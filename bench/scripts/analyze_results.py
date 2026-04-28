#!/usr/bin/env python3
"""Aggregate TTFT bench results from {fsdax,devdax}/<timestamp>/dist_outputs/.

Reads each rank's summary.rank{r}.json, prints p50/p95 of TTFT and end-to-
end latency per condition, and dumps a CSV side-by-side at
``bench/results/aggregated.csv``. Also greps each rank's vllm server log
for cache-hit signals (``num_cached_tokens``, ``need to load``,
``hybrid_mamba_state_io``) and reports counts.

Usage:
    python bench/scripts/analyze_results.py
"""
from __future__ import annotations

import csv
import json
import re
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    k = (len(s) - 1) * p / 100
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] if f == c else s[f] + (s[c] - s[f]) * (k - f)


def latest_run(cond_dir: Path) -> Path | None:
    """Return the most recent timestamp dir under ``cond_dir`` that has
    at least one summary.rank*.json under dist_outputs/."""
    candidates = []
    for d in cond_dir.iterdir():
        if not d.is_dir():
            continue
        if any(d.glob("dist_outputs/rank_*/summary.rank*.json")):
            candidates.append(d)
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.name)


def load_rank_summaries(run_dir: Path) -> list[dict]:
    out = []
    for r in range(4):
        f = run_dir / "dist_outputs" / f"rank_{r}" / f"summary.rank{r}.json"
        if not f.exists():
            continue
        with open(f) as fh:
            d = json.load(fh)
        out.append({"rank": r, "data": d})
    return out


def collect_metric(rank_summaries: list[dict], key: str) -> list[float]:
    vals: list[float] = []
    for rs in rank_summaries:
        for r in rs["data"].get("results", []):
            v = r.get(key)
            if v is not None:
                vals.append(float(v))
    return vals


CACHE_HIT_PATTERNS = (
    re.compile(r"num_cached_tokens\s*[=:]\s*(\d+)"),
    re.compile(r"need to load\s*[=:]\s*(\d+)"),
    re.compile(r"hybrid_mamba_state_io: restored Mamba state for (\d+)"),
)


def cache_hit_counts(run_dir: Path) -> dict[str, int]:
    counts = {"num_cached_tokens_lines": 0, "need_to_load_lines": 0,
              "mamba_restored_lines": 0}
    for log in run_dir.glob("rank_logs/rank_*/vllm_server.stderr.log"):
        text = log.read_text(errors="ignore")
        counts["num_cached_tokens_lines"] += sum(
            1 for _ in CACHE_HIT_PATTERNS[0].finditer(text)
        )
        counts["need_to_load_lines"] += sum(
            1 for _ in CACHE_HIT_PATTERNS[1].finditer(text)
        )
        counts["mamba_restored_lines"] += sum(
            1 for _ in CACHE_HIT_PATTERNS[2].finditer(text)
        )
    return counts


def main() -> None:
    rows = []
    for cond in ("fsdax", "devdax"):
        cond_dir = RESULTS / cond
        if not cond_dir.exists():
            print(f"[skip] {cond}: no results dir")
            continue
        run = latest_run(cond_dir)
        if run is None:
            print(f"[skip] {cond}: no completed runs under {cond_dir}")
            continue
        rs = load_rank_summaries(run)
        ttfts = collect_metric(rs, "ttft")
        lats = collect_metric(rs, "latency")
        gens = collect_metric(rs, "generation_time")
        cache = cache_hit_counts(run)
        row = {
            "cond": cond,
            "run": run.name,
            "n_ranks": len(rs),
            "n_requests": sum(len(r["data"].get("results", [])) for r in rs),
            "ttft_p50": percentile(ttfts, 50),
            "ttft_p95": percentile(ttfts, 95),
            "ttft_mean": st.mean(ttfts) if ttfts else float("nan"),
            "lat_p50": percentile(lats, 50),
            "lat_p95": percentile(lats, 95),
            "gen_p50": percentile(gens, 50),
            **cache,
        }
        rows.append(row)
        print(f"\n=== {cond} ({run.name}) ===")
        print(f"  ranks={row['n_ranks']}, requests={row['n_requests']}")
        print(f"  TTFT p50={row['ttft_p50']:.3f}s, p95={row['ttft_p95']:.3f}s, "
              f"mean={row['ttft_mean']:.3f}s")
        print(f"  Latency p50={row['lat_p50']:.3f}s, p95={row['lat_p95']:.3f}s")
        print(f"  GenTime p50={row['gen_p50']:.3f}s")
        print(f"  Cache-hit signals: "
              f"num_cached_tokens={row['num_cached_tokens_lines']}, "
              f"need_to_load={row['need_to_load_lines']}, "
              f"mamba_restored={row['mamba_restored_lines']}")

    if rows:
        out = RESULTS / "aggregated.csv"
        with open(out, "w") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\n→ wrote {out}")
        if len(rows) == 2:
            f50, d50 = rows[0]["ttft_p50"], rows[1]["ttft_p50"]
            speedup = f50 / d50 if d50 else float("nan")
            print(
                f"\nTTFT p50 speedup ({rows[1]['cond']} vs {rows[0]['cond']}): "
                f"{speedup:.2f}x  ({f50:.3f}s → {d50:.3f}s)"
            )


if __name__ == "__main__":
    main()
