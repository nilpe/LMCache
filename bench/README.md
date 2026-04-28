# TTFT verification: ralph branches × {fsdax, devdax}

This directory holds the artefacts to verify whether attaching the
DevDAX backend to the nilpe ralph branches (LMCache + vLLM hybrid Mamba
state I/O) shortens TTFT relative to vanilla LocalDiskBackend on FSDAX.

## What is being compared

| Condition | Disk backend | Where data lives | Expected raw BW |
|-----------|--------------|------------------|-----------------|
| `fsdax`   | `LocalDiskBackend` (vanilla, ralph baseline) | `/pmem/...` (FSDAX) | ~3.3 GB/s |
| `devdax`  | `DevDaxBackend` (this fork's port) | `/dev/dax0.0` slab | ~34 GB/s (BAR1 push) |

Both conditions exercise the same vLLM hybrid path:

- `nilpe-vllm` branch `ralph/hybrid-mamba-state-io-v0.19.1` — exposes
  `register_external_mamba_state_{save,restore}_hook`.
- `nilpe-lmcache` branch `ykogi/ralph-devdax` (this branch) — adds
  the YAML-driven sidecar dispatch (db58a844) on top of nilpe's ralph
  branch, plus the cherry-picked devdax/bar1/gpu_dma stack.

## Files

Two pairs of configs, mirroring the existing `lmcache-perf-analysis/ttft_bench`
directory shape so the same `multi-decode` rust harness drives both:

- `configs/multidecode_fsdax.toml` + `multidecode_fsdax.yaml` — vanilla
  `LocalDiskBackend` on `/pmem/...`. Pure-attention (Qwen2.5-7B-1M).
- `configs/multidecode_devdax.toml` + `multidecode_devdax.yaml` —
  `DevDaxBackend` on `/dev/dax0.0` via `extra_config.disk_backend_type`.
- `configs/qwen35_9b_fsdax.yaml` / `qwen35_9b_devdax.yaml` — hybrid
  Mamba state I/O configs (Qwen3.5-9B). Used by the simpler
  `run_ttft.sh` harness; not driven by `multi-decode` because that
  binary's TOML expects a model that vLLM serve can load — Qwen3.5-9B
  needs to be downloaded first.
- `prompts/long_prefix.txt` — symlink to a 32k-token Gutenberg text.
- `scripts/setup_venv.sh` — one-off venv bootstrapper. Uses
  `VLLM_USE_PRECOMPILED=1` + `NO_CUDA_EXT=1` so neither vLLM nor
  LMCache's CUDA kernels rebuild from source. Run on a login node.
- `scripts/run_one.sh` — multi-decode runner for a single condition.
- `scripts/submit_devdax_vs_fsdax.sh` — PBS job that calls
  `run_one.sh fsdax` then `run_one.sh devdax`. **Not auto-submitted.**

## Pre-flight

1. Build the venv (one-off, on a login node):

   ```bash
   bash bench/scripts/setup_venv.sh
   ```

   Smoke test at the end prints `ok hybrid_mamba_state_io`,
   `ok DevDaxBackend`, `ok vLLM ralph hooks present`. If any of those
   miss, re-check the env vars at the top of `setup_venv.sh`.

2. Verify the dispatch wiring without launching vLLM:

   ```bash
   source /work/0/NBB/kogi/workspace/nilpe-bench-venv/venv/bin/activate
   python -c "
   import yaml
   from lmcache.v1.config import LMCacheEngineConfig
   from lmcache.integration.vllm.hybrid_mamba_state_io import _build_sidecar_config
   raw = yaml.safe_load(open('bench/configs/qwen35_9b_devdax.yaml'))
   parent = LMCacheEngineConfig.from_defaults()
   for k, v in raw.items():
       if k == 'hybrid_mamba_state_io_config': continue
       setattr(parent, k, v)
   side = _build_sidecar_config(parent, raw['hybrid_mamba_state_io_config'])
   assert side.extra_config.get('disk_backend_type') == 'devdax'
   print('OK')
   "
   ```

3. (Hybrid-only) Download Qwen3.5-9B on a node with outbound HTTP:

   ```bash
   huggingface-cli download Qwen/Qwen3.5-9B \
       --local-dir /home/NBB/kogi/work/workspace/archived/test/hf_models/Qwen/Qwen3.5-9B
   ```

   Then update `qwen35_9b_*.yaml` and the `run_ttft.sh` `MODEL` env if
   needed.

## Running the bench

The pure-attention multi-decode bench (parallels `lmcache-perf-analysis`):

```bash
qsub bench/scripts/submit_devdax_vs_fsdax.sh
```

The hybrid Mamba-state lite bench (single curl, after Qwen3.5-9B is
downloaded):

```bash
qsub bench/scripts/job_ttft_devdax_vs_fsdax.sh
```

## What to look for

For each condition, `summary.txt` should contain:

- `phaseA_e2e_seconds`: warm-pass end-to-end latency (cache cold).
- `phaseB_e2e_seconds`: cold-restart end-to-end latency (cache warm
  on disk).
- `num_cached_tokens > 0` and `need to load > 0` from the LMCache log
  during Phase B — both signals together mean the prefix was served
  from disk.

Verdict:

- If `devdax: phaseB_e2e_seconds < fsdax: phaseB_e2e_seconds`, devdax
  shortens TTFT for hybrid models (the hypothesis under test).
- If both conditions report similar `phaseB`, the disk-to-GPU path is
  not the bottleneck for the chosen prefix length; widen the prompt
  or tighten the rest of the pipeline (model-load, scheduler).
