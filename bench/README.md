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

- `configs/qwen35_9b_fsdax.yaml` — baseline LMCache config (FSDAX
  for both attn KV and Mamba state).
- `configs/qwen35_9b_devdax.yaml` — DevDAX for both attn KV and
  Mamba state. Parent slab at offset 0, sidecar slab at offset 1 TiB.
- `prompts/long_prefix.txt` — symlink to a 32k-token Gutenberg text
  used as the shared prefix.
- `scripts/run_ttft.sh` — runs Phase A (warm) then Phase B (cold-
  restart, same disk path) for one condition. Reports phase-B TTFT
  and the LMCache cache-hit log lines.
- `scripts/job_ttft_devdax_vs_fsdax.sh` — PBS wrapper that runs
  `run_ttft.sh` for both `fsdax` and `devdax`. **Not auto-submitted.**

## Pre-flight (no qsub)

1. Verify the dispatch wiring is live:

   ```bash
   source venv/bin/activate
   python -c "
   from lmcache.integration.vllm.hybrid_mamba_state_io import (
       _build_disk_backend, _build_sidecar_config,
   )
   import yaml
   from lmcache.v1.config import LMCacheEngineConfig
   raw = yaml.safe_load(open('bench/configs/qwen35_9b_devdax.yaml'))
   parent = LMCacheEngineConfig.from_defaults()
   for k, v in raw.items():
       if k == 'hybrid_mamba_state_io_config': continue
       setattr(parent, k, v)
   side = _build_sidecar_config(parent, raw['hybrid_mamba_state_io_config'])
   assert side.extra_config.get('disk_backend_type') == 'devdax'
   print('OK: sidecar would dispatch to devdax')
   "
   ```

2. Download the model (one-off, on a node with network):

   ```bash
   huggingface-cli download Qwen/Qwen3.5-9B
   ```

3. Confirm the vLLM venv has the companion fork installed:

   ```bash
   python -c "
   from vllm.v1.worker.gpu_model_runner import register_external_mamba_state_save_hook
   from vllm.v1.worker.mamba_utils import register_external_mamba_state_restore_hook
   print('OK: vLLM hooks present')
   "
   ```

## Running the bench

After review, submit:

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
