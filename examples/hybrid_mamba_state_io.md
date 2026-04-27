# Hybrid model (Mamba + Attention) state I/O

When this fork is paired with the companion vLLM fork branch
`ralph/hybrid-mamba-state-io-v0.19.1`, an LMCache disk tier can hold
**both** attention KV and Mamba/GatedDeltaNet recurrent state across
process restarts for hybrid models such as Qwen3-Next, Qwen3.5/3.6,
LFM2, etc.

## What you get

A fresh vLLM process started against a previously-warmed LMCache disk
path produces output **bit-identical** to a no-cache baseline AND
skips forward compute for the cached prefix on **both** lanes.
Verified end-to-end on `Qwen/Qwen3.5-9B`: Phase B (cold restart)
reports `num_cached_tokens=528` (one Mamba block) and the LMCache
worker log line shows `need to load = 528`.

## Required setup

1. Install this fork of LMCache (branch
   `ralph/hybrid-attn-mamba-disk-persist-v0.4.4`).
2. Install the companion vLLM fork (branch
   `ralph/hybrid-mamba-state-io-v0.19.1`).
3. **Set `PYTHONHASHSEED=0`** in every process that touches the same
   disk path. LMCache's chunk-hash computation uses Python's built-in
   `hash()` which is salted at interpreter startup; without a fixed
   seed, two processes compute different hashes for the same tokens
   and never share entries.

## YAML

Save as `lmcache.yaml` and point `LMCACHE_CONFIG_FILE` at it:

```yaml
# Chunk size — must equal the model's MambaSpec.block_size for the
# attn-lane cache hits to land on Mamba block boundaries. Set to
# ``auto`` to let LMCache read the value out of vLLM's
# kv_cache_config at connector init time (works for any hybrid model
# vLLM supports). Pure-attention models with ``auto`` fall back to
# the conventional 256.
chunk_size: auto    # or set explicitly: e.g. 528 for Qwen/Qwen3.5-9B

# Memory tier (CPU staging area). Required.
local_cpu: true
max_local_cpu_size: 2.0

# Persistent attention KV tier on disk.
local_disk: /var/lmcache/qwen3.5-9b
max_local_disk_size: 8.0

# Save partial chunks too — important for short prompts.
save_unfull_chunk: true

# === Hybrid model state I/O (this fork) ====================
# Turn on so the recurrent (Mamba) state is also externalised.
# Requires the companion vLLM fork that exposes
# register_external_mamba_state_{save,restore}_hook in
# vllm.v1.worker.gpu_model_runner / vllm.v1.worker.mamba_utils.
enable_hybrid_mamba_state_io: true
# Where Mamba state files live. ``null`` means "share the directory
# with local_disk", which is the simplest option.
hybrid_mamba_state_io_path: null
hybrid_mamba_state_io_size_gb: 4.0
```

## vLLM CLI

```bash
PYTHONHASHSEED=0 \
LMCACHE_CONFIG_FILE=lmcache.yaml \
vllm serve Qwen/Qwen3.5-9B \
  --enable-prefix-caching \
  --no-disable-hybrid-kv-cache-manager \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both","kv_connector_extra_config":{"lmcache.local_cpu":true,"lmcache.max_local_cpu_size":2.0,"lmcache.local_disk":"/var/lmcache/qwen3.5-9b","lmcache.max_local_disk_size":8.0,"lmcache.chunk_size":528,"lmcache.save_unfull_chunk":true,"lmcache.enable_hybrid_mamba_state_io":true,"lmcache.hybrid_mamba_state_io_size_gb":4.0}}'
```

(Either `LMCACHE_CONFIG_FILE` or the `kv_connector_extra_config` form
work; `extra_config` keys are prefixed `lmcache.`.)

## Verify

Two prefix-sharing prompts in two separate processes against the same
`local_disk`. The second process should report
`num_cached_tokens > 0` and `need to load > 0` in the LMCache log
line — both signals together mean LMCache served the prefix from
disk. Output equals a baseline run without LMCache attached.

## Falling back

Set `enable_hybrid_mamba_state_io: false` (or omit the key — the
default is False) to disable the Mamba-state externalisation. The
attention lane still benefits from the disk-tier sidecar persistence
(also added in this fork) so cross-process retrieve works for
pure-attention models too.

## Caveats

* `chunk_size: auto` is the recommended setting for hybrid models —
  it reads the model's Mamba ``block_size`` directly from vLLM's
  ``kv_cache_config`` at connector init time. Setting it manually is
  only useful when you need a non-default value for a pure-attention
  model.
* Save granularity is per Mamba block (e.g. 528 tokens for
  Qwen3.5-9B). Prompts shorter than one block do not benefit from
  the Mamba half of the cache, only attn.
* Single-request prefill is verified end-to-end. Multi-request
  batches sharing prefixes should work by the same logic but are not
  exercised by the current test set.
