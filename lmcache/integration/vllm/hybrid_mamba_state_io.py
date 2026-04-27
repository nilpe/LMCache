"""Externalise vLLM hybrid model Mamba state through LMCache disk.

vLLM 0.19.1 hybrid models (Qwen3-Next, Qwen3.5/3.6, LFM2, ...) interleave
full-attention layers with Mamba/GatedDeltaNet layers. The companion
LMCache patches in this branch already make the attention lane work
end-to-end (filtering Mamba slots from ``register_kv_caches``,
overriding ``num_layer`` to attention-only count, persisting attention
KV across process restarts via the disk-tier sidecar). What's missing
to make a hybrid restart actually skip compute on cache hits is the
recurrent-state half of the story: vLLM honours an external KV hit by
skipping forward compute for the cached prefix, but the Mamba layers
have no state for that prefix unless something restores it into the
right block.

This module supplies that missing half. It registers two hooks on the
companion vLLM patch series:

  * ``register_external_mamba_state_restore_hook`` (preprocess_mamba)
    — called at the start of every step; for each cache-hit request,
    look up the saved state under ``(model, layer_name, dtype, shape,
    prefix_token_hash)`` and write it into
    ``block_ids[group_id][prev_state_idx]``. vLLM's existing block-copy
    machinery then propagates that into ``curr_state_idx`` and the
    forward picks up correct state.

  * ``register_external_mamba_state_save_hook`` (execute_model)
    — called after every step; if the request just landed on a Mamba
    block boundary (``new_total % block_size == 0``), read state from
    ``block_ids[group_id][mamba_state_idx]`` and store it under the
    same key.

Storage uses the existing ``LocalDiskBackend`` from this fork (the one
with the JSONL sidecar that lets a fresh process re-register what
previous processes wrote) so a single disk path warms up across runs.

Activated by ``enable_hybrid_mamba_state_io: true`` in lmcache.yaml.
The connector calls ``maybe_install`` once during init; a no-op when
the flag is off OR when vLLM is too old to expose the hook registration
functions.
"""
from __future__ import annotations

import asyncio
import hashlib
import os
import threading
from typing import TYPE_CHECKING, Any, Optional

import torch

from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import MemoryFormat
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from lmcache.v1.storage_backend.local_disk_backend import LocalDiskBackend

logger = init_logger(__name__)

if TYPE_CHECKING:
    pass  # only used for typing in hook signatures


_INSTALLED = False
_DISK_BACKEND: Optional[LocalDiskBackend] = None
_LOOP: Optional[asyncio.AbstractEventLoop] = None
_LOOP_THREAD: Optional[threading.Thread] = None
_MODEL_NAME: str = "lmcache-hybrid-mamba"


# ── Helpers ────────────────────────────────────────────────────────────


def _hash_prefix_tokens(token_ids) -> str:
    """SHA-256 hex of a token-id sequence, fixed-width 4-byte LE encoding
    so [1,23] and [12,3] do not collide."""
    h = hashlib.sha256()
    for t in token_ids:
        if not isinstance(t, int) or t < 0 or t >= 2**31:
            return ""
        h.update(int(t).to_bytes(4, byteorder="little", signed=False))
    return h.hexdigest()


def _is_mamba_spec(spec: Any) -> bool:
    return type(spec).__name__ == "MambaSpec"


def _make_state_key(
    *, layer_name: str, dtype: str, sub_idx: int,
    state_shape: tuple, prefix_token_hash: str,
) -> CacheEngineKey:
    """Build a CacheEngineKey for one Mamba sub-state of one layer.

    The full identity goes into ``model_name`` so it survives
    LMCache's hashing / disk filename layout unchanged. ``world_size``,
    ``worker_id``, ``chunk_hash`` are constants since per-layer Mamba
    state is not chunked the way attn KV is. ``dtype`` matches the
    sub-state's dtype.
    """
    sh = "x".join(str(d) for d in state_shape)
    name = (
        f"mamba:{_MODEL_NAME}:{layer_name}:{dtype}:s{sub_idx}:"
        f"{sh}:{prefix_token_hash}"
    )
    # CacheEngineKey expects torch.dtype for the dtype field.
    torch_dtype = getattr(torch, dtype.replace("torch.", ""), torch.bfloat16)
    return CacheEngineKey(
        model_name=name,
        world_size=1,
        worker_id=0,
        chunk_hash=0,
        dtype=torch_dtype,
    )


def _get_layer_kv_tensors(attn_module: Any) -> Optional[list[torch.Tensor]]:
    """Pull ``[conv_state, ssm_state]`` out of a Mamba attention module.

    vLLM stores the per-virtual-engine kv_cache as either a flat list
    (lab path; PP=1) or nested ``list[list[Tensor]]`` (PP > 1). Pick
    the first virtual engine in either case.
    """
    layer_kv = getattr(attn_module, "kv_cache", None)
    if layer_kv is None:
        return None
    if isinstance(layer_kv, list) and layer_kv and isinstance(
        layer_kv[0], list
    ):
        layer_kv = layer_kv[0]
    if not isinstance(layer_kv, (list, tuple)):
        return None
    return layer_kv


# ── Save / restore implementations ────────────────────────────────────


def _restore_one_request(
    *, req_state, kv_cache_config, forward_context,
) -> int:
    """Pull cached Mamba state for one request and write into
    ``block_ids[group_id][prev_state_idx]``. Returns the count of
    layers fully restored across all Mamba groups."""
    if _DISK_BACKEND is None:
        return 0
    cached_prefix_len = int(req_state.num_computed_tokens)
    if cached_prefix_len == 0 or not req_state.prompt_token_ids:
        return 0
    prefix_hash = _hash_prefix_tokens(
        req_state.prompt_token_ids[:cached_prefix_len]
    )
    if not prefix_hash:
        return 0
    restored = 0
    for group_id, group in enumerate(kv_cache_config.kv_cache_groups):
        spec = group.kv_cache_spec
        if not _is_mamba_spec(spec):
            continue
        block_size = getattr(spec, "block_size", 0)
        if block_size <= 0:
            continue
        prev_state_idx = (cached_prefix_len - 1) // block_size
        if prev_state_idx < 0:
            continue
        block_ids = req_state.block_ids[group_id]
        if prev_state_idx >= len(block_ids):
            continue
        block_id = block_ids[prev_state_idx]
        sub_shapes = list(getattr(spec, "shapes", []) or [])
        sub_dtypes = list(getattr(spec, "dtypes", []) or [])
        if not sub_shapes or not sub_dtypes:
            continue
        for layer_name in group.layer_names:
            attn_module = forward_context.get(layer_name)
            if attn_module is None:
                continue
            layer_kv = _get_layer_kv_tensors(attn_module)
            if layer_kv is None:
                continue
            ok_count = 0
            for sub_idx, (state_tensor, sub_shape, sub_dtype) in enumerate(
                zip(layer_kv, sub_shapes, sub_dtypes)
            ):
                key = _make_state_key(
                    layer_name=layer_name, dtype=str(sub_dtype),
                    sub_idx=sub_idx,
                    state_shape=tuple(sub_shape),
                    prefix_token_hash=prefix_hash,
                )
                obj = _DISK_BACKEND.get_blocking(key)
                if obj is None:
                    break
                target = state_tensor[block_id]
                src = obj.tensor
                if src.shape != target.shape:
                    obj.ref_count_down()
                    break
                target.copy_(src.to(target.device))
                obj.ref_count_down()
                ok_count += 1
            if ok_count == len(sub_shapes):
                restored += 1
    return restored


def _save_one_request(
    *, req_state, mamba_state_idx_for_req: int,
    kv_cache_config, forward_context, new_total_tokens: int,
) -> int:
    """Snapshot Mamba state for one request iff ``new_total`` is on a
    block boundary. Returns count of layer-instances saved."""
    if _DISK_BACKEND is None:
        return 0
    if not req_state.prompt_token_ids or new_total_tokens <= 0:
        return 0
    if new_total_tokens > len(req_state.prompt_token_ids):
        return 0
    saved = 0
    for group_id, group in enumerate(kv_cache_config.kv_cache_groups):
        spec = group.kv_cache_spec
        if not _is_mamba_spec(spec):
            continue
        block_size = getattr(spec, "block_size", 0)
        if block_size <= 0 or new_total_tokens % block_size != 0:
            continue
        snapshot_prefix_len = new_total_tokens
        block_ids = req_state.block_ids[group_id]
        if mamba_state_idx_for_req >= len(block_ids):
            continue
        block_id = block_ids[mamba_state_idx_for_req]
        prefix_hash = _hash_prefix_tokens(
            req_state.prompt_token_ids[:snapshot_prefix_len]
        )
        if not prefix_hash:
            continue
        sub_shapes = list(getattr(spec, "shapes", []) or [])
        sub_dtypes = list(getattr(spec, "dtypes", []) or [])
        if not sub_shapes or not sub_dtypes:
            continue
        for layer_name in group.layer_names:
            attn_module = forward_context.get(layer_name)
            if attn_module is None:
                continue
            layer_kv = _get_layer_kv_tensors(attn_module)
            if layer_kv is None:
                continue
            ok_count = 0
            for sub_idx, (state_tensor, sub_shape, sub_dtype) in enumerate(
                zip(layer_kv, sub_shapes, sub_dtypes)
            ):
                src = state_tensor[block_id].detach().contiguous()
                key = _make_state_key(
                    layer_name=layer_name, dtype=str(sub_dtype),
                    sub_idx=sub_idx, state_shape=tuple(sub_shape),
                    prefix_token_hash=prefix_hash,
                )
                # Allocate a CPU MemoryObj via the disk backend's local
                # CPU allocator and copy state into it.
                cpu_alloc = _DISK_BACKEND.local_cpu_backend
                obj = cpu_alloc.allocate(src.shape, src.dtype)
                if obj is None:
                    break
                obj.tensor.copy_(src)
                # remove any stale entry at this key first
                if _DISK_BACKEND.contains(key):
                    _DISK_BACKEND.remove(key, force=True)
                _DISK_BACKEND.submit_put_task(key, obj)
                obj.ref_count_down()
                ok_count += 1
            if ok_count == len(sub_shapes):
                saved += 1
    return saved


# ── Hook callbacks (wired into vLLM's hook registries) ────────────────


def _restore_hook(
    scheduler_output, kv_cache_config, mamba_state_idx,
    input_batch, requests, forward_context,
):
    """Adapter matching ``ExternalMambaStateRestoreHook`` from
    ``vllm.v1.worker.mamba_utils``."""
    total = 0
    for req_id in input_batch.req_ids:
        req_state = requests.get(req_id)
        if req_state is None:
            continue
        # Skip already-running requests; only newly-arrived ones need
        # external restore.
        if mamba_state_idx.get(req_id) is not None:
            continue
        try:
            total += _restore_one_request(
                req_state=req_state,
                kv_cache_config=kv_cache_config,
                forward_context=forward_context,
            )
        except Exception as e:
            logger.warning(
                "hybrid_mamba_state_io: restore failed for %s: %r",
                req_id, e,
            )
    if total:
        logger.info(
            "hybrid_mamba_state_io: restored Mamba state for %d "
            "layer-instance(s) from external storage", total,
        )


def _save_hook(runner, scheduler_output):
    """Adapter matching ``ExternalMambaStateSaveHook`` from
    ``vllm.v1.worker.gpu_model_runner``."""
    kv_cache_config = runner.kv_cache_config
    requests = runner.requests
    input_batch = runner.input_batch
    mamba_state_idx = getattr(runner, "mamba_state_idx", {})
    forward_context = runner.compilation_config.static_forward_context
    num_scheduled = scheduler_output.num_scheduled_tokens
    # Quick exit when no Mamba groups are present (pure-attn model).
    if not any(
        _is_mamba_spec(g.kv_cache_spec) for g in kv_cache_config.kv_cache_groups
    ):
        return
    total = 0
    for req_id in input_batch.req_ids:
        req_state = requests.get(req_id)
        if req_state is None:
            continue
        state_idx = mamba_state_idx.get(req_id)
        if state_idx is None:
            continue
        new_total = (
            req_state.num_computed_tokens
            + int(num_scheduled.get(req_id, 0))
        )
        try:
            total += _save_one_request(
                req_state=req_state,
                mamba_state_idx_for_req=state_idx,
                kv_cache_config=kv_cache_config,
                forward_context=forward_context,
                new_total_tokens=new_total,
            )
        except Exception as e:
            logger.warning(
                "hybrid_mamba_state_io: save failed for %s: %r",
                req_id, e,
            )
    if total:
        logger.debug(
            "hybrid_mamba_state_io: saved Mamba state for %d "
            "layer-instance(s)", total,
        )


# ── Install / dispose ─────────────────────────────────────────────────


def _start_loop_thread() -> asyncio.AbstractEventLoop:
    loop = asyncio.new_event_loop()

    def _run():
        asyncio.set_event_loop(loop)
        loop.run_forever()

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    global _LOOP_THREAD
    _LOOP_THREAD = t
    return loop


def maybe_install(
    config: LMCacheEngineConfig,
    metadata: LMCacheMetadata,
) -> bool:
    """Install hybrid Mamba state I/O hooks if the config opts in AND
    the host vLLM exposes the registration functions.

    Idempotent: subsequent calls are no-ops. Returns True iff freshly
    installed in this process."""
    global _INSTALLED, _DISK_BACKEND, _LOOP, _MODEL_NAME

    if not getattr(config, "enable_hybrid_mamba_state_io", False):
        return False
    if _INSTALLED:
        return False

    try:
        from vllm.v1.worker.gpu_model_runner import (
            register_external_mamba_state_save_hook,
        )
        from vllm.v1.worker.mamba_utils import (
            register_external_mamba_state_restore_hook,
        )
    except ImportError:
        logger.warning(
            "hybrid_mamba_state_io: vLLM does not expose "
            "register_external_mamba_state_*_hook; skipping. Install "
            "the companion vLLM patch series."
        )
        return False

    # Decide where the Mamba state files live.
    disk_path = getattr(config, "hybrid_mamba_state_io_path", None) or config.local_disk
    if disk_path is None:
        logger.warning(
            "hybrid_mamba_state_io: neither hybrid_mamba_state_io_path nor "
            "local_disk is set; skipping"
        )
        return False
    os.makedirs(disk_path, exist_ok=True)

    # Build a dedicated LocalDiskBackend for Mamba state, wired through
    # a lightweight CPU allocator. We reuse the user's
    # ``hybrid_mamba_state_io_size_gb`` budget here; it's separate from
    # the attention-lane budget.
    sub_config = LMCacheEngineConfig.from_defaults()
    sub_config.local_cpu = True
    sub_config.max_local_cpu_size = 0.5  # only used for staging
    sub_config.local_disk = disk_path
    sub_config.max_local_disk_size = float(
        getattr(config, "hybrid_mamba_state_io_size_gb", 4.0)
    )
    sub_config.chunk_size = config.chunk_size
    sub_config.cache_policy = config.cache_policy

    _LOOP = _start_loop_thread()

    cpu = LocalCPUBackend(sub_config, metadata, dst_device="cpu")
    _DISK_BACKEND = LocalDiskBackend(
        sub_config, loop=_LOOP, local_cpu_backend=cpu,
        dst_device="cpu", metadata=metadata,
    )

    # Capture model name for keys (used in _make_state_key).
    _MODEL_NAME = metadata.model_name or "lmcache-hybrid-mamba"

    register_external_mamba_state_restore_hook(_restore_hook)
    register_external_mamba_state_save_hook(_save_hook)

    _INSTALLED = True
    logger.info(
        "hybrid_mamba_state_io: installed; disk=%s size=%.1f GiB chunk_size=%d",
        disk_path, sub_config.max_local_disk_size, sub_config.chunk_size,
    )
    return True
