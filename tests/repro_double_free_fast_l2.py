"""
Reproduction: feeding a "fast" (synchronous-completing) secondary storage
backend into StorageManager.batched_put causes the MemoryObj reference
count to go negative, triggering the
   "Double free occurred somewhere."
warning in lmcache.v1.memory_management.

Setup:
  - Primary allocator backend: real LocalCPUBackend (uses MixedMemoryAllocator)
  - Mock model: just a config + metadata pair (no actual model)
  - Secondary backend: an in-process FastMockBackend that *synchronously*
    finishes its put — i.e. it follows the LocalDiskBackend convention of
    ref_count_up()/ref_count_down() but does the down() right inside
    submit_put_task instead of via the asyncio loop.

We then call StorageManager.batched_put once, and inspect the ref_count
of the cached MemoryObj after the call returns.
"""

import logging
import threading
from collections import OrderedDict
from typing import Any, Callable, List, Optional, Sequence

import torch

from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import MemoryFormat, MemoryObj
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.abstract_backend import (
    AllocatorBackendInterface,
    StorageBackendInterface,
)
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from lmcache.v1.storage_backend.storage_manager import StorageManager

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

# ---- bind warnings so we can detect "Double free" ------------------------
caught_warnings: list[str] = []


class CatchHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:  # noqa: D401
        msg = record.getMessage()
        if "Double free" in msg or "is negative" in msg:
            caught_warnings.append(msg)


logging.getLogger("lmcache.v1.memory_management").addHandler(CatchHandler())


# ---- a "fast" (synchronous-completing) secondary backend -----------------
class FastMockBackend(StorageBackendInterface):
    """
    Mimics the LocalDiskBackend put protocol (ref_count_up + later
    ref_count_down) but completes its put synchronously, exactly like a
    storage tier whose write latency is essentially zero.
    """

    def __init__(self, allocator_backend: AllocatorBackendInterface):
        super().__init__("cpu")
        self._allocator = allocator_backend
        self.store: dict[CacheEngineKey, MemoryObj] = {}
        self._lock = threading.Lock()
        self._in_flight: set[CacheEngineKey] = set()

    def get_allocator_backend(self) -> AllocatorBackendInterface:
        return self._allocator

    def contains(self, key: CacheEngineKey, pin: bool = False) -> bool:
        with self._lock:
            return key in self.store

    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        with self._lock:
            return key in self._in_flight

    def batched_submit_put_task(
        self,
        keys: Sequence[CacheEngineKey],
        objs: List[MemoryObj],
        transfer_spec: Any = None,
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]] = None,
    ) -> None:
        """
        Mimics a "too-fast" L2 storage tier:
          - The write completes synchronously, on the StorageManager thread.
          - Because the put finishes before submit_put_task returns, the
            backend follows the "skip if already in flight" early-out path
            on the SECOND call for the same key.

        Implementation note: the LocalDisk-style protocol is
            ref_count_up()  -> schedule async write that does ref_count_down()
        For a synchronous L2 we'd compress this to ref_up + ref_down
        bracketed around the immediate write. The bug surfaces when the
        backend was written assuming the StorageManager hands off ownership
        of the +1 reference (so it does *only* a final ref_count_down, no
        matching ref_count_up). This is exactly the protocol confusion
        that pops up when somebody bolts a synchronous L2 onto the put
        pipeline.
        """
        for key, memory_obj in zip(keys, objs, strict=False):
            with self._lock:
                if key in self._in_flight or key in self.store:
                    continue  # like LocalDisk's exists_in_put_tasks early-out
                self._in_flight.add(key)
                self.store[key] = memory_obj
                self._in_flight.discard(key)
            # "consume" the reference that was handed in by the caller --
            # this matches how a couple of L2 connectors decrement the
            # ref count after persisting (mooncakestore_connector,
            # s3_connector, fs_connector all call ref_count_down inside
            # their put path), instead of doing the up/down dance.
            memory_obj.ref_count_down()

    def get_blocking(self, key):  # pragma: no cover - unused in repro
        with self._lock:
            return self.store.get(key)

    def pin(self, key):  # pragma: no cover
        return False

    def unpin(self, key):  # pragma: no cover
        return False

    def remove(self, key, force=True):  # pragma: no cover
        with self._lock:
            return self.store.pop(key, None) is not None

    def close(self):  # pragma: no cover
        pass

    def touch_cache(self):  # pragma: no cover
        pass


# ---- mock model: minimal config + metadata --------------------------------
def make_mock_config_and_metadata():
    cfg = LMCacheEngineConfig.from_defaults()
    cfg.local_cpu = True
    cfg.max_local_cpu_size = 0.5  # GiB

    meta = LMCacheMetadata(
        model_name="mock-model",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.float16,
        kv_shape=(2, 1, 256, 8, 64),  # (2, num_layers, chunk, n_heads, head_dim)
        use_mla=False,
    )
    return cfg, meta


# ---- build a minimal StorageManager (skip __init__) -----------------------
def build_storage_manager(cfg, meta):
    sm = StorageManager.__new__(StorageManager)

    cpu = LocalCPUBackend(config=cfg, metadata=meta, dst_device="cpu")
    fast = FastMockBackend(allocator_backend=cpu)

    sm.config = cfg
    sm.metadata = meta
    sm.storage_backends = OrderedDict([("LocalCPUBackend", cpu), ("FastMock", fast)])
    sm.allocator_backend = cpu
    sm.local_cpu_backend = cpu
    sm.internal_copy_stream = None
    sm._bypassed_backends = set()
    sm._bypass_lock = threading.RLock()
    sm._freeze = False
    sm._freeze_lock = threading.RLock()
    return sm, cpu, fast


def main() -> int:
    cfg, meta = make_mock_config_and_metadata()
    sm, cpu, fast = build_storage_manager(cfg, meta)

    # Allocate one MemoryObj using the real allocator. The caller-owned
    # ref_count starts at 1, exactly as a real producer would hand it off
    # to StorageManager.batched_put.
    shape = torch.Size([2, 1, 256, 8, 64])
    dtype = torch.float16
    obj = cpu.allocate(shape, dtype, fmt=MemoryFormat.KV_2LTD)
    assert obj is not None
    print(f"[before put] ref_count = {obj.get_ref_count()}")

    key = CacheEngineKey(model_name="mock", world_size=1,
                         worker_id=0, chunk_hash=0xC0FFEE,
                         dtype=torch.float16)

    sm.batched_put([key], [obj])

    print(f"[after put]  ref_count = {obj.get_ref_count()}")
    print(f"[after put]  in cpu.hot_cache: {key in cpu.hot_cache}")
    print(f"[after put]  in fast.store:    {key in fast.store}")

    # Now exercise eviction from LocalCPU's hot_cache. In a real run this
    # happens implicitly the moment something else tries to allocate and
    # the LRU policy picks our key. Calling remove() directly is the same
    # ref_count_down() path.
    print("\n[evict] removing the key from LocalCPUBackend.hot_cache ...")
    cpu.remove(key)
    print(f"[after evict] ref_count = {obj.get_ref_count()}")

    if caught_warnings:
        print("\n!!! DOUBLE FREE WARNINGS CAUGHT !!!")
        for w in caught_warnings:
            print("   ", w)
        return 1
    print("\nNo double-free warning (bug NOT reproduced).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
