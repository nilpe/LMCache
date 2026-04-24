# SPDX-License-Identifier: Apache-2.0
"""
DevDAX Storage Backend with AVX-512 optimized reads.

FSDAX の per-file open/read/close を完全にバイパスし、
DevDAX mmap 上に直接 KV-cache チャンクを配置。
Retrieve 時は AVX-512 NT store で pinned DRAM にコピーし、
GPU が DRAM から DMA 転送する。

Usage:
  extra_config:
    disk_backend_type: "devdax"
    devdax_path: "/dev/dax0.0"
    devdax_copy_mode: "avx512_nt"  # or "memcpy", "avx512"
    disk_read_threads: 8
"""

from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, List, Optional, Sequence

import asyncio
import ctypes
import os
import pathlib
import struct
import threading
import time

import torch

from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey, DiskCacheMetadata
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import MemoryFormat, MemoryObj
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.abstract_backend import StorageBackendInterface
from lmcache.v1.storage_backend.cache_policy import get_cache_policy
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend

if TYPE_CHECKING:
    from lmcache.v1.cache_controller.worker import LMCacheWorker

logger = init_logger(__name__)

# Load fast_read.so for AVX-512 memcpy
_fast_read = None
_so_path = pathlib.Path(__file__).parent / "fast_read.so"
if _so_path.exists():
    _fast_read = ctypes.CDLL(str(_so_path))
    _fast_read.fast_memcpy_devdax.restype = ctypes.c_int
    _fast_read.fast_memcpy_devdax.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int
    ]

# Load pipeline_native.so for native double-buffer pipeline
_pipeline_native = None
_BATCH_CB_TYPE = ctypes.CFUNCTYPE(
    None, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p
)
_pipe_so_path = pathlib.Path(__file__).parent / "pipeline_native.so"
if _pipe_so_path.exists():
    _pipeline_native = ctypes.CDLL(str(_pipe_so_path))
    _pipeline_native.pipeline_double_buffer.restype = ctypes.c_int
    _pipeline_native.pipeline_double_buffer.argtypes = [
        ctypes.c_void_p,                    # base
        ctypes.POINTER(ctypes.c_int64),     # offsets
        ctypes.POINTER(ctypes.c_void_p),    # dsts
        ctypes.POINTER(ctypes.c_int64),     # sizes
        ctypes.c_int,                       # count
        ctypes.c_int,                       # batch_size
        ctypes.c_int,                       # max_threads
        ctypes.c_int,                       # mode
        _BATCH_CB_TYPE,                     # callback
        ctypes.c_void_p,                    # user_data
    ]

# Load bar1_bridge.so for GDRCopy BAR1 direct write
_bar1_bridge = None
_bar1_so_path = pathlib.Path(__file__).parent / "bar1_bridge.so"
if _bar1_so_path.exists():
    try:
        _bar1_bridge = ctypes.CDLL(str(_bar1_so_path))
        _bar1_bridge.bar1_init.restype = ctypes.c_int
        _bar1_bridge.bar1_init.argtypes = [ctypes.c_size_t]
        _bar1_bridge.bar1_get_gpu_ptr.restype = ctypes.c_ulonglong
        _bar1_bridge.bar1_get_bar1_ptr.restype = ctypes.c_ulonglong
        _bar1_bridge.bar1_get_size.restype = ctypes.c_size_t
        _bar1_bridge.bar1_copy_scatter.restype = ctypes.c_int
        _bar1_bridge.bar1_copy_scatter.argtypes = [
            ctypes.c_void_p,                    # pmem_base
            ctypes.POINTER(ctypes.c_int64),     # pmem_offsets
            ctypes.POINTER(ctypes.c_int64),     # gpu_offsets
            ctypes.POINTER(ctypes.c_int64),     # sizes
            ctypes.c_int,                       # count
            ctypes.c_int,                       # max_threads
        ]
        _bar1_bridge.bar1_cleanup.restype = None
        # H2D per-thread mode
        if hasattr(_bar1_bridge, "h2d_init"):
            _bar1_bridge.h2d_init.restype = ctypes.c_int
            _bar1_bridge.h2d_init.argtypes = [ctypes.c_size_t, ctypes.c_int]
            _bar1_bridge.parallel_h2d_chunked.restype = ctypes.c_int
            _bar1_bridge.parallel_h2d_chunked.argtypes = [
                ctypes.c_void_p,                    # pmem_base
                ctypes.c_void_p,                    # gpu_dst_base
                ctypes.POINTER(ctypes.c_int64),     # pmem_offsets
                ctypes.POINTER(ctypes.c_int64),     # gpu_offsets
                ctypes.POINTER(ctypes.c_int64),     # sizes
                ctypes.c_int,                       # count
                ctypes.c_int,                       # max_threads
            ]
            _bar1_bridge.h2d_cleanup.restype = None
        # GPU DMA direct mode (cudaHostRegister on PMEM + cudaMemcpyAsync)
        if hasattr(_bar1_bridge, "gpu_dma_init"):
            _bar1_bridge.gpu_dma_init.restype = ctypes.c_int
            _bar1_bridge.gpu_dma_init.argtypes = [
                ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int
            ]
            _bar1_bridge.parallel_gpu_dma_chunked.restype = ctypes.c_int
            _bar1_bridge.parallel_gpu_dma_chunked.argtypes = [
                ctypes.c_void_p,                    # pmem_base (unused, global)
                ctypes.c_void_p,                    # gpu_dst_base
                ctypes.POINTER(ctypes.c_int64),     # pmem_offsets
                ctypes.POINTER(ctypes.c_int64),     # gpu_offsets
                ctypes.POINTER(ctypes.c_int64),     # sizes
                ctypes.c_int,                       # count
                ctypes.c_int,                       # max_threads
            ]
            _bar1_bridge.gpu_dma_cleanup.restype = None
    except Exception:
        _bar1_bridge = None

def _load_cudart():
    """Load libcudart trying versioned names first."""
    for name in ("libcudart.so.12", "libcudart.so.11", "libcudart.so"):
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    raise OSError("libcudart not found (tried .so.12, .so.11, .so)")


# libc for mmap
_libc = ctypes.CDLL("libc.so.6", use_errno=True)
_libc.mmap.restype = ctypes.c_void_p
_libc.mmap.argtypes = [
    ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
    ctypes.c_int, ctypes.c_int, ctypes.c_long
]
_libc.munmap.restype = ctypes.c_int
_libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]

MAP_SHARED = 0x01
PROT_READ = 0x1
PROT_WRITE = 0x2


class DevDaxBackend(StorageBackendInterface):
    """DevDAX + AVX-512 高速ストレージバックエンド。

    DevDAX デバイスを丸ごと mmap し、スラブアロケータで
    チャンクを配置。Retrieve 時に AVX-512 NT store で
    pinned DRAM にコピーする。
    """

    def __init__(
        self,
        config: LMCacheEngineConfig,
        loop: asyncio.AbstractEventLoop,
        local_cpu_backend: LocalCPUBackend,
        dst_device: str = "cuda",
        lmcache_worker: Optional["LMCacheWorker"] = None,
        metadata: Optional[LMCacheMetadata] = None,
    ):
        if torch.cuda.is_available():
            super().__init__(dst_device)
        else:
            super().__init__("cpu")

        self.local_cpu_backend = local_cpu_backend
        self.cache_policy = get_cache_policy(config.cache_policy)
        self.dict = self.cache_policy.init_mutable_mapping()
        self.lock = threading.Lock()

        extra = config.extra_config or {}
        self.devdax_path = extra.get("devdax_path", "/dev/dax0.0")
        # Offset (in bytes) where this backend's slab starts inside the DAX
        # device. Non-zero is used to split one DevDAX device across multiple
        # LMCacheEngine instances (e.g. attention + Mamba sidecar). Must be a
        # 2 MiB multiple so the mmap remains PMD-huge-page aligned.
        self.devdax_offset = int(extra.get("devdax_offset", 0))
        if self.devdax_offset < 0 or self.devdax_offset % (2 * 1024 * 1024) != 0:
            raise ValueError(
                f"devdax_offset must be a non-negative 2 MiB multiple; "
                f"got 0x{self.devdax_offset:x}"
            )
        copy_mode = extra.get("devdax_copy_mode", "avx512_nt")
        self.num_threads = extra.get("disk_read_threads", 8)

        # Copy mode: 0=memcpy, 1=avx512, 2=avx512_nt, 3=read_only(prefetch)
        self._copy_mode = {"memcpy": 0, "avx512": 1, "avx512_nt": 2,
                           "prefetch": 3}.get(copy_mode, 2)

        # Read engine: "python" (ThreadPoolExecutor), "c_pthread", "rust_rayon"
        self._read_engine = extra.get("read_engine", "python")
        self._c_parallel = None
        self._rust_read = None

        if self._read_engine == "c_pthread" and _fast_read:
            _fast_read.parallel_devdax_copy.restype = ctypes.c_int
            _fast_read.parallel_devdax_copy.argtypes = [
                ctypes.c_void_p,                    # base
                ctypes.POINTER(ctypes.c_int64),     # offsets
                ctypes.POINTER(ctypes.c_void_p),    # dsts
                ctypes.POINTER(ctypes.c_int64),     # sizes
                ctypes.c_int,                       # count
                ctypes.c_int,                       # max_threads
                ctypes.c_int,                       # mode
            ]
            self._c_parallel = _fast_read.parallel_devdax_copy
            logger.info("DevDaxBackend: using C pthread parallel copy")

        elif self._read_engine == "rust_rayon":
            try:
                import lmcache_fast_read
                self._rust_read = lmcache_fast_read
                logger.info("DevDaxBackend: using Rust rayon parallel copy")
            except ImportError:
                logger.warning("Rust extension not available, falling back to python")
                self._read_engine = "python"

        # BAR1 direct write mode
        self._use_bar1 = extra.get("use_bar1", False)
        self._bar1_threads = extra.get("bar1_threads", 16)  # 16T optimal for BAR1
        self._bar1_ready = False
        if self._use_bar1 and _bar1_bridge:
            # Lazy init: will be initialized on first use (needs CUDA context)
            logger.info("DevDaxBackend: BAR1 direct write mode enabled (lazy init)")

        # Per-thread H2D (Approach A): each C thread does
        # PMEM → pinned DRAM (NT) → cudaMemcpyAsync GPU on its own stream
        self._use_h2d = extra.get("use_h2d_per_thread", False)
        self._h2d_threads = extra.get("h2d_threads", 32)
        self._h2d_streams = extra.get("h2d_streams", 16)
        self._h2d_ready = False
        if self._use_h2d and _bar1_bridge and hasattr(_bar1_bridge, "h2d_init"):
            logger.info(
                f"DevDaxBackend: H2D per-thread mode enabled "
                f"(threads={self._h2d_threads}, streams={self._h2d_streams}, lazy init)"
            )

        # GPU DMA direct: cudaHostRegister(PMEM) + cudaMemcpyAsync per chunk.
        # GPU copy engine pulls PMEM directly; no CPU staging (no DRAM bounce).
        self._use_gpu_dma = extra.get("use_gpu_dma_direct", False)
        self._gpu_dma_threads = extra.get("gpu_dma_threads", 16)
        self._gpu_dma_streams = extra.get("gpu_dma_streams", 16)
        # Register size: how many bytes of PMEM to pin for GPU visibility.
        # H100 BAR empirically ≤123 GiB; default 64 GiB covers snapshots.
        self._gpu_dma_register_gb = extra.get("gpu_dma_register_gb", 64)
        self._gpu_dma_ready = False
        if self._use_gpu_dma:
            if not _bar1_bridge or not hasattr(_bar1_bridge, "gpu_dma_init"):
                raise RuntimeError(
                    "use_gpu_dma_direct=true but bar1_bridge has no "
                    "gpu_dma_init symbol. Rebuild bar1_bridge.so."
                )
            logger.info(
                f"DevDaxBackend: GPU DMA direct mode enabled "
                f"(threads={self._gpu_dma_threads}, streams={self._gpu_dma_streams}, "
                f"register={self._gpu_dma_register_gb} GiB, eager init)"
            )

        self._thread_pool = ThreadPoolExecutor(max_workers=self.num_threads)

        # mmap DevDAX
        self._fd = os.open(self.devdax_path, os.O_RDWR)

        # Device size detection
        dev_size = 2 * 1024 * 1024 * 1024 * 1024  # 2 TiB default
        for sysfs in ["/sys/class/dax/dax0.0/size",
                      "/sys/bus/dax/devices/dax0.0/size"]:
            try:
                with open(sysfs) as f:
                    dev_size = int(f.read().strip())
                break
            except Exception:
                pass

        max_size = int(config.max_local_disk_size * 1024**3)
        available = dev_size - self.devdax_offset
        if available <= 0:
            raise ValueError(
                f"devdax_offset 0x{self.devdax_offset:x} exceeds device size "
                f"0x{dev_size:x} ({self.devdax_path})"
            )
        self._map_size = min(max_size, available)
        ALIGN = 2 * 1024 * 1024
        self._map_size = (self._map_size // ALIGN) * ALIGN

        # libpmem2 style: MAP_SHARED_VALIDATE | MAP_SYNC for DAX mapping
        # + 2MB alignment for PMD (huge page) TLB entries
        MAP_SHARED_VALIDATE = 0x03
        MAP_SYNC = 0x80000

        self._base = _libc.mmap(
            None, self._map_size, PROT_READ | PROT_WRITE,
            MAP_SHARED_VALIDATE | MAP_SYNC, self._fd, self.devdax_offset
        )
        if self._base == ctypes.c_void_p(-1).value:
            # Fallback to MAP_SHARED
            logger.warning("MAP_SYNC failed, falling back to MAP_SHARED")
            self._base = _libc.mmap(
                None, self._map_size, PROT_READ | PROT_WRITE,
                MAP_SHARED, self._fd, self.devdax_offset
            )
        if self._base == ctypes.c_void_p(-1).value:
            raise RuntimeError(f"DevDAX mmap failed: {os.strerror(ctypes.get_errno())}")

        # Check if 2MB aligned (for PMD huge page mapping)
        if self._base % (2 * 1024 * 1024) == 0:
            logger.info("DevDAX mmap is 2MB aligned → PMD huge page mapping likely")
        else:
            logger.warning(f"DevDAX mmap not 2MB aligned: {hex(self._base)}")

        # Some harnesses (e.g. Claude Code's sandbox) set PR_SET_THP_DISABLE=1,
        # which causes SIGBUS on the very first write to DAX 2MB pages.
        # Re-enable THP for this process so DAX page faults succeed.
        try:
            PR_SET_THP_DISABLE = 41
            _libc.prctl(PR_SET_THP_DISABLE, 0, 0, 0, 0)
        except Exception:
            pass

        # Touch (read-fault) to install PTEs without modifying data.
        # Write would corrupt pre-existing DAX data (snapshot restore case).
        for i in range(0, min(self._map_size, 1024 * 4096), 4096):
            _ = ctypes.c_char.from_address(self._base + i).value

        # Simple slab allocator
        self._alloc_offset = 0
        self._alloc_lock = threading.Lock()

        # Async loading support: disk_worker for non-blocking prefetch
        from lmcache.v1.storage_backend.local_disk_backend import LocalDiskWorker
        self.loop = loop
        self.disk_worker = LocalDiskWorker(loop)
        self.keys_in_request: List[CacheEngineKey] = []

        logger.info(
            f"DevDaxBackend: {self.devdax_path}"
            f"[+{self.devdax_offset / (1024**3):.1f} GiB], "
            f"mapped {self._map_size / (1024**3):.1f} GB, "
            f"copy_mode={copy_mode}({self._copy_mode}), threads={self.num_threads}"
        )

        # State snapshot: SIGUSR1 でメタデータ+アロケータ状態ダンプ、起動時に自動復元
        from lmcache.v1.storage_backend.state_snapshot import (
            register_backend_for_snapshot,
            load_backend_state,
        )
        n = load_backend_state(self)
        if n > 0:
            logger.info(
                f"Restored {n} chunks from snapshot "
                f"(alloc_offset={self._alloc_offset / (1024**3):.2f} GB)"
            )
        register_backend_for_snapshot(self)

        # GPU DMA direct: eager one-time cudaHostRegister of full configured range.
        # Lazy re-registering on staging growth was ~3-4s per occurrence and was
        # polluting retrieve latency (turn boundaries). Register once, grow
        # staging tensor independently.
        if self._use_gpu_dma:
            import time as _time
            register_bytes = int(self._gpu_dma_register_gb * (1 << 30))
            register_bytes = min(register_bytes, self._map_size)
            t0 = _time.perf_counter()
            ret = _bar1_bridge.gpu_dma_init(
                self._base, register_bytes, self._gpu_dma_streams
            )
            if ret != 0:
                raise RuntimeError(
                    f"gpu_dma_init failed (register_bytes={register_bytes}, "
                    f"streams={self._gpu_dma_streams}). "
                    "Check cudaHostRegister support for DevDAX region."
                )
            self._gpu_dma_registered_size = register_bytes
            self._gpu_dma_ready = True
            logger.info(
                f"GPU DMA init: registered {register_bytes >> 30} GiB PMEM "
                f"in {_time.perf_counter() - t0:.2f}s, streams={self._gpu_dma_streams}"
            )

    def __str__(self):
        return "DevDaxBackend"

    def _allocate_region(self, size: int) -> Optional[int]:
        """スラブアロケータ: offset を返す"""
        size = (size + 255) & ~255  # 256B align (XPLine)
        with self._alloc_lock:
            if self._alloc_offset + size > self._map_size:
                return None
            offset = self._alloc_offset
            self._alloc_offset += size
            return offset

    def _key_to_path(self, key: CacheEngineKey) -> str:
        return key.to_string().replace("/", "-")

    def contains(self, key: CacheEngineKey, pin: bool = False) -> bool:
        with self.lock:
            if key not in self.dict:
                return False
            if pin:
                self.dict[key].pin()
            return True

    def submit_put_task(
        self,
        key: CacheEngineKey,
        memory_obj: MemoryObj,
        transfer_spec=None,
        on_complete_callback=None,
    ) -> None:
        """Store: pinned DRAM → DevDAX (AVX-512 memcpy)"""
        kv_chunk = memory_obj.tensor
        assert kv_chunk is not None
        buffer = memory_obj.byte_array
        size = len(buffer)

        offset = self._allocate_region(size)
        if offset is None:
            logger.warning("DevDaxBackend: no space left")
            memory_obj.ref_count_down()
            return

        # Copy to DevDAX
        dst = self._base + offset
        src = ctypes.addressof(ctypes.c_ubyte.from_buffer(buffer))
        if _fast_read:
            _fast_read.fast_memcpy_devdax(dst, src, size, self._copy_mode)
        else:
            ctypes.memmove(dst, src, size)

        # Store metadata
        shape = memory_obj.metadata.shape
        dtype = memory_obj.metadata.dtype
        fmt = memory_obj.metadata.fmt
        cached_positions = memory_obj.metadata.cached_positions
        memory_obj.ref_count_down()

        with self.lock:
            self.dict[key] = DiskCacheMetadata(
                path=str(offset),  # offset as "path"
                size=size,
                shape=shape,
                dtype=dtype,
                fmt=fmt,
                cached_positions=cached_positions,
            )
            self.cache_policy.update_on_put(key)

        if on_complete_callback:
            try:
                on_complete_callback(key)
            except Exception as e:
                logger.warning(f"callback failed: {e}")

    def batched_submit_put_task(self, keys, memory_objs, transfer_spec=None,
                                on_complete_callback=None, **kwargs):
        for key, mo in zip(keys, memory_objs):
            self.submit_put_task(key, mo, transfer_spec=transfer_spec,
                                on_complete_callback=on_complete_callback)

    def get_blocking(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        with self.lock:
            if key not in self.dict:
                return None
            self.cache_policy.update_on_hit(key, self.dict)
            meta = self.dict[key]
            offset = int(meta.path)
            dtype = meta.dtype
            shape = meta.shape
            fmt = meta.fmt

        # Allocate pinned DRAM buffer
        memory_obj = self.local_cpu_backend.allocate(shape, dtype, fmt)
        assert memory_obj is not None

        buffer = memory_obj.byte_array
        size = len(buffer)

        # AVX-512 copy: DevDAX → pinned DRAM
        src = self._base + offset
        dst = ctypes.addressof(ctypes.c_ubyte.from_buffer(buffer))
        if _fast_read:
            _fast_read.fast_memcpy_devdax(dst, src, size, self._copy_mode)
        else:
            ctypes.memmove(dst, src, size)

        # Recover metadata
        with self.lock:
            memory_obj.metadata.cached_positions = self.dict[key].cached_positions

        return memory_obj

    def _init_bar1(self, total_size: int):
        """Lazy init BAR1 staging buffer (needs CUDA context)"""
        if self._bar1_ready:
            return True
        if not _bar1_bridge:
            return False
        ret = _bar1_bridge.bar1_init(total_size)
        if ret == 0:
            self._bar1_ready = True
            self._bar1_gpu_ptr = _bar1_bridge.bar1_get_gpu_ptr()
            self._bar1_cpu_ptr = _bar1_bridge.bar1_get_bar1_ptr()
            logger.info(f"BAR1 init OK: gpu=0x{self._bar1_gpu_ptr:x}, size={total_size>>20} MB")
            return True
        logger.warning("BAR1 init failed, falling back to DRAM path")
        return False

    def batched_get_blocking(
        self,
        keys: List[CacheEngineKey],
    ) -> Optional[List[Optional[MemoryObj]]]:
        """ThreadPoolExecutor で並列に DevDAX → pinned DRAM コピー
        BAR1 モード時は DevDAX → GPU BAR1 直接書き込み"""
        if not keys:
            return None

        t0 = time.perf_counter()

        tasks = []
        with self.lock:
            for key in keys:
                if key not in self.dict:
                    tasks.append(None)
                    continue
                self.cache_policy.update_on_hit(key, self.dict)
                meta = self.dict[key]
                tasks.append((key, int(meta.path), meta.dtype, meta.shape, meta.fmt))

        t_lock = time.perf_counter()

        if all(t is None for t in tasks):
            return None

        # Allocate all memory objects first
        valid_tasks = [(i, t) for i, t in enumerate(tasks) if t is not None]
        memory_objs = [None] * len(tasks)
        offsets_list = []
        dst_ptrs_list = []
        sizes_list = []

        for idx, (key, offset, dtype, shape, fmt) in valid_tasks:
            mo = self.local_cpu_backend.allocate(shape, dtype, fmt)
            assert mo is not None
            memory_objs[idx] = mo
            buf = mo.byte_array
            offsets_list.append(offset)
            dst_ptrs_list.append(ctypes.addressof(ctypes.c_ubyte.from_buffer(buf)))
            sizes_list.append(len(buf))

        t_alloc = time.perf_counter()
        n = len(valid_tasks)

        if n > 0 and self._read_engine == "c_pthread" and self._c_parallel:
            # Phase 2: C pthread parallel copy
            c_offsets = (ctypes.c_int64 * n)(*offsets_list)
            c_dsts = (ctypes.c_void_p * n)(*dst_ptrs_list)
            c_sizes = (ctypes.c_int64 * n)(*sizes_list)
            self._c_parallel(
                self._base, c_offsets, c_dsts, c_sizes,
                n, self.num_threads, self._copy_mode
            )

        elif n > 0 and self._read_engine == "rust_rayon" and self._rust_read:
            # Phase 3: Rust rayon parallel copy
            self._rust_read.parallel_devdax_copy(
                self._base, offsets_list, dst_ptrs_list,
                sizes_list, self.num_threads,
            )

        else:
            # Default: Python ThreadPoolExecutor
            def _read_one(args):
                idx, (key, offset, dtype, shape, fmt) = args
                mo = memory_objs[idx]
                src = self._base + offset
                dst = ctypes.addressof(ctypes.c_ubyte.from_buffer(mo.byte_array))
                if _fast_read:
                    _fast_read.fast_memcpy_devdax(dst, src, len(mo.byte_array), self._copy_mode)
                else:
                    ctypes.memmove(dst, src, len(mo.byte_array))
                return mo

            list(self._thread_pool.map(_read_one, valid_tasks))

        t_copy = time.perf_counter()

        # Recover metadata
        for idx, (key, offset, dtype, shape, fmt) in valid_tasks:
            with self.lock:
                memory_objs[idx].metadata.cached_positions = self.dict[key].cached_positions

        elapsed = (time.perf_counter() - t0) * 1000
        total_bytes = sum(sizes_list)
        copy_ms = (t_copy - t_alloc) * 1000
        copy_bw = (total_bytes / (1024**3)) / (t_copy - t_alloc) if t_copy > t_alloc else 0
        logger.info(
            f"DevDaxBackend batched_get ({self._read_engine}): {n} chunks in {elapsed:.1f}ms "
            f"[lock={((t_lock - t0) * 1000):.1f}ms, alloc={((t_alloc - t_lock) * 1000):.1f}ms, "
            f"copy={copy_ms:.1f}ms ({copy_bw:.1f} GB/s)]"
        )

        return memory_objs if any(m is not None for m in memory_objs) else None

    # ---- Native pipeline (Plan A/C) ----

    def pipeline_get_blocking(
        self,
        keys: List[CacheEngineKey],
        batch_size: int,
        gpu_callback,
    ) -> Optional[List[Optional[MemoryObj]]]:
        """C ネイティブ double-buffer pipeline.

        1. 全チャンクのメモリを一括確保
        2. C 側で work-stealing disk copy + サブバッチ完了コールバック
        3. コールバック内で GPU 転送を発行
        4. Python ループゼロ
        """
        if not keys or _pipeline_native is None:
            return None

        t0 = time.perf_counter()

        # Metadata lookup
        tasks = []
        with self.lock:
            for key in keys:
                if key not in self.dict:
                    tasks.append(None)
                    continue
                self.cache_policy.update_on_hit(key, self.dict)
                meta = self.dict[key]
                tasks.append((key, int(meta.path), meta.dtype, meta.shape, meta.fmt))

        t_lock = time.perf_counter()

        valid_tasks = [(i, t) for i, t in enumerate(tasks) if t is not None]
        if not valid_tasks:
            return None

        # Pre-allocate ALL memory objects at once
        memory_objs = [None] * len(tasks)
        offsets_list = []
        dst_ptrs_list = []
        sizes_list = []

        for idx, (key, offset, dtype, shape, fmt) in valid_tasks:
            mo = self.local_cpu_backend.allocate(shape, dtype, fmt)
            assert mo is not None
            memory_objs[idx] = mo
            buf = mo.byte_array
            offsets_list.append(offset)
            dst_ptrs_list.append(ctypes.addressof(ctypes.c_ubyte.from_buffer(buf)))
            sizes_list.append(len(buf))

        t_alloc = time.perf_counter()

        n = len(valid_tasks)
        c_offsets = (ctypes.c_int64 * n)(*offsets_list)
        c_dsts = (ctypes.c_void_p * n)(*dst_ptrs_list)
        c_sizes = (ctypes.c_int64 * n)(*sizes_list)

        # Callback: C calls this after each sub-batch disk copy completes
        # The callback runs in a C pthread, ctypes auto-acquires GIL
        def _on_subbatch_ready(batch_idx, start, end, user_data):
            sub_valid = valid_tasks[start:end]
            sub_objs = [memory_objs[idx] for idx, _ in sub_valid]
            sub_blocks = [(tasks[idx], idx) for idx, _ in sub_valid]

            # Recover metadata for this sub-batch
            for idx, (key, offset, dtype, shape, fmt) in sub_valid:
                with self.lock:
                    memory_objs[idx].metadata.cached_positions = \
                        self.dict[key].cached_positions

            # Call GPU transfer
            if gpu_callback:
                gpu_callback(batch_idx, sub_objs, sub_valid)

        cb = _BATCH_CB_TYPE(_on_subbatch_ready)

        # Single C call: handles all sub-batches with double-buffering
        _pipeline_native.pipeline_double_buffer(
            self._base, c_offsets, c_dsts, c_sizes,
            n, batch_size, self.num_threads, self._copy_mode,
            cb, None,
        )

        t_done = time.perf_counter()

        total_bytes = sum(sizes_list)
        copy_ms = (t_done - t_alloc) * 1000
        logger.info(
            f"DevDaxBackend pipeline_get (native): {n} chunks, "
            f"batch={batch_size}, total={copy_ms:.1f}ms "
            f"[lock={((t_lock - t0) * 1000):.1f}ms, "
            f"alloc={((t_alloc - t_lock) * 1000):.1f}ms]"
        )

        return memory_objs if any(m is not None for m in memory_objs) else None

    # ---- BAR1 direct retrieve ----

    def _ensure_bar1(self, total_size: int):
        """BAR1 staging: PyTorch GPU tensor + GDRCopy BAR1 マップ"""
        if self._bar1_ready and hasattr(self, '_bar1_staging') and \
           self._bar1_staging.nbytes >= total_size:
            return True

        if not _bar1_bridge:
            return False

        # Setup bar1_init_external
        if not hasattr(_bar1_bridge, '_ext_setup'):
            _bar1_bridge.bar1_init_external.restype = ctypes.c_int
            _bar1_bridge.bar1_init_external.argtypes = [
                ctypes.c_ulonglong, ctypes.c_size_t
            ]
            _bar1_bridge._ext_setup = True

        # Cleanup previous
        if self._bar1_ready:
            _bar1_bridge.bar1_cleanup()
            self._bar1_ready = False

        # Use cuMemAlloc for BAR1 (faster WC mapping) + D2D copy to PyTorch
        align = 64 * 1024
        alloc_size = ((total_size + align - 1) // align) * align

        ret = _bar1_bridge.bar1_init(alloc_size)
        if ret != 0:
            logger.warning("BAR1 init failed")
            self._use_bar1 = False
            return False

        # Also allocate a PyTorch GPU tensor for D2D copy target
        self._bar1_staging = torch.empty(alloc_size, dtype=torch.uint8, device="cuda")

        self._bar1_gpu_ptr = _bar1_bridge.bar1_get_gpu_ptr()
        self._bar1_cpu_ptr = _bar1_bridge.bar1_get_bar1_ptr()
        self._bar1_buf_size = _bar1_bridge.bar1_get_size()
        self._bar1_ready = True
        logger.info(
            f"BAR1 init: gpu=0x{self._bar1_gpu_ptr:x}, "
            f"bar1=0x{self._bar1_cpu_ptr:x}, size={alloc_size>>20} MB, "
            f"pytorch_owned={'bar1_init_external' in str(ret)}"
        )
        return True

    def bar1_batched_get(
        self,
        keys: List[CacheEngineKey],
    ):
        """PMEM → GPU BAR1 直接書き込み (DRAM バイパス)。

        Returns: (valid_tasks, gpu_offsets, sizes, total_size, metadata_list)
        GPU staging buffer に書き込み済みの状態で返す。
        呼び出し元で staging buffer → KV cache の reshape を行う。
        """
        if not keys or not self._use_bar1 or not _bar1_bridge:
            return None

        t0 = time.perf_counter()

        # Metadata lookup
        valid = []
        with self.lock:
            for key in keys:
                if key not in self.dict:
                    break
                self.cache_policy.update_on_hit(key, self.dict)
                meta = self.dict[key]
                valid.append((key, int(meta.path), meta.dtype, meta.shape,
                              meta.fmt, meta.cached_positions))

        if not valid:
            return None

        t_lock = time.perf_counter()

        # Compute sizes and GPU staging offsets
        n = len(valid)
        pmem_offsets = []
        gpu_offsets = []
        sizes_list = []
        gpu_off = 0

        for key, offset, dtype, shape, fmt, cpos in valid:
            numel = 1
            for s in shape:
                numel *= s
            elem_size = torch.tensor([], dtype=dtype).element_size()
            chunk_bytes = numel * elem_size
            chunk_bytes = (chunk_bytes + 255) & ~255

            pmem_offsets.append(offset)
            gpu_offsets.append(gpu_off)
            sizes_list.append(chunk_bytes)
            gpu_off += chunk_bytes

        total_size = gpu_off

        # Init BAR1
        if not self._ensure_bar1(total_size):
            return None

        t_init = time.perf_counter()

        # PMEM → BAR1 scatter copy (AVX-512 NT store, 16T)
        c_pmem = (ctypes.c_int64 * n)(*pmem_offsets)
        c_gpu = (ctypes.c_int64 * n)(*gpu_offsets)
        c_sizes = (ctypes.c_int64 * n)(*sizes_list)

        _bar1_bridge.bar1_copy_scatter(
            self._base, c_pmem, c_gpu, c_sizes,
            n, self._bar1_threads,
        )

        t_copy = time.perf_counter()

        # D2D bulk copy: cuMemAlloc staging → PyTorch tensor (HBM, ~3 TB/s)
        _cudart = _load_cudart()
        _cudart.cudaMemcpy.restype = ctypes.c_int
        _cudart.cudaMemcpy.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int
        ]
        _cudart.cudaMemcpy(
            self._bar1_staging.data_ptr(),
            self._bar1_gpu_ptr,
            total_size,
            3,  # cudaMemcpyDeviceToDevice
        )

        t_d2d = time.perf_counter()

        copy_bw = (total_size / (1024**3)) / (t_copy - t_init) if t_copy > t_init else 0
        d2d_ms = (t_d2d - t_copy) * 1000
        logger.info(
            f"DevDaxBackend bar1_copy: {n} chunks, "
            f"lock={((t_lock - t0)*1000):.1f}ms, "
            f"bar1={((t_copy - t_init)*1000):.1f}ms ({copy_bw:.1f} GB/s), "
            f"d2d={d2d_ms:.1f}ms"
        )

        return {
            "valid": valid,
            "gpu_offsets": gpu_offsets,
            "sizes": sizes_list,
            "total_size": total_size,
            "gpu_ptr": self._bar1_gpu_ptr,
            "staging_tensor": self._bar1_staging,  # PyTorch GPU tensor (D2D copied)
        }

    # ---- Per-thread H2D (Approach A) ----

    def _ensure_h2d(self, total_size: int):
        """Lazy init H2D staging: PyTorch GPU tensor + pinned DRAM bounce buf"""
        if self._h2d_ready and hasattr(self, "_h2d_staging") and \
           self._h2d_staging.nbytes >= total_size:
            return True

        if not _bar1_bridge or not hasattr(_bar1_bridge, "h2d_init"):
            return False

        align = 64 * 1024
        alloc_size = ((total_size + align - 1) // align) * align

        # Cleanup previous if growing
        if self._h2d_ready:
            _bar1_bridge.h2d_cleanup()
            self._h2d_ready = False

        # Allocate GPU staging tensor (PyTorch owns it; we pass its data_ptr)
        self._h2d_staging = torch.empty(alloc_size, dtype=torch.uint8, device="cuda")

        # Init pinned DRAM bounce buffer + CUDA streams
        ret = _bar1_bridge.h2d_init(alloc_size, self._h2d_streams)
        if ret != 0:
            logger.warning("h2d_init failed")
            self._use_h2d = False
            return False

        self._h2d_buf_size = alloc_size
        self._h2d_ready = True
        logger.info(
            f"H2D init: gpu_staging={alloc_size>>20} MB, "
            f"streams={self._h2d_streams}"
        )
        return True

    def h2d_batched_get(self, keys: List[CacheEngineKey]):
        """PMEM → pinned DRAM → GPU staging buffer via per-thread cudaMemcpyAsync.

        Returns: dict with 'valid', 'gpu_offsets', 'sizes', 'total_size',
                 'staging_tensor' (GPU tensor with all chunks concatenated).
        """
        if not keys or not self._use_h2d or not _bar1_bridge:
            return None

        t0 = time.perf_counter()

        valid = []
        with self.lock:
            for key in keys:
                if key not in self.dict:
                    break
                self.cache_policy.update_on_hit(key, self.dict)
                meta = self.dict[key]
                valid.append((key, int(meta.path), meta.dtype, meta.shape,
                              meta.fmt, meta.cached_positions))

        if not valid:
            return None

        t_lock = time.perf_counter()

        n = len(valid)
        pmem_offsets = []
        gpu_offsets = []
        sizes_list = []
        gpu_off = 0

        for key, offset, dtype, shape, fmt, cpos in valid:
            numel = 1
            for s in shape:
                numel *= s
            elem_size = torch.tensor([], dtype=dtype).element_size()
            chunk_bytes = numel * elem_size
            chunk_bytes = (chunk_bytes + 255) & ~255

            pmem_offsets.append(offset)
            gpu_offsets.append(gpu_off)
            sizes_list.append(chunk_bytes)
            gpu_off += chunk_bytes

        total_size = gpu_off

        if not self._ensure_h2d(total_size):
            return None

        t_init = time.perf_counter()

        c_pmem = (ctypes.c_int64 * n)(*pmem_offsets)
        c_gpu = (ctypes.c_int64 * n)(*gpu_offsets)
        c_sizes = (ctypes.c_int64 * n)(*sizes_list)

        _bar1_bridge.parallel_h2d_chunked(
            self._base,
            self._h2d_staging.data_ptr(),
            c_pmem, c_gpu, c_sizes,
            n, self._h2d_threads,
        )

        t_copy = time.perf_counter()

        copy_bw = (total_size / (1024**3)) / (t_copy - t_init) if t_copy > t_init else 0
        logger.info(
            f"DevDaxBackend h2d_get: {n} chunks, "
            f"lock={((t_lock - t0)*1000):.1f}ms, "
            f"h2d={((t_copy - t_init)*1000):.1f}ms ({copy_bw:.1f} GB/s)"
        )

        return {
            "valid": valid,
            "gpu_offsets": gpu_offsets,
            "sizes": sizes_list,
            "total_size": total_size,
            "staging_tensor": self._h2d_staging,
        }

    # ---- GPU DMA direct retrieve ----

    def _ensure_gpu_dma(self, total_size: int):
        """Grow GPU staging tensor if needed. PMEM is registered eagerly in __init__."""
        if not self._gpu_dma_ready:
            raise RuntimeError(
                "GPU DMA not initialized (should be done in __init__). "
                "Check that use_gpu_dma_direct was true when backend was constructed."
            )

        align = 64 * 1024
        alloc_size = ((total_size + align - 1) // align) * align

        if hasattr(self, "_gpu_dma_staging") and self._gpu_dma_staging.nbytes >= alloc_size:
            return True

        # Grow staging tensor. GPU alloc is fast (~ms), PMEM registration is NOT touched.
        if hasattr(self, "_gpu_dma_staging"):
            del self._gpu_dma_staging
        self._gpu_dma_staging = torch.empty(
            alloc_size, dtype=torch.uint8, device="cuda"
        )
        logger.info(f"GPU DMA staging grew to {alloc_size >> 20} MB")
        return True

    def gpu_dma_batched_get(self, keys: List[CacheEngineKey]):
        """PMEM → GPU staging buffer via cudaMemcpyAsync (GPU copy engine pull).

        No CPU staging, no DRAM bounce. Source is registered PMEM device ptr.
        Returns: dict with 'valid', 'gpu_offsets', 'sizes', 'total_size',
                 'staging_tensor' (GPU tensor, chunks concatenated).
        """
        if not keys or not self._use_gpu_dma or not _bar1_bridge:
            return None

        t0 = time.perf_counter()

        valid = []
        with self.lock:
            for key in keys:
                if key not in self.dict:
                    break
                self.cache_policy.update_on_hit(key, self.dict)
                meta = self.dict[key]
                valid.append((key, int(meta.path), meta.dtype, meta.shape,
                              meta.fmt, meta.cached_positions))

        if not valid:
            return None

        t_lock = time.perf_counter()

        n = len(valid)
        pmem_offsets = []
        gpu_offsets = []
        sizes_list = []
        gpu_off = 0

        for key, offset, dtype, shape, fmt, cpos in valid:
            numel = 1
            for s in shape:
                numel *= s
            elem_size = torch.tensor([], dtype=dtype).element_size()
            chunk_bytes = numel * elem_size
            chunk_bytes = (chunk_bytes + 255) & ~255

            pmem_offsets.append(offset)
            gpu_offsets.append(gpu_off)
            sizes_list.append(chunk_bytes)
            gpu_off += chunk_bytes

        total_size = gpu_off

        # Check PMEM offsets are within registered region
        max_pmem_off = max(po + sz for po, sz in zip(pmem_offsets, sizes_list))
        if not self._ensure_gpu_dma(total_size):
            raise RuntimeError(
                "GPU DMA direct: _ensure_gpu_dma failed "
                f"(total_size={total_size}, register_gb="
                f"{self._gpu_dma_register_gb})"
            )
        if max_pmem_off > self._gpu_dma_registered_size:
            raise RuntimeError(
                f"GPU DMA direct: chunk offset {max_pmem_off} "
                f"({max_pmem_off / (1024 ** 3):.2f} GB) exceeds registered "
                f"region {self._gpu_dma_registered_size} "
                f"({self._gpu_dma_registered_size / (1024 ** 3):.2f} GB). "
                "Increase gpu_dma_register_gb or reduce workload."
            )

        t_init = time.perf_counter()

        c_pmem = (ctypes.c_int64 * n)(*pmem_offsets)
        c_gpu = (ctypes.c_int64 * n)(*gpu_offsets)
        c_sizes = (ctypes.c_int64 * n)(*sizes_list)

        _bar1_bridge.parallel_gpu_dma_chunked(
            self._base,  # unused on C side; global registered ptr used
            self._gpu_dma_staging.data_ptr(),
            c_pmem, c_gpu, c_sizes,
            n, self._gpu_dma_threads,
        )

        t_copy = time.perf_counter()

        copy_bw = (total_size / (1024**3)) / (t_copy - t_init) \
            if t_copy > t_init else 0
        logger.info(
            f"DevDaxBackend gpu_dma_get: {n} chunks, "
            f"lock={((t_lock - t0) * 1000):.1f}ms, "
            f"init={((t_init - t_lock) * 1000):.1f}ms, "
            f"dma={((t_copy - t_init) * 1000):.1f}ms ({copy_bw:.1f} GB/s)"
        )

        return {
            "valid": valid,
            "gpu_offsets": gpu_offsets,
            "sizes": sizes_list,
            "total_size": total_size,
            "staging_tensor": self._gpu_dma_staging,
        }

    # ---- Async loading support ----

    async def batched_async_contains(
        self,
        lookup_id: str,
        keys: List[CacheEngineKey],
        pin: bool = False,
    ) -> int:
        """async パスで呼ばれる contains チェック"""
        num_hit = 0
        with self.lock:
            for key in keys:
                if key not in self.dict:
                    return num_hit
                if pin:
                    self.dict[key].pin()
                    self.keys_in_request.append(key)
                num_hit += 1
        return num_hit

    async def batched_get_non_blocking(
        self,
        lookup_id: str,
        keys: List[CacheEngineKey],
        transfer_spec: Any = None,
    ) -> List[MemoryObj]:
        """async パスで呼ばれる非同期プリフェッチ。
        disk_worker 経由で _batched_load をスレッドプール上で実行。"""
        mem_objs: List[MemoryObj] = []
        tasks = []

        with self.lock:
            for key in keys:
                assert key in self.dict, f"Key {key} not found in DevDaxBackend"
                meta = self.dict[key]
                offset = int(meta.path)
                dtype = meta.dtype
                shape = meta.shape
                fmt = meta.fmt

                memory_obj = self.local_cpu_backend.allocate(shape, dtype, fmt)
                assert memory_obj is not None

                self.dict[key].pin()
                self.cache_policy.update_on_hit(key, self.dict)

                memory_obj.pin()
                mem_objs.append(memory_obj)
                tasks.append((key, offset, len(memory_obj.byte_array)))

        return await self.disk_worker.submit_task(
            "prefetch",
            self._batched_load,
            tasks=tasks,
            keys=keys,
            memory_objs=mem_objs,
        )

    def _batched_load(
        self,
        tasks: List,
        keys: List[CacheEngineKey],
        memory_objs: List[MemoryObj],
    ) -> List[MemoryObj]:
        """実際の DevDAX → DRAM コピー (read_engine に応じて並列化)"""
        t0 = time.perf_counter()
        n = len(tasks)

        # Build arrays for C/Rust engines
        offsets_list = [t[1] for t in tasks]
        sizes_list = [t[2] for t in tasks]
        dst_ptrs_list = [
            ctypes.addressof(ctypes.c_ubyte.from_buffer(mo.byte_array))
            for mo in memory_objs
        ]

        if n > 0 and self._read_engine == "c_pthread" and self._c_parallel:
            c_offsets = (ctypes.c_int64 * n)(*offsets_list)
            c_dsts = (ctypes.c_void_p * n)(*dst_ptrs_list)
            c_sizes = (ctypes.c_int64 * n)(*sizes_list)
            self._c_parallel(
                self._base, c_offsets, c_dsts, c_sizes,
                n, self.num_threads, self._copy_mode
            )

        elif n > 0 and self._read_engine == "rust_rayon" and self._rust_read:
            self._rust_read.parallel_devdax_copy(
                self._base, offsets_list, dst_ptrs_list,
                sizes_list, self.num_threads,
            )

        else:
            def _copy_one(args):
                (key, offset, size), mem_obj = args
                buffer = mem_obj.byte_array
                src = self._base + offset
                dst = ctypes.addressof(ctypes.c_ubyte.from_buffer(buffer))
                if _fast_read:
                    _fast_read.fast_memcpy_devdax(dst, src, size, self._copy_mode)
                else:
                    ctypes.memmove(dst, src, size)

            list(self._thread_pool.map(
                _copy_one,
                zip(tasks, memory_objs, strict=False),
            ))

        # Recover metadata and unpin
        for (key, offset, size), mem_obj in zip(tasks, memory_objs):
            mem_obj.metadata.cached_positions = self.dict[key].cached_positions
            with self.lock:
                self.dict[key].unpin()

        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(
            f"DevDaxBackend async_load ({self._read_engine}): "
            f"{n} chunks in {elapsed:.1f}ms"
        )
        return memory_objs

    # ---- Abstract method implementations ----

    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        return False  # DevDAX put is synchronous, no pending tasks

    def get_allocator_backend(self):
        return self.local_cpu_backend

    def pin(self, key: CacheEngineKey) -> bool:
        with self.lock:
            if key in self.dict:
                self.dict[key].pin()
                return True
        return False

    def unpin(self, key: CacheEngineKey) -> bool:
        with self.lock:
            if key in self.dict:
                self.dict[key].unpin()
                return True
        return False

    def remove(self, key: CacheEngineKey, force: bool = True) -> bool:
        with self.lock:
            if key in self.dict:
                del self.dict[key]
                return True
        return False

    def close(self):
        self._thread_pool.shutdown(wait=False)
        if self._h2d_ready and _bar1_bridge and hasattr(_bar1_bridge, "h2d_cleanup"):
            _bar1_bridge.h2d_cleanup()
            self._h2d_ready = False
        if self._gpu_dma_ready and _bar1_bridge and \
                hasattr(_bar1_bridge, "gpu_dma_cleanup"):
            _bar1_bridge.gpu_dma_cleanup()
            self._gpu_dma_ready = False
        _libc.munmap(self._base, self._map_size)
        os.close(self._fd)
