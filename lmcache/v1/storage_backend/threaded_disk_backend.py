# SPDX-License-Identifier: Apache-2.0
"""
Threaded / mmap variants of LocalDiskBackend for retrieve bandwidth improvement.

All variants maintain the same POSIX semantics as LocalDiskBackend:
  - 1 chunk = 1 file
  - Each read does open → read → close (no fd caching)
  - Files may be deleted by external processes at any time

Three backends:

  ThreadedDiskBackend (案A):
    Uses ThreadPoolExecutor to read multiple chunk files in parallel.
    Single-thread FSDAX limit is ~4 GB/s; 8 threads → ~24 GB/s.

  MmapDiskBackend (案B):
    Uses mmap + MAP_POPULATE per file. The kernel parallelizes page fault
    handling internally, achieving ~17 GB/s on FSDAX with a single thread.

  ThreadedMmapDiskBackend (案C):
    Combines ThreadPoolExecutor with mmap + MAP_POPULATE.

Usage:
  Set extra_config.disk_backend_type to select:
    "threaded"      → ThreadedDiskBackend
    "mmap"          → MmapDiskBackend
    "threaded_mmap" → ThreadedMmapDiskBackend
    (default)       → LocalDiskBackend (original)

  Set extra_config.disk_read_threads for thread count (default: 8).
"""

from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, List, Optional, Sequence

import asyncio
import ctypes
import mmap
import os
import time

import torch

from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey, _lmcache_nvtx_annotate
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import MemoryFormat, MemoryObj
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.local_disk_backend import LocalDiskBackend
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend

if TYPE_CHECKING:
    from lmcache.v1.cache_controller.worker import LMCacheWorker

logger = init_logger(__name__)


class ThreadedDiskBackend(LocalDiskBackend):
    """案A: ThreadPoolExecutor による並列 read.

    各ワーカーが独立にファイルを open → read → close するため、
    POSIX セマンティクスを完全に維持する。
    Python の GIL は read() syscall 中に解放されるので、
    IO バウンドな処理は真に並列に実行される。
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
        super().__init__(
            config, loop, local_cpu_backend, dst_device,
            lmcache_worker, metadata,
        )
        extra = config.extra_config or {}
        self.num_read_threads = extra.get("disk_read_threads", 8)
        self._thread_pool = ThreadPoolExecutor(
            max_workers=self.num_read_threads
        )

        # Read mode: "python" (default), "c_read", "c_mmap_memcpy",
        #            "c_mmap_avx512", "c_mmap_avx512_nt"
        self.read_mode = extra.get("disk_read_mode", "python")
        self._fast_read_lib = None
        if self.read_mode != "python":
            self._init_fast_read()

        logger.info(
            "ThreadedDiskBackend initialized with %d read threads, mode=%s",
            self.num_read_threads, self.read_mode,
        )

    def _init_fast_read(self):
        """fast_read.so を読み込む"""
        import pathlib
        so_path = pathlib.Path(__file__).parent / "fast_read.so"
        if not so_path.exists():
            logger.warning(f"fast_read.so not found at {so_path}, falling back to python mode")
            self.read_mode = "python"
            return
        self._fast_read_lib = ctypes.CDLL(str(so_path))
        self._fast_read_lib.fast_read_file.restype = ctypes.c_int
        self._fast_read_lib.fast_read_file.argtypes = [
            ctypes.c_char_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int
        ]
        mode_map = {
            "c_read": 0,
            "c_mmap_memcpy": 1,
            "c_mmap_avx512": 2,
            "c_mmap_avx512_nt": 3,
        }
        self._fast_read_mode_id = mode_map.get(self.read_mode, 0)
        logger.info(f"fast_read.so loaded, mode_id={self._fast_read_mode_id}")

    def _fast_read_file(self, buffer, path):
        """C extension でファイルを読み出し、buffer に書き込む"""
        # buffer は ctypes memoryview — data_ptr を取得
        dst_addr = ctypes.addressof(ctypes.c_ubyte.from_buffer(buffer))
        size = len(buffer)
        ret = self._fast_read_lib.fast_read_file(
            path.encode(), dst_addr, size, self._fast_read_mode_id
        )
        if ret != 0:
            logger.warning(f"fast_read_file failed for {path}")

    def read_file(self, key, buffer, path):
        """read_mode に応じて Python or C で読み出す"""
        if self._fast_read_lib is not None:
            start_time = time.time()
            self._fast_read_file(buffer, path)
            disk_read_time = time.time() - start_time
            size = len(buffer)
            logger.debug(
                f"Fast disk read size: {size} bytes, "
                f"Bandwidth: {size / disk_read_time / 1e6:.2f} MB/s"
            )
        else:
            super().read_file(key, buffer, path)

    def __str__(self):
        return "ThreadedDiskBackend"

    def batched_get_blocking(
        self,
        keys: List[CacheEngineKey],
    ) -> Optional[List[Optional[MemoryObj]]]:
        """ThreadPoolExecutor で複数チャンクを並列に読み出す."""
        if not keys:
            return None

        import time as _time

        t_lock_start = _time.perf_counter()

        # メタデータ収集は disk_lock 下で一括で行う
        # cached_positions も lock 下で取得 (key が消える race condition 回避)
        tasks = []
        with self.disk_lock:
            for key in keys:
                if key not in self.dict:
                    tasks.append(None)
                    continue
                self.cache_policy.update_on_hit(key, self.dict)
                meta = self.dict[key]
                tasks.append((
                    key, meta.path, meta.dtype, meta.shape, meta.fmt,
                    meta.cached_positions,
                ))

        t_lock_end = _time.perf_counter()

        # 存在しないキーがあったら None で埋めて返す
        if all(t is None for t in tasks):
            return None

        # Per-chunk timing
        chunk_times = []

        def _read_one(task):
            if task is None:
                return None
            key, path, dtype, shape, fmt, cached_positions = task

            t0 = _time.perf_counter()
            memory_obj = self.local_cpu_backend.allocate(shape, dtype, fmt)
            t_alloc = _time.perf_counter()

            assert memory_obj is not None
            buffer = memory_obj.byte_array
            t_bytearray = _time.perf_counter()

            self.read_file(key, buffer, path)
            t_read = _time.perf_counter()

            memory_obj.metadata.cached_positions = cached_positions

            chunk_times.append({
                'alloc': t_alloc - t0,
                'bytearray': t_bytearray - t_alloc,
                'read': t_read - t_bytearray,
                'total': t_read - t0,
            })
            return memory_obj

        t_map_start = _time.perf_counter()
        results = list(self._thread_pool.map(_read_one, tasks))
        t_map_end = _time.perf_counter()

        n_chunks = len([t for t in tasks if t is not None])
        if chunk_times:
            avg_alloc = sum(c['alloc'] for c in chunk_times) / len(chunk_times) * 1000
            avg_ba = sum(c['bytearray'] for c in chunk_times) / len(chunk_times) * 1000
            avg_read = sum(c['read'] for c in chunk_times) / len(chunk_times) * 1000
            avg_total = sum(c['total'] for c in chunk_times) / len(chunk_times) * 1000
            lock_ms = (t_lock_end - t_lock_start) * 1000
            map_ms = (t_map_end - t_map_start) * 1000
            logger.info(
                f"ThreadedDiskBackend batched_get: {n_chunks} chunks, "
                f"lock={lock_ms:.2f}ms, threadpool_map={map_ms:.2f}ms, "
                f"per-chunk avg: alloc={avg_alloc:.3f}ms, "
                f"byte_array={avg_ba:.3f}ms, read={avg_read:.3f}ms, "
                f"total={avg_total:.3f}ms"
            )

        return results if any(r is not None for r in results) else None

    def batched_async_load_bytes_from_disk(
        self,
        paths: list[str],
        keys: list[CacheEngineKey],
        memory_objs: list[MemoryObj],
        write_back: bool = False,
    ) -> list[MemoryObj]:
        """Async path も並列化する."""

        def _read_one(args):
            path, key, mem_obj = args
            buffer = mem_obj.byte_array
            self.read_file(key, buffer, path)
            cached_positions = self.dict[key].cached_positions
            mem_obj.metadata.cached_positions = cached_positions
            with self.disk_lock:
                self.dict[key].unpin()
            return mem_obj

        results = list(self._thread_pool.map(
            _read_one,
            zip(paths, keys, memory_objs, strict=False),
        ))
        return results

    def close(self) -> None:
        self._thread_pool.shutdown(wait=False)
        super().close()


class MmapDiskBackend(LocalDiskBackend):
    """案B: mmap + MAP_POPULATE による高速読み出し.

    MAP_POPULATE を指定すると、カーネルが mmap 時にページフォルトを
    並列処理し、シングルスレッドでも ~17 GB/s を達成する (FSDAX 実測)。
    mmap 後に pinned buffer へ memcpy する必要がある。
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
        super().__init__(
            config, loop, local_cpu_backend, dst_device,
            lmcache_worker, metadata,
        )
        logger.info("MmapDiskBackend initialized")

    def __str__(self):
        return "MmapDiskBackend"

    def read_file(self, key, buffer, path):
        """mmap + MAP_POPULATE で読み出し、buffer に memcpy."""
        start_time = time.time()
        size = len(buffer)

        try:
            fd = os.open(path, os.O_RDONLY)
            try:
                mm = mmap.mmap(
                    fd, size,
                    flags=mmap.MAP_PRIVATE | mmap.MAP_POPULATE,
                    prot=mmap.PROT_READ,
                )
                try:
                    # mmap region → pinned buffer への memcpy
                    # buffer は ctypes c_ubyte array の memoryview
                    # mmap.mmap には readinto() がないため、
                    # ctypes.memmove でゼロコピー転送する
                    dst_addr = ctypes.addressof(
                        ctypes.c_ubyte.from_buffer(buffer)
                    )
                    ctypes.memmove(dst_addr, mm, size)
                finally:
                    mm.close()
            finally:
                os.close(fd)
        except FileNotFoundError:
            logger.warning(f"File not found on disk: {path}")
            if self.dict.get(key, None):
                self.dict.pop(key)
            return

        disk_read_time = time.time() - start_time
        logger.debug(
            f"Disk mmap read size: {size} bytes, "
            f"Bandwidth: {size / disk_read_time / 1e6:.2f} MB/s"
        )


class ThreadedMmapDiskBackend(ThreadedDiskBackend, MmapDiskBackend):
    """案C: ThreadPoolExecutor + mmap + MAP_POPULATE.

    ThreadedDiskBackend の並列読み出しと MmapDiskBackend の
    mmap+MAP_POPULATE を組み合わせる。
    MRO により read_file は MmapDiskBackend のものが使われ、
    batched_get_blocking は ThreadedDiskBackend のものが使われる。
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
        # ThreadedDiskBackend.__init__ → LocalDiskBackend.__init__
        ThreadedDiskBackend.__init__(
            self, config, loop, local_cpu_backend, dst_device,
            lmcache_worker, metadata,
        )
        logger.info(
            "ThreadedMmapDiskBackend initialized with %d threads",
            self.num_read_threads,
        )

    def __str__(self):
        return "ThreadedMmapDiskBackend"
