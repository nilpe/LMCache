# GDSBackend 仕様書

> GPU Direct Storage: cuFile DMA による超高速GPU直結ストレージ

**ファイル**: `/home/user/LMCache/lmcache/v1/storage_backend/gds_backend.py`
**クラス**: `GdsBackend`, `WekaGdsBackend`
**責務**: cuFile管理、DMA転送、GPU メモリ直結

---

## 🎯 概要

GDSBackend は、NVIDIA cuFile を使用した GPU Direct Storage を実装。NVMe ストレージから GPU メモリへ CPU 経由なく直接DMA転送。

**特徴**:
- **超高速**: 100 μs （DMA直結）
- **低遅延**: CPU バイパス
- **DMA対応**: NVMe → GPU 直接転送

---

## 📖 主要メソッド

### `get_blocking(key) → Optional[MemoryObj]`

```python
def get_blocking(self, key: CacheEngineKey) -> Optional[MemoryObj]:
    with self.hot_lock:
        entry = self.hot_cache.get(key)
    if entry is None:
        return None

    path = entry.path
    dtype = entry.dtype
    shape = entry.shape

    # GPU メモリ割当
    memory_obj = self.memory_allocator.allocate(shape, dtype)
    if memory_obj is None:
        return None

    # DMA 転送
    ret = self._load_gds_cufile(path, 4096, memory_obj.tensor.data_ptr(),
                                memory_obj.get_size(), 0)

    if ret != memory_obj.get_size():
        return None

    return memory_obj

def _load_gds_cufile(self, path, offset, gpu_ptr, size, dev_offset):
    with self.cufile.CuFile(path, "r") as f:
        # DMA 転送: NVMe → GPU (CPU なし)
        return f.read(
            ctypes.c_void_p(gpu_ptr),
            size,
            file_offset=offset,
            dev_offset=dev_offset,
        )
```

### `submit_put_task(key, memory_obj) → Future`

```python
def submit_put_task(self, key, memory_obj) -> Optional[Future]:
    memory_obj.ref_count_up()

    # 非同期保存
    future = asyncio.run_coroutine_threadsafe(
        self._async_save_bytes_to_disk(key, memory_obj),
        self.loop
    )

    future.add_done_callback(lambda f: memory_obj.ref_count_down())
    return future

async def _async_save_bytes_to_disk(self, key, memory_obj):
    path = self._key_to_path(key)

    # メタデータ書込（CPU）
    metadata = pack_metadata(memory_obj.tensor.shape, memory_obj.tensor.dtype)
    with open(path + ".tmp", "wb") as f:
        f.write(metadata)

    # KV 書込（DMA）
    with self.cufile.CuFile(path + ".tmp", "r+") as f:
        f.write(
            ctypes.c_void_p(memory_obj.tensor.data_ptr()),
            memory_obj.get_size(),
            file_offset=4096,
            dev_offset=0,
        )

    os.rename(path + ".tmp", path)
```

---

## 🏗️ ファイル構成

### ファイルフォーマット

```
{key_hash}.kvcache.bin:
┌────────────────────────────────────┐
│ [0:4096]       メタデータ           │
│ ├─ shape: [2, num_layers, ...]     │
│ ├─ dtype: float16                  │
│ └─ fmt: MemoryFormat               │
├────────────────────────────────────┤
│ [4096:end]     KV テンソル（バイナリ）│
│ DMA転送により高速読込                │
└────────────────────────────────────┘
```

### ディレクトリ構造（GDS パス）

```
/tmp/gds/
├─ e3/
│  ├─ 22/
│  │  └─ abc123...kvcache.bin
│  └─ 45/
│     └─ ...
└─ f7/
   └─ ...
```

---

## ⚙️ cuFile 登録

### CuFileMemoryAllocator

```python
class CuFileMemoryAllocator(GPUMemoryAllocator):
    def __init__(self, size: int, device=None):
        from cufile.bindings import cuFileBufRegister

        # GPU メモリ確保（4096バイトアライン）
        super().__init__(size, device, align_bytes=4096)

        # cuFile に登録（DMA対応化）
        self.base_pointer = self.tensor.data_ptr()
        cuFileBufRegister(
            ctypes.c_void_p(self.base_pointer),
            size,
            flags=0
        )
```

---

## 🔄 DMA フロー図

```
NVMe Storage
  │ (DMA: CPU なし)
  ├─ cuFile が I/O 制御
  ├─ NVMe DMA エンジン起動
  ├─ データ転送（GPU に直結）
  └─
      ↓
  GPU VRAM
  ├─ cuFile 登録済みメモリ
  ├─ GPU compute units で直接使用可
  └─
```

---

## 📊 パフォーマンス特性

| メトリクス | 値 |
|----------|-----|
| レイテンシ | ~100 μs |
| スループット | 10+ GB/s |
| 容量 | TB 級 |
| CPU負荷 | 低（DMA） |
| DMA対応 | ✓ |

---

## WekaGdsBackend

```python
class WekaGdsBackend(GdsBackend):
    """Weka FileSystem 特化版

    特徴:
      - Weka FS 最適化
      - GPU ダイレクト設計
    """
```

---

**バージョン**: v1
**最終更新**: 2025-11-23

