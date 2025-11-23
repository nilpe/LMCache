# LocalDiskBackend 仕様書

> ウォームキャッシュ: ローカルNVMe/SSD 上の非同期I/O

**ファイル**: `/home/user/LMCache/lmcache/v1/storage_backend/local_disk_backend.py`
**クラス**: `LocalDiskBackend`
**責務**: ローカルディスク管理、非同期I/O、aiofiles

---

## 🎯 概要

LocalDiskBackend は、ローカル NVMe/SSD のウォームキャッシュを管理。非同期I/O により CPU ブロッキング回避。

**特徴**:
- **中速**: ~500 μs レイテンシ
- **大容量**: TB 級ストレージ対応
- **非同期**: asyncio による非ブロッキングI/O

---

## 📖 主要メソッド

### `get_blocking(key) → Optional[MemoryObj]`

```python
def get_blocking(self, key: CacheEngineKey) -> Optional[MemoryObj]:
    self.disk_lock.acquire()

    if key not in self.dict:
        self.disk_lock.release()
        return None

    metadata = self.dict[key]
    self.evictor.update_on_hit(key, self.dict)

    # ディスク → CPUメモリ読込
    memory_obj = self.load_bytes_from_disk(
        path=metadata.path,
        dtype=metadata.dtype,
        shape=metadata.shape,
    )
    self.disk_lock.release()
    return memory_obj

def load_bytes_from_disk(self, path, dtype, shape):
    # メモリ割当
    memory_obj = self.local_cpu_backend.allocate(shape, dtype)

    # ディスク読込（同期）
    with open(path, "rb") as f:
        f.readinto(memory_obj.byte_array)

    return memory_obj
```

### `submit_put_task(key, memory_obj) → Future`

```python
def submit_put_task(self, key, memory_obj) -> Optional[Future]:
    memory_obj.ref_count_up()

    # 非同期保存タスク送信
    future = asyncio.run_coroutine_threadsafe(
        self._async_save_bytes_to_disk(key, memory_obj),
        self.loop
    )

    # コールバック: 完了時に ref_count_down()
    future.add_done_callback(
        lambda f: memory_obj.ref_count_down()
    )

    return future

async def _async_save_bytes_to_disk(self, key, memory_obj):
    path = self._key_to_path(key)

    # aiofiles で非同期書込
    async with aiofiles.open(path, "wb") as f:
        await f.write(memory_obj.byte_array)
```

---

## 🏗️ ファイル構成

### ディレクトリ構造

```
cache_dir/
├─ e3/
│  ├─ 22/
│  │  ├─ abc123...pt
│  │  └─ def456...pt
│  └─ 45/
│     └─ ...
└─ f7/
   └─ ...

キー → パス変換:
  key.to_string() = "KV_BLOB@model@1@0@abc123def456..."
  path = "cache_dir/e3/22/abc123def456...pt"
```

### DiskCacheMetadata

```python
@dataclass
class DiskCacheMetadata:
    path: str              # ファイルパス
    size: int              # ファイルサイズ (bytes)
    shape: torch.Size      # テンソル形状
    dtype: torch.dtype     # データ型
    fmt: MemoryFormat      # メモリフォーマット
    is_pinned: bool        # ピン状態
```

---

## 📊 パフォーマンス特性

| メトリクス | 値 |
|----------|-----|
| レイテンシ | ~500 μs |
| スループット | ~5 GB/s |
| 容量 | TB 級 |
| CPU負荷 | 中程度 |

---

**バージョン**: v1
**最終更新**: 2025-11-23

