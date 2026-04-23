# LMCache LocalDiskBackend Retrieve 帯域改善計画

## 問題のサマリ

FSDAX (PMEM) 上の LocalDiskBackend からの KV-Cache Retrieve が遅い。

| 項目 | 測定値 |
|------|--------|
| LMCache Retrieve (実測) | 3.3 GB/s |
| FSDAX read() (1スレッド, C/Python) | 4.1 GB/s |
| open/close オーバーヘッド込み (strace) | 2.06 GB/s |
| FSDAX read() (8スレッド) | 24.6 GB/s |
| FSDAX mmap+MAP_POPULATE (C) | 16.8 GB/s |
| GPU transfer (CPU→GPU) | 51 GB/s |
| PMEM 理論帯域 (Load, 48T) | 80 GiB/s |

### ボトルネックの内訳 (strace 実測, 1.72 GB, 126チャンク)

```
open():   220 ms (26.4%)  ← FS メタデータアクセス
read():   474 ms (56.8%)  ← 3.62 GB/s (single-thread limit)
close():  140 ms (16.7%)  ← fd 解放
total:    835 ms
```

### 現在のコードの問題点

1. `batched_get_blocking()` は `get_blocking()` をチャンクごとに逐次呼出
2. 各チャンクで open → read → close のフルサイクル
3. read と GPU transfer が完全直列
4. io_uring/AIO は FSDAX では効果なし (DAX パスでは同期フォールバック)

## 改善案

### 案1: スレッドプール並列 read (最も確実, 高効果)

**効果見積: 4.1 → 24+ GB/s (6x)**

GDS Backend が既に `ThreadPoolExecutor` を使っている。同じパターンを LocalDiskBackend に適用。

```python
# local_disk_backend.py の batched_get_blocking() を変更
def batched_get_blocking(self, keys):
    # 現在: 逐次
    # for key in keys: self.get_blocking(key)

    # 改善: ThreadPoolExecutor で並列
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(self.get_blocking, keys))
    return results
```

**利点**: 既存の GDS Backend にパターンがある。Python GIL は IO 中に解放されるので read() は並列に走る。
**リスク**: 低。read() はスレッドセーフ。disk_lock をチャンク単位ではなくバッチ単位に変更する必要あり。

---

### 案2: fd プール (open/close 削除)

**効果見積: 43% のオーバーヘッド削減 → 2.06 → 3.6+ GB/s**

チャンクファイルの fd を初回 open 後にキャッシュし、再利用する。

```python
class LocalDiskBackend:
    def __init__(self, ...):
        self.fd_cache = {}  # path → fd

    def _get_fd(self, path):
        if path not in self.fd_cache:
            self.fd_cache[path] = os.open(path, os.O_RDONLY)
        return self.fd_cache[path]

    def read_file(self, key, buffer, path):
        fd = self._get_fd(path)
        os.lseek(fd, 0, os.SEEK_SET)
        os.readv(fd, [buffer])
```

**利点**: 変更が小さい。open/close を完全に除去。
**リスク**: fd リーク防止のための生存管理が必要。LRU eviction と連動させる。

---

### 案3: チャンク統合ファイル (1ファイル + offset)

**効果見積: open/close 削除 + seek ベース → 4+ GB/s**

128チャンクを 1 ファイルにまとめ、offset + size のインデックスでアクセス。

```python
class PackedDiskBackend:
    def __init__(self, path):
        self.fd = os.open(path, os.O_RDONLY)
        self.index = {}  # key → (offset, size)

    def get_blocking(self, key):
        offset, size = self.index[key]
        buf = allocate(size)
        os.pread(self.fd, size, offset)
        return buf
```

**利点**: open/close 完全排除。pread でオフセット指定。
**リスク**: 既存のファイルレイアウトと非互換。マイグレーションが必要。

---

### 案4: mmap + MAP_POPULATE (シングルスレッドで高帯域)

**効果見積: 4.1 → 16.8 GB/s (C 実測値)**

カーネルが MAP_POPULATE で内部的にページフォルトを並列処理するため、
シングルスレッドでも高帯域が出る。

```python
def read_file_mmap(self, key, buffer, path):
    fd = os.open(path, os.O_RDONLY)
    size = os.fstat(fd).st_size
    mm = mmap.mmap(fd, size, mmap.MAP_PRIVATE | mmap.MAP_POPULATE,
                   mmap.PROT_READ)
    # mm → buffer へのコピーが必要 (ctypes.memmove)
    ctypes.memmove(buffer.data_ptr(), mm, size)
    mm.close()
    os.close(fd)
```

**利点**: カーネルレベルの並列化。Python 側の変更が小さい。
**リスク**: memcpy が追加で必要 (直接 pinned memory に mmap できない場合)。
open/close オーバーヘッドは残る。

---

### 案5: read と GPU transfer のパイプライン化

**効果見積: レイテンシ削減 (帯域は変わらないが総時間短縮)**

現在: [read chunk 0..127] → [GPU transfer 0..127]
改善: [read 0] → [GPU 0 | read 1] → [GPU 1 | read 2] → ...

```python
def batched_retrieve_pipelined(self, keys):
    load_stream = torch.cuda.Stream()

    # Read first chunk
    mem0 = self.read_chunk(keys[0])

    for i in range(1, len(keys)):
        # Overlap: GPU transfer of i-1 with disk read of i
        with torch.cuda.stream(load_stream):
            self.to_gpu(mem_objs[i-1])
        mem_objs[i] = self.read_chunk(keys[i])

    # Final GPU transfer
    with torch.cuda.stream(load_stream):
        self.to_gpu(mem_objs[-1])
    load_stream.synchronize()
```

**利点**: GPU transfer 32ms を隠蔽可能。
**リスク**: cache_engine と gpu_connector の連携変更が必要。

---

### 案6: 案1 + 案2 + 案5 の統合 (推奨)

**効果見積: 24+ GB/s (スレッド並列) + パイプライン**

```
現在:  [open+read+close] × 128 (逐次) → [GPU transfer]
       = 835ms + 33ms = 868ms (2.0 GB/s)

改善:  ThreadPool(8) × [pread from cached fd] → パイプライン GPU
       = ~70ms (read 並列) + 33ms (GPU, 隠蔽) ≈ 70ms (24 GB/s)
```

## 実装優先度

| 優先度 | 案 | 効果 | 実装コスト | リスク |
|--------|-----|------|-----------|--------|
| **1** | 案1: スレッドプール | 6x | 小 | 低 |
| **2** | 案2: fd プール | 1.7x | 小 | 低 |
| **3** | 案5: パイプライン | レイテンシ | 中 | 中 |
| 4 | 案4: mmap | 4x | 小 | 中 |
| 5 | 案3: チャンク統合 | 1.7x | 大 | 高 |

**推奨**: まず案1 (スレッドプール) を実装し、効果を確認。
その後 案2 (fd プール) を追加して open/close を排除。
