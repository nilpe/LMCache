# LocalCPUBackend 仕様書

> ホットキャッシュ: CPU メモリ上の高速メモリキャッシュ

**ファイル**: `/home/user/LMCache/lmcache/v1/storage_backend/local_cpu_backend.py`
**クラス**: `LocalCPUBackend`
**責務**: ホットキャッシュ管理、LRU制御、メモリ効率

---

## 🎯 概要

LocalCPUBackend は、CPU メモリ上のホットキャッシュを管理。複数バックエンド中最速で、LRU により最頻使用データを保持します。

**特徴**:
- **超高速**: <1 μs アクセス
- **LRU管理**: メモリオーバーフロー時に自動削除
- **参照カウント統合**: 他バックエンド使用中は削除不可

---

## 📖 主要メソッド

### `get_blocking(key) → Optional[MemoryObj]`

```python
def get_blocking(self, key: CacheEngineKey) -> Optional[MemoryObj]:
    with self.cpu_lock:
        if key not in self.hot_cache:
            return None

        memory_obj = self.hot_cache[key]
        memory_obj.ref_count_up()
        self.hot_cache.move_to_end(key)  # LRU更新
        return memory_obj
```

### `contains(key) → bool`

```python
def contains(self, key: CacheEngineKey, search_range=None, pin=False) -> bool:
    with self.cpu_lock:
        return key in self.hot_cache
```

### `submit_put_task(key, memory_obj) → None`

```python
def submit_put_task(self, key: CacheEngineKey, memory_obj: MemoryObj):
    with self.cpu_lock:
        if key in self.hot_cache:
            old_obj = self.hot_cache.pop(key)
            old_obj.ref_count_down()

        self.hot_cache[key] = memory_obj
        # → ref_count はすでに +1 済み（put時）
```

---

## 🏗️ データ構造

### hot_cache

```python
hot_cache: OrderedDict[CacheEngineKey, MemoryObj]

特性:
  - Python 3.7+: 挿入順序を保持
  - move_to_end(): 要素を末尾に移動
  - LRU: 最初=LRU, 最後=MRU
  - max_size: GB単位
  - eviction: LRU削除ポリシー
```

### LRU 管理フロー

```
初期状態:
  hot_cache = {key1, key2, key3}
              (LRU)         (MRU)

access(key1):
  move_to_end(key1)
  → hot_cache = {key2, key3, key1}
                (LRU)         (MRU)

eviction (メモリ超過):
  next(iter(hot_cache)) = key2 (LRU)
  → hot_cache.pop(key2)
  → ref_count_down(key2)
```

---

## 📊 パフォーマンス特性

| メトリクス | 値 |
|----------|-----|
| アクセス速度 | <1 μs |
| サイズ | GB 級 |
| キャパシティ | メモリ制限 |

---

**バージョン**: v1
**最終更新**: 2025-11-23

