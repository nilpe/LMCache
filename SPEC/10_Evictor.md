# Evictor 仕様書

> キャッシュ削除ポリシー: LRU / LFU

**ファイル**: `/home/user/LMCache/lmcache/v1/storage_backend/evictor/`
**クラス**: `BaseEvictor`, `LRUEvictor`, `LFUEvictor`
**責務**: キャッシュサイズ管理、自動削除ポリシー

---

## 🎯 概要

Evictor は、ストレージ容量超過時に自動的に古い（または少ない）キャッシュを削除するコンポーネント。LRU（Least Recently Used）と LFU（Least Frequently Used）の2つのポリシーをサポート。

---

## 📖 主要メソッド

### BaseEvictor（抽象）

```python
class BaseEvictor(ABC):
    @abstractmethod
    def update_on_hit(self, key, cache_dict) -> None:
        """キャッシュ hit 時の更新"""
        pass

    @abstractmethod
    def update_on_put(self, cache_dict, cache_size) -> Tuple[List, PutStatus]:
        """put時の eviction 判定

        Returns:
            (evict_keys, status)
        """
        pass
```

### LRUEvictor（実装）

**ファイル**: `/home/user/LMCache/lmcache/v1/storage_backend/evictor/lru_evictor.py`

```python
class LRUEvictor(BaseEvictor):
    """Least Recently Used ポリシー"""

    def __init__(self, max_cache_size: float = 10.0):
        self.MAX_CACHE_SIZE = int(max_cache_size * 1024**3)  # GB → bytes
        self.current_cache_size = 0.0

    def update_on_hit(self, key, cache_dict) -> None:
        """アクセス時: MRU に昇格"""
        cache_dict.move_to_end(key)  # OrderedDict の最後に移動

    def update_on_put(self, cache_dict, new_size) -> Tuple[List[CacheEngineKey], PutStatus]:
        """put時: サイズオーバーなら LRU から削除"""
        evict_keys = []

        if new_size > self.MAX_CACHE_SIZE:
            return [], PutStatus.ILLEGAL  # 単一キャッシュが大きすぎる

        # サイズオーバーチェック
        while new_size + self.current_cache_size > self.MAX_CACHE_SIZE:
            # LRU（最初のキー）を削除
            evict_key = next(iter(cache_dict))
            evict_metadata = cache_dict[evict_key]

            if evict_metadata.is_pinned:
                break  # pinned はスキップ

            self.current_cache_size -= evict_metadata.size
            evict_keys.append(evict_key)

        self.current_cache_size += new_size
        return evict_keys, PutStatus.LEGAL
```

---

## 🔄 フロー

### Hit 時

```
get_blocking(key)
│
├─ MemoryObj 取得
│
└─ Evictor.update_on_hit(key)
   └─ OrderedDict.move_to_end(key)
      └─ key を MRU (最後) に移動
```

### Put 時

```
put(key, memory_obj)
│
├─ Evictor.update_on_put(cache_dict, size)
│  │
│  ├─ size > MAX_CACHE_SIZE? → ILLEGAL
│  │
│  └─ サイズチェック:
│     └─ while current + new > MAX:
│        ├─ LRU 候補取得: next(iter(cache_dict))
│        ├─ pinned? → skip
│        ├─ evict_keys に追加
│        └─ current_cache_size -= evict_size
│
└─ evict_keys を削除
   └─ ref_count_down()
```

---

## 📊 ポリシー比較

| 特性 | LRU | LFU |
|------|-----|-----|
| 判定基準 | 最後アクセス時刻 | アクセス頻度 |
| メモリ効率 | 中程度 | 良（頻度基準） |
| キャッシュヒット率 | 中程度 | 高 |
| 実装複雑度 | 低 | 中 |
| CPU オーバーヘッド | 低 | 中 |
| 用途 | 標準 | ワーキングセット明確 |

---

## ⚙️ パラメータ

```yaml
# Evictor 設定
max_local_cpu_size: 10         # GB単位
max_local_disk_size: 500       # GB単位

evictor:
  enable_lfu: false            # False=LRU, True=LFU
  cache_evict_interval: 100    # イテレーション間隔
```

---

## 🎯 安全性保証

### Pinned 保護

```python
if evict_metadata.is_pinned:
    break  # pinned は削除不可
```

### ref_count 保護

```python
# バックエンド側で検査
if obj.get_ref_count() > 1:
    continue  # 使用中は削除不可
```

---

**バージョン**: v1
**最終更新**: 2025-11-23

