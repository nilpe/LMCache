# TokenDatabase 仕様書

> トークンID列 → キャッシュキー変換エンジン

**ファイル**: `/home/user/LMCache/lmcache/v1/token_database.py`
**クラス**: `TokenDatabase`, `ChunkedTokenDatabase`
**責務**: トークン→キー変換、チャンク分割、SHA256累積ハッシング

---

## 🎯 概要

TokenDatabase は、vLLM から受け取ったトークンID列を、ストレージで使用できるキャッシュキーに変換するコンポーネント。

**主要責務**:
1. トークンチャンク分割
2. SHA256 累積ハッシング
3. CacheEngineKey 生成
4. キャッシュ可能判定

---

## 📐 アーキテクチャ

### TokenDatabase（抽象）

```python
class TokenDatabase(ABC):
    """トークン処理の抽象インターフェース"""

    @abstractmethod
    def process_tokens(
        self,
        tokens: Union[torch.Tensor, List[int]],
        mask: Optional[torch.Tensor] = None,
        make_key: bool = True,
    ) -> Iterable[Tuple[int, int, Union[CacheEngineKey, str]]]:
        """
        トークンをプロセス（チャンク分割 + キー生成）

        Yields:
            (start, end, key): チャンク情報
        """
        pass
```

### ChunkedTokenDatabase（実装）

**ファイル行**: 67-178

```python
class ChunkedTokenDatabase(TokenDatabase):
    """
    トークンをチャンク単位で分割し、累積ハッシュでキーを生成

    フロー:
        1. トークン列をチャンク分割 (chunk_size単位)
        2. 累積SHA256ハッシング
        3. CacheEngineKey生成
    """

    def __init__(self, chunk_size: int = 256, config: Optional[Dict] = None):
        self.chunk_size = chunk_size
        self.config = config or {}

    def process_tokens(
        self,
        tokens: Union[torch.Tensor, List[int]],
        mask: Optional[torch.Tensor] = None,
        make_key: bool = True,
    ) -> Iterable[Tuple[int, int, Union[CacheEngineKey, str]]]:
        """
        トークン列をチャンク単位でプロセス

        Args:
            tokens: トークンID列 [token_0, token_1, ..., token_N]
            mask: 処理対象マスク（True=処理, False=スキップ）
            make_key: キー生成するか

        Yields:
            (start, end, key): チャンク単位の (開始, 終了, キー)

        例:
            tokens: [1, 2, ..., 1024], chunk_size=256
            ├─ yield (0, 256, key_hash1)
            ├─ yield (256, 512, key_hash2)
            ├─ yield (512, 768, key_hash3)
            └─ yield (768, 1024, key_hash4)
        """
        # ステップ1: トークン量の正規化
        if isinstance(tokens, torch.Tensor):
            token_list = tokens.cpu().tolist()
        else:
            token_list = tokens

        # ステップ2: マスク適用
        if mask is not None:
            token_list = [
                t for i, t in enumerate(token_list)
                if mask[i]
            ]

        # ステップ3: チャンク分割と累積ハッシング
        prefix_hash = ""  # 初期ハッシュ（空文字列）

        for chunk_id in range(0, len(token_list), self.chunk_size):
            start = chunk_id
            end = min(chunk_id + self.chunk_size, len(token_list))

            # チャンク抽出
            chunk_tokens = token_list[start:end]

            # SHA256 累積ハッシング
            chunk_hash = self._hash(chunk_tokens, prefix_hash)
            prefix_hash = chunk_hash  # 次のチャンクの前ハッシュに

            # キー生成
            if make_key:
                key = self._make_key_by_hash(chunk_hash)
            else:
                key = chunk_hash

            yield start, end, key
```

---

## 🔗 SHA256 累積ハッシング

### アルゴリズム

```python
def _hash(self, tokens: List[int], prefix_hash: str) -> str:
    """
    SHA256 累積ハッシング

    Args:
        tokens: トークン列
        prefix_hash: 前のチャンクのハッシュ値

    Returns:
        このチャンクのハッシュ値

    フロー:
        1. トークンをバイト列に変換
        2. prefix_hash + tokens_bytes をハッシング
        3. SHA256ハッシュを返す
    """
    # ステップ1: トークンをバイト列に変換
    if isinstance(tokens, torch.Tensor):
        tokens_bytes = tokens.cpu().to(torch.uint32).numpy().tobytes()
    else:
        tokens_bytes = array.array("I", tokens).tobytes()
        # "I" = unsigned int (4バイト)

    # ステップ2: 累積ハッシング
    combined = prefix_hash.encode("ascii") + tokens_bytes

    # ステップ3: SHA256
    return hashlib.sha256(combined).hexdigest()
```

### 累積ハッシング例

```
チャンク1: [token_0, token_1, ..., token_255] (256個)
  hash1 = SHA256("" + chunk1_bytes)
  hash1 = "abc123def456..."

チャンク2: [token_256, token_257, ..., token_511]
  hash2 = SHA256(hash1 + chunk2_bytes)
  hash2 = "fed789cba012..."

チャンク3: [token_512, token_513, ..., token_767]
  hash3 = SHA256(hash2 + chunk3_bytes)
  hash3 = "012345fedcba..."

特性:
  - チャンク順序が重要（順序変更 → 異なるハッシュ）
  - 追加チャンク（末尾）のみ異なるハッシュ
  - 削除チャンク（末尾）でハッシュ更新
```

### 効率性

```
フルハッシング（毎回全トークン）:
  tokens: [1, 2, ..., 10000]
  → SHA256計算 × 10000 回 = 遅い

累積ハッシング（チャンク単位）:
  tokens: [1, 2, ..., 10000], chunk_size=256
  → SHA256計算 × (10000/256) = ~39 回 = 高速
```

---

## 🔑 CacheEngineKey 生成

### CacheEngineKey 構造

```python
@dataclass
class CacheEngineKey:
    """キャッシュの統一キー"""
    fmt: MemoryFormat                  # KV_2LTD, KV_T2D, など
    model_name: str                    # モデル名（"bert-large"など）
    world_size: int                    # 分散時の全ワーカー数
    worker_id: int                     # ワーカーID
    chunk_hash: str                    # SHA256ハッシュ（累積）

    def to_string(self) -> str:
        """文字列表現"""
        return f"{self.fmt.value}@{self.model_name}@{self.world_size}@{self.worker_id}@{self.chunk_hash}"
        # 例: "KV_BLOB@bert-large@8@0@abc123def456..."
```

### キー生成メソッド

```python
def _make_key_by_hash(self, chunk_hash: str) -> CacheEngineKey:
    """
    ハッシュ値から CacheEngineKey を生成

    Args:
        chunk_hash: SHA256 ハッシュ値

    Returns:
        CacheEngineKey
    """
    return CacheEngineKey(
        fmt=self.config.get("fmt", MemoryFormat.KV_2LTD),
        model_name=self.config.get("model_name", "unknown"),
        world_size=self.config.get("world_size", 1),
        worker_id=self.config.get("worker_id", 0),
        chunk_hash=chunk_hash,
    )
```

### キーの例

```
Model: "llama-7b"
Tokens: [1, 2, 3, ..., 256]
  ↓ SHA256
  hash1 = "abc123def456789..."
  ↓ CacheEngineKey
  key = CacheEngineKey(
      fmt=KV_2LTD,
      model_name="llama-7b",
      world_size=1,
      worker_id=0,
      chunk_hash="abc123def456789...",
  )

string: "KV_2LTD@llama-7b@1@0@abc123def456789..."
```

---

## 🔄 フロー詳細

### Lookup フロー

```
lookup(tokens=[1, 2, ..., 1024])
│
├─ ChunkedTokenDatabase.process_tokens()
│  │
│  ├─ チャンク1: [1-256]
│  │  ├─ hash1 = SHA256("" + bytes([1-256]))
│  │  └─ key1 = CacheEngineKey(..., hash1)
│  │
│  ├─ チャンク2: [257-512]
│  │  ├─ hash2 = SHA256(hash1 + bytes([257-512]))
│  │  └─ key2 = CacheEngineKey(..., hash2)
│  │
│  ├─ チャンク3: [513-768]
│  │  ├─ hash3 = SHA256(hash2 + bytes([513-768]))
│  │  └─ key3 = CacheEngineKey(..., hash3)
│  │
│  └─ チャンク4: [769-1024]
│     ├─ hash4 = SHA256(hash3 + bytes([769-1024]))
│     └─ key4 = CacheEngineKey(..., hash4)
│
├─ for key in [key1, key2, key3, key4]:
│  │
│  ├─ key1: StorageManager.contains()? → YES
│  ├─ key2: StorageManager.contains()? → YES
│  ├─ key3: StorageManager.contains()? → YES
│  ├─ key4: StorageManager.contains()? → NO ← ここで終了
│  │
│  └─ return 768 (key4のstart)
```

### Retrieve フロー

```
retrieve(tokens=[1,...,1024], mask=[F,F,...,T,T,...])
│
├─ ChunkedTokenDatabase.process_tokens(tokens, mask)
│  │
│  ├─ mask の True 位置のみプロセス
│  ├─ 各チャンク用のキー生成
│  │
│  └─ yield (start, end, key)
│
├─ for start, end, key in ...:
│  │
│  ├─ StorageManager.get(key)
│  ├─ GPU に転送
│  └─ ref_count_down()
│
└─ return
```

---

## ⚙️ 設定・初期化

### 設定パラメータ

```yaml
# TokenDatabase 設定
chunk_size: 256                # チャンク分割サイズ

# メタデータ
model_name: "llama-7b"         # モデル名
world_size: 1                  # ワーカー数（分散時）
worker_id: 0                   # このワーカーのID
```

### 初期化例

```python
config = {
    "chunk_size": 256,
    "model_name": "bert-large",
    "world_size": 8,
    "worker_id": 0,
    "fmt": MemoryFormat.KV_2LTD,
}

token_db = ChunkedTokenDatabase(chunk_size=256, config=config)
```

---

## 🎯 使用パターン

### パターン1: 単純なプロセッシング

```python
tokens = torch.tensor([1, 2, 3, ..., 1024])

for start, end, key in token_database.process_tokens(tokens):
    print(f"Chunk [{start}:{end}] → key={key.to_string()}")

# Output:
# Chunk [0:256] → key=KV_2LTD@bert@1@0@abc123...
# Chunk [256:512] → key=KV_2LTD@bert@1@0@def456...
# Chunk [512:768] → key=KV_2LTD@bert@1@0@ghi789...
# Chunk [768:1024] → key=KV_2LTD@bert@1@0@jkl012...
```

### パターン2: マスク適用

```python
tokens = torch.tensor([1, 2, ..., 1024])
mask = torch.tensor([F, F, ..., T, T, ...])  # [0:768] スキップ

for start, end, key in token_database.process_tokens(tokens, mask=mask):
    # mask の True 部分のみ処理
    print(f"Processing chunk [{start}:{end}]")

# Output:
# Processing chunk [768:1024]  (マスク後の値)
```

### パターン3: Lookup 実装

```python
def lookup(tokens):
    num_cached = 0

    for start, end, key in token_database.process_tokens(tokens):
        if storage_manager.contains(key):
            num_cached = end  # このチャンクまでキャッシュ済み
        else:
            break  # このチャンクから先は未キャッシュ

    return num_cached
```

---

## 📊 パフォーマンス特性

### チャンクサイズの影響

```
chunk_size = 64:
  - ハッシング回数: 多（1024/64 = 16）
  - キャッシュキー数: 多
  - 粒度: 細かい
  - 利点: キャッシュヒット率向上

chunk_size = 256:
  - ハッシング回数: 中（1024/256 = 4）
  - キャッシュキー数: 中
  - 粒度: バランス型
  - 利点: スイートスポット

chunk_size = 1024:
  - ハッシング回数: 少（1024/1024 = 1）
  - キャッシュキー数: 少
  - 粒度: 粗い
  - 利点: キー管理簡潔

推奨: chunk_size = 256
```

### ハッシング速度

```
1024 トークン処理:
  累積ハッシング: ~4 × SHA256 = ~0.1 ms
  フルハッシング: ~1024 × SHA256 = ~1 ms
  差: 10倍高速化
```

---

## 🔗 関連ドキュメント

- **[06_CacheEngine.md](./06_CacheEngine.md)** - コアエンジン
- **[01_StorageManager.md](./01_StorageManager.md)** - ストレージ統合

---

**バージョン**: v1
**最終更新**: 2025-11-23

