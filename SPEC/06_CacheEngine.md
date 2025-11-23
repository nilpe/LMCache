# CacheEngine 仕様書

> LMCache のコアエンジン: lookup, retrieve, store の3つの核操作

**ファイル**: `/home/user/LMCache/lmcache/v1/cache_engine.py`
**クラス**: `CacheEngine`
**責務**: キャッシュ管理ロジック、トークンバッファリング、I/Oオーケストレーション

---

## 🎯 概要

CacheEngine は、vLLM の lookup/retrieve/store 要求を処理するコアコンポーネント。TokenDatabase と StorageManager を組み合わせ、KV-Cache のライフサイクルを管理します。

**3つの核操作**:
1. **lookup(token_ids)** → キャッシュ済みトークン数
2. **retrieve(token_ids, mask, ...)** → KV テンソル取得
3. **store(token_ids, mask, kv_cache, ...)** → KV テンソル保存

---

## 🏗️ クラス構造

```python
class CacheEngine:
    # 依存コンポーネント
    token_database: TokenDatabase
    storage_manager: StorageManager
    gpu_connector: GPUConnectorInterface

    # 設定
    config: LMCacheEngineConfig
    metadata: LMCacheEngineMetadata

    # 統計情報
    stats_monitor: Optional[StatsMonitor]
```

---

## 📖 主要メソッド

### 1. `lookup(tokens) → int`

**責務**: キャッシュ済みトークンインデックスを返却

**ファイル行**: 382-428

```python
def lookup(
    self,
    tokens: Union[torch.Tensor, List[int]],
    search_range: Optional[List[str]] = None,
    pin: bool = False,
) -> int:
    """
    トークン列に対して、キャッシュされているトークン数を返す

    Args:
        tokens: トークンID列 [token_0, token_1, ..., token_N]
        search_range: 検索対象バックエンド ("p2p", "local", etc)
        pin: ピン状態を要求するか

    Returns:
        int: キャッシュ済みトークンインデックス
             例: 256なら、[0:256]がキャッシュ済み、256:以降未キャッシュ

    フロー:
        1. TokenDatabase でトークンをチャンクに分割
        2. 各チャンクに対してキャッシュキー生成
        3. StorageManager で存在確認（短絡評価）
        4. 最初の未キャッシュインデックスを返す
    """
    end = 0
    search_local = True
    search_p2p = self.enable_p2p and (search_range is None or "p2p" in search_range)

    # ステップ1: TokenDatabase でキャッシュキー生成
    for start, end, key in self.token_database.process_tokens(tokens):
        # ステップ2: ローカルストレージをチェック（LocalCPU/LocalDisk）
        if search_local:
            if self.storage_manager.contains(key, search_range=["local"], pin=pin):
                continue  # 見つかった、次へ
            else:
                search_local = False  # 以降はp2pのみ

        # ステップ3: P2P/リモートストレージをチェック
        if search_p2p:
            if self.lookup_server.lookup(key):
                continue  # 見つかった、次へ

        # ステップ4: 見つからなかった → このインデックスから先は未キャッシュ
        return start

    # ステップ5: すべてのトークンがキャッシュされている
    return end
```

**フロー図**:

```
lookup([token_0, ..., token_1023])
│
├─ process_tokens() 呼び出し
│  ├─ Chunk 1: [token_0-255]  → key_1
│  ├─ Chunk 2: [token_256-511] → key_2
│  ├─ Chunk 3: [token_512-767] → key_3
│  └─ Chunk 4: [token_768-1023] → key_4
│
├─ key_1: contains() → YES
├─ key_2: contains() → YES
├─ key_3: contains() → YES
├─ key_4: contains() → NO ← ここから未キャッシュ
│
└─ return 768 (key_4 の start インデックス)
```

**返り値の意味**:

```
lookup() 返り値 = N

意味: [0:N] はキャッシュ済み、[N:end] は未キャッシュ

例:
  lookup() = 256 → [0:256] キャッシュ済み, [256:] 未キャッシュ
  lookup() = 0   → すべて未キャッシュ
  lookup() = 1024 → すべてキャッシュ済み
```

### 2. `retrieve(tokens, mask, **kwargs) → torch.Tensor`

**責務**: KV テンソル取得（ブロッキング）

**ファイル行**: 295-367

```python
@torch.inference_mode()
def retrieve(
    self,
    tokens: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    **kwargs,
) -> torch.Tensor:
    """
    トークン列に対応する KV キャッシュを ストレージから取得

    Args:
        tokens: トークンID列
        mask: 取得対象位置マスク
              True = 取得対象, False = スキップ
        **kwargs: GPU転送時のパラメータ
                  - kvcaches: vLLM Paged KV Buffer
                  - slot_mapping: スロットマッピング

    Returns:
        torch.Tensor: 取得できたトークンマスク

    フロー:
        1. TokenDatabase でキャッシュキー生成
        2. StorageManager.get() でMemoryObj取得
        3. GPUConnector でGPUに転送
        4. mask 更新（どれが取得できたか）
    """
    ret_mask = torch.zeros_like(tokens, dtype=torch.bool, device="cpu")

    # ステップ1: TokenDatabase でキャッシュキー生成
    for start, end, key in self.token_database.process_tokens(tokens, mask):
        # ステップ2: ストレージから MemoryObj 取得（ブロッキング）
        memory_obj = self.storage_manager.get(key)

        if memory_obj is None:
            # キャッシュ未検出 → この位置以降は取得不可
            break

        # ステップ3: mask 更新（この位置は取得可）
        ret_mask[start:end] = True

        # ステップ4: GPU に転送
        self.gpu_connector.to_gpu(memory_obj, start, end, **kwargs)

        # ステップ5: メモリ解放（ref_count_down）
        memory_obj.ref_count_down()

    return ret_mask
```

**パラメータ詳細**:

```python
# kvcaches パラメータ（vLLMからの指定）
kwargs = {
    "kvcaches": [
        (k_tensor_layer0, v_tensor_layer0),  # layer 0
        (k_tensor_layer1, v_tensor_layer1),  # layer 1
        ...
    ],
    "slot_mapping": [0, 1, 2, ..., slot_N],  # スロット番号列
}

# mask パラメータ（取得対象の指定）
mask = torch.tensor([
    False,  # token 0: スキップ（既にGPUにある）
    False,  # token 1: スキップ
    True,   # token 2: 取得対象
    True,   # token 3: 取得対象
    ...
])
```

**フロー図**:

```
retrieve(tokens=[0,1,...,1023], mask=[F,F,T,T,...])
│
├─ process_tokens() → キャッシュキーを生成
│  ├─ [0-255]   → key_1
│  ├─ [256-511] → key_2
│  ├─ [512-767] → key_3
│  └─ [768-1023] → key_4
│
├─ key_1.get_blocking() → MemoryObj or None
│  ├─ 見つかった → to_gpu() + ref_count_down()
│  │               ret_mask[0:256] = True
│  └─ 見つからない → break
│
└─ return ret_mask
   └─ 取得できた位置が True
```

### 3. `store(tokens, mask, kv_cache, **kwargs) → None`

**責務**: KV テンソル保存（非同期）

**ファイル行**: 429-544

```python
@torch.inference_mode()
def store(
    self,
    tokens: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    kv_cache: Optional[torch.Tensor] = None,
    **kwargs,
) -> None:
    """
    KV キャッシュを ストレージに保存

    Args:
        tokens: トークンID列
        mask: 保存対象位置マスク
        kv_cache: KV テンソル（デバイス上）
        **kwargs: GPU転送時のパラメータ

    特徴:
        - 非同期保存（完了待たず）
        - 複数バックエンドに並列保存
        - 既キャッシュ部分はスキップ

    フロー:
        1. lookup() で既キャッシュ部分を確認
        2. skip_leading_tokens以降のみ保存
        3. allocate() でメモリ割当
        4. GPU → CPU/GPU メモリコピー
        5. StorageManager.put() で非同期保存
    """
    # ステップ0: 保存設定確認
    if not self.config.enable_save:
        return

    # ステップ1: lookup で既キャッシュ部分を確認
    skip_leading_tokens = self.lookup(tokens)
    # → skip_leading_tokens まではスキップ

    # ステップ2: save_mask 作成
    save_mask = torch.zeros_like(tokens, dtype=torch.bool)
    save_mask[skip_leading_tokens:] = True

    # ステップ3: TokenDatabase でキャッシュキー生成
    for start, end, key in self.token_database.process_tokens(tokens, save_mask):
        # ステップ4: ストレージに既に存在するか確認
        if self.storage_manager.contains(key):
            continue  # 既存 → スキップ

        # ステップ5: メモリ割当
        memory_obj = self.storage_manager.allocate(
            self._infer_shape(kv_cache),  # 形状推論
            kv_cache.dtype,
        )
        if memory_obj is None:
            logger.warning("Failed to allocate memory")
            continue

        # ステップ6: GPU → memory_obj へコピー
        self.gpu_connector.from_gpu(
            memory_obj,
            start, end,
            kv_cache=kv_cache,
            slot_mapping=kwargs.get("slot_mapping"),
        )

        # ステップ7: 非同期保存を開始
        self.storage_manager.put(key, memory_obj)
        # → 各バックエンドが非同期に保存開始
        # → 完了時に自動的に ref_count_down()
```

**保存フロー図**:

```
store(tokens, kv_cache)
│
├─ lookup(tokens) → skip_leading_tokens = 512
│  └─ [0:512] は既キャッシュ, [512:] が新規
│
├─ save_mask 作成
│  ├─ [0:512]   = False (スキップ)
│  └─ [512:end] = True  (保存対象)
│
├─ for each chunk in [512:end]:
│  │
│  ├─ contains(key) → 既存?
│  │  ├─ YES → skip
│  │  └─ NO → 保存
│  │
│  ├─ allocate(shape, dtype)
│  │  └─ CPU/GPU メモリ割当
│  │
│  ├─ from_gpu(memory_obj, kv_cache)
│  │  └─ GPU → memory_obj コピー
│  │
│  └─ put(key, memory_obj)
│     └─ 全バックエンド非同期保存開始
│
└─ return (完了待たず)
```

---

## 📊 トークン処理フロー

### TokenDatabase との連携

```python
# process_tokens() 呼び出し例

tokens = torch.tensor([1, 2, 3, ..., 1024])
chunk_size = 256

for start, end, key in token_database.process_tokens(tokens):
    # start=0,   end=256,  key=CacheEngineKey(hash1)
    # start=256, end=512,  key=CacheEngineKey(hash2)
    # start=512, end=768,  key=CacheEngineKey(hash3)
    # start=768, end=1024, key=CacheEngineKey(hash4)

    # 各チャンク [start:end] に対してキャッシュキー key が対応
```

### SHA256 累積ハッシング

```
Chunk 1: [token_0-255]
  hash1 = SHA256("" + chunk1_bytes)

Chunk 2: [token_256-511]
  hash2 = SHA256(hash1 + chunk2_bytes)

Chunk 3: [token_512-767]
  hash3 = SHA256(hash2 + chunk3_bytes)

特性:
  - チャンク追加時のみハッシュ変更
  - 前チャンクと同じなら同じハッシュ（再利用可能）
  - チャンク削除は新しいハッシュ（別キャッシュ）
```

---

## 🔄 ライフサイクル管理

### トークンバッファリング

```
Request:
  token_ids: [1, 2, 3, ..., 1024]

Phase 1: Lookup
  ├─ キャッシュ確認
  ├─ lookup() = 768
  └─ [0:768] キャッシュ済み, [768:1024] 新規

Phase 2: Retrieve
  ├─ [0:768] を storage_manager.get()
  └─ GPU に転送

Phase 3: Forward
  ├─ GPU で推論実行
  └─ 新規 KV キャッシュ生成

Phase 4: Store
  ├─ 新規 KV を storage_manager.put()
  └─ 全バックエンド非同期保存
```

---

## ⚙️ 設定パラメータ

```yaml
# CacheEngine 設定
enable_save: true              # 保存機能有効化
enable_load: true              # 読込機能有効化
chunk_size: 256                # チャンク分割サイズ

# P2P / 分散
enable_p2p: true               # P2P検索有効化
enable_controller: true        # コントローラー機能

# メモリ管理
use_layerwise: false           # レイヤー単位保存
enable_blending: false         # キャッシュブレンディング
```

---

## 🎯 使用パターン

### パターン1: 通常のリクエスト処理

```python
engine = CacheEngine(config, ...)

request_tokens = torch.tensor([1, 2, 3, ..., 1024])
kv_cache_new = torch.randn(2, 32, 256, 128)  # GPU上

# ステップ1: Lookup
num_cached = engine.lookup(request_tokens)
print(f"Cached: {num_cached}, New: {len(request_tokens) - num_cached}")

# ステップ2: Retrieve
mask = engine.retrieve(
    request_tokens,
    kvcaches=vllm_paged_kv,
    slot_mapping=slot_map,
)
print(f"Retrieved mask: {mask}")

# ステップ3: Forward
# ... GPU 推論 ...

# ステップ4: Store
engine.store(
    request_tokens,
    kv_cache=kv_cache_new,
    slot_mapping=slot_map,
)
# → 非同期保存開始
```

### パターン2: Lookup のみ（vLLM Scheduler側）

```python
# Scheduler スレッド
num_cached_lmcache = engine.lookup(request.prompt_token_ids)

load_spec = LoadSpec(
    vllm_cached_tokens=num_computed_tokens,  # vLLMのキャッシュ
    lmcache_cached_tokens=num_cached_lmcache,  # LMCacheのキャッシュ
)
```

---

## 📈 パフォーマンス特性

### Lookup パフォーマンス

```
Token数       処理時間     ハッシュ計算
100           ~0.1 ms      ~0.5 ms
1000          ~1 ms        ~5 ms
10000         ~10 ms       ~50 ms

ボトルネック: SHA256 ハッシング
最適化: チャンク単位でハッシング（全トークン不要）
```

### Retrieve パフォーマンス

```
バックエンド別:
  LocalCPU    <1 μs/MB    (メモリアクセス)
  GDS         ~100 μs     (DMA)
  LocalDisk   ~1 ms/MB    (I/O)
  Remote      ~10 ms      (RPC)
```

---

## 🔗 関連クラス

- **TokenDatabase**: トークン処理
- **StorageManager**: ストレージ統合
- **GPUConnector**: GPU転送
- **CacheEngineConfig**: 設定

---

**バージョン**: v1
**最終更新**: 2025-11-23

