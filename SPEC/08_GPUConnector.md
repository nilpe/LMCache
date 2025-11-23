# GPUConnector 仕様書

> MemoryObj ↔ GPU テンソル間のデータ転送

**ファイル**: `/home/user/LMCache/lmcache/v1/gpu_connector.py`
**クラス**: `GPUConnectorInterface`, `VLLMPagedMemGPUConnectorV2`
**責務**: MemoryObj とvLLM Paged KV バッファ間のデータ転送

---

## 🎯 概要

GPUConnector は、LMCache の MemoryObj（CPU/GPU）と vLLM のページ化KVバッファ間のデータ転送を担当するコンポーネント。

**主要責務**:
1. **to_gpu()**: ストレージ → vLLM GPU メモリ
2. **from_gpu()**: vLLM GPU メモリ → ストレージ
3. slot_mapping による位置指定転送

---

## 🏗️ クラス構造

```python
class GPUConnectorInterface(ABC):
    """GPU転送の抽象インターフェース"""

    @abstractmethod
    def to_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        """ストレージ → GPU"""
        pass

    @abstractmethod
    def from_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        """GPU → ストレージ"""
        pass

class VLLMPagedMemGPUConnectorV2(GPUConnectorInterface):
    """vLLM Paged Memory対応実装"""
    pass
```

---

## 📖 主要メソッド

### 1. `to_gpu()` - ストレージ → GPU

**ファイル行**: 192-228

```python
class VLLMPagedMemGPUConnectorV2(GPUConnectorInterface):
    def to_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        """
        ストレージ（CPU/GPU）からvLLM GPU バッファへ転送

        Args:
            memory_obj: ストレージから取得したMemoryObj
            start: 転送範囲の開始インデックス
            end: 転送範囲の終了インデックス
            **kwargs:
                - kvcaches: vLLM のKVテンソルペアリスト
                - slot_mapping: スロット番号列

        フロー:
            1. memory_obj からテンソルを抽出
            2. slot_mapping で物理位置を決定
            3. vLLM paged buffer にコピー
        """
        # ステップ1: パラメータ抽出
        kvcaches = kwargs["kvcaches"]          # [(k0, v0), (k1, v1), ...]
        slot_mapping = kwargs["slot_mapping"]  # [slot_0, slot_1, ...]

        # ステップ2: memory_obj のテンソル抽出
        # memory_obj.tensor 形状: [2, num_layers, tokens, hidden_size]
        #                      or [num_layers, 2, tokens, hidden_size]
        tensor = memory_obj.tensor

        # ステップ3: レイヤーごとにコピー
        for layer_id in range(len(kvcaches)):
            k_target, v_target = kvcaches[layer_id]
            # k_target: [num_blocks, block_size, num_heads, head_size]
            # v_target: [num_blocks, block_size, num_heads, head_size]

            # MemoryObj から該当レイヤーを抽出
            if tensor.shape[0] == 2:  # [2, num_layers, ...]
                k_src = tensor[0, layer_id]  # [tokens, hidden_size]
                v_src = tensor[1, layer_id]
            else:  # [num_layers, 2, ...]
                k_src = tensor[layer_id, 0]
                v_src = tensor[layer_id, 1]

            # slot_mapping でインデックス指定して copy
            # slot_mapping: [slot_0, slot_1, ..., slot_{end-start-1}]
            # → vLLM buffer のどこにコピーするか指定
            k_target[slot_mapping[start:end]].copy_(
                k_src.reshape(-1, *k_src.shape[-2:]),
                non_blocking=False,  # CPU↔GPU 同期転送
            )
            v_target[slot_mapping[start:end]].copy_(
                v_src.reshape(-1, *v_src.shape[-2:]),
                non_blocking=False,
            )
```

**slot_mapping の説明**:

```
vLLM ページ化KVバッファ:
┌────────────────────────────────────┐
│ Block 0: [slot 0, slot 1, ...]     │
│ Block 1: [slot 256, slot 257, ...] │
│ Block 2: [slot 512, slot 513, ...] │
└────────────────────────────────────┘

slot_mapping: [0, 1, 2, 256, 257, 258, ...]
  ↓
  MemoryObj の [token_0, token_1, token_2, ...]
  ↓
  vLLM の [slot_0, slot_1, slot_2, slot_256, ...]

copy_:
  k_target[slot_mapping[start:end]] = k_src[start:end]
  ↓
  k_target[[0, 1, 2, ...]] = k_src[[0, 1, 2, ...]]
```

**メモリレイアウト図**:

```
MemoryObj (ストレージから取得):
┌───────────────────────────────────┐
│ Layer 0:                          │
│  K: [token_0, token_1, ...]       │
│  V: [token_0, token_1, ...]       │
│                                   │
│ Layer 1:                          │
│  K: [token_0, token_1, ...]       │
│  V: [token_0, token_1, ...]       │
└───────────────────────────────────┘
            ↓ to_gpu()
vLLM Paged KV Buffer (GPU):
┌───────────────────────────────────┐
│ Layer 0:                          │
│  K: [block_0, block_1, ...]       │
│     ↓ slot_mapping 指定           │
│     slot_0, slot_1, ...           │
│  V: [block_0, block_1, ...]       │
│                                   │
│ Layer 1: (同様)                   │
└───────────────────────────────────┘
```

### 2. `from_gpu()` - GPU → ストレージ

**ファイル行**: 229-260

```python
class VLLMPagedMemGPUConnectorV2(GPUConnectorInterface):
    def from_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        """
        vLLM GPU バッファからストレージへ転送

        Args:
            memory_obj: 転送先（割り当て済み）のMemoryObj
            start: 転送範囲の開始インデックス
            end: 転送範囲の終了インデックス
            **kwargs:
                - kv_cache: vLLM のKV テンソル
                - slot_mapping: スロット番号列

        フロー:
            1. vLLM GPU バッファからデータ抽出
            2. MemoryObj にコピー
            3. 非同期転送対応
        """
        # ステップ1: パラメータ抽出
        kv_cache = kwargs.get("kv_cache")
        slot_mapping = kwargs.get("slot_mapping")

        # ステップ2: memory_obj の初期化
        tensor = memory_obj.tensor

        # ステップ3: レイヤーごとにコピー
        for layer_id in range(len(kv_cache)):
            k_src, v_src = kv_cache[layer_id]
            # k_src: [num_blocks, block_size, num_heads, head_size]

            # 抽出してメモリオブジェクトにコピー
            if tensor.shape[0] == 2:
                tensor[0, layer_id].reshape(-1, *tensor.shape[-2:]).copy_(
                    k_src[slot_mapping[start:end]],
                    non_blocking=True,  # 非同期転送（完了待たず）
                )
                tensor[1, layer_id].reshape(-1, *tensor.shape[-2:]).copy_(
                    v_src[slot_mapping[start:end]],
                    non_blocking=True,
                )
            else:
                tensor[layer_id, 0].reshape(-1, ...).copy_(..., non_blocking=True)
                tensor[layer_id, 1].reshape(-1, ...).copy_(..., non_blocking=True)
```

**パラメータ詳細**:

```python
# kv_cache (vLLMのKVテンソル)
kv_cache = [
    (k_layer0, v_layer0),  # Layer 0
    (k_layer1, v_layer1),  # Layer 1
    ...
]

# 各k, v の形状: [num_blocks, block_size, num_heads, head_size]
#              例: [32, 16, 32, 128]

# slot_mapping
slot_mapping = [
    0, 1, 2, 256, 257, 258, ...
]
# → どの物理スロットからデータを取得するか
```

---

## 🔄 データフロー図

### Retrieve フロー（to_gpu）

```
StorageManager.get(key)
│
└─ MemoryObj (CPU Pinned or GPU)
   ├─ tensor: [2, num_layers, tokens, hidden_size]
   │
   GPUConnector.to_gpu()
   │
   ├─ for each layer:
   │  │
   │  ├─ k_src = tensor[0, layer_id]  [tokens, hidden_size]
   │  ├─ v_src = tensor[1, layer_id]
   │  │
   │  ├─ k_target = kvcaches[layer_id][0]  [blocks, slot, heads, head_size]
   │  ├─ v_target = kvcaches[layer_id][1]
   │  │
   │  ├─ slot_mapping = [0, 1, 2, ...]
   │  │
   │  └─ k_target[slot_mapping].copy_(k_src)
   │     v_target[slot_mapping].copy_(v_src)
   │
   └─ vLLM Paged KV Buffer (GPU)
      ├─ Layer 0: [block_0, block_1, ...]
      ├─ Layer 1: [block_0, block_1, ...]
      └─ Layer N: [block_0, block_1, ...]
```

### Store フロー（from_gpu）

```
vLLM forward()
│
└─ kv_cache_new
   ├─ 形状: [num_layers, 2, tokens, hidden_size]
   │
   StorageManager.allocate(shape, dtype)
   │
   └─ MemoryObj (未初期化)
      │
      GPUConnector.from_gpu()
      │
      ├─ for each layer:
      │  │
      │  ├─ k_src = kv_cache_new[layer_id][0]
      │  ├─ v_src = kv_cache_new[layer_id][1]
      │  │
      │  ├─ k_target = memory_obj.tensor[0, layer_id]
      │  ├─ v_target = memory_obj.tensor[1, layer_id]
      │  │
      │  └─ k_target.copy_(k_src[slot_mapping], non_blocking=True)
      │     v_target.copy_(v_src[slot_mapping], non_blocking=True)
      │
      └─ MemoryObj (初期化完了)
         │
         StorageManager.put(key, memory_obj)
         │
         └─ 各バックエンドに非同期保存
```

---

## 🎯 テンソル形状変換

### 入力形状（vLLM Paged Memory）

```
vLLM KV テンソル:
  k_tensor: [num_blocks, block_size, num_heads, head_size]
           例: [32, 16, 32, 128]  (512トークン分)

  v_tensor: [num_blocks, block_size, num_heads, head_size]
           例: [32, 16, 32, 128]
```

### 出力形状（MemoryObj）

```
MemoryObj テンソル:
  tensor: [2, num_layers, num_tokens, hidden_size]
         例: [2, 32, 256, 4096]
                ↑ 2 = K, V
                ↑ 32 = num_layers
                ↑ 256 = num_tokens (このチャンク内)
                ↑ 4096 = hidden_size (全head合算)

変換:
  k_src: [tokens, hidden_size]
         [256, 4096]
  ↓ reshape(-1, num_heads, head_size)
  [256, 32, 128]

  vs

  k_target[slots]: [num_slots, num_heads, head_size]
                   [256, 32, 128]  (slot_mappingで選択)
```

---

## ⚙️ 設定・初期化

### 初期化パラメータ

```python
class VLLMPagedMemGPUConnectorV2:
    def __init__(self, config: LMCacheEngineConfig):
        self.config = config
        self.device = config.device  # "cuda:0"など

        # vLLM との互換性確認
        self.num_heads = config.num_heads
        self.head_size = config.head_size
        self.hidden_size = num_heads * head_size
```

### パラメータ計算

```
num_heads: 32        (マルチヘッドアテンション)
head_size: 128       (各ヘッドの次元)
hidden_size: 32 * 128 = 4096

num_layers: 32       (トランスフォーマーレイヤー数)
num_tokens: 256      (このチャンク内のトークン数)

MemoryObj テンソルサイズ:
  2 * 32 * 256 * 4096 * 2bytes(float16)
  = 67,108,864 bytes = 64 MB
```

---

## 🔗 非同期転送制御

### to_gpu（同期）

```python
k_target[slot_mapping].copy_(k_src, non_blocking=False)
```

**理由**: ストレージ取得直後、推論の準備が必要
- GPU での直後のアクセスを保証
- ストール回避のため同期が必要

### from_gpu（非同期）

```python
k_target.copy_(k_src, non_blocking=True)
```

**理由**: 推論後の保存、次の操作と並列化可能
- 完了待たずに次のリクエスト処理へ
- バックエンド保存と並列実行

---

## 🎯 使用パターン

### パターン1: Retrieve（読込）

```python
# ストレージから取得
memory_obj = storage_manager.get(key)

# GPU に転送
gpu_connector.to_gpu(
    memory_obj,
    start=0,
    end=256,
    kvcaches=vllm_paged_buffer,
    slot_mapping=slot_map,
)
# → memory_obj の [0:256] が vLLM GPU buffer に転送

# 使用終了
memory_obj.ref_count_down()
```

### パターン2: Store（保存）

```python
# メモリ割当
memory_obj = storage_manager.allocate(shape, dtype)

# GPU から転送
gpu_connector.from_gpu(
    memory_obj,
    start=768,
    end=1024,
    kv_cache=new_kv_tensor,
    slot_mapping=slot_map,
)
# → new_kv_tensor の [768:1024] が memory_obj に転送

# 保存開始
storage_manager.put(key, memory_obj)
```

---

## 📊 パフォーマンス特性

### 転送速度

```
CPU Pinned → GPU:  ~12 GB/s  (PCIe Gen3)
GPU → GPU:         >100 GB/s (GPU Memory)
GPU → CPU Pinned:  ~12 GB/s  (PCIe Gen3)

例: 64 MB 転送
  CPU→GPU: 64 MB / 12 GB/s ≈ 5.3 ms
```

### スケーラビリティ

```
並列度:
  - 複数レイヤーは順次処理
  - 複数リクエストは並列処理
  - slot_mapping により複数スロット同時転送

ボトルネック:
  - PCIe bandwidth（CPU Pinned）
  - GPU memory bandwidth（GPU内）
```

---

## 🔗 関連ドキュメント

- **[06_CacheEngine.md](./06_CacheEngine.md)** - コアエンジン
- **[02_MemoryManagement.md](./02_MemoryManagement.md)** - メモリモデル

---

**バージョン**: v1
**最終更新**: 2025-11-23

