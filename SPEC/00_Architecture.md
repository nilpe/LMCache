# LMCache アーキテクチャ仕様書

> LMCache 全体システムアーキテクチャと主要コンポーネント間の関係

**ファイル**: 複数（アーキテクチャレベルのドキュメント）
**対応クラス**: システム全体
**バージョン**: v1

---

## 📐 全体アーキテクチャ

```
┌────────────────────────────────────────────────────────────────┐
│                         vLLM (Llama Model)                     │
│ ┌──────────────────────────────────────────────────────────┐   │
│ │  Scheduler/Worker                                        │   │
│ │  ├─ request.prompt_token_ids                            │   │
│ │  ├─ request.slot_mapping                                │   │
│ │  └─ forward(kv_cache)                                   │   │
│ └────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────┘
                              ↓ (ZMQ RPC)
┌────────────────────────────────────────────────────────────────┐
│                    LMCache Integration Layer                    │
│ ┌──────────────────────────────────────────────────────────┐   │
│ │  LMCacheConnectorV1Impl (vllm_v1_adapter.py)           │   │
│ │  ├─ LMCacheLookupClient/Server (ZMQ)                   │   │
│ │  ├─ start_load_kv() / wait_for_save()                  │   │
│ │  └─ GPUConnector インスタンス                           │   │
│ └──────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────┐
│                      LMCache Core Engine                        │
│ ┌──────────────────────────────────────────────────────────┐   │
│ │  CacheEngine (cache_engine.py)                           │   │
│ │  ├─ lookup(token_ids) → キャッシュ済みトークン数          │   │
│ │  ├─ retrieve(token_ids) → KV テンソル (GPU)             │   │
│ │  └─ store(token_ids, kv_tensor) → 永続化                │   │
│ │                                                          │   │
│ │  TokenDatabase (token_database.py)                       │   │
│ │  ├─ ChunkedTokenDatabase                                │   │
│ │  ├─ process_tokens() → キャッシュキー生成                │   │
│ │  └─ SHA256 累積ハッシング                                │   │
│ │                                                          │   │
│ │  GPUConnector (gpu_connector.py)                         │   │
│ │  ├─ MemoryObj ↔ GPU Tensor 転送                         │   │
│ │  └─ vLLM Paged KV バッファ対応                          │   │
│ └──────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────┐
│                  Storage Abstraction Layer                      │
│ ┌──────────────────────────────────────────────────────────┐   │
│ │  StorageManager (storage_manager.py)                     │   │
│ │  ├─ get(key) → MemoryObj (バックエンド順序探索)          │   │
│ │  ├─ put(key, memory_obj) → 全バックエンド保存            │   │
│ │  └─ allocate(shape, dtype) → メモリ割当                 │   │
│ │                                                          │   │
│ │  MemoryObj + メモリアロケータ                            │   │
│ │  ├─ TensorMemoryObj                                      │   │
│ │  ├─ MemoryAllocatorInterface                            │   │
│ │  │  ├─ TensorMemoryAllocator (First-fit)                │   │
│ │  │  ├─ CuFileMemoryAllocator (GDS)                      │   │
│ │  │  └─ MixedMemoryAllocator (Pinned + Buffer)           │   │
│ │  └─ LRU Evictor                                          │   │
│ └──────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────┐
│               Concrete Storage Backends                         │
│ ┌──────────────┐ ┌──────────────┐ ┌──────────┐ ┌────────────┐ │
│ │ LocalCPU     │ │ LocalDisk    │ │ Remote   │ │ GDS        │ │
│ │ Backend      │ │ Backend      │ │ Backend  │ │ Backend    │ │
│ │              │ │              │ │          │ │            │ │
│ │ • hot_cache  │ │ • /cache/    │ │ • Redis  │ │ • /tmp/gds │ │
│ │ • OrderedDict│ │ • Async I/O  │ │ • Msgpack│ │ • cuFile   │ │
│ │ • LRU        │ │ • aiofiles   │ │ • async  │ │ • DMA      │ │
│ │ • CPU Memory │ │ • CPU Memory │ │ • RPC    │ │ • GPU Mem  │ │
│ └──────────────┘ └──────────────┘ └──────────┘ └────────────┘ │
│                                                                 │
│                    Storage Backends Interface                   │
│             (StorageBackendInterface - abstract)               │
│  ├─ get_blocking(key) → Optional[MemoryObj]                    │
│  ├─ contains(key) → bool                                       │
│  └─ submit_put_task(key, memory_obj) → Future                  │
└────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────┐
│              Physical Storage & Memory                          │
│                                                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────────┐   │
│  │CPU VRAM  │  │NVMe SSD  │  │Redis/    │  │GPU VRAM +   │   │
│  │(Pinned)  │  │(Local)   │  │P2P       │  │cuFile       │   │
│  └──────────┘  └──────────┘  └──────────┘  └─────────────┘   │
│                                                                 │
│  速度: GPU >> LocalDisk ≈ GDS > CPU > Remote                  │
│  容量: Remote > LocalDisk > GPU > LocalCPU                     │
│  遅延: LocalCPU < GDS << LocalDisk << Remote                   │
└────────────────────────────────────────────────────────────────┘
```

---

## 🔄 主要フロー

### 1️⃣ Lookup フロー（キャッシュヒット確認）

```
vLLM Scheduler
  |
  | request.prompt_token_ids = [1, 2, 3, ..., 1024]
  |
  ↓ (ZMQ REQ)
LMCacheLookupClient
  |
  | token_ids を Msgpack エンコード
  |
  ↓ (IPC: ipc://lmcache_rpc_port_*)
LMCacheLookupServer (別プロセス)
  |
  ↓
CacheEngine.lookup(token_ids)
  |
  ├─ TokenDatabase.process_tokens(token_ids)
  │  └─ token_ids → CacheEngineKey リスト
  │     (SHA256 累積ハッシング)
  │
  ├─ for each key:
  │  │
  │  ├─ LocalCPUBackend.contains(key)?
  │  │  └─ YES → 継続
  │  │
  │  ├─ LocalDiskBackend.contains(key)?
  │  │  └─ YES → 継続
  │  │
  │  ├─ RemoteBackend.lookup(key)?
  │  │  └─ YES → 継続
  │  │
  │  └─ 見つからない → return start_index
  │
  └─ すべて見つかった → return end_index

  ↓ (4バイト整数)
int (キャッシュ済みトークン数)

  ↓ (ZMQ REP)
vLLM
  |
  └─ LoadSpec 更新
     └─ vllm_cached_tokens, lmcache_cached_tokens
```

### 2️⃣ Retrieve フロー（KV テンソル取得）

```
vLLM Worker
  |
  | forward() 開始
  |
  ↓
LMCacheConnectorV1Impl.start_load_kv(forward_context)
  |
  ├─ request.load_spec 確認
  │
  ↓
CacheEngine.retrieve(token_ids, mask, ...)
  |
  ├─ TokenDatabase.process_tokens(token_ids)
  │  └─ キャッシュキー列生成
  │
  ├─ for each key:
  │  │
  │  ├─ StorageManager.get(key)
  │  │  │
  │  │  ├─ LocalCPUBackend.get_blocking()
  │  │  │  └─ hot_cache[key] → 即返却
  │  │  │
  │  │  ├─ LocalDiskBackend.get_blocking()
  │  │  │  ├─ ディスク読込（async）
  │  │  │  ├─ allocate(shape, dtype)
  │  │  │  └─ write_back(key, memory_obj)
  │  │  │
  │  │  ├─ RemoteBackend.get_blocking()
  │  │  │  ├─ Redis/P2P 取得（async）
  │  │  │  ├─ deserialize()
  │  │  │  └─ write_back()
  │  │  │
  │  │  └─ GDSBackend.get_blocking()
  │  │     ├─ allocate(GPU Memory)
  │  │     ├─ DMA 転送（cuFile）
  │  │     └─ return GPU MemoryObj
  │  │
  │  ├─ MemoryObj 取得成功
  │  │
  │  └─ GPUConnector.to_gpu(memory_obj, ...)
  │     │
  │     ├─ memory_obj.tensor [CPU/GPU]
  │     │
  │     ├─ vLLM Paged KV Buffer コピー
  │     │
  │     └─ memory_obj.ref_count_down()
  │
  └─ return mask

  ↓
vLLM Worker
  |
  └─ GPU 推論実行可能
```

### 3️⃣ Store フロー（KV キャッシュ保存）

```
vLLM Worker
  |
  | forward() 完了
  | kv_cache = model.forward(...)
  |
  ↓
LMCacheConnectorV1Impl.wait_for_save()
  |
  ├─ request.save_spec 確認
  │
  ↓
CacheEngine.store(token_ids, mask, kv_cache, ...)
  |
  ├─ TokenDatabase.process_tokens(token_ids)
  │  └─ キャッシュキー列生成
  │
  ├─ for each key:
  │  │
  │  ├─ StorageManager.allocate(shape, dtype)
  │  │  └─ CPU/GPU メモリ割当
  │  │
  │  ├─ GPUConnector.from_gpu(memory_obj, ...)
  │  │  ├─ GPU kv_cache → memory_obj.tensor コピー
  │  │  └─ non_blocking=True
  │  │
  │  ├─ StorageManager.put(key, memory_obj)
  │  │  │
  │  │  ├─ LocalCPUBackend.submit_put_task()
  │  │  │  └─ hot_cache[key] = memory_obj (同期)
  │  │  │
  │  │  ├─ LocalDiskBackend.submit_put_task()
  │  │  │  ├─ asyncio で非同期保存
  │  │  │  ├─ aiofiles で書込
  │  │  │  └─ Future 返却
  │  │  │
  │  │  ├─ RemoteBackend.submit_put_task()
  │  │  │  ├─ serializer.serialize()
  │  │  │  ├─ Redis へ async 送信
  │  │  │  └─ Future 返却
  │  │  │
  │  │  └─ GDSBackend.submit_put_task()
  │  │     ├─ allocate(GPU Memory)
  │  │     ├─ cuFile で DMA 書込
  │  │     └─ Future 返却
  │  │
  │  └─ Future.add_done_callback()
  │     └─ memory_obj.ref_count_down()
  │
  └─ return

  ↓
vLLM
  |
  └─ 次リクエストへ（非同期保存は継続中）
```

---

## 🏗️ コンポーネント責務分離

### vLLM Integration Layer
**責務**: vLLM との通信、リクエスト/レスポンス処理

- `vllm_v1_adapter.py`
  - ZMQ サーバー/クライアント管理
  - lookup/retrieve/store API 提供
  - Forward Context → LMCache オペレーション変換

### Core Engine Layer
**責務**: キャッシュロジック、トークン処理、KV転送

- `cache_engine.py`: lookup/retrieve/store 核
- `token_database.py`: トークン → キー変換
- `gpu_connector.py`: MemoryObj ↔ GPU テンソル

### Storage Abstraction Layer
**責務**: 統一インターフェース、バックエンド管理、メモリ割当

- `storage_manager.py`: バックエンド統合、メモリ割当
- `memory_management.py`: MemoryObj, Allocator, 参照カウント
- `local_cpu_backend.py`, `local_disk_backend.py`, など

### Concrete Backend Layer
**責務**: 具体的なストレージI/O実装

- `local_cpu_backend.py`: OrderedDict (LRU) キャッシュ
- `local_disk_backend.py`: ファイルI/O, async
- `remote_backend.py`: Redis/P2P RPC
- `gds_backend.py`: cuFile DMA

---

## 🔐 データフロー保証

### 参照カウント管理

```
allocate()           ref_count = 1 (アロケータ保有)
  ↓
ref_count_up()       ref_count = 2 (バックエンド保有)
  ↓
...
  ↓
ref_count_down()     ref_count = 1
  ↓
ref_count_down()     ref_count = 0 → free() (自動解放)
```

### メモリ有効期間

```
allocate()
  ├─ GPU/CPU メモリ物理割当
  ├─ metadata 初期化
  └─ ref_count = 1

put_task()
  ├─ ref_count_up() (複数バックエンド)
  ├─ 非同期転送
  └─ コールバック: ref_count_down()

get_blocking()
  ├─ ref_count_up() (呼び出し元)
  ├─ 返却
  └─ 呼び出し元: ref_count_down()

ref_count = 0
  └─ parent_allocator.free() (自動解放)
```

### マルチスレッド安全性

```
StorageManager
├─ lock: ロック全体
├─ prefetch_tasks: スレッド安全な辞書

LocalCPUBackend
├─ cpu_lock: hot_cache アクセス
└─ hot_cache: OrderedDict (LRU)

LocalDiskBackend
├─ disk_lock: dict アクセス
├─ loop: asyncio イベントループ
└─ thread: 実行スレッド

MemoryObj
└─ lock: ref_count 更新
```

---

## ⏱️ パフォーマンス階層

```
速度（アクセス）:
  GPU (GDS)       100 μs   (最速、DMA直接)
  LocalCPU        <1 μs    (メモリアクセス)
  LocalDisk       500 μs   (ディスクI/O, CPU経由)
  Remote          1-100 ms (ネットワークRPC)

容量（ストレージ）:
  LocalCPU        数GB～ GB級    (メモリ制限)
  LocalDisk       TB級          (SSD容量)
  Remote          無制限        (サーバー次第)
  GDS             TB級          (NVMe容量)

推奨用途:
  LocalCPU        ホット（頻繁使用）
  LocalDisk       ウォーム（時々使用）
  GDS             超高速推論（GPU直結）
  Remote          バックアップ（P2P分散）
```

---

## 🌐 設定・初期化フロー

```
startup()
  │
  ├─ LMCacheConfig 読込 (yaml)
  │  └─ chunk_size, backends, device等
  │
  ├─ CacheEngine 初期化
  │  ├─ TokenDatabase 作成
  │  ├─ StorageManager 作成
  │  │  ├─ メモリアロケータ作成
  │  │  │  └─ GPU/CPU メモリプール確保
  │  │  │
  │  │  ├─ バックエンド作成 (CreateStorageBackends)
  │  │  │  ├─ LocalCPUBackend (常に作成)
  │  │  │  ├─ LocalDiskBackend (config.local_disk)
  │  │  │  ├─ GDSBackend (config.gds_path)
  │  │  │  └─ RemoteBackend (config.remote_url)
  │  │  │
  │  │  └─ Evictor 作成 (LRU/LFU)
  │  │
  │  └─ GPUConnector 作成
  │
  ├─ vLLM 統合初期化
  │  ├─ LMCacheLookupServer 起動
  │  ├─ LMCacheLookupClient 接続確認
  │  └─ ZMQ ソケット確保
  │
  └─ event loop 起動 (asyncio)
     ├─ LocalDiskBackend async
     └─ RemoteBackend async
```

---

## 📋 主要インターフェース

### StorageBackendInterface

```python
class StorageBackendInterface:
    def get_blocking(self, key: CacheEngineKey) -> Optional[MemoryObj]
    def contains(self, key: CacheEngineKey) -> bool
    def submit_put_task(self, key, memory_obj) -> Optional[Future]
    def remove(self, key: CacheEngineKey) -> None
```

### MemoryObj

```python
class MemoryObj:
    @property
    def tensor(self) -> torch.Tensor  # 論理テンソル

    @property
    def byte_array(self)  # バイト配列（I/O用）

    def ref_count_up(self) -> None
    def ref_count_down(self) -> None
    def get_ref_count(self) -> int
```

### CacheEngine

```python
class CacheEngine:
    def lookup(self, tokens: torch.Tensor) -> int
        # → キャッシュ済みトークン数

    def retrieve(self, tokens, mask, **kwargs) -> torch.Tensor
        # → KV テンソル取得フロー

    def store(self, tokens, mask, kv_cache, **kwargs) -> None
        # → KV テンソル保存フロー
```

---

## 🔗 関連ドキュメント

次に読むべきドキュメント:

1. **[01_StorageManager.md](./01_StorageManager.md)** - バックエンド管理
2. **[02_MemoryManagement.md](./02_MemoryManagement.md)** - メモリモデル
3. **[06_CacheEngine.md](./06_CacheEngine.md)** - 核エンジン
4. **[07_TokenDatabase.md](./07_TokenDatabase.md)** - トークンハッシング
5. **[04_vLLMIntegration.md](./04_vLLMIntegration.md)** - vLLM連携

---

**バージョン**: v1
**最終更新**: 2025-11-23

