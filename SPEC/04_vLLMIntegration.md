# vLLM Integration 仕様書

> vLLMからのKV-Cacheクエリ処理インターフェース

**ファイル**: `/home/user/LMCache/lmcache/integration/vllm/vllm_v1_adapter.py`
**クラス**: `LMCacheConnectorV1Impl`, `LMCacheLookupClient/Server`
**責務**: vLLM との通信、クエリプロトコル処理、LoadSpec管理

---

## 🎯 概要

vLLM Integration は、vLLMの Scheduler/Worker と LMCache の通信層。ZMQ RPC（Request-Reply パターン）を使用して、メインプロセスと別プロセスで動作する LMCache エンジンと通信します。

**3つの通信フェーズ**:
1. **Lookup フェーズ** (リクエスト処理前)
   - vLLM Scheduler が lookup クエリ送信
   - LMCache が キャッシュ済みトークン数を応答

2. **Retrieve フェーズ** (フォワード実行時)
   - vLLM Worker が KV キャッシュ取得
   - LMCache が GPU に直接転送

3. **Store フェーズ** (フォワード後)
   - vLLM Worker が新規 KV キャッシュ保存
   - LMCache が 非同期で複数バックエンドに保存

---

## 🏗️ クラス構造

```
LMCacheConnectorV1Impl
├─ LMCacheLookupClient (ZMQ REQ)
├─ LMCacheLookupServer (ZMQ REP, 別プロセス)
├─ CacheEngine (コアエンジン)
├─ GPUConnector (GPU転送)
└─ LoadSpec 管理

通信プロトコル:
├─ Msgpack (トークン ID エンコーディング)
├─ ZMQ RPC (IPC: ipc://lmcache_rpc_port_*)
└─ 構造化メッセージ (ClientMetaMessage, ServerMetaMessage)
```

---

## 📡 プロトコル詳細

### Lookup クエリ (ZMQ REQ/REP)

#### 送信側 (vLLM Scheduler)

```python
# LMCacheLookupClient.lookup()

request = {
    "type": "lookup",
    "token_ids": torch.tensor([1, 2, 3, ..., 1024], dtype=torch.int32),
}

# Msgpack エンコーディング
encoded = msgpack_encoder.encode(request["token_ids"])

# ZMQ 送信（ブロッキング）
socket.send_multipart(encoded)

# レスポンス待機（ブロッキング）
response = socket.recv()  # 4バイト整数

# デコード
num_cached = int.from_bytes(response, "big")
print(f"Cache hit: {num_cached} tokens")
```

#### 受信側 (LMCache Server)

```python
# LMCacheLookupServer.process_request()

# ZMQ 受信（ブロッキング）
frames = socket.recv_multipart()

# デコード
token_ids = msgpack_decoder.decode(frames)

# Lookup 実行
num_cached = cache_engine.lookup(token_ids)

# レスポンス作成
response = num_cached.to_bytes(4, "big")

# ZMQ 送信（ブロッキング）
socket.send(response)
```

#### メッセージフォーマット

```
Request:
┌─────────────────────────────────┐
│ Msgpack encoded token_ids       │
│ Frame 0: dtype, shape (metadata)│
│ Frame 1+: token data (int32)    │
└─────────────────────────────────┘

Response:
┌─────────────────────────────────┐
│ 4-byte integer (big-endian)     │
│ Example: int.to_bytes(256, "big")│
└─────────────────────────────────┘
```

### Retrieve / Store クエリ (Direct GPU Memory)

#### メモリオブジェクト転送

```
vLLM Worker
  └─ GPU KV バッファ
      ↓ (GPU ↔ GPU direct copy, ページロックメモリ経由)

LMCache GPU Connector
  └─ MemoryObj (GPU Memory)
      ↓ (DMA or cuFile)

StorageManager
  └─ Storage Backend
      ├─ LocalCPU
      ├─ LocalDisk
      ├─ Remote
      └─ GDS
```

---

## 🔄 フロー詳細

### フェーズ 1: Lookup（初期化フェーズ）

```
vLLM Scheduler
│
├─ request = Request(prompt_token_ids=[1,2,...,1024])
│
├─ get_num_new_matched_tokens(request, num_computed_tokens=128)
│  │
│  ├─ token_ids = torch.tensor(request.prompt_token_ids)
│  │
│  ├─ LMCacheLookupClient.lookup(token_ids)
│  │  │
│  │  └─ ZMQ REQ
│  │     └─ ipc://lmcache_rpc_port_xxxx
│  │
│  ├─ LMCacheLookupServer (別プロセス)
│  │  │
│  │  └─ CacheEngine.lookup(token_ids)
│  │     └─ キャッシュ済みトークン数を検索
│  │
│  ├─ ZMQ REP
│  │  └─ int (例: 256)
│  │
│  └─ num_external_hit_tokens = 256
│
├─ LoadSpec 作成・保存
│  │
│  └─ LoadSpec(
│         vllm_cached_tokens=128,        # vLLMキャッシュ
│         lmcache_cached_tokens=256,     # LMCacheキャッシュ
│         can_load=True,
│     )
│
└─ request.load_spec = LoadSpec
```

**LoadSpec の意味**:

```
vLLM          LMCache
Cache:        Cache:       Action:
[0:128]       [0:128]      No load needed (all cached)
[128:256]     [256:256]    Load from LMCache
[256:1024]    -            Compute

合計キャッシュ: max(128, 256) = 256 トークン
未計算: 1024 - 256 = 768 トークン
```

### フェーズ 2: Retrieve（フォワード実行フェーズ）

```
vLLM Worker
│
├─ forward_context = context
│
├─ start_load_kv(forward_context)
│  │
│  ├─ for request in metadata.requests:
│  │  │
│  │  ├─ if request.load_spec is None:
│  │  │  │
│  │  │  └─ continue
│  │  │
│  │  ├─ tokens = request.token_ids
│  │  ├─ slot_mapping = request.slot_mapping
│  │  ├─ token_mask = [False, False, ..., True, True]
│  │  │
│  │  ├─ CacheEngine.retrieve(
│  │  │      tokens,
│  │  │      mask=token_mask,
│  │  │      kvcaches=vllm_paged_kv_buffer,
│  │  │      slot_mapping=slot_mapping,
│  │  │  )
│  │  │  │
│  │  │  ├─ StorageManager.get(key)
│  │  │  │  │
│  │  │  │  ├─ LocalCPUBackend
│  │  │  │  ├─ LocalDiskBackend
│  │  │  │  ├─ RemoteBackend
│  │  │  │  └─ GDSBackend
│  │  │  │
│  │  │  ├─ GPUConnector.to_gpu()
│  │  │  │  │
│  │  │  │  └─ MemoryObj → vLLM Paged KV buffer コピー
│  │  │  │
│  │  │  └─ return mask
│  │  │
│  │  └─ num_retrieved = 768
│  │
│  └─ return
│
├─ model.forward(input_ids, kv_cache, ...)
│  │
│  └─ GPU 推論実行
│
└─ kv_cache_new = kv_cache
```

### フェーズ 3: Store（保存フェーズ）

```
vLLM Worker
│
├─ wait_for_save()
│  │
│  ├─ for request in metadata.requests:
│  │  │
│  │  ├─ if request.save_spec is None or not request.save_spec.can_save:
│  │  │  │
│  │  │  └─ continue
│  │  │
│  │  ├─ token_ids = request.token_ids
│  │  ├─ skip_leading_tokens = CacheEngine.lookup(token_ids)
│  │  │
│  │  ├─ store_mask = [False, False, ..., True, True]
│  │  │
│  │  ├─ CacheEngine.store(
│  │  │      token_ids,
│  │  │      mask=store_mask,
│  │  │      kv_cache=kv_cache_new,
│  │  │      slot_mapping=slot_mapping,
│  │  │  )
│  │  │  │
│  │  │  ├─ allocate(shape, dtype)
│  │  │  │
│  │  │  ├─ from_gpu(memory_obj, kv_cache)
│  │  │  │
│  │  │  ├─ StorageManager.put(key, memory_obj)
│  │  │  │  │
│  │  │  │  ├─ LocalCPUBackend.submit_put_task()
│  │  │  │  ├─ LocalDiskBackend.submit_put_task() (async)
│  │  │  │  ├─ RemoteBackend.submit_put_task() (async)
│  │  │  │  └─ GDSBackend.submit_put_task() (async)
│  │  │  │
│  │  │  └─ Future.add_done_callback() (完了時)
│  │  │
│  │  └─ return
│  │
│  └─ return (すべての保存タスクが非同期で実行中)
│
└─ 次リクエストへ（保存を待たない）
```

---

## 🔧 ZMQ RPC 設定

### ソケットパス

```python
# IPC (Inter-Process Communication)
ipc://lmcache_rpc_port_xxxx

構成:
  ├─ Base URL: /tmp/lmcache_rpc
  ├─ Role: scheduler (S) / worker (W)
  ├─ Instance ID: (worker-0, worker-1, ...)
  └─ RPC Port: (default 50000)

例:
  ipc:///tmp/lmcache_rpc_scheduler_50000
  ipc:///tmp/lmcache_rpc_worker_0_50000
  ipc:///tmp/lmcache_rpc_worker_1_50000
```

### パターン: REQ/REP

```
ZMQ REQ/REP:
  ├─ 同期的なリクエスト・レスポンス
  ├─ クライアント（vLLM）がリクエスト送信
  ├─ サーバー（LMCache）がレスポンス返送
  └─ ブロッキング: 応答を待つ

利点:
  ├─ シンプル
  ├─ 接続管理不要
  └─ 低遅延（IPC）
```

---

## 🎯 LoadSpec 管理

```python
@dataclass
class LoadSpec:
    vllm_cached_tokens: int      # vLLM キャッシュ済みトークン数
    lmcache_cached_tokens: int   # LMCache キャッシュ済みトークン数
    can_load: bool               # 読込可能か

    @property
    def num_new_to_load(self) -> int:
        """LMCacheから読込すべき新規トークン数"""
        return self.lmcache_cached_tokens - self.vllm_cached_tokens

# 例:
load_spec = LoadSpec(
    vllm_cached_tokens=128,      # vLLMに [0:128] がある
    lmcache_cached_tokens=256,   # LMCacheに [0:256] がある
    can_load=True,
)

num_new = load_spec.num_new_to_load
# → 256 - 128 = 128 トークンを LMCacheから読込可能
```

---

## 📊 通信フロー図

```
Phase 1: Lookup
┌─────────────────┐              ┌──────────────────┐
│  vLLM Scheduler │              │ LMCache (server) │
├─────────────────┤              ├──────────────────┤
│                 │  ZMQ REQ     │                  │
│ Lookup Request  ├─────────────→│ lookup()         │
│ [token_ids]     │              │ ↓               │
│                 │              │ StorageManager   │
│                 │  ZMQ REP     │                  │
│ LoadSpec        │←─────────────┤ Response: 256    │
│ (saved)         │              │                  │
└─────────────────┘              └──────────────────┘

Phase 2: Retrieve (implicit)
┌─────────────────┐              ┌──────────────────┐
│ vLLM Worker     │              │ CacheEngine      │
├─────────────────┤              ├──────────────────┤
│                 │  Direct GPU  │                  │
│ retrieve()      ├─────────────→│ to_gpu()         │
│ kv_cache []     │  memory      │ StorageManager   │
│                 │              │                  │
└─────────────────┘              └──────────────────┘

Phase 3: Store (implicit)
┌─────────────────┐              ┌──────────────────┐
│ vLLM Worker     │              │ CacheEngine      │
├─────────────────┤              ├──────────────────┤
│                 │  Direct GPU  │                  │
│ store()         ├─────────────→│ from_gpu()       │
│ kv_cache_new    │  memory      │ StorageManager   │
│                 │              │ (async)          │
│ wait_for_save() │              │                  │
└─────────────────┘              └──────────────────┘
```

---

## 🔌 初期化フロー

```python
def __init__(self, config: VllmConfig):
    """
    v1 Adapter 初期化
    """
    # ステップ1: ZMQ RPC パス決定
    socket_path = get_zmq_rpc_path_lmcache(
        role=KVConnectorRole.SCHEDULER,  # or WORKER
        is_tp=False,
        vllm_config=config,
    )
    # → ipc:///tmp/lmcache_rpc_scheduler_50000

    # ステップ2: LMCacheLookupClient 初期化
    self.lookup_client = LMCacheLookupClient(
        role=KVConnectorRole.SCHEDULER,
        is_tp=False,
        vllm_config=config,
    )
    # → ZMQ REQ ソケット作成、LMCache サーバーに接続

    # ステップ3: LMCacheLookupServer 起動（別プロセス）
    self.lookup_server_process = threading.Thread(
        target=run_lmcache_server,
        args=(config,),
    )
    self.lookup_server_process.start()

    # ステップ4: CacheEngine 初期化
    self.cache_engine = CacheEngine(config)

    # ステップ5: GPUConnector 初期化
    self.gpu_connector = VLLMPagedMemGPUConnectorV2(...)

    # ステップ6: LoadSpec 管理
    self.load_specs: Dict[str, LoadSpec] = {}
```

---

## 📋 インターフェース定義

### LMCacheLookupClient

```python
class LMCacheLookupClient:
    def __init__(self, role: KVConnectorRole, is_tp: bool, vllm_config):
        self.encoder = MsgpackEncoder()
        self.ctx = zmq.Context()
        self.socket = make_zmq_socket(
            self.ctx,
            socket_path,
            zmq.REQ,
            bind=False,  # クライアント（接続側）
        )

    def lookup(self, token_ids: torch.Tensor) -> int:
        """
        Lookup クエリ（ブロッキング）

        Args:
            token_ids: [token_0, token_1, ..., token_N]

        Returns:
            キャッシュ済みトークン数
        """
        request = self.encoder.encode(token_ids)
        self.socket.send_multipart(request, copy=False)
        resp = self.socket.recv()
        return int.from_bytes(resp, "big")
```

### LMCacheLookupServer

```python
class LMCacheLookupServer:
    def __init__(self, lmcache_engine, socket_path):
        self.lmcache_engine = lmcache_engine
        self.decoder = MsgpackDecoder()
        self.socket = make_zmq_socket(
            ctx,
            socket_path,
            zmq.REP,
            bind=True,  # サーバー（バインド側）
        )

    def process_request(self):
        """リクエスト処理ループ"""
        while self.running:
            frames = self.socket.recv_multipart(copy=False)
            token_ids = self.decoder.decode(frames)

            # 核処理
            result = self.lmcache_engine.lookup(token_ids)

            # レスポンス送信
            response = result.to_bytes(4, "big")
            self.socket.send(response)
```

---

## 🎯 使用パターン

### Scheduler（Lookup のみ）

```python
# Scheduler スレッド
connector = LMCacheConnectorV1Impl(config)

for request in batch:
    # Lookup クエリ
    num_cached = connector.lookup_client.lookup(
        torch.tensor(request.prompt_token_ids)
    )

    # LoadSpec 作成・保存
    request.load_spec = LoadSpec(
        vllm_cached_tokens=computed_tokens,
        lmcache_cached_tokens=num_cached,
        can_load=num_cached > computed_tokens,
    )
```

### Worker（Retrieve + Store）

```python
# Worker スレッド
for request in metadata.requests:
    if request.load_spec is None:
        continue

    # Retrieve
    connector.cache_engine.retrieve(
        request.token_ids,
        kvcaches=vllm_paged_buffer,
        slot_mapping=request.slot_mapping,
    )

    # ... GPU Forward ...

    # Store
    connector.cache_engine.store(
        request.token_ids,
        kv_cache=new_kv_cache,
        slot_mapping=request.slot_mapping,
    )
```

---

## ⚙️ 設定（config.yaml）

```yaml
# vLLM Integration 設定
lmcache_config: /path/to/lmcache.yaml

# LookupServer
lookup_server_host: localhost
lookup_server_port: 50000

# ZMQ
zmq_timeout: 1000              # ms

# キャッシュ
chunk_size: 256
enable_p2p: true
enable_controller: true
```

---

## 🔗 関連ドキュメント

- **[09_Protocol.md](./09_Protocol.md)** - プロトコル詳細
- **[06_CacheEngine.md](./06_CacheEngine.md)** - コアエンジン
- **[08_GPUConnector.md](./08_GPUConnector.md)** - GPU転送

---

**バージョン**: v1
**最終更新**: 2025-11-23

