# StorageManager 仕様書

> 複数ストレージバックエンドの統一インターフェース、メモリ割当、バックエンド管理

**ファイル**: `/home/user/LMCache/lmcache/v1/storage_backend/storage_manager.py`
**クラス**: `StorageManager`
**責務**: バックエンド統合、MemoryObj 取得/保存、メモリ割当管理

---

## 🎯 概要

StorageManager は、複数のストレージバックエンド（LocalCPU, LocalDisk, Remote, GDS）を統一インターフェースで管理するコンポーネント。

**核となる責務**:
1. バックエンド検索（優先度順、短絡評価）
2. MemoryObj 取得・保存（ブロッキング/非同期）
3. メモリ割当・解放管理
4. Eviction（キャッシュサイズ超過時）
5. Prefetch タスク管理

---

## 🏗️ クラス構造

```python
class StorageManager:
    # バックエンド管理
    storage_backends: OrderedDict[str, StorageBackendInterface]

    # メモリ管理
    local_cpu_backend: LocalCPUBackend
    memory_allocator: MemoryAllocatorInterface
    evictor: BaseEvictor

    # 非同期処理
    loop: asyncio.AbstractEventLoop
    thread: threading.Thread

    # 同期化
    manager_lock: threading.Lock
    prefetch_tasks: Dict[CacheEngineKey, Future]
```

---

## 📖 主要メソッド

### 1. `__init__(config, ...)`

**責務**: 初期化、バックエンド生成、メモリプール確保

```python
def __init__(self, config: LMCacheEngineConfig, loop: asyncio.AbstractEventLoop,
             memory_allocator: MemoryAllocatorInterface, dst_device: str = "cuda"):
    """
    StorageManager 初期化

    Args:
        config: LMCacheEngineConfig（ストレージ設定）
        loop: asyncio イベントループ
        memory_allocator: メモリアロケータ（CPU/GPU）
        dst_device: ターゲットデバイス（"cuda"など）
    """
    self.config = config
    self.dst_device = dst_device
    self.memory_allocator = memory_allocator
    self.loop = loop

    # ステップ1: Evictor 作成
    self.evictor = BaseEvictor.Create(config)  # LRU/LFU

    # ステップ2: バックエンド作成（優先度順）
    self.storage_backends = CreateStorageBackends(
        config,
        self.evictor,
        loop,
        memory_allocator,
        dst_device,
    )

    # ステップ3: LocalCPU バックエンド確保
    assert "LocalCPUBackend" in self.storage_backends
    self.local_cpu_backend = self.storage_backends["LocalCPUBackend"]

    # ステップ4: 非同期タスク管理
    self.prefetch_tasks: Dict[CacheEngineKey, Future] = {}
    self.manager_lock = threading.Lock()
```

**生成されるバックエンド** (初期化.yaml に基づいて):
```
1️⃣ LocalCPUBackend      (常に作成)
2️⃣ LocalDiskBackend     (config.local_disk == True)
3️⃣ WekaGdsBackend       (config.weka_path != None)
4️⃣ GdsBackend           (config.gds_path != None)
5️⃣ RemoteBackend        (config.remote_url != None)
```

### 2. `get(key) → Optional[MemoryObj]`

**責務**: ブロッキングでMemoryObjを取得（バックエンド順序探索）

**ファイル行**: 179-211

```python
def get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
    """
    キー に対応する MemoryObj をブロッキング取得

    Args:
        key: CacheEngineKey

    Returns:
        MemoryObj（見つかった場合）, None（見つからない場合）

    フロー:
        1. prefetch タスク完了待機（存在する場合）
        2. バックエンド順序探索（短絡評価）
        3. LocalCPU 以外から取得時は write_back
        4. MemoryObj 返却
    """
    # ステップ1: プリフェッチ完了待機
    self.manager_lock.acquire()
    prefetch_task = self.prefetch_tasks.get(key, None)
    self.manager_lock.release()

    if prefetch_task is not None:
        logger.debug(f"Waiting for prefetching result of {key}...")
        prefetch_task.result(timeout=1)  # ブロッキング待機

    # ステップ2: バックエンド順序探索（短絡評価）
    for backend_name, backend in self.storage_backends.items():
        memory_obj = backend.get_blocking(key)
        if memory_obj is not None:
            # ステップ3: LocalCPU 以外からは write_back
            if backend_name != "LocalCPUBackend":
                self.local_cpu_backend.write_back(key, memory_obj)

            # ステップ4: 返却
            return memory_obj

    # すべてMiss
    return None
```

**バックエンド検索順序の重要性**:

```
検索順序: OrderedDict.items() の順序に準ずる

局所性の原理に基づく最適化:
  ├─ LocalCPU       (最速, メモリ内)
  ├─ LocalDisk      (中速, ローカルI/O)
  ├─ GDS            (超高速, DMA)
  └─ Remote         (遅い, RPC)

短絡評価: 最初に見つかったバックエンドから即座に返却
  └─ 後続バックエンドは検索しない
```

### 3. `put(key, memory_obj) → None`

**責務**: 全バックエンドに保存（非同期並列）

**ファイル行**: 121-138

```python
def put(self, key: CacheEngineKey, memory_obj: MemoryObj) -> None:
    """
    MemoryObj を全バックエンドに保存

    Args:
        key: CacheEngineKey
        memory_obj: MemoryObj

    特徴:
        - 各バックエンドに並列で保存
        - 非同期タスク返却（完了待たず）
        - ref_count管理は各バックエンドで実施
    """
    for backend_name, backend in self.storage_backends.items():
        # ステップ1: バックエンド判定
        if not backend.should_put(key):
            continue  # 不要なバックエンドはスキップ

        # ステップ2: 非同期保存タスク送信
        future = backend.submit_put_task(key, memory_obj)
        # → LocalCPU: 同期完了
        # → LocalDisk: asyncio Future
        # → Remote: asyncio Future
        # → GDS: asyncio Future
```

**バックエンド別の動作**:

| Backend | 実行 | 同期性 | ref_count |
|---------|------|--------|-----------|
| LocalCPU | 同期 | ブロッキング | get時に+1 |
| LocalDisk | 非同期 | Future返却 | 完了時に-1 |
| Remote | 非同期 | Future返却 | 完了時に-1 |
| GDS | 非同期 | Future返却 | 完了時に-1 |

### 4. `allocate(shape, dtype, ...) → Optional[MemoryObj]`

**責務**: メモリ割当（eviction可能）

**ファイル行**: 106-120

```python
def allocate(self, shape: torch.Size, dtype: torch.dtype,
             fmt: MemoryFormat = MemoryFormat.KV_2LTD,
             eviction: bool = True) -> Optional[MemoryObj]:
    """
    新規メモリ割当

    Args:
        shape: テンソル形状 [2, num_layers, tokens, hidden_size]
        dtype: データ型（torch.float16など）
        fmt: メモリフォーマット
        eviction: メモリ不足時に LRU から削除するか

    Returns:
        MemoryObj（成功）, None（失敗）

    フロー:
        1. LocalCPUBackend に委譲
        2. メモリ不足 && eviction==True の場合
        3. LRU 候補を削除して再試行
    """
    assert isinstance(self.local_cpu_backend, LocalCPUBackend)

    # ステップ1: LocalCPUBackend.allocate() に委譲
    memory_obj = self.local_cpu_backend.allocate(
        shape, dtype, fmt, eviction=eviction
    )
    return memory_obj

    # LocalCPUBackend.allocate() 内部:
    # ├─ memory_allocator.allocate()
    # ├─ 成功時: return memory_obj
    # └─ 失敗時:
    #    ├─ hot_cache から LRU削除候補を探索
    #    ├─ ref_count > 1 なら（他バックエンド使用中）スキップ
    #    ├─ evict_keys に追加
    #    ├─ memory_obj.ref_count_down()
    #    ├─ memory_allocator.allocate() 再試行
    #    └─ lookup_server.batched_remove(evict_keys)
```

### 5. `contains(key) → bool`

**責務**: キーが任意のバックエンドに存在するか確認

```python
def contains(self, key: CacheEngineKey, search_range: Optional[List[str]] = None) -> bool:
    """
    キーがいずれかのバックエンドに存在するか確認

    Args:
        key: CacheEngineKey
        search_range: 検索対象バックエンド（None=全て）

    Returns:
        bool（存在する場合 True）
    """
    for backend_name, backend in self.storage_backends.items():
        if search_range is not None and backend_name not in search_range:
            continue

        if backend.contains(key):
            return True

    return False
```

---

## 🔄 MemoryObj フロー詳細

### Get フロー（取得）

```
StorageManager.get(key)
│
├─[1] Prefetch 完了待機
│     if key in prefetch_tasks:
│        future.result(timeout=1)  # ブロック
│
├─[2] バックエンド順序探索
│     for backend_name, backend in storage_backends.items():
│
│     ├─ backend.get_blocking(key)
│     │  │
│     │  ├─ LocalCPUBackend.get_blocking()
│     │  │  ├─ hot_cache[key]?
│     │  │  ├─ YES → ref_count_up() + move_to_end() + return
│     │  │  └─ NO → continue
│     │  │
│     │  ├─ LocalDiskBackend.get_blocking()
│     │  │  ├─ dict[key]メタデータ?
│     │  │  ├─ YES → allocate() + f.readinto() + return
│     │  │  └─ NO → continue
│     │  │
│     │  └─ RemoteBackend.get_blocking()
│     │     ├─ asyncio.run_coroutine_threadsafe()
│     │     ├─ connection.get(key)
│     │     ├─ deserialize()
│     │     └─ return
│     │
│     └─ 見つかった?
│        ├─ YES → write_back() + return
│        └─ NO → 次バックエンド
│
├─[3] Write-back（非LocalCPU時）
│     if backend_name != "LocalCPUBackend":
│        local_cpu_backend.write_back(key, memory_obj)
│        └─ ホットキャッシュに昇格
│
└─ return MemoryObj (or None)
```

### Put フロー（保存）

```
StorageManager.put(key, memory_obj)
│
└─ for each backend:
   │
   ├─ LocalCPUBackend.submit_put_task()
   │  └─ hot_cache[key] = memory_obj (同期)
   │     └─ ref_count はすでに +1 済み
   │
   ├─ LocalDiskBackend.submit_put_task()
   │  ├─ memory_obj.ref_count_up()
   │  ├─ asyncio.run_coroutine_threadsafe(async_save_bytes_to_disk)
   │  └─ future.add_done_callback(lambda f: ref_count_down())
   │
   ├─ RemoteBackend.submit_put_task()
   │  ├─ serializer.serialize(memory_obj)
   │  ├─ asyncio.run_coroutine_threadsafe(connection.put)
   │  └─ future.add_done_callback(lambda f: ref_count_down())
   │
   └─ GDSBackend.submit_put_task()
      ├─ allocate(GPU Memory)
      ├─ asyncio.run_coroutine_threadsafe(_async_save_bytes_to_disk)
      └─ future.add_done_callback(lambda f: ref_count_down())
```

---

## 💾 バックエンド生成（CreateStorageBackends）

**ファイル**: `/home/user/LMCache/lmcache/v1/storage_backend/__init__.py` (行43-98)

```python
def CreateStorageBackends(
    config: LMCacheEngineConfig,
    evictor: Optional[BaseEvictor],
    loop: asyncio.AbstractEventLoop,
    memory_allocator: MemoryAllocatorInterface,
    dst_device: str = "cuda",
) -> OrderedDict[str, StorageBackendInterface]:
    """
    設定に基づいてストレージバックエンド を生成

    Returns:
        OrderedDict (順序が重要: 検索優先度)
    """
    storage_backends = OrderedDict()

    # [1] LocalCPUBackend（常に作成）
    storage_backends["LocalCPUBackend"] = LocalCPUBackend(
        config, evictor, loop, memory_allocator
    )

    # [2] LocalDiskBackend（オプション）
    if config.local_disk and config.max_local_disk_size > 0:
        storage_backends["LocalDiskBackend"] = LocalDiskBackend(
            config, evictor, loop, memory_allocator
        )

    # [3] WekaGdsBackend（オプション）
    if config.weka_path is not None:
        storage_backends["WekaGdsBackend"] = WekaGdsBackend(
            config, loop, memory_allocator, dst_device
        )

    # [4] GdsBackend（オプション）
    if config.gds_path is not None:
        storage_backends["GdsBackend"] = GdsBackend(
            config, loop, memory_allocator, dst_device
        )

    # [5] RemoteBackend（オプション）
    if config.remote_url is not None:
        storage_backends["RemoteBackend"] = RemoteBackend(
            config, loop, memory_allocator, dst_device
        )

    return storage_backends
```

**生成順序の意味**:
- リスト順 = get_blocking() での検索優先度
- OrderedDict により順序が保証される（Python 3.7+）

---

## 🔐 スレッド安全性

### Lock 管理

```
StorageManager:
├─ manager_lock (threading.Lock)
│  └─ prefetch_tasks アクセス時のみ
│
LocalCPUBackend:
├─ cpu_lock (threading.Lock)
│  └─ hot_cache 全アクセス
│
LocalDiskBackend:
├─ disk_lock (threading.Lock)
│  └─ dict, put_tasks アクセス
│
MemoryObj:
└─ lock (threading.Lock)
   └─ ref_count 更新のみ
```

### 非同期実行

```
StorageManager初期化:
├─ loop = asyncio.new_event_loop()
├─ thread = threading.Thread(target=loop.run_forever)
└─ thread.start()

バックエンド内の非同期操作:
├─ asyncio.run_coroutine_threadsafe(coro, loop)
│  └─ メインスレッド以外から呼び出し可
│
└─ future.result(timeout=T)
   └─ メインスレッドがブロッキング待機
```

---

## 🎯 使用パターン

### パターン1: 単純な取得と使用

```python
storage_manager = StorageManager(config, loop, allocator)
key = CacheEngineKey(...)

# 取得（ブロッキング）
memory_obj = storage_manager.get(key)
if memory_obj is None:
    # キャッシュ未検出
    memory_obj = storage_manager.allocate(shape, dtype)

# 使用
tensor = memory_obj.tensor
# ...

# 使用終了
memory_obj.ref_count_down()
```

### パターン2: 保存と非同期完了

```python
# メモリ割当
memory_obj = storage_manager.allocate(shape, dtype)

# データ書き込み
memory_obj.tensor.copy_(kv_cache)

# 保存開始（非同期並列）
storage_manager.put(key, memory_obj)
# → 即座に返却
# → 各バックエンド が非同期に保存を開始

# メインスレッドは次処理へ（保存待たず）
# → 完了時に自動的に ref_count_down()
```

### パターン3: 存在確認

```python
if storage_manager.contains(key):
    memory_obj = storage_manager.get(key)
    # ...
else:
    # キャッシュ未検出
    memory_obj = storage_manager.allocate(...)
```

---

## ⚙️ 設定パラメータ

**関連設定**:

```yaml
# バックエンド選択
local_cpu: true                    # LocalCPU 有効
local_disk: true                   # LocalDisk 有効
local_disk_path: "/path/to/cache"
max_local_disk_size: 500           # GB単位

gds_path: "/tmp/gds"               # GDS 有効
cufile_buffer_size: 128            # MiB単位

weka_path: "/mnt/weka"             # Weka 有効

remote_url: "redis://localhost"    # Redis URL

# キャッシュ管理
max_local_cpu_size: 10             # GB単位
lru_cache_config:
  enable_lfu: False                # True=LFU, False=LRU
  cache_evict_interval: 100        # イテレーション間隔
```

---

## 📊 パフォーマンス特性

### アクセス速度

```
LocalCPUBackend     <1 μs       最速（メモリ）
GDSBackend          ~100 μs     超高速（DMA）
LocalDiskBackend    ~500 μs     速（ローカルI/O）
RemoteBackend       ~1-100 ms   遅（RPC）
```

### スケーラビリティ

```
バックエンド数:      最大5個（実運用は3-4個推奨）
ストレージ容量:      Remote > LocalDisk >> GDS > LocalCPU
並列保存:            全バックエンド並列可（asyncio）
メモリ割当:          O(1) First-fit
バックエンド検索:    O(n) n=バックエンド数（短絡評価あり）
```

---

## 🔗 関連クラス

- **StorageBackendInterface**: 各バックエンドの抽象インターフェース
- **LocalCPUBackend**: ホットキャッシュ
- **LocalDiskBackend**: ウォームキャッシュ
- **RemoteBackend**: リモートストレージ
- **GDSBackend**: GPU Direct Storage
- **MemoryObj**: 統一メモリ表現
- **MemoryAllocatorInterface**: メモリ割当

---

## 📝 トラブルシューティング

### 問題: メモリ割当失敗

```
症状: allocate() が None を返す
原因: メモリプール枯渇、eviction 失敗
解決:
  1. max_local_cpu_size を増加
  2. ref_count > 1 のオブジェクト確認
  3. 使用済みメモリが ref_count_down() されているか確認
```

### 問題: キャッシュヒット率低下

```
症状: get() が常に None を返す
原因: バックエンド未初期化、キーの不一致
解決:
  1. TokenDatabase の SHA256 ハッシング確認
  2. バックエンド contains() を個別テスト
  3. ホットキャッシュサイズ確認
```

### 問題: 非同期保存の未完了

```
症状: put() 後すぐに削除されてしまう
原因: ref_count 管理の不適切
解決:
  1. future.result() で完了待機
  2. コールバック内で ref_count_down()
  3. スレッド安全性確認
```

---

## 🔗 関連ドキュメント

- **[02_MemoryManagement.md](./02_MemoryManagement.md)** - MemoryObj詳細
- **[03a_LocalCPUBackend.md](./03_StorageBackends/03a_LocalCPUBackend.md)** - ホットキャッシュ
- **[03b_LocalDiskBackend.md](./03_StorageBackends/03b_LocalDiskBackend.md)** - ローカルディスク
- **[05_ReferenceCountingAndAsync.md](./05_ReferenceCountingAndAsync.md)** - 参照カウント詳細

---

**バージョン**: v1
**最終更新**: 2025-11-23

