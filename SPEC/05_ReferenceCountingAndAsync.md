# Reference Counting と Non-blocking I/O 仕様書

> メモリライフサイクル管理とスレッド安全な非同期処理

**ファイル**: 複数（memory_management.py, storage_backend/*.py）
**責務**: 参照カウント、非同期タスク管理、スレッド同期

---

## 🎯 概要

Reference Counting は、複数のバックエンドがメモリオブジェクトを保有する際に、メモリ安全性を保証するメカニズム。非同期I/Oと組み合わせて、ノンブロッキングな並列処理を実現します。

---

## 🔄 参照カウント管理

### ライフサイクル

```
[1] allocate()
    └─ ref_count = 1 (アロケータが保有)

[2] put_task (バックエンド保存開始)
    ├─ ref_count_up()
    └─ ref_count = 2

[3] put_task (複数バックエンド)
    ├─ LocalCPU: ref_count = 2 (同期)
    ├─ LocalDisk: ref_count_up() → 3
    ├─ Remote: ref_count_up() → 4
    └─ GDS: ref_count_up() → 5

[4] put_callback (完了時)
    ├─ LocalDisk: ref_count_down() → 4
    ├─ Remote: ref_count_down() → 3
    └─ GDS: ref_count_down() → 2

[5] LocalCPU.put (同期完了)
    └─ ref_count_down() → 1

[6] get_blocking (呼び出し元で使用)
    ├─ ref_count_up() → 2
    └─ 返却

[7] 使用終了
    └─ ref_count_down() → 1

[8] ref_count = 0
    └─ parent_allocator.free() (自動解放)
```

### スレッド安全性

```python
def ref_count_down(self) -> None:
    with self.lock:  # スレッド保護
        self.meta.ref_count -= 1
        if (self.meta.ref_count == 0 and
            self.parent_allocator is not None and
            self.meta.is_pin is False):
            self.parent_allocator.free(self)
```

---

## 🚀 Non-blocking I/O

### 非同期処理フロー

```
vLLM Worker
  │
  ├─ store(tokens, kv_cache)
  │  ├─ allocate(shape) [同期]
  │  ├─ from_gpu() [同期]
  │  └─ put(key, memory_obj) [非同期開始]
  │     │
  │     ├─ LocalCPU.submit_put_task()
  │     │  └─ hot_cache[key] = obj [即座]
  │     │
  │     ├─ LocalDisk.submit_put_task()
  │     │  └─ asyncio タスク送信 [Future返却]
  │     │     └─ async_save_bytes_to_disk()
  │     │        ├─ allocate() [非同期]
  │     │        ├─ aiofiles.open() [非同期]
  │     │        ├─ f.write() [非同期]
  │     │        └─ コールバック: ref_count_down()
  │     │
  │     ├─ Remote.submit_put_task()
  │     │  └─ asyncio タスク送信 [Future返却]
  │     │     └─ connection.put() [非同期]
  │     │        └─ コールバック: ref_count_down()
  │     │
  │     └─ return [即座に返却]
  │
  └─ [保存タスク実行継続]
     └─ メインスレッド次処理へ
```

### asyncio イベントループ

```python
# StorageManager 初期化時

self.loop = asyncio.new_event_loop()
self.thread = threading.Thread(target=self.loop.run_forever)
self.thread.start()

# バックエンド内の非同期実行

future = asyncio.run_coroutine_threadsafe(
    async_coro(),
    self.loop
)
# → メインスレッドがブロッキングなし

# ブロッキング待機（必要な場合）

result = future.result(timeout=1)  # タイムアウト付き
```

---

## 🔐 Lock 戦略

### Lock の粒度

```
StorageManager:
├─ manager_lock (粗粒度)
│  └─ prefetch_tasks 辞書アクセスのみ

LocalCPUBackend:
├─ cpu_lock (粗粒度)
│  └─ hot_cache OrderedDict アクセス全体

LocalDiskBackend:
├─ disk_lock (粗粒度)
│  └─ dict, put_tasks アクセス

MemoryObj:
└─ lock (細粒度)
   └─ ref_count 更新のみ
```

### デッドロック回避

```
Rule 1: 常に同じ順序でロック取得
  ✓ manager_lock → cpu_lock → disk_lock
  ✗ cpu_lock → manager_lock (逆順注意)

Rule 2: Lock保有中は長時間処理しない
  ✗ with lock:
        result = slow_io_operation()  # 不可
  ✓ with lock:
        task = slow_io_operation_async()
     task.result()  # Lock外で待機

Rule 3: Callback内では Lock取得しない
  ✗ future.add_done_callback(lambda f: with lock: ...)
  ✓ future.add_done_callback(lambda f: simple_operation())
```

---

## 🎯 Eviction 時の保護

### 安全な削除判定

```python
# LocalCPUBackend.allocate() 内

for candidate_key in hot_cache:
    obj = hot_cache[candidate_key]

    # ref_count > 1 ならまだ使用中
    # → eviction不可
    if obj.get_ref_count() > 1:
        continue  # 次の候補へ

    # ref_count == 1 なら安全に削除可
    evict_keys.append(candidate_key)
    obj.ref_count_down()  # ref_count = 0 → free()

    # 再度割当試行
    memory_obj = allocator.allocate(shape, dtype)
    if memory_obj is not None:
        break
```

---

## 📊 シナリオ分析

### シナリオ1: Local-to-Disk 転送

```
[T0] allocate()          ref_count = 1
[T1] put()
     ├─ LocalCPU: ref_count は +1 なし（同期）
     ├─ LocalDisk: ref_count_up() → 2
     │            asyncio.run_coroutine_threadsafe()
     └─ return [即座]

[T1-T100ms] メインスレッド: 次処理へ
[T100ms] LocalDisk 非同期タスク完了
         └─ callback: ref_count_down() → 1
         └─ aiofiles.write() 完了

[T200ms] 呼び出し元: ref_count_down() → 0
         └─ free() [自動]
```

### シナリオ2: Eviction 中の参照

```
[T0] hot_cache[key1] に access
     └─ ref_count_up() → 2
     └─ 呼び出し元保有

[T1] allocate(new) でメモリ不足
     └─ LRU候補: key2 (ref_count=1)
     └─ evict可能
     └─ key2.ref_count_down() → 0
     └─ free()

[T2] key1 使用完了
     └─ ref_count_down() → 1
     └─ (自動には free() されない、LocalCPUで保有中)

[T3] key1 LRU削除
     └─ ref_count_down() → 0
     └─ free()
```

---

## 🔗 関連ドキュメント

- **[02_MemoryManagement.md](./02_MemoryManagement.md)** - メモリモデル
- **[01_StorageManager.md](./01_StorageManager.md)** - ストレージ管理

---

**バージョン**: v1
**最終更新**: 2025-11-23

