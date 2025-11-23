# RemoteBackend 仕様書

> リモート分散キャッシュ: Redis/P2P ネットワークストレージ

**ファイル**: `/home/user/LMCache/lmcache/v1/storage_backend/remote_backend.py`
**クラス**: `RemoteBackend`
**責務**: リモート接続管理、シリアライゼーション、P2P

---

## 🎯 概要

RemoteBackend は、Redis や P2P ネットワークを通じたリモートストレージを管理。複数ワーカー間でKVキャッシュ共有可能。

**特徴**:
- **無制限容量**: サーバー次第
- **分散**: 複数ワーカー共有
- **遅延**: ~1-100 ms （ネットワークレイテンシ）

---

## 📖 主要メソッド

### `get_blocking(key) → Optional[MemoryObj]`

```python
def get_blocking(self, key: CacheEngineKey) -> Optional[MemoryObj]:
    if self.connection is None:
        return None

    # 非同期get()をスレッドセーフに実行
    future = asyncio.run_coroutine_threadsafe(
        self.connection.get(key),
        self.loop
    )

    # ブロッキング待機
    try:
        memory_obj = future.result(timeout=1)
    except Exception as e:
        self.connection = None
        return None

    # デシリアライズ（圧縮解除）
    decompressed = self.deserializer.deserialize(memory_obj)
    return decompressed
```

### `submit_put_task(key, memory_obj) → Future`

```python
def submit_put_task(self, key, memory_obj) -> Optional[Future]:
    if self.connection is None:
        return None

    memory_obj.ref_count_up()

    # シリアライズ（圧縮）
    compressed = self.serializer.serialize(memory_obj)
    memory_obj.ref_count_down()

    # 非同期送信
    future = asyncio.run_coroutine_threadsafe(
        self.connection.put(key, compressed),
        self.loop
    )

    # コールバック
    future.add_done_callback(
        lambda f: memory_obj.ref_count_down()
    )

    return future
```

---

## 🔌 コネクター実装

### RedisConnector

**ファイル**: `/home/user/LMCache/lmcache/v1/storage_backend/connector/redis_connector.py`

```python
class RedisConnector:
    async def get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        key_str = key.to_string()

        # メタデータ取得
        metadata_bytes = self.connection.get(key_str + "metadata")
        if metadata_bytes is None:
            return None

        metadata = RemoteMetadata.deserialize(metadata_bytes)

        # CPU メモリ割当
        memory_obj = self.local_cpu_backend.allocate(
            metadata.shape,
            metadata.dtype,
        )

        # KV データ取得
        kv_bytes = self.connection.get(key_str + "kv_bytes")
        view = memoryview(memory_obj.byte_array)
        view[:metadata.length] = kv_bytes

        return memory_obj

    async def put(self, key: CacheEngineKey, memory_obj):
        key_str = key.to_string()

        # メタデータ保存
        metadata = RemoteMetadata(
            shape=memory_obj.get_shape(),
            dtype=memory_obj.get_dtype(),
            length=memory_obj.get_size(),
        )
        self.connection.set(
            key_str + "metadata",
            metadata.serialize()
        )

        # KV データ保存
        self.connection.set(
            key_str + "kv_bytes",
            bytes(memory_obj.byte_array)
        )
```

### Redis キー構造

```
Redis Keys:
  {key_string}metadata    → RemoteMetadata (シリアライズ)
  {key_string}kv_bytes    → KV テンソルのバイト列

例:
  KV_BLOB@bert@1@0@abc123metadata    (メタデータ)
  KV_BLOB@bert@1@0@abc123kv_bytes    (データ)
```

---

## 🔄 シリアライゼーション

### Naive Serde

```python
class NaiveSerializer:
    def serialize(self, memory_obj) -> bytes:
        """CPU メモリ → バイト列"""
        return bytes(memory_obj.byte_array)

    def deserialize(self, data: bytes, metadata) -> MemoryObj:
        """バイト列 → CPU メモリ"""
        memory_obj = self.allocator.allocate(
            metadata.shape,
            metadata.dtype,
        )
        memory_obj.byte_array[:] = data
        return memory_obj
```

---

## 📊 パフォーマンス特性

| メトリクス | 値 |
|----------|-----|
| レイテンシ | ~1-100 ms |
| スループット | ~1 GB/s |
| 容量 | 無制限 |
| CPU負荷 | 低（ネットワーク支配） |

---

**バージョン**: v1
**最終更新**: 2025-11-23

