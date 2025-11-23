# Protocol 仕様書

> 通信プロトコル定義: v1 API, メッセージ形式

**ファイル**: `/home/user/LMCache/lmcache/v1/protocol.py`, `v1/v1_api/`
**クラス**: `ClientMetaMessage`, `ServerMetaMessage`
**責務**: 構造化メッセージ定義、シリアライゼーション

---

## 🎯 概要

Protocol は、LMCache の内部通信（ストレージバックエンド間）と外部インターフェース（vLLM）の共通メッセージ定義。

**2つのプロトコル**:
1. **v1 Protocol**: ストレージ層の通信
2. **Lookup RPC**: vLLM との lookup 通信

---

## 📖 v1 Protocol メッセージ

### ClientMetaMessage (クライアント → サーバー)

```python
@dataclass
class ClientMetaMessage:
    """LMCServerConnectorからLMCacheServerへの制御メッセージ"""

    command: int                    # CLIENT_PUT(1), CLIENT_GET(2), CLIENT_EXIST(3)
    key: CacheEngineKey            # キャッシュキー
    length: int                    # データ長（バイト）
    fmt: MemoryFormat              # メモリ形式（KV_2LTD, KV_T2Dなど）
    dtype: Optional[torch.dtype]   # データ型（fp16, bf16など）
    shape: torch.Size              # 形状 [num_layers, 2, num_tokens, hidden_size]

    def serialize(self) -> bytes:
        """182バイトのバイナリシリアライズ"""
        return struct.pack(
            f"iiiiiiii{MAX_KEY_LENGTH}s",
            self.command,
            self.length,
            int(self.fmt.value),
            DTYPE_TO_INT[self.dtype],
            self.shape[0], self.shape[1], self.shape[2], self.shape[3],
            key_str.encode().ljust(MAX_KEY_LENGTH),
        )
```

**シリアライズ形式**:
```
[4 bytes] command          (int32)
[4 bytes] length           (int32)
[4 bytes] fmt              (int32)
[4 bytes] dtype            (int32)
[4 bytes] shape[0]         (int32)
[4 bytes] shape[1]         (int32)
[4 bytes] shape[2]         (int32)
[4 bytes] shape[3]         (int32)
[150 bytes] key_string     (固定)
───────────────────────
182 bytes 合計
```

### ServerMetaMessage (サーバー → クライアント)

```python
@dataclass
class ServerMetaMessage:
    """LMCacheServerからLMCServerConnectorへのレスポンス"""

    code: int                      # SERVER_SUCCESS(200), SERVER_FAIL(400)
    length: int                    # レスポンスデータ長
    fmt: MemoryFormat              # メモリ形式
    dtype: Optional[torch.dtype]   # データ型
    shape: torch.Size              # 形状

    def serialize(self) -> bytes:
        """32バイトのバイナリシリアライズ"""
        return struct.pack(
            "iiiiiiii",
            self.code,
            self.length,
            int(self.fmt.value),
            DTYPE_TO_INT[self.dtype],
            self.shape[0], self.shape[1], self.shape[2], self.shape[3],
        )
```

---

## 🔌 Lookup RPC プロトコル

### Msgpack Encoding

```python
# トークンID配列を Msgpack でエンコード

request = {
    "type": "lookup",
    "tokens": torch.tensor([1, 2, 3, ..., 1024], dtype=torch.int32),
}

encoded = msgpack.packb(request)
# → フレーム0: メタデータ
# → フレーム1+: データ
```

### ZMQ REQ/REP

```
Client (vLLM):
  socket.send_multipart(encoded_frames)
  response = socket.recv()

Server (LMCache):
  frames = socket.recv_multipart()
  token_ids = decode(frames)
  result = lookup(token_ids)
  socket.send(result.to_bytes(4, "big"))
```

---

## 🏗️ APIエンドポイント

### REST API (v1)

```
POST /lookup
Content-Type: application/json

Request:
{
  "tokens": [1, 2, 3, ..., 1024],
  "model_name": "bert-large",
  "worker_id": 0,
}

Response:
{
  "num_cached": 256,
  "layout_info": {...}
}
```

### Socket API (従来版)

```
TCP://localhost:50000

Request:
  ClientMetaMessage (182 bytes) + KV data

Response:
  ServerMetaMessage (32 bytes) + KV data
```

---

## 📊 メッセージ形式一覧

| メッセージ | 用途 | サイズ |
|-----------|------|--------|
| ClientMetaMessage | PUT/GET制御 | 182 bytes |
| ServerMetaMessage | レスポンス | 32 bytes |
| Msgpack Frame | トークンエンコーディング | 可変 |
| int.to_bytes() | Lookup応答 | 4 bytes |

---

**バージョン**: v1
**最終更新**: 2025-11-23

