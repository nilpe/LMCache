# LMCache 仕様書

> vLLM と連携した KV-Cache 外部ストレージシステムの完全仕様書

## 📑 ドキュメント構成

このスペック群は、LMCacheのコアメカニズムを**ファイル/クラス単位**で詳細に解説しています。

### 必読順序

#### コア基礎（必読）
1. **[00_Architecture.md](./00_Architecture.md)** ⭐⭐⭐ 最優先
   - LMCache 全体アーキテクチャ
   - vLLM との統合フロー
   - コンポーネント間の関係図

2. **[01_StorageManager.md](./01_StorageManager.md)** ⭐⭐
   - StorageManager の責務と設計
   - MemoryObj 取得フロー（get_blocking）
   - バックエンド検索順序

3. **[02_MemoryManagement.md](./02_MemoryManagement.md)** ⭐⭐
   - MemoryObj クラス階層
   - メモリアロケータ（Allocator）の実装
   - 参照カウント管理

#### コアエンジン層
4. **[06_CacheEngine.md](./06_CacheEngine.md)**
   - CacheEngine: lookup / retrieve / store
   - トークンバッファリング
   - ライフサイクル管理

5. **[07_TokenDatabase.md](./07_TokenDatabase.md)**
   - token_ids → CacheEngineKey 変換
   - チャンク分割と SHA256 ハッシング
   - 累積ハッシュ計算

6. **[08_GPUConnector.md](./08_GPUConnector.md)**
   - MemoryObj → GPU テンソル転送
   - vLLM Paged KV バッファ対応
   - GPU↔CPU ビルディングブロック

#### ストレージバックエンド
7. **Storage Backend 仕様** (5つのバックエンド)
   - **[03a_LocalCPUBackend.md](./03_StorageBackends/03a_LocalCPUBackend.md)**
     - ホットキャッシュ（メモリ上）
     - LRU 管理

   - **[03b_LocalDiskBackend.md](./03_StorageBackends/03b_LocalDiskBackend.md)**
     - ローカルディスク（CPU 経由）
     - 非同期I/O

   - **[03c_RemoteBackend.md](./03_StorageBackends/03c_RemoteBackend.md)**
     - Redis / P2P リモートストレージ
     - シリアライゼーション

   - **[03d_GDSBackend.md](./03_StorageBackends/03d_GDSBackend.md)** ⭐⭐ GPU Direct Storage
     - cuFile DMA 転送
     - GPU メモリ直接アクセス

   - **[03e_WekaGDSBackend.md](./03_StorageBackends/03e_WekaGDSBackend.md)**
     - Weka FileSystem 特化版

#### インテグレーション層
8. **[04_vLLMIntegration.md](./04_vLLMIntegration.md)** ⭐
   - vLLMからのクエリプロトコル
   - Lookup / Retrieve / Store フロー
   - v1 Adapter（ZMQ RPC）

9. **[09_Protocol.md](./09_Protocol.md)**
   - ClientMetaMessage / ServerMetaMessage
   - Msgpack エンコーディング
   - API エンドポイント定義

#### 高度なトピック
10. **[05_ReferenceCountingAndAsync.md](./05_ReferenceCountingAndAsync.md)**
    - 参照カウント管理機構
    - 非同期処理とスレッド管理
    - メモリライフサイクル

11. **[10_Evictor.md](./10_Evictor.md)**
    - LRU / LFU キャッシュ削除
    - メモリ圧力時の制御
    - キャッシュサイズ管理

---

## 🎯 主要なコンポーネント

### ファイル配置

```
/home/user/LMCache/
├── lmcache/
│   ├── integration/vllm/
│   │   ├── vllm_v1_adapter.py      ← vLLM統合
│   │   └── vllm_adapter.py
│   │
│   ├── v1/
│   │   ├── cache_engine.py         ← コアエンジン
│   │   ├── cache_interface.py
│   │   ├── token_database.py       ← トークンハッシング
│   │   ├── protocol.py             ← プロトコル定義
│   │   ├── memory_management.py    ← メモリ管理
│   │   ├── gpu_connector.py        ← GPU転送
│   │   │
│   │   └── storage_backend/
│   │       ├── storage_manager.py  ← ストレージ統合インターフェース
│   │       ├── abstract_backend.py
│   │       ├── local_cpu_backend.py
│   │       ├── local_disk_backend.py
│   │       ├── remote_backend.py
│   │       ├── gds_backend.py      ← GPU Direct Storage
│   │       ├── weka_gds_backend.py
│   │       ├── evictor/
│   │       │   ├── base_evictor.py
│   │       │   └── lru_evictor.py
│   │       ├── connector/
│   │       │   ├── base_connector.py
│   │       │   └── redis_connector.py
│   │       └── naive_serde/
│   │
│   ├── server/
│   │   └── __main__.py             ← ソケット API
│   │
│   └── utils.py
```

### クラス図（簡略）

```
StorageBackendInterface (抽象)
├── LocalCPUBackend (ホットキャッシュ, LRU)
├── LocalDiskBackend (ローカルディスク, 非同期)
├── RemoteBackend (Redis/P2P, シリアライズ)
├── GdsBackend (GPU Direct Storage, DMA)
└── WekaGdsBackend (Weka 特化)

MemoryObj (抽象)
├── TensorMemoryObj (CPU/GPU テンソル)
└── BufferMemoryObj (バイトバッファ)

MemoryAllocatorInterface (抽象)
├── TensorMemoryAllocator (First-fit メモリプール)
├── BufferAllocator (Bytearray ベース)
├── GPUMemoryAllocator (GPU VRAM)
├── CuFileMemoryAllocator (GPU + cuFile登録)
├── PinMemoryAllocator (CPU ピンド)
└── MixedMemoryAllocator (Pinned + Buffer)

CacheEngine (コア実装)
├── lookup() → キャッシュ済みトークン数
├── retrieve() → KV テンソル取得
└── store() → KV テンソル保存
```

---

## 🔄 主要フロー

### 1. vLLM → LMCache Lookup フロー

```
vLLM Scheduler
  └─ request.prompt_token_ids = [t0, t1, ..., tN]
      ↓ ZMQ REQ
      LMCacheLookupClient.lookup([tokens])
      ↓
      LMCacheLookupServer (別プロセス)
        ├─ TokenDatabase: token_ids → キャッシュキーに変換
        ├─ StorageManager.lookup(): バックエンド検索
        └─ 「256トークンキャッシュ済み」
      ↓ ZMQ REP
      vLLM に結果返却
```

### 2. StorageManager.get(key) → MemoryObj フロー

```
StorageManager.get(key)
  ├─[1] プリフェッチタスク待機
  ├─[2] バックエンド順序探索（短絡評価）
  │  ├─ LocalCPUBackend.get_blocking()
  │  │   └─ hot_cache[key] → 即返却
  │  ├─ LocalDiskBackend.get_blocking()
  │  │   └─ ディスク読込 + メモリ割当
  │  ├─ RemoteBackend.get_blocking()
  │  │   └─ Redis取得 + デシリアライズ
  │  └─ GdsBackend.get_blocking()
  │      └─ DMA転送（GPU直接）
  └─[3] ライトバック（非LocalCPU時）
        └─ LocalCPU にコピー
```

### 3. GPU Direct Storage (GDS) フロー

```
NVMe Storage
  └─ (DMA: CPU なし)
      ↓
GPU Memory
  └─ cuFile 登録
      ↓
      推論即座に使用可能

vs 従来:
NVMe → CPU RAM → GPU（CPU を経由, 遅い）
```

---

## 📊 パフォーマンス特性

### ストレージバックエンド比較

| 特性 | LocalCPU | LocalDisk | Remote | GDS |
|------|----------|-----------|--------|-----|
| **アクセス速度** | 最速 | 速 | 遅 | 最速 |
| **レイテンシ** | <1 μs | ~500 μs | ~1-100 ms | ~100 μs |
| **スループット** | >100 GB/s | ~5 GB/s | ~1 GB/s | 10+ GB/s |
| **ストレージ容量** | メモリ制限 | TB級 | 無制限 | TB級 |
| **CPU負荷** | N/A | 高 | 中 | 低 |
| **DMA対応** | N/A | × | × | ✓ |
| **用途** | ホットキャッシュ | ウォームキャッシュ | バックアップ | 高速推論 |

---

## 🔑 重要な概念

### 参照カウント（Reference Counting）

```
allocate()           → ref_count = 1
  ↓
ref_count_up()      → ref_count = 2 (複数バックエンド保有)
  ↓
ref_count_down()    → ref_count = 1
  ↓
ref_count_down()    → ref_count = 0 → メモリ自動解放
```

### メモリプール（Memory Pool）

```
TensorMemoryAllocator
├─ Buffer: 連続した GPU/CPU メモリ
├─ explicit_list: フリーブロック一覧（ソート済み）
├─ allocate(): First-fit 検索
├─ free(): ブロック統合
└─ align_bytes: アライメント（4096バイト）
```

### LRU キャッシュ管理

```
OrderedDict（Python 3.7+）
├─ 挿入順序を保持
├─ move_to_end(): MRU に更新
├─ FIFO削除: LRU から削除対象
└─ eviction: サイズ超過時に LRU から削除
```

---

## ⚙️ 設定・初期化

### 基本設定（config.yaml）

```yaml
# ストレージ選択
chunk_size: 256
local_cpu: true                    # LocalCPU 有効
max_local_cpu_size: 10            # 10 GB
local_disk: true                  # LocalDisk 有効
local_disk_path: "/path/to/cache"
gds_path: "/tmp/gds"              # GDS 有効
cufile_buffer_size: 128           # 128 MiB

# vLLM 統合
enable_p2p: true
enable_controller: true
```

### vLLM プロトコル（v1 API）

```python
# ZMQ RPC
ipc://lmcache_rpc_port_xxxx

# リクエスト形式: Msgpack(token_ids)
# レスポンス形式: int (4バイト)
```

---

## 📝 図解・ダイアグラム

各スペック内には以下の図解を含みます：

- **アーキテクチャ図**: コンポーネント間の関係
- **フロー図**: 実行フロー（時系列）
- **メモリレイアウト**: メモリプール構造
- **タイムラインダイアグラム**: 非同期処理フロー

---

## 🔗 関連リソース

- **ソースコード**: `/home/user/LMCache/lmcache/v1/`
- **テストコード**: `/home/user/LMCache/tests/v1/`
- **設定例**: `/home/user/LMCache/tests/v1/data/`

---

**バージョン**: LMCache V1
**最終更新**: 2025-11-23
**言語**: 日本語（英数字混用）

---

## 📖 ドキュメント使用方法

```
初心者向け:
  1. 00_Architecture.md から開始
  2. 01_StorageManager.md で基本フローを理解
  3. 各バックエンドを順番に学習

開発者向け:
  1. 02_MemoryManagement.md でメモリモデルを理解
  2. 各バックエンド仕様から実装詳細を確認
  3. 04_vLLMIntegration.md でインターフェースを確認

GDS検討者向け:
  1. 03d_GDSBackend.md を優先
  2. 05_ReferenceCountingAndAsync.md で非同期処理を理解
  3. パフォーマンス比較表を参照
```

