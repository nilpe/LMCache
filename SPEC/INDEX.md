# LMCache 仕様書 - 完全インデックス

> すべてのドキュメントへのリンクと検索ガイド

---

## 📑 ドキュメント一覧（推奨順）

### ⭐ 必読（基礎3点セット）

1. **[README.md](./README.md)** - 仕様書ガイド・全体構成
2. **[00_Architecture.md](./00_Architecture.md)** - システムアーキテクチャ全体図
3. **[01_StorageManager.md](./01_StorageManager.md)** - ストレージ統合インターフェース

### 🔧 コアエンジン（推奨順）

4. **[06_CacheEngine.md](./06_CacheEngine.md)** - lookup / retrieve / store 核処理
5. **[07_TokenDatabase.md](./07_TokenDatabase.md)** - トークン → キャッシュキー変換
6. **[08_GPUConnector.md](./08_GPUConnector.md)** - GPU メモリ転送

### 💾 メモリ・ストレージ層

7. **[02_MemoryManagement.md](./02_MemoryManagement.md)** - MemoryObj, メモリアロケータ
8. **[10_Evictor.md](./10_Evictor.md)** - キャッシュ削除ポリシー（LRU/LFU）

### 🏗️ ストレージバックエンド（4つ）

9. **[03_StorageBackends/03a_LocalCPUBackend.md](./03_StorageBackends/03a_LocalCPUBackend.md)** - ホットキャッシュ（CPU メモリ）
10. **[03_StorageBackends/03b_LocalDiskBackend.md](./03_StorageBackends/03b_LocalDiskBackend.md)** - ウォームキャッシュ（ローカルディスク）
11. **[03_StorageBackends/03c_RemoteBackend.md](./03_StorageBackends/03c_RemoteBackend.md)** - リモートキャッシュ（Redis/P2P）
12. **[03_StorageBackends/03d_GDSBackend.md](./03_StorageBackends/03d_GDSBackend.md)** - GPU Direct Storage (超高速DMA)

### 🔌 インテグレーション層

13. **[04_vLLMIntegration.md](./04_vLLMIntegration.md)** - vLLM との統合（Lookup/Retrieve/Store）
14. **[09_Protocol.md](./09_Protocol.md)** - 通信プロトコル定義（v1 API）

### 🎯 高度なトピック

15. **[05_ReferenceCountingAndAsync.md](./05_ReferenceCountingAndAsync.md)** - 参照カウント・非同期処理詳細

---

## 🔍 機能別検索ガイド

### vLLM ユーザー向け

```
Q: vLLMからのクエリはどう処理される？
A: [04_vLLMIntegration.md](./04_vLLMIntegration.md) +
   [06_CacheEngine.md](./06_CacheEngine.md) (lookup/retrieve/store)

Q: 推論速度を最大化するには?
A: [03_StorageBackends/03d_GDSBackend.md](./03_StorageBackends/03d_GDSBackend.md) (GPU Direct Storage)

Q: キャッシュヒット率を改善するには?
A: [07_TokenDatabase.md](./07_TokenDatabase.md) (chunk_size設定) +
   [10_Evictor.md](./10_Evictor.md) (LRU/LFU選択)
```

### 開発者向け

```
Q: 新しいバックエンドを追加するには?
A: [01_StorageManager.md](./01_StorageManager.md) +
   [03_StorageBackends/](./03_StorageBackends/) (実装例参考)

Q: メモリ管理の仕組みは?
A: [02_MemoryManagement.md](./02_MemoryManagement.md) +
   [05_ReferenceCountingAndAsync.md](./05_ReferenceCountingAndAsync.md)

Q: GPU 転送の最適化は?
A: [08_GPUConnector.md](./08_GPUConnector.md) +
   [02_MemoryManagement.md](./02_MemoryManagement.md)
```

### 運用・デプロイ向け

```
Q: キャッシュ容量はどう設定する?
A: [10_Evictor.md](./10_Evictor.md) (max_cache_size) +
   [01_StorageManager.md](./01_StorageManager.md) (バックエンド選択)

Q: パフォーマンスプロファイリングは?
A: 各ドキュメントの「パフォーマンス特性」セクション

Q: トラブルシューティング?
A: [01_StorageManager.md](./01_StorageManager.md) (トラブル対応表)
```

---

## 🎓 学習パス（3段階）

### Lv1: 初心者向け（3時間）

1. [README.md](./README.md) を読む
2. [00_Architecture.md](./00_Architecture.md) でシステム理解
3. [04_vLLMIntegration.md](./04_vLLMIntegration.md) で外部インターフェース確認

### Lv2: 実装者向け（1日）

1. [01_StorageManager.md](./01_StorageManager.md) で全体管理
2. [06_CacheEngine.md](./06_CacheEngine.md) でコア処理
3. [02_MemoryManagement.md](./02_MemoryManagement.md) でメモリモデル
4. [03_StorageBackends/](./03_StorageBackends/) で各バックエンド理解

### Lv3: 専門家向け（深掘り）

1. [05_ReferenceCountingAndAsync.md](./05_ReferenceCountingAndAsync.md) で並行制御
2. [07_TokenDatabase.md](./07_TokenDatabase.md) でハッシングアルゴリズム
3. [10_Evictor.md](./10_Evictor.md) でメモリ管理戦略
4. [03_StorageBackends/03d_GDSBackend.md](./03_StorageBackends/03d_GDSBackend.md) で GPU 最適化

---

## 📊 ドキュメント概要表

| No. | ドキュメント | 対象クラス | 行数 | 難易度 |
|-----|-----------|----------|------|--------|
| 00 | Architecture | システム | - | ★☆☆ |
| 01 | StorageManager | `StorageManager` | ~500 | ★★☆ |
| 02 | MemoryManagement | `MemoryObj`, Allocator | ~800 | ★★★ |
| 03a | LocalCPUBackend | `LocalCPUBackend` | ~250 | ★☆☆ |
| 03b | LocalDiskBackend | `LocalDiskBackend` | ~300 | ★★☆ |
| 03c | RemoteBackend | `RemoteBackend` | ~300 | ★★☆ |
| 03d | GDSBackend | `GdsBackend` | ~300 | ★★★ |
| 04 | vLLMIntegration | v1 Adapter | ~400 | ★★☆ |
| 05 | RefCounting & Async | 並行制御 | ~300 | ★★★ |
| 06 | CacheEngine | `CacheEngine` | ~350 | ★★☆ |
| 07 | TokenDatabase | `TokenDatabase` | ~350 | ★★☆ |
| 08 | GPUConnector | `GPUConnector` | ~300 | ★★☆ |
| 09 | Protocol | メッセージ定義 | ~200 | ★☆☆ |
| 10 | Evictor | `Evictor` | ~200 | ★★☆ |

---

## 🔗 クロスリファレンス

### StorageManager から参照すべきドキュメント

- [02_MemoryManagement.md](./02_MemoryManagement.md) - MemoryObj詳細
- [10_Evictor.md](./10_Evictor.md) - Eviction戦略
- [03_StorageBackends/](./03_StorageBackends/) - 各バックエンド実装

### CacheEngine から参照すべきドキュメント

- [07_TokenDatabase.md](./07_TokenDatabase.md) - トークン処理
- [01_StorageManager.md](./01_StorageManager.md) - ストレージ統合
- [08_GPUConnector.md](./08_GPUConnector.md) - GPU転送

### vLLMIntegration から参照すべきドキュメント

- [06_CacheEngine.md](./06_CacheEngine.md) - コア処理
- [09_Protocol.md](./09_Protocol.md) - プロトコル定義
- [08_GPUConnector.md](./08_GPUConnector.md) - GPU転送

---

## 🎯 実装タスク別チェックリスト

### 新バックエンド実装

- [ ] [01_StorageManager.md](./01_StorageManager.md) - インターフェース確認
- [ ] [02_MemoryManagement.md](./02_MemoryManagement.md) - MemoryObj 操作
- [ ] [03_StorageBackends/](./03_StorageBackends/) - 既存実装参考
- [ ] [05_ReferenceCountingAndAsync.md](./05_ReferenceCountingAndAsync.md) - 参照カウント管理

### GPU 最適化

- [ ] [08_GPUConnector.md](./08_GPUConnector.md) - 転送実装
- [ ] [03_StorageBackends/03d_GDSBackend.md](./03_StorageBackends/03d_GDSBackend.md) - DMA 実装
- [ ] [02_MemoryManagement.md](./02_MemoryManagement.md) - GPU メモリ割当

### キャッシュ戦略

- [ ] [10_Evictor.md](./10_Evictor.md) - ポリシー選択
- [ ] [07_TokenDatabase.md](./07_TokenDatabase.md) - チャンク設定
- [ ] [01_StorageManager.md](./01_StorageManager.md) - バックエンド優先度

### vLLM 統合

- [ ] [04_vLLMIntegration.md](./04_vLLMIntegration.md) - プロトコル
- [ ] [09_Protocol.md](./09_Protocol.md) - メッセージ定義
- [ ] [06_CacheEngine.md](./06_CacheEngine.md) - API 実装

---

## 📝 ドキュメント更新履歴

| 日付 | 更新内容 |
|------|---------|
| 2025-11-23 | 初版作成、全14ドキュメント |

---

## 💡 Tips & Tricks

### パフォーマンス最適化

```
GDS (GPU Direct Storage) が利用可能な場合:
  ├─ cufile_buffer_size を 256 MiB に設定
  ├─ 4096 バイトアライメントを確認
  └─ ローカルディスク不要（GDSが高速）

CPU のみの環境:
  ├─ chunk_size = 256 を推奨
  ├─ LocalCPU + LocalDisk を併用
  └─ Remote は最後の手段
```

### デバッグ tips

```
キャッシュヒット率が低い:
  1. TokenDatabase.chunk_size 確認
  2. 参照カウント漏れ（Eviction不可）確認
  3. バックエンド優先度確認

メモリ leak の疑い:
  1. ref_count_down() が呼ばれているか確認
  2. Eviction 時の ref_count > 1 チェック
  3. コールバック内の ref_count_down() 確認
```

---

## 🔗 外部リンク

- **ソースコード**: `/home/user/LMCache/lmcache/v1/`
- **テストコード**: `/home/user/LMCache/tests/v1/`

---

**バージョン**: v1
**最終更新**: 2025-11-23
**ドキュメント数**: 14

