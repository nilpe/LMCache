# MemoryManagement 仕様書

> MemoryObj, メモリアロケータ, 参照カウント管理

**ファイル**: `/home/user/LMCache/lmcache/v1/memory_management.py`
**クラス**: `MemoryObj`, `TensorMemoryObj`, `MemoryAllocatorInterface`, 各 Allocator
**責務**: メモリ割当・解放、参照カウント管理、メモリプール管理

---

## 🎯 概要

MemoryManagement は、LMCache全体のメモリライフサイクルを管理するコンポーネント。CPU/GPUメモリの確保、参照カウントトラッキング、メモリプール管理を提供します。

**主要責務**:
1. MemoryObj: 統一メモリ表現（CPU/GPU）
2. MemoryAllocator: First-fit メモリプール割当
3. Reference Counting: マルチバックエンド対応
4. Eviction: メモリ圧力時の削除

---

## 🏗️ クラス階層

```
MemoryObj (抽象基底クラス)
├─ TensorMemoryObj (具体実装: テンソル)
│  └─ GPU/CPU Tensor で実装
│
└─ BufferMemoryObj (具体実装: バイトバッファ)
   └─ bytearray で実装

MemoryAllocatorInterface (抽象基底クラス)
├─ TensorMemoryAllocator (First-fit)
├─ BufferAllocator (bytearray)
├─ HostMemoryAllocator (CPU)
├─ PinMemoryAllocator (CPU Pinned)
├─ GPUMemoryAllocator (GPU)
├─ CuFileMemoryAllocator (GPU + cuFile)
└─ MixedMemoryAllocator (Pinned + Buffer)
```

---

## 📖 MemoryObj

### MemoryObjMetadata

```python
@dataclass
class MemoryObjMetadata:
    shape: torch.Size                  # 論理形状
    dtype: Optional[torch.dtype]       # データ型
    address: int                       # 物理メモリ開始アドレス
    phy_size: int                      # 物理サイズ（アライン済み）
    ref_count: int                     # 参照カウント
    is_pin: bool = False               # ピン状態
    fmt: MemoryFormat = UNDEFINED      # メモリフォーマット
    cached_positions: Optional[torch.Tensor] = None
```

### TensorMemoryObj（実装）

```python
class TensorMemoryObj(MemoryObj):
    """
    Tensorベースの MemoryObj 実装

    特徴:
    - GPU/CPU Tensor 両対応
    - メモリの効率的なビュー
    - インプレイス操作可能
    """

    def __init__(self, raw_data: torch.Tensor, metadata: MemoryObjMetadata,
                 parent_allocator: Optional[MemoryAllocatorInterface] = None):
        self.raw_data = raw_data              # フラット uint8 テンソル
        self.meta = metadata
        self.parent_allocator = parent_allocator
        self.valid = True
        self.lock = threading.Lock()

    @property
    def tensor(self) -> torch.Tensor:
        """
        論理テンソルを返す（ビュー）

        [物理形状] → [論理形状]にリシェイプして返却
        """
        return self.raw_data[:self.meta.phy_size].view(
            self.meta.dtype
        ).reshape(self.meta.shape)

    @property
    def byte_array(self):
        """
        バイト配列（ファイルI/O用）
        """
        return self.raw_data[:self.meta.phy_size]

    def get_shape(self) -> torch.Size:
        return self.meta.shape

    def get_dtype(self) -> torch.dtype:
        return self.meta.dtype

    def get_size(self) -> int:
        """バイト数"""
        return self.meta.phy_size

    def ref_count_up(self) -> None:
        with self.lock:
            self.meta.ref_count += 1

    def ref_count_down(self) -> None:
        with self.lock:
            self.meta.ref_count -= 1
            # ref_count=0 かつ unpinned のみメモリ解放
            if (self.meta.ref_count == 0 and
                self.parent_allocator is not None and
                self.meta.is_pin is False):
                self.parent_allocator.free(self)

    def get_ref_count(self) -> int:
        with self.lock:
            return self.meta.ref_count
```

---

## 💾 MemoryAllocator

### TensorMemoryAllocator（First-fit）

**ファイル行**: 496-651

```python
class TensorMemoryAllocator:
    """
    First-fit アルゴリズムを使用したメモリプール管理

    フリーブロックを ソート済みリスト（SortedList）で管理
    """

    ALIGN_BYTES = 512  # アライメント単位（NVMe ブロック等）

    def __init__(self, tensor: torch.Tensor, align_bytes: int = 512):
        """
        Args:
            tensor: メモリプール用テンソル
            align_bytes: アライメント（通常512）
        """
        self.buffer = tensor.view(torch.uint8).flatten()
        self.explicit_list = sortedcontainers.SortedList(key=lambda x: x.start)

        # 初期状態: 全体が1つのフリーブロック
        self.explicit_list.add(FreeBlock(start=0, size=self.buffer.numel()))

    def allocate(self, shape: torch.Size, dtype: torch.dtype,
                 fmt: MemoryFormat) -> Optional[TensorMemoryObj]:
        """
        メモリ割当（First-fit）

        Returns:
            TensorMemoryObj（成功）, None（失敗）

        アルゴリズム:
            1. 必要サイズ計算
            2. explicit_list から First-fit 検索
            3. ブロック分割
            4. TensorMemoryObj 作成
        """
        # ステップ1: 必要サイズ計算
        raw_size = shape.numel() * dtype.itemsize
        aligned_size = (raw_size + self.ALIGN_BYTES - 1) & ~(self.ALIGN_BYTES - 1)

        # ステップ2: First-fit 検索
        selected_block = None
        for block in self.explicit_list:
            if block.size >= aligned_size:
                selected_block = block
                break

        if selected_block is None:
            return None  # メモリ不足

        # ステップ3: ブロック割当と分割
        self.explicit_list.remove(selected_block)

        if selected_block.size > aligned_size:
            # 余ったブロックをフリーリストに戻す
            self.explicit_list.add(
                FreeBlock(
                    start=selected_block.start + aligned_size,
                    size=selected_block.size - aligned_size
                )
            )

        # ステップ4: TensorMemoryObj 作成
        obj_tensor = self.buffer[
            selected_block.start : selected_block.start + raw_size
        ]

        return TensorMemoryObj(
            raw_data=obj_tensor,
            metadata=MemoryObjMetadata(
                shape=shape,
                dtype=dtype,
                address=selected_block.start,
                phy_size=aligned_size,
                ref_count=1,
                is_pin=False,
                fmt=fmt,
            ),
            parent_allocator=self,
        )

    def free(self, memory_obj: TensorMemoryObj) -> None:
        """
        メモリ解放（ブロック統合）

        フリーブロックリストに戻す
        隣接ブロックがあれば統合
        """
        metadata = memory_obj.meta
        start = metadata.address
        size = metadata.phy_size

        # フリーブロックを追加
        free_block = FreeBlock(start=start, size=size)
        self.explicit_list.add(free_block)

        # 隣接ブロック統合（簡略版）
        # 実装では分裂による外部フラグメンテーション対策
```

**メモリプール状態遷移例**:

```
初期:
  [Free: 0-8GB]

allocate(2GB):
  [Allocated: 0-2GB] [Free: 2-8GB]

allocate(3GB):
  [Allocated: 0-2GB] [Allocated: 2-5GB] [Free: 5-8GB]

free(0-2GB):
  [Free: 0-2GB] [Allocated: 2-5GB] [Free: 5-8GB]

allocate(1.5GB):
  [Allocated: 0-1.5GB] [Free: 1.5-2GB] [Allocated: 2-5GB] [Free: 5-8GB]
```

### CuFileMemoryAllocator（GDS対応）

**ファイル行**: 997-1014

```python
class CuFileMemoryAllocator(GPUMemoryAllocator):
    """
    GPU Direct Storage（GDS）対応アロケータ

    特徴:
    - GPU メモリを確保
    - cuFile に登録（DMA対応）
    - NVMe ストレージと直結可能
    """

    def __init__(self, size: int, device=None):
        from cufile.bindings import cuFileBufRegister

        if device is None:
            device = f"cuda:{torch.cuda.current_device()}"

        # GPU メモリを確保（4096バイト境界アライン）
        super().__init__(size, device, align_bytes=4096)

        # cuFile に登録（DMA転送対応化）
        self.base_pointer = self.tensor.data_ptr()
        cuFileBufRegister(
            ctypes.c_void_p(self.base_pointer),
            size,
            flags=0
        )
```

### MixedMemoryAllocator（Pinned + Buffer）

```python
class MixedMemoryAllocator(MemoryAllocatorInterface):
    """
    CPU Pinned メモリとバッファを組み合わせたアロケータ
    """

    def __init__(self, size: int):
        # Pinned CPU メモリ
        buffer = torch.empty(size, dtype=torch.uint8, pin_memory=True)
        self.pin_allocator = TensorMemoryAllocator(buffer)

        # バッファ（バイト配列）
        self.buffer_allocator = BufferAllocator("cpu")

    def allocate(self, shape: torch.Size, dtype: Optional[torch.dtype],
                 fmt: MemoryFormat = MemoryFormat.KV_2LTD) -> Optional[MemoryObj]:
        """
        メモリ形式に基づいて適切なアロケータを選択
        """
        if fmt == MemoryFormat.BINARY_BUFFER:
            return self.buffer_allocator.allocate(shape, dtype, fmt)
        else:
            return self.pin_allocator.allocate(shape, dtype, fmt)
```

---

## 🔄 参照カウント管理

### ライフサイクル

```
allocate()
  ├─ ref_count = 1 (アロケータが保有)
  │
  ├─ LocalCPUBackend.submit_put_task()
  │  └─ ref_count_up() → ref_count = 2
  │
  ├─ LocalDiskBackend.submit_put_task()
  │  └─ ref_count_up() → ref_count = 3
  │
  ├─ RemoteBackend.submit_put_task()
  │  └─ ref_count_up() → ref_count = 4
  │
  ├─ コールバック完了（各バックエンド）
  │  └─ ref_count_down() × 3 → ref_count = 1
  │
  └─ ref_count_down() → ref_count = 0
     └─ parent_allocator.free() (自動メモリ解放)
```

### Eviction 時の保護

```python
def allocate(self, shape, dtype, eviction=True):
    # メモリ割当試行
    memory_obj = allocator.allocate(shape, dtype)
    if memory_obj is not None or not eviction:
        return memory_obj

    # Eviction 候補探索
    evict_keys = []
    for candidate_key in hot_cache:
        candidate_obj = hot_cache[candidate_key]

        # ref_count > 1 なら他バックエンドで使用中
        # → eviction 不可
        if candidate_obj.get_ref_count() > 1:
            continue

        # 安全に削除可能
        evict_keys.append(candidate_key)
        candidate_obj.ref_count_down()

        # 再度割当試行
        memory_obj = allocator.allocate(shape, dtype)
        if memory_obj is not None:
            break

    return memory_obj
```

---

## 📊 メモリレイアウト図

### CPU Pinned Memory

```
┌─────────────────────────────────────────┐
│     CPU VRAM (例: 10GB, Pinned)          │
├─────────────────────────────────────────┤
│ ┌───────────────────────────────────┐   │
│ │ MixedMemoryAllocator              │   │
│ ├───────────────────────────────────┤   │
│ │ ┌─────────────┐ ┌──────────────┐  │   │
│ │ │ Allocated   │ │ Allocated    │  │   │
│ │ │ 2 GB        │ │ 3 GB         │  │   │
│ │ └─────────────┘ └──────────────┘  │   │
│ │                                    │   │
│ │ ┌──────────────────────────────┐  │   │
│ │ │ Free                         │  │   │
│ │ │ 5 GB                         │  │   │
│ │ └──────────────────────────────┘  │   │
│ │                                    │   │
│ │ TensorMemoryAllocator:             │   │
│ │ ├─ explicit_list: [(Free blocks)]  │   │
│ │ ├─ align_bytes: 512                │   │
│ │ └─ device_mem_lock: Lock           │   │
│ └───────────────────────────────────┘   │
└─────────────────────────────────────────┘
```

### GPU Memory (GDS)

```
┌─────────────────────────────────────────┐
│     GPU VRAM (例: 8GB + cuFile)         │
├─────────────────────────────────────────┤
│ ┌───────────────────────────────────┐   │
│ │ CuFileMemoryAllocator             │   │
│ ├───────────────────────────────────┤   │
│ │ base_pointer: 0x7f123...          │   │
│ │                                    │   │
│ │ ┌─────────────┐ ┌──────────────┐  │   │
│ │ │ Allocated   │ │ Allocated    │  │   │
│ │ │ 64 MiB      │ │ 32 MiB       │  │   │
│ │ └─────────────┘ └──────────────┘  │   │
│ │                                    │   │
│ │ ┌──────────────────────────────┐  │   │
│ │ │ Free                         │  │   │
│ │ │ 32 MiB                       │  │   │
│ │ └──────────────────────────────┘  │   │
│ │                                    │   │
│ │ cuFile登録: DMA対応化              │   │
│ └───────────────────────────────────┘   │
└─────────────────────────────────────────┘
```

---

## 🎯 使用パターン

### パターン1: CPU Memory 割当

```python
allocator = MixedMemoryAllocator(10 * 1024**3)  # 10 GB

shape = torch.Size([2, 32, 1024, 64])
dtype = torch.float16

memory_obj = allocator.allocate(shape, dtype)
# → raw_data: CPU Pinned テンソル
# → ref_count: 1
# → dtype: float16
```

### パターン2: GPU Memory 割当（GDS）

```python
allocator = CuFileMemoryAllocator(128 * 1024**2)  # 128 MiB

memory_obj = allocator.allocate(shape, dtype)
# → raw_data: GPU テンソル
# → cuFile登録済み（DMA可能）
# → align_bytes: 4096
```

### パターン3: 参照カウント管理

```python
# 割当
memory_obj = allocator.allocate(shape, dtype)  # ref_count=1

# バックエンド保有
memory_obj.ref_count_up()  # ref_count=2

# 使用
tensor = memory_obj.tensor
# ...

# 使用終了
memory_obj.ref_count_down()  # ref_count=1

# バックエンド削除
memory_obj.ref_count_down()  # ref_count=0 → free()
```

---

## 🔗 関連ドキュメント

- **[01_StorageManager.md](./01_StorageManager.md)** - 統合管理
- **[05_ReferenceCountingAndAsync.md](./05_ReferenceCountingAndAsync.md)** - 詳細マニュアル

---

**バージョン**: v1
**最終更新**: 2025-11-23

