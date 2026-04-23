# SPDX-License-Identifier: Apache-2.0
"""
State snapshot: SIGUSR1 でメタデータをダンプ、起動時に復元。

FSDAX: dict のみダンプ (データは .pt ファイルとして残る)
DevDAX: dict + _alloc_offset をダンプ (データは /dev/dax0.0 上に残る)

環境変数:
  LMCACHE_DUMP_PATH: ダンプファイルの保存先ディレクトリ
                     デフォルト: /tmp/lmcache_snapshot/

使い方:
  # ダンプ (プロセス外から)
  kill -USR1 <EngineCore PID>

  # 復元 (自動: ダンプファイルが存在すれば起動時に読み込む)
  export LMCACHE_DUMP_PATH=/work/.../snapshot/rank_0/
  # → LMCache 起動時に自動復元
"""

import os
import pickle
import signal
import threading
from pathlib import Path
from typing import Optional

from lmcache.logging import init_logger

logger = init_logger(__name__)

_DEFAULT_DUMP_DIR = "/tmp/lmcache_snapshot"


def _get_dump_path() -> Path:
    d = Path(os.environ.get("LMCACHE_DUMP_PATH", _DEFAULT_DUMP_DIR))
    d.mkdir(parents=True, exist_ok=True)
    # ホスト名を含めて 4 ランクが同一ディレクトリに書いても衝突しない
    import socket
    hostname = socket.gethostname()
    return d / f"lmcache_state_{hostname}.pkl"


def dump_backend_state(backend) -> Optional[str]:
    """Backend の状態を pickle でダンプ。

    Returns: 保存先パス, or None on failure.
    """
    dump_path = _get_dump_path()

    state = {
        "backend_type": type(backend).__name__,
        "dict": {},
    }

    # dict の中身をシリアライズ可能な形式に変換
    for key, meta in backend.dict.items():
        state["dict"][key] = {
            "path": meta.path,
            "size": meta.size,
            "shape": tuple(meta.shape) if meta.shape is not None else None,
            "dtype": meta.dtype,
            "fmt": meta.fmt,
            "cached_positions": (
                meta.cached_positions.tolist()
                if meta.cached_positions is not None
                else None
            ),
        }

    # DevDaxBackend 固有の状態
    if hasattr(backend, "_alloc_offset"):
        state["alloc_offset"] = backend._alloc_offset
        state["map_size"] = backend._map_size

    # FSDAX: path 情報
    if hasattr(backend, "path"):
        state["disk_path"] = backend.path

    try:
        with open(dump_path, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        n_chunks = len(state["dict"])
        logger.info(
            f"State dumped: {n_chunks} chunks → {dump_path} "
            f"({dump_path.stat().st_size / 1024:.1f} KB)"
        )
        return str(dump_path)
    except Exception as e:
        logger.error(f"State dump failed: {e}")
        return None


def load_backend_state(backend) -> int:
    """ダンプファイルからメタデータを復元。

    ホスト名付き pkl を優先。なければディレクトリ内の任意の pkl をフォールバック。
    Returns: 復元したチャンク数, or 0 if no dump found.
    """
    import torch
    from lmcache.utils import DiskCacheMetadata

    dump_path = _get_dump_path()
    if not dump_path.exists():
        # フォールバック: 同一ディレクトリ内の任意の pkl
        d = dump_path.parent
        pkls = sorted(d.glob("lmcache_state_*.pkl"))
        if pkls:
            dump_path = pkls[0]
            logger.info(f"Hostname pkl not found, using fallback: {dump_path}")
        else:
            return 0

    try:
        with open(dump_path, "rb") as f:
            state = pickle.load(f)
    except Exception as e:
        logger.warning(f"Failed to load state dump: {e}")
        return 0

    backend_type = state.get("backend_type", "")
    current_type = type(backend).__name__

    # DevDaxBackend: alloc_offset を復元 (上書き防止)
    if hasattr(backend, "_alloc_offset") and "alloc_offset" in state:
        old_offset = backend._alloc_offset
        backend._alloc_offset = state["alloc_offset"]
        logger.info(
            f"Restored alloc_offset: {old_offset} → {state['alloc_offset']} "
            f"({state['alloc_offset'] / (1024**3):.2f} GB used)"
        )

    # dict を復元
    n_restored = 0
    for key, meta_dict in state["dict"].items():
        shape = (
            torch.Size(meta_dict["shape"])
            if meta_dict["shape"] is not None
            else None
        )
        cached_positions = (
            torch.tensor(meta_dict["cached_positions"])
            if meta_dict["cached_positions"] is not None
            else None
        )

        # FSDAX: ファイルが実際に存在するか確認
        if hasattr(backend, "path") and not meta_dict["path"].startswith("/dev/"):
            if not os.path.exists(meta_dict["path"]):
                logger.debug(f"Skipping missing file: {meta_dict['path']}")
                continue

        meta = DiskCacheMetadata(
            path=meta_dict["path"],
            size=meta_dict["size"],
            shape=shape,
            dtype=meta_dict["dtype"],
            fmt=meta_dict["fmt"],
            cached_positions=cached_positions,
        )
        backend.dict[key] = meta
        n_restored += 1

    logger.info(
        f"Restored {n_restored} chunks from {dump_path} "
        f"(backend={current_type}, dump_backend={backend_type})"
    )
    return n_restored


# ============================================================
# Signal handler registration
# ============================================================

_registered_backends = []
_handler_lock = threading.Lock()


def register_backend_for_snapshot(backend):
    """Backend をシグナルハンドラに登録。"""
    with _handler_lock:
        _registered_backends.append(backend)
        if len(_registered_backends) == 1:
            _install_signal_handler()


def _install_signal_handler():
    """SIGUSR1 ハンドラをインストール。"""
    prev_handler = signal.getsignal(signal.SIGUSR1)

    def _handler(signum, frame):
        logger.info("SIGUSR1 received: dumping state...")
        for backend in _registered_backends:
            dump_backend_state(backend)
        # Chain to previous handler if any
        if callable(prev_handler) and prev_handler not in (
            signal.SIG_DFL,
            signal.SIG_IGN,
        ):
            prev_handler(signum, frame)

    signal.signal(signal.SIGUSR1, _handler)
    logger.info("SIGUSR1 handler installed for state snapshot")
