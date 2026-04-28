#!/bin/bash
# setup_venv.sh — build the bench venv on the login node.
#
# Lays down a Python 3.13 venv with:
#   - nilpe-vllm (ralph/hybrid-mamba-state-io-v0.19.1) editable, using
#     VLLM_USE_PRECOMPILED=1 so the heavy CUDA kernel compile is skipped
#     and vLLM's prebuilt .so wheels are downloaded instead.
#   - nilpe-lmcache (ykogi/ralph-devdax) editable, with NO_CUDA_EXT=1
#     so the cachegen CUDA kernels are skipped (we don't use cachegen
#     in the hybrid Mamba state I/O path).
#
# Run on a login node — compute nodes don't always have outbound HTTP.
# Once the venv exists, both compute and login nodes can use it.

set -euo pipefail

VENV_DIR="${VENV_DIR:-/work/0/NBB/kogi/workspace/nilpe-bench-venv/venv}"
LMCACHE_SRC="${LMCACHE_SRC:-/work/0/NBB/kogi/workspace/nilpe-lmcache}"
VLLM_SRC="${VLLM_SRC:-/work/0/NBB/kogi/workspace/nilpe-vllm}"

eval "$(spack load --sh python@3.13.5)"
module load cuda/12.9.1

if [ ! -d "${VENV_DIR}" ]; then
    uv venv --python "$(which python)" "${VENV_DIR}"
fi
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

export TORCH_CUDA_ARCH_LIST="9.0"   # H100 — login nodes have no GPU to autodetect
export VLLM_USE_PRECOMPILED=1       # download vLLM kernel .so, skip nvcc
export NO_CUDA_EXT=1                # skip cachegen CUDA kernels in lmcache

echo "[setup] installing vLLM (ralph) editable..."
uv pip install -e "${VLLM_SRC}"

echo "[setup] installing LMCache (ykogi/ralph-devdax) editable..."
uv pip install -e "${LMCACHE_SRC}"

echo "[setup] linking pre-built .so files for the devdax stack..."
for so in fast_read.so pipeline_native.so bar1_bridge.so; do
    if [ ! -e "${LMCACHE_SRC}/lmcache/v1/storage_backend/${so}" ]; then
        cp "/work/0/NBB/kogi/workspace/lmcache-dev/lmcache/v1/storage_backend/${so}" \
           "${LMCACHE_SRC}/lmcache/v1/storage_backend/${so}"
    fi
done

echo "[setup] smoke test imports..."
python -c "
from lmcache.integration.vllm.hybrid_mamba_state_io import (
    _build_disk_backend, _build_sidecar_config, maybe_install,
)
print('  ok hybrid_mamba_state_io')
from lmcache.v1.storage_backend.devdax_backend import DevDaxBackend
print('  ok DevDaxBackend')
from vllm.v1.worker.gpu_model_runner import (
    register_external_mamba_state_save_hook,
)
from vllm.v1.worker.mamba_utils import (
    register_external_mamba_state_restore_hook,
)
print('  ok vLLM ralph hooks present')
print('venv ready: ${VENV_DIR}')
"
