#!/bin/bash
#PBS -A NBB
#PBS -q gpu
#PBS -l elapstim_req=2:00:00
#PBS -N rl_fsdax
#PBS -o /work/0/NBB/kogi/workspace/nilpe-lmcache/bench/results/rl_fsdax.log
#PBS -e /work/0/NBB/kogi/workspace/nilpe-lmcache/bench/results/rl_fsdax.err
#PBS -M ykogi@hpcs.cs.tsukuba.ac.jp
#PBS -m e
#PBS -b 4
#PBS -T openmpi
#PBS -v NQSV_MPI_VER=5.0.10/intel2023.0.0-cuda12.9.1
# NOTE: USE_DEVDAX=pmemkv intentionally NOT set — that flag mounts PMEM as
# raw DAX (/dev/dax0.0) and makes /pmem/ unwritable. fsdax baseline needs
# the FSDAX /pmem/ mount, which is what the gpu queue ships when
# USE_DEVDAX is unset.

set -uo pipefail
module load openmpi/5.0.10/intel2023.0.0-cuda12.9.1
export no_proxy=localhost,127.0.0.1
export NO_PROXY=localhost,127.0.0.1

ROOT="/work/0/NBB/kogi/workspace/nilpe-lmcache/bench"
RUN_ONE="${ROOT}/scripts/run_one.sh"

echo "============================================================"
echo " ralph + fsdax TTFT bench (4 nodes)"
echo " Started: $(date)"
echo "============================================================"

bash "${RUN_ONE}" fsdax || echo "[WARN] fsdax failed"

# Final cleanup
mpirun ${NQSV_MPIOPTS} -np 4 -npernode 1 --bind-to none \
    bash -c "rm -rf /pmem/${USER}_lmcache_qwen36 2>/dev/null" 2>&1 || true

echo ""
echo "============================================================"
echo " fsdax done at $(date)"
echo "============================================================"
