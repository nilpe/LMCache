#!/bin/bash
#PBS -A NBB
#PBS -q gpu
#PBS -l elapstim_req=2:00:00
#PBS -N rl_dvf
#PBS -o /work/0/NBB/kogi/workspace/nilpe-lmcache/bench/results/rl_dvf.log
#PBS -e /work/0/NBB/kogi/workspace/nilpe-lmcache/bench/results/rl_dvf.err
#PBS -M ykogi@hpcs.cs.tsukuba.ac.jp
#PBS -m e
#PBS -b 4
#PBS -T openmpi
#PBS -v NQSV_MPI_VER=5.0.10/intel2023.0.0-cuda12.9.1
#PBS -v USE_DEVDAX=pmemkv

set -uo pipefail
module load openmpi/5.0.10/intel2023.0.0-cuda12.9.1
export no_proxy=localhost,127.0.0.1
export NO_PROXY=localhost,127.0.0.1

ROOT="/work/0/NBB/kogi/workspace/nilpe-lmcache/bench"
RUN_ONE="${ROOT}/scripts/run_one.sh"

echo "============================================================"
echo " ralph + {fsdax, devdax} TTFT bench (4 nodes)"
echo " Started: $(date)"
echo "============================================================"

# Order: fsdax first (cleans /pmem/lmcache each time), then devdax (slab
# persists across runs but each run resets prefix-token domain via the
# multi-decode rust harness's session-id rotation).
for label in fsdax devdax; do
    bash "${RUN_ONE}" "${label}" || echo "[WARN] ${label} failed, continuing"
done

# Final cleanup: don't leave the user-prefixed PMEM dir behind.
mpirun ${NQSV_MPIOPTS} -np 4 -npernode 1 --bind-to none \
    bash -c "rm -rf /pmem/${USER}_lmcache_qwen36" 2>&1 || true

echo ""
echo "============================================================"
echo " all done at $(date)"
echo "============================================================"
