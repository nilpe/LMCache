#!/bin/bash
#PBS -A NBB
#PBS -q gpu
#PBS -l elapstim_req=2:00:00
#PBS -N rl_devdax
#PBS -o /work/0/NBB/kogi/workspace/nilpe-lmcache/bench/results/rl_devdax.log
#PBS -e /work/0/NBB/kogi/workspace/nilpe-lmcache/bench/results/rl_devdax.err
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
echo " ralph + devdax TTFT bench (4 nodes)"
echo " Started: $(date)"
echo "============================================================"

bash "${RUN_ONE}" devdax || echo "[WARN] devdax failed"

echo ""
echo "============================================================"
echo " devdax done at $(date)"
echo "============================================================"
