#!/bin/bash
# job_ttft_devdax_vs_fsdax.sh — PBS job comparing fsdax vs devdax on a
# PMEM-equipped Pegasus node, using the ralph branches with this fork's
# YAML-driven sidecar dispatch (db58a844).
#
# Submit with: qsub bench/scripts/job_ttft_devdax_vs_fsdax.sh
#
# Output: bench/results/<timestamp>_<cond>/summary.txt per condition.
#
# This script DOES NOT submit itself. The user said qsub on hold; submit
# manually after reviewing.

#PBS -A NBBG
#PBS -q gen_S
#PBS -l elapstim_req=2:00:00
#PBS -N ttft_dvf
#PBS -o /work/0/NBB/kogi/workspace/nilpe-lmcache/bench/results/ttft_dvf.log
#PBS -e /work/0/NBB/kogi/workspace/nilpe-lmcache/bench/results/ttft_dvf.err
#PBS -M ykogi@hpcs.cs.tsukuba.ac.jp
#PBS -m e
#PBS -T openmpi
#PBS -v NQSV_MPI_VER=5.0.10/intel2023.0.0-cuda12.9.1
#PBS -v USE_DEVDAX=pmemkv

set -uo pipefail

ROOT="/work/0/NBB/kogi/workspace/nilpe-lmcache"
PROMPT_FILE="${ROOT}/bench/prompts/long_prefix.txt"

module load openmpi/5.0.10/intel2023.0.0-cuda12.9.1
module load cuda/12.9.1

# Source the venv that has nilpe-lmcache (this fork) + nilpe-vllm
# installed editable. Adjust if you keep a different venv name.
source "${ROOT}/venv/bin/activate"

# Sanity: dax device should be present on this node
if [ ! -e /dev/dax0.0 ]; then
    echo "[FATAL] /dev/dax0.0 not present — wrong node?" >&2
    exit 1
fi

run_one() {
    local cond="$1"
    echo ""
    echo "================================================================"
    echo " ${cond} starting at $(date)"
    echo "================================================================"
    COND="${cond}" \
    MODEL="Qwen/Qwen3.5-9B" \
    ROOT="${ROOT}" \
    PROMPT_FILE="${PROMPT_FILE}" \
    bash "${ROOT}/bench/scripts/run_ttft.sh"
}

run_one fsdax
run_one devdax

echo ""
echo "================================================================"
echo " all-done at $(date)"
echo "================================================================"
