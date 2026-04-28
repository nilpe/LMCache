#!/bin/bash
# run_one.sh - Run a single multi-decode TOML across 4 ranks with mpstat.
# Mirrors lmcache-perf-analysis/ttft_bench/scripts/run_one_config.sh.
# Fail-fast: any rank's vLLM crash → exit non-zero.
#
# Args:
#   $1 = label (fsdax | devdax)

set -uo pipefail

LABEL="$1"

ROOT="/work/0/NBB/kogi/workspace/nilpe-lmcache/bench"
BIN="/home/NBB/kogi/work/workspace/archived/prog/multi-decode/target/release/multi-decode"

CFG_TOML="${ROOT}/configs/multidecode_${LABEL}.toml"
WORKDIR="${ROOT}/results/${LABEL}/$(date +%Y%m%d_%H%M%S)"
mkdir -p "${WORKDIR}"

CPU_LOG_DIR="${WORKDIR}/cpu_logs"
mkdir -p "${CPU_LOG_DIR}"

# fsdax writes blob files under /pmem/${USER}_lmcache_qwen36/{attn,mamba}.
# devdax writes raw bytes into /dev/dax0.0 directly via mmap, so there's
# nothing to prepare on the filesystem side. With USE_DEVDAX=pmemkv set
# (devdax bench), /pmem/ is mounted as raw DAX and is NOT writable as a
# filesystem — skip the cleanup branch there.
if [ "${LABEL}" = "fsdax" ]; then
    PMEM_ROOT="/pmem/${USER}_lmcache_qwen36"
    mpirun ${NQSV_MPIOPTS} -np 4 -npernode 1 --bind-to none \
        bash -c "
            if [ -d '${PMEM_ROOT}' ] && [ -O '${PMEM_ROOT}' ]; then
                rm -rf '${PMEM_ROOT}' 2>/dev/null
            fi
            mkdir -p '${PMEM_ROOT}/attn' '${PMEM_ROOT}/mamba'
            ls -lad '${PMEM_ROOT}' 2>&1
        " 2>&1 || echo "[WARN] could not prepare ${PMEM_ROOT}; relying on existing state"
fi

# Patch workdir into the TOML on a temp copy so the rust harness writes
# under the timestamped result dir.
TMPCFG="${WORKDIR}/config.toml"
sed "s|workdir = \".*\"|workdir = \"${WORKDIR}\"|" "${CFG_TOML}" > "${TMPCFG}"

echo ""
echo "============================================================"
echo " ${LABEL}"
echo " $(date)"
echo "============================================================"

# mpstat per-node CPU time series
echo "Starting mpstat..."
mpirun ${NQSV_MPIOPTS} -np 4 -npernode 1 --bind-to none \
    bash -c "
        HOST=\$(hostname)
        nohup mpstat -P ALL 2 > ${CPU_LOG_DIR}/mpstat_\${HOST}.log 2>&1 &
        echo \$! > ${CPU_LOG_DIR}/mpstat_\${HOST}.pid
    " 2>&1 || echo "mpstat start may have warnings"

# multi-decode runs the actual workload
echo "Running multi-decode..."
mpirun ${NQSV_MPIOPTS} -np 4 -npernode 1 --bind-to none \
    "${BIN}" --config "${TMPCFG}"
RC=$?

echo "Stopping mpstat..."
mpirun ${NQSV_MPIOPTS} -np 4 -npernode 1 --bind-to none \
    bash -c "
        HOST=\$(hostname)
        if [ -f ${CPU_LOG_DIR}/mpstat_\${HOST}.pid ]; then
            kill \$(cat ${CPU_LOG_DIR}/mpstat_\${HOST}.pid) 2>/dev/null || true
        fi
    " 2>&1 || true

# Verify all 4 ranks produced summary.json
N_OK=0
for r in 0 1 2 3; do
    if [ -f "${WORKDIR}/dist_outputs/rank_${r}/summary.rank${r}.json" ]; then
        N_OK=$((N_OK + 1))
    fi
done

echo "${LABEL} done at $(date), rc=${RC}, ranks_ok=${N_OK}/4"

if [ "${N_OK}" -ne 4 ] || [ "${RC}" -ne 0 ]; then
    echo "[FAIL] ${LABEL}: only ${N_OK}/4 ranks succeeded, rc=${RC}"
    exit 1
fi

exit 0
