#!/bin/bash
# run_ttft.sh — TTFT verification for ralph branches × {fsdax,devdax}.
#
# Two phases per condition:
#   Phase A (warm)   : empty cache  → vLLM serves N requests sharing a long
#                      prefix. Both attn KV and Mamba state get written
#                      under local_disk (fsdax) or DevDAX slab (devdax).
#                      Process is killed after the warm pass.
#   Phase B (cold)   : same disk path as Phase A, *fresh* vLLM process.
#                      First request with the same prefix should hit
#                      LMCache (need_to_load > 0, num_cached_tokens > 0)
#                      and report a lower TTFT than Phase A's first call.
#
# Args (env):
#   COND           : fsdax | devdax            (which YAML to use)
#   MODEL          : HF model id or local path (hybrid: Qwen3.5-9B / 3.6-27B)
#   ROOT           : repo root (default = nilpe-lmcache)
#   VLLM_VENV_BIN  : path to vllm venv's bin/activate
#   PROMPT_FILE    : path to a long-prefix prompt (.txt)
#   LMCACHE_LOG    : where to capture LMCache log lines
#
# qsub is intentionally NOT issued here — meant to be invoked from the
# job script (job_ttft_devdax_vs_fsdax.sh) on a PMEM compute node.

set -uo pipefail

COND="${COND:?COND must be fsdax or devdax}"
MODEL="${MODEL:-Qwen/Qwen3.5-9B}"
ROOT="${ROOT:-/work/0/NBB/kogi/workspace/nilpe-lmcache}"
PROMPT_FILE="${PROMPT_FILE:?PROMPT_FILE must point at a long prompt .txt}"
TP_SIZE="${TP_SIZE:-1}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
PORT="${PORT:-8123}"
HOST="${HOST:-127.0.0.1}"

CONFIG="${ROOT}/bench/configs/qwen35_9b_${COND}.yaml"
RESULTS="${ROOT}/bench/results/$(date +%Y%m%d_%H%M%S)_${COND}"
mkdir -p "${RESULTS}"

# --- env baseline ------------------------------------------------------
# PYTHONHASHSEED MUST be fixed across the warm/cold phases for the prefix
# token hash to agree. Forgetting this is the most common reason for
# need_to_load=0 in cold-restart runs (per the ralph patch series notes).
export PYTHONHASHSEED=0
export LMCACHE_CONFIG_FILE="${CONFIG}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export no_proxy=localhost,127.0.0.1
export NO_PROXY=localhost,127.0.0.1

if [ -n "${VLLM_VENV_BIN:-}" ]; then
    # shellcheck disable=SC1090
    source "${VLLM_VENV_BIN}"
fi

cleanup_disk() {
    # Wipe the previously-warmed cache so each Phase-A starts cold.
    if [ "${COND}" = "fsdax" ]; then
        rm -rf /pmem/lmcache_qwen35
        mkdir -p /pmem/lmcache_qwen35/attn /pmem/lmcache_qwen35/mamba
    else
        # DevDAX slab: just bump the in-band epoch by deleting the JSONL
        # sidecar / state_snapshot pkl. The slab itself is reused.
        # _ralph_kv_index.jsonl lives at the local_disk path which is the
        # dax device (a misnomer — devdax backend ignores it). Instead
        # zero the in-process state by removing per-rank pkl/sidecar
        # state if present in /pmem/lmcache_qwen35.
        rm -rf /pmem/lmcache_qwen35
        mkdir -p /pmem/lmcache_qwen35
    fi
}

start_vllm() {
    local logf="$1"
    vllm serve "${MODEL}" \
        --host "${HOST}" --port "${PORT}" \
        --tensor-parallel-size "${TP_SIZE}" \
        --gpu-memory-utilization "${GPU_MEM_UTIL}" \
        --max-model-len "${MAX_MODEL_LEN}" \
        --enable-prefix-caching \
        --no-disable-hybrid-kv-cache-manager \
        --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}' \
        > "${logf}" 2>&1 &
    echo $!
}

wait_ready() {
    local tries=240
    while ! curl -fsS "http://${HOST}:${PORT}/v1/models" >/dev/null 2>&1; do
        tries=$((tries - 1))
        if [ "${tries}" -le 0 ]; then
            echo "[FAIL] vLLM didn't become ready" >&2
            return 1
        fi
        sleep 1
    done
}

stop_vllm() {
    local pid="$1"
    kill -INT "${pid}" 2>/dev/null
    for _ in $(seq 1 60); do
        if ! kill -0 "${pid}" 2>/dev/null; then return 0; fi
        sleep 1
    done
    kill -KILL "${pid}" 2>/dev/null
}

send_prompt() {
    # Returns TTFT (s) on stdout; full response saved to $2.
    local out="$1"
    local prompt
    prompt=$(jq -Rs . < "${PROMPT_FILE}")
    curl -sS -N "http://${HOST}:${PORT}/v1/completions" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"${MODEL}\",
            \"prompt\": ${prompt},
            \"max_tokens\": 32,
            \"temperature\": 0.0,
            \"stream\": true
        }" \
        > "${out}" 2>&1
}

# --- Phase A: warm -----------------------------------------------------
cleanup_disk
PHASE_A_LOG="${RESULTS}/phaseA_vllm.log"
PHASE_A_RES="${RESULTS}/phaseA_response.txt"
echo "=== Phase A (warm) ${COND} $(date) ==="
PID=$(start_vllm "${PHASE_A_LOG}")
wait_ready || { stop_vllm "${PID}"; exit 1; }
T0=$(date +%s.%N); send_prompt "${PHASE_A_RES}"; T1=$(date +%s.%N)
echo "phaseA_e2e_seconds: $(echo "${T1} - ${T0}" | bc)" | tee -a "${RESULTS}/summary.txt"
stop_vllm "${PID}"

# --- Phase B: cold restart ---------------------------------------------
PHASE_B_LOG="${RESULTS}/phaseB_vllm.log"
PHASE_B_RES="${RESULTS}/phaseB_response.txt"
echo "=== Phase B (cold-restart) ${COND} $(date) ==="
PID=$(start_vllm "${PHASE_B_LOG}")
wait_ready || { stop_vllm "${PID}"; exit 1; }
T0=$(date +%s.%N); send_prompt "${PHASE_B_RES}"; T1=$(date +%s.%N)
echo "phaseB_e2e_seconds: $(echo "${T1} - ${T0}" | bc)" | tee -a "${RESULTS}/summary.txt"
stop_vllm "${PID}"

# --- Cache-hit signals from LMCache log -------------------------------
echo "=== Cache-hit signals ===" | tee -a "${RESULTS}/summary.txt"
grep -E "num_cached_tokens|need to load|hybrid_mamba_state_io" "${PHASE_B_LOG}" \
    | tee -a "${RESULTS}/summary.txt" || true

echo "=== Done ${COND} $(date) ==="
