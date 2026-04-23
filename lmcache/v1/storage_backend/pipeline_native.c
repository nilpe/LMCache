/*
 * pipeline_native.c - Python ループを排除したネイティブ pipeline
 *
 * 全チャンクの PMEM → DRAM コピーを C 側で管理し、
 * サブバッチ完了時に Python コールバックを呼び出す。
 *
 * Python 側: 1回の C 呼び出しで pipeline 全体が完了。
 * ループ回数分の Python↔C 遷移を排除。
 *
 * gcc -O2 -march=native -mavx512f -shared -fPIC -o pipeline_native.so pipeline_native.c -lpthread
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <immintrin.h>
#include <pthread.h>

/* Profiling (same pattern as bar1_bridge.cu). Gated by LMCACHE_PROFILE=1. */
static int g_profile_enabled = -1;
static inline int profile_enabled(void) {
    if (__builtin_expect(g_profile_enabled < 0, 0)) {
        const char *e = getenv("LMCACHE_PROFILE");
        g_profile_enabled = (e && e[0] && e[0] != '0') ? 1 : 0;
    }
    return g_profile_enabled;
}
static inline double prof_now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

/* Import memcpy variants (same as fast_read.c) */
static void avx512_memcpy_nt(void *dst, const void *src, size_t n) {
    char *d = (char *)dst;
    const char *s = (const char *)src;
    size_t i;
    for (i = 0; i + 256 <= n; i += 256) {
        __m512i z0 = _mm512_load_si512((__m512i *)(s + i));
        __m512i z1 = _mm512_load_si512((__m512i *)(s + i + 64));
        __m512i z2 = _mm512_load_si512((__m512i *)(s + i + 128));
        __m512i z3 = _mm512_load_si512((__m512i *)(s + i + 192));
        _mm512_stream_si512((__m512i *)(d + i), z0);
        _mm512_stream_si512((__m512i *)(d + i + 64), z1);
        _mm512_stream_si512((__m512i *)(d + i + 128), z2);
        _mm512_stream_si512((__m512i *)(d + i + 192), z3);
    }
    _mm_sfence();
    if (i < n) memcpy(d + i, s + i, n - i);
}

static void avx512_memcpy_temporal(void *dst, const void *src, size_t n) {
    char *d = (char *)dst;
    const char *s = (const char *)src;
    size_t i;
    for (i = 0; i + 256 <= n; i += 256) {
        __m512i z0 = _mm512_load_si512((__m512i *)(s + i));
        __m512i z1 = _mm512_load_si512((__m512i *)(s + i + 64));
        __m512i z2 = _mm512_load_si512((__m512i *)(s + i + 128));
        __m512i z3 = _mm512_load_si512((__m512i *)(s + i + 192));
        _mm512_store_si512((__m512i *)(d + i), z0);
        _mm512_store_si512((__m512i *)(d + i + 64), z1);
        _mm512_store_si512((__m512i *)(d + i + 128), z2);
        _mm512_store_si512((__m512i *)(d + i + 192), z3);
    }
    if (i < n) memcpy(d + i, s + i, n - i);
}

static inline void do_memcpy(void *dst, const void *src, size_t size, int mode) {
    switch (mode) {
    case 0: memcpy(dst, src, size); break;
    case 1: avx512_memcpy_temporal(dst, src, size); break;
    case 2: avx512_memcpy_nt(dst, src, size); break;
    default: memcpy(dst, src, size); break;
    }
}

/* ============================================================
 * Work-stealing sub-batch copy
 * ============================================================ */

typedef struct {
    const void *base;       /* DevDAX mmap base */
    const int64_t *offsets;
    void **dsts;
    const int64_t *sizes;
    volatile int *next_task;
    int start;              /* first task index in this sub-batch */
    int end;                /* one past last task index */
    int mode;
} subbatch_ws_t;

static void *_subbatch_worker(void *arg) {
    subbatch_ws_t *ws = (subbatch_ws_t *)arg;
    while (1) {
        int idx = __sync_fetch_and_add(ws->next_task, 1);
        if (idx >= ws->end) break;
        const void *src = (const char *)ws->base + ws->offsets[idx];
        do_memcpy(ws->dsts[idx], src, (size_t)ws->sizes[idx], ws->mode);
    }
    return NULL;
}

/* Copy one sub-batch using work-stealing threads */
static void copy_subbatch(
    const void *base,
    const int64_t *offsets,
    void **dsts,
    const int64_t *sizes,
    int start, int end,
    int max_threads, int mode,
    pthread_t *threads  /* pre-allocated thread array */
) {
    int count = end - start;
    int nt = max_threads < count ? max_threads : count;

    volatile int next_task = start;
    subbatch_ws_t ws = {
        .base = base, .offsets = offsets, .dsts = dsts, .sizes = sizes,
        .next_task = &next_task, .start = start, .end = end, .mode = mode
    };

    for (int i = 0; i < nt; i++)
        pthread_create(&threads[i], NULL, _subbatch_worker, &ws);
    for (int i = 0; i < nt; i++)
        pthread_join(threads[i], NULL);
}

/*
 * pipeline_copy_with_callback:
 *   全チャンクをサブバッチに分割し、C 側で順次コピー。
 *   各サブバッチ完了時に Python コールバックを呼ぶ。
 *
 *   Python 側は 1 回の C 呼び出しで済む。
 *   コールバックは GIL を取得して GPU transfer を発行する。
 *
 * callback(batch_idx, start, end):
 *   start..end のチャンクの DRAM コピーが完了したことを通知。
 *   Python 側で batched_to_gpu を呼ぶ。
 *
 * Returns: 0 on success
 */
typedef void (*batch_callback_fn)(int batch_idx, int start, int end, void *user_data);

int pipeline_copy_with_callback(
    const void *base,
    const int64_t *offsets,
    void **dsts,
    const int64_t *sizes,
    int count,
    int batch_size,
    int max_threads,
    int mode,
    batch_callback_fn callback,
    void *user_data
) {
    pthread_t *threads = (pthread_t *)malloc(max_threads * sizeof(pthread_t));
    if (!threads) return -1;

    int batch_idx = 0;
    for (int start = 0; start < count; start += batch_size) {
        int end = start + batch_size;
        if (end > count) end = count;

        /* Copy this sub-batch (blocking, uses all threads) */
        copy_subbatch(base, offsets, dsts, sizes,
                      start, end, max_threads, mode, threads);

        /* Notify Python: this sub-batch is ready for GPU transfer */
        if (callback)
            callback(batch_idx, start, end, user_data);

        batch_idx++;
    }

    free(threads);
    return 0;
}

/*
 * pipeline_double_buffer:
 *   真のダブルバッファ: disk read[i+1] と callback[i] を並行実行。
 *
 *   Thread 0: callback (GPU transfer) を呼ぶ
 *   Threads 1..N: 次のサブバッチの disk read
 *
 *   Returns: 0 on success
 */

typedef struct {
    batch_callback_fn callback;
    void *user_data;
    int batch_idx;
    int start;
    int end;
} gpu_cb_arg_t;

static void *_gpu_callback_thread(void *arg) {
    gpu_cb_arg_t *cb = (gpu_cb_arg_t *)arg;
    cb->callback(cb->batch_idx, cb->start, cb->end, cb->user_data);
    return NULL;
}

int pipeline_double_buffer(
    const void *base,
    const int64_t *offsets,
    void **dsts,
    const int64_t *sizes,
    int count,
    int batch_size,
    int max_threads,
    int mode,
    batch_callback_fn callback,
    void *user_data
) {
    pthread_t *threads = (pthread_t *)malloc(max_threads * sizeof(pthread_t));
    if (!threads) return -1;

    pthread_t gpu_thread;
    gpu_cb_arg_t gpu_arg;
    int gpu_running = 0;

    int profile = profile_enabled();
    double p_entry = profile ? prof_now_ms() : 0;
    double acc_disk_ms = 0, acc_wait_prev_ms = 0, acc_cb_issue_ms = 0;
    int n_batches = 0;
    size_t total_bytes = 0;

    int batch_idx = 0;
    for (int start = 0; start < count; start += batch_size) {
        int end = start + batch_size;
        if (end > count) end = count;

        double t0 = profile ? prof_now_ms() : 0;

        /* Copy this sub-batch */
        copy_subbatch(base, offsets, dsts, sizes,
                      start, end, max_threads, mode, threads);

        double t1 = profile ? prof_now_ms() : 0;

        /* Wait for previous GPU transfer to complete */
        if (gpu_running) {
            pthread_join(gpu_thread, NULL);
            gpu_running = 0;
        }

        double t2 = profile ? prof_now_ms() : 0;

        /* Start GPU transfer for this sub-batch in background */
        if (callback) {
            gpu_arg = (gpu_cb_arg_t){
                .callback = callback, .user_data = user_data,
                .batch_idx = batch_idx, .start = start, .end = end
            };
            pthread_create(&gpu_thread, NULL, _gpu_callback_thread, &gpu_arg);
            gpu_running = 1;
        }

        double t3 = profile ? prof_now_ms() : 0;

        if (profile) {
            acc_disk_ms += t1 - t0;
            acc_wait_prev_ms += t2 - t1;
            acc_cb_issue_ms += t3 - t2;
            n_batches++;
            for (int j = start; j < end; j++) total_bytes += (size_t)sizes[j];
        }

        batch_idx++;
    }

    /* Wait for last GPU transfer */
    double t_final_wait_start = profile ? prof_now_ms() : 0;
    if (gpu_running)
        pthread_join(gpu_thread, NULL);
    double t_final_wait_end = profile ? prof_now_ms() : 0;

    if (profile) {
        double total_ms = t_final_wait_end - p_entry;
        double total_gb = (double)total_bytes / (1024.0*1024.0*1024.0);
        double agg_bw = total_ms > 0 ? total_gb / (total_ms / 1000.0) : 0;
        fprintf(stderr,
                "[PROF pipeline_double_buffer] n=%d nt=%d batch=%d n_sub=%d | "
                "disk_total=%.2fms wait_prev_total=%.2fms cb_issue_total=%.2fms "
                "final_wait=%.2fms walltime=%.2fms | %.2f GB @ %.2f GB/s\n",
                count, max_threads, batch_size, n_batches,
                acc_disk_ms, acc_wait_prev_ms, acc_cb_issue_ms,
                t_final_wait_end - t_final_wait_start,
                total_ms, total_gb, agg_bw);
    }

    free(threads);
    return 0;
}
