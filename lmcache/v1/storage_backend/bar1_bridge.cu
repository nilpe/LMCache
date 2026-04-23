/*
 * bar1_bridge.cu - GDRCopy BAR1 bridge for LMCache
 *
 * GPU staging buffer を BAR1 マップし、CPU から直接書き込み可能にする。
 * PMEM → GPU を DRAM バイパスで実現。
 *
 * Build:
 *   nvcc -O2 -arch=sm_90 -Xcompiler="-mavx512f -fPIC -lpthread" \
 *        -shared -I/usr/local/include -L/usr/local/lib \
 *        -lgdrapi -lcuda -o bar1_bridge.so bar1_bridge.cu
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <gdrapi.h>
#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

#define GPU_PAGE (64UL * 1024)

/* ============================================================
 * Global state (one staging buffer per process)
 * ============================================================ */
static gdr_t g_gdr = NULL;
static gdr_mh_t g_mh;
static CUdeviceptr g_gpu_buf = 0;
static void *g_bar1_ptr = NULL;   /* BAR1 WC-mapped CPU pointer */
static void *g_bar1_base = NULL;  /* adjusted for offset */
static size_t g_buf_size = 0;
static int g_initialized = 0;

/* ============================================================
 * AVX-512 NT store copy (PMEM → BAR1)
 * ============================================================ */
__attribute__((target("avx512f")))
static void avx512_nt_copy(void *dst, const void *src, size_t n) {
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

/* ============================================================
 * Work-stealing parallel copy
 * ============================================================ */
typedef struct {
    const void *src;
    void *dst;
    size_t size;
} copy_task_t;

typedef struct {
    copy_task_t *tasks;
    volatile int *next_task;
    int count;
} ws_arg_t;

static void *ws_worker(void *arg) {
    ws_arg_t *ws = (ws_arg_t *)arg;
    while (1) {
        int idx = __sync_fetch_and_add(ws->next_task, 1);
        if (idx >= ws->count) break;
        avx512_nt_copy(ws->tasks[idx].dst, ws->tasks[idx].src, ws->tasks[idx].size);
    }
    return NULL;
}

/* ============================================================
 * Public API (extern "C" for ctypes)
 * ============================================================ */
extern "C" {

/*
 * bar1_init: GPU staging buffer を確保し BAR1 マップ
 * Returns 0 on success, -1 on error
 */
int bar1_init(size_t size) {
    if (g_initialized) {
        if (size <= g_buf_size) return 0;  /* already big enough */
        /* Need to grow - cleanup first */
        gdr_unmap(g_gdr, g_mh, g_bar1_ptr, g_buf_size);
        gdr_unpin_buffer(g_gdr, g_mh);
        cuMemFree(g_gpu_buf);
        g_initialized = 0;
    }

    /* Align to GPU page */
    size = (size + GPU_PAGE - 1) & ~(GPU_PAGE - 1);

    /* Open GDRCopy */
    if (!g_gdr) {
        g_gdr = gdr_open();
        if (!g_gdr) {
            fprintf(stderr, "bar1_bridge: gdr_open failed\n");
            return -1;
        }
    }

    /* Ensure we have a CUDA context (PyTorch may have created one) */
    CUcontext ctx;
    CUresult cerr = cuCtxGetCurrent(&ctx);
    if (cerr != CUDA_SUCCESS || ctx == NULL) {
        /* No current context; attach to device 0's primary context */
        CUdevice dev;
        cuInit(0);
        cuDeviceGet(&dev, 0);
        cuDevicePrimaryCtxRetain(&ctx, dev);
        cuCtxSetCurrent(ctx);
        fprintf(stderr, "bar1_bridge: attached to primary context\n");
    }

    /* Allocate GPU memory */
    CUresult err = cuMemAlloc(&g_gpu_buf, size);
    if (err != CUDA_SUCCESS) {
        const char *s;
        cuGetErrorString(err, &s);
        fprintf(stderr, "bar1_bridge: cuMemAlloc(%zu) failed: %s\n", size, s);
        return -1;
    }

    /* Pin and map to BAR1 */
    int ret = gdr_pin_buffer(g_gdr, g_gpu_buf, size, 0, 0, &g_mh);
    if (ret) {
        fprintf(stderr, "bar1_bridge: gdr_pin_buffer failed: %d\n", ret);
        cuMemFree(g_gpu_buf);
        return -1;
    }

    ret = gdr_map(g_gdr, g_mh, &g_bar1_ptr, size);
    if (ret) {
        fprintf(stderr, "bar1_bridge: gdr_map failed: %d\n", ret);
        gdr_unpin_buffer(g_gdr, g_mh);
        cuMemFree(g_gpu_buf);
        return -1;
    }

    /* Adjust for offset within page */
    gdr_info_t info;
    gdr_get_info(g_gdr, g_mh, &info);
    size_t off = g_gpu_buf - info.va;
    g_bar1_base = (char *)g_bar1_ptr + off;

    g_buf_size = size;
    g_initialized = 1;

    fprintf(stderr, "bar1_bridge: init OK, gpu=0x%lx, bar1=%p, size=%zu MB\n",
            (unsigned long)g_gpu_buf, g_bar1_base, size >> 20);
    return 0;
}

/*
 * bar1_init_external: 外部 (PyTorch) で確保した GPU ポインタを BAR1 マップ
 * D2D memcpy なしで PyTorch tensor を直接 BAR1 経由で書き込める。
 */
int bar1_init_external(unsigned long long gpu_ptr, size_t size) {
    if (g_initialized) {
        gdr_unmap(g_gdr, g_mh, g_bar1_ptr, g_buf_size);
        gdr_unpin_buffer(g_gdr, g_mh);
        /* Don't free - we don't own this memory */
        g_initialized = 0;
    }

    size = (size + GPU_PAGE - 1) & ~(GPU_PAGE - 1);

    if (!g_gdr) {
        g_gdr = gdr_open();
        if (!g_gdr) {
            fprintf(stderr, "bar1_bridge: gdr_open failed\n");
            return -1;
        }
    }

    g_gpu_buf = (CUdeviceptr)gpu_ptr;

    int ret = gdr_pin_buffer(g_gdr, g_gpu_buf, size, 0, 0, &g_mh);
    if (ret) {
        fprintf(stderr, "bar1_bridge: gdr_pin_buffer(ext 0x%llx, %zu) failed: %d\n",
                gpu_ptr, size, ret);
        return -1;
    }

    ret = gdr_map(g_gdr, g_mh, &g_bar1_ptr, size);
    if (ret) {
        fprintf(stderr, "bar1_bridge: gdr_map(ext) failed: %d\n", ret);
        gdr_unpin_buffer(g_gdr, g_mh);
        return -1;
    }

    gdr_info_t info;
    gdr_get_info(g_gdr, g_mh, &info);
    size_t off = g_gpu_buf - info.va;
    g_bar1_base = (char *)g_bar1_ptr + off;

    g_buf_size = size;
    g_initialized = 1;

    fprintf(stderr, "bar1_bridge: init_external OK, gpu=0x%llx, bar1=%p, size=%zu MB\n",
            gpu_ptr, g_bar1_base, size >> 20);
    return 0;
}

/*
 * bar1_get_gpu_ptr: GPU buffer のデバイスポインタを返す
 * Python から torch.tensor のストレージとして使える
 */
unsigned long long bar1_get_gpu_ptr(void) {
    return (unsigned long long)g_gpu_buf;
}

/*
 * bar1_get_bar1_ptr: BAR1 の CPU ポインタを返す
 */
unsigned long long bar1_get_bar1_ptr(void) {
    return (unsigned long long)g_bar1_base;
}

size_t bar1_get_size(void) {
    return g_buf_size;
}

/*
 * bar1_copy_scatter: PMEM → BAR1 scatter copy (LMCache pattern)
 * Fine-grained task stealing: 各チャンクを SUB_TASK_SIZE に分割し、
 * 遅いスレッド (NUMA ペナルティ等) の影響を軽減。
 */
#define SUB_TASK_SIZE (1UL * 1024 * 1024)  /* 1MB sub-task granularity */

typedef struct {
    const char *src_base;
    char *dst_base;
    size_t chunk_size;
    int chunk_idx;
    size_t sub_offset;
    size_t sub_size;
} fine_task_t;

typedef struct {
    fine_task_t *tasks;
    volatile int *next_task;
    int count;
} fine_ws_arg_t;

static void *fine_ws_worker(void *arg) {
    fine_ws_arg_t *ws = (fine_ws_arg_t *)arg;
    while (1) {
        int idx = __sync_fetch_and_add(ws->next_task, 1);
        if (idx >= ws->count) break;
        fine_task_t *t = &ws->tasks[idx];
        avx512_nt_copy(t->dst_base + t->sub_offset,
                       t->src_base + t->sub_offset,
                       t->sub_size);
    }
    return NULL;
}

int bar1_copy_scatter(
    const void *pmem_base,
    const long long *pmem_offsets,
    const long long *gpu_offsets,
    const long long *sizes,
    int count,
    int max_threads
) {
    if (!g_initialized) return -1;

    /* Build fine-grained task list */
    int total_tasks = 0;
    for (int i = 0; i < count; i++)
        total_tasks += ((size_t)sizes[i] + SUB_TASK_SIZE - 1) / SUB_TASK_SIZE;

    fine_task_t *tasks = (fine_task_t *)malloc(total_tasks * sizeof(fine_task_t));
    if (!tasks) return -1;

    int ti = 0;
    for (int i = 0; i < count; i++) {
        const char *src = (const char *)pmem_base + pmem_offsets[i];
        char *dst = (char *)g_bar1_base + gpu_offsets[i];
        size_t remaining = (size_t)sizes[i];
        size_t off = 0;
        while (remaining > 0) {
            size_t sz = remaining < SUB_TASK_SIZE ? remaining : SUB_TASK_SIZE;
            tasks[ti].src_base = src;
            tasks[ti].dst_base = dst;
            tasks[ti].chunk_size = (size_t)sizes[i];
            tasks[ti].chunk_idx = i;
            tasks[ti].sub_offset = off;
            tasks[ti].sub_size = sz;
            ti++;
            off += sz;
            remaining -= sz;
        }
    }

    int nt = max_threads;
    if (nt > total_tasks) nt = total_tasks;

    pthread_t *threads = (pthread_t *)malloc(nt * sizeof(pthread_t));
    volatile int next = 0;
    fine_ws_arg_t ws = { .tasks = tasks, .next_task = &next, .count = total_tasks };

    for (int i = 0; i < nt; i++)
        pthread_create(&threads[i], NULL, fine_ws_worker, &ws);
    for (int i = 0; i < nt; i++)
        pthread_join(threads[i], NULL);

    free(threads);
    free(tasks);
    return 0;
}

/*
 * bar1_copy_scatter_coarse: 元の粗い粒度版 (比較用)
 */
int bar1_copy_scatter_coarse(
    const void *pmem_base,
    const long long *pmem_offsets,
    const long long *gpu_offsets,
    const long long *sizes,
    int count,
    int max_threads
) {
    if (!g_initialized) return -1;

    copy_task_t *tasks = (copy_task_t *)malloc(count * sizeof(copy_task_t));
    if (!tasks) return -1;

    for (int i = 0; i < count; i++) {
        tasks[i].src = (const char *)pmem_base + pmem_offsets[i];
        tasks[i].dst = (char *)g_bar1_base + gpu_offsets[i];
        tasks[i].size = (size_t)sizes[i];
    }

    int nt = max_threads;
    if (nt > count) nt = count;

    pthread_t *threads = (pthread_t *)malloc(nt * sizeof(pthread_t));
    volatile int next = 0;
    ws_arg_t ws = { .tasks = tasks, .next_task = &next, .count = count };

    for (int i = 0; i < nt; i++)
        pthread_create(&threads[i], NULL, ws_worker, &ws);
    for (int i = 0; i < nt; i++)
        pthread_join(threads[i], NULL);

    free(threads);
    free(tasks);
    return 0;
}

/* ============================================================
 * Per-thread H2D mode (Approach A): each C thread does
 *   PMEM → pinned DRAM (AVX-512 NT) → cudaMemcpyAsync → GPU staging
 * Multiple CUDA streams allow parallel H2D transfers.
 * ============================================================ */

#define MAX_H2D_STREAMS 64
static cudaStream_t g_h2d_streams[MAX_H2D_STREAMS];
static int g_h2d_num_streams = 0;
static void *g_h2d_pinned_dram = NULL;  /* per-chunk bounce buffers (concatenated) */
static size_t g_h2d_pinned_size = 0;

int h2d_init(size_t buf_size, int num_streams) {
    if (num_streams > MAX_H2D_STREAMS) num_streams = MAX_H2D_STREAMS;

    /* Ensure CUDA context */
    CUcontext ctx;
    if (cuCtxGetCurrent(&ctx) != CUDA_SUCCESS || ctx == NULL) {
        CUdevice dev;
        cuInit(0);
        cuDeviceGet(&dev, 0);
        cuDevicePrimaryCtxRetain(&ctx, dev);
        cuCtxSetCurrent(ctx);
    }

    /* Free old streams if any */
    for (int i = 0; i < g_h2d_num_streams; i++) {
        cudaStreamDestroy(g_h2d_streams[i]);
    }
    if (g_h2d_pinned_dram) {
        cudaFreeHost(g_h2d_pinned_dram);
        g_h2d_pinned_dram = NULL;
    }

    g_h2d_num_streams = num_streams;
    for (int i = 0; i < num_streams; i++) {
        cudaError_t e = cudaStreamCreate(&g_h2d_streams[i]);
        if (e != cudaSuccess) {
            fprintf(stderr, "h2d_init: cudaStreamCreate failed: %s\n", cudaGetErrorString(e));
            return -1;
        }
    }

    /* Pinned DRAM bounce buffer */
    cudaError_t e = cudaHostAlloc(&g_h2d_pinned_dram, buf_size, cudaHostAllocDefault);
    if (e != cudaSuccess) {
        fprintf(stderr, "h2d_init: cudaHostAlloc(%zu) failed: %s\n", buf_size, cudaGetErrorString(e));
        return -1;
    }
    g_h2d_pinned_size = buf_size;

    fprintf(stderr, "h2d_init: %d streams, %zu MB pinned DRAM\n", num_streams, buf_size >> 20);
    return 0;
}

typedef struct {
    const void *pmem_base;
    void *gpu_dst_base;       /* GPU staging buffer device pointer */
    const long long *pmem_offsets;
    const long long *gpu_offsets;
    const long long *sizes;
    volatile int *next_task;
    int count;
    int thread_id;            /* used to pick stream */
    CUcontext ctx;            /* shared CUDA context for this worker */
} h2d_arg_t;

static void *h2d_worker(void *arg) {
    h2d_arg_t *a = (h2d_arg_t *)arg;
    /* Attach this pthread to the CUDA context so cudaMemcpyAsync can use it */
    if (a->ctx) cuCtxSetCurrent(a->ctx);
    cudaSetDevice(0);
    cudaStream_t stream = g_h2d_streams[a->thread_id % g_h2d_num_streams];

    while (1) {
        int idx = __sync_fetch_and_add(a->next_task, 1);
        if (idx >= a->count) break;

        const char *src = (const char *)a->pmem_base + a->pmem_offsets[idx];
        size_t off = (size_t)a->gpu_offsets[idx];
        size_t sz = (size_t)a->sizes[idx];
        char *dram = (char *)g_h2d_pinned_dram + off;
        char *gpu  = (char *)a->gpu_dst_base + off;

        /* PMEM → pinned DRAM (AVX-512 NT store) */
        avx512_nt_copy(dram, src, sz);

        /* DRAM → GPU async */
        cudaMemcpyAsync(gpu, dram, sz, cudaMemcpyHostToDevice, stream);
    }
    return NULL;
}

/*
 * parallel_h2d_chunked: per-thread PMEM → DRAM → GPU
 *
 * gpu_dst_base: pre-allocated GPU staging buffer (cudaMalloc'd)
 */
int parallel_h2d_chunked(
    const void *pmem_base,
    void *gpu_dst_base,
    const long long *pmem_offsets,
    const long long *gpu_offsets,
    const long long *sizes,
    int count,
    int max_threads
) {
    if (g_h2d_num_streams == 0 || g_h2d_pinned_dram == NULL) {
        fprintf(stderr, "parallel_h2d_chunked: h2d_init not called\n");
        return -1;
    }

    int nt = max_threads;
    if (nt > count) nt = count;

    pthread_t *threads = (pthread_t *)malloc(nt * sizeof(pthread_t));
    h2d_arg_t *args = (h2d_arg_t *)malloc(nt * sizeof(h2d_arg_t));
    volatile int next = 0;

    /* Capture current CUDA context from main thread */
    CUcontext ctx = NULL;
    cuCtxGetCurrent(&ctx);

    for (int i = 0; i < nt; i++) {
        args[i].pmem_base = pmem_base;
        args[i].gpu_dst_base = gpu_dst_base;
        args[i].pmem_offsets = pmem_offsets;
        args[i].gpu_offsets = gpu_offsets;
        args[i].sizes = sizes;
        args[i].next_task = &next;
        args[i].count = count;
        args[i].thread_id = i;
        args[i].ctx = ctx;
        pthread_create(&threads[i], NULL, h2d_worker, &args[i]);
    }
    for (int i = 0; i < nt; i++)
        pthread_join(threads[i], NULL);

    /* Sync all streams to ensure GPU has all data */
    for (int i = 0; i < g_h2d_num_streams; i++)
        cudaStreamSynchronize(g_h2d_streams[i]);

    free(threads);
    free(args);
    return 0;
}

void h2d_cleanup(void) {
    for (int i = 0; i < g_h2d_num_streams; i++)
        cudaStreamDestroy(g_h2d_streams[i]);
    g_h2d_num_streams = 0;
    if (g_h2d_pinned_dram) {
        cudaFreeHost(g_h2d_pinned_dram);
        g_h2d_pinned_dram = NULL;
    }
}

/* ============================================================
 * GPU DMA direct mode: PMEM mmap → cudaHostRegister → cudaMemcpyAsync
 * GPU copy engine pulls directly from PMEM; no CPU staging.
 * ============================================================ */

#define MAX_GPU_DMA_STREAMS 64
static cudaStream_t g_gpu_dma_streams[MAX_GPU_DMA_STREAMS];
static int g_gpu_dma_num_streams = 0;
static void *g_gpu_dma_pmem_host = NULL;  /* registered PMEM host ptr */
static void *g_gpu_dma_pmem_dev = NULL;   /* PMEM device ptr (unified addressing) */
static size_t g_gpu_dma_pmem_size = 0;

/*
 * gpu_dma_init: register PMEM mmap region with cudaHostRegister so GPU can pull
 *   pmem_base:  PMEM mmap address (must be 2MB aligned for PMD)
 *   pmem_size:  size to register (must fit in BAR; <~123 GiB observed on H100)
 *   num_streams: CUDA streams (parallel cudaMemcpyAsync)
 */
int gpu_dma_init(void *pmem_base, size_t pmem_size, int num_streams) {
    if (num_streams > MAX_GPU_DMA_STREAMS) num_streams = MAX_GPU_DMA_STREAMS;

    /* Ensure CUDA context */
    CUcontext ctx;
    if (cuCtxGetCurrent(&ctx) != CUDA_SUCCESS || ctx == NULL) {
        CUdevice dev;
        cuInit(0);
        cuDeviceGet(&dev, 0);
        cuDevicePrimaryCtxRetain(&ctx, dev);
        cuCtxSetCurrent(ctx);
    }

    /* Free old state if re-init */
    for (int i = 0; i < g_gpu_dma_num_streams; i++) {
        cudaStreamDestroy(g_gpu_dma_streams[i]);
    }
    g_gpu_dma_num_streams = 0;
    if (g_gpu_dma_pmem_host) {
        cudaHostUnregister(g_gpu_dma_pmem_host);
        g_gpu_dma_pmem_host = NULL;
        g_gpu_dma_pmem_dev = NULL;
        g_gpu_dma_pmem_size = 0;
    }

    /* Register PMEM mmap (cudaHostRegisterPortable=1 only; ReadOnly=4 fails on DevDAX) */
    cudaError_t e = cudaHostRegister(pmem_base, pmem_size, cudaHostRegisterPortable);
    if (e != cudaSuccess) {
        fprintf(stderr, "gpu_dma_init: cudaHostRegister(%zu MB) failed: %s\n",
                pmem_size >> 20, cudaGetErrorString(e));
        return -1;
    }
    e = cudaHostGetDevicePointer(&g_gpu_dma_pmem_dev, pmem_base, 0);
    if (e != cudaSuccess) {
        fprintf(stderr, "gpu_dma_init: cudaHostGetDevicePointer failed: %s\n",
                cudaGetErrorString(e));
        cudaHostUnregister(pmem_base);
        return -1;
    }
    g_gpu_dma_pmem_host = pmem_base;
    g_gpu_dma_pmem_size = pmem_size;

    /* Create streams */
    for (int i = 0; i < num_streams; i++) {
        cudaError_t se = cudaStreamCreate(&g_gpu_dma_streams[i]);
        if (se != cudaSuccess) {
            fprintf(stderr, "gpu_dma_init: cudaStreamCreate failed: %s\n",
                    cudaGetErrorString(se));
            return -1;
        }
    }
    g_gpu_dma_num_streams = num_streams;

    fprintf(stderr, "gpu_dma_init: registered %zu MB PMEM (dev=%p), %d streams\n",
            pmem_size >> 20, g_gpu_dma_pmem_dev, num_streams);
    return 0;
}

typedef struct {
    void *gpu_dst_base;               /* GPU staging buffer device ptr */
    const long long *pmem_offsets;    /* offsets into registered PMEM */
    const long long *gpu_offsets;     /* offsets into gpu_dst_base */
    const long long *sizes;
    volatile int *next_task;
    int count;
    int thread_id;
    CUcontext ctx;
} gpu_dma_arg_t;

static void *gpu_dma_worker(void *arg) {
    gpu_dma_arg_t *a = (gpu_dma_arg_t *)arg;
    if (a->ctx) cuCtxSetCurrent(a->ctx);
    cudaSetDevice(0);
    cudaStream_t stream = g_gpu_dma_streams[a->thread_id % g_gpu_dma_num_streams];

    while (1) {
        int idx = __sync_fetch_and_add(a->next_task, 1);
        if (idx >= a->count) break;
        const char *src = (const char *)g_gpu_dma_pmem_dev + a->pmem_offsets[idx];
        char *dst = (char *)a->gpu_dst_base + a->gpu_offsets[idx];
        size_t sz = (size_t)a->sizes[idx];
        cudaMemcpyAsync(dst, src, sz, cudaMemcpyHostToDevice, stream);
    }
    return NULL;
}

/*
 * parallel_gpu_dma_chunked: per-thread cudaMemcpyAsync from PMEM to GPU staging
 *   Multiple CUDA streams let GPU copy-engine parallelize PMEM pulls.
 *   pmem_base arg is ignored (global from gpu_dma_init is used) — kept for API parity.
 */
int parallel_gpu_dma_chunked(
    const void *pmem_base,   /* unused — uses globally-registered region */
    void *gpu_dst_base,
    const long long *pmem_offsets,
    const long long *gpu_offsets,
    const long long *sizes,
    int count,
    int max_threads
) {
    (void)pmem_base;
    if (g_gpu_dma_num_streams == 0 || g_gpu_dma_pmem_dev == NULL) {
        fprintf(stderr, "parallel_gpu_dma_chunked: gpu_dma_init not called\n");
        return -1;
    }

    int nt = max_threads;
    if (nt > count) nt = count;
    if (nt < 1) nt = 1;

    pthread_t *threads = (pthread_t *)malloc(nt * sizeof(pthread_t));
    gpu_dma_arg_t *args = (gpu_dma_arg_t *)malloc(nt * sizeof(gpu_dma_arg_t));
    volatile int next = 0;

    CUcontext ctx = NULL;
    cuCtxGetCurrent(&ctx);

    for (int i = 0; i < nt; i++) {
        args[i].gpu_dst_base = gpu_dst_base;
        args[i].pmem_offsets = pmem_offsets;
        args[i].gpu_offsets = gpu_offsets;
        args[i].sizes = sizes;
        args[i].next_task = &next;
        args[i].count = count;
        args[i].thread_id = i;
        args[i].ctx = ctx;
        pthread_create(&threads[i], NULL, gpu_dma_worker, &args[i]);
    }
    for (int i = 0; i < nt; i++)
        pthread_join(threads[i], NULL);

    for (int i = 0; i < g_gpu_dma_num_streams; i++)
        cudaStreamSynchronize(g_gpu_dma_streams[i]);

    free(threads);
    free(args);
    return 0;
}

void gpu_dma_cleanup(void) {
    for (int i = 0; i < g_gpu_dma_num_streams; i++)
        cudaStreamDestroy(g_gpu_dma_streams[i]);
    g_gpu_dma_num_streams = 0;
    if (g_gpu_dma_pmem_host) {
        cudaHostUnregister(g_gpu_dma_pmem_host);
        g_gpu_dma_pmem_host = NULL;
        g_gpu_dma_pmem_dev = NULL;
        g_gpu_dma_pmem_size = 0;
    }
}

/*
 * bar1_cleanup: リソース解放
 */
void bar1_cleanup(void) {
    if (g_initialized) {
        gdr_unmap(g_gdr, g_mh, g_bar1_ptr, g_buf_size);
        gdr_unpin_buffer(g_gdr, g_mh);
        cuMemFree(g_gpu_buf);
        g_initialized = 0;
    }
    if (g_gdr) {
        gdr_close(g_gdr);
        g_gdr = NULL;
    }
}

}  /* extern "C" */
