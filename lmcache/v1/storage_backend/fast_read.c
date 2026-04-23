/*
 * fast_read.c - AVX-512 高速ファイル読み出し C extension
 *
 * Python の file I/O を完全にバイパスし、
 * open → mmap(MAP_POPULATE) → AVX-512 memcpy → munmap → close を C で直接実行。
 * または open → read → close を C で直接実行。
 *
 * gcc -O2 -march=native -mavx512f -shared -fPIC -o fast_read.so fast_read.c
 *
 * Python から ctypes で呼び出す:
 *   lib = ctypes.CDLL("./fast_read.so")
 *   lib.fast_read_file(path_bytes, dst_ptr, size)
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <immintrin.h>

/* AVX-512 memcpy: 256B unrolled (XPLine aligned) */
static void avx512_memcpy(void *dst, const void *src, size_t n) {
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
    /* remainder */
    if (i < n) memcpy(d + i, s + i, n - i);
}

/* AVX-512 memcpy with NT store (don't pollute destination cache) */
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

/* libpmem2 style: 32x64B=2048B unroll with WC barrier per 12 CL */
static inline void _pmem2_nt_2048b(char *d, const char *s) {
    __m512i z0  = _mm512_loadu_si512((const __m512i *)(s));
    __m512i z1  = _mm512_loadu_si512((const __m512i *)(s + 64));
    __m512i z2  = _mm512_loadu_si512((const __m512i *)(s + 128));
    __m512i z3  = _mm512_loadu_si512((const __m512i *)(s + 192));
    __m512i z4  = _mm512_loadu_si512((const __m512i *)(s + 256));
    __m512i z5  = _mm512_loadu_si512((const __m512i *)(s + 320));
    __m512i z6  = _mm512_loadu_si512((const __m512i *)(s + 384));
    __m512i z7  = _mm512_loadu_si512((const __m512i *)(s + 448));
    __m512i z8  = _mm512_loadu_si512((const __m512i *)(s + 512));
    __m512i z9  = _mm512_loadu_si512((const __m512i *)(s + 576));
    __m512i z10 = _mm512_loadu_si512((const __m512i *)(s + 640));
    __m512i z11 = _mm512_loadu_si512((const __m512i *)(s + 704));
    __m512i z12 = _mm512_loadu_si512((const __m512i *)(s + 768));
    __m512i z13 = _mm512_loadu_si512((const __m512i *)(s + 832));
    __m512i z14 = _mm512_loadu_si512((const __m512i *)(s + 896));
    __m512i z15 = _mm512_loadu_si512((const __m512i *)(s + 960));
    __m512i z16 = _mm512_loadu_si512((const __m512i *)(s + 1024));
    __m512i z17 = _mm512_loadu_si512((const __m512i *)(s + 1088));
    __m512i z18 = _mm512_loadu_si512((const __m512i *)(s + 1152));
    __m512i z19 = _mm512_loadu_si512((const __m512i *)(s + 1216));
    __m512i z20 = _mm512_loadu_si512((const __m512i *)(s + 1280));
    __m512i z21 = _mm512_loadu_si512((const __m512i *)(s + 1344));
    __m512i z22 = _mm512_loadu_si512((const __m512i *)(s + 1408));
    __m512i z23 = _mm512_loadu_si512((const __m512i *)(s + 1472));
    __m512i z24 = _mm512_loadu_si512((const __m512i *)(s + 1536));
    __m512i z25 = _mm512_loadu_si512((const __m512i *)(s + 1600));
    __m512i z26 = _mm512_loadu_si512((const __m512i *)(s + 1664));
    __m512i z27 = _mm512_loadu_si512((const __m512i *)(s + 1728));
    __m512i z28 = _mm512_loadu_si512((const __m512i *)(s + 1792));
    __m512i z29 = _mm512_loadu_si512((const __m512i *)(s + 1856));
    __m512i z30 = _mm512_loadu_si512((const __m512i *)(s + 1920));
    __m512i z31 = _mm512_loadu_si512((const __m512i *)(s + 1984));

    _mm512_stream_si512((__m512i *)(d),       z0);
    _mm512_stream_si512((__m512i *)(d + 64),  z1);
    _mm512_stream_si512((__m512i *)(d + 128), z2);
    _mm512_stream_si512((__m512i *)(d + 192), z3);
    _mm512_stream_si512((__m512i *)(d + 256), z4);
    _mm512_stream_si512((__m512i *)(d + 320), z5);
    _mm512_stream_si512((__m512i *)(d + 384), z6);
    _mm512_stream_si512((__m512i *)(d + 448), z7);
    _mm512_stream_si512((__m512i *)(d + 512), z8);
    _mm512_stream_si512((__m512i *)(d + 576), z9);
    _mm512_stream_si512((__m512i *)(d + 640), z10);
    _mm512_stream_si512((__m512i *)(d + 704), z11);
    _mm_sfence(); /* WC barrier after 12 CL */
    _mm512_stream_si512((__m512i *)(d + 768),  z12);
    _mm512_stream_si512((__m512i *)(d + 832),  z13);
    _mm512_stream_si512((__m512i *)(d + 896),  z14);
    _mm512_stream_si512((__m512i *)(d + 960),  z15);
    _mm512_stream_si512((__m512i *)(d + 1024), z16);
    _mm512_stream_si512((__m512i *)(d + 1088), z17);
    _mm512_stream_si512((__m512i *)(d + 1152), z18);
    _mm512_stream_si512((__m512i *)(d + 1216), z19);
    _mm512_stream_si512((__m512i *)(d + 1280), z20);
    _mm512_stream_si512((__m512i *)(d + 1344), z21);
    _mm512_stream_si512((__m512i *)(d + 1408), z22);
    _mm512_stream_si512((__m512i *)(d + 1472), z23);
    _mm_sfence(); /* WC barrier after 12 CL */
    _mm512_stream_si512((__m512i *)(d + 1536), z24);
    _mm512_stream_si512((__m512i *)(d + 1600), z25);
    _mm512_stream_si512((__m512i *)(d + 1664), z26);
    _mm512_stream_si512((__m512i *)(d + 1728), z27);
    _mm512_stream_si512((__m512i *)(d + 1792), z28);
    _mm512_stream_si512((__m512i *)(d + 1856), z29);
    _mm512_stream_si512((__m512i *)(d + 1920), z30);
    _mm512_stream_si512((__m512i *)(d + 1984), z31);
    _mm_sfence(); /* WC barrier after 8 CL */
}

static void avx512_memcpy_nt_pmem2(void *dst, const void *src, size_t n) {
    char *d = (char *)dst;
    const char *s = (const char *)src;
    /* Align destination to 64B */
    size_t cnt = (uintptr_t)d & 63;
    if (cnt) {
        cnt = 64 - cnt;
        if (cnt > n) cnt = n;
        memcpy(d, s, cnt);
        d += cnt; s += cnt; n -= cnt;
    }
    /* Main loop: 2048B blocks */
    while (n >= 2048) {
        _pmem2_nt_2048b(d, s);
        d += 2048; s += 2048; n -= 2048;
    }
    /* Tail: 256B blocks */
    while (n >= 256) {
        __m512i z0 = _mm512_loadu_si512((const __m512i *)s);
        __m512i z1 = _mm512_loadu_si512((const __m512i *)(s + 64));
        __m512i z2 = _mm512_loadu_si512((const __m512i *)(s + 128));
        __m512i z3 = _mm512_loadu_si512((const __m512i *)(s + 192));
        _mm512_stream_si512((__m512i *)d, z0);
        _mm512_stream_si512((__m512i *)(d + 64), z1);
        _mm512_stream_si512((__m512i *)(d + 128), z2);
        _mm512_stream_si512((__m512i *)(d + 192), z3);
        d += 256; s += 256; n -= 256;
    }
    _mm_sfence();
    if (n > 0) memcpy(d, s, n);
}

/*
 * Mode 0: open → read → close (baseline, same as Python readinto)
 * Mode 1: open → mmap(MAP_POPULATE) → memcpy → munmap → close
 * Mode 2: open → mmap(MAP_POPULATE) → AVX-512 memcpy → munmap → close
 * Mode 3: open → mmap(MAP_POPULATE) → AVX-512 NT store → munmap → close
 * Mode 4: open → read → close (C, no Python overhead)
 *
 * Returns 0 on success, -1 on error.
 */
int fast_read_file(const char *path, void *dst, size_t size, int mode) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return -1;

    if (mode == 0 || mode == 4) {
        /* Direct read */
        ssize_t total = 0;
        while (total < (ssize_t)size) {
            ssize_t n = read(fd, (char *)dst + total, size - total);
            if (n <= 0) break;
            total += n;
        }
        close(fd);
        return (total == (ssize_t)size) ? 0 : -1;
    }

    /* mmap modes */
    void *mm = mmap(NULL, size, PROT_READ, MAP_PRIVATE | MAP_POPULATE, fd, 0);
    if (mm == MAP_FAILED) { close(fd); return -1; }

    switch (mode) {
    case 1: memcpy(dst, mm, size); break;
    case 2: avx512_memcpy(dst, mm, size); break;
    case 3: avx512_memcpy_nt(dst, mm, size); break;
    }

    munmap(mm, size);
    close(fd);
    return 0;
}

/*
 * Batch read: read multiple files in sequence (single thread).
 */
int fast_read_batch(const char **paths, void *dst, int count, size_t size, int mode) {
    for (int i = 0; i < count; i++) {
        int ret = fast_read_file(paths[i], (char *)dst + (size_t)i * size, size, mode);
        if (ret != 0) return ret;
    }
    return 0;
}

/*
 * DevDAX 直接メモリコピー: mmap 済み DevDAX 領域から dst へコピー
 * open/close/mmap/munmap 不要。純粋なメモリコピーのみ。
 *
 * mode:
 *   0 = glibc memcpy
 *   1 = AVX-512 temporal (256B XPLine aligned, load+store)
 *   2 = AVX-512 NT store (temporal load, non-temporal store to dst)
 *   3 = AVX-512 read-only (LLC prefetch, store なし)
 */
/*
 * Phase 2: pthread 並列ファイル読み出し
 * Python GIL を完全に回避して複数チャンクを同時に読む。
 */

#include <pthread.h>

typedef struct {
    const char *path;    /* FSDAX: file path, DevDAX: NULL */
    const void *src;     /* DevDAX: source address in mmap */
    void *dst;           /* destination (pinned DRAM) */
    size_t size;
    int mode;            /* 0=read, 1=mmap+memcpy, 2=mmap+avx512, 3=mmap+avx512_nt */
    int is_devdax;       /* 1 if DevDAX (no file open), 0 if FSDAX */
} read_task_t;

static void *_parallel_read_worker(void *arg) {
    read_task_t *t = (read_task_t *)arg;
    if (t->is_devdax) {
        /* DevDAX: direct memory copy from mmap'd region */
        switch (t->mode) {
        case 0: memcpy(t->dst, t->src, t->size); break;
        case 1: avx512_memcpy(t->dst, t->src, t->size); break;
        case 2: avx512_memcpy_nt(t->dst, t->src, t->size); break;
        case 3: avx512_memcpy_nt_pmem2(t->dst, t->src, t->size); break;
        default: memcpy(t->dst, t->src, t->size); break;
        }
    } else {
        /* FSDAX: file-based read */
        fast_read_file(t->path, t->dst, t->size, t->mode);
    }
    return NULL;
}

/*
 * Work-stealing parallel read: atomic counter で各スレッドが自律的にタスクを取得。
 * pthread create/join は 1 回のみ (batch barrier 方式の 3 回→1 回に削減)。
 */
typedef struct {
    read_task_t *tasks;
    volatile int *next_task;  /* atomic counter */
    int count;
} ws_arg_t;

static void *_ws_worker(void *arg) {
    ws_arg_t *ws = (ws_arg_t *)arg;
    while (1) {
        int idx = __sync_fetch_and_add(ws->next_task, 1);
        if (idx >= ws->count) break;
        _parallel_read_worker(&ws->tasks[idx]);
    }
    return NULL;
}

int parallel_read(read_task_t *tasks, int count, int max_threads) {
    if (max_threads <= 0 || max_threads > count) max_threads = count;

    volatile int next_task = 0;
    ws_arg_t ws = { .tasks = tasks, .next_task = &next_task, .count = count };

    pthread_t *threads = (pthread_t *)malloc(max_threads * sizeof(pthread_t));
    if (!threads) return -1;

    for (int i = 0; i < max_threads; i++)
        pthread_create(&threads[i], NULL, _ws_worker, &ws);
    for (int i = 0; i < max_threads; i++)
        pthread_join(threads[i], NULL);

    free(threads);
    return 0;
}

/*
 * parallel_devdax_copy: DevDAX mmap 領域から pinned DRAM へ並列コピー
 *
 * base: DevDAX mmap base address
 * offsets: 各チャンクの offset (base からの)
 * dsts: 各チャンクの destination address (pinned DRAM)
 * sizes: 各チャンクのサイズ
 * count: チャンク数
 * max_threads: 最大スレッド数
 * mode: 0=memcpy, 1=avx512, 2=avx512_nt
 */
int parallel_devdax_copy(
    const void *base,
    const int64_t *offsets,
    void **dsts,
    const int64_t *sizes,
    int count,
    int max_threads,
    int mode
) {
    read_task_t *tasks = (read_task_t *)malloc(count * sizeof(read_task_t));
    if (!tasks) return -1;

    for (int i = 0; i < count; i++) {
        tasks[i].path = NULL;
        tasks[i].src = (const char *)base + offsets[i];
        tasks[i].dst = dsts[i];
        tasks[i].size = (size_t)sizes[i];
        tasks[i].mode = mode;
        tasks[i].is_devdax = 1;
    }

    int ret = parallel_read(tasks, count, max_threads);
    free(tasks);
    return ret;
}

int fast_memcpy_devdax(void *dst, const void *src, size_t size, int mode) {
    switch (mode) {
    case 0:
        memcpy(dst, src, size);
        break;
    case 1:
        avx512_memcpy(dst, src, size);
        break;
    case 2:
        avx512_memcpy_nt(dst, src, size);
        break;
    case 4:
        avx512_memcpy_nt_pmem2(dst, src, size);
        break;
    case 3: {
        /* Read-only: LLC prefetch (load without store) */
        const char *s = (const char *)src;
        __m512i acc = _mm512_setzero_si512();
        size_t i;
        for (i = 0; i + 256 <= size; i += 256) {
            acc = _mm512_add_epi64(acc, _mm512_load_si512((__m512i *)(s + i)));
            acc = _mm512_add_epi64(acc, _mm512_load_si512((__m512i *)(s + i + 64)));
            acc = _mm512_add_epi64(acc, _mm512_load_si512((__m512i *)(s + i + 128)));
            acc = _mm512_add_epi64(acc, _mm512_load_si512((__m512i *)(s + i + 192)));
        }
        /* Prevent DCE */
        volatile uint64_t sink;
        uint64_t tmp[8];
        _mm512_store_si512(tmp, acc);
        sink = tmp[0];
        (void)sink;
        break;
    }
    }
    return 0;
}
