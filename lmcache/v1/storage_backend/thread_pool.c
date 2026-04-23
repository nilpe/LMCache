/* Persistent thread pool implementation. See thread_pool.h. */
#define _GNU_SOURCE
#include "thread_pool.h"

#include <pthread.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

struct thread_pool {
    int nthreads;
    pthread_t *threads;

    pthread_mutex_t mu;
    pthread_cond_t cv_work;       /* workers wait here for next job */
    pthread_cond_t cv_done;       /* submitter waits here for completion */

    /* Current job: bumped generation prevents lost-wakeup. */
    tp_task_fn fn;
    void *arg;
    unsigned long long generation;
    int workers_done;
    int shutdown;
};

typedef struct {
    thread_pool_t *pool;
    int tid;
    int cpu;
} tp_worker_ctx_t;

static void *tp_worker_main(void *arg) {
    tp_worker_ctx_t *ctx = (tp_worker_ctx_t *)arg;
    thread_pool_t *pool = ctx->pool;

    /* Pin to a specific CPU (best-effort). */
    if (ctx->cpu >= 0) {
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(ctx->cpu, &set);
        (void)pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
    }

    unsigned long long last_gen = 0;
    while (1) {
        pthread_mutex_lock(&pool->mu);
        while (pool->generation == last_gen && !pool->shutdown)
            pthread_cond_wait(&pool->cv_work, &pool->mu);
        if (pool->shutdown) {
            pthread_mutex_unlock(&pool->mu);
            break;
        }
        last_gen = pool->generation;
        tp_task_fn fn = pool->fn;
        void *a = pool->arg;
        pthread_mutex_unlock(&pool->mu);

        fn(a);

        pthread_mutex_lock(&pool->mu);
        pool->workers_done++;
        if (pool->workers_done == pool->nthreads)
            pthread_cond_signal(&pool->cv_done);
        pthread_mutex_unlock(&pool->mu);
    }

    free(ctx);
    return NULL;
}

static thread_pool_t *g_pool = NULL;  /* per-.so singleton */
static pthread_mutex_t g_init_mu = PTHREAD_MUTEX_INITIALIZER;

thread_pool_t *thread_pool_get(int nthreads) {
    pthread_mutex_lock(&g_init_mu);
    if (g_pool) {
        pthread_mutex_unlock(&g_init_mu);
        return g_pool;
    }

    int ncpu = (int)sysconf(_SC_NPROCESSORS_ONLN);
    if (nthreads <= 0) nthreads = ncpu;
    if (nthreads > ncpu) nthreads = ncpu;

    thread_pool_t *p = (thread_pool_t *)calloc(1, sizeof(*p));
    p->nthreads = nthreads;
    p->threads = (pthread_t *)calloc(nthreads, sizeof(pthread_t));
    pthread_mutex_init(&p->mu, NULL);
    pthread_cond_init(&p->cv_work, NULL);
    pthread_cond_init(&p->cv_done, NULL);

    for (int i = 0; i < nthreads; i++) {
        tp_worker_ctx_t *ctx = (tp_worker_ctx_t *)malloc(sizeof(*ctx));
        ctx->pool = p;
        ctx->tid = i;
        /* Spread across cores: tid i → cpu i % ncpu. */
        ctx->cpu = i % ncpu;
        pthread_create(&p->threads[i], NULL, tp_worker_main, ctx);
    }

    g_pool = p;
    fprintf(stderr, "thread_pool: started %d pinned workers (ncpu=%d)\n",
            nthreads, ncpu);
    pthread_mutex_unlock(&g_init_mu);
    return p;
}

void thread_pool_parallel_for(thread_pool_t *pool, tp_task_fn fn, void *arg) {
    pthread_mutex_lock(&pool->mu);
    pool->fn = fn;
    pool->arg = arg;
    pool->workers_done = 0;
    pool->generation++;
    pthread_cond_broadcast(&pool->cv_work);
    while (pool->workers_done < pool->nthreads)
        pthread_cond_wait(&pool->cv_done, &pool->mu);
    pthread_mutex_unlock(&pool->mu);
}

int thread_pool_size(thread_pool_t *pool) {
    return pool ? pool->nthreads : 0;
}
