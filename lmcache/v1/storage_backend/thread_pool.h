/* Shared persistent thread pool for DevDAX copy paths.
 *
 * Usage:
 *   thread_pool_t *pool = thread_pool_get(47);  // lazy singleton per .so
 *   thread_pool_parallel_for(pool, my_worker_fn, &arg);  // blocks until all done
 *
 * Design:
 *   - N worker threads, created once at first call.
 *   - Each worker pins to a specific CPU core (affinity) for locality.
 *   - Submit: mutex + broadcast condvar + wait for completion condvar.
 *     All N workers call fn(arg) once; fn itself implements work-stealing
 *     on shared atomic counters inside arg.
 *
 *   Pool persists for the process lifetime (no teardown needed since
 *   LMCache backends live for entire vLLM process).
 */
#ifndef LMCACHE_THREAD_POOL_H
#define LMCACHE_THREAD_POOL_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct thread_pool thread_pool_t;
typedef void (*tp_task_fn)(void *arg);

/* Lazy singleton: first call creates the pool with `nthreads` workers. */
thread_pool_t *thread_pool_get(int nthreads);

/* Run fn(arg) on every worker; block until all N workers have returned. */
void thread_pool_parallel_for(thread_pool_t *pool, tp_task_fn fn, void *arg);

/* Returns worker count (0 if pool not yet created). */
int thread_pool_size(thread_pool_t *pool);

#ifdef __cplusplus
}
#endif

#endif /* LMCACHE_THREAD_POOL_H */
