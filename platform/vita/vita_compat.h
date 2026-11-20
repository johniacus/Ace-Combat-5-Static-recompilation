/* Force-included into every runtime and generated source on the Vita build
 * (-include vita_compat.h). Keeps the shared runtime free of #ifdefs for
 * things that only matter on this platform. */
#ifndef PS2_VITA_COMPAT_H
#define PS2_VITA_COMPAT_H

#include <limits.h>
#ifndef PATH_MAX
#define PATH_MAX 1024
#endif

#ifndef __cplusplus
#include <pthread.h>
#include <stdlib.h>

/* The runtime runs every guest EE thread, plus the vblank timer, gfx queue
 * and watchdog, on host pthreads created with default attributes. The Vita's
 * default pthread stack is far smaller than the 8 MB desktop default, and the
 * recompiled code is not shy with stack, so give every thread created without
 * explicit attributes a fixed, generous stack.
 *
 * Every thread also starts through vita_thread_tramp (vita_threads.c), which
 * records its kernel thread id and entry point for the per-thread CPU report. */
#ifndef PS2_VITA_THREAD_STACK
#define PS2_VITA_THREAD_STACK (1024 * 1024)
#endif

typedef struct vita_thread_start {
    void *(*fn)(void *);
    void *arg;
} vita_thread_start;
void *vita_thread_tramp(void *start);

static inline int ps2_vita_pthread_create(pthread_t *th,
                                          const pthread_attr_t *attr,
                                          void *(*fn)(void *), void *arg) {
    vita_thread_start *s = (vita_thread_start *)malloc(sizeof *s);
    if (!s) return -1;
    s->fn = fn;
    s->arg = arg;
    int rc;
    if (attr) {
        rc = pthread_create(th, attr, vita_thread_tramp, s);
    } else {
        pthread_attr_t a;
        pthread_attr_init(&a);
        pthread_attr_setstacksize(&a, PS2_VITA_THREAD_STACK);
        rc = pthread_create(th, &a, vita_thread_tramp, s);
        pthread_attr_destroy(&a);
    }
    if (rc != 0) free(s);
    return rc;
}
#define pthread_create ps2_vita_pthread_create
#endif /* !__cplusplus */

#endif
