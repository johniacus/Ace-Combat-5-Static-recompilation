#ifndef PS2_GSQ_H
#define PS2_GSQ_H

/* The GS stage of the graphics machine on a thread of its own.
 *
 * The gfxq worker runs VIF, VU1 and the GIF parser; everything they hand the
 * GS (register writes, image data, native mesh draws, presents, privileged
 * writes, emitter tags) is queued here and done by a second thread, so the
 * GS half of a field runs while the worker is already on the next one. On
 * the Vita the worker had one core ~86% busy while the renderer's core sat
 * ~60% idle. AC5_GSQ=0 keeps the GS on the worker as before. */

#include "ps2_runtime.h"

struct ps2_vk_state;
struct ps2_vk_mesh_block;

#include <stdint.h>
#if !defined(__arm__)
#include <pthread.h>
#endif

extern int ps2_gsq_on;
extern _Thread_local int ps2_gsq_producer;   /* the gfxq worker thread */
/* Which thread is this, cheaply: every GS register write asks. _Thread_local
 * costs a library call per access on the Vita (emutls); the ARM user
 * thread-ID register (TPIDRURO) is one instruction and differs per thread. */
extern uintptr_t ps2_gsq_worker_tp;
static inline uintptr_t ps2_gsq_tp(void) {
#if defined(__arm__)
    uintptr_t v;
    __asm__ volatile("mrc p15, 0, %0, c13, c0, 3" : "=r"(v));
    return v;
#else
    return (uintptr_t)pthread_self();
#endif
}
#define PS2_GSQ_DEFER() (ps2_gsq_on && ps2_gsq_tp() == ps2_gsq_worker_tp)

void ps2_gsq_wreg(u32 reg, u64 val);
void ps2_gsq_hwpair(u64 lo, u64 hi);
void ps2_gsq_hwstream(const void *data, u32 qwc);
void ps2_gsq_nstate(void);
void ps2_gsq_mark_frame(void);
void ps2_gsq_nvertex(float x, float y, double z, float fog, int kind, int adc);
void ps2_gsq_drawmesh(const struct ps2_vk_state *st, const struct ps2_vk_mesh_block *b,
                      u32 vbase, u32 ifirst, u32 n, int cull, int topo);
u32  ps2_gsq_mstore(const u8 *recs, u32 nv, const u32 *idx, u32 ni);
void ps2_gsq_priv(u32 addr, u64 val);
void ps2_gsq_tag(u32 kind, u32 a, u32 b);
void ps2_gsq_intent(const u8 *rec, u32 len);
void ps2_gsq_field(void);
/* producer only: wait until the GS thread has done everything queued */
void ps2_gsq_sync(const char *why);
/* EE side: wait until the GS queue is empty and the thread idle */
void ps2_gsq_drain(void);
void ps2_gsq_init(void);
void ps2_gsq_worker_started(void);
void ps2_gsq_report(char *buf, int n);

/* mesh store tokens handed to the worker in place of arena offsets */
#define PS2_GSQ_TOKEN 0x80000000u
int  ps2_gsq_resolve(u32 v, int which, u32 *out);

#endif
