/* Per-thread CPU accounting for the Vita build.
 *
 * Every pthread the runtime creates starts in vita_thread_tramp (see
 * vita_compat.h), which records the kernel thread id and the entry function.
 * vita_threads_report() then logs each thread's share of a core since the
 * last report, from the kernel's runClocks, plus which core it last ran on. */
#include "ps2_runtime.h"

#include <psp2/kernel/threadmgr.h>
#include <psp2/kernel/processmgr.h>

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <psp2/io/fcntl.h>

/* vita_compat.h's pthread_create wrapper must not wrap our own trampoline. */
#undef pthread_create

#define MAX_TRACKED 64

typedef struct {
    SceUID id;
    const void *fn;
    char name[32];
    u64 last_clocks;
} tracked;

static tracked list[MAX_TRACKED];
static int count;
static pthread_mutex_t list_lock = PTHREAD_MUTEX_INITIALIZER;

void vita_thread_register(const char *name, const void *fn) {
    pthread_mutex_lock(&list_lock);
    if (count < MAX_TRACKED) {
        tracked *t = &list[count++];
        memset(t, 0, sizeof *t);
        t->id = sceKernelGetThreadId();
        t->fn = fn;
        if (name) strncpy(t->name, name, sizeof t->name - 1);
    }
    pthread_mutex_unlock(&list_lock);
}

/* Pin the calling thread to one user core (0-2). The Vita scheduler does
 * not move threads between cores on its own, and new threads inherit their
 * creator's core: in build 10 the EE thread and the graphics worker often
 * shared a core and took turns, holding cutscenes and missions to ~40
 * fields/s while each sat half idle. */
/* Thread priorities (lower number = more urgent). The Vita does not
 * time-slice threads of equal priority on a core, so a guest thread that
 * spins starves everything of its priority pinned to its core: in build 14
 * guest threads were pinned to core 2 next to the vblank timer, the display
 * thread and the watchdog, and the game froze with all three starved.
 * Our service threads now outrank the guest threads, and guest threads may
 * run on any core. */
#define PRIO_TIMER   112    /* vblank timer (core 2) */
#define PRIO_WORKER  140    /* graphics worker (core 1) */
#define PRIO_EE      150    /* main EE thread (core 0) */
#define PRIO_GUEST   160    /* every other runtime thread, any core */

void vita_pin_current(int core) {
    if (core < 0 || core > 2) return;
    sceKernelChangeThreadCpuAffinityMask(sceKernelGetThreadId(),
                                         SCE_KERNEL_CPU_MASK_USER_0 << core);
    sceKernelChangeThreadPriority(sceKernelGetThreadId(),
                                  core == 1 ? PRIO_WORKER : core == 2 ? PRIO_TIMER : PRIO_EE);
}

void vita_pin_named(const char *name, int core, int prio) {
    pthread_mutex_lock(&list_lock);
    for (int i = 0; i < count; i++)
        if (strcmp(list[i].name, name) == 0) {
            sceKernelChangeThreadCpuAffinityMask(list[i].id, SCE_KERNEL_CPU_MASK_USER_0 << core);
            sceKernelChangeThreadPriority(list[i].id, prio);
        }
    pthread_mutex_unlock(&list_lock);
}

void vita_set_priority(int prio) {
    sceKernelChangeThreadPriority(sceKernelGetThreadId(), prio);
}

void *vita_thread_tramp(void *p) {
    vita_thread_start s = *(vita_thread_start *)p;
    free(p);
    /* Every runtime thread (guest threads, helpers) may use any core at a
     * priority below ours; the EE (core 0), the graphics worker (core 1)
     * and the vblank timer (core 2) pin themselves. */
    /* cores 0 and 1 only (AC5_GUEST_ALLCORES=1: any): core 2 belongs to the
     * renderer and the GS thread, and a guest thread there either starved the
     * GS (GS below it) or stuttered the audio (GS above it) -- build 83/84 */
    {   static int all = -1;
        if (all < 0) all = getenv("AC5_GUEST_ALLCORES") ? 1 : 0;
        sceKernelChangeThreadCpuAffinityMask(sceKernelGetThreadId(),
            all ? SCE_KERNEL_CPU_MASK_USER_ALL : (SCE_KERNEL_CPU_MASK_USER_0 | SCE_KERNEL_CPU_MASK_USER_1));
    }
    sceKernelChangeThreadPriority(sceKernelGetThreadId(), PRIO_GUEST);
    vita_thread_register(NULL, (const void *)s.fn);
    return s.fn(s.arg);
}

void vita_threads_report(void) {
    static u64 last_us;
    u64 now = sceKernelGetProcessTimeWide();
    u64 span = last_us ? now - last_us : 0;
    last_us = now;

    pthread_mutex_lock(&list_lock);
    for (int i = 0; i < count; i++) {
        tracked *t = &list[i];
        SceKernelThreadInfo info;
        memset(&info, 0, sizeof info);
        info.size = sizeof info;
        if (sceKernelGetThreadInfo(t->id, &info) < 0) continue;
        u64 clocks = info.runClocks;          /* microseconds */
        u64 used = clocks - t->last_clocks;
        t->last_clocks = clocks;
        if (!span) continue;
        if (used * 100 < span) continue;      /* hide threads under 1% */
        ps2_log("vita cpu: %5.1f%%  core %d  mask %08X  %-16s fn %08X",
                100.0 * (double)used / (double)span, info.lastExecutedCpuId,
                (unsigned)info.currentCpuAffinityMask,
                t->name[0] ? t->name : info.name,
                (unsigned)(uintptr_t)t->fn);
    }
    pthread_mutex_unlock(&list_lock);
}

/* ------------------------------------------------------------ freeze dump */

volatile u32 vita_display_ticks;
u64 ps2_kernel_vblank_count(void);
void vita_gl_interval(u64 *frames, u64 *draws, u64 *uploads, int *variants);
int ps2_vk_enabled(void);

static void raw_write(SceUID fd, const char *s) { sceIoWrite(fd, s, strlen(s)); }

/* Tiny formatters: the freeze report must not touch stdio, malloc or
 * newlib's per-thread state, any of which a stuck thread may be holding. */
static char *put_s(char *o, const char *s) { while (*s) *o++ = *s++; return o; }
static char *put_u(char *o, unsigned long long v) {
    char t[24]; int n = 0;
    do { t[n++] = (char)('0' + v % 10u); v /= 10u; } while (v);
    while (n) *o++ = t[--n];
    return o;
}
static char *put_x(char *o, unsigned v) {
    for (int i = 7; i >= 0; i--) *o++ = "0123456789ABCDEF"[(v >> (i * 4)) & 15u];
    return o;
}

static void dump_threads(SceUID fd) {
    char line[200];
    for (int i = 0; i < count; i++) {
        tracked *t = &list[i];
        SceKernelThreadInfo info;
        memset(&info, 0, sizeof info);
        info.size = sizeof info;
        if (sceKernelGetThreadInfo(t->id, &info) < 0) continue;
        char *o = put_s(line, "  ");
        o = put_s(o, t->name[0] ? t->name : info.name);
        o = put_s(o, " id "); o = put_x(o, (unsigned)t->id);
        o = put_s(o, " status "); o = put_u(o, info.status);
        o = put_s(o, " prio "); o = put_u(o, (unsigned)info.currentPriority);
        o = put_s(o, " wait type "); o = put_u(o, info.waitType);
        o = put_s(o, " on "); o = put_x(o, (unsigned)info.waitId);
        o = put_s(o, " cpu "); o = put_u(o, (unsigned)info.lastExecutedCpuId);
        o = put_s(o, " fn "); o = put_x(o, (unsigned)(uintptr_t)t->fn);
        o = put_s(o, "\n");
        sceIoWrite(fd, line, (SceSize)(o - line));
    }
}

static int watchdog_main(SceSize args, void *argp) {
    (void)args; (void)argp;
    u32 last_disp = 0, disp_still = 0;
    u64 last_fields = 0, last_frames = 0;
    u32 fields_still = 0, frames_still = 0;
    const char *e = getenv("AC5_FREEZE_DUMP");
    int armed = !(e && *e == '0');
    ps2_log("vita watchdog: running (%s); a freeze writes ux0:data/ac5/freeze.txt",
            armed ? "stops the game for a crash dump on a freeze" : "report only");
    for (;;) {
        sceKernelDelayThread(1000000);
        u64 fields = ps2_kernel_vblank_count();
        u64 frames = 0;
        if (fields < 120) continue;              /* still starting up */
        int gpu = ps2_vk_enabled();
        if (gpu) vita_gl_interval(&frames, NULL, NULL, NULL);
        u32 d = vita_display_ticks;
        disp_still = d == last_disp ? disp_still + 1 : 0;
        fields_still = fields == last_fields ? fields_still + 1 : 0;
        frames_still = (gpu && frames == last_frames) ? frames_still + 1 : 0;
        last_disp = d; last_fields = fields; last_frames = frames;
        const char *why = NULL;
        if (disp_still >= 8) why = "the display/input thread has not run for 8 s";
        else if (fields_still >= 12) why = "the game has not advanced a field for 12 s";
        else if (frames_still >= 12) why = "the GPU renderer has not finished a frame for 12 s";
        if (!why) continue;
        SceUID fd = sceIoOpen("ux0:data/ac5/freeze.txt", SCE_O_WRONLY | SCE_O_CREAT | SCE_O_TRUNC, 0666);
        if (fd >= 0) {
            char line[200];
            char *o = put_s(line, "freeze: ");
            o = put_s(o, why);
            o = put_s(o, " (field "); o = put_u(o, fields);
            o = put_s(o, ", gl frames "); o = put_u(o, frames);
            o = put_s(o, ")\n");
            sceIoWrite(fd, line, (SceSize)(o - line));
            dump_threads(fd);
            raw_write(fd, armed ? "freeze: stopping on purpose so the Vita writes a crash dump "
                                  "with every thread's stack\n"
                                : "freeze: AC5_FREEZE_DUMP=0, not stopping\n");
            sceIoClose(fd);
        }
        if (armed) {
            *(volatile u32 *)0 = 0xF4EE2Eu;      /* deliberate: core dump */
        }
        disp_still = fields_still = frames_still = 0;
    }
    return 0;
}

void vita_watchdog_start(void) {
    SceUID th = sceKernelCreateThread("ac5_watchdog", watchdog_main, 96, 0x4000, 0,
                                      SCE_KERNEL_CPU_MASK_USER_ALL, NULL);
    if (th >= 0) sceKernelStartThread(th, 0, NULL);
}

u64 vita_tsc(void) { return sceKernelGetProcessTimeWide(); }
