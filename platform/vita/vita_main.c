/* PS Vita entry point. Sets up clocks, memory and file locations, then hands
 * over to the runtime's own main (ps2_main.c compiled with -Dmain=ps2_main).
 *
 * Files on the memory card, all under ux0:data/ac5/:
 *   ps2_image.bin   from the recompiler's generated/ folder
 *   ac5.iso         your SLUS-20851 disc image   (or)
 *   disc/           the extracted disc files
 *   args.txt        optional: extra command-line flags, whitespace separated
 *   log.txt         written on every run
 */
#include <psp2/kernel/processmgr.h>
#include <psp2/kernel/threadmgr.h>
#include <psp2/power.h>
#include <psp2/io/stat.h>
#include <psp2/apputil.h>
#include <psp2/kernel/cpu.h>
#include <psp2/kernel/sysmem.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>

/* newlib heap. Build 9 measured 41 MB of system main RAM still free with a
 * 192 MB heap while the heap itself filled up in missions, so it gets 16 MB
 * more. (Set here rather than through the CMake PS2_VITA_HEAP_MB option, which
 * is a target-wide define: changing it recompiles all 43 generated files.) */
#undef PS2_VITA_HEAP_MB
#define PS2_VITA_HEAP_MB 208

/* newlib heap; everything the runtime callocs comes out of this. */
int _newlib_heap_size_user = PS2_VITA_HEAP_MB * 1024 * 1024;
/* ps2_dispatch runs the main guest thread on this thread's stack. */
unsigned int sceUserMainThreadStackSize = 4 * 1024 * 1024;

#define AC5_DIR "ux0:data/ac5"

extern int ps2_main(int argc, char **argv);
extern void vita_display_start(void);
extern void vita_thread_register(const char *name, const void *fn);

static int exists(const char *path) {
    SceIoStat st;
    return sceIoGetstat(path, &st) >= 0;
}

#define MAX_ARGS 64

/* Startup check that __thread really is per-thread (see vita_shims.c). */
static __thread int tls_probe = 1;
static void *tls_probe_thread(void *out) {
    tls_probe = 2;
    *(int *)out = tls_probe;
    return NULL;
}
static void check_tls(void) {
    pthread_t th;
    int seen = 0;
    tls_probe = 7;
    if (pthread_create(&th, NULL, tls_probe_thread, &seen) != 0) {
        fprintf(stderr, "ac5 vita: TLS check could not start a thread\n");
        return;
    }
    pthread_join(th, NULL);
    fprintf(stderr, "ac5 vita: thread-local storage %s (main=%d, worker=%d)\n",
            tls_probe == 7 && seen == 2 ? "OK, per-thread"
                                        : "BROKEN, shared between threads",
            tls_probe, seen);
}

int main(int argc, char **argv) {
    (void)argc; (void)argv;
    static char *av[MAX_ARGS];
    static char extra[4096];
    int ac = 0;

    scePowerSetArmClockFrequency(444);
    scePowerSetBusClockFrequency(222);
    scePowerSetGpuClockFrequency(222);
    scePowerSetGpuXbarClockFrequency(166);

    /* The main thread runs the EE: give it core 0 to itself. Threads it
     * starts are moved to core 2 by vita_thread_tramp, and the graphics
     * worker moves itself to core 1. */
    sceKernelChangeThreadCpuAffinityMask(sceKernelGetThreadId(), SCE_KERNEL_CPU_MASK_USER_ALL);
    /* (the display thread moves this, the EE thread, to core 0 once the
     * guest is running: threads created before then -- SDL's audio thread
     * among them -- inherit all three cores rather than the EE's one) */

    /* Runtime diagnostics that are on by default in this build. */
    setenv("PS2_SPEED_LOG", "1", 0);
    setenv("PS2_PROFILE_GUEST", "1", 0);
    setenv("PS2_PROFILE_PHASES", "0", 0);  /* "phases" lines: two clock reads per phase in one field of 8 (AC5_PROF-style cost) */
    setenv("PS2_ASYNC_FIELDS", "5", 0);
    setenv("PS2_VU_CENSUS", "0", 0);       /* record VU1 microprograms for tools/vurecomp */
    setenv("AC5_GL_NOVBO", "1", 0);        /* frame vertex buffer: slower and broke the HUD in build 41 */
    setenv("PS2_RN_SCREEN", "0", 0);
    setenv("PS2_RN_CENSUS", "0", 0);       /* per-primitive pixel accounting: a PC diagnostic */       /* 2D stays on the GS path; only 3D goes native mesh */
    setenv("PS2_VU_RECOMP", "0", 0);       /* recompiled VU1 programs drew the mission wrongly in build 41 */    /* let the EE run a field further ahead of the graphics worker */   /* periodic "guest profile" log lines */
    /* The PC default 64 MB graphics ring is far more than the Vita can spare. */
    setenv("PS2_ASYNC_MB", "16", 0);

    sceIoMkdir(AC5_DIR, 0777);
    chdir(AC5_DIR);

    /* ps2_log writes to stderr; send both streams to the log file. */
    freopen(AC5_DIR "/log.txt", "w", stderr);
    freopen(AC5_DIR "/stdout.txt", "w", stdout);
    setvbuf(stderr, NULL, _IONBF, 0);

    fprintf(stderr, "ac5 vita: heap %d MB, main stack %u KB\n",
            PS2_VITA_HEAP_MB, sceUserMainThreadStackSize / 1024);
    check_tls();
    {   SceKernelFreeMemorySizeInfo fm;
        memset(&fm, 0, sizeof fm);
        fm.size = sizeof fm;
        sceKernelGetFreeMemorySize(&fm);
        fprintf(stderr, "ac5 vita: free memory at start: main %d MB, CDRAM %d MB, "
                        "phycont %d MB\n", fm.size_user >> 20, fm.size_cdram >> 20,
                fm.size_phycont >> 20);
    }
    vita_thread_register("ee-main", (const void *)main);

    av[ac++] = (char *)"ac5";
    av[ac++] = (char *)"--data";
    av[ac++] = (char *)AC5_DIR;
    if (exists(AC5_DIR "/ac5.iso")) {
        av[ac++] = (char *)"--disc";
        av[ac++] = (char *)AC5_DIR "/ac5.iso";
    } else if (exists(AC5_DIR "/disc")) {
        av[ac++] = (char *)"--disc";
        av[ac++] = (char *)AC5_DIR "/disc";
    } else {
        fprintf(stderr, "ac5 vita: no disc found at " AC5_DIR "/ac5.iso or "
                        AC5_DIR "/disc/\n");
    }
    av[ac++] = (char *)"--watchdog";
    av[ac++] = (char *)"0";

    /* Optional extra flags, e.g. "--verbose --vblanks 600". */
    {
        FILE *fp = fopen(AC5_DIR "/args.txt", "r");
        if (fp) {
            size_t n = fread(extra, 1, sizeof extra - 1, fp);
            fclose(fp);
            extra[n] = 0;
            for (char *t = strtok(extra, " \t\r\n"); t && ac < MAX_ARGS - 1;
                 t = strtok(NULL, " \t\r\n")) {
                /* NAME=value sets an environment variable (PS2_* switches). */
                char *eq = t[0] != '-' ? strchr(t, '=') : NULL;
                if (eq) {
                    *eq = 0;
                    setenv(t, eq + 1, 1);
                    fprintf(stderr, "env: %s=%s\n", t, eq + 1);
                    continue;
                }
                av[ac++] = t;
            }
        }
    }
    av[ac] = NULL;
    {   extern int ac5_prof;
        ac5_prof = getenv("AC5_PROF") != NULL; }

    for (int i = 0; i < ac; i++) fprintf(stderr, "%s%s", i ? " " : "args: ", av[i]);
    fprintf(stderr, "\n");

    vita_display_start();
    {   extern void vita_watchdog_start(void); vita_watchdog_start(); }
    if (!getenv("AC5_NO_BENCH")) {
        extern void ps2_ipu_bench(void), ps2_gs_bench(void);
        ps2_ipu_bench();
        ps2_gs_bench();
    }
    int rc = ps2_main(ac, av);
    fprintf(stderr, "ac5 vita: ps2_main returned %d\n", rc);
    fflush(stderr);
    sceKernelExitProcess(0);
    return rc;
}

/* Layout padding: vita-elf-create appends its SCE tables to the end of the
 * text segment and fails if that end lands within a few KB of the data
 * segment. Adjust the size of this block if it ever reports
 * "segment 1 overlaps" again. */
__attribute__((used)) const unsigned char ac5_layout_pad[20480] = { 1 };
