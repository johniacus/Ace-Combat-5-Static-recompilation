/* Stage 1.5 output for the Vita: no GPU renderer yet, so show what the
 * runtime's software Graphics Synthesizer has drawn into its emulated VRAM.
 *
 * With ps2_vk_enabled() returning 0, ps2_gs.c rasterises every primitive in
 * software into GS VRAM, exactly as the PC build does with --novideo. This
 * thread copies the displayed buffer (DISPFB1/2 + DISPLAY1/2) to the Vita
 * screen through sceDisplay, scaled to 4:3 at 544 lines, and feeds the
 * controls into the pad HLE. It's slow and unfiltered, but it shows the game.
 *
 * Quit: hold START + SELECT + L + R for one second. */
#include "ps2_runtime.h"
#include "ps2_hle.h"
#include "ps2_settings.h"

#include <psp2/ctrl.h>
#include <psp2/display.h>
#include <psp2/kernel/processmgr.h>
#include <psp2/kernel/sysmem.h>
#include <psp2/kernel/threadmgr.h>
#include <psp2/touch.h>

#include <stdio.h>
#include <string.h>
#include <malloc.h>
#include <psp2/kernel/sysmem.h>

const u8 *ps2_gs_framebuffer(u32 *w, u32 *h, u32 *stride);
u32 ps2_gs_display_width(void);
u32 ps2_gs_display_height(void);
u64 ps2_gs_priv_read(u32 addr);
u64 ps2_kernel_vblank_count(void);
void ps2_pad_publish(int port, const ps2_pad_state *st);
void ps2_gs_stats(u64 *prims, u64 *pixels, u64 *regs);
void vita_thread_register(const char *name, const void *fn);
void vita_threads_report(void);
void vita_gl_interval(u64 *frames, u64 *draws, u64 *uploads, int *variants);
void vita_gl_mem(char *buf, size_t n);
int ps2_vk_enabled(void);

static volatile int want_vram_dump;

/* Log the GS display setup whenever it changes, so the picture mapping can
 * be checked against what the game actually programmed. */
static void log_display_regs(void) {
    static u64 last[4];
    u64 r[4] = {
        ps2_gs_priv_read(0x12000000u),  /* PMODE */
        ps2_gs_priv_read(0x12000020u),  /* SMODE2 */
        0, 0 };
    int two = (r[0] & 2ull) != 0;
    r[2] = ps2_gs_priv_read(two ? 0x12000090u : 0x12000070u);  /* DISPFB */
    r[3] = ps2_gs_priv_read(two ? 0x120000A0u : 0x12000080u);  /* DISPLAY */
    if (!memcmp(r, last, sizeof r)) return;
    memcpy(last, r, sizeof r);
    ps2_log("vita display: PMODE=%llX SMODE2=%llX (int=%u ffmd=%u) circuit=%d",
            (unsigned long long)r[0], (unsigned long long)r[1],
            (unsigned)(r[1] & 1ull), (unsigned)((r[1] >> 1) & 1ull), two ? 2 : 1);
    ps2_log("vita display: DISPFB fbp=%u (byte %u) fbw=%u psm=%02X dbx=%u dby=%u",
            (unsigned)(r[2] & 0x1FFull), (unsigned)(r[2] & 0x1FFull) * 8192u,
            (unsigned)((r[2] >> 9) & 0x3Full) * 64u, (unsigned)((r[2] >> 15) & 0x1Full),
            (unsigned)((r[2] >> 32) & 0x7FFull), (unsigned)((r[2] >> 43) & 0x7FFull));
    ps2_log("vita display: DISPLAY dx=%u dy=%u magh=%u magv=%u dw=%u dh=%u -> %ux%u",
            (unsigned)(r[3] & 0xFFFull), (unsigned)((r[3] >> 12) & 0x7FFull),
            (unsigned)((r[3] >> 23) & 0xFull) + 1u, (unsigned)((r[3] >> 27) & 3ull) + 1u,
            (unsigned)((r[3] >> 32) & 0xFFFull) + 1u, (unsigned)((r[3] >> 44) & 0x7FFull) + 1u,
            ps2_gs_display_width(), ps2_gs_display_height());
}

/* SELECT + TRIANGLE: write all 4 MB of GS VRAM and the display registers. */
static void dump_vram(void) {
    static int n;
    u32 w, h, stride;
    const u8 *at = ps2_gs_framebuffer(&w, &h, &stride);
    if (!at) return;
    u64 pmode = ps2_gs_priv_read(0x12000000u);
    u64 fb = ps2_gs_priv_read((pmode & 2ull) ? 0x12000090u : 0x12000070u);
    const u8 *vram = at - (u32)(fb & 0x1FFull) * 8192u;
    char path[96];
    snprintf(path, sizeof path, "ux0:data/ac5/vram_%03d.bin", n);
    FILE *f = fopen(path, "wb");
    if (!f) return;
    fwrite(vram, 1, 4u * 1024u * 1024u, f);
    fclose(f);
    snprintf(path, sizeof path, "ux0:data/ac5/vram_%03d.txt", n);
    f = fopen(path, "w");
    if (f) {
        static const u32 regs[] = { 0x12000000u, 0x12000020u, 0x12000070u,
                                    0x12000080u, 0x12000090u, 0x120000A0u };
        static const char *names[] = { "PMODE", "SMODE2", "DISPFB1",
                                       "DISPLAY1", "DISPFB2", "DISPLAY2" };
        for (int i = 0; i < 6; i++)
            fprintf(f, "%s %016llX\n", names[i],
                    (unsigned long long)ps2_gs_priv_read(regs[i]));
        fprintf(f, "field %llu\n", (unsigned long long)ps2_kernel_vblank_count());
        fclose(f);
    }
    ps2_log("vita: GS VRAM written to vram_%03d.bin", n);
    n++;
}

#define SCR_W 960
#define SCR_H 544
#define FB_BYTES (SCR_W * SCR_H * 4)
#define FB_ALLOC ((FB_BYTES + 0x3FFFF) & ~0x3FFFF)

static u32 *screens[2];
static int xmap[SCR_W];
static int ymap[SCR_H];
static u32 map_w, map_h;
static int out_x0, out_w;

static int alloc_screens(void) {
    for (int i = 0; i < 2; i++) {
        SceUID blk = sceKernelAllocMemBlock("ac5_display",
                                            SCE_KERNEL_MEMBLOCK_TYPE_USER_CDRAM_RW,
                                            FB_ALLOC, NULL);
        if (blk < 0) {
            ps2_log("vita display: cannot allocate a framebuffer (%08X)", blk);
            return -1;
        }
        sceKernelGetMemBlockBase(blk, (void **)&screens[i]);
        memset(screens[i], 0, FB_BYTES);
    }
    return 0;
}

static void build_maps(u32 w, u32 h) {
    /* 4:3 picture, full height, centred. */
    out_w = SCR_H * 4 / 3;
    out_x0 = (SCR_W - out_w) / 2;
    for (int x = 0; x < out_w; x++) xmap[x] = (int)((u64)x * w / (u32)out_w);
    for (int y = 0; y < SCR_H; y++) ymap[y] = (int)((u64)y * h / SCR_H);
    map_w = w;
    map_h = h;
}

static inline u32 c16to32(u16 p) {
    u32 r = (p & 0x1Fu) << 3, g = ((p >> 5) & 0x1Fu) << 3, b = ((p >> 10) & 0x1Fu) << 3;
    return r | (r >> 5) | ((g | (g >> 5)) << 8) | ((b | (b >> 5)) << 16) | 0xFF000000u;
}

static void blit(u32 *dst) {
    u32 w, h, stride_unused;
    const u8 *at = ps2_gs_framebuffer(&w, &h, &stride_unused);
    if (!at) return;

    /* Re-derive the layout from the display registers: ps2_gs_framebuffer
     * assumes 32-bit pixels and no display offset. */
    u64 pmode = ps2_gs_priv_read(0x12000000u);
    u64 fb = ps2_gs_priv_read((pmode & 2ull) ? 0x12000090u : 0x12000070u);
    u32 fbp = (u32)(fb & 0x1FFull) * 2048u * 4u;
    u32 fbw = (u32)((fb >> 9) & 0x3Full) * 64u;
    u32 psm = (u32)((fb >> 15) & 0x1Full);
    u32 dbx = (u32)((fb >> 32) & 0x7FFull), dby = (u32)((fb >> 43) & 0x7FFull);
    const u8 *vram = at - fbp;
    int bpp = (psm == 0x02 || psm == 0x0A) ? 2 : 4;
    if (!fbw) fbw = 640;
    if (w != map_w || h != map_h) build_maps(w, h);

    u32 base = fbp + (dby * fbw + dbx) * (u32)bpp;
    u32 pitch = fbw * (u32)bpp;
    const u32 vram_size = 4u * 1024u * 1024u;

    for (int y = 0; y < SCR_H; y++) {
        u32 *row = dst + y * SCR_W + out_x0;
        u32 off = base + (u32)ymap[y] * pitch;
        if (off + w * (u32)bpp > vram_size) { memset(row, 0, (size_t)out_w * 4); continue; }
        const u8 *src = vram + off;
        if (bpp == 4) {
            const u32 *s = (const u32 *)src;
            for (int x = 0; x < out_w; x++) row[x] = s[xmap[x]] | 0xFF000000u;
        } else {
            const u16 *s = (const u16 *)src;
            for (int x = 0; x < out_w; x++) row[x] = c16to32(s[xmap[x]]);
        }
    }
}

static u8 axis(u8 v) { return v; }

static void poll_pad(void) {
    SceCtrlData c;
    SceTouchData back;
    ps2_pad_state st;
    memset(&st, 0, sizeof st);
    st.connected = 1;
    st.lx = st.ly = st.rx = st.ry = PS2_PAD_ANALOG_NEUTRAL;
    if (sceCtrlPeekBufferPositive(0, &c, 1) < 0) return;

    struct { u32 vita; int act; } map[] = {
        { SCE_CTRL_SELECT,   PS2_ACT_SELECT },   { SCE_CTRL_START,  PS2_ACT_START },
        { SCE_CTRL_UP,       PS2_ACT_UP },       { SCE_CTRL_RIGHT,  PS2_ACT_RIGHT },
        { SCE_CTRL_DOWN,     PS2_ACT_DOWN },     { SCE_CTRL_LEFT,   PS2_ACT_LEFT },
        { SCE_CTRL_LTRIGGER, PS2_ACT_L1 },       { SCE_CTRL_RTRIGGER, PS2_ACT_R1 },
        { SCE_CTRL_TRIANGLE, PS2_ACT_TRIANGLE }, { SCE_CTRL_CIRCLE, PS2_ACT_CIRCLE },
        { SCE_CTRL_CROSS,    PS2_ACT_CROSS },    { SCE_CTRL_SQUARE, PS2_ACT_SQUARE },
    };
    {   static int was;
        const u32 combo = SCE_CTRL_SELECT | SCE_CTRL_TRIANGLE;
        int now_held = (c.buttons & combo) == combo;
        if (now_held && !was) {
            extern void vita_gl_request_dump(void);
            want_vram_dump = 1;
            if (ps2_vk_enabled()) vita_gl_request_dump();
        }
        was = now_held;
        if (now_held) return;   /* don't pass the combo to the game */
    }
    for (unsigned i = 0; i < sizeof map / sizeof map[0]; i++)
        if (c.buttons & map[i].vita) st.buttons |= (u16)(1u << map[i].act);

    /* Rear touchpad: left half = L2, right half = R2 (throttle in AC5). */
    if (sceTouchPeek(SCE_TOUCH_PORT_BACK, &back, 1) >= 0) {
        for (unsigned i = 0; i < back.reportNum; i++) {
            if (back.report[i].x < 960) { st.buttons |= 1u << PS2_ACT_L2; st.l2 = 0xFF; }
            else                        { st.buttons |= 1u << PS2_ACT_R2; st.r2 = 0xFF; }
        }
    }

    st.lx = axis(c.lx); st.ly = axis(c.ly);
    st.rx = axis(c.rx); st.ry = axis(c.ry);
    ps2_pad_publish(0, &st);

    {   static u64 quit_since;
        const u32 combo = SCE_CTRL_START | SCE_CTRL_SELECT | SCE_CTRL_LTRIGGER | SCE_CTRL_RTRIGGER;
        u64 now = sceKernelGetProcessTimeWide();
        if ((c.buttons & combo) == combo) {
            if (!quit_since) quit_since = now;
            else if (now - quit_since > 1000000ull) {
                ps2_log("vita: quit combo held, exiting");
                fflush(stderr);
                sceKernelExitProcess(0);
            }
        } else quit_since = 0;
    }
}

static int display_main(SceSize args, void *argp) {
    (void)args; (void)argp;
    int cur = 0;
    u64 last_report = sceKernelGetProcessTimeWide(), last_fields = 0;
    unsigned frames = 0;

    vita_thread_register("display", (const void *)display_main);
    sceCtrlSetSamplingMode(SCE_CTRL_MODE_ANALOG_WIDE);
    sceTouchSetSamplingState(SCE_TOUCH_PORT_BACK, SCE_TOUCH_SAMPLING_STATE_START);

    /* The runtime decides on video before it enters the guest, so once
     * fields are flowing we know whether the GPU renderer owns the screen.
     * Only without it does this thread draw the software GS picture. */
    while (ps2_kernel_vblank_count() == 0) sceKernelDelayThread(10000);
    {   extern void vita_pin_named(const char *name, int core, int prio);
        vita_pin_named("ee-main", 0, 150);   /* EE on core 0, above guest helpers */
    }
    int software = !ps2_vk_enabled();
    ps2_log("vita display: %s", software ? "software GS picture (no GPU renderer)"
                                         : "GPU renderer owns the screen");
    if (software && alloc_screens() != 0) return 0;

    for (;;) {
        /* The pad HLE is ready once the kernel delivers fields. */
        poll_pad();
        if (software) {
            log_display_regs();
            if (want_vram_dump) { want_vram_dump = 0; dump_vram(); }
            blit(screens[cur]);

            SceDisplayFrameBuf f;
            memset(&f, 0, sizeof f);
            f.size = sizeof f;
            f.base = screens[cur];
            f.pitch = SCR_W;
            f.pixelformat = SCE_DISPLAY_PIXELFORMAT_A8B8G8R8;
            f.width = SCR_W;
            f.height = SCR_H;
            sceDisplaySetFrameBuf(&f, SCE_DISPLAY_SETBUF_NEXTFRAME);
        }
        sceDisplayWaitVblankStart();
        {   extern volatile u32 vita_display_ticks; vita_display_ticks++; }
        cur ^= 1;
        frames++;

        u64 now = sceKernelGetProcessTimeWide();
        if (now - last_report >= 5000000ull) {
            u64 fields = ps2_kernel_vblank_count();
            double secs = (double)(now - last_report) / 1e6;
            u64 prims, pixels, regs;
            static u64 last_prims, last_pixels;
            ps2_gs_stats(&prims, &pixels, &regs);
            ps2_log("vita: %.1f game fields/s (60 = full speed), %.1f screen fps, "
                    "GS %.0f prims/s %.2f Mpixels/s",
                    (double)(fields - last_fields) / secs, frames / secs,
                    (double)(prims - last_prims) / secs,
                    (double)(pixels - last_pixels) / secs / 1e6);
            last_prims = prims;
            last_pixels = pixels;
            if (!software) {
                static u64 lf, ld, lu;
                u64 f2, d2, u2;
                int nv;
                vita_gl_interval(&f2, &d2, &u2, &nv);
                ps2_log("vita gl: %.1f frames/s, %.0f draws/frame, %.1f texture "
                        "uploads/s, %d shader variants",
                        (double)(f2 - lf) / secs,
                        f2 > lf ? (double)(d2 - ld) / (double)(f2 - lf) : 0.0,
                        (double)(u2 - lu) / secs, nv);
                lf = f2; ld = d2; lu = u2;
            }
            {   /* heap and system memory, to find what is eating it */
                struct mallinfo mi = mallinfo();
                SceKernelFreeMemorySizeInfo fm;
                memset(&fm, 0, sizeof fm);
                fm.size = sizeof fm;
                sceKernelGetFreeMemorySize(&fm);
                ps2_log("vita mem: heap %.1f MB used, %.1f MB free in arena of %.1f MB; "
                        "system free main %d MB, CDRAM %d MB",
                        mi.uordblks / 1048576.0, mi.fordblks / 1048576.0,
                        mi.arena / 1048576.0, fm.size_user >> 20, fm.size_cdram >> 20);
                if (!software) {
                    char gm[480];
                    vita_gl_mem(gm, sizeof gm);
                    ps2_log("vita mem: %s", gm);
                    {   extern void vita_gl_detail(char *buf, size_t n);
                        char gd[3000];
                        vita_gl_detail(gd, sizeof gd);
                        ps2_log("vita gl: %s", gd);
                        {   extern void vita_gl_timing(char *buf, size_t n);
                            vita_gl_timing(gd, sizeof gd);
                            ps2_log("vita gl: %s", gd);
                        }
                    }
                }
            }
            vita_threads_report();
            {   extern void ps2_sampler_window(unsigned top);
                ps2_sampler_window(10);
                {   extern u64 gs_palc_hits, gs_palc_misses;
                    static u64 lph, lpm;
                    if (gs_palc_hits != lph || gs_palc_misses != lpm)
                        ps2_log("gs: palettes reused %llu, worked out %llu",
                                (unsigned long long)(gs_palc_hits - lph),
                                (unsigned long long)(gs_palc_misses - lpm));
                    lph = gs_palc_hits; lpm = gs_palc_misses;
                }
                {   extern u64 ps2_gs_rawc_bytes;
                    static u64 lbytes;
                    if (ps2_gs_rawc_bytes != lbytes)
                        ps2_log("gs: hashed %.1f MB of VRAM to recognise textures (%.2f MB/field)",
                                (double)(ps2_gs_rawc_bytes - lbytes) / 1048576.0,
                                (double)(ps2_gs_rawc_bytes - lbytes) / 1048576.0
                                / ((double)(fields - last_fields) > 0 ? (double)(fields - last_fields) : 1.0));
                    lbytes = ps2_gs_rawc_bytes;
                }
                {   extern u64 ps2_gs_rawc_hits, ps2_gs_rawc_misses;
                    static u64 lh, lmi;
                    u64 h2 = ps2_gs_rawc_hits, m2 = ps2_gs_rawc_misses;
                    if (h2 != lh || m2 != lmi)
                        ps2_log("gs: re-uploaded textures reused without decoding %llu, decoded %llu",
                                (unsigned long long)(h2 - lh), (unsigned long long)(m2 - lmi));
                    lh = h2; lmi = m2;
                }
                {   extern u64 ps2_gs_upc_hits, ps2_gs_upc_misses, ps2_gs_upc_none;
                    static u64 a0, b0, c0;
                    if (ps2_gs_upc_hits != a0 || ps2_gs_upc_misses != b0 || ps2_gs_upc_none != c0)
                    {   extern void ps2_gsq_report(char *buf, int n);
                        char gq[400];
                        ps2_gsq_report(gq, sizeof gq);
                        if (gq[0]) ps2_log("%s", gq);
                    }
                    {   extern u64 ps2_gs_upid_pick, ps2_gs_upid_blocks;
                        static u64 p0, q0;
                        ps2_log("gs: textures named by their uploads %llu, new %llu, not coverable %llu"
                                " (named from their own upload %llu, from the uploads owning their blocks %llu)",
                                (unsigned long long)(ps2_gs_upc_hits - a0),
                                (unsigned long long)(ps2_gs_upc_misses - b0),
                                (unsigned long long)(ps2_gs_upc_none - c0),
                                (unsigned long long)(ps2_gs_upid_pick - p0),
                                (unsigned long long)(ps2_gs_upid_blocks - q0));
                        p0 = ps2_gs_upid_pick; q0 = ps2_gs_upid_blocks;
                    }
                    a0 = ps2_gs_upc_hits; b0 = ps2_gs_upc_misses; c0 = ps2_gs_upc_none;
                }
                {   extern u64 vita_mc_calls[4], vita_mc_fallbacks[4];
                    static u64 lc[4], lf[4];
                    u64 c = 0, f = 0;
                    for (int k = 0; k < 4; k++) { c += vita_mc_calls[k] - lc[k]; f += vita_mc_fallbacks[k] - lf[k];
                                                  lc[k] = vita_mc_calls[k]; lf[k] = vita_mc_fallbacks[k]; }
                    if (c || f)
                        ps2_log("mpeg: native motion-compensation loops ran %llu times (%llu fell back to the recompiled code)",
                                (unsigned long long)c, (unsigned long long)f);
                }
                {   extern u64 ps2_gs_trx_unchanged, ps2_gs_trx_skipped_px;
                    static u64 lu, lp;
                    if (ps2_gs_trx_unchanged != lu)
                        ps2_log("gs: %llu image uploads changed nothing and were not re-processed"
                                " (%.1f Mpixels of writes skipped)",
                                (unsigned long long)(ps2_gs_trx_unchanged - lu),
                                (double)(ps2_gs_trx_skipped_px - lp) / 1e6);
                    lu = ps2_gs_trx_unchanged; lp = ps2_gs_trx_skipped_px;
                }
                {   extern u64 ps2_ipu_busy_us, ps2_ipu_mb_count;
                    static u64 lb, lm;
                    u64 b2 = ps2_ipu_busy_us, m2 = ps2_ipu_mb_count;
                    extern u64 ps2_ipu_blocks, ps2_ipu_flat_blocks;
                    static u64 lbk, lfb;
                    if (m2 != lm)
                        ps2_log("ipu: %.0f ms/s decoding on the EE thread, %.0f macroblocks/s (%.1f us each); %.0f%% of blocks flat",
                                (double)(b2 - lb) / 1000.0 / secs, (double)(m2 - lm) / secs,
                                (double)(b2 - lb) / (double)(m2 - lm),
                                ps2_ipu_blocks != lbk ? 100.0 * (double)(ps2_ipu_flat_blocks - lfb)
                                                        / (double)(ps2_ipu_blocks - lbk) : 0.0);
                    lbk = ps2_ipu_blocks; lfb = ps2_ipu_flat_blocks;
                    lb = b2; lm = m2;
                }
                {   extern u64 ps2_ipu_cmd_us[16], ps2_ipu_cmd_n[16], ps2_ipu_retries;
                    static u64 lu[16], ln[16], lr;
                    static const char *nm[10] = { "BCLR", "IDEC", "BDEC", "VDEC", "FDEC",
                                                  "SETIQ", "SETVQ", "CSC", "PACK", "SETTH" };
                    char line[400];
                    int n = 0, k;
                    for (k = 0; k < 10; k++) {
                        u64 du = ps2_ipu_cmd_us[k] - lu[k], dn = ps2_ipu_cmd_n[k] - ln[k];
                        if (dn && n < (int)sizeof line - 48)
                            n += snprintf(line + n, sizeof line - (size_t)n, " %s %.0fms/%llu",
                                          nm[k], (double)du / 1000.0, (unsigned long long)dn);
                        lu[k] = ps2_ipu_cmd_us[k]; ln[k] = ps2_ipu_cmd_n[k];
                    }
                    if (n) {
                        extern u64 ps2_ipu_csc_helper_mbs, ps2_ipu_csc_mbs;
                        static u64 lhm, lcm;
                        ps2_log("ipu per command (5 s, time/runs incl. retries):%s | %llu retries"
                                " | colour conversion: helper core did %llu of %llu macroblocks",
                                line, (unsigned long long)(ps2_ipu_retries - lr),
                                (unsigned long long)(ps2_ipu_csc_helper_mbs - lhm),
                                (unsigned long long)(ps2_ipu_csc_mbs - lcm));
                        lhm = ps2_ipu_csc_helper_mbs; lcm = ps2_ipu_csc_mbs;
                    }
                    lr = ps2_ipu_retries;
                }
                {   extern void ps2_vu_recomp_window(void); ps2_vu_recomp_window(); }
                extern void ps2_phase_window(u64 fields);
                ps2_phase_window(ps2_kernel_vblank_count());
            }
            last_fields = fields;
            last_report = now;
            frames = 0;
        }
    }
    return 0;
}

void vita_display_start(void) {
    /* Lower priority than the game threads and pinned to the third core, so
     * the copy only uses time the game doesn't. */
    SceUID th = sceKernelCreateThread("ac5_display", display_main, 120,
                                      0x10000, 0, SCE_KERNEL_CPU_MASK_USER_2, NULL);
    if (th < 0) {
        fprintf(stderr, "vita display: cannot create thread (%08X)\n", th);
        return;
    }
    sceKernelStartThread(th, 0, NULL);
}
