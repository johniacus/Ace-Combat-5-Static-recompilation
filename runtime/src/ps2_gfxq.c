#include "ps2_gfxq.h"

#ifdef PS2_VITA
/* ARM has no rdtsc here: time the phase profiler with the process clock. */
u64 vita_tsc(void);
#undef ps2_tsc
#define ps2_tsc() vita_tsc()
#endif

#include "rn.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <time.h>
#ifdef PS2_VITA
#include <psp2/kernel/threadmgr.h>
#endif
#include "ps2_gsq.h"

void ps2_vif_transfer(int which, const ps2_reg128 *q, u32 qwc);
void ps2_vif_write(int which, u32 word);
void ps2_vif_clear_stall(int which);
void ps2_vif_fbrst(int which, u32 val);
void ps2_gif_transfer(const ps2_reg128 *data, u32 qwc);
void ps2_gs_priv_write_now(u32 addr, u64 val);
void ps2_gif_ctrl_write(u32 addr, u32 val);
int  ps2_gs_present_now(void);
void ps2_request_exit(void);

int ps2_gfxq_on;

enum {
    GQ_WRAP = 0,
    GQ_VIF_QW,
    GQ_GIF_QW,
    GQ_VIF_W,
    GQ_FBRST,
    GQ_VIFCLR,
    GQ_GS_PRIV,
    GQ_MMIO,
    GQ_FIELD,
    GQ_TAG,
    GQ_INTENT
};

static u8 *q_buf;
static size_t q_cap;
static size_t q_head, q_tail;
static int q_busy;
static int q_running;
static pthread_t q_thread;
static _Thread_local int q_is_worker;
static pthread_mutex_t q_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t q_not_empty = PTHREAD_COND_INITIALIZER;
static pthread_cond_t q_not_full  = PTHREAD_COND_INITIALIZER;
static pthread_cond_t q_idle      = PTHREAD_COND_INITIALIZER;

static u64 st_records, st_bytes, st_producer_waits, st_drains, st_drain_waits;
static size_t st_peak_fill;
static int q_fields_pending;
static int q_field_depth = 2;
static u64 st_field_waits;
static u64 st_field_wait_ns;

static u64 gq_now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (u64)ts.tv_sec * 1000000000ull + (u64)ts.tv_nsec;
}
static u64 st_producer_ns, st_drain_ns;

static u64 speed_present_ns, speed_present_n, speed_work_ns;
static u64 speed_vu_runs, speed_vu_insns, speed_vu_guards;
void ps2_vu1_speed_counters(u64 *, u64 *, u64 *);
int ps2_speed_enabled(void) {
    static int enabled = -1;
    int value = __atomic_load_n(&enabled, __ATOMIC_RELAXED);
    if (value < 0) {
        const char *e = getenv("PS2_SPEED_LOG"); value = e && atoi(e) != 0;
        __atomic_store_n(&enabled, value, __ATOMIC_RELAXED);
    }
    return value;
}
void ps2_scene_speed_report(double dt);
void ps2_speed_sample(u64 fields, u64 dropped) {
    static u64 prev_ns, prev_fields, prev_dropped, prev_present, prev_n, prev_work;
    static u64 prev_field_wait, prev_ring_wait, prev_drain, prev_bytes;
    static u64 prev_vu_runs, prev_vu_insns, prev_vu_guards;
    if (!ps2_speed_enabled()) return;
    u64 now = gq_now_ns();
    if (!prev_ns) {
        ps2_log("speed: diagnostics v1; 5s intervals, scene dispatch is not simulation FPS; collision/VIF fixes 2026-09-06");
        ps2_log("speed: EE/present/worker ms are interval totals; present_ms is GS display-list handoff, not Vulkan GPU time. See vk-speed for renderer host stages.");
        prev_ns = now; prev_fields = fields; prev_dropped = dropped;
        prev_field_wait=st_field_wait_ns; prev_ring_wait=st_producer_ns;
        prev_drain=st_drain_ns; prev_bytes=st_bytes;
        pthread_mutex_lock(&q_lock);
        prev_present=speed_present_ns; prev_n=speed_present_n; prev_work=speed_work_ns;
        prev_vu_runs=speed_vu_runs; prev_vu_insns=speed_vu_insns; prev_vu_guards=speed_vu_guards;
        pthread_mutex_unlock(&q_lock);
        ps2_scene_speed_report(0);
        return;
    }
    if (now - prev_ns < 5000000000ull) return;
    double dt = (now - prev_ns) / 1e9;
    pthread_mutex_lock(&q_lock);
    u64 pn = speed_present_n, pt = speed_present_ns, work = speed_work_ns;
    u64 vr=speed_vu_runs, vi=speed_vu_insns, vg=speed_vu_guards;
    size_t fill = q_head - q_tail;
    int pending = q_fields_pending;
    pthread_mutex_unlock(&q_lock);
    ps2_log("speed: dt=%.2fs fields=%.2f/s dropped=%llu gfxq=%d pending=%d ring_KB=%lu queued_MB=%.2f",
        dt, (fields-prev_fields)/dt, (unsigned long long)(dropped-prev_dropped),
        ps2_gfxq_on, pending, (unsigned long)(fill >> 10), (st_bytes-prev_bytes)/1048576.0);
    ps2_log("speed: EE wait_ms field=%.2f ring=%.2f drain=%.2f; present=%.2f/s present_ms=%.2f worker_active_ms=%.2f",
        (st_field_wait_ns-prev_field_wait)/1e6, (st_producer_ns-prev_ring_wait)/1e6,
        (st_drain_ns-prev_drain)/1e6, (pn-prev_n)/dt, (pt-prev_present)/1e6, (work-prev_work)/1e6);
    ps2_scene_speed_report(dt);
    ps2_log("speed: VU1 runs=%.2f/s instructions_M=%.3f guard_hits=%llu",
        (vr-prev_vu_runs)/dt, (vi-prev_vu_insns)/1e6, (unsigned long long)(vg-prev_vu_guards));
    prev_vu_runs=vr; prev_vu_insns=vi; prev_vu_guards=vg;
    prev_ns=now; prev_fields=fields; prev_dropped=dropped; prev_present=pt; prev_n=pn;
    prev_field_wait=st_field_wait_ns; prev_ring_wait=st_producer_ns; prev_drain=st_drain_ns;
    prev_bytes=st_bytes;
    prev_work=work;
}

static size_t q_producer_tail, q_reserved_end;
static int q_wake_sent;     /* as g_wake_sent: one signal per idle spell of the worker */
static u8 *q_reserve(u32 n) {
    size_t off = q_head & (q_cap - 1);
    size_t pad = off + n > q_cap ? q_cap - off : 0;
    size_t need = pad + n;
    if (q_cap - (q_head - q_producer_tail) < need) {
        pthread_mutex_lock(&q_lock);
        q_producer_tail = q_tail;
        while (q_cap - (q_head - q_producer_tail) < need) {
            u64 t0 = gq_now_ns();
            st_producer_waits++;
            pthread_cond_wait(&q_not_full, &q_lock);
            st_producer_ns += gq_now_ns() - t0;
            q_producer_tail = q_tail;
        }
        pthread_mutex_unlock(&q_lock);
    }
    q_reserved_end = q_head + need;
    if (pad) {
        q_buf[off] = GQ_WRAP;
        off = 0;
    }
    return q_buf + off;
}

static void q_commit(u32 n) {
    size_t fill;
    __atomic_store_n(&q_head, q_reserved_end, __ATOMIC_SEQ_CST);
    st_records++;
    st_bytes += n;
    fill = q_reserved_end - __atomic_load_n(&q_tail, __ATOMIC_RELAXED);
    if (fill > st_peak_fill) st_peak_fill = fill;
    if (!__atomic_load_n(&q_busy, __ATOMIC_SEQ_CST)
        && !__atomic_exchange_n(&q_wake_sent, 1, __ATOMIC_SEQ_CST)) {
        pthread_mutex_lock(&q_lock);
        pthread_cond_signal(&q_not_empty);
        pthread_mutex_unlock(&q_lock);
    }
}

static inline void put8(u8 *p, u32 off, u8 v)   { p[off] = v; }
static inline void put32(u8 *p, u32 off, u32 v) { memcpy(p + off, &v, 4); }
static inline void put64(u8 *p, u32 off, u64 v) { memcpy(p + off, &v, 8); }
static inline u32  get32(const u8 *p, u32 off)  { u32 v; memcpy(&v, p + off, 4); return v; }
static inline u64  get64(const u8 *p, u32 off)  { u64 v; memcpy(&v, p + off, 8); return v; }

static u32 q_apply(const u8 *r) {
    switch (r[0]) {
    case GQ_VIF_QW: {
        int ch = (int)r[1];
        u32 qwc = get32(r, 2);
        const ps2_reg128 *qw = (const ps2_reg128 *)(r + 6);
        ps2_vif_transfer(ch, qw, qwc);
        return 6u + qwc * 16u;
    }
    case GQ_GIF_QW: {
        u32 qwc = get32(r, 1);
        ps2_gif_transfer((const ps2_reg128 *)(r + 5), qwc);
        return 5u + qwc * 16u;
    }
    case GQ_VIF_W:   ps2_vif_write((int)r[1], get32(r, 2));        return 6u;
    case GQ_FBRST:   ps2_vif_fbrst((int)r[1], get32(r, 2));        return 6u;
    case GQ_VIFCLR:  ps2_vif_clear_stall((int)r[1]);               return 2u;
    case GQ_GS_PRIV:
        if (ps2_gsq_on) ps2_gsq_priv(get32(r, 1), get64(r, 5));
        else ps2_gs_priv_write_now(get32(r, 1), get64(r, 5));
        return 13u;
    case GQ_MMIO:    ps2_gif_ctrl_write(get32(r, 1), get32(r, 5)); return 9u;
    case GQ_TAG:
        if (ps2_gsq_on) ps2_gsq_tag(get32(r, 1), get32(r, 5), get32(r, 9));
        else rn_gs_tag(get32(r, 1), get32(r, 5), get32(r, 9));
        return 13u;
    case GQ_INTENT: {
        u32 len = get32(r, 1);
        if (ps2_gsq_on) ps2_gsq_intent(r + 5, len);
        else rn_gs_intent(r + 5, len);
        return 5u + len;
    }
    case GQ_FIELD:
        {   static u32 nf;
            extern void rn_vp_timing_report(u32 fields);
            if (++nf % 300u == 0u) rn_vp_timing_report(300u);
        }
        ps2_phase_field_tick();
        {   extern void ps2_vu_census_tick(void); ps2_vu_census_tick(); }
        if (ps2_gsq_on) ps2_gsq_field();      /* the GS thread presents and counts it done */
        else if (!ps2_gs_present_now()) ps2_request_exit();
        return 1u;
    default:
        ps2_log("gfxq: corrupt record opcode %u -- stopping the worker", r[0]);
        return 0u;
    }
}

#ifdef PS2_VITA
void vita_pin_current(int core);
#endif
/* Slow-field finder: time per record kind within a field; a field that
 * takes over 50 ms is logged with its breakdown and the biggest record. */
static u64 fp_ns[16], fp_n[16], fp_field0, fp_big_ns, fp_shown_ns;
static u32 fp_big_op, fp_big_len, fp_logged;
static u64 fp_run0, fp_runn0, fp_st0, fp_stn0, fp_tex0, fp_texn0;
static u64 fp_fq0, fp_sw0, fp_cw0, fp_dq0, fp_ff0, fp_vif0, fp_tl[4], fp_te[5], fp_dx[7], fp_px[6];
static void fp_field_end(u64 now) {
    u64 tot = now - fp_field0;
    if (fp_field0 && tot > 50000000ull && (fp_logged < 40u || now - fp_shown_ns > 10000000000ull)) {
        static const char *nm[] = { "wrap", "VIF", "GIF", "VIFw", "FBRST", "VIFCLR", "GSpriv",
                                    "MMIO", "field", "tag", "intent" };
        char b[2800];
        int o = snprintf(b, sizeof b, "gfxq: slow field %.1f ms:", (double)tot / 1e6);
        for (int k = 0; k < 11 && o < (int)sizeof b; k++)
            if (fp_n[k]) o += snprintf(b + o, sizeof b - (size_t)o, " %s %llux %.1f ms", nm[k],
                                       (unsigned long long)fp_n[k], (double)fp_ns[k] / 1e6);
        if (o < (int)sizeof b)
            o += snprintf(b + o, sizeof b - (size_t)o, " | biggest one: %s %.1f ms (%u bytes)",
                          fp_big_op < 11u ? nm[fp_big_op] : "?", (double)fp_big_ns / 1e6, fp_big_len);
        {   extern u64 ps2_vt_run_us[2], ps2_vt_run_n[2], rn_tq_state_us, rn_tq_state_n,
                       ps2_gs_tex_us, ps2_gs_tex_n;
            if (o < (int)sizeof b)
                o += snprintf(b + o, sizeof b - (size_t)o,
                              " | in it: VU1 runs %.1f ms/%llu, native GS state %.1f ms/%llu, texture lookups %.1f ms/%llu, programs:",
                              (ps2_vt_run_us[1] - fp_run0) / 1000.0, (unsigned long long)(ps2_vt_run_n[1] - fp_runn0),
                              (rn_tq_state_us - fp_st0) / 1000.0, (unsigned long long)(rn_tq_state_n - fp_stn0),
                              (ps2_gs_tex_us - fp_tex0) / 1000.0, (unsigned long long)(ps2_gs_tex_n - fp_texn0));
        }
        {   extern int rn_vp_field_breakdown(char *b, int n);
            if (o < (int)sizeof b) o += rn_vp_field_breakdown(b + o, (int)sizeof b - o);
        }
        {   extern u64 ps2_vt_direct_us, ps2_gs_img_us, ps2_gs_img_fast_qw, ps2_gs_img_slow_qw, ps2_gs_trxbeg_us,
                       ps2_gs_img_calls, ps2_gs_dirty_us;
            if (o < (int)sizeof b)
                o += snprintf(b + o, sizeof b - (size_t)o,
                              " | direct-to-GS %.1f ms, of it image data %.1f ms (fast qw %llu, slow qw %llu, in %llu pieces, dirty marking %.1f ms), transfer starts %.1f ms",
                              (ps2_vt_direct_us - fp_dx[0]) / 1000.0, (ps2_gs_img_us - fp_dx[1]) / 1000.0,
                              (unsigned long long)(ps2_gs_img_fast_qw - fp_dx[2]),
                              (unsigned long long)(ps2_gs_img_slow_qw - fp_dx[3]),
                              (unsigned long long)(ps2_gs_img_calls - fp_dx[5]), (ps2_gs_dirty_us - fp_dx[6]) / 1000.0,
                              (ps2_gs_trxbeg_us - fp_dx[4]) / 1000.0);
        }
        {   extern u64 ps2_vc_slow_fmt[64], ps2_vc_code_cmd[128];
            static u64 sf[64], cc[128];
            {   extern u64 ps2_vc_elem_words;
                static u64 ew;
                if (o < (int)sizeof b) o += snprintf(b + o, sizeof b - (size_t)o, " | whole-element unpacks %llu words",
                                                     (unsigned long long)(ps2_vc_elem_words - ew));
                ew = ps2_vc_elem_words;
            }
            if (o < (int)sizeof b) o += snprintf(b + o, sizeof b - (size_t)o, " | word-by-word unpacks (vn vl mask mode):");
            for (int k = 0; k < 64; k++) {
                u64 d = ps2_vc_slow_fmt[k] - sf[k];
                if (d >= 500 && o < (int)sizeof b)
                    o += snprintf(b + o, sizeof b - (size_t)o, " V%d-%d%s%s x%llu", (k & 3) + 1,
                                  32 >> ((k >> 2) & 3), (k & 16) ? " m" : "", (k & 32) ? " mode" : "",
                                  (unsigned long long)d);
                sf[k] = ps2_vc_slow_fmt[k];
            }
            if (o < (int)sizeof b) o += snprintf(b + o, sizeof b - (size_t)o, " | VIFcodes:");
            for (int k = 0; k < 128; k++) {
                u64 d = ps2_vc_code_cmd[k] - cc[k];
                if (d >= 300 && o < (int)sizeof b)
                    o += snprintf(b + o, sizeof b - (size_t)o, " %02X x%llu", k, (unsigned long long)d);
                cc[k] = ps2_vc_code_cmd[k];
            }
        }
        {   extern u64 ps2_gs_dec_psm[64];
            static u64 seen[64];
            if (o < (int)sizeof b) o += snprintf(b + o, sizeof b - (size_t)o, " | decoded here by format:");
            for (int k = 0; k < 64; k++) {
                if (ps2_gs_dec_psm[k] != seen[k] && o < (int)sizeof b)
                    o += snprintf(b + o, sizeof b - (size_t)o, " psm%d x%llu", k,
                                  (unsigned long long)(ps2_gs_dec_psm[k] - seen[k]));
                seen[k] = ps2_gs_dec_psm[k];
            }
        }
        {   extern u64 ps2_gs_img_px[4], ps2_gs_img_cmp_px, ps2_gs_img_skip_px;
            u64 *px0 = fp_px, c0 = fp_px[4], k0 = fp_px[5];
            if (o < (int)sizeof b)
                o += snprintf(b + o, sizeof b - (size_t)o,
                              " | image pixels 32-bit %llu 16-bit %llu 8-bit %llu 4-bit %llu, compared %llu, skipped unchanged %llu",
                              (unsigned long long)(ps2_gs_img_px[0] - px0[0]), (unsigned long long)(ps2_gs_img_px[1] - px0[1]),
                              (unsigned long long)(ps2_gs_img_px[2] - px0[2]), (unsigned long long)(ps2_gs_img_px[3] - px0[3]),
                              (unsigned long long)(ps2_gs_img_cmp_px - c0), (unsigned long long)(ps2_gs_img_skip_px - k0));
        }
        {   extern u64 ps2_tl_us[4], ps2_tl_end[5];
            if (o < (int)sizeof b)
                o += snprintf(b + o, sizeof b - (size_t)o,
                              " | texture lookup split: cache probe %.1f ms (%llu ended there), upload naming %.1f ms (%llu),"
                              " page hashing %.1f ms (%llu), decoding %.1f ms (%llu)",
                              (ps2_tl_us[0] - fp_tl[0]) / 1000.0, (unsigned long long)(ps2_tl_end[0] - fp_te[0]),
                              (ps2_tl_us[1] - fp_tl[1]) / 1000.0, (unsigned long long)(ps2_tl_end[1] - fp_te[1]),
                              (ps2_tl_us[2] - fp_tl[2]) / 1000.0, (unsigned long long)(ps2_tl_end[2] - fp_te[2]),
                              (ps2_tl_us[3] - fp_tl[3]) / 1000.0, (unsigned long long)(ps2_tl_end[3] - fp_te[3]));
        }
        {   extern u64 ps2_vif_fast_qw, ps2_vc_slow_words, ps2_vc_vifcode_words, ps2_vc_direct_qw,
                       ps2_vc_fifo_qw, ps2_vt_vif_us[2];
            if (o < (int)sizeof b)
                o += snprintf(b + o, sizeof b - (size_t)o,
                              " | VIF1 %.1f ms; unpacked qwords fast %llu, word-by-word words %llu; VIFcode words %llu;"
                              " direct-to-GS qwords %llu; FIFO qwords %llu",
                              (ps2_vt_vif_us[1] - fp_vif0) / 1000.0,
                              (unsigned long long)(ps2_vif_fast_qw - fp_fq0),
                              (unsigned long long)(ps2_vc_slow_words - fp_sw0),
                              (unsigned long long)(ps2_vc_vifcode_words - fp_cw0),
                              (unsigned long long)(ps2_vc_direct_qw - fp_dq0),
                              (unsigned long long)(ps2_vc_fifo_qw - fp_ff0));
        }
        ps2_log("%s", b);
        fp_logged++;
        fp_shown_ns = now;
    }
    else {
        extern int rn_vp_field_breakdown(char *b, int n);
        char tmp[8];
        rn_vp_field_breakdown(tmp, 0);
    }
    memset(fp_ns, 0, sizeof fp_ns); memset(fp_n, 0, sizeof fp_n);
    fp_big_ns = 0; fp_big_op = 0; fp_big_len = 0;
    fp_field0 = now;
    {   extern u64 ps2_vt_run_us[2], ps2_vt_run_n[2], rn_tq_state_us, rn_tq_state_n,
                   ps2_gs_tex_us, ps2_gs_tex_n;
        fp_run0 = ps2_vt_run_us[1]; fp_runn0 = ps2_vt_run_n[1];
        fp_st0 = rn_tq_state_us; fp_stn0 = rn_tq_state_n;
        fp_tex0 = ps2_gs_tex_us; fp_texn0 = ps2_gs_tex_n;
    }
    {   extern u64 ps2_vif_fast_qw, ps2_vc_slow_words, ps2_vc_vifcode_words, ps2_vc_direct_qw,
                   ps2_vc_fifo_qw, ps2_vt_vif_us[2];
        fp_fq0 = ps2_vif_fast_qw; fp_sw0 = ps2_vc_slow_words; fp_cw0 = ps2_vc_vifcode_words;
        fp_dq0 = ps2_vc_direct_qw; fp_ff0 = ps2_vc_fifo_qw; fp_vif0 = ps2_vt_vif_us[1];
    }
    {   extern u64 ps2_tl_us[4], ps2_tl_end[5];
        memcpy(fp_tl, ps2_tl_us, sizeof fp_tl); memcpy(fp_te, ps2_tl_end, sizeof fp_te);
    }
    {   extern u64 ps2_vt_direct_us, ps2_gs_img_us, ps2_gs_img_fast_qw, ps2_gs_img_slow_qw, ps2_gs_trxbeg_us;
        fp_dx[0] = ps2_vt_direct_us; fp_dx[1] = ps2_gs_img_us; fp_dx[2] = ps2_gs_img_fast_qw;
        fp_dx[3] = ps2_gs_img_slow_qw; fp_dx[4] = ps2_gs_trxbeg_us;
        {   extern u64 ps2_gs_img_calls, ps2_gs_dirty_us;
            fp_dx[5] = ps2_gs_img_calls; fp_dx[6] = ps2_gs_dirty_us; }
        {   extern u64 ps2_gs_img_px[4], ps2_gs_img_cmp_px, ps2_gs_img_skip_px;
            memcpy(fp_px, ps2_gs_img_px, 4 * sizeof(u64)); fp_px[4] = ps2_gs_img_cmp_px; fp_px[5] = ps2_gs_img_skip_px; }
    }
}

static void *q_main(void *unused) {
    (void)unused;
#ifdef PS2_VITA
    vita_pin_current(1);    /* VIF/VU1/GS work on its own core */
#endif
    q_is_worker = 1;
    ps2_gsq_producer = 1;
    ps2_gsq_worker_started();
    ps2_host_prof_attach("gfx");
    u64 speed_burst=0;
    int speed_on=ps2_speed_enabled();
    pthread_mutex_lock(&q_lock);
    for (;;) {
        while (q_running && __atomic_load_n(&q_head, __ATOMIC_SEQ_CST) == q_tail) {
            if (speed_burst) { speed_work_ns += gq_now_ns()-speed_burst; speed_burst=0; }
            __atomic_store_n(&q_busy, 0, __ATOMIC_SEQ_CST);
            __atomic_store_n(&q_wake_sent, 0, __ATOMIC_SEQ_CST);
            pthread_cond_broadcast(&q_idle);
            if (__atomic_load_n(&q_head, __ATOMIC_SEQ_CST) != q_tail) break;
            pthread_cond_wait(&q_not_empty, &q_lock);
        }
        if (!q_running && __atomic_load_n(&q_head, __ATOMIC_SEQ_CST) == q_tail) break;
        __atomic_store_n(&q_busy, 1, __ATOMIC_SEQ_CST);
        if (!speed_burst && speed_on) speed_burst=gq_now_ns();
        size_t tail = q_tail, head = __atomic_load_n(&q_head, __ATOMIC_ACQUIRE);
        unsigned records = 0;
        int field = 0, corrupt = 0;
        u64 speed_elapsed = 0;
        pthread_mutex_unlock(&q_lock);
        while (tail != head && records < 256) {
            const u8 *r = q_buf + (tail & (q_cap - 1));
            if (*r == GQ_WRAP) {
                tail += q_cap - (tail & (q_cap - 1));
                continue;
            }
            field = *r == GQ_FIELD;
            u64 start = field && speed_on ? gq_now_ns() : 0;
            u32 n;
            static int fprof = -1;
            if (fprof < 0) fprof = getenv("AC5_FIELD_PROF") ? 1 : 0;
            u64 fp0 = fprof ? gq_now_ns() : 0;
            u8 op = *r;
            { PS2_PHASE_BEGIN(PS2_PH_GFXQ);
              n = q_apply(r);
              PS2_PHASE_END(PS2_PH_GFXQ); }
            if (fprof) {
                u64 fp1 = gq_now_ns(), d = fp1 - fp0;
                if (op < 16) { fp_ns[op] += d; fp_n[op]++; }
                if (d > fp_big_ns) { fp_big_ns = d; fp_big_op = op; fp_big_len = n; }
                if (op == GQ_FIELD) fp_field_end(fp1);
            } else if (op == GQ_FIELD) {
                fp_field_end(gq_now_ns());   /* slow fields still logged, without the per-record split */
            }
            if (start) speed_elapsed = gq_now_ns() - start;
            if (!n) { corrupt = 1; break; }
            tail += n;
            records++;
            if (field) break;
        }
        pthread_mutex_lock(&q_lock);
        __atomic_store_n(&q_tail, tail, __ATOMIC_RELEASE);
        if (field && !corrupt) {
            if (speed_on) {
                speed_present_ns += speed_elapsed; speed_present_n++;
                ps2_vu1_speed_counters(&speed_vu_runs,&speed_vu_insns,&speed_vu_guards);
            }
            if (speed_burst) { u64 now=gq_now_ns(); speed_work_ns+=now-speed_burst; speed_burst=now; }
            if (!ps2_gsq_on) q_fields_pending--;
        }
        pthread_cond_broadcast(&q_not_full);
        if (corrupt) { q_running = 0; break; }
    }
    __atomic_store_n(&q_busy, 0, __ATOMIC_SEQ_CST);
    pthread_cond_broadcast(&q_idle);
    pthread_mutex_unlock(&q_lock);
    return NULL;
}

static const char *q_forbidden;
void ps2_gfxq_forbid(const char *why) { if (!q_forbidden) q_forbidden = why; }

void ps2_gfxq_init(void) {
    const char *e = getenv("PS2_ASYNC_GFX");
    size_t mb;
    if (e && *e == '0') {
        ps2_log("gfxq: PS2_ASYNC_GFX=0 -- the graphics machine runs on the EE "
                "thread, as it did before.");
        return;
    }

    if (q_forbidden) {
        ps2_log("gfxq: PS2_ASYNC_GFX is set but %s, so the graphics machine "
                "stays on the EE thread.", q_forbidden);
        return;
    }

    mb = 64;
    { const char *m = getenv("PS2_ASYNC_MB");
      if (m && *m) { double v = atof(m); if (v >= 1.0 && v <= 1024.0) mb = (size_t)v; } }
    { const char *d = getenv("PS2_ASYNC_FIELDS");
      if (d && *d) { int v = atoi(d); if (v >= 1 && v <= 8) q_field_depth = v; } }
    q_cap = 1;
    while (q_cap < mb * 1024u * 1024u) q_cap <<= 1;
    q_buf = (u8 *)malloc(q_cap);
    if (!q_buf) {
        ps2_log("gfxq: cannot allocate a %lu MB ring; staying synchronous",
                (unsigned long)(q_cap >> 20));
        return;
    }
    q_head = q_tail = q_producer_tail = q_reserved_end = 0;
    ps2_gsq_init();     /* before the worker starts producing for it */
    q_running = 1;
    if (pthread_create(&q_thread, NULL, q_main, NULL) != 0) {
        ps2_log("gfxq: cannot start the worker thread; staying synchronous");
        free(q_buf); q_buf = NULL; q_running = 0;
        return;
    }
    ps2_gfxq_on = 1;
    ps2_log("gfxq: the graphics machine (VIF, VU1, GIF, GS) runs on its own "
            "thread; %lu MB ring", (unsigned long)(q_cap >> 20));
}

void ps2_gfxq_drain(void) {
    u64 t0;
    if (!ps2_gfxq_on || q_is_worker) return;
    t0 = gq_now_ns();
    pthread_mutex_lock(&q_lock);
    st_drains++;
    if (q_head != q_tail || q_busy || q_fields_pending) {
        st_drain_waits++;
        pthread_cond_signal(&q_not_empty);
        while (q_running && (q_head != q_tail || q_busy || q_fields_pending))
            pthread_cond_wait(&q_idle, &q_lock);
    }
    pthread_mutex_unlock(&q_lock);
    ps2_gsq_drain();
    st_drain_ns += gq_now_ns() - t0;
}

void ps2_gfxq_shutdown(void) {
    if (!ps2_gfxq_on) return;
    ps2_gfxq_drain();
    pthread_mutex_lock(&q_lock);
    q_running = 0;
    pthread_cond_broadcast(&q_not_empty);
    pthread_mutex_unlock(&q_lock);
    pthread_join(q_thread, NULL);
    ps2_gfxq_on = 0;
}

int ps2_gfxq_blocks_capture(void) { return ps2_gfxq_on; }

void ps2_gfxq_report(void) {
    if (!st_records) return;
    ps2_log("gfxq: %llu records, %llu MB queued, peak fill %lu KB of %lu MB",
            (unsigned long long)st_records,
            (unsigned long long)(st_bytes >> 20),
            (unsigned long)(st_peak_fill >> 10), (unsigned long)(q_cap >> 20));
    ps2_log("gfxq: %llu drains, %llu of them actually waited (%.1f ms total); "
            "the EE waited on a full ring %llu times (%.1f ms total)",
            (unsigned long long)st_drains, (unsigned long long)st_drain_waits,
            (double)st_drain_ns / 1e6,
            (unsigned long long)st_producer_waits,
            (double)st_producer_ns / 1e6);
    ps2_log("gfxq: the EE waited for the graphics machine to catch up on "
            "fields %llu times (%.1f ms total), at a depth of %d",
            (unsigned long long)st_field_waits,
            (double)st_field_wait_ns / 1e6, q_field_depth);
    if (st_producer_waits)
        ps2_log("gfxq:   a non-zero full-ring count means the graphics machine "
                "is the constraint, not the EE -- a bigger ring will not help.");
}

void ps2_gfxq_vif_qw(int ch, const ps2_reg128 *qw, u32 qwc) {
    if (!ps2_gfxq_on) {
        ps2_vif_transfer(ch, qw, qwc);
        return;
    }
    if (!qwc) return;
    {
        u32 n = 6u + qwc * 16u;
        u8 *r = q_reserve(n);
        put8(r, 0, GQ_VIF_QW);
        put8(r, 1, (u8)ch);
        put32(r, 2, qwc);
        memcpy(r + 6, qw, (size_t)qwc * 16u);
        q_commit(n);
    }
}

void ps2_gfxq_gif_qw(const ps2_reg128 *qw, u32 qwc) {
    if (!ps2_gfxq_on) { ps2_gif_transfer(qw, qwc); return; }
    if (!qwc) return;
    {
        u32 n = 5u + qwc * 16u;
        u8 *r = q_reserve(n);
        put8(r, 0, GQ_GIF_QW);
        put32(r, 1, qwc);
        memcpy(r + 5, qw, (size_t)qwc * 16u);
        q_commit(n);
    }
}

void ps2_gfxq_vif_word(int ch, u32 word) {
    if (!ps2_gfxq_on) { ps2_vif_write(ch, word); return; }
    { u8 *r = q_reserve(6);
      put8(r, 0, GQ_VIF_W); put8(r, 1, (u8)ch); put32(r, 2, word);
      q_commit(6); }
}

void ps2_gfxq_fbrst(int ch, u32 val) {
    if (!ps2_gfxq_on) { ps2_vif_fbrst(ch, val); return; }
    { u8 *r = q_reserve(6);
      put8(r, 0, GQ_FBRST); put8(r, 1, (u8)ch); put32(r, 2, val);
      q_commit(6); }
}

void ps2_gfxq_vif_clear(int ch) {
    if (!ps2_gfxq_on) { ps2_vif_clear_stall(ch); return; }
    { u8 *r = q_reserve(2);
      put8(r, 0, GQ_VIFCLR); put8(r, 1, (u8)ch);
      q_commit(2); }
}

void ps2_gfxq_gs_priv(u32 addr, u64 val) {
    if (!ps2_gfxq_on) { ps2_gs_priv_write_now(addr, val); return; }
    { u8 *r = q_reserve(13);
      put8(r, 0, GQ_GS_PRIV); put32(r, 1, addr); put64(r, 5, val);
      q_commit(13); }
}

int ps2_gfxq_field(void) {
    if (!ps2_gfxq_on) {
        u64 start = ps2_speed_enabled() ? gq_now_ns() : 0;
        int result = ps2_gs_present_now();
        if (start) {
            speed_present_ns += gq_now_ns()-start; speed_present_n++;
            ps2_vu1_speed_counters(&speed_vu_runs,&speed_vu_insns,&speed_vu_guards);
        }
        return result;
    }
    pthread_mutex_lock(&q_lock);
    if (q_fields_pending >= q_field_depth) {
        u64 t0 = gq_now_ns();
        st_field_waits++;
        while (q_running && q_fields_pending >= q_field_depth)
            pthread_cond_wait(&q_not_full, &q_lock);
        st_field_wait_ns += gq_now_ns() - t0;
    }
    q_fields_pending++;
    pthread_mutex_unlock(&q_lock);
    { u8 *r = q_reserve(1);
      put8(r, 0, GQ_FIELD);
      q_commit(1); }
    return 1;
}

void ps2_gfxq_mmio(u32 addr, u32 val) {
    if (!ps2_gfxq_on) { ps2_gif_ctrl_write(addr, val); return; }
    { u8 *r = q_reserve(9);
      put8(r, 0, GQ_MMIO); put32(r, 1, addr); put32(r, 5, val);
      q_commit(9); }
}

void ps2_gfxq_tag(u32 kind, u32 a, u32 b) {
    if (!ps2_gfxq_on) { rn_gs_tag(kind, a, b); return; }
    { u8 *r = q_reserve(13);
      put8(r, 0, GQ_TAG); put32(r, 1, kind); put32(r, 5, a); put32(r, 9, b);
      q_commit(13); }
}

void ps2_gfxq_intent(const u8 *rec, u32 len) {
    if (!ps2_gfxq_on) { rn_gs_intent(rec, len); return; }
    { u8 *r = q_reserve(5u + len);
      put8(r, 0, GQ_INTENT); put32(r, 1, len);
      memcpy(r + 5, rec, len);
      q_commit(5u + len); }
}

/* ===================================================================== */
/* The GS stage on its own thread (see ps2_gsq.h).                        */
/* ===================================================================== */
#include "ps2_gsq.h"
#include "ps2_vk.h"

void ps2_gs_write_reg(u32 reg, u64 val);
void ps2_gs_hwreg_pair(u64 lo, u64 hi);
void ps2_gs_hwreg_stream(const void *data, u32 qwc);
void ps2_gs_native_state(ps2_vk_state *st);
void ps2_gs_native_frame(float *xoff, float *yoff, float *zmax);
void ps2_gs_native_vertex(float x, float y, double z, float fog, int kind, int adc);

int ps2_gsq_on;
_Thread_local int ps2_gsq_producer;
uintptr_t ps2_gsq_worker_tp = (uintptr_t)-1;
static volatile uintptr_t gsq_gs_tp;
static volatile int gsq_gs_started;
void ps2_gsq_worker_started(void);
static _Thread_local int gsq_consumer;

enum { GS_WRAP = 0, GS_WREG, GS_HWPAIR, GS_HWSTREAM, GS_NSTATE, GS_NVERTEX, GS_DRAWMESH,
       GS_MSTORE, GS_PRIV, GS_TAG, GS_INTENT, GS_FIELD };

/* What the GS thread is doing, for a 1 kHz sampler on its core (build 89):
 * the record kind, or a finer stage set inside ps2_gs.c (20+), 31 idle.
 * Each slow field's log line lists the samples it collected, and how many
 * fell while the renderer (same core, higher priority) was drawing or
 * decoding, which is time the GS thread did not get. */
volatile u8 ps2_gs_ph;
static volatile u16 g_smp[64][4];   /* [phase][GS thread running / ready / waiting / other] */
#ifdef PS2_VITA
extern volatile int rn_busy_state;
static volatile SceUID g_gs_thid = -1;
/* On core 0 (build 97), so the GS thread's state is seen as it is: running,
 * ready but held off its core by another thread, or waiting in the kernel
 * (a lock, a semaphore, file I/O). */
static void *g_sampler(void *unused) {
    (void)unused;
    {   const char *c = getenv("AC5_SAMPLER_CORE"); vita_pin_current(c ? atoi(c) : 0); }
    {   extern void vita_set_priority(int prio); vita_set_priority(100); }
    for (;;) {
        sceKernelDelayThread(1000);
        {   u32 ph = ps2_gs_ph & 63u, st = 3u;
            if (g_gs_thid >= 0) {
                SceKernelThreadInfo ti;
                memset(&ti, 0, sizeof ti);
                ti.size = sizeof ti;
                if (sceKernelGetThreadInfo(g_gs_thid, &ti) >= 0)
                    st = (ti.status & SCE_THREAD_RUNNING) ? 0u : (ti.status & SCE_THREAD_READY) ? 1u
                       : (ti.status & SCE_THREAD_WAITING) ? 2u : 3u;
            }
            {   extern void rn_sample(void); rn_sample(); }
            if (g_smp[ph][st] < 60000u) g_smp[ph][st]++;
        }
    }
    return NULL;
}
#endif
static const char *g_ph_name(u32 k) {
    static const char *kn[] = { "-", "reg", "hwpair", "image", "nstate", "nvertex",
                                "meshdraw", "meshstore", "priv", "tag", "intent", "present" };
    if (k < 12) return kn[k];
    switch (k) {
    case 20: return "texlookup"; case 21: return "tex-hash"; case 22: return "tex-copy";
    case 23: return "tex-decode"; case 24: return "tex-store"; case 25: return "image-slow";
    case 26: return "trx-start"; case 31: return "idle";
    case 27: return "tex-palette"; case 28: return "tex-shadow"; case 29: return "tex-pagehash"; case 30: return "tex-rawc";
    case 42: return "upload-record"; case 43: return "dirty-mark"; case 44: return "trx-changed";
    case 45: return "trx-unchanged";
    case 50: return "img-entry"; case 51: return "img-skip-cmp"; case 52: return "img-compare";
    case 53: return "img-write"; case 54: return "img-record"; case 55: return "trx-record-start";
    case 56: return "trx-reupload"; case 57: return "img-runs";
    case 58: return "img-band"; case 49: return "img-start"; case 59: return "img-done"; case 60: return "present-gs"; case 61: return "present-vk";
    case 62: return "field-done-lock"; case 63: return "field-done-wake"; case 64: return "after-field";
    case 65: return "ring-lock"; case 66: return "list-lock"; case 67: return "publish";
    default: return "other";
    }
}

static u8 *g_buf;
static size_t g_cap, g_head, g_tail, g_ptail, g_rend;
static volatile int g_busy;
/* one wake-up per idle spell: the worker commits a record per GS register
 * write, and each one took the lock to signal a GS thread already waking */
static int g_wake_sent;
static int g_running;
static pthread_t g_thread;
static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_not_empty = PTHREAD_COND_INITIALIZER;
static pthread_cond_t g_not_full = PTHREAD_COND_INITIALIZER;
static pthread_cond_t g_idle = PTHREAD_COND_INITIALIZER;
static u64 gst_records, gst_bytes, gst_full_waits, gst_full_ns, gst_syncs, gst_sync_ns,
           gst_busy_ns, gst_fields, gst_token_miss, gst_slow, gst_store_fail;
static const char *gst_sync_why;
static u64 gst_sync_why_n;

/* producer state for the native draw sequence: native_state / native_frame
 * hand back placeholders and the draw that follows picks them up */
static int p_nstate, p_frame;
static u32 p_store_seq;

static u8 *g_reserve(u32 n) {
    size_t off = g_head & (g_cap - 1);
    size_t pad = off + n > g_cap ? g_cap - off : 0;
    size_t need = pad + n;
    if (g_cap - (g_head - g_ptail) < need) {
        pthread_mutex_lock(&g_lock);
        g_ptail = g_tail;
        while (g_cap - (g_head - g_ptail) < need) {
            u64 t0 = gq_now_ns();
            gst_full_waits++;
            pthread_cond_signal(&g_not_empty);
            pthread_cond_wait(&g_not_full, &g_lock);
            gst_full_ns += gq_now_ns() - t0;
            g_ptail = g_tail;
        }
        pthread_mutex_unlock(&g_lock);
    }
    g_rend = g_head + need;
    if (pad) { g_buf[off] = GS_WRAP; off = 0; }
    return g_buf + off;
}

static void g_commit(u32 n) {
    __atomic_store_n(&g_head, g_rend, __ATOMIC_SEQ_CST);
    gst_records++;
    gst_bytes += n;
    if (!__atomic_load_n(&g_busy, __ATOMIC_SEQ_CST)
        && !__atomic_exchange_n(&g_wake_sent, 1, __ATOMIC_SEQ_CST)) {
        pthread_mutex_lock(&g_lock);
        pthread_cond_signal(&g_not_empty);
        pthread_mutex_unlock(&g_lock);
    }
}

static inline u32 g_round(u32 n) { return (n + 7u) & ~7u; }

void ps2_gsq_wreg(u32 reg, u64 val) {
    u8 *r = g_reserve(16);
    r[0] = GS_WREG; put32(r, 4, reg); put64(r, 8, val);
    g_commit(16);
}
void ps2_gsq_hwpair(u64 lo, u64 hi) {
    u8 *r = g_reserve(24);
    r[0] = GS_HWPAIR; put64(r, 8, lo); put64(r, 16, hi);
    g_commit(24);
}
void ps2_gsq_hwstream(const void *data, u32 qwc) {
    /* big images go in pieces, so a single record never needs the whole ring */
    const u8 *d = (const u8 *)data;
    while (qwc) {
        u32 n = qwc > 4096u ? 4096u : qwc;
        u8 *r = g_reserve(8u + n * 16u);
        r[0] = GS_HWSTREAM; put32(r, 4, n);
        memcpy(r + 8, d, (size_t)n * 16u);
        g_commit(8u + n * 16u);
        d += (size_t)n * 16u;
        qwc -= n;
    }
}
void ps2_gsq_nstate(void) {
    u8 *r = g_reserve(8);
    r[0] = GS_NSTATE;
    g_commit(8);
    p_nstate = 1;
}
void ps2_gsq_mark_frame(void) { p_frame = 1; }
void ps2_gsq_nvertex(float x, float y, double z, float fog, int kind, int adc) {
    u8 *r = g_reserve(32);
    r[0] = GS_NVERTEX; put32(r, 4, (u32)kind);
    memcpy(r + 8, &x, 4); memcpy(r + 12, &y, 4); memcpy(r + 16, &z, 8);
    memcpy(r + 24, &fog, 4); put32(r, 28, (u32)adc);
    g_commit(32);
}
void ps2_gsq_drawmesh(const ps2_vk_state *st, const ps2_vk_mesh_block *b,
                      u32 vbase, u32 ifirst, u32 n, int cull, int topo) {
    int use_n = p_nstate;     /* st is native_state's placeholder then */
    u32 len = 32u + (u32)sizeof(ps2_vk_mesh_block) + (use_n ? 0u : (u32)sizeof(ps2_vk_state));
    u8 *r;
    len = g_round(len);
    r = g_reserve(len);
    r[0] = GS_DRAWMESH;
    r[1] = (u8)((use_n ? 1u : 0u) | (p_frame ? 2u : 0u));
    put32(r, 4, n); put32(r, 8, vbase); put32(r, 12, ifirst);
    put32(r, 16, (u32)cull); put32(r, 20, (u32)topo);
    memcpy(r + 32, b, sizeof(ps2_vk_mesh_block));
    if (!use_n) memcpy(r + 32 + sizeof(ps2_vk_mesh_block), st, sizeof(ps2_vk_state));
    g_commit(len);
    p_nstate = 0;
    p_frame = 0;
}
u32 ps2_gsq_mstore(const u8 *recs, u32 nv, const u32 *idx, u32 ni) {
    u32 seq = p_store_seq++ & 0x7FFFu;
    u32 len = g_round(16u + nv * 64u + ni * 4u);
    u8 *r = g_reserve(len);
    r[0] = GS_MSTORE; put32(r, 4, seq); put32(r, 8, nv); put32(r, 12, ni);
    memcpy(r + 16, recs, (size_t)nv * 64u);
    memcpy(r + 16 + (size_t)nv * 64u, idx, (size_t)ni * 4u);
    g_commit(len);
    return PS2_GSQ_TOKEN | (seq << 16);
}
void ps2_gsq_priv(u32 addr, u64 val) {
    u8 *r = g_reserve(16);
    r[0] = GS_PRIV; put32(r, 4, addr); put64(r, 8, val);
    g_commit(16);
}
void ps2_gsq_tag(u32 kind, u32 a, u32 b) {
    u8 *r = g_reserve(16);
    r[0] = GS_TAG; put32(r, 4, kind); put32(r, 8, a); put32(r, 12, b);
    g_commit(16);
}
void ps2_gsq_intent(const u8 *rec, u32 len) {
    u32 n = g_round(8u + len);
    u8 *r = g_reserve(n);
    r[0] = GS_INTENT; put32(r, 4, len);
    memcpy(r + 8, rec, len);
    g_commit(n);
}
void ps2_gsq_field(void) {
    u8 *r;
    p_nstate = p_frame = 0;   /* a native state or frame never carries across fields */
    r = g_reserve(8);
    r[0] = GS_FIELD;
    g_commit(8);
}

/* consumer: results of the worker's mesh stores, by token */
static struct { u32 vbase, ifirst; u8 ok; } g_store[0x8000];
int ps2_gsq_resolve(u32 v, int which, u32 *out) {
    if (!(v & PS2_GSQ_TOKEN)) { *out = v; return 1; }
    {   u32 s = (v >> 16) & 0x7FFFu;
        if (!g_store[s].ok) { gst_token_miss++; return 0; }
        *out = (which ? g_store[s].ifirst : g_store[s].vbase) + (v & 0xFFFFu);
        return 1;
    }
}

int ps2_gs_present_now(void);
static ps2_vk_state g_nst;
static void gfxq_field_done(void) {
    pthread_mutex_lock(&q_lock);
    ps2_gs_ph = 63;
    if (q_fields_pending > 0) q_fields_pending--;
    pthread_cond_broadcast(&q_not_full);
    pthread_cond_broadcast(&q_idle);
    pthread_mutex_unlock(&q_lock);
}

/* AC5_PROF=1: GS-thread time and count per record kind within a field */
static u64 g_kind_us[16], g_kind_n[16];
static u32 g_apply_inner(const u8 *r);
static u32 g_apply(const u8 *r) {
    extern int ac5_prof;
    ps2_gs_ph = r[0];
    if (!ac5_prof) return g_apply_inner(r);
    {   u8 op = r[0];
        u64 t0 = gq_now_ns();
        u32 n = g_apply_inner(r);
        if (op < 16) { g_kind_us[op] += (gq_now_ns() - t0) / 1000u; g_kind_n[op]++; }
        return n;
    }
}
static u32 g_apply_inner(const u8 *r) {
    switch (r[0]) {
    case GS_WREG:     ps2_gs_write_reg(get32(r, 4), get64(r, 8)); return 16u;
    case GS_HWPAIR:   ps2_gs_hwreg_pair(get64(r, 8), get64(r, 16)); return 24u;
    case GS_HWSTREAM: { u32 n = get32(r, 4); ps2_gs_hwreg_stream(r + 8, n); ps2_gs_ph = 59; return 8u + n * 16u; }
    case GS_NSTATE:   ps2_gs_native_state(&g_nst); return 8u;
    case GS_NVERTEX: {
        float x, y, fog; double z;
        memcpy(&x, r + 8, 4); memcpy(&y, r + 12, 4); memcpy(&z, r + 16, 8); memcpy(&fog, r + 24, 4);
        ps2_gs_native_vertex(x, y, z, fog, (int)get32(r, 4), (int)get32(r, 28));
        return 32u;
    }
    case GS_DRAWMESH: {
        u32 use_n = r[1] & 1u, frame = r[1] & 2u;
        u32 len = g_round(32u + (u32)sizeof(ps2_vk_mesh_block) + (use_n ? 0u : (u32)sizeof(ps2_vk_state)));
        ps2_vk_mesh_block b;
        ps2_vk_state st;
        u32 vb, ifi;
        memcpy(&b, r + 32, sizeof b);
        if (use_n) st = g_nst;
        else memcpy(&st, r + 32 + sizeof b, sizeof st);
        if (frame) ps2_gs_native_frame(&b.xoff, &b.yoff, &b.zmax);
        if (ps2_gsq_resolve(get32(r, 8), 0, &vb) && ps2_gsq_resolve(get32(r, 12), 1, &ifi))
            ps2_vk_draw_mesh_topo(&st, &b, vb, ifi, get32(r, 4), (int)get32(r, 16), (int)get32(r, 20));
        return len;
    }
    case GS_MSTORE: {
        u32 seq = get32(r, 4), nv = get32(r, 8), ni = get32(r, 12);
        u32 len = g_round(16u + nv * 64u + ni * 4u);
        const u32 *idx = (const u32 *)(const void *)(r + 16 + (size_t)nv * 64u);
        u32 vb = 0, ifi = 0;
        int ok = ps2_vk_mesh_store(r + 16, nv, idx, ni, &vb, &ifi) == 0;
        if (!ok) gst_store_fail++;
        g_store[seq & 0x7FFFu].ok = (u8)ok;
        g_store[seq & 0x7FFFu].vbase = vb;
        g_store[seq & 0x7FFFu].ifirst = ifi;
        return len;
    }
    case GS_PRIV:   ps2_gs_priv_write_now(get32(r, 4), get64(r, 8)); return 16u;
    case GS_TAG:    rn_gs_tag(get32(r, 4), get32(r, 8), get32(r, 12)); return 16u;
    case GS_INTENT: { u32 len = get32(r, 4); rn_gs_intent(r + 8, len); return g_round(8u + len); }
    case GS_FIELD: {
        int ok;
        ps2_gs_ph = 60;
        ok = ps2_gs_present_now();
        gst_fields++;
        ps2_gs_ph = 62;
        gfxq_field_done();
        ps2_gs_ph = 64;
        if (!ok) ps2_request_exit();
        return 8u;
    }
    default:
        ps2_log("gsq: corrupt record opcode %u -- stopping the GS thread", r[0]);
        return 0u;
    }
}

static void *g_main(void *unused) {
    (void)unused;
#ifdef PS2_VITA
    {   extern void vita_set_priority(int prio);
        const char *c = getenv("AC5_GSQ_CORE"), *p = getenv("AC5_GSQ_PRIO");
        vita_pin_current(c ? atoi(c) : 2);
        /* just below the renderer (140): guest threads (audio) no longer run
         * on core 2, so this no longer starves them (build 83's stutter), and
         * the renderer is all that can hold it up (at 170 it stalled, build 84) */
        vita_set_priority(p ? atoi(p) : 141);
    }
#endif
    gsq_consumer = 1;
#ifdef PS2_VITA
    g_gs_thid = sceKernelGetThreadId();
#endif
    gsq_gs_tp = ps2_gsq_tp();
    __atomic_store_n(&gsq_gs_started, 1, __ATOMIC_SEQ_CST);
    u64 fbusy = 0;      /* GS time spent on the field so far (not waiting) */
    pthread_mutex_lock(&g_lock);
    for (;;) {
        while (g_running && __atomic_load_n(&g_head, __ATOMIC_SEQ_CST) == g_tail) {
            __atomic_store_n(&g_busy, 0, __ATOMIC_SEQ_CST);
            __atomic_store_n(&g_wake_sent, 0, __ATOMIC_SEQ_CST);
            pthread_cond_broadcast(&g_idle);
            if (__atomic_load_n(&g_head, __ATOMIC_SEQ_CST) != g_tail) break;
            ps2_gs_ph = 31;
            pthread_cond_wait(&g_not_empty, &g_lock);
        }
        if (!g_running && __atomic_load_n(&g_head, __ATOMIC_SEQ_CST) == g_tail) break;
        __atomic_store_n(&g_busy, 1, __ATOMIC_SEQ_CST);
        size_t tail = g_tail, head = __atomic_load_n(&g_head, __ATOMIC_ACQUIRE);
        int corrupt = 0;
        unsigned records = 0;
        u64 t0 = gq_now_ns(), b0 = t0;
        pthread_mutex_unlock(&g_lock);
        while (tail != head && records < 512) {
            const u8 *r = g_buf + (tail & (g_cap - 1));
            u32 n;
            if (*r == GS_WRAP) { tail += g_cap - (tail & (g_cap - 1)); continue; }
            {   int field = *r == GS_FIELD;
                n = g_apply(r);
                if (field) {
                    u64 now = gq_now_ns();
                    fbusy += now - t0;
                    t0 = now;
                    {   /* with AC5_PROF=1 the split of the GS thread's own work */
                        extern u64 ps2_gs_tex_us, rn_tq_state_us, rn_tq_draw_us, ps2_gs_img_us, ps2_gs_dirty_us,
                                   ps2_gs_trxbeg_us;
                        static u64 c[6];
                        u64 now6[6] = { ps2_gs_tex_us, rn_tq_state_us, rn_tq_draw_us, ps2_gs_img_us,
                                        ps2_gs_dirty_us, ps2_gs_trxbeg_us };
                        if (fbusy > 50000000ull && (gst_slow++ < 40u || (gst_slow & 15u) == 0)) {
                            char kb[900];
                            int ko = 0;
                            kb[0] = 0;
                            for (int k = 1; k < 12 && ko < 400; k++)
                                if (g_kind_n[k])
                                    ko += snprintf(kb + ko, sizeof kb - (size_t)ko, " %s %.1f ms/%llu",
                                                   g_ph_name((u32)k), g_kind_us[k] / 1000.0, (unsigned long long)g_kind_n[k]);
                            ko += snprintf(kb + ko, sizeof kb - (size_t)ko, " | 1 ms samples (running/held off core/waiting):");
                            for (u32 k = 0; k < 64 && ko < (int)sizeof kb - 40; k++) {
                                u32 a0 = g_smp[k][0], a1 = g_smp[k][1], a2 = g_smp[k][2], a3 = g_smp[k][3];
                                if (a0 + a1 + a2 + a3)
                                    ko += snprintf(kb + ko, sizeof kb - (size_t)ko, " %s %u (%u/%u/%u)", g_ph_name(k),
                                                   a0 + a1 + a2 + a3, a0, a1, a2);
                            }
                            ps2_log("gsq: slow GS field, %.1f ms of GS work | native state %.1f ms (texture lookups"
                                    " %.1f ms), mesh draws %.1f ms, image data %.1f ms (dirty marking %.1f ms),"
                                    " transfer starts %.1f ms | by record:%s",
                                    (double)fbusy / 1e6, (now6[1] - c[1]) / 1000.0, (now6[0] - c[0]) / 1000.0,
                                    (now6[2] - c[2]) / 1000.0, (now6[3] - c[3]) / 1000.0, (now6[4] - c[4]) / 1000.0,
                                    (now6[5] - c[5]) / 1000.0, kb);
                        }
                        memset((void *)g_smp, 0, sizeof g_smp);
                        memcpy(c, now6, sizeof c);
                        memset(g_kind_us, 0, sizeof g_kind_us);
                        memset(g_kind_n, 0, sizeof g_kind_n);
                    }
                    fbusy = 0;
                }
            }
            if (!n) { corrupt = 1; break; }
            tail += n;
            records++;
        }
        {   u64 now = gq_now_ns();
            gst_busy_ns += now - b0;
            fbusy += now - t0;
        }
        ps2_gs_ph = 65;
        pthread_mutex_lock(&g_lock);
        __atomic_store_n(&g_tail, tail, __ATOMIC_RELEASE);
        pthread_cond_broadcast(&g_not_full);
        if (corrupt) { g_running = 0; break; }
    }
    __atomic_store_n(&g_busy, 0, __ATOMIC_SEQ_CST);
    pthread_cond_broadcast(&g_idle);
    pthread_mutex_unlock(&g_lock);
    return NULL;
}

static void g_wait_idle(void) {
    pthread_mutex_lock(&g_lock);
    if (g_head != g_tail || g_busy) {
        pthread_cond_signal(&g_not_empty);
        while (g_running && (__atomic_load_n(&g_head, __ATOMIC_SEQ_CST) != g_tail || g_busy))
            pthread_cond_wait(&g_idle, &g_lock);
    }
    pthread_mutex_unlock(&g_lock);
}

void ps2_gsq_sync(const char *why) {
    u64 t0;
    if (!ps2_gsq_on || gsq_consumer) return;
    t0 = gq_now_ns();
    gst_syncs++;
    if (why && why != gst_sync_why) { gst_sync_why = why; gst_sync_why_n = 0; }
    gst_sync_why_n++;
    g_wait_idle();
    gst_sync_ns += gq_now_ns() - t0;
}

void ps2_gsq_drain(void) {
    if (!ps2_gsq_on || gsq_consumer) return;
    g_wait_idle();
}

/* Called first thing on the worker: from here the worker's GS calls are
 * queued. If this thread cannot be told apart from the GS thread by its
 * thread register, the GS stays on the worker. */
void ps2_gsq_worker_started(void) {
    uintptr_t me = ps2_gsq_tp();
    if (!ps2_gsq_on) return;
    while (!__atomic_load_n(&gsq_gs_started, __ATOMIC_SEQ_CST)) {
        struct timespec ts = { 0, 1000000 };
        nanosleep(&ts, NULL);
    }
    if (!me || me == gsq_gs_tp) {
        ps2_log("gsq: the worker and the GS thread share a thread register (%08lX) -- GS stays on the worker",
                (unsigned long)me);
        ps2_gsq_on = 0;
        return;
    }
    ps2_gsq_worker_tp = me;
}

/* the GS thread has records waiting (the renderer gives way, build 108) */
int ps2_gsq_backlog(void) {
    return ps2_gsq_on && __atomic_load_n(&g_head, __ATOMIC_ACQUIRE) != __atomic_load_n(&g_tail, __ATOMIC_ACQUIRE);
}

void ps2_gsq_init(void) {
    const char *e = getenv("AC5_GSQ");
    size_t mb = 4;      /* a slow field queues ~1-2 MB; main RAM is tight */
    if (e && *e == '0') { ps2_log("gsq: AC5_GSQ=0 -- the GS stays on the graphics worker"); return; }
    {   const char *m = getenv("AC5_GSQ_MB");
        if (m && *m) { int v = atoi(m); if (v >= 1 && v <= 64) mb = (size_t)v; } }
    g_cap = 1;
    while (g_cap < mb * 1024u * 1024u) g_cap <<= 1;
    g_buf = (u8 *)malloc(g_cap);
    if (!g_buf) { ps2_log("gsq: no memory for a %lu MB ring; GS stays on the worker", (unsigned long)mb); return; }
    g_head = g_tail = g_ptail = g_rend = 0;
    g_running = 1;
    if (pthread_create(&g_thread, NULL, g_main, NULL) != 0) {
        ps2_log("gsq: cannot start the GS thread; GS stays on the worker");
        free(g_buf); g_buf = NULL; g_running = 0;
        return;
    }
    ps2_gsq_on = 1;
#ifdef PS2_VITA
    {   const char *e2 = getenv("AC5_GS_SAMPLER");
        pthread_t st;
        /* measurement only: off in the release build (AC5_GS_SAMPLER=1 turns it on) */
        if (e2 && *e2 == '1' && pthread_create(&st, NULL, g_sampler, NULL) == 0) pthread_detach(st);
    }
#endif
    ps2_log("gsq: the GS runs on its own thread (core %s), %lu MB ring; AC5_GSQ=0 turns it off",
            getenv("AC5_GSQ_CORE") ? getenv("AC5_GSQ_CORE") : "2", (unsigned long)(g_cap >> 20));
}

void ps2_gsq_report(char *buf, int n) {
    static u64 b0, r0, w0, wn0, s0, sn0, f0, bz0, tm0, sf0;
    if (!ps2_gsq_on) { if (n) buf[0] = 0; return; }
    snprintf(buf, (size_t)n,
             "gsq: %llu records %.1f MB, GS thread busy %.0f ms, %llu fields; worker waited on a full GS ring %llu times"
             " (%.1f ms), synced %llu times (%.1f ms, last for %s x%llu); token misses %llu, failed stores %llu",
             (unsigned long long)(gst_records - r0), (double)(gst_bytes - b0) / 1048576.0,
             (double)(gst_busy_ns - bz0) / 1e6, (unsigned long long)(gst_fields - f0),
             (unsigned long long)(gst_full_waits - wn0), (double)(gst_full_ns - w0) / 1e6,
             (unsigned long long)(gst_syncs - sn0), (double)(gst_sync_ns - s0) / 1e6,
             gst_sync_why ? gst_sync_why : "-", (unsigned long long)gst_sync_why_n,
             (unsigned long long)(gst_token_miss - tm0), (unsigned long long)(gst_store_fail - sf0));
    b0 = gst_bytes; r0 = gst_records; w0 = gst_full_ns; wn0 = gst_full_waits; s0 = gst_sync_ns;
    sn0 = gst_syncs; f0 = gst_fields; bz0 = gst_busy_ns; tm0 = gst_token_miss; sf0 = gst_store_fail;
}
