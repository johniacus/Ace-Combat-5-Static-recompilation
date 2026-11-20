/* Stress test of the GS queue: a producer thread (standing in for the gfxq
 * worker) issues random GS commands; the consumer stubs log what the GS
 * thread executes; the two logs must match exactly, syncs must see every
 * earlier command done, and mesh-store tokens must resolve to the offsets
 * the store returned. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <pthread.h>
#include <unistd.h>
#include "ps2_runtime.h"
#include "ps2_vk.h"
#include "ps2_gsq.h"

/* ---- stubs the gfxq file links against ---- */
u64 ps2_gs_dec_psm[64], ps2_gs_dirty_us, ps2_gs_img_calls, ps2_gs_img_cmp_px, ps2_gs_img_fast_qw,
    ps2_gs_img_px[4], ps2_gs_img_skip_px, ps2_gs_img_slow_qw, ps2_gs_img_us, ps2_gs_tex_n, ps2_gs_tex_us,
    ps2_gs_trxbeg_us, ps2_tl_end[5], ps2_tl_us[4], ps2_vc_code_cmd[128], ps2_vc_direct_qw, ps2_vc_elem_words,
    ps2_vc_fifo_qw, ps2_vc_slow_fmt[64], ps2_vc_slow_words, ps2_vc_vifcode_words, ps2_vif_fast_qw,
    ps2_vt_direct_us, ps2_vt_run_n[2], ps2_vt_run_us[2], ps2_vt_vif_us[2], rn_tq_state_n, rn_tq_state_us, rn_tq_draw_us;
int ps2_phase_on; u64 ps2_phase_calls[PS2_PH_N], ps2_phase_cycles[PS2_PH_N], ps2_phase_self[PS2_PH_N];
__thread u64 ps2_phase_child;
void ps2_gif_ctrl_write(u32 a, u32 v) { (void)a; (void)v; }
void ps2_gif_transfer(const ps2_reg128 *d, u32 q) { (void)d; (void)q; }
void ps2_host_prof_attach(const char *n) { (void)n; }
void ps2_log(const char *f, ...) { (void)f; }
void ps2_phase_field_tick(void) {}
void ps2_request_exit(void) {}
void ps2_scene_speed_report(double d) { (void)d; }
void ps2_vif_clear_stall(int w) { (void)w; }
void ps2_vif_fbrst(int w, u32 v) { (void)w; (void)v; }
void ps2_vif_transfer(int w, const ps2_reg128 *q, u32 n) { (void)w; for (u32 i = 0; i < n; i++) { u32 x[4]; memcpy(x, (const u8 *)q + i * 16, 16); if (!PS2_GSQ_DEFER()) abort(); ps2_gsq_wreg(x[0] & 0x63, ((u64)x[2] << 32) | x[1]); } }
void ps2_vif_write(int w, u32 v) { (void)w; (void)v; }
void ps2_vu1_speed_counters(u64 *a, u64 *b, u64 *c) { (void)a; (void)b; (void)c; }
void ps2_vu_census_tick(void) {}
void rn_vp_timing_report(u32 f) { (void)f; }
int rn_vp_field_breakdown(char *b, int n) { (void)b; (void)n; return 0; }

/* consumer-side log */
#define LOGN (1u << 22)
static u64 act[LOGN]; static volatile u32 nact;
static u64 exp_[LOGN]; static u32 nexp;
void ps2_gsq_worker_started(void);
static int consumer_defers;
static void A(u64 v) { if (PS2_GSQ_DEFER()) consumer_defers++; act[nact] = v; if ((nact & 255u) == 0 && (nact & 0x10000u)) usleep(1000); __atomic_add_fetch(&nact, 1, __ATOMIC_RELEASE); }
static void E(u64 v) { exp_[nexp++] = v; }
static u32 store_next = 100;
static u64 mix(u64 a, u64 b) { return (a * 0x9E3779B97F4A7C15ull) ^ b; }

void ps2_gs_write_reg(u32 reg, u64 val) { A(mix(1, mix(reg, val))); }
void ps2_gs_hwreg_pair(u64 lo, u64 hi) { A(mix(2, mix(lo, hi))); }
void ps2_gs_hwreg_stream(const void *data, u32 qwc) {
    const u8 *p = data; u64 h = 3;
    for (u32 i = 0; i < qwc * 16u; i++) h = mix(h, p[i]);
    A(mix(h, qwc));
}
static int nstate_val;
void ps2_gs_native_state(ps2_vk_state *st) { memset(st, 0, sizeof *st); st->tex_index = (u32)(++nstate_val); A(mix(4, (u64)nstate_val)); }
void ps2_gs_native_frame(float *x, float *y, float *z) { *x = 1.5f; *y = 2.5f; *z = 3.0f; }
void ps2_gs_native_vertex(float x, float y, double z, float fog, int kind, int adc) {
    A(mix(5, mix((u64)(x * 16), mix((u64)(y * 16), mix((u64)z, mix((u64)(fog * 100), (u64)(kind * 7 + adc)))))));
}
void ps2_vk_draw_mesh_topo(const ps2_vk_state *st, const ps2_vk_mesh_block *b, u32 vb, u32 ifi, u32 n, int cull, int topo) {
    A(mix(6, mix(st->tex_index, mix((u64)(b->xoff * 2), mix(b->pass, mix(vb, mix(ifi, mix(n, (u64)(cull * 3 + topo)))))))));
}
int ps2_vk_mesh_store(const u8 *recs, u32 nv, const u32 *idx, u32 ni, u32 *vb, u32 *ifi) {
    u64 h = 7;
    for (u32 i = 0; i < nv * 64u; i++) h = mix(h, recs[i]);
    for (u32 i = 0; i < ni; i++) h = mix(h, idx[i]);
    *vb = store_next; *ifi = store_next * 2u; store_next += nv;
    A(h);
    return 0;
}
int ps2_gs_present_now(void) { A(8); return 1; }
void ps2_gs_priv_write_now(u32 a, u64 v) { A(mix(9, mix(a, v))); }
void rn_gs_tag(u32 k, u32 a, u32 b) { A(mix(10, mix(k, mix(a, b)))); }
void rn_gs_intent(const u8 *r, u32 len) { u64 h = 11; for (u32 i = 0; i < len; i++) h = mix(h, r[i]); A(mix(h, len)); }


#include "ps2_gfxq.h"
static u32 rnd(void) { static __thread u32 s = 4242; s ^= s << 13; s ^= s >> 17; s ^= s << 5; return s; }
int main(void) {
    setenv("AC5_GSQ_MB", "1", 1);
    setenv("PS2_ASYNC_MB", "1", 1);
    setenv("PS2_ASYNC_FIELDS", "3", 1);
    ps2_gfxq_init();
    if (!ps2_gfxq_on || !ps2_gsq_on) { printf("init failed %d %d\n", ps2_gfxq_on, ps2_gsq_on); return 1; }
    static ps2_reg128 q[300];
    int bad = 0;
    for (int it = 0; it < 200000 && !bad && nexp < LOGN - 1000; it++) {
        u32 k = rnd() % 6;
        if (k == 0 || k == 1) {
            u32 n = 1 + rnd() % 200;
            for (u32 i = 0; i < n; i++) {
                q[i].uw[0] = rnd(); q[i].uw[1] = rnd(); q[i].uw[2] = rnd(); q[i].uw[3] = 0;
                E(mix(1, mix(q[i].uw[0] & 0x63, ((u64)q[i].uw[2] << 32) | q[i].uw[1])));
            }
            ps2_gfxq_vif_qw(1, q, n);
        } else if (k == 2) { u32 a = rnd(), b = rnd(), c = rnd(); ps2_gfxq_tag(a, b, c); E(mix(10, mix(a, mix(b, c)))); }
        else if (k == 3) { u32 a = rnd(); u64 v = rnd(); ps2_gfxq_gs_priv(a, v); E(mix(9, mix(a, v))); }
        else if (k == 4) { ps2_gfxq_field(); E(8); }
        else if (rnd() % 500 == 0) {
            ps2_gfxq_drain();
            if (__atomic_load_n(&nact, __ATOMIC_ACQUIRE) != nexp) { printf("DRAIN: %u of %u\n", nact, nexp); bad = 1; }
        }
    }
    ps2_gfxq_drain();
    if (nact != nexp) { printf("count %u vs %u\n", nact, nexp); bad = 1; }
    for (u32 i = 0; i < nexp && i < nact; i++) if (act[i] != exp_[i]) { printf("mismatch at %u\n", i); bad = 1; break; }
    printf("%s %u\n", bad ? "FAIL" : "OK", nexp);
    return bad;
}
