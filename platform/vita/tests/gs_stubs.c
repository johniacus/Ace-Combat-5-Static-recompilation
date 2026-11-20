#include "ps2_runtime.h"
#include "ps2_vk.h"
#include "ps2_settings.h"
#include <stdint.h>
#include <string.h>
int g_cap_shallow, ps2_gif_path, ps2_gsq_on, ps2_verbose, rn_census_on, rn_claims_on, rn_dump_on;
uintptr_t ps2_gsq_worker_tp;
u32 rn_gs_emitter;
ps2_settings ps2_cfg;
int ps2_phase_on; u64 ps2_phase_calls[PS2_PH_N], ps2_phase_cycles[PS2_PH_N], ps2_phase_self[PS2_PH_N];
__thread u64 ps2_phase_child;
void ps2_cap_maybe_start(void) {}
void ps2_cap_present(void) {}
void ps2_cap_reg(u32 r, u64 v) { (void)r; (void)v; }
void ps2_cap_request(void) {}
void ps2_dump_trace(const char *w) { (void)w; }
void ps2_fatal(const char *f, ...) { (void)f; }
int ps2_gfxq_field(void) { return 1; }
void ps2_gs_priv_cap_load(const u64 *a, u64 b, u64 c) { (void)a; (void)b; (void)c; }
void ps2_gs_priv_cap_save(u64 *a, u64 *b, u64 *c) { (void)a; (void)b; (void)c; }
u64 ps2_gs_priv_read(u32 a) { (void)a; return 0; }
void ps2_gsq_hwpair(u64 a, u64 b) { (void)a; (void)b; }
void ps2_gsq_hwstream(const void *d, u32 q) { (void)d; (void)q; }
void ps2_gsq_mark_frame(void) {}
void ps2_gsq_nstate(void) {}
void ps2_gsq_nvertex(float x, float y, double z, float f, int k, int a) { (void)x; (void)y; (void)z; (void)f; (void)k; (void)a; }
void ps2_gsq_sync(const char *w) { (void)w; }
void ps2_gsq_wreg(u32 r, u64 v) { (void)r; (void)v; }
void ps2_log(const char *f, ...) { (void)f; }
const char *ps2_statecap_dir(void) { return "."; }
void ps2_statecap_gs_done(void) {}
int ps2_statecap_gs_pending(void) { return 0; }
int ps2_vk_closed(void) { return 0; }
void ps2_vk_draw(int t, const ps2_vk_state *s, const ps2_vk_vertex *v, int n) { (void)t; (void)s; (void)v; (void)n; }
int ps2_vk_enabled(void) { return 1; }
int ps2_vk_mesh_material_ok(const ps2_vk_state *s) { (void)s; return 0; }
int ps2_vk_present(u32 a, u32 b, u32 c, u32 d, u32 e) { (void)a; (void)b; (void)c; (void)d; (void)e; return 1; }
void ps2_vk_statecap(const char *d) { (void)d; }
static u32 ntex;
u32 ps2_vk_texture(u64 k, u32 h, const u8 *p, u32 w, u32 hh) { (void)k; (void)h; (void)p; (void)w; (void)hh; return ntex++ % 1000; }
int ps2_vk_texture_is(u32 i, u64 k, u32 h) { (void)i; (void)k; (void)h; return 1; }
u32 ps2_vk_texture_native(u32 s, u32 x, u32 y, u32 w, u32 h) { (void)s; (void)x; (void)y; (void)w; (void)h; return 0xFFFFFFFFu; }
u32 ps2_vk_texture_native_find(u64 k, u32 h, u32 w, u32 hh) { (void)k; (void)h; (void)w; (void)hh; return 0xFFFFFFFFu; }
u32 ps2_vk_texture_native_rgba(u64 k, u32 h, const u8 *p, u32 w, u32 hh) { (void)k; (void)h; (void)p; (void)w; (void)hh; return 0xFFFFFFFFu; }
const u8 *ps2_vk_texture_pixels(u32 i) { (void)i; return NULL; }
void ps2_vk_texture_size(u32 i, u32 *w, u32 *h) { (void)i; *w = 64; *h = 64; }
void rn_census_frame(void) {}
void rn_census_prim_slow(const struct ps2_vk_state *s, int k, const struct ps2_vk_vertex *v, int n, int c) { (void)s; (void)k; (void)v; (void)n; (void)c; }
int rn_claim_prim_slow(const struct ps2_vk_state *s, int k, const struct ps2_vk_vertex *v, int n) { (void)s; (void)k; (void)v; (void)n; return 0; }
void rn_claims_init(void) {}
void rn_dump_frame(void) {}
void rn_dump_gsreg(u32 r, u64 v) { (void)r; (void)v; }
volatile unsigned char ps2_gs_ph;
