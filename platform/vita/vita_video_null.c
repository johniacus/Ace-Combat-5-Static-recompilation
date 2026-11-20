/* Headless stand-in for ps2_video.c (the Vulkan renderer) on PS Vita.
 *
 * Stage one of the port only needs the game to boot and run, so every entry
 * point the rest of the runtime calls into the renderer is here with "no
 * renderer" behaviour: ps2_video_init fails, ps2_main then runs headless, and
 * the emulated GS keeps rasterising into its own framebuffer as it does with
 * --novideo on the PC. The GXM renderer replaces this file later. */
#include "ps2_runtime.h"
#include "ps2_vk.h"
#include "ps2_hle.h"

#include <stdio.h>

float ps2_vk_display_aspect = 4.0f / 3.0f;
int ps2_vk_lockstep = 0;
void (*ps2_vk_before_record)(void) = 0;

int  ps2_video_init(const char *title, int width, int height) {
    (void)title; (void)width; (void)height;
    ps2_log("vita: renderer not implemented yet; running headless");
    return -1;
}
void ps2_video_hidden(int on) { (void)on; }
void ps2_video_shutdown(void) {}
void ps2_video_report(void) { ps2_log("video: headless (vita null renderer)"); }

int  ps2_vk_enabled(void) { return 0; }
int  ps2_vk_closed(void) { return 0; }

void ps2_vk_draw(int topology, const ps2_vk_state *st,
                 const ps2_vk_vertex *v, int n) {
    (void)topology; (void)st; (void)v; (void)n;
}
void ps2_vk_draw_2d(int topology, const ps2_vk_state *st,
                    const ps2_vk_vertex *v, int n) {
    (void)topology; (void)st; (void)v; (void)n;
}
void ps2_vk_native_draw(const ps2_vk_native *nd, const ps2_vk_vertex *v, int n) {
    (void)nd; (void)v; (void)n;
}

int  ps2_vk_mesh_available(void) { return 0; }
int  ps2_vk_mesh_material_ok(const ps2_vk_state *st) { (void)st; return 0; }
int  ps2_vk_mesh_store(const u8 *records, u32 n_verts, const u32 *indices,
                       u32 n_indices, u32 *vbase, u32 *ifirst) {
    (void)records; (void)n_verts; (void)indices; (void)n_indices;
    if (vbase) *vbase = 0;
    if (ifirst) *ifirst = 0;
    return -1;
}
int  ps2_vk_mesh_room(u32 n_verts, u32 n_indices, u32 n_draws) {
    (void)n_verts; (void)n_indices; (void)n_draws; return 0;
}
void ps2_vk_draw_mesh(const ps2_vk_state *st, const ps2_vk_mesh_block *b,
                      u32 vbase, u32 ifirst, u32 n_indices, int cull) {
    (void)st; (void)b; (void)vbase; (void)ifirst; (void)n_indices; (void)cull;
}
void ps2_vk_draw_mesh_topo(const ps2_vk_state *st, const ps2_vk_mesh_block *b,
                           u32 vbase, u32 ifirst, u32 n_indices, int cull,
                           int topology) {
    (void)st; (void)b; (void)vbase; (void)ifirst; (void)n_indices; (void)cull;
    (void)topology;
}

u32  ps2_vk_texture(u64 key, u32 hash, const u8 *rgba, u32 w, u32 h) {
    (void)key; (void)hash; (void)rgba; (void)w; (void)h; return 0;
}
int  ps2_vk_texture_is(u32 idx, u64 key, u32 hash) {
    (void)idx; (void)key; (void)hash; return 0;
}
u32  ps2_vk_texture_native(u32 src, u32 x, u32 y, u32 w, u32 h) {
    (void)src; (void)x; (void)y; (void)w; (void)h; return 0;
}
u32  ps2_vk_texture_native_find(u64 key, u32 hash, u32 w, u32 h) {
    (void)key; (void)hash; (void)w; (void)h; return 0;
}
u32  ps2_vk_texture_native_rgba(u64 key, u32 hash, const u8 *rgba, u32 w, u32 h) {
    (void)key; (void)hash; (void)rgba; (void)w; (void)h; return 0;
}
const u8 *ps2_vk_texture_pixels(u32 idx) { (void)idx; return 0; }
void ps2_vk_texture_size(u32 idx, u32 *w, u32 *h) {
    (void)idx;
    if (w) *w = 0;
    if (h) *h = 0;
}

void ps2_vk_invalidate(u32 base, u32 size) { (void)base; (void)size; }

int  ps2_vk_present(u32 disp_x, u32 disp_y, u32 disp_w, u32 disp_h, u32 rt) {
    (void)disp_x; (void)disp_y; (void)disp_w; (void)disp_h; (void)rt;
    return 0;
}

int  ps2_vk_screenshot(const char *path) { (void)path; return -1; }
void ps2_vk_stats(u64 *frames, u64 *draws, u64 *verts, u64 *textures) {
    if (frames) *frames = 0;
    if (draws) *draws = 0;
    if (verts) *verts = 0;
    if (textures) *textures = 0;
}
void ps2_vk_state_save(ps2_state_put put, void *ud) { (void)put; (void)ud; }
void ps2_vk_statecap(const char *dir) { (void)dir; }
