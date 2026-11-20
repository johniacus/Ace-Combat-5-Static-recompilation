/* PS Vita renderer: the ps2_vk_* interface implemented with vitaGL (GXM).
 *
 * This mirrors the structure of runtime/src/ps2_video.c, the Vulkan renderer:
 *
 *  - the emulated GS (on the gfx-queue thread) calls ps2_vk_draw() with
 *    vertices in GS pixel space and a ps2_vk_state; draws are recorded into a
 *    CPU display list, merging consecutive draws with identical state;
 *  - ps2_vk_texture() stores decoded RGBA textures CPU-side, keyed and hashed
 *    exactly like the Vulkan store; the renderer uploads them on use;
 *  - ps2_vk_present() publishes the list (three rotating slots) to the
 *    renderer thread, which owns the GL context and replays it into one
 *    render target per GS buffer ("slot"), then shows the displayed slot.
 *
 * Stage 1 scope. Implemented: textured/untextured triangles, lines and
 * points; TFX/TCC; fog; alpha test incl. the AFAIL split; GS blending
 * (A-B)*C+D for every A/B/C/D the Vulkan path handles; depth test/write;
 * colour/alpha write masks; scissor; sampling another render target; reading
 * the target being drawn (via a snapshot copy); present with 4:3 letterbox.
 *
 * Not yet: the native mesh path (ps2_vk_mesh_available() is 0, so the VU1
 * output goes through the GS like everything else), native sun/sky passes,
 * the HUD overlay composite, destination-alpha test (DATE), RG channel
 * shuffles, sampling depth, the 0..2 colour range clamp (colclip), and
 * internal-resolution upscaling.
 *
 * Known approximation: GXM has no constant blend colour, so a FIX blend
 * factor is delivered through the shader's alpha output, which also becomes
 * the alpha written to the target for those draws. */
#include "ps2_runtime.h"
#include "ps2_vk.h"
#include "ps2_gsq.h"
#include "ps2_hle.h"
#include "ps2_settings.h"

#include <vitaGL.h>
#include <psp2/io/stat.h>
#include <psp2/kernel/processmgr.h>
#include <psp2/kernel/sysmem.h>
#include <psp2/kernel/threadmgr.h>

#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ------------------------------------------------------------------ limits */

#define GL_MAX_VERTS    (96u * 1024u)
#define GL_MAX_DRAWS    (12u * 1024u)
#define GL_MAX_TEXTURES 4096u
#define GL_TEX_BYTES    (48u << 20)     /* CPU copies, both halves counted */
#define RT_SLOTS        32
#define RT_MAX          1024u
#define LIST_SLOTS      3
#define SCREEN_W        960
#define SCREEN_H        544

float ps2_vk_display_aspect = 4.0f / 3.0f;
int ps2_vk_lockstep = 0;
void (*ps2_vk_before_record)(void) = 0;

/* --------------------------------------------------------------- the store */

typedef struct gl_texture {
    u64 key;
    u32 hash, w, h;
    u8 *rgba, *rgba_alt;
    u32 generation;         /* bumped by the recorder on new pixels */
    u32 uploaded_gen;       /* renderer side */
    u32 rec_seq;
    GLuint name;            /* renderer side */
    u32 gl_w, gl_h;         /* renderer side: size of the GL texture's storage */
    int filter;             /* renderer side: last GL filter set, -1 unset */
    volatile u32 pending;   /* decode jobs still to fill its pixels */
} gl_texture;

static gl_texture textures[GL_MAX_TEXTURES];
static u32 ntextures;
static size_t tex_bytes;
static u32 tex_rec_seq = 1;

#define TEX_HASH_BITS 12
static s32 tex_head[1u << TEX_HASH_BITS];
static s32 tex_next[GL_MAX_TEXTURES];

static u32 tex_bucket(u64 key) {
    return (u32)((key * 0x9E3779B97F4A7C15ull) >> (64 - TEX_HASH_BITS));
}

/* ------------------------------------------------------------ display list */

typedef struct gl_draw {
    u16 topology;
    u16 tex;            /* texture store index, 0xFFFF none */
    u16 tex_rt;         /* 1 + slot when sampling a render target */
    u16 rt, zrt;
    u8  tex_self, tex_depth, skip;
    u8  native;
    u32 first, count;
    u32 variant;        /* fragment shader variant key */
    u32 blend;          /* abe<<8 | A | B<<2 | C<<4 | D<<6 */
    u32 fix;
    u32 depth;          /* ztst | zwrite<<2 | rgb<<3 | a<<4 | chanmask<<5 */
    u32 aref;
    u32 fogcol;
    u16 minu, maxu, minv, maxv;
    float tex_w, tex_h;
    s32 scissor[4];
    u32 fb_w;
    int point_filter;
    /* native mesh draws: first/count index the slot's mesh indices */
    u32 mvbase, mvn, mblock;
    u8  mcull;
} gl_draw;

static ps2_vk_vertex *slot_verts[LIST_SLOTS];
static gl_draw *slot_draws[LIST_SLOTS];

static ps2_vk_vertex *rec_verts, *pub_verts, *rep_verts;
static gl_draw *rec_draws, *pub_draws, *rep_draws;
static u32 rec_nverts, rec_ndraws, pub_nverts, pub_ndraws, rep_nverts, rep_ndraws;
static int rec_slot, pub_slot;
static int rec_overflow;
static u16 rec_native_mark;

/* Native mesh arenas (one per display-list slot, swapped with it): the VU1
 * programs ported to C store their vertices here untransformed, and the
 * mesh vertex program does the transform, lighting and fog on the GPU.
 * Records are the PC renderer's 64-byte layout with the integer fields
 * turned into floats (sceGxm has no 32-bit integer attributes). */
#define MESH_V_MAX  20480u
#define MESH_I_MAX  49152u
#define MESH_B_MAX  1536u
typedef struct { float pos[4], nrm[4], col[4], uv[4]; } mesh_rec;
typedef struct {
    mesh_rec *recs;
    u16 *idx;
    ps2_vk_mesh_block *blocks;
    u32 nrecs, nidx, nblocks;
} mesh_arena;
static mesh_arena slot_mesh[LIST_SLOTS];
static mesh_arena *rec_mesh, *pub_mesh, *rep_mesh;
static volatile int mesh_ready;     /* renderer built the mesh programs */
static int mesh_cpu_ok;             /* ... and the pre-transformed one */
static int mesh_alloc_ok;
static u64 stat_mesh_draws, stat_mesh_overflow, stat_mesh_stores, stat_mesh_verts;
static struct { u32 vbase, ifirst, icount, block, vn; u8 cull; } rec_mesh_pending;

static pthread_mutex_t list_lock = PTHREAD_MUTEX_INITIALIZER;
/* Wakes the renderer when a frame is published. The pthread condition
 * variable's timed wait did not wake on a broadcast here: the renderer sat
 * out ~40 ms per frame before noticing a new one (build 56 log). */
static SceUID frame_sema = -1;
static pthread_cond_t list_cv = PTHREAD_COND_INITIALIZER;
static int frame_requested, frame_done;
static volatile int renderer_running;   /* 0 starting, 1 up, -1 failed/ended */
static volatile int gl_ready;

static u32 present_x, present_y, present_w = 640, present_h = 448;
static u32 present_rt;
static int present_blank;
/* second display circuit (see ps2_gs_present_now): pending from the
 * recorder, published with the frame */
static int ov_pend_rt = -1, present_ov_rt = -1;
static u32 ov_pend_x, ov_pend_y, present_ov_x, present_ov_y;
static float ov_pend_a = -1.0f, present_ov_a = -1.0f;
void ps2_vk_present_overlay(int rt, u32 x, u32 y, float alpha) {
    static int off = -1;
    if (off < 0) off = getenv("AC5_OVERLAY") ? 0 : 1;   /* opt-in: not how the briefing map is shown */
    ov_pend_rt = !off && rt < RT_SLOTS ? rt : -1;
    ov_pend_x = x; ov_pend_y = y; ov_pend_a = alpha;
}

static u64 stat_frames, stat_draws, stat_verts, stat_dropped, stat_uploads;
static u64 rl_lock_us, rl_idle_us, rl_wall_us, rl_frames;
static u64 rl_pres0, rl_drop0, rl_gap_max, rl_last_pub;
static u64 rl_gap_hist[5];          /* publish intervals: <10, <20, <40, <80, >=80 ms */
static u64 rl_age_us, rl_wake_us, rl_wakes;   /* renderer loop: where the wall time goes */
static u64 stat_presents, stat_snapshots, stat_snap_skips, stat_rt_binds, stat_tex_deletes;
static volatile u32 stat_gpu_tex_kb, stat_vgl_vram_kb, stat_vgl_ram_kb, stat_vgl_phy_kb;
#define SNAP_MAX_PER_FRAME 8
static int pool_ok[3];   /* CDRAM, main, phycont pools really allocated */

/* ------------------------------------------------------------ recorder API */

int ps2_vk_enabled(void) { return gl_ready && renderer_running > 0; }
int ps2_vk_closed(void) { return 0; }

static u32 variant_of(const ps2_vk_state *st, int discard_on_fail, int afonly) {
    u32 v = 0;
    int tme = st->tme || st->tex_rt;
    if (tme) {
        v |= 1u;
        v |= (st->tfx & 3u) << 1;
        if (st->tcc) v |= 1u << 3;
        if (st->fst) v |= 1u << 11;
        if (st->tex_point) v |= 1u << 12;
        v |= (st->wms & 3u) << 13;
        v |= (st->wmt & 3u) << 15;
        if (st->tex_rt ? !st->tex_depth : st->tme) v |= 1u << 17;   /* half texel */
    }
    if (st->fge) v |= 1u << 4;
    if (st->ate && st->atst != 1u) {
        v |= 1u << 8;
        v |= (st->atst & 7u) << 5;
        if (discard_on_fail) v |= 1u << 9;
        if (afonly) v |= 1u << 10;
    }
    if (st->abe && st->alpha_c == 2u) v |= 1u << 18;   /* alpha out = FIX */
    if (st->fba && !st->abe) v |= 1u << 19;
    return v;
}

static void afail_mask(u32 afail, int *rgb, int *a, int *z) {
    if (afail == 1u) { *z = 0; }
    else if (afail == 2u) { *rgb = 0; *a = 0; }
    else { *a = 0; *z = 0; }
}

static void rec_one(int topology, const ps2_vk_state *st,
                    const ps2_vk_vertex *v, int n, u32 variant, u32 depth) {
    gl_draw *d;
    u16 tex = (u16)(st->tme && !st->tex_rt && st->tex_index < ntextures
                    ? st->tex_index : 0xFFFFu);
    u32 blend = ((u32)st->abe << 8) | st->alpha_a | (st->alpha_b << 2)
              | (st->alpha_c << 4) | (st->alpha_d << 6);
    u8 skip = (u8)(st->shuffle_rg || (st->tex_rt && st->tex_depth));
    if (tex != 0xFFFFu) textures[tex].rec_seq = tex_rec_seq;
    for (u32 ch = 0; ch < 4; ch++)
        if (((st->fb_mask >> (ch * 8u)) & 255u) == 255u)
            depth |= 1u << (5u + ch);

    if (rec_ndraws && rec_native_mark != PS2_VK_NATIVE_MESH) {
        d = &rec_draws[rec_ndraws - 1];
        if (d->topology == (u16)topology && d->tex == tex
            && d->tex_rt == (u16)st->tex_rt && d->rt == (u16)st->rt
            && d->zrt == (u16)st->zrt && d->tex_self == (u8)st->tex_self
            && d->variant == variant && d->blend == blend && d->fix == st->alpha_fix
            && d->depth == depth && d->aref == st->aref && d->fogcol == st->fogcol
            && d->minu == (u16)st->minu && d->maxu == (u16)st->maxu
            && d->minv == (u16)st->minv && d->maxv == (u16)st->maxv
            && d->tex_w == st->tex_w && d->tex_h == st->tex_h
            && d->scissor[0] == st->scissor[0] && d->scissor[1] == st->scissor[1]
            && d->scissor[2] == st->scissor[2] && d->scissor[3] == st->scissor[3]
            && d->native == rec_native_mark && d->skip == skip
            && d->first + d->count == rec_nverts) {
            memcpy(&rec_verts[rec_nverts], v, sizeof(*v) * (size_t)n);
            rec_nverts += (u32)n;
            d->count += (u32)n;
            return;
        }
    }
    d = &rec_draws[rec_ndraws++];
    memset(d, 0, sizeof *d);
    d->topology = (u16)topology;
    d->tex = tex;
    d->tex_rt = (u16)st->tex_rt;
    d->rt = (u16)st->rt;
    d->zrt = (u16)st->zrt;
    d->tex_self = (u8)st->tex_self;
    d->tex_depth = (u8)st->tex_depth;
    d->skip = skip;
    d->native = (u8)rec_native_mark;
    d->first = rec_nverts;
    d->count = (u32)n;
    d->variant = variant;
    d->blend = blend;
    d->fix = st->alpha_fix;
    d->depth = depth;
    d->aref = st->aref;
    d->fogcol = st->fogcol;
    d->minu = (u16)st->minu; d->maxu = (u16)st->maxu;
    d->minv = (u16)st->minv; d->maxv = (u16)st->maxv;
    d->tex_w = st->tex_w;
    d->tex_h = st->tex_h;
    memcpy(d->scissor, st->scissor, sizeof d->scissor);
    d->fb_w = st->fb_w;
    d->point_filter = st->tex_point;
    if (rec_native_mark == PS2_VK_NATIVE_MESH) {
        d->first = rec_mesh_pending.ifirst;
        d->count = rec_mesh_pending.icount;
        d->mvbase = rec_mesh_pending.vbase;
        d->mvn = rec_mesh_pending.vn;
        d->mblock = rec_mesh_pending.block;
        d->mcull = rec_mesh_pending.cull;
        return;
    }
    memcpy(&rec_verts[rec_nverts], v, sizeof(*v) * (size_t)n);
    rec_nverts += (u32)n;
}

static void draw_inner(int topology, const ps2_vk_state *st,
                       const ps2_vk_vertex *v, int n) {
    int split, passes, discard_on_fail = 1, write_rgb = 1, write_a = 1, write_z;
    u32 ztst, depth;
    if (!gl_ready || n <= 0) return;
    split = st->ate && st->atst != 0u && st->atst != 1u && st->afail != 0u;
    passes = split ? 2 : 1;
    if ((rec_native_mark != PS2_VK_NATIVE_MESH && rec_nverts + (u32)(n * passes) > GL_MAX_VERTS)
        || rec_ndraws + (u32)passes > GL_MAX_DRAWS) {
        rec_overflow = 1;
        return;
    }
    write_z = st->zwrite ? 1 : 0;
    ztst = (st->zte ? st->ztst : 1u) & 3u;
    depth = ztst | ((u32)write_z << 2) | ((u32)write_rgb << 3) | ((u32)write_a << 4);
    rec_one(topology, st, v, n, variant_of(st, discard_on_fail, 0), depth);
    if (split) {
        write_rgb = 1; write_a = 1; write_z = st->zwrite ? 1 : 0;
        afail_mask(st->afail, &write_rgb, &write_a, &write_z);
        depth = ztst | ((u32)write_z << 2) | ((u32)write_rgb << 3) | ((u32)write_a << 4);
        rec_one(topology, st, v, n, variant_of(st, 0, 1), depth);
    }
}

void ps2_vk_draw(int topology, const ps2_vk_state *st,
                 const ps2_vk_vertex *v, int n) {
    if (PS2_GSQ_DEFER()) ps2_gsq_sync("vk_draw");
    if (ps2_vk_before_record) ps2_vk_before_record();
    draw_inner(topology, st, v, n);
}

void ps2_vk_draw_2d(int topology, const ps2_vk_state *st,
                    const ps2_vk_vertex *v, int n) {
    if (PS2_GSQ_DEFER()) ps2_gsq_sync("vk_draw_2d");
    if (ps2_vk_before_record) ps2_vk_before_record();
    rec_native_mark = PS2_VK_NATIVE_2D;
    draw_inner(topology, st, v, n);
    rec_native_mark = 0;
}

void ps2_vk_native_draw(const ps2_vk_native *nd, const ps2_vk_vertex *v, int n) {
    (void)nd; (void)v; (void)n;   /* native sun/ghost passes: stage 3 */
}

/* Native mesh path. The recorder stores untransformed vertices and a
 * per-draw block (matrices, light, fog, program); the renderer draws them
 * with the mesh vertex program. AC5_MESH=0 in args.txt turns it off, which
 * sends VU1 output down the ordinary GS path as before. */
static int mesh_switch(void) {
    static int sw = -1;
    if (sw < 0) {
        const char *e = getenv("AC5_MESH");
        sw = !(e && *e == '0');
    }
    return sw;
}
int  ps2_vk_mesh_available(void) {
    return mesh_switch() && mesh_alloc_ok && mesh_ready && gl_ready && rec_mesh;
}
int  ps2_vk_mesh_material_ok(const ps2_vk_state *st) { (void)st; return 0; }
int  ps2_vk_mesh_store(const u8 *records, u32 n_verts, const u32 *indices,
                       u32 n_indices, u32 *vbase, u32 *ifirst) {
    mesh_arena *a = rec_mesh;
    if (vbase) *vbase = 0;
    if (ifirst) *ifirst = 0;
    if (!ps2_vk_mesh_available() || n_verts > 65535u) return -1;
    if (PS2_GSQ_DEFER()) {
        /* stored on the GS thread, into the arena its draws record against;
         * the worker gets tokens that thread turns back into offsets */
        u32 tok;
        for (u32 i = 0; i < n_indices; i++)
            if (indices[i] >= n_verts) return -1;
        tok = ps2_gsq_mstore(records, n_verts, indices, n_indices);
        if (vbase) *vbase = tok;
        if (ifirst) *ifirst = tok;
        return 0;
    }
    if (a->nrecs + n_verts > MESH_V_MAX || a->nidx + n_indices > MESH_I_MAX) {
        stat_mesh_overflow++;
        return -1;
    }
    for (u32 i = 0; i < n_indices; i++)
        if (indices[i] >= n_verts) return -1;
    {   mesh_rec *o = a->recs + a->nrecs;
        for (u32 i = 0; i < n_verts; i++, records += 64) {
            float f[4];
            s32 k[4];
            memcpy(o[i].pos, records, 16);
            memcpy(k, records + 16, 16);
            for (int j = 0; j < 4; j++) f[j] = (float)k[j];
            memcpy(o[i].nrm, f, 16);
            memcpy(k, records + 32, 16);
            for (int j = 0; j < 4; j++) f[j] = (float)k[j];
            memcpy(o[i].col, f, 16);
            memcpy(o[i].uv, records + 48, 16);
        }
    }
    {   u16 *o = a->idx + a->nidx;
        for (u32 i = 0; i < n_indices; i++) o[i] = (u16)indices[i];
    }
    *vbase = a->nrecs;
    *ifirst = a->nidx;
    a->nrecs += n_verts;
    a->nidx += n_indices;
    stat_mesh_stores++;
    stat_mesh_verts += n_verts;
    return 0;
}
int  ps2_vk_mesh_room(u32 n_verts, u32 n_indices, u32 n_draws) {
    const mesh_arena *a = rec_mesh;
    if (!ps2_vk_mesh_available()) return 0;
    if (PS2_GSQ_DEFER()) return 1;   /* the arena belongs to the GS thread; it drops what does not fit */
    return a->nrecs + n_verts <= MESH_V_MAX && a->nidx + n_indices <= MESH_I_MAX
        && a->nblocks + n_draws * 2u <= MESH_B_MAX
        && rec_ndraws + n_draws * 2u + 8u <= GL_MAX_DRAWS;
}
void ps2_vk_draw_mesh(const ps2_vk_state *st, const ps2_vk_mesh_block *b,
                      u32 vbase, u32 ifirst, u32 n_indices, int cull) {
    ps2_vk_draw_mesh_topo(st, b, vbase, ifirst, n_indices, cull, PS2_VK_TRIANGLES);
}
static void draw_mesh_topo_inner(const ps2_vk_state *st, const ps2_vk_mesh_block *b,
                                 u32 vbase, u32 ifirst, u32 n_indices, int cull, int topology);
extern u64 rn_tq_draw_us, rn_tq_draw_n;
void ps2_vk_draw_mesh_topo(const ps2_vk_state *st, const ps2_vk_mesh_block *b,
                           u32 vbase, u32 ifirst, u32 n_indices, int cull,
                           int topology) {
    extern int ac5_prof;
    if (PS2_GSQ_DEFER()) { ps2_gsq_drawmesh(st, b, vbase, ifirst, n_indices, cull, topology); return; }
    if (!ac5_prof) { draw_mesh_topo_inner(st, b, vbase, ifirst, n_indices, cull, topology); return; }
    u64 t0 = sceKernelGetProcessTimeWide();
    draw_mesh_topo_inner(st, b, vbase, ifirst, n_indices, cull, topology);
    rn_tq_draw_us += sceKernelGetProcessTimeWide() - t0;
    rn_tq_draw_n++;
}
static void draw_mesh_topo_inner(const ps2_vk_state *st, const ps2_vk_mesh_block *b,
                                 u32 vbase, u32 ifirst, u32 n_indices, int cull, int topology) {
    mesh_arena *a = rec_mesh;
    u32 vn = 0;
    if (!ps2_vk_mesh_available() || !n_indices) return;
    if (ifirst + n_indices > a->nidx || vbase >= a->nrecs) return;
    if (ps2_vk_before_record) ps2_vk_before_record();
    if (a->nblocks >= MESH_B_MAX) { stat_mesh_overflow++; return; }
    for (u32 i = 0; i < n_indices; i++)
        if (a->idx[ifirst + i] >= vn) vn = a->idx[ifirst + i] + 1u;
    if (vbase + vn > a->nrecs) return;
    a->blocks[a->nblocks] = *b;
    rec_mesh_pending.vbase = vbase;
    rec_mesh_pending.ifirst = ifirst;
    rec_mesh_pending.icount = n_indices;
    rec_mesh_pending.block = a->nblocks++;
    rec_mesh_pending.vn = vn;
    rec_mesh_pending.cull = (u8)((cull & PS2_VK_MESH_CULL) && topology == PS2_VK_TRIANGLES);
    rec_native_mark = PS2_VK_NATIVE_MESH;
    draw_inner(topology, st, NULL, (int)n_indices);
    rec_native_mark = 0;
}

/* ---------------------------------------------------------------- textures */

static s32 tex_first(u64 key) { return tex_head[tex_bucket(key)]; }

static void tex_index_add(u32 i) {
    u32 b = tex_bucket(textures[i].key);
    tex_next[i] = tex_head[b];
    tex_head[b] = (s32)i;
}

static void tex_unlink(u32 i) {
    u32 b = tex_bucket(textures[i].key);
    s32 *pp = &tex_head[b];
    while (*pp >= 0) {
        if ((u32)*pp == i) { *pp = tex_next[i]; return; }
        pp = &tex_next[*pp];
    }
}

static u64 stat_evictions;

/* Free entry i's pixels and unhook it; the slot becomes dead (rgba == NULL)
 * until tex_reuse() fills it again. Only called for entries no display
 * list in flight can reference (unused for more than 3 fields). */
void vita_pin_current(int core);
/* ---- texture decoding on the helper core (ps2_vk_texture_async) ---- */
static struct { int on; u8 *dst; int swap; u32 idx; } tex_defer;
#define TJ_N 256u
typedef struct { void (*fill)(void *, u8 *); void *job; u8 *dst; u32 idx; int swap; } tex_jobq;
static tex_jobq tjq[TJ_N];
static volatile u32 tj_head;                   /* next to add (recorder) */
static volatile u32 tj_claim;                  /* next to hand to a decoder */
static volatile u32 tj_submitted, tj_done, pub_jobs, rep_jobs;
static SceUID tj_sema = -1, tj_done_sema = -1;
volatile int rn_busy_state;   /* renderer: 0 waiting, 1 drawing, 2 decoding (for the GS sampler) */
static u64 stat_tj_jobs, stat_tj_us, stat_tj_waits_us, stat_tj_stolen, stat_tj_jobs2;
/* Several decoders (build 89): the helper on core 1, a second helper on
 * core 0 that runs while the game thread waits for the graphics, and the
 * renderer itself, which decodes the jobs its frame needs instead of
 * waiting for them (it waited 50-90 ms a second in build 88). Jobs finish
 * out of order, so "every job before k is done" is the smallest of the
 * claim counter and the jobs still in a decoder's hands. */
#define TJ_DEC 3
#define TJ_NONE 0xFFFFFFFFu
static volatile u32 tj_busy[TJ_DEC] = { TJ_NONE, TJ_NONE, TJ_NONE };
static u32 tj_prefix(void) {
    u32 c = __atomic_load_n(&tj_claim, __ATOMIC_SEQ_CST), m = c;
    for (int k = 0; k < TJ_DEC; k++) {
        u32 b = __atomic_load_n(&tj_busy[k], __ATOMIC_SEQ_CST);
        if (b != TJ_NONE && (s32)(b - m) < 0) m = b;
    }
    return m;
}
/* claim and run one job below 'limit' (any job if limit is TJ_NONE) */
static int tj_run_one(int me, u32 limit) {
    u32 c;
    for (;;) {
        c = __atomic_load_n(&tj_claim, __ATOMIC_SEQ_CST);
        if (c == __atomic_load_n(&tj_head, __ATOMIC_ACQUIRE)) return 0;
        if (limit != TJ_NONE && (s32)(c - limit) >= 0) return 0;
        __atomic_store_n(&tj_busy[me], c, __ATOMIC_SEQ_CST);
        if (__atomic_compare_exchange_n(&tj_claim, &c, c + 1u, 0, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST))
            break;
    }
    {   tex_jobq *q = &tjq[c % TJ_N];
        u64 t0 = sceKernelGetProcessTimeWide();
        gl_texture *t = &textures[q->idx];
        q->fill(q->job, q->dst);
        free(q->job);
        if (q->swap) {
            u8 *now_live = t->rgba_alt;
            t->rgba_alt = t->rgba;
            __atomic_store_n(&t->rgba, now_live, __ATOMIC_RELEASE);
        }
        __atomic_add_fetch(&t->generation, 1, __ATOMIC_RELEASE);
        __atomic_sub_fetch(&t->pending, 1, __ATOMIC_RELEASE);
        {   u64 d = sceKernelGetProcessTimeWide() - t0;
            __atomic_add_fetch(&stat_tj_us, d, __ATOMIC_RELAXED);
        }
        __atomic_add_fetch(&stat_tj_jobs, 1, __ATOMIC_RELAXED);
        if (me == 1) stat_tj_jobs2++;
        if (me == 2) stat_tj_stolen++;
        __atomic_store_n(&tj_busy[me], TJ_NONE, __ATOMIC_SEQ_CST);
        __atomic_add_fetch(&tj_done, 1, __ATOMIC_RELEASE);
        sceKernelSignalSema(tj_done_sema, 1);
    }
    return 1;
}
static void *tex_decoder_main(void *arg) {
    int me = (int)(intptr_t)arg;
    {   extern int ps2_gsq_on;
        extern void vita_set_priority(int prio);
        if (me == 0) {
            /* core 1 by default: with the GS on its own thread on core 2, the
             * graphics worker's core is the one with time to spare */
            const char *c = getenv("AC5_TEXHELPER_CORE");
            /* core 0 just above the game thread (build 106): below the GS on
             * core 2 it was starved and the renderer waited 120-200 ms per 5 s
             * for decodes (build 105); a decode is a few ms, the game has slack */
            const char *p = getenv("AC5_TEXHELPER_PRIO");
            vita_pin_current(c ? atoi(c) : 0);
            vita_set_priority(p ? atoi(p) : 145);
        } else {
            /* core 0, just below the game thread: it runs in the game's
             * waits for the graphics (a quarter to a third of the time) */
            const char *c = getenv("AC5_TEXHELPER2_CORE"), *p = getenv("AC5_TEXHELPER2_PRIO");
            vita_pin_current(c ? atoi(c) : 1);   /* core 1 below the worker and renderer (build 106) */
            vita_set_priority(p ? atoi(p) : 142);
        }
    }
    for (;;) {
        sceKernelWaitSema(tj_sema, 1, NULL);
        while (tj_run_one(me, TJ_NONE)) {}
    }
    return NULL;
}
static int tex_decoder_start(void) {
    static int state;
    pthread_t th;
    pthread_attr_t a;
    int two;
    if (state) return state > 0;
    state = -1;
    tj_sema = sceKernelCreateSema("ac5_texjobs", 0, 0, 0x7FFFFFFF, NULL);
    tj_done_sema = sceKernelCreateSema("ac5_texdone", 0, 0, 0x7FFFFFFF, NULL);
    if (tj_sema < 0 || tj_done_sema < 0) return 0;
    pthread_attr_init(&a);
    pthread_attr_setstacksize(&a, 64 * 1024);
    if (pthread_create(&th, &a, tex_decoder_main, (void *)(intptr_t)0) != 0) return 0;
    pthread_detach(th);
    {   const char *e = getenv("AC5_TEXHELPER2_CORE");
        two = !(e && atoi(e) < 0);
    }
    if (two && pthread_create(&th, &a, tex_decoder_main, (void *)(intptr_t)1) == 0) pthread_detach(th);
    else two = 0;
    state = 1;
    ps2_log("gl: texture decoding on helper threads (core 1%s; AC5_TEXHELPER_CORE, AC5_TEXHELPER2_CORE=-1 drops the second) "
            "and on the renderer while it waits (AC5_TEX_STEAL=0 stops that); AC5_TEX_ASYNC=0 keeps it on the GS",
            two ? " and core 0" : "");
    return 1;
}
/* wait for every decode job submitted so far (CPU readers of pixels) */
static void tex_jobs_drain(void) {
    u32 want = __atomic_load_n(&tj_submitted, __ATOMIC_ACQUIRE);
    while ((s32)(tj_prefix() - want) < 0) {
        SceUInt to = 2000;
        sceKernelWaitSema(tj_done_sema, 1, &to);
    }
}

static void tex_evict(u32 i) {
    gl_texture *t = &textures[i];
    u8 *live = t->rgba;
    tex_unlink(i);
    __atomic_store_n(&t->rgba, (u8 *)NULL, __ATOMIC_RELEASE);
    free(live);
    free(t->rgba_alt);
    t->rgba_alt = NULL;
    tex_bytes -= (size_t)t->w * t->h * 4u * 2u;
    t->key = 0;
    t->hash = 0;
    stat_evictions++;
}

/* Fill slot i (live or dead) with a new texture. The renderer re-uploads it
 * when it sees the generation change, re-sizing the GL texture if needed. */
static u32 tex_reuse(u32 i, u64 key, u32 hash, const u8 *rgba, u32 w, u32 h) {
    gl_texture *t = &textures[i];
    size_t bytes = (size_t)w * h * 4u;
    if (t->rgba) tex_evict(i);
    u8 *a = (u8 *)malloc(bytes), *b = (u8 *)malloc(bytes);
    if (!a || !b) { free(a); free(b); return 0xFFFFFFFFu; }
    if (tex_defer.on) { tex_defer.dst = a; tex_defer.swap = 0; tex_defer.idx = i; }
    else memcpy(a, rgba, bytes);
    t->rgba_alt = b;
    t->key = key;
    t->hash = hash;
    t->w = w;
    t->h = h;
    t->rec_seq = tex_rec_seq;
    tex_bytes += bytes * 2u;
    tex_index_add(i);
    __atomic_store_n(&t->rgba, a, __ATOMIC_RELEASE);
    __atomic_add_fetch(&t->generation, 1, __ATOMIC_RELEASE);
    return i;
}

static u32 tex_lru(void) {
    u32 victim = 0xFFFFFFFFu, oldest = 3u;
    for (u32 k = 0; k < ntextures; k++) {
        u32 age = tex_rec_seq - textures[k].rec_seq;
        if (textures[k].rgba && !textures[k].pending && age > oldest) { oldest = age; victim = k; }
    }
    return victim;
}

u32 ps2_vk_texture(u64 key, u32 hash, const u8 *rgba, u32 w, u32 h) {
    size_t bytes = (size_t)w * h * 4u;
    u32 i, spare = 0xFFFFFFFFu;
    if (PS2_GSQ_DEFER()) ps2_gsq_sync("vk_texture");
    if (!gl_ready || !w || !h) return 0xFFFFFFFFu;
    for (s32 c = tex_first(key); c >= 0; c = tex_next[c]) {
        i = (u32)c;
        if (textures[i].key != key || textures[i].w != w || textures[i].h != h)
            continue;
        if (textures[i].hash == hash) {
            textures[i].rec_seq = tex_rec_seq;
            return i;
        }
        if (spare == 0xFFFFFFFFu && textures[i].rgba && !textures[i].pending
            && (u32)(tex_rec_seq - textures[i].rec_seq) > 2u)
            spare = i;
    }
    if (spare != 0xFFFFFFFFu && tex_defer.on) {
        i = spare;
        tex_defer.dst = textures[i].rgba_alt; tex_defer.swap = 1; tex_defer.idx = i;
        textures[i].hash = hash;
        textures[i].rec_seq = tex_rec_seq;
        return i;
    }
    if (spare != 0xFFFFFFFFu) {
        i = spare;
        memcpy(textures[i].rgba_alt, rgba, bytes);
        {   u8 *now_live = textures[i].rgba_alt;
            textures[i].rgba_alt = textures[i].rgba;
            __atomic_store_n(&textures[i].rgba, now_live, __ATOMIC_RELEASE);
        }
        textures[i].hash = hash;
        textures[i].rec_seq = tex_rec_seq;
        __atomic_add_fetch(&textures[i].generation, 1, __ATOMIC_RELEASE);
        return i;
    }
    /* Full: evict least recently used entries (never one a list in flight
     * may still reference) until the new texture fits. */
    while (tex_bytes + bytes * 2u > GL_TEX_BYTES) {
        u32 victim = tex_lru();
        if (victim == 0xFFFFFFFFu) {
            static int said;
            if (!said++)
                ps2_log("gl: texture store full and nothing old enough to evict; "
                        "a %ux%u texture was dropped", w, h);
            return 0xFFFFFFFFu;
        }
        tex_evict(victim);
    }
    /* Reuse a dead slot before growing the table. */
    for (u32 k = 0; k < ntextures; k++)
        if (!textures[k].rgba && !textures[k].key) return tex_reuse(k, key, hash, rgba, w, h);
    if (ntextures >= GL_MAX_TEXTURES) {
        u32 victim = tex_lru();
        if (victim == 0xFFFFFFFFu) return 0xFFFFFFFFu;
        return tex_reuse(victim, key, hash, rgba, w, h);
    }
    i = ntextures;
    textures[i].key = key;
    textures[i].hash = hash;
    textures[i].w = w;
    textures[i].h = h;
    textures[i].rgba = (u8 *)malloc(bytes);
    textures[i].rgba_alt = (u8 *)malloc(bytes);
    if (!textures[i].rgba || !textures[i].rgba_alt) {
        free(textures[i].rgba);
        free(textures[i].rgba_alt);
        textures[i].rgba = textures[i].rgba_alt = NULL;
        return 0xFFFFFFFFu;
    }
    if (tex_defer.on) { tex_defer.dst = textures[i].rgba; tex_defer.swap = 0; tex_defer.idx = i; }
    else memcpy(textures[i].rgba, rgba, bytes);
    textures[i].uploaded_gen = 0;
    textures[i].rec_seq = tex_rec_seq;
    textures[i].name = 0;
    textures[i].filter = -1;
    __atomic_store_n(&textures[i].generation, 1u, __ATOMIC_RELEASE);
    tex_index_add(i);
    tex_bytes += bytes * 2u;
    __atomic_store_n(&ntextures, i + 1u, __ATOMIC_RELEASE);
    return i;
}

u32 ps2_vk_texture_async(u64 key, u32 hash, u32 w, u32 h,
                         void (*fill)(void *job, u8 *rgba), void *job) {
    u32 i;
    if (!gl_ready || !w || !h || !tex_decoder_start()) return 0xFFFFFFFFu;
    if (tj_head - tj_prefix() >= TJ_N - 1u) return 0xFFFFFFFFu;  /* queue full */
    tex_defer.on = 1; tex_defer.dst = NULL; tex_defer.swap = 0; tex_defer.idx = 0;
    i = ps2_vk_texture(key, hash, NULL, w, h);
    tex_defer.on = 0;
    if (i == 0xFFFFFFFFu) return i;
    if (!tex_defer.dst) { free(job); return i; }       /* already stored: nothing to decode */
    {   tex_jobq *q = &tjq[tj_head % TJ_N];
        q->fill = fill; q->job = job; q->dst = tex_defer.dst; q->idx = tex_defer.idx; q->swap = tex_defer.swap;
        __atomic_add_fetch(&textures[i].pending, 1, __ATOMIC_RELEASE);
        __atomic_store_n(&tj_head, tj_head + 1u, __ATOMIC_RELEASE);
        __atomic_add_fetch(&tj_submitted, 1, __ATOMIC_RELEASE);
        sceKernelSignalSema(tj_sema, 1);
    }
    return i;
}

/* The native (mesh-path) texture variants exist for stage 2. Until then they
 * report "no texture", which is what the mesh path being off implies. */
u32 ps2_vk_texture_native(u32 src, u32 x, u32 y, u32 w, u32 h) {
    (void)src; (void)x; (void)y; (void)w; (void)h; return 0xFFFFFFFFu;
}
u32 ps2_vk_texture_native_find(u64 key, u32 hash, u32 w, u32 h) {
    (void)key; (void)hash; (void)w; (void)h; return 0xFFFFFFFFu;
}
u32 ps2_vk_texture_native_rgba(u64 key, u32 hash, const u8 *rgba, u32 w, u32 h) {
    (void)key; (void)hash; (void)rgba; (void)w; (void)h; return 0xFFFFFFFFu;
}
int ps2_vk_texture_is(u32 idx, u64 key, u32 hash) {
    return idx < ntextures && textures[idx].key == key && textures[idx].hash == hash;
}
const u8 *ps2_vk_texture_pixels(u32 idx) {
    if (idx < ntextures && textures[idx].pending) tex_jobs_drain();
    return idx < ntextures ? __atomic_load_n(&textures[idx].rgba, __ATOMIC_ACQUIRE) : NULL;
}
void ps2_vk_texture_size(u32 idx, u32 *w, u32 *h) {
    if (w) *w = idx < ntextures ? textures[idx].w : 0u;
    if (h) *h = idx < ntextures ? textures[idx].h : 0u;
}
void ps2_vk_invalidate(u32 base, u32 size) { (void)base; (void)size; }

/* ----------------------------------------------------------------- present */

int ps2_vk_present(u32 disp_x, u32 disp_y, u32 disp_w, u32 disp_h, u32 rt) {
    int replace;
    if (!gl_ready) return 0;
{ extern volatile unsigned char ps2_gs_ph; ps2_gs_ph = 61; }
    if (ps2_vk_before_record) ps2_vk_before_record();
    stat_presents++;
    { extern volatile unsigned char ps2_gs_ph; ps2_gs_ph = 66; }
    pthread_mutex_lock(&list_lock);
    { extern volatile unsigned char ps2_gs_ph; ps2_gs_ph = 67; }
    if (frame_requested && renderer_running > 0 && ps2_vk_lockstep) {
        while (frame_requested && renderer_running > 0)
            pthread_cond_wait(&list_cv, &list_lock);
    }
    replace = frame_requested && renderer_running > 0;
    if (replace) stat_dropped++;
    pub_verts = rec_verts;
    pub_draws = rec_draws;
    pub_nverts = rec_nverts;
    pub_ndraws = rec_ndraws;
    {   int next = replace ? pub_slot : (0 + 1 + 2) - pub_slot - rec_slot;
        pub_slot = rec_slot;
        rec_slot = next;
    }
    pub_mesh = rec_mesh;
    rec_verts = slot_verts[rec_slot];
    rec_draws = slot_draws[rec_slot];
    rec_mesh = &slot_mesh[rec_slot];
    rec_mesh->nrecs = rec_mesh->nidx = rec_mesh->nblocks = 0;
    rec_nverts = 0;
    rec_ndraws = 0;
    if (disp_w) present_w = disp_w;
    if (disp_h) present_h = disp_h;
    present_x = disp_x;
    present_y = disp_y;
    present_blank = rt == PS2_VK_PRESENT_BLANK;
    present_rt = rt < RT_SLOTS ? rt : 0;
    present_ov_rt = ov_pend_rt;
    present_ov_x = ov_pend_x; present_ov_y = ov_pend_y; present_ov_a = ov_pend_a;
    ov_pend_rt = -1;
    if (!replace) tex_rec_seq++;
    pub_jobs = __atomic_load_n(&tj_submitted, __ATOMIC_ACQUIRE);
    frame_requested = 1;
    frame_done = 0;
    {   u64 now = sceKernelGetProcessTimeWide();
        if (rl_last_pub) {
            u64 g = now - rl_last_pub;
            if (g > rl_gap_max) rl_gap_max = g;
            rl_gap_hist[g < 10000 ? 0 : g < 20000 ? 1 : g < 40000 ? 2 : g < 80000 ? 3 : 4]++;
        }
        rl_last_pub = now;
    }
    pthread_cond_broadcast(&list_cv);
    if (frame_sema >= 0) {
        SceKernelSemaInfo si;
        si.size = sizeof si;
        if (sceKernelGetSemaInfo(frame_sema, &si) < 0 || si.currentCount < 1)
            sceKernelSignalSema(frame_sema, 1);
    }
    if (ps2_vk_lockstep) {
        while (!frame_done && renderer_running > 0)
            pthread_cond_wait(&list_cv, &list_lock);
    }
    pthread_mutex_unlock(&list_lock);
    if (rec_overflow) {
        static int said;
        if (said++ < 8)
            ps2_log("gl: display list overflowed (%u verts / %u draws); the "
                    "field was truncated", GL_MAX_VERTS, GL_MAX_DRAWS);
        rec_overflow = 0;
    }
    return 1;
}

/* ============================================================ renderer side */

typedef struct gl_target {
    int created;
    GLuint attached;    /* colour texture currently on fbo (see render_list) */
    GLuint color, fbo;  /* vitaGL gives every framebuffer its own hidden
                         * depth buffer; renderbuffers are ignored */
    u32 dirty_seq;      /* bumped on every draw into it */
    int depth_kept;     /* its depth buffer is loaded/stored across scenes */
} gl_target;

/* One shared snapshot for self-reads (a draw sampling the target it draws
 * into): snap_slot/snap_seq say whose contents it holds. */
static GLuint snap_tex, snap_fbo;
static int snap_slot = -1;
static u32 snap_seq, snaps_this_frame;

static gl_target rts[RT_SLOTS];
static u32 rt_w = 640, rt_h = 448;
/* Internal resolution: GL storage is rt_w x rt_h scaled by this (args.txt
 * AC5_GL_SCALE=0.5 .. 1). Positions stay in GS pixels; the viewport does
 * the scaling, and render-target sampling is normalised, so nothing else
 * needs to know. */
static float rt_scale = 0.625f;   /* default since build 106: 400x280, divides evenly; AC5_GL_SCALE overrides */
#define PW ((GLsizei)((float)rt_w * rt_scale + 0.5f))
#define PH ((GLsizei)((float)rt_h * rt_scale + 0.5f))
static u64 stat_fill;
/* where draws go and where the renderer's time goes (since the last report) */
enum { SK_SHUFFLE, SK_NATIVE, SK_NORT, SK_NOSNAP, SK_NOPROG, SK_SCISSOR, SK_N };
static u64 stat_skip[SK_N], stat_drawn_tri, stat_drawn_line, stat_drawn_point;
static u64 t_submit_us, t_swap_us, t_frames_timed;
     /* estimated pixels covered by draws (GS pixels) */

static GLuint white_tex;
static GLuint ov_prog;
static GLint ov_u_mode;
static u64 stat_overlays;
static GLuint copy_prog;
static GLint copy_u_tex;

typedef struct gl_variant {
    GLuint prog;
    GLint u_xform, u_texinfo, u_clamp, u_misc, u_fog, u_tex, u_f1, u_f2, u_f3, u_blk;
    float uc[8][4];     /* last value set for each uniform (uset bit) */
    u32 uset;
} gl_variant;

/* Set a vec4 uniform only if it changed for this program: vitaGL rebuilds
 * the uniform buffer for a draw when any uniform is set, and missions
 * issue ~1,300 draws a frame. */
static u64 stat_uniform_skips;
static inline void set_u4(gl_variant *v, int slot, GLint loc, const float *x) {
    if (loc < 0) return;
    if ((v->uset & (1u << slot)) && !memcmp(v->uc[slot], x, 16)) { stat_uniform_skips++; return; }
    memcpy(v->uc[slot], x, 16);
    v->uset |= 1u << slot;
    glUniform4fv(loc, 1, x);
}
/* [textured][vertex program: 0 triangles/lines, 1 points, 2 mesh] */
static gl_variant progs[2][4];
static int nvariants;
static GLuint vs_tri, vs_point, vs_mesh, vs_clip;

static const char *VS_SRC =
    "void main(float3 aPos, float4 aColor, float3 aStq, float aFog,\n"
    "          uniform float4 uXform,\n"
    "          float4 out vPos : POSITION,\n"
    "          float4 out vColor : TEXCOORD0,\n"
    "          float3 out vStq : TEXCOORD1,\n"
    "          float out vFog : TEXCOORD2\n"
    "#ifdef POINTS\n"
    "          , float out vSize : PSIZE\n"
    "#endif\n"
    "          ) {\n"
    "    vPos = float4(aPos.x * uXform.x - 1.0, aPos.y * uXform.y - 1.0,\n"
    "                  aPos.z * 2.0 - 1.0, 1.0);\n"
    "    vColor = aColor / 128.0;\n"
    "    vStq = aStq;\n"
    "    vFog = aFog;\n"
    "#ifdef POINTS\n"
    "    vSize = uXform.z;\n"
    "#endif\n"
    "}\n";

/* Pre-transformed mesh vertices: the renderer runs the mesh program on the
 * CPU for small meshes, so runs of them with the same state go to the GPU
 * as one draw (vitaGL's cost is per draw, not per vertex). */
static const char *CLIP_VS_SRC =
    "void main(float4 aPos, float4 aColor, float3 aStq, float aFog,\n"
    "          float4 out vPos : POSITION,\n"
    "          float4 out vColor : TEXCOORD0,\n"
    "          float3 out vStq : TEXCOORD1,\n"
    "          float out vFog : TEXCOORD2) {\n"
    "    vPos = aPos;\n"
    "    vColor = aColor;\n"
    "    vStq = aStq;\n"
    "    vFog = aFog;\n"
    "}\n";

/* Native mesh vertex program: runtime/shaders/mesh.vert in Cg. uB is the
 * draw's ps2_vk_mesh_block as float4s:
 *   0-3 m (to GS screen space), 4-7 c (clip), 8-10 l, 11-14 r, 15-17 e,
 *   18 fog0 fog1 zscale zmax, 19 xoff yoff spec_alpha -, 20 mode pass clip prog,
 *   21-24 p.
 * Positions follow VS_SRC: GS pixels times 2/size minus one, and GL depth
 * 2z-1 (here in clip space, so 2Z - w). Colours come out /128 like VS_SRC.
 * col & 255 is done in float (the values are integers well under 2^24). */
static const char *MESH_VS_SRC =
    "float4 lo8(float4 c) { return c - 256.0 * floor(c / 256.0); }\n"
    "void main(float4 aPos, float4 aNrm, float4 aCol, float4 aUv,\n"
    "          uniform float4 uXform, uniform float4 uB[25],\n"
    "          float4 out vPos : POSITION,\n"
    "          float4 out vColor : TEXCOORD0,\n"
    "          float3 out vStq : TEXCOORD1,\n"
    "          float out vFog : TEXCOORD2) {\n"
    "    float3 p = aPos.xyz;\n"
    "    float4 F = uB[18], O = uB[19], K = uB[20];\n"
    "    float4 s = uB[0] * p.x + uB[1] * p.y + uB[2] * p.z + uB[3];\n"
    "    float w = s.w;\n"
    "    float zlim = min(16777215.0, F.w);\n"
    "    float X = (s.x + (0.5 - O.x) * w) * uXform.x - w;\n"
    "    float Y = (s.y + (0.5 - O.y) * w) * uXform.y - w;\n"
    "    float Z = min(max(s.z * F.z, 0.0), zlim * max(w, 0.0)) / F.w;\n"
    "    vPos = float4(X, Y, 2.0 * Z - w, w);\n"
    "    float4 cc = uB[4] * p.x + uB[5] * p.y + uB[6] * p.z + uB[7];\n"
    "    float fog = clamp(F.x + w * F.y, 0.0, 255.0);\n"
    "    vFog = fog / 255.0;\n"
    "    float prog = K.w, pss = K.y;\n"
    "    if (prog > 4.5) {\n"
    "        vPos = float4(aPos.x * uXform.x - 1.0, aPos.y * uXform.y - 1.0, aPos.z * 2.0 - 1.0, 1.0);\n"
    "        vStq = aUv.xyz;\n"
    "        vColor = aCol / 128.0;\n"
    "        vFog = aUv.w;\n"
    "    } else if (prog > 3.5) {\n"
    "        vStq = aUv.xyz;\n"
    "        vColor = lo8(aCol) / 128.0;\n"
    "    } else if (prog > 2.5) {\n"
    "        float iw = 1.0 / w;\n"
    "        float3 c = s.xyz * iw;\n"
    "        float2 corner = c.xy + aNrm.xy * uB[21].xy * aUv.z * iw;\n"
    "        float zg = clamp(floor(floor(c.z * 16.0) / 8.0), uB[21].z, zlim);\n"
    "        vPos = c.z > 0.0\n"
    "            ? float4((corner.x + 0.5 - O.x) * uXform.x - 1.0,\n"
    "                     (corner.y + 0.5 - O.y) * uXform.y - 1.0, 2.0 * zg / F.w - 1.0, 1.0)\n"
    "            : float4(2.0, 2.0, 2.0, 1.0);\n"
    "        vStq = float3(aUv.xy, 1.0);\n"
    "        vColor = lo8(aCol) / 128.0;\n"
    "        vFog = 1.0;\n"
    "    } else if (prog > 1.5) {\n"
    "        if (pss < 0.5) {\n"
    "            vStq = float3(aUv.xy, 1.0);\n"
    "            vColor = lo8(aCol) / 128.0;\n"
    "        } else {\n"
    "            vStq = float3(aUv.xy * 2.0, 1.0);\n"
    "            float g = clamp((uB[21].z - cc.w) * uB[21].w * 128.0, 0.0, 128.0);\n"
    "            vColor = float4(g, g, g, 128.0) / 128.0;\n"
    "        }\n"
    "    } else if (prog > 0.5) {\n"
    "        float2 uv;\n"
    "        float4 col;\n"
    "        if (pss < 0.5) {\n"
    "            uv = aUv.xy + aUv.zw;\n"
    "            col = lo8(aCol);\n"
    "        } else if (pss < 1.5) {\n"
    "            uv = aUv.xy * 16.0;\n"
    "            float g = clamp((uB[21].z - cc.w) * uB[21].w * 128.0, 0.0, 128.0);\n"
    "            col = float4(g, g, g, uB[24].x);\n"
    "        } else if (pss < 2.5) {\n"
    "            uv = aUv.xy * 16.0 + uB[22].xy;\n"
    "            float g = clamp((uB[22].z - cc.w) * uB[22].w * 128.0, 0.0, 128.0);\n"
    "            col = float4(g, g, g, fog * 0.5);\n"
    "        } else {\n"
    "            uv = (p.xz - uB[23].xz) * uB[23].w;\n"
    "            col = float4(fog * 0.5, fog * 0.5, fog * 0.5, fog * 0.5);\n"
    "        }\n"
    "        vStq = float3(uv, 1.0);\n"
    "        vColor = col / 128.0;\n"
    "    } else if (pss < 0.5) {\n"
    "        vStq = float3(aUv.xy, 1.0);\n"
    "        if (K.x < 0.5) {\n"
    "            vColor = lo8(aCol) / 128.0;\n"
    "        } else {\n"
    "            float3 n = aNrm.xyz / 4096.0;\n"
    "            float4 l = float4(uB[8].xyz * n.x + uB[9].xyz * n.y + uB[10].xyz * n.z, 1.0);\n"
    "            l = l * l * l;\n"
    "            l.xyz *= 2.0;\n"
    "            l = max(l, 0.0);\n"
    "            float4 li = max(uB[11] * l.x + uB[12] * l.y + uB[13] * l.z + uB[14] * l.w, 0.0);\n"
    "            float3 c = min(aCol.xyz * li.xyz, float3(255.0, 255.0, 255.0));\n"
    "            vColor = float4(c, 128.0) / 128.0;\n"
    "        }\n"
    "    } else {\n"
    "        float3 n = aNrm.xyz / 4096.0;\n"
    "        float3 nb = uB[15].xyz * n.x + uB[16].xyz * n.y + uB[17].xyz * n.z;\n"
    "        vStq = float3((nb.x + 1.0) * 0.5, (nb.z + 1.0) * 0.5, 1.0);\n"
    "        float sp = min(max(nb.y, 0.0) * 1280.0, 128.0);\n"
    "        vColor = float4(sp, sp, sp, O.z) / 128.0;\n"
    "    }\n"
    "}\n";

/* Fragment program: one textured and one untextured build, with every other
 * piece of GS state passed as uniforms. All four programs (x points/tris)
 * are compiled once at start-up, while memory is free: compiling per state
 * combination at run time failed once vitaGL's pool was under pressure.
 * Uniforms:
 *   uF1 = tfx, tcc, fst, fge
 *   uF2 = ate, atst, afmode (1 keep = discard on fail, 2 = discard on pass), outfix
 *   uF3 = fba, point sampling, half-texel offset, -
 * Note: Cg reserves `pass` (and `technique`...), hence `ok`. */
/* One source, two uses. Built with only TME defined it is the "uber"
 * shader: every GS mode is a uniform and the shader branches per pixel.
 * Built with the K_* macros defined as constants for one variant key, the
 * Cg compiler folds every branch away -- several times cheaper per pixel,
 * which matters on the SGX: missions draw the screen over many times. */
static const char *FS_SRC =
    "#ifndef K_TFX\n"
    "#define K_TFX uF1.x\n#define K_TCC uF1.y\n#define K_FST uF1.z\n#define K_FGE uF1.w\n"
    "#define K_ATE uF2.x\n#define K_ATST uF2.y\n#define K_AFMODE uF2.z\n#define K_OUTFIX uF2.w\n"
    "#define K_FBA uF3.x\n#define K_POINT uF3.y\n#define K_HALF uF3.z\n"
    "#define K_WMS uMisc.z\n#define K_WMT uMisc.w\n"
    "#endif\n"
    "#ifndef K_TRIM\n#define K_TRIM 1.0\n#endif\n"
    "#ifndef K_NREP\n#define K_NREP 0.0\n#endif\n"
    "float wrapc(float t, float mode, float lo, float hi, float size) {\n"
    "    if (mode < 0.5) return t - size * floor(t / max(size, 1.0));\n"
    "    if (mode < 1.5) return clamp(t, 0.0, max(size - 1.0, 0.0));\n"
    "    if (mode < 2.5) return clamp(t, lo, max(hi, lo));\n"
    "    float p = lo + 1.0;\n"
    "    float f = floor(t);\n"
    "    return f - p * floor(f / p) + hi + (t - f);\n"
    "}\n"
    "float4 main(float4 vColor : TEXCOORD0, float3 vStq : TEXCOORD1,\n"
    "            float vFog : TEXCOORD2,\n"
    "            uniform sampler2D uTex,\n"
    "            uniform float4 uTexInfo,\n"   /* tex_w, tex_h, 1/img_w, 1/img_h */
    "            uniform float4 uClamp,\n"     /* minu, maxu, minv, maxv */
    "            uniform float4 uMisc,\n"      /* aref, fix, wms, wmt */
    "            uniform float4 uFog,\n"
    "            uniform float4 uF1, uniform float4 uF2, uniform float4 uF3) : COLOR {\n"
    "    float4 c = vColor;\n"
    "#ifdef TME\n"
    "    float2 tc = K_FST > 0.5 ? vStq.xy\n"
    "                            : (vStq.xy / max(vStq.z, 1e-9)) * uTexInfo.xy;\n"
    "    if (K_HALF > 0.5 && K_POINT < 0.5) tc -= 0.5;\n"
    /* K_NREP: a texture stored at exactly its GS size with GL_REPEAT, where
     * the sampler's own wrap does what wrapc's REPEAT case does */
    "    if (!(K_NREP > 0.5 && K_WMS < 0.5)) tc.x = wrapc(tc.x, K_WMS, uClamp.x, uClamp.y, uTexInfo.x);\n"
    "    if (!(K_NREP > 0.5 && K_WMT < 0.5)) tc.y = wrapc(tc.y, K_WMT, uClamp.z, uClamp.w, uTexInfo.y);\n"
    /* The stored picture can be smaller than the GS texture (only the
     * uploaded part is kept). Texels past it are not in the GL texture, and
     * GL_REPEAT used to fold them back onto the opposite edge -- the line
     * down the side of the fading title logo. Treat them as empty and keep
     * filtering inside the picture. */
    "    float outside = 0.0;\n"
    "    if (K_TRIM > 0.5) {\n"         /* only variants for textures stored smaller */
    "        float2 isz = 1.0 / uTexInfo.zw;\n"
    "        float2 trim = step(isz + 0.5, uTexInfo.xy);\n"
    "        float2 past = trim * step(isz - 0.5, tc);\n"
    "        outside = max(past.x, past.y);\n"
    "        tc = lerp(tc, min(tc, isz - 1.0), trim);\n"
    "    }\n"
    "    float2 uv = K_POINT > 0.5 ? (floor(tc + 1.0 / 32.0) + 0.5) * uTexInfo.zw\n"
    "                              : (tc + 0.5) * uTexInfo.zw;\n"
    "    float4 t = tex2D(uTex, uv) * (1.0 - outside);\n"
    "    float af = c.a;\n"
    "    float tfx = K_TFX;\n"
    "    float3 rgb;\n"
    "    float a;\n"
    "    if (tfx < 0.5)      { rgb = t.rgb * c.rgb; a = t.a * af; }\n"
    "    else if (tfx < 1.5) { rgb = t.rgb;         a = t.a; }\n"
    "    else if (tfx < 2.5) { rgb = t.rgb * c.rgb + af * (128.0 / 255.0); a = t.a + af; }\n"
    "    else                { rgb = t.rgb * c.rgb + af * (128.0 / 255.0); a = t.a; }\n"
    "    c.rgb = rgb;\n"
    "    c.a = K_TCC > 0.5 ? a : af;\n"
    "#else\n"
    "    c.rgb *= 128.0 / 255.0;\n"
    "#endif\n"
    "    if (K_FGE > 0.5) c.rgb = lerp(uFog.rgb, c.rgb, saturate(vFog));\n"
    "    if (K_ATE > 0.5) {\n"
    "        float aref = uMisc.x;\n"
    "        float m = K_ATST;\n"
    "        float d = c.a - aref;\n"
    "        float ok;\n"
    "        if (m < 0.5)      ok = 0.0;\n"
    "        else if (m < 1.5) ok = 1.0;\n"
    "        else if (m < 2.5) ok = d < 0.0 ? 1.0 : 0.0;\n"
    "        else if (m < 3.5) ok = d <= 0.0 ? 1.0 : 0.0;\n"
    "        else if (m < 4.5) ok = abs(d) < (0.5 / 128.0) ? 1.0 : 0.0;\n"
    "        else if (m < 5.5) ok = d >= 0.0 ? 1.0 : 0.0;\n"
    "        else if (m < 6.5) ok = d > 0.0 ? 1.0 : 0.0;\n"
    "        else              ok = abs(d) >= (0.5 / 128.0) ? 1.0 : 0.0;\n"
    "        if (K_AFMODE > 0.5 && K_AFMODE < 1.5 && ok < 0.5) discard;\n"
    "        if (K_AFMODE > 1.5 && ok > 0.5) discard;\n"
    "    }\n"
    "    float oa = saturate(c.a);\n"
    "    if (K_OUTFIX > 0.5) oa = uMisc.y;\n"
    "    if (K_FBA > 0.5) oa = 1.0;\n"
    "    return float4(saturate(c.rgb), oa);\n"
    "}\n";

/* AC5_GL_SIMPLEFS=1: a texture-times-colour shader in place of the GS one,
 * to measure how much of the GPU's time the GS shader's arithmetic takes
 * (colours, fog and alpha test will be wrong). */
static const char *SIMPLE_FS =
    "float4 main(float4 vColor : TEXCOORD0, float3 vStq : TEXCOORD1,\n"
    "            float vFog : TEXCOORD2, uniform sampler2D uTex,\n"
    "            uniform float4 uTexInfo) : COLOR {\n"
    "#ifdef TME\n"
    "    return tex2D(uTex, vStq.xy / max(vStq.z, 1e-9)) * vColor * 2.0;\n"
    "#else\n"
    "    return vColor;\n"
    "#endif\n"
    "}\n";
static const char *fs_src(void) {
    static int simple = -1;
    if (simple < 0) simple = getenv("AC5_GL_SIMPLEFS") ? 1 : 0;
    return simple ? SIMPLE_FS : FS_SRC;
}

static const char *COPY_VS =
    "void main(float2 aPos, float2 aUv, float4 out vPos : POSITION,\n"
    "          float2 out vUv : TEXCOORD0) {\n"
    "    vPos = float4(aPos, 0.0, 1.0);\n"
    "    vUv = aUv;\n"
    "}\n";
/* the second display circuit over the picture: alpha from the texel
 * (uMode.x 0, PS2 0x80 = 1) or fixed (uMode.x 1, uMode.y) */
static const char *OV_FS =
    "float4 main(float2 vUv : TEXCOORD0, uniform sampler2D uTex, uniform float4 uMode) : COLOR {\n"
    "    float4 t = tex2D(uTex, vUv);\n"
    "    return float4(t.rgb, uMode.x > 0.5 ? uMode.y : t.a);\n"
    "}\n";
static const char *COPY_FS =
    "float4 main(float2 vUv : TEXCOORD0, uniform sampler2D uTex) : COLOR {\n"
    "    return tex2D(uTex, vUv);\n"
    "}\n";

static GLuint compile(GLenum type, const char *defines, const char *src) {
    GLuint s = glCreateShader(type);
    size_t dl = defines ? strlen(defines) : 0, sl = strlen(src);
    char *full = (char *)malloc(dl + sl + 1);
    GLint ok = 0, len;
    if (!full) return 0;
    if (dl) memcpy(full, defines, dl);
    memcpy(full + dl, src, sl + 1);
    len = (GLint)(dl + sl);
    {   const char *one = full;
        glShaderSource(s, 1, &one, &len); }
    free(full);
    glCompileShader(s);
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[1024];
        GLsizei n = 0;
        glGetShaderInfoLog(s, sizeof log - 1, &n, log);
        log[n > 0 ? n : 0] = 0;
        ps2_log("gl: shader compile failed (%s): %s",
                defines && *defines ? defines : "base", log);
        glDeleteShader(s);
        return 0;
    }
    return s;
}

static GLuint vs_for(int vsel) {
    return vsel == 3 ? vs_clip : vsel == 2 ? vs_mesh : vsel == 1 ? vs_point : vs_tri;
}
static GLuint link_gs(GLuint vs, GLuint fs) {
    GLuint p = glCreateProgram();
    GLint ok = 0;
    glAttachShader(p, vs);
    glAttachShader(p, fs);
    glBindAttribLocation(p, 0, "aPos");
    if (vs == vs_mesh) {
        glBindAttribLocation(p, 1, "aNrm");
        glBindAttribLocation(p, 2, "aCol");
        glBindAttribLocation(p, 3, "aUv");
    } else {
        glBindAttribLocation(p, 1, "aColor");
        glBindAttribLocation(p, 2, "aStq");
        glBindAttribLocation(p, 3, "aFog");
    }
    glLinkProgram(p);
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        ps2_log("gl: program link failed");
        glDeleteProgram(p);
        return 0;
    }
    return p;
}

static int build_program(gl_variant *v, int textured, int points) {
    GLuint fs = compile(GL_CG_FRAGMENT_SHADER_EXT, textured ? "#define TME 1\n" : NULL, fs_src());
    if (!fs) return -1;
    v->prog = link_gs(vs_for(points), fs);
    if (!v->prog) return -1;
    v->u_xform = glGetUniformLocation(v->prog, "uXform");
    v->u_texinfo = glGetUniformLocation(v->prog, "uTexInfo");
    v->u_clamp = glGetUniformLocation(v->prog, "uClamp");
    v->u_misc = glGetUniformLocation(v->prog, "uMisc");
    v->u_fog = glGetUniformLocation(v->prog, "uFog");
    v->u_tex = glGetUniformLocation(v->prog, "uTex");
    v->u_f1 = glGetUniformLocation(v->prog, "uF1");
    v->u_f2 = glGetUniformLocation(v->prog, "uF2");
    v->u_f3 = glGetUniformLocation(v->prog, "uF3");
    v->u_blk = glGetUniformLocation(v->prog, "uB");
    glUseProgram(v->prog);
    if (v->u_tex >= 0) glUniform1i(v->u_tex, 0);
    nvariants++;
    return 0;
}

/* Specialised programs, built on first use (the compiled binaries land in
 * vitaGL's shader cache, so later runs load them instead). A few per frame
 * at most; past that, or if one fails, the uber program draws. */
#define SPEC_MAX 192
#define SPEC_PER_FRAME 3
typedef struct { u32 key; int used; gl_variant v; } spec_entry;
static spec_entry spec[SPEC_MAX];
static int nspec, spec_built_this_frame, spec_failed;

static u32 spec_key(u32 key, int points) {
    /* the bits the shader cares about; bit 12 (point sampling) included */
    return (key & 0x3FFFFFu) | (points == 1 ? 0x80000000u : points == 2 ? 0x40000000u
                               : points == 3 ? 0x20000000u : 0u);
}

static int build_spec(gl_variant *v, u32 key, int points) {
    char def[512];
    u32 tme = key & 1u, tfx = (key >> 1) & 3u, tcc = (key >> 3) & 1u, fge = (key >> 4) & 1u;
    u32 atst = (key >> 5) & 7u, ate = (key >> 8) & 1u;
    u32 afmode = (key >> 10) & 1u ? 2u : ((key >> 9) & 1u ? 1u : 0u);
    u32 fst = (key >> 11) & 1u, pt = (key >> 12) & 1u, wms = (key >> 13) & 3u, wmt = (key >> 15) & 3u;
    u32 half = (key >> 17) & 1u, outfix = (key >> 18) & 1u, fba = (key >> 19) & 1u;
    u32 trim = (key >> 20) & 1u, nrep = (key >> 21) & 1u;
    snprintf(def, sizeof def,
             "%s#define K_TFX %u.0\n#define K_TCC %u.0\n#define K_FST %u.0\n#define K_FGE %u.0\n"
             "#define K_ATE %u.0\n#define K_ATST %u.0\n#define K_AFMODE %u.0\n#define K_OUTFIX %u.0\n"
             "#define K_FBA %u.0\n#define K_POINT %u.0\n#define K_HALF %u.0\n"
             "#define K_WMS %u.0\n#define K_WMT %u.0\n#define K_TRIM %u.0\n#define K_NREP %u.0\n",
             tme ? "#define TME 1\n" : "", tfx, tcc, fst, fge, ate, atst, afmode, outfix,
             fba, pt, half, wms, wmt, trim, nrep);
    GLuint fs = compile(GL_CG_FRAGMENT_SHADER_EXT, def, fs_src());
    if (!fs) return -1;
    v->prog = link_gs(vs_for(points), fs);
    if (!v->prog) return -1;
    v->u_xform = glGetUniformLocation(v->prog, "uXform");
    v->u_texinfo = glGetUniformLocation(v->prog, "uTexInfo");
    v->u_clamp = glGetUniformLocation(v->prog, "uClamp");
    v->u_misc = glGetUniformLocation(v->prog, "uMisc");
    v->u_fog = glGetUniformLocation(v->prog, "uFog");
    v->u_tex = glGetUniformLocation(v->prog, "uTex");
    v->u_f1 = glGetUniformLocation(v->prog, "uF1");
    v->u_f2 = glGetUniformLocation(v->prog, "uF2");
    v->u_f3 = glGetUniformLocation(v->prog, "uF3");
    v->u_blk = glGetUniformLocation(v->prog, "uB");
    glUseProgram(v->prog);
    if (v->u_tex >= 0) glUniform1i(v->u_tex, 0);
    return 0;
}

/* Shader variants seen in earlier runs are listed in VARIANTS_FILE and
 * compiled at start-up, so a screen that needs new ones (the main menu
 * brings in nine) does not stall on the shader compiler mid-game. */
#define VARIANTS_FILE "ux0:data/ac5/gl_variants.txt"
static int prewarming;
static void remember_variant(u32 key, int points) {
    FILE *f = fopen(VARIANTS_FILE, "a");
    if (!f) return;
    fprintf(f, "%08X %d\n", key, points);
    fclose(f);
}
static gl_variant *get_variant(u32 key, int points);
/* Compiling them all at start-up held the boot screen for ~15 s; read
 * the list then, and compile one per frame while frames are light (the
 * logos run at full speed with a few draws each). */
static u32 pw_key[256];
static u8 pw_pts[256];
static int pw_n, pw_next, pw_done;
static u64 pw_us;
static void prewarm_variants(void) {
    FILE *f = fopen(VARIANTS_FILE, "r");
    unsigned key;
    int pts;
    if (!f) return;
    while (pw_n < 256 && fscanf(f, "%x %d", &key, &pts) == 2) {
        pw_key[pw_n] = (u32)key;
        pw_pts[pw_n] = (u8)(pts >= 1 && pts <= 3 ? pts : 0);
        pw_n++;
    }
    fclose(f);
    ps2_log("gl: %d shader variants from earlier runs queued to compile during light frames", pw_n);
}
static void prewarm_step(u32 draws_this_frame) {
    u64 t0;
    if (pw_next >= pw_n || draws_this_frame > 64u) return;
    t0 = sceKernelGetProcessTimeWide();
    prewarming = 1;
    spec_built_this_frame = 0;
    get_variant(pw_key[pw_next], pw_pts[pw_next]);
    prewarming = 0;
    spec_built_this_frame = 0;
    pw_next++;
    pw_us += sceKernelGetProcessTimeWide() - t0;
    if (pw_next == pw_n && !pw_done) {
        pw_done = 1;
        ps2_log("gl: finished compiling the %d queued shader variants (%.0f ms in all)",
                pw_n, (double)pw_us / 1000.0);
    }
}

static gl_variant *get_variant(u32 key, int points) {
    u32 k = spec_key(key, points);
    u32 h = (k * 2654435761u) % SPEC_MAX;
    for (int n = 0; n < SPEC_MAX; n++, h = (h + 1u) % SPEC_MAX) {
        spec_entry *e = &spec[h];
        if (e->used && e->key == k) {
            if (e->v.prog) return &e->v;
            break;                                   /* failed before: uber */
        }
        if (!e->used) {
            if (nspec >= SPEC_MAX - 8 || spec_built_this_frame >= SPEC_PER_FRAME) break;
            spec_built_this_frame++;
            e->used = 1;
            e->key = k;
            nspec++;
            if (build_spec(&e->v, key, points) != 0) {
                e->v.prog = 0;
                if (spec_failed++ < 4)
                    ps2_log("gl: specialised shader %08X failed; using the general one", k);
                break;
            }
            nvariants++;
            if (!prewarming) remember_variant(key, points);
            return &e->v;
        }
    }
    gl_variant *v = &progs[(key & 1u) ? 1 : 0][points >= 1 && points <= 3 ? points : 0];
    return v->prog ? v : NULL;
}

static int make_copy_program(void) {
    GLuint vs = compile(GL_CG_VERTEX_SHADER_EXT, NULL, COPY_VS);
    GLuint fs = compile(GL_CG_FRAGMENT_SHADER_EXT, NULL, COPY_FS);
    GLint ok = 0;
    if (!vs || !fs) return -1;
    copy_prog = glCreateProgram();
    glAttachShader(copy_prog, vs);
    glAttachShader(copy_prog, fs);
    glBindAttribLocation(copy_prog, 0, "aPos");
    glBindAttribLocation(copy_prog, 1, "aUv");
    glLinkProgram(copy_prog);
    glGetProgramiv(copy_prog, GL_LINK_STATUS, &ok);
    if (!ok) return -1;
    copy_u_tex = glGetUniformLocation(copy_prog, "uTex");
    glUseProgram(copy_prog);
    glUniform1i(copy_u_tex, 0);
    {   GLuint ofs = compile(GL_CG_FRAGMENT_SHADER_EXT, NULL, OV_FS);
        if (ofs) {
            ov_prog = glCreateProgram();
            glAttachShader(ov_prog, vs);
            glAttachShader(ov_prog, ofs);
            glBindAttribLocation(ov_prog, 0, "aPos");
            glBindAttribLocation(ov_prog, 1, "aUv");
            glLinkProgram(ov_prog);
            glGetProgramiv(ov_prog, GL_LINK_STATUS, &ok);
            if (!ok) ov_prog = 0;
            else {
                ov_u_mode = glGetUniformLocation(ov_prog, "uMode");
                glUseProgram(ov_prog);
                glUniform1i(glGetUniformLocation(ov_prog, "uTex"), 0);
            }
        }
    }
    return 0;
}

/* Draw the uv rectangle (u0,v0)-(u1,v1) of `tex` onto the NDC rectangle
 * (x0,y0)-(x1,y1): the top edge (y1) samples v0, the bottom edge v1. */
static void draw_copy_with(GLuint prog, GLuint tex, float u0, float v0, float u1, float v1,
                           float x0, float y0, float x1, float y1, int linear);
static void draw_copy(GLuint tex, float u0, float v0, float u1, float v1,
                      float x0, float y0, float x1, float y1, int linear) {
    draw_copy_with(copy_prog, tex, u0, v0, u1, v1, x0, y0, x1, y1, linear);
}
static void draw_copy_with(GLuint prog, GLuint tex, float u0, float v0, float u1, float v1,
                           float x0, float y0, float x1, float y1, int linear) {
    float q[16] = {
        x0, y0, u0, v1,
        x1, y0, u1, v1,
        x0, y1, u0, v0,
        x1, y1, u1, v0,
    };
    glUseProgram(prog);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, linear ? GL_LINEAR : GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, linear ? GL_LINEAR : GL_NEAREST);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glDisableVertexAttribArray(2);
    glDisableVertexAttribArray(3);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 16, &q[0]);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 16, &q[2]);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
}

static GLuint new_color_tex(u32 w, u32 h) {
    GLuint t;
    glGenTextures(1, &t);
    glBindTexture(GL_TEXTURE_2D, t);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, (GLsizei)w, (GLsizei)h, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    return t;
}

static int create_target(gl_target *t) {
    if (t->created) return 0;
    t->color = new_color_tex((u32)PW, (u32)PH);
    glGenFramebuffers(1, &t->fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, t->fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, t->color, 0);
    t->attached = t->color;
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        ps2_log("gl: render target %ux%u is incomplete", rt_w, rt_h);
    glViewport(0, 0, PW, PH);
    glDisable(GL_SCISSOR_TEST);
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    glDepthMask(GL_TRUE);
    glClearColor(0, 0, 0, 0);
    glClearDepthf(0.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    t->created = 1;
    t->dirty_seq = 1;
    stat_rt_binds++;
    return 0;
}

static void destroy_targets(void) {
    for (int i = 0; i < RT_SLOTS; i++) {
        gl_target *t = &rts[i];
        if (!t->created) continue;
        glDeleteFramebuffers(1, &t->fbo);
        glDeleteTextures(1, &t->color);
        memset(t, 0, sizeof *t);
    }
    if (snap_fbo) glDeleteFramebuffers(1, &snap_fbo);
    if (snap_tex) glDeleteTextures(1, &snap_tex);
    snap_fbo = snap_tex = 0;
    snap_slot = -1;
}

/* Grow the shared target size when a field needs more (as the Vulkan
 * renderer does). Contents are lost; this only happens at mode changes. */
static void size_targets(void) {
    u32 need_w = present_x + present_w, need_h = present_y + present_h;
    for (u32 k = 0; k < rep_ndraws; k++) {
        u32 w = rep_draws[k].fb_w;
        u32 h = (u32)(rep_draws[k].scissor[3] + 1);
        if (w > need_w && w <= RT_MAX) need_w = w;
        if (h > need_h && h <= RT_MAX) need_h = h;
    }
    if (need_w > RT_MAX) need_w = RT_MAX;
    if (need_h > RT_MAX) need_h = RT_MAX;
    if (need_w <= rt_w && need_h <= rt_h) return;
    if (need_w < rt_w) need_w = rt_w;
    if (need_h < rt_h) need_h = rt_h;
    destroy_targets();
    rt_w = need_w;
    rt_h = need_h;
    ps2_log("gl: render targets now %ux%u", rt_w, rt_h);
}

static u64 stat_uploads_p8, stat_defer_ms;
static void upload_texture(gl_texture *t) {
    u32 gen = __atomic_load_n(&t->generation, __ATOMIC_ACQUIRE);
    const u8 *px = __atomic_load_n(&t->rgba, __ATOMIC_ACQUIRE);
    if (!px) return;
    if (!t->name) {
        glGenTextures(1, &t->name);
        glBindTexture(GL_TEXTURE_2D, t->name);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        t->filter = -1;
    } else {
        glBindTexture(GL_TEXTURE_2D, t->name);
    }
    {   extern const u32 ps2_tex_p8_magic[4];
        if (!memcmp(px, ps2_tex_p8_magic, 16)) {
            /* indices + palette (ps2_gs.c tex_job_fill_p8): a P8 GPU texture,
             * new storage each time (vitaGL frees the old once the GPU is done) */
            glCompressedTexImage2D(GL_TEXTURE_2D, 0, GL_PALETTE8_RGBA8_OES, (GLsizei)t->w, (GLsizei)t->h, 0,
                                   (GLsizei)(1024u + t->w * t->h), px + 16);
            t->gl_w = 0;            /* storage is P8: an RGBA upload re-specifies */
            t->gl_h = 0;
            t->filter = -1;
            t->uploaded_gen = gen;
            stat_uploads++;
            stat_uploads_p8++;
            return;
        }
    }
    /* Same size: update in place. Re-specifying with glTexImage2D makes
     * vitaGL allocate fresh storage and free the old only once the GPU is
     * done with it, which under heavy texture streaming exhausted its pool. */
    if (t->gl_w == t->w && t->gl_h == t->h) {
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, (GLsizei)t->w, (GLsizei)t->h,
                        GL_RGBA, GL_UNSIGNED_BYTE, px);
    } else {
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, (GLsizei)t->w, (GLsizei)t->h, 0,
                     GL_RGBA, GL_UNSIGNED_BYTE, px);
        t->gl_w = t->w;
        t->gl_h = t->h;
    }
    t->uploaded_gen = gen;
    stat_uploads++;
}

static void set_filter(GLuint name, int *cached, int point) {
    int want = point ? GL_NEAREST : GL_LINEAR;
    glBindTexture(GL_TEXTURE_2D, name);
    if (cached && *cached == want) return;
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, want);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, want);
    if (cached) *cached = want;
}

/* GS blend (A - B) * C + D with A,B,D in {Cs, Cd, 0} and C in {As, Ad, FIX},
 * as GL factors. Same case analysis as translate_blend() in ps2_video.c. */
static void translate_blend(u32 a, u32 b, u32 c, u32 d,
                            GLenum *src, GLenum *dst, GLenum *op) {
    GLenum f = c == 1 ? GL_DST_ALPHA : GL_SRC_ALPHA;
    GLenum nf = c == 1 ? GL_ONE_MINUS_DST_ALPHA : GL_ONE_MINUS_SRC_ALPHA;
    *op = GL_FUNC_ADD;
    if (a == b) {
        *src = d == 0 ? GL_ONE : GL_ZERO;
        *dst = d == 1 ? GL_ONE : GL_ZERO;
        return;
    }
    switch (a * 3u + b) {
    case 0 * 3 + 1:
        if (d == 1) { *src = f; *dst = nf; return; }
        if (d == 2) { *src = f; *dst = f; *op = GL_FUNC_SUBTRACT; return; }
        break;
    case 0 * 3 + 2:
        if (d == 1) { *src = f; *dst = GL_ONE; return; }
        if (d == 2) { *src = f; *dst = GL_ZERO; return; }
        break;
    case 1 * 3 + 0:
        if (d == 0) { *src = nf; *dst = f; return; }
        if (d == 2) { *src = f; *dst = f; *op = GL_FUNC_REVERSE_SUBTRACT; return; }
        break;
    case 1 * 3 + 2:
        if (d == 0) { *src = GL_ONE; *dst = f; return; }
        if (d == 2) { *src = GL_ZERO; *dst = f; return; }
        break;
    case 2 * 3 + 0:
        if (d == 0) { *src = nf; *dst = GL_ZERO; return; }
        if (d == 1) { *src = f; *dst = GL_ONE; *op = GL_FUNC_REVERSE_SUBTRACT; return; }
        *src = f; *dst = GL_ZERO; *op = GL_FUNC_REVERSE_SUBTRACT;
        return;
    case 2 * 3 + 1:
        if (d == 0) { *src = GL_ONE; *dst = f; *op = GL_FUNC_SUBTRACT; return; }
        if (d == 1) { *src = GL_ZERO; *dst = nf; return; }
        *src = GL_ZERO; *dst = f; *op = GL_FUNC_REVERSE_SUBTRACT;
        return;
    default:
        break;
    }
    if (a == 0 && b == 1) { *src = GL_ONE; *dst = f; *op = GL_FUNC_SUBTRACT; return; }
    if (a == 0 && b == 2) { *src = GL_ONE; *dst = GL_ZERO; return; }
    if (a == 1 && b == 0) { *src = f; *dst = GL_ONE; *op = GL_FUNC_REVERSE_SUBTRACT; return; }
    *src = GL_ZERO; *dst = GL_ONE;
}

static GLenum depth_func(u32 ztst) {
    switch (ztst & 3u) {
    case 0:  return GL_NEVER;
    case 1:  return GL_ALWAYS;
    case 2:  return GL_GEQUAL;
    default: return GL_GREATER;
    }
}

/* Vertices are given to vitaGL as client-side arrays, pointed at the draw's
 * own range: vitaGL copies them into its per-frame scratch memory, which is
 * recycled every frame. (A buffer object re-filled with glBufferData every
 * frame left vitaGL's pool full of orphaned copies until it ran out.) */
/* The frame's vertices go to the GPU once, in one buffer; draws point into
 * it. Client-side arrays made vitaGL copy (and allocate for) every draw. */
static GLuint frame_vbo;
static int frame_vbo_on;
static void point_gs_vertices(const ps2_vk_vertex *v) {
    const GLsizei stride = (GLsizei)sizeof(ps2_vk_vertex);
    if (frame_vbo_on) {
        uintptr_t base = (uintptr_t)(v - rep_verts) * sizeof(ps2_vk_vertex);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, (const void *)(base + offsetof(ps2_vk_vertex, x)));
        glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, stride, (const void *)(base + offsetof(ps2_vk_vertex, r)));
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, stride, (const void *)(base + offsetof(ps2_vk_vertex, s)));
        glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, stride, (const void *)(base + offsetof(ps2_vk_vertex, fog)));
        return;
    }
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, &v->x);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, stride, &v->r);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, stride, &v->s);
    glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, stride, &v->fog);
}
static void enable_gs_attribs(void) {
    glBindBuffer(GL_ARRAY_BUFFER, frame_vbo_on ? frame_vbo : 0);
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glEnableVertexAttribArray(2);
    glEnableVertexAttribArray(3);
}

/* Copy a target's colour into its snapshot texture, for draws that read the
 * target they draw into. */
/* Returns 0 when the per-frame budget is spent and no usable copy exists:
 * each copy is a separate GPU scene (a full tile load/store pass). */
static GLuint snapshot(int slot, int *rebound) {
    gl_target *t = &rts[slot];
    *rebound = 0;
    if (snap_tex && snap_slot == slot && snap_seq == t->dirty_seq) return snap_tex;
    if (snaps_this_frame >= SNAP_MAX_PER_FRAME) {
        stat_snap_skips++;
        return snap_tex && snap_slot == slot ? snap_tex : 0;  /* stale is ok */
    }
    if (!snap_tex) {
        snap_tex = new_color_tex((u32)PW, (u32)PH);
        glGenFramebuffers(1, &snap_fbo);
        glBindFramebuffer(GL_FRAMEBUFFER, snap_fbo);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, snap_tex, 0);
    }
    *rebound = 1;
    vglSetNextSceneRegion(0, 0, 0);
    snaps_this_frame++;
    stat_snapshots++;
    stat_rt_binds++;
    glBindFramebuffer(GL_FRAMEBUFFER, snap_fbo);
    glViewport(0, 0, PW, PH);
    glDisable(GL_SCISSOR_TEST);
    glDisable(GL_BLEND);
    glDisable(GL_DEPTH_TEST);
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    draw_copy(t->color, 0, 1, 1, 0, -1, -1, 1, 1, 0);
    snap_slot = slot;
    snap_seq = t->dirty_seq;
    return snap_tex;
}

/* Give back the GL storage of store entries the recorder evicted, and keep
 * a tally of what the textures hold on the GPU. Without this every texture
 * ever seen kept its vitaGL storage, which filled the pools in missions. */
static void sweep_textures(void) {
    u32 n = __atomic_load_n(&ntextures, __ATOMIC_ACQUIRE);
    u64 bytes = 0;
    for (u32 k = 0; k < n; k++) {
        gl_texture *t = &textures[k];
        if (!t->name) continue;
        if (!__atomic_load_n(&t->rgba, __ATOMIC_ACQUIRE)) {
            glDeleteTextures(1, &t->name);
            t->name = 0;
            t->gl_w = t->gl_h = 0;
            t->uploaded_gen = 0;
            stat_tex_deletes++;
            continue;
        }
        bytes += (u64)t->gl_w * t->gl_h * 4u;
    }
    stat_gpu_tex_kb = (u32)(bytes >> 10);
}

static u64 stat_zclears, stat_zswaps;

/* Select+Triangle in GPU mode: write every render target drawn in the next
 * frame as a TGA, plus that frame's draw list as text, to ux0:data/ac5/dump. */
static volatile int dump_requested;
static int dump_active;
static FILE *dump_list;
static u8 dump_rt_used[RT_SLOTS];
void vita_gl_request_dump(void) { dump_requested = 1; }

static void write_tga(const char *path, const u8 *rgba, int w, int h) {
    FILE *f = fopen(path, "wb");
    if (!f) return;
    u8 hdr[18] = {0};
    hdr[2] = 2; hdr[12] = (u8)w; hdr[13] = (u8)(w >> 8); hdr[14] = (u8)h; hdr[15] = (u8)(h >> 8);
    hdr[16] = 32; hdr[17] = 8 | 0x20;   /* top-left origin: GL row 0 is GS row 0 */
    fwrite(hdr, 1, 18, f);
    u8 *row = (u8 *)malloc((size_t)w * 4u);
    if (row) {
        for (int y = 0; y < h; y++) {
            const u8 *src = rgba + (size_t)y * w * 4u;
            for (int x = 0; x < w; x++) {
                row[x * 4 + 0] = src[x * 4 + 2]; row[x * 4 + 1] = src[x * 4 + 1];
                row[x * 4 + 2] = src[x * 4 + 0]; row[x * 4 + 3] = src[x * 4 + 3];
            }
            fwrite(row, 1, (size_t)w * 4u, f);
        }
        free(row);
    }
    fclose(f);
}

static u8 dump_tex_done[512];
/* The texture as the GPU holds it (read back through a scratch FBO), so a
 * dump shows exactly what the draws sampled. */
static void dump_texture(u32 idx) {
    gl_texture *t = &textures[idx];
    u32 w = t->gl_w ? t->gl_w : t->w, h = t->gl_h ? t->gl_h : t->h;
    GLuint fbo = 0;
    u8 *buf;
    char path[96];
    if (!t->name || !w || !h || w > 2048u || h > 2048u) return;
    buf = (u8 *)malloc((size_t)w * h * 4u);
    if (!buf) return;
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, t->name, 0);
    glReadPixels(0, 0, (GLsizei)w, (GLsizei)h, GL_RGBA, GL_UNSIGNED_BYTE, buf);
    snprintf(path, sizeof path, "ux0:data/ac5/dump/tex%03u_%ux%u.tga", idx, t->w, t->h);
    write_tga(path, buf, (int)w, (int)h);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDeleteFramebuffers(1, &fbo);
    free(buf);
}

static void dump_targets(void) {
    int w = PW, h = PH;
    u8 *buf = (u8 *)malloc((size_t)w * h * 4u);
    if (!buf) return;
    for (int i = 0; i < RT_SLOTS; i++) {
        if (!rts[i].created || !dump_rt_used[i]) continue;
        GLuint fbo = 0;
        glGenFramebuffers(1, &fbo);
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, rts[i].color, 0);
        glReadPixels(0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, buf);
        char path[96];
        snprintf(path, sizeof path, "ux0:data/ac5/dump/rt%02d%s.tga", i, i == (int)present_rt ? "_shown" : "");
        write_tga(path, buf, w, h);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glDeleteFramebuffers(1, &fbo);
    }
    free(buf);
    for (u32 k = 0; k < 4096u; k++)
        if (dump_tex_done[k >> 3] & (1u << (k & 7u))) dump_texture(k);
    ps2_log("gl: dumped render targets, textures and the draw list to ux0:data/ac5/dump");
}
static u8 is_zbuf[RT_SLOTS];
static u32 frame_3d_tris, wide_run, flat_run;
/* where 3D goes: triangles per target, and depth tests used, since the last report */
static u64 stat_3d_rt[RT_SLOTS], stat_3d_ztst[4], stat_present_rt[RT_SLOTS];
static void note_3d(const gl_draw *d, u32 tris) {
    if (d->rt < RT_SLOTS) stat_3d_rt[d->rt] += tris;
    stat_3d_ztst[d->depth & 3u] += tris;
}
static int wide_mode;


static int draw_uses_z(const gl_draw *d) {
    return (d->depth & 3u) != 1u || ((d->depth >> 2) & 1u);
}

static void upload_frame_vertices(void) {
    static int off = -1;
    if (off < 0) off = getenv("AC5_GL_NOVBO") ? 1 : 0;
    frame_vbo_on = 0;
    if (off || !rep_nverts) return;
    if (!frame_vbo) glGenBuffers(1, &frame_vbo);
    if (!frame_vbo) return;
    glBindBuffer(GL_ARRAY_BUFFER, frame_vbo);
    glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)(rep_nverts * sizeof(ps2_vk_vertex)), rep_verts, GL_STREAM_DRAW);
    frame_vbo_on = 1;
}

/* One native mesh draw: the block as uniforms, the arena's records as
 * client arrays from the draw's base vertex, and its 16-bit indices. */
static int mesh_cull_mode = -1;     /* AC5_MESH_CULL: 0 off, 1 CW front, 2 CCW front */
static void draw_mesh_gl(gl_variant *v, const gl_draw *d, GLenum mode) {
    const ps2_vk_mesh_block *b = &rep_mesh->blocks[d->mblock];
    const mesh_rec *r = rep_mesh->recs + d->mvbase;
    float u[25 * 4];
    const GLsizei stride = (GLsizei)sizeof(mesh_rec);
    memcpy(u, b, 20 * 16);                               /* m c l r e, fog, offsets */
    u[80] = (float)b->mode; u[81] = (float)b->pass;
    u[82] = (float)b->clip; u[83] = (float)b->prog;
    memcpy(u + 84, b->p, 16 * 4);
    if (v->u_blk >= 0) glUniform4fv(v->u_blk, 25, u);
    if (mesh_cull_mode < 0) {
        const char *e = getenv("AC5_MESH_CULL");
        mesh_cull_mode = e ? (*e - '0') : 0;
    }
    if (mesh_cull_mode > 0 && d->mcull && mode == GL_TRIANGLES) {
        glEnable(GL_CULL_FACE);
        glFrontFace(mesh_cull_mode == 1 ? GL_CW : GL_CCW);
        glCullFace(GL_BACK);
    }
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, stride, r->pos);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, stride, r->nrm);
    glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, stride, r->col);
    glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, stride, r->uv);
    glDrawRangeElements(mode, 0, d->mvn - 1u, (GLsizei)d->count, GL_UNSIGNED_SHORT,
                        rep_mesh->idx + d->first);
    if (mesh_cull_mode > 0 && d->mcull && mode == GL_TRIANGLES) glDisable(GL_CULL_FACE);
    if (mode == GL_TRIANGLES) { stat_drawn_tri++; frame_3d_tris += d->count / 3u; note_3d(d, d->count / 3u); }
    else if (mode == GL_LINES) stat_drawn_line++;
    else stat_drawn_point++;
    stat_mesh_draws++;
    stat_draws++;
}

/* Draw order. Every change of target is a new GPU scene, and the Vita
 * writes out and reloads the whole colour and depth buffer for each one;
 * missions changed target ~20 times a frame. Draws are grouped back onto
 * an earlier run for the same target whenever nothing drawn in between
 * touches what they read or write (colour or depth of any target), so
 * the picture is unchanged and there are fewer scenes. */
static u32 ord[GL_MAX_DRAWS];
#define SEG_MAX 96
#define SEG_LOOKBACK 12
typedef struct { u16 rt; u32 rc, rd, wc, wd; u32 first_k, n; } seg_info;
static seg_info segs[SEG_MAX];
static u32 seg_draw_next[GL_MAX_DRAWS];   /* linked list of draws per segment */
static u32 seg_head[SEG_MAX], seg_tail[SEG_MAX];
static u64 stat_segs, stat_segs_before, stat_moved;
static int no_reorder = -1;

static int draw_uses_z(const gl_draw *d);
static void draw_masks(const gl_draw *d, u32 *rc, u32 *rd, u32 *wc, u32 *wd) {
    u32 ztst = d->depth & 3u, zw = (d->depth >> 2) & 1u;
    *rc = *rd = *wc = *wd = 0;
    if (d->rt < RT_SLOTS) {
        *wc |= 1u << d->rt;
        *rc |= 1u << d->rt;              /* blending, partial masks */
        if (is_zbuf[d->rt]) *wd |= 1u << d->rt;   /* depth stamps */
    }
    if (d->tex_rt && d->tex_rt <= RT_SLOTS) *rc |= 1u << (d->tex_rt - 1u);
    if (d->zrt < RT_SLOTS) {
        if (ztst != 1u) *rd |= 1u << d->zrt;
        if (zw) *wd |= 1u << d->zrt;
    }
}

/* Solid 3D (no blending, depth tested and written, all channels) looks the
 * same in any order, so a stretch of such mesh draws on one target is
 * sorted by texture and state: neighbours then share state and are drawn
 * as one. (Exact ties in depth could pick a different winner; decals on
 * the PS2 are blended, so they are not in these runs.) */
static int no_sort = -1;
static u64 stat_sorted, stat_overlays_moved;
/* A mesh's first pass: depth tested and written, all channels. Mission
 * terrain draws its first pass with alpha blending on, but at full alpha
 * (the ground is solid), so it is grouped like solid geometry; tiles only
 * meet at their edges. AC5_GL_SORT_SOLID=1 limits this to unblended. */
static int opaque_mesh(const gl_draw *d) {
    u32 ztst = d->depth & 3u;
    static int solid_only = -1;
    if (solid_only < 0) solid_only = getenv("AC5_GL_SORT_SOLID") ? 1 : 0;
    return d->native == PS2_VK_NATIVE_MESH && (!solid_only || !((d->blend >> 8) & 1u))
        && (ztst == 2u || ztst == 3u) && ((d->depth >> 2) & 1u)
        && ((d->depth >> 3) & 1u) && ((d->depth >> 4) & 1u) && !((d->depth >> 5) & 0xFu)
        && !(d->tex_rt && (u32)(d->tex_rt - 1u) == d->rt) && !d->skip;
}
static int state_cmp(const gl_draw *a, const gl_draw *b) {
#define C(f) if (a->f != b->f) return a->f < b->f ? -1 : 1
    C(tex); C(tex_rt); C(variant); C(depth); C(zrt); C(aref); C(fogcol); C(fix);
    C(minu); C(maxu); C(minv); C(maxv); C(point_filter);
#undef C
    if (a->tex_w != b->tex_w) return a->tex_w < b->tex_w ? -1 : 1;
    if (a->tex_h != b->tex_h) return a->tex_h < b->tex_h ? -1 : 1;
    return memcmp(a->scissor, b->scissor, sizeof a->scissor);
}
/* A later pass of a mesh drawn over its own first pass (shine, detail,
 * light-map layers): depth-tested against that geometry, so it can only
 * land where its own first pass is front-most. */
static int overlay_pass(const gl_draw *d) {
    u32 ztst = d->depth & 3u;
    return d->native == PS2_VK_NATIVE_MESH && rep_mesh && d->mblock < rep_mesh->nblocks
        && rep_mesh->blocks[d->mblock].pass >= 1u && (ztst == 2u || ztst == 3u)
        && !(d->tex_rt && (u32)(d->tex_rt - 1u) == d->rt) && !d->skip;
}
static u32 sort_tmp[GL_MAX_DRAWS];
static void sort_opaque_runs(void) {
    if (no_sort < 0) no_sort = getenv("AC5_GL_NOSORT") ? 1 : 0;
    if (no_sort) return;
    for (u32 a = 0; a < rep_ndraws; ) {
        const gl_draw *h = &rep_draws[ord[a]];
        u32 b = a, nop = 0, nov = 0;
        if (!opaque_mesh(h)) { a++; continue; }
        /* the run: first passes and their overlay passes, one target */
        while (b < rep_ndraws) {
            const gl_draw *e = &rep_draws[ord[b]];
            if (e->rt != h->rt) break;
            if (opaque_mesh(e) && !(rep_mesh && e->mblock < rep_mesh->nblocks
                                    && rep_mesh->blocks[e->mblock].pass >= 1u)) nop++;
            else if (overlay_pass(e)) nov++;
            else break;
            b++;
        }
        /* Every first pass ahead of every overlay: an overlay still comes
         * after its own first pass (it did before), and a first pass moved
         * ahead of another mesh's overlay either covers it (then the
         * overlay's depth test fails, as the first pass would have painted
         * over it) or is behind that mesh's own first pass and fails. */
        {   u32 k = a, j;
            for (j = a; j < b; j++) {
                const gl_draw *e = &rep_draws[ord[j]];
                if (!overlay_pass(e) || (rep_mesh && rep_mesh->blocks[e->mblock].pass == 0u))
                    sort_tmp[k++] = ord[j];
            }
            for (j = a; j < b; j++) {
                const gl_draw *e = &rep_draws[ord[j]];
                if (overlay_pass(e) && !(rep_mesh && rep_mesh->blocks[e->mblock].pass == 0u))
                    sort_tmp[k++] = ord[j];
            }
            memcpy(&ord[a], &sort_tmp[a], (size_t)(b - a) * sizeof(u32));
        }
        /* first passes by state, so equal ones sit together and merge */
        {   u32 e1 = a + nop;
            for (u32 x = a + 1; x < e1; x++) {
                u32 v = ord[x], y = x;
                while (y > a && state_cmp(&rep_draws[ord[y - 1]], &rep_draws[v]) > 0) {
                    ord[y] = ord[y - 1];
                    y--;
                }
                ord[y] = v;
            }
        }
        stat_sorted += nop;
        stat_overlays_moved += nov;
        a = b > a ? b : a + 1;
    }
}

static void build_order(void) {
    u32 nseg = 0, before = 0;
    int last_rt = -1;
    if (no_reorder < 0) no_reorder = getenv("AC5_GL_NOREORDER") ? 1 : 0;
    for (u32 i = 0; i < rep_ndraws; i++) {
        const gl_draw *d = &rep_draws[i];
        u32 rc, rd, wc, wd;
        int target = -1;
        if ((int)d->rt != last_rt) { before++; last_rt = (int)d->rt; }
        draw_masks(d, &rc, &rd, &wc, &wd);
        if (!no_reorder && nseg) {
            /* walk back over later segments while D can pass them */
            int lo = (int)nseg - SEG_LOOKBACK; if (lo < 0) lo = 0;
            for (int g = (int)nseg - 1; g >= lo; g--) {
                seg_info *x = &segs[g];
                if (x->rt == d->rt) { target = g; break; }
                if ((wc & (x->rc | x->wc)) || (wd & (x->rd | x->wd))
                    || (x->wc & rc) || (x->wd & rd))
                    break;
            }
            if (target >= 0 && target != (int)nseg - 1) {
                /* D must also not conflict with the target segment's later
                 * siblings -- checked above; and must not jump a draw of
                 * its own target, which a same-rt segment would be. */
                stat_moved++;
            }
        }
        if (target < 0 && nseg && segs[nseg - 1].rt == d->rt) target = (int)nseg - 1;
        if (target < 0) {
            if (nseg >= SEG_MAX) {
                /* out of segments: fall back to the original order */
                for (u32 k = 0; k < rep_ndraws; k++) ord[k] = k;
                stat_segs += before;
                stat_segs_before += before;
                return;
            }
            target = (int)nseg++;
            segs[target].rt = d->rt;
            segs[target].rc = segs[target].rd = segs[target].wc = segs[target].wd = 0;
            seg_head[target] = seg_tail[target] = 0xFFFFFFFFu;
        }
        {   seg_info *x = &segs[target];
            x->rc |= rc; x->rd |= rd; x->wc |= wc; x->wd |= wd;
            seg_draw_next[i] = 0xFFFFFFFFu;
            if (seg_tail[target] == 0xFFFFFFFFu) seg_head[target] = i;
            else seg_draw_next[seg_tail[target]] = i;
            seg_tail[target] = i;
        }
    }
    {   u32 k = 0;
        for (u32 g = 0; g < nseg; g++)
            for (u32 i = seg_head[g]; i != 0xFFFFFFFFu; i = seg_draw_next[i]) ord[k++] = i;
    }
    sort_opaque_runs();
    stat_segs += nseg;
    stat_segs_before += before;
}

/* where the renderer's time goes, per category, since the last report */
enum { TM_UPLOAD, TM_PICK, TM_BIND, TM_STATE, TM_DRAW, TM_MESHRUN, TM_MESHGPU, TM_OTHER, TM_N };
static u64 tm_us[TM_N], tm_last, tm_frames;
static int no_sticky = -1;
extern int ac5_prof;
/* Renderer sampling (build 99): the last timing mark passed, read by the
 * 1 kHz sampler (ps2_gfxq.c) together with the renderer thread's state, so
 * the log shows where a frame's time goes without clock reads per draw. */
volatile u8 rn_ph;
static volatile SceUID rn_thid = -1;
static u16 rn_smp[16][3];
static const char *rn_ph_name[16] = { "upload", "texpick", "bind", "state", "draw", "meshrun", "meshgpu",
                                      "other", "?", "?", "start", "swap", "after-swap", "decode-wait", "?", "?" };
void rn_sample(void) {
    u32 st = 2u;
    if (rn_busy_state != 1 || rn_thid < 0) return;
    {   SceKernelThreadInfo ti;
        memset(&ti, 0, sizeof ti);
        ti.size = sizeof ti;
        if (sceKernelGetThreadInfo(rn_thid, &ti) >= 0)
            st = (ti.status & SCE_THREAD_RUNNING) ? 0u : (ti.status & SCE_THREAD_READY) ? 1u : 2u;
    }
    if (rn_smp[rn_ph & 15u][st] < 60000u) rn_smp[rn_ph & 15u][st]++;
}
static inline void tm_mark(int c) {
    rn_ph = (u8)(c + 1 < 8 ? c + 1 : 7);     /* the segment after mark c; see the report */
    if (!ac5_prof) return;      /* ~6 clock reads a draw, thousands a frame */
    u64 now = sceKernelGetProcessTimeWide();
    tm_us[c] += now - tm_last;
    tm_last = now;
}
/* how long single draw calls take: buckets <50us, <250us, <1ms, >=1ms
 * (count and total time), and the time of draws that open a new scene */
static u64 dh_n[4], dh_us[4], dh_first_us, dh_first_n;
static int scene_new;
static inline void draw_timed_arrays(GLenum mode, GLsizei n) {
    u64 t0, dt;
    int b;
    if (!ac5_prof) { glDrawArrays(mode, 0, n); return; }
    t0 = sceKernelGetProcessTimeWide();
    glDrawArrays(mode, 0, n);
    dt = sceKernelGetProcessTimeWide() - t0;
    b = dt < 50 ? 0 : dt < 250 ? 1 : dt < 1000 ? 2 : 3;
    dh_n[b]++; dh_us[b] += dt;
    if (scene_new) { dh_first_us += dt; dh_first_n++; scene_new = 0; }
}
/* why neighbouring mesh draws could not be merged */
enum { BR_NOTMESH, BR_TEX, BR_BLEND, BR_VARIANT, BR_DEPTH, BR_TARGET, BR_REGION, BR_BIG, BR_SELF, BR_N };
static u64 stat_break[BR_N];
static int break_reason(const gl_draw *a, const gl_draw *b) {
    if (a->tex != b->tex || a->tex_rt != b->tex_rt) return BR_TEX;
    if (a->blend != b->blend || a->fix != b->fix) return BR_BLEND;
    if (a->variant != b->variant || a->aref != b->aref || a->fogcol != b->fogcol) return BR_VARIANT;
    if (a->depth != b->depth) return BR_DEPTH;
    if (a->rt != b->rt || a->zrt != b->zrt) return BR_TARGET;
    return BR_REGION;
}

/* ---- pre-transformed meshes: mesh.vert on the CPU, merged into runs ---- */
typedef struct { float pos[4], col[4], stq[3], fog; } clip_vert;
/* Runs stay under vitaGL's 32 KB "draw straight from client memory" mark,
 * so it copies them and the scratch can be refilled for the next run. */
#define MESH_RUN_MAX 639u
#define MESH_CPU_MAX_IDX 600u
static clip_vert mesh_scratch[MESH_RUN_MAX + 3];
static clip_vert mesh_vcache[MESH_CPU_MAX_IDX];
static u64 stat_mesh_runs;

static int mesh_cpu_wanted(const gl_draw *d) {
    static int sw = -1;
    if (sw < 0) {
        const char *e = getenv("AC5_MESH_CPU");
        sw = !(e && *e == '0');
    }
    return sw && mesh_cpu_ok && d->topology == PS2_VK_TRIANGLES
        && d->count <= MESH_CPU_MAX_IDX && d->mvn <= MESH_CPU_MAX_IDX;
}

static inline float lo8f(float c) { return c - 256.0f * floorf(c / 256.0f); }
static inline float clampf(float x, float lo, float hi) { return x < lo ? lo : x > hi ? hi : x; }

static void mesh_xform(const ps2_vk_mesh_block *k, const mesh_rec *r, float ix, float iy,
                       clip_vert *o) {
    const float px = r->pos[0], py = r->pos[1], pz = r->pos[2];
    float s[4], ccw;
    for (int j = 0; j < 4; j++)
        s[j] = k->m[j] * px + k->m[4 + j] * py + k->m[8 + j] * pz + k->m[12 + j];
    const float w = s[3];
    const float zlim = fminf(16777215.0f, k->zmax);
    {
        float X = (s[0] + (0.5f - k->xoff) * w) * ix - w;
        float Y = (s[1] + (0.5f - k->yoff) * w) * iy - w;
        float Z = fminf(fmaxf(s[2] * k->zscale, 0.0f), zlim * fmaxf(w, 0.0f)) / k->zmax;
        o->pos[0] = X; o->pos[1] = Y; o->pos[2] = 2.0f * Z - w; o->pos[3] = w;
    }
    ccw = k->c[3] * px + k->c[7] * py + k->c[11] * pz + k->c[15];
    const float fog = clampf(k->fog0 + w * k->fog1, 0.0f, 255.0f);
    o->fog = fog / 255.0f;
    const float m8 = 1.0f / 128.0f;
#define COLM() do { for (int j = 0; j < 4; j++) o->col[j] = lo8f(r->col[j]) * m8; } while (0)
    if (k->prog == 5u) {
        o->pos[0] = r->pos[0] * ix - 1.0f; o->pos[1] = r->pos[1] * iy - 1.0f;
        o->pos[2] = r->pos[2] * 2.0f - 1.0f; o->pos[3] = 1.0f;
        o->stq[0] = r->uv[0]; o->stq[1] = r->uv[1]; o->stq[2] = r->uv[2];
        for (int j = 0; j < 4; j++) o->col[j] = r->col[j] * m8;
        o->fog = r->uv[3];
    } else if (k->prog == 4u) {
        o->stq[0] = r->uv[0]; o->stq[1] = r->uv[1]; o->stq[2] = r->uv[2];
        COLM();
    } else if (k->prog == 3u) {
        float iw = 1.0f / w;
        float cx = s[0] * iw, cy = s[1] * iw, cz = s[2] * iw;
        float kx = cx + r->nrm[0] * k->p[0] * r->uv[2] * iw;
        float ky = cy + r->nrm[1] * k->p[1] * r->uv[2] * iw;
        float zg = clampf(floorf(floorf(cz * 16.0f) / 8.0f), k->p[2], zlim);
        if (cz > 0.0f) {
            o->pos[0] = (kx + 0.5f - k->xoff) * ix - 1.0f;
            o->pos[1] = (ky + 0.5f - k->yoff) * iy - 1.0f;
            o->pos[2] = 2.0f * zg / k->zmax - 1.0f;
        } else {
            o->pos[0] = o->pos[1] = o->pos[2] = 2.0f;
        }
        o->pos[3] = 1.0f;
        o->stq[0] = r->uv[0]; o->stq[1] = r->uv[1]; o->stq[2] = 1.0f;
        COLM();
        o->fog = 1.0f;
    } else if (k->prog == 2u) {
        if (k->pass == 0u) {
            o->stq[0] = r->uv[0]; o->stq[1] = r->uv[1]; o->stq[2] = 1.0f;
            COLM();
        } else {
            float g = clampf((k->p[2] - ccw) * k->p[3] * 128.0f, 0.0f, 128.0f) * m8;
            o->stq[0] = r->uv[0] * 2.0f; o->stq[1] = r->uv[1] * 2.0f; o->stq[2] = 1.0f;
            o->col[0] = o->col[1] = o->col[2] = g; o->col[3] = 1.0f;
        }
    } else if (k->prog == 1u) {
        float u, v, c0, c3;
        if (k->pass == 0u) {
            u = r->uv[0] + r->uv[2]; v = r->uv[1] + r->uv[3];
            COLM();
        } else {
            if (k->pass == 1u) {
                u = r->uv[0] * 16.0f; v = r->uv[1] * 16.0f;
                c0 = clampf((k->p[2] - ccw) * k->p[3] * 128.0f, 0.0f, 128.0f);
                c3 = k->p[12];
            } else if (k->pass == 2u) {
                u = r->uv[0] * 16.0f + k->p[4]; v = r->uv[1] * 16.0f + k->p[5];
                c0 = clampf((k->p[6] - ccw) * k->p[7] * 128.0f, 0.0f, 128.0f);
                c3 = fog * 0.5f;
            } else {
                u = (px - k->p[8]) * k->p[11]; v = (pz - k->p[10]) * k->p[11];
                c0 = c3 = fog * 0.5f;
            }
            o->col[0] = o->col[1] = o->col[2] = c0 * m8;
            o->col[3] = c3 * m8;
        }
        o->stq[0] = u; o->stq[1] = v; o->stq[2] = 1.0f;
    } else if (k->pass == 0u) {
        o->stq[0] = r->uv[0]; o->stq[1] = r->uv[1]; o->stq[2] = 1.0f;
        if (k->mode == 0u) {
            COLM();
        } else {
            float n0 = r->nrm[0] / 4096.0f, n1 = r->nrm[1] / 4096.0f, n2 = r->nrm[2] / 4096.0f;
            float l[4];
            for (int j = 0; j < 3; j++) l[j] = k->l[j] * n0 + k->l[4 + j] * n1 + k->l[8 + j] * n2;
            l[3] = 1.0f;
            for (int j = 0; j < 4; j++) {
                l[j] = l[j] * l[j] * l[j];
                if (j < 3) l[j] *= 2.0f;
                l[j] = fmaxf(l[j], 0.0f);
            }
            for (int j = 0; j < 3; j++) {
                float li = fmaxf(k->r[j] * l[0] + k->r[4 + j] * l[1] + k->r[8 + j] * l[2]
                                 + k->r[12 + j] * l[3], 0.0f);
                o->col[j] = fminf(r->col[j] * li, 255.0f) * m8;
            }
            o->col[3] = 1.0f;
        }
    } else {
        float n0 = r->nrm[0] / 4096.0f, n1 = r->nrm[1] / 4096.0f, n2 = r->nrm[2] / 4096.0f;
        float nb[3];
        for (int j = 0; j < 3; j++) nb[j] = k->e[j] * n0 + k->e[4 + j] * n1 + k->e[8 + j] * n2;
        o->stq[0] = (nb[0] + 1.0f) * 0.5f; o->stq[1] = (nb[2] + 1.0f) * 0.5f; o->stq[2] = 1.0f;
        float sp = fminf(fmaxf(nb[1], 0.0f) * 1280.0f, 128.0f) * m8;
        o->col[0] = o->col[1] = o->col[2] = sp;
        o->col[3] = k->spec_alpha * m8;
    }
#undef COLM
}

static int mesh_draw_ok(const gl_draw *d) {
    return rep_mesh && d->mblock < rep_mesh->nblocks && d->first + d->count <= rep_mesh->nidx
        && d->mvbase + d->mvn <= rep_mesh->nrecs && d->mvn;
}

/* Everything that decides how a draw is drawn, apart from its geometry. */
static int same_draw_state(const gl_draw *a, const gl_draw *b) {
    return a->topology == b->topology && a->tex == b->tex && a->tex_rt == b->tex_rt
        && a->rt == b->rt && a->zrt == b->zrt && a->tex_self == b->tex_self
        && a->tex_depth == b->tex_depth && a->skip == b->skip && a->native == b->native
        && a->variant == b->variant && a->blend == b->blend && a->fix == b->fix
        && a->depth == b->depth && a->aref == b->aref && a->fogcol == b->fogcol
        && a->minu == b->minu && a->maxu == b->maxu && a->minv == b->minv && a->maxv == b->maxv
        && a->tex_w == b->tex_w && a->tex_h == b->tex_h
        && !memcmp(a->scissor, b->scissor, sizeof a->scissor)
        && a->point_filter == b->point_filter;
}

static void mesh_run_flush(u32 n) {
    const GLsizei stride = (GLsizei)sizeof(clip_vert);
    if (!n) return;
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, stride, mesh_scratch[0].pos);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, stride, mesh_scratch[0].col);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, stride, mesh_scratch[0].stq);
    glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, stride, &mesh_scratch[0].fog);
    draw_timed_arrays(GL_TRIANGLES, (GLsizei)n);
    stat_draws++;
    stat_mesh_runs++;
}

/* Draws mesh draw i and every following one with the same state; returns
 * the index of the last one taken. */
static u32 draw_mesh_run(u32 i) {
    const gl_draw *h = &rep_draws[ord[i]];
    const float ix = 2.0f / (float)rt_w, iy = 2.0f / (float)rt_h;
    const int self = h->tex_rt && (u32)(h->tex_rt - 1u) == h->rt;
    u32 n = 0, j = i;
    for (;;) {
        const gl_draw *d = &rep_draws[ord[j]];
        const ps2_vk_mesh_block *k = &rep_mesh->blocks[d->mblock];
        const mesh_rec *r = rep_mesh->recs + d->mvbase;
        const u16 *ix16 = rep_mesh->idx + d->first;
        u32 cnt = d->count - d->count % 3u;
        for (u32 q = 0; q < d->mvn; q++) mesh_xform(k, &r[q], ix, iy, &mesh_vcache[q]);
        for (u32 q = 0; q < cnt; q++) {
            if (n >= MESH_RUN_MAX - 2u && q % 3u == 0u) { mesh_run_flush(n); n = 0; }
            mesh_scratch[n++] = mesh_vcache[ix16[q]];
        }
        stat_drawn_tri++;
        frame_3d_tris += cnt / 3u;
        note_3d(d, cnt / 3u);
        stat_mesh_draws++;
        if (self) { stat_break[BR_SELF]++; break; }
        if (j + 1u >= rep_ndraws) break;
        {   const gl_draw *nx = &rep_draws[ord[j + 1u]];
            if (nx->native != PS2_VK_NATIVE_MESH) { stat_break[BR_NOTMESH]++; break; }
            if (!mesh_draw_ok(nx) || !mesh_cpu_wanted(nx)) { stat_break[BR_BIG]++; break; }
            if (!same_draw_state(h, nx)) { stat_break[break_reason(h, nx)]++; break; }
        }
        j++;
    }
    mesh_run_flush(n);
    return j;
}

/* The GPU loads and stores only the part of a target a scene touches when
 * told: the union of the scissors of the draws from here until the target
 * changes (a scene can only end earlier, so this covers it). PS2 effect
 * buffers -- glow, blur, the radar -- are small corners of their target. */
static u64 stat_region_px, stat_region_full_px, stat_region_scenes;
static void scene_region_hint(u32 k, u32 rt) {
    static int off = -1;
    s32 x1 = 0, y1 = 0;
    if (off < 0) off = getenv("AC5_GL_NOREGION") ? 1 : 0;
    if (off) { vglSetNextSceneRegion(0, 0, 0); return; }
    for (u32 j = k; j < rep_ndraws; j++) {
        const gl_draw *e = &rep_draws[ord[j]];
        s32 ex, ey;
        if (e->rt != rt) break;
        ex = e->scissor[1] + 1; ey = e->scissor[3] + 1;
        if (ex > x1) x1 = ex;
        if (ey > y1) y1 = ey;
    }
    if (x1 > (s32)rt_w) x1 = (s32)rt_w;
    if (y1 > (s32)rt_h) y1 = (s32)rt_h;
    if (x1 <= 0 || y1 <= 0) { vglSetNextSceneRegion(0, 0, 0); return; }
    {   u32 xm = (u32)((float)x1 * rt_scale + 0.999f), ym = (u32)((float)y1 * rt_scale + 0.999f);
        if (xm > (u32)PW) xm = (u32)PW;
        if (ym > (u32)PH) ym = (u32)PH;
        vglSetNextSceneRegion(1, xm - 1u, ym - 1u);
        stat_region_px += (u64)xm * ym;
        stat_region_full_px += (u64)PW * PH;
        stat_region_scenes++;
    }
}

/* Neighbouring GS draws with identical state and adjacent vertices are one
 * glDrawArrays: each vitaGL draw costs ~70 us, and missions issue hundreds
 * of small ones (HUD, effects, lines). AC5_GL_NOMERGE turns it off. */
static u64 stat_gs_merged, stat_gs_break[9];
/* why the next GS draw was not merged: 0 vertices not adjacent, 1 texture,
 * 2 shader variant, 3 blend, 4 depth, 5 scissor, 6 UV clamp, 7 target, 8 other */
static void gs_break_note(const gl_draw *a, const gl_draw *b, u32 next_first) {
    int r;
    if (!b->count || b->skip || b->native != a->native || a->topology != b->topology) return;
    if (b->first != next_first) r = 0;
    else if (a->tex != b->tex || a->tex_rt != b->tex_rt) r = 1;
    else if (a->variant != b->variant) r = 2;
    else if (a->blend != b->blend || a->fix != b->fix) r = 3;
    else if (a->depth != b->depth || a->aref != b->aref) r = 4;
    else if (memcmp(a->scissor, b->scissor, sizeof a->scissor)) r = 5;
    else if (a->minu != b->minu || a->maxu != b->maxu || a->minv != b->minv || a->maxv != b->maxv
             || a->tex_w != b->tex_w || a->tex_h != b->tex_h) r = 6;
    else if (a->rt != b->rt || a->zrt != b->zrt) r = 7;
    else r = 8;
    stat_gs_break[r]++;
}
static int gs_same(const gl_draw *a, const gl_draw *b) {
    return b->count && !b->skip && a->native == b->native && a->topology == b->topology
        && a->tex == b->tex && a->tex_rt == b->tex_rt && a->rt == b->rt && a->zrt == b->zrt
        && a->tex_self == b->tex_self && a->tex_depth == b->tex_depth
        && a->variant == b->variant && a->blend == b->blend && a->fix == b->fix
        && a->depth == b->depth && a->aref == b->aref && a->fogcol == b->fogcol
        && a->minu == b->minu && a->maxu == b->maxu && a->minv == b->minv && a->maxv == b->maxv
        && a->tex_w == b->tex_w && a->tex_h == b->tex_h && a->fb_w == b->fb_w
        && a->point_filter == b->point_filter
        && a->scissor[0] == b->scissor[0] && a->scissor[1] == b->scissor[1]
        && a->scissor[2] == b->scissor[2] && a->scissor[3] == b->scissor[3];
}
static void render_list(void) {
    if (no_sticky < 0) no_sticky = getenv("AC5_GL_NOSTICKY") ? 1 : 0;
    tm_frames++;
    tm_last = sceKernelGetProcessTimeWide();
    int cur_rt = -1, cur_owner = -1;
    if (dump_requested) {
        dump_requested = 0;
        sceIoMkdir("ux0:data/ac5/dump", 0777);
        dump_list = fopen("ux0:data/ac5/dump/drawlist.txt", "w");
        dump_active = 1;
        memset(dump_rt_used, 0, sizeof dump_rt_used);
        memset(dump_tex_done, 0, sizeof dump_tex_done);
        if (dump_list)
            fprintf(dump_list, "# frame: %u draws, %u verts, present rt %u at %u,%u %ux%u, targets %ux%u\n"
                    "# i rt zrt topo count variant blend fix depth aref texrt tex(w x h) scissor bbox z(min..max) skip native\n",
                    rep_ndraws, rep_nverts, present_rt, present_x, present_y, present_w, present_h, rt_w, rt_h);
    }
    gl_variant *bound = NULL;
    GLuint bound_tex = 0xFFFFFFFFu;
    u32 bound_blend = 0xFFFFFFFFu, bound_depth = 0xFFFFFFFFu;
    int layout_bound = 0;

    upload_frame_vertices();
    for (u32 i = 0; i < rep_ndraws; i++) {
        gl_draw *d = &rep_draws[i];
        u16 ti = d->tex;
        if (ti != 0xFFFFu && ti < ntextures) {
            gl_texture *t = &textures[ti];
            if (t->rgba && t->uploaded_gen != __atomic_load_n(&t->generation, __ATOMIC_ACQUIRE))
                upload_texture(t);
        }
        if (d->rt < RT_SLOTS) create_target(&rts[d->rt]);
        if (d->tex_rt && d->tex_rt <= RT_SLOTS) create_target(&rts[d->tex_rt - 1]);
        if (d->zrt < RT_SLOTS && d->zrt != d->rt && draw_uses_z(d)) {
            create_target(&rts[d->zrt]);
            is_zbuf[d->zrt] = 1;
        }
    }

    tm_mark(TM_UPLOAD);
    build_order();
    glEnable(GL_SCISSOR_TEST);
    for (u32 k = 0; k < rep_ndraws; k++) {
        const u32 i = ord[k];
        gl_draw *d = &rep_draws[i];
        const int is_mesh = d->native == PS2_VK_NATIVE_MESH;
        tm_mark(TM_OTHER);
        if (dump_active && dump_list && is_mesh) {
            fprintf(dump_list, "%u %u %u %u %u %05X %03X %u %03X %u %u %u mesh vbase %u verts %u block %u prog %u pass %u\n",
                    i, d->rt, d->zrt, d->topology, d->count, d->variant, d->blend, d->fix, d->depth,
                    d->aref, d->tex_rt, d->tex, d->mvbase, d->mvn, d->mblock,
                    rep_mesh && d->mblock < rep_mesh->nblocks ? rep_mesh->blocks[d->mblock].prog : 99u,
                    rep_mesh && d->mblock < rep_mesh->nblocks ? rep_mesh->blocks[d->mblock].pass : 99u);
        } else if (dump_active && dump_list) {
            const ps2_vk_vertex *vv = &rep_verts[d->first];
            float mnx = 1e9f, mny = 1e9f, mxx = -1e9f, mxy = -1e9f, mnz = 1e9f, mxz = -1e9f;
            for (u32 k = 0; k < d->count; k++) {
                if (vv[k].x < mnx) mnx = vv[k].x;
                if (vv[k].x > mxx) mxx = vv[k].x;
                if (vv[k].y < mny) mny = vv[k].y;
                if (vv[k].y > mxy) mxy = vv[k].y;
                if (vv[k].z < mnz) mnz = vv[k].z;
                if (vv[k].z > mxz) mxz = vv[k].z;
            }
            u32 tw = 0, th = 0;
            if (d->tex != 0xFFFFu && d->tex < ntextures) { tw = textures[d->tex].w; th = textures[d->tex].h; }
            fprintf(dump_list, "%u %u %u %u %u %05X %03X %u %03X %u %u %u(%ux%u) %d,%d-%d,%d %.0f,%.0f-%.0f,%.0f %.6f..%.6f %u %u\n",
                    i, d->rt, d->zrt, d->topology, d->count, d->variant, d->blend, d->fix, d->depth, d->aref,
                    d->tex_rt, d->tex, tw, th, d->scissor[0], d->scissor[2], d->scissor[1], d->scissor[3],
                    mnx, mny, mxx, mxy, mnz, mxz, d->skip, d->native);
            if (d->count) {
                const ps2_vk_vertex *a = &vv[0], *b = &vv[d->count - 1];
                fprintf(dump_list, "#   v0 xy %.1f,%.1f stq %.4f,%.4f,%.4f rgba %.0f,%.0f,%.0f,%.0f | vN xy %.1f,%.1f stq %.4f,%.4f,%.4f rgba %.0f,%.0f,%.0f,%.0f\n",
                        a->x, a->y, a->s, a->t, a->q, a->r, a->g, a->b, a->a,
                        b->x, b->y, b->s, b->t, b->q, b->r, b->g, b->b, b->a);
            }
            if (d->tex != 0xFFFFu && d->tex < ntextures && d->tex < 4096u
                && !(dump_tex_done[d->tex >> 3] & (1u << (d->tex & 7u)))) {
                dump_tex_done[d->tex >> 3] |= (u8)(1u << (d->tex & 7u));
            }
            if (d->rt < RT_SLOTS) dump_rt_used[d->rt] = 1;
            if (d->zrt < RT_SLOTS) dump_rt_used[d->zrt] = 1;
        }
        if (!d->count) continue;
        if (d->skip) { stat_skip[SK_SHUFFLE]++; continue; }
        if (is_mesh) {
            if (!mesh_draw_ok(d)) { stat_skip[SK_NATIVE]++; continue; }
        } else if (d->native && d->native != PS2_VK_NATIVE_2D) { stat_skip[SK_NATIVE]++; continue; }
        if (d->rt >= RT_SLOTS || !rts[d->rt].created) { stat_skip[SK_NORT]++; continue; }
        {   /* AC5_GL_NOFX=1 (an experiment): leave out the passes into the small
             * effect targets and the draws that read them back (glow, blur,
             * heat haze), to see what they cost the GPU */
            static int nofx = -1;
            if (nofx < 0) nofx = getenv("AC5_GL_NOFX") ? 1 : 0;
            if (nofx && (d->rt != present_rt
                         || (d->tex_rt && (u32)(d->tex_rt - 1) != present_rt))) { stat_skip[SK_NORT]++; continue; }
        }
        gl_target *t = &rts[d->rt];

        tm_mark(TM_OTHER);
        /* Pick the texture first: a self-read needs a snapshot, which
         * changes the bound framebuffer. */
        GLuint tex = 0;
        float img_w = 1.0f, img_h = 1.0f;
        int *filter_cache = NULL;
        if (d->variant & 1u) {
            if (d->tex_rt && d->tex_rt <= RT_SLOTS) {
                gl_target *s = &rts[d->tex_rt - 1];
                if ((int)(d->tex_rt - 1) == (int)d->rt) {
                    int rebound;
                    tex = snapshot((int)d->rt, &rebound);
                    if (rebound) {
                        /* the copy rebinds framebuffer, program, buffers,
                         * texture and turns state off: forget all of it */
                        cur_rt = -1;
                        cur_owner = -1;
                        bound = NULL;
                        bound_tex = 0xFFFFFFFFu;
                        bound_blend = bound_depth = 0xFFFFFFFFu;
                        layout_bound = 0;
                        glEnable(GL_SCISSOR_TEST);
                    }
                    if (!tex) { stat_skip[SK_NOSNAP]++; continue; }
                } else {
                    tex = s->color;
                }
                img_w = (float)rt_w;
                img_h = (float)rt_h;
            } else if (d->tex != 0xFFFFu && d->tex < ntextures && textures[d->tex].name) {
                tex = textures[d->tex].name;
                filter_cache = &textures[d->tex].filter;
                img_w = (float)textures[d->tex].w;
                img_h = (float)textures[d->tex].h;
            }
        }

        /* The GS keeps depth in ZBUF, shared by whatever FRAME is drawn
         * with it; vitaGL keeps one hidden depth buffer per framebuffer.
         * So the depth "owner" is the zrt slot's framebuffer, and the colour
         * target being drawn is attached to it for the draw (same size, so
         * vitaGL keeps the depth buffer). Draws that do not touch depth use
         * the colour target's own framebuffer. */
        tm_mark(TM_PICK);
        {
            int owner = (int)d->rt;
            if (draw_uses_z(d) && d->zrt < RT_SLOTS && d->zrt != d->rt && rts[d->zrt].created)
                owner = (int)d->zrt;
            /* A draw that does not touch depth can go into whichever
             * framebuffer already has this colour target attached: every
             * framebuffer change is a new GPU scene, and missions were
             * flipping between the depth owner and the target's own one
             * for each depth-less draw. */
            else if (!draw_uses_z(d) && cur_rt == (int)d->rt && cur_owner >= 0 && !no_sticky)
                owner = cur_owner;
            /* Only framebuffers that hold a depth buffer between scenes pay
             * for loading and storing it (see the vitaGL patch). Checked on
             * every depth draw, not only on a change of framebuffer: a depth
             * draw can land in the framebuffer that is already bound. The
             * flag takes effect at the start of the next scene, so a scene
             * already under way is ended to make this one keep its depth. */
            if (draw_uses_z(d) && !rts[owner].depth_kept) {
                vglSetFramebufferDepthPersistence(rts[owner].fbo, GL_TRUE);
                rts[owner].depth_kept = 1;
                if (owner == cur_owner) { cur_owner = -1; glBindFramebuffer(GL_FRAMEBUFFER, 0); }
            }
            if (owner != cur_owner || (int)d->rt != cur_rt) {
                gl_target *o = &rts[owner];
                scene_region_hint(k, d->rt);
                glBindFramebuffer(GL_FRAMEBUFFER, o->fbo);
                if (o->attached != t->color) {
                    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, t->color, 0);
                    o->attached = t->color;
                    stat_zswaps++;
                }
                glViewport(0, 0, PW, PH);
                cur_rt = (int)d->rt;
                cur_owner = owner;
                layout_bound = 0;
                stat_rt_binds++;
                scene_new = 1;
            }
        }
        /* A draw into a Z buffer as if it were a colour buffer (how games
         * clear or stamp depth) changes depth on the GS: do the same to the
         * depth buffer of that slot's framebuffer. */
        tm_mark(TM_BIND);
        if (!is_mesh && is_zbuf[d->rt] && !(d->variant & 1u) && ((d->depth >> 3) & 1u)) {
            const ps2_vk_vertex *vv = &rep_verts[d->first];
            float mnx = 1e9f, mny = 1e9f, mxx = -1e9f, mxy = -1e9f;
            for (u32 k = 0; k < d->count; k++) {
                if (vv[k].x < mnx) mnx = vv[k].x;
                if (vv[k].x > mxx) mxx = vv[k].x;
                if (vv[k].y < mny) mny = vv[k].y;
                if (vv[k].y > mxy) mxy = vv[k].y;
            }
            if (mxx > mnx && mxy > mny) {
                double zv = ((double)vv[0].r + (double)vv[0].g * 256.0 + (double)vv[0].b * 65536.0)
                          / 16777215.0;
                if (zv > 1.0) zv = 1.0;
                if (zv < 0.0) zv = 0.0;
                gl_target *zt = &rts[d->rt];
                vglSetNextSceneRegion(0, 0, 0);
                glBindFramebuffer(GL_FRAMEBUFFER, zt->fbo);
                if (!zt->depth_kept) { vglSetFramebufferDepthPersistence(zt->fbo, GL_TRUE); zt->depth_kept = 1; }
                glViewport(0, 0, PW, PH);
                glScissor((GLint)(mnx * rt_scale), (GLint)(mny * rt_scale),
                          (GLsizei)((mxx - mnx) * rt_scale + 0.999f), (GLsizei)((mxy - mny) * rt_scale + 0.999f));
                glDepthMask(GL_TRUE);
                glClearDepthf((GLfloat)zv);
                glClear(GL_DEPTH_BUFFER_BIT);
                stat_zclears++;
                bound_depth = 0xFFFFFFFFu;
                cur_owner = -1;
                cur_rt = -1;
                /* rebind for the colour draw itself */
                glBindFramebuffer(GL_FRAMEBUFFER, zt->fbo);
                if (zt->attached != t->color) {
                    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, t->color, 0);
                    zt->attached = t->color;
                }
                cur_owner = cur_rt = (int)d->rt;
                layout_bound = 0;
            }
        }
        t->dirty_seq++;

        const int mesh_cpu = is_mesh && mesh_cpu_wanted(d);
        /* bit 20: the stored picture is smaller than the GS texture, so the
         * shader must trim (most textures are stored whole and skip that) */
        u32 vkey = d->variant;
        if (d->variant & 1u) {
            const float gw = d->tex_w > 0.0f ? d->tex_w : 1.0f, gh = d->tex_h > 0.0f ? d->tex_h : 1.0f;
            if (img_w + 0.5f <= gw || img_h + 0.5f <= gh) vkey |= 0x100000u;
            /* bit 21: an ordinary texture stored at exactly its GS size (GL_REPEAT) */
            else if (!d->tex_rt && tex != white_tex && img_w == gw && img_h == gh) vkey |= 0x200000u;
        }
        gl_variant *v = get_variant(vkey, mesh_cpu ? 3 : is_mesh ? 2 : d->topology == PS2_VK_POINTS);
        if (!v) { stat_skip[SK_NOPROG]++; continue; }
        if (v != bound) {
            glUseProgram(v->prog);
            bound = v;
        }
        if (!layout_bound) { enable_gs_attribs(); layout_bound = 1; }

        /* scissor, in GS pixels: GS row y is GL row y in our targets */
        {
            s32 x0 = d->scissor[0] < 0 ? 0 : d->scissor[0];
            s32 y0 = d->scissor[2] < 0 ? 0 : d->scissor[2];
            s32 x1 = d->scissor[1] + 1, y1 = d->scissor[3] + 1;
            if (x1 > (s32)rt_w) x1 = (s32)rt_w;
            if (y1 > (s32)rt_h) y1 = (s32)rt_h;
            if (x1 <= x0 || y1 <= y0) { stat_skip[SK_SCISSOR]++; continue; }
            if (rt_scale == 1.0f) {
                glScissor(x0, y0, x1 - x0, y1 - y0);
            } else {
                GLint sx0 = (GLint)((float)x0 * rt_scale), sy0 = (GLint)((float)y0 * rt_scale);
                GLint sx1 = (GLint)((float)x1 * rt_scale + 0.999f), sy1 = (GLint)((float)y1 * rt_scale + 0.999f);
                glScissor(sx0, sy0, sx1 - sx0, sy1 - sy0);
            }
            if (!is_mesh) {   /* fill estimate: the draw's bounding box inside the scissor */
                const ps2_vk_vertex *vv = &rep_verts[d->first];
                float mnx = 1e9f, mny = 1e9f, mxx = -1e9f, mxy = -1e9f;
                for (u32 k = 0; k < d->count; k++) {
                    if (vv[k].x < mnx) mnx = vv[k].x;
                    if (vv[k].x > mxx) mxx = vv[k].x;
                    if (vv[k].y < mny) mny = vv[k].y;
                    if (vv[k].y > mxy) mxy = vv[k].y;
                }
                if (mnx < (float)x0) mnx = (float)x0;
                if (mny < (float)y0) mny = (float)y0;
                if (mxx > (float)x1) mxx = (float)x1;
                if (mxy > (float)y1) mxy = (float)y1;
                if (mxx > mnx && mxy > mny) stat_fill += (u64)((mxx - mnx) * (mxy - mny));
            }
        }

        if (d->blend != bound_blend) {
            if ((d->blend >> 8) & 1u) {
                GLenum s, dd, op;
                translate_blend(d->blend & 3u, (d->blend >> 2) & 3u,
                                (d->blend >> 4) & 3u, (d->blend >> 6) & 3u,
                                &s, &dd, &op);
                glEnable(GL_BLEND);
                glBlendEquationSeparate(op, GL_FUNC_ADD);
                glBlendFuncSeparate(s, dd, GL_ONE, GL_ZERO);
            } else {
                glDisable(GL_BLEND);
            }
            bound_blend = d->blend;
        }
        if (d->depth != bound_depth) {
            u32 k = d->depth;
            int rgb = (k >> 3) & 1u, a = (k >> 4) & 1u;
            static int nodepth = -1;
            if (nodepth < 0) nodepth = getenv("AC5_GL_NODEPTH") ? 1 : 0;
            glEnable(GL_DEPTH_TEST);
            glDepthFunc(nodepth ? GL_ALWAYS : depth_func(k & 3u));
            glDepthMask(((k >> 2) & 1u) ? GL_TRUE : GL_FALSE);
            glColorMask(rgb && !((k >> 5) & 1u), rgb && !((k >> 6) & 1u),
                        rgb && !((k >> 7) & 1u), a && !((k >> 8) & 1u));
            bound_depth = d->depth;
        }

        {
            float xf[4] = { 2.0f / (float)rt_w, 2.0f / (float)rt_h, 1.0f, 0.0f };
            set_u4(v, 0, v->u_xform, xf);
        }
        if (d->variant & 1u) {
            float ti4[4] = { d->tex_w > 0.0f ? d->tex_w : 1.0f, d->tex_h > 0.0f ? d->tex_h : 1.0f,
                             1.0f / img_w, 1.0f / img_h };
            float cl[4] = { (float)d->minu, (float)d->maxu, (float)d->minv, (float)d->maxv };
            set_u4(v, 1, v->u_texinfo, ti4);
            set_u4(v, 2, v->u_clamp, cl);
            glActiveTexture(GL_TEXTURE0);
            if (!tex) tex = white_tex;
            if (tex != bound_tex || filter_cache || tex == white_tex) {
                if (tex == white_tex) glBindTexture(GL_TEXTURE_2D, white_tex);
                else set_filter(tex, filter_cache, d->point_filter);
                bound_tex = tex;
            }
        }
        {
            float misc[4] = { (float)(d->aref & 0xFFu) / 128.0f, (float)d->fix / 128.0f,
                              (float)((d->variant >> 13) & 3u), (float)((d->variant >> 15) & 3u) };
            float fog[4] = { (float)((d->fogcol >> 0) & 0xFF) / 255.0f,
                             (float)((d->fogcol >> 8) & 0xFF) / 255.0f,
                             (float)((d->fogcol >> 16) & 0xFF) / 255.0f, 1.0f };
            if (misc[1] > 1.0f) misc[1] = 1.0f;
            {   u32 k = d->variant;
                float f1[4] = { (float)((k >> 1) & 3u), (k >> 3) & 1u ? 1.0f : 0.0f,
                                (k >> 11) & 1u ? 1.0f : 0.0f, (k >> 4) & 1u ? 1.0f : 0.0f };
                float f2[4] = { (k >> 8) & 1u ? 1.0f : 0.0f, (float)((k >> 5) & 7u),
                                (k >> 10) & 1u ? 2.0f : ((k >> 9) & 1u ? 1.0f : 0.0f),
                                (k >> 18) & 1u ? 1.0f : 0.0f };
                float f3[4] = { (k >> 19) & 1u ? 1.0f : 0.0f, (k >> 12) & 1u ? 1.0f : 0.0f,
                                (k >> 17) & 1u ? 1.0f : 0.0f, 0.0f };
                set_u4(v, 3, v->u_f1, f1);
                set_u4(v, 4, v->u_f2, f2);
                set_u4(v, 5, v->u_f3, f3);
            }
            set_u4(v, 6, v->u_misc, misc);
            set_u4(v, 7, v->u_fog, fog);
        }

        GLenum mode = d->topology == PS2_VK_POINTS ? GL_POINTS
                    : d->topology == PS2_VK_LINES ? GL_LINES : GL_TRIANGLES;
        tm_mark(TM_STATE);
        if (mesh_cpu) {
            k = draw_mesh_run(k);
            layout_bound = 0;
            tm_mark(TM_MESHRUN);
            continue;
        }
        if (is_mesh) {
            draw_mesh_gl(v, d, mode);
            layout_bound = 0;
            tm_mark(TM_MESHGPU);
            continue;
        }
        if (mode == GL_LINES) stat_drawn_line++;
        else if (mode == GL_POINTS) stat_drawn_point++;
        else {
            stat_drawn_tri++;
            if (d->count >= 3u) {   /* depth or perspective varies: a 3D scene */
                const ps2_vk_vertex *w = &rep_verts[d->first];
                if (w[0].z != w[1].z || w[0].z != w[2].z || w[0].q != w[1].q || w[0].q != w[2].q) {
                    frame_3d_tris += d->count / 3u;
                    note_3d(d, d->count / 3u);
                }
            }
        }
        {   static int nomerge = -1;
            u32 cnt = d->count;
            if (nomerge < 0) nomerge = getenv("AC5_GL_NOMERGE") ? 1 : 0;
            if (!nomerge && !d->tex_self && !(d->tex_rt && (int)(d->tex_rt - 1) == (int)d->rt)) {
                while (k + 1u < rep_ndraws) {
                    const gl_draw *e = &rep_draws[ord[k + 1u]];
                    if (e->first != d->first + cnt || !gs_same(d, e)) {
                        gs_break_note(d, e, d->first + cnt);
                        break;
                    }
                    cnt += e->count;
                    k++;
                    stat_gs_merged++;
                }
            }
            point_gs_vertices(&rep_verts[d->first]);
            draw_timed_arrays(mode, (GLsizei)cnt);
        }
        stat_draws++;
        tm_mark(TM_DRAW);
    }
    glDisable(GL_SCISSOR_TEST);
    stat_verts += rep_nverts;
    /* Widescreen applies to 3D scenes only: frames that are all 2D (movies,
     * the title, flat menus) keep their 4:3 shape with side borders.
     * Hysteresis keeps a fade or a short 2D moment from flipping it. */
    {   /* back to 4:3 only after 2 s without 3D, not 20 frames: at 20+ fps
         * a briefing map loading in passed 20 flat frames (build 83) */
        static u64 flat_since;
        u64 now = sceKernelGetProcessTimeWide();
        /* a flat frame between 3D ones (a mission at 512x448 alternated, build
         * 101: it stayed 4:3 with 3000 3D triangles a frame) no longer resets
         * the 3D count; only 0.3 s without 3D does */
        if (frame_3d_tris >= 64u) { wide_run++; flat_run = 0; flat_since = 0; }
        else {
            flat_run++;
            if (!flat_since) flat_since = now;
            if (now - flat_since >= 300000u) wide_run = 0;
        }
        {   int was = wide_mode;
            static u64 last_note;
            /* one 3D frame is enough (build 104): in the 512x448 scenes most
             * frames the renderer picked up had no 3D, so three never came */
            if (!wide_mode && wide_run >= 1u) wide_mode = 1;
            /* 5 s without 3D before 4:3 again (build 106): missions flipped
             * between the two with 2 s (build 105) */
            if (wide_mode && flat_run >= 20u && flat_since && now - flat_since >= 5000000u) wide_mode = 0;
            /* aspect diagnostics (build 100): each switch, and every 5 s the
             * state, with what decided it -- for the 4:3 in-engine cutscenes */
            if (was != wide_mode || now - last_note >= 5000000u) {
                last_note = now;
                ps2_log("gl: aspect %s%s | 3D tris this frame %u, present %ux%u at %u,%u from target %d%s",
                        wide_mode ? "16:9" : "4:3", was != wide_mode ? " (switched)" : "",
                        frame_3d_tris, present_w, present_h, present_x, present_y, present_rt,
                        present_blank ? " (blank)" : "");
            }
        }
    }
    frame_3d_tris = 0;
    memset(is_zbuf, 0, sizeof is_zbuf);
    if (dump_active) {
        dump_active = 0;
        if (dump_list) { fclose(dump_list); dump_list = NULL; }
        dump_rt_used[present_rt < RT_SLOTS ? present_rt : 0] = 1;
        dump_targets();
    }
}

static void present_frame(void) {
    /* see the replay's end for wide_mode */
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glViewport(0, 0, SCREEN_W, SCREEN_H);
    glDisable(GL_SCISSOR_TEST);
    glDisable(GL_BLEND);
    glDisable(GL_DEPTH_TEST);
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    if (!present_blank) {
        gl_target *t = &rts[present_rt < RT_SLOTS && rts[present_rt].created ? present_rt : 0];
        stat_present_rt[present_rt < RT_SLOTS ? present_rt : 0]++;
        if (t->created) {
            float aspect = ps2_cfg.widescreen && wide_mode ? 16.0f / 9.0f : 4.0f / 3.0f;
            float out_w = (float)SCREEN_H * aspect;
            if (out_w > (float)SCREEN_W) out_w = (float)SCREEN_W;
            float half = out_w / (float)SCREEN_W;
            float u0 = (float)present_x / (float)rt_w;
            float u1 = (float)(present_x + present_w) / (float)rt_w;
            float v0 = (float)present_y / (float)rt_h;
            float v1 = (float)(present_y + present_h) / (float)rt_h;
            ps2_vk_display_aspect = aspect;
            /* screen top (+1) shows GS row present_y */
            draw_copy(t->color, u0, v0, u1, v1, -half, -1.0f, half, 1.0f, 1);
            if (present_ov_rt >= 0 && rts[present_ov_rt].created && ov_prog) {
                float a0 = (float)present_ov_x / (float)rt_w, a1 = (float)(present_ov_x + present_w) / (float)rt_w;
                float b0 = (float)present_ov_y / (float)rt_h, b1 = (float)(present_ov_y + present_h) / (float)rt_h;
                float mode[4] = { present_ov_a >= 0.0f ? 1.0f : 0.0f, present_ov_a, 0.0f, 0.0f };
                glEnable(GL_BLEND);
                glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
                glUseProgram(ov_prog);
                glUniform4fv(ov_u_mode, 1, mode);
                draw_copy_with(ov_prog, rts[present_ov_rt].color, a0, b0, a1, b1, -half, -1.0f, half, 1.0f, 1);
                glDisable(GL_BLEND);
                stat_overlays++;
            }
        }
    }
    {   /* frame cap (AC5_FPS_CAP, default 30; 0 = off): a steady 30 looks
         * smoother than 20-40, and the Vita's 60 Hz shows 30 evenly */
        static int cap = -1;
        static u64 last;
        /* off by default until frames arrive evenly: with frames coming in bursts
         * after a stall, holding the quick ones back only cost time */
        if (cap < 0) { const char *e = getenv("AC5_FPS_CAP"); cap = e ? atoi(e) : 0; }
        if (cap > 0 && last) {
            u64 want = 1000000u / (u32)cap, now = sceKernelGetProcessTimeWide();
            if (now - last < want) sceKernelDelayThread((SceUInt)(want - (now - last)));
        }
        last = sceKernelGetProcessTimeWide();
    }
    vglSwapBuffers(GL_FALSE);
    stat_frames++;
}

static int shader_compiler_present(void) {
    static const char *paths[] = {
        "ur0:data/external/libshacccg.suprx",
        "ur0:data/libshacccg.suprx",
    };
    SceIoStat st;
    for (unsigned i = 0; i < sizeof paths / sizeof paths[0]; i++)
        if (sceIoGetstat(paths[i], &st) >= 0) return 1;
    return 0;
}

static int gl_init(void) {
    if (!shader_compiler_present()) {
        ps2_log("gl: the Vita shader compiler (libshacccg.suprx) is not "
                "installed. Extract it with the ShaRKF00D homebrew, then run "
                "again. Falling back to the software picture.");
        return -1;
    }
    vglSetupRuntimeShaderCompiler(SHARK_OPT_FAST, SHARK_ENABLE, SHARK_ENABLE, SHARK_ENABLE);
    {   const char *e = getenv("AC5_GL_SCALE");
        if (e && *e) {
            float f = (float)atof(e);
            if (f >= 0.25f && f <= 1.0f) rt_scale = f;
        }
        ps2_log("gl: internal resolution scale %.2f (AC5_GL_SCALE)", (double)rt_scale);
    }
    {   /* Size vitaGL's pools from what is actually free, keeping some main
         * RAM back for the thread stacks the game creates later and some
         * CDRAM for the software-picture fallback. */
        SceKernelFreeMemorySizeInfo fm;
        const int MB = 1024 * 1024;
        memset(&fm, 0, sizeof fm);
        fm.size = sizeof fm;
        sceKernelGetFreeMemorySize(&fm);
        int ram = fm.size_user - 24 * MB;
        int cdram = fm.size_cdram - 8 * MB;
        int phy = fm.size_phycont - 2 * MB;
        {   /* The free total is not what one block can take: build 10 asked
             * for 104 MB, got nothing (CDRAM stayed 96 MB free) and vitaGL
             * spilled into the C heap. Find the largest block that really
             * allocates, and leave room for the display buffers. */
            int got = 0;
            for (int sz = cdram > 0 ? cdram & ~(MB - 1) : 0; sz >= 16 * MB; sz -= 4 * MB) {
                SceUID b = sceKernelAllocMemBlock("ac5_probe", SCE_KERNEL_MEMBLOCK_TYPE_USER_CDRAM_RW,
                                                  (SceSize)sz, NULL);
                if (b >= 0) { sceKernelFreeMemBlock(b); got = sz; break; }
            }
            ps2_log("gl: largest CDRAM block that allocates: %d MB", got / MB);
            /* and vitaGL starts sceGxm before making its pools: the GXM
             * parameter buffer (16 MB) comes out of CDRAM first. Build 13
             * asked for 100 MB of the 112 MB and got nothing. */
            if (got < cdram) cdram = got;
            cdram -= 16 * MB + 4 * MB;
        }
        if (ram > 32 * MB) ram = 32 * MB;
        if (phy > 16 * MB) phy = 16 * MB;
        if (phy < 0) phy = 0;
        ps2_log("gl: free memory before init: main %d MB, CDRAM %d MB, phycont %d MB "
                "-> vitaGL pools main %d MB, CDRAM %d MB, phycont %d MB",
                fm.size_user / MB, fm.size_cdram / MB, fm.size_phycont / MB,
                ram / MB, cdram / MB, phy / MB);
        if (ram < 4 * MB || cdram < 16 * MB) {
            ps2_log("gl: not enough free memory for vitaGL");
            return -1;
        }
        /* In this vitaGL the return value is not a success flag: it is TRUE
         * only when the requested resolution had to be clamped to the
         * display's maximum (disassembled: vglInitWithCustomSizes.part.0
         * returns 1 for width/height above sceDisplayGetMaximumFrameBuf-
         * Resolution, else 0, after a complete init either way). So check
         * that the context answers instead. */
        /* vitaGL's garbage collector off core 2 (renderer + GS thread), where
         * at default priority it could fall behind and stall the swap */
        vglSetupGarbageCollector(0x10000100, SCE_KERNEL_CPU_MASK_USER_0 | SCE_KERNEL_CPU_MASK_USER_1);
        GLboolean clamped = vglInitWithCustomSizes(0, SCREEN_W, SCREEN_H, ram, cdram, phy, 0,
                                                   SCE_GXM_MULTISAMPLE_NONE);
        const GLubyte *ver = glGetString(GL_VERSION);
        const GLubyte *ren = glGetString(GL_RENDERER);
        ps2_log("gl: vitaGL init done%s: %s / %s", clamped ? " (resolution clamped)" : "",
                ver ? (const char *)ver : "?", ren ? (const char *)ren : "?");
        {   /* vitaGL records a pool's size before allocating it and
             * vglMemFree() on a pool whose block failed dereferences a NULL
             * mspace (build 11 crashed on its first frame that way), so
             * work out which pools really exist from what the system lost. */
            SceKernelFreeMemorySizeInfo f2;
            memset(&f2, 0, sizeof f2);
            f2.size = sizeof f2;
            sceKernelGetFreeMemorySize(&f2);
            pool_ok[0] = cdram > 0 && fm.size_cdram - f2.size_cdram >= cdram;
            pool_ok[1] = ram > 0 && fm.size_user - f2.size_user >= ram;
            pool_ok[2] = phy > 0 && fm.size_phycont - f2.size_phycont >= phy;
            ps2_log("gl: vitaGL pools: CDRAM %d MB %s, main %d MB %s, phycont %d MB %s "
                    "(system now free: main %d MB, CDRAM %d MB, phycont %d MB)",
                    cdram / MB, pool_ok[0] ? "made" : "FAILED",
                    ram / MB, pool_ok[1] ? "made" : "FAILED",
                    phy / MB, pool_ok[2] ? "made" : "FAILED",
                    f2.size_user / MB, f2.size_cdram / MB, f2.size_phycont / MB);
        }
        if (!ver) {
            ps2_log("gl: vitaGL did not come up (no GL_VERSION)");
            return -1;
        }
    }
    vglWaitVblankStart(GL_FALSE);
    vs_tri = compile(GL_CG_VERTEX_SHADER_EXT, NULL, VS_SRC);
    vs_point = compile(GL_CG_VERTEX_SHADER_EXT, "#define POINTS 1\n", VS_SRC);
    if (!vs_tri || !vs_point || make_copy_program() != 0) {
        ps2_log("gl: base shaders failed to compile");
        return -1;
    }
    for (int tex = 0; tex < 2; tex++)
        for (int pts = 0; pts < 2; pts++)
            if (build_program(&progs[tex][pts], tex, pts) != 0) {
                ps2_log("gl: GS shader (%s, %s) failed to compile",
                        tex ? "textured" : "untextured", pts ? "points" : "triangles");
                return -1;
            }
    if (mesh_switch() && mesh_alloc_ok) {
        vs_mesh = compile(GL_CG_VERTEX_SHADER_EXT, NULL, MESH_VS_SRC);
        if (vs_mesh && build_program(&progs[0][2], 0, 2) == 0
            && build_program(&progs[1][2], 1, 2) == 0 && progs[0][2].u_blk >= 0) {
            vs_clip = compile(GL_CG_VERTEX_SHADER_EXT, NULL, CLIP_VS_SRC);
            mesh_cpu_ok = vs_clip && build_program(&progs[0][3], 0, 3) == 0
                       && build_program(&progs[1][3], 1, 3) == 0;
            if (!mesh_cpu_ok) ps2_log("gl: pre-transformed mesh program failed; every mesh draws on its own");
            mesh_ready = 1;
            ps2_log("gl: native mesh path on (AC5_MESH=0 turns it off)");
        } else {
            ps2_log("gl: mesh vertex program failed; meshes go the GS way");
        }
    }
    ps2_log("gl: GS shaders compiled (%d programs)", nvariants);
    {   static const u8 white[4] = { 255, 255, 255, 255 };
        glGenTextures(1, &white_tex);
        glBindTexture(GL_TEXTURE_2D, white_tex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 1, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, white);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST); }
    glDisable(GL_CULL_FACE);
    prewarm_variants();
    ps2_log("gl: vitaGL renderer up (%dx%d screen)", SCREEN_W, SCREEN_H);
    return 0;
}

void vita_pin_current(int core);
static void *renderer_main(void *arg) {
    (void)arg;
    /* Core 0 below the game thread by default (build 98): on core 2 the
     * renderer held the GS thread off its core ~25 ms in every heavy field
     * (build 97's thread-state samples); the game thread leaves core 0 idle
     * over half the time. AC5_RENDER_CORE / AC5_RENDER_PRIO (2 / 140 = old). */
    {   extern int ps2_gsq_on;
        const char *c = getenv("AC5_RENDER_CORE"), *p = getenv("AC5_RENDER_PRIO");
        /* core 1 with the graphics worker (build 104): the GS thread gets core 2
         * to itself; heavy scenes ran ~30% faster than with the renderer on
         * core 2 (build 103's test). Core 0 starved it (build 98). */
        /* back on core 2 (build 107): on core 1 (104-106) heavy scenes were
         * faster but normal flight felt worse than build 96 */
        (void)ps2_gsq_on;
        vita_pin_current(c ? atoi(c) : 2);
        {   extern void vita_set_priority(int prio); vita_set_priority(p ? atoi(p) : 140); }
    }
    rn_thid = sceKernelGetThreadId();
    if (gl_init() != 0) {
        pthread_mutex_lock(&list_lock);
        renderer_running = -1;
        pthread_cond_broadcast(&list_cv);
        pthread_mutex_unlock(&list_lock);
        return NULL;
    }
    gl_ready = 1;
    pthread_mutex_lock(&list_lock);
    renderer_running = 1;
    pthread_cond_broadcast(&list_cv);
    pthread_mutex_unlock(&list_lock);

    frame_sema = sceKernelCreateSema("ac5_frame", 0, 0, 1, NULL);
    u64 loop_t = sceKernelGetProcessTimeWide();
    for (;;) {
        struct timespec ts;
        u64 w0 = sceKernelGetProcessTimeWide();
        pthread_mutex_lock(&list_lock);
        rl_lock_us += sceKernelGetProcessTimeWide() - w0;
        if (renderer_running <= 0) { pthread_mutex_unlock(&list_lock); break; }
        if (!frame_requested) {
            clock_gettime(CLOCK_REALTIME, &ts);
            ts.tv_nsec += 4 * 1000 * 1000;
            if (ts.tv_nsec >= 1000000000L) { ts.tv_sec++; ts.tv_nsec -= 1000000000L; }
            {   u64 i0 = sceKernelGetProcessTimeWide();
                if (frame_sema >= 0) {
                    SceUInt to = 4000;
                    pthread_mutex_unlock(&list_lock);
                    sceKernelWaitSema(frame_sema, 1, &to);
                    pthread_mutex_lock(&list_lock);
                } else {
                    pthread_cond_timedwait(&list_cv, &list_lock, &ts);
                }
                rl_idle_us += sceKernelGetProcessTimeWide() - i0;
            }
            if (!frame_requested) { pthread_mutex_unlock(&list_lock); continue; }
            rl_wake_us += sceKernelGetProcessTimeWide() - rl_last_pub;
            rl_wakes++;
        }
        rl_age_us += sceKernelGetProcessTimeWide() - rl_last_pub;
        rep_verts = pub_verts;
        rep_draws = pub_draws;
        rep_mesh = pub_mesh;
        rep_jobs = pub_jobs;
        rep_nverts = pub_nverts;
        rep_ndraws = pub_ndraws;
        frame_requested = 0;
        pthread_cond_broadcast(&list_cv);
        pthread_mutex_unlock(&list_lock);

        size_targets();
        snaps_this_frame = 0;
        spec_built_this_frame = 0;
        prewarm_step(rep_ndraws);
        sweep_textures();
        if ((stat_presents & 31u) == 0) {
            stat_vgl_vram_kb = pool_ok[0] ? (u32)(vglMemFree(VGL_MEM_VRAM) >> 10) : 0;
            stat_vgl_ram_kb = pool_ok[1] ? (u32)(vglMemFree(VGL_MEM_RAM) >> 10) : 0;
            stat_vgl_phy_kb = pool_ok[2] ? (u32)(vglMemFree(VGL_MEM_PHYCONT) >> 10) : 0;
        }
        {   /* the frame's textures must be decoded before it is drawn */
            u64 w0 = sceKernelGetProcessTimeWide();
            static int steal = -1;
            if (steal < 0) { const char *e = getenv("AC5_TEX_STEAL"); steal = !(e && *e == '0'); }
            while ((s32)(tj_prefix() - rep_jobs) < 0) {
                SceUInt to = 2000;
                if (steal) {
                    rn_busy_state = 2;
                    if (tj_run_one(2, rep_jobs)) continue;
                    rn_busy_state = 0;
                }
                sceKernelWaitSema(tj_done_sema, 1, &to);
            }
            stat_tj_waits_us += sceKernelGetProcessTimeWide() - w0;
        }
        {   /* Give way to the GS thread (build 108): on the same core and
             * below the renderer, it was held off ~30 ms whenever a frame
             * came in while it still had fields queued. Wait up to
             * AC5_RENDER_DEFER_MS (default 8) for it to catch up. */
            extern int ps2_gsq_backlog(void);
            static int defer_ms = -1;
            if (defer_ms < 0) { const char *e = getenv("AC5_RENDER_DEFER_MS"); defer_ms = e ? atoi(e) : 8; }
            for (int w = 0; w < defer_ms && ps2_gsq_backlog(); w++) { sceKernelDelayThread(1000); stat_defer_ms++; }
        }
        rn_busy_state = 1;
        rn_ph = 10;
        {   u64 t0 = sceKernelGetProcessTimeWide();
            render_list();
            u64 t1 = sceKernelGetProcessTimeWide();
            rn_ph = 11;
            present_frame();
            rn_ph = 12;
            u64 t2 = sceKernelGetProcessTimeWide();
            t_submit_us += t1 - t0;
            t_swap_us += t2 - t1;
            t_frames_timed++;
        }

        {   u64 now = sceKernelGetProcessTimeWide();
            rl_wall_us += now - loop_t;
            loop_t = now;
            rl_frames++;
        }
        rn_busy_state = 0;
        pthread_mutex_lock(&list_lock);
        frame_done = 1;
        pthread_cond_broadcast(&list_cv);
        pthread_mutex_unlock(&list_lock);
    }
    return NULL;
}

static pthread_t renderer_thread;

int ps2_video_init(const char *title, int width, int height) {
    (void)title;
    present_w = (u32)(width > 0 ? width : 640);
    present_h = (u32)(height > 0 ? height : 448);
    memset(tex_head, 0xFF, sizeof tex_head);
    for (int s = 0; s < LIST_SLOTS; s++) {
        slot_verts[s] = (ps2_vk_vertex *)malloc(sizeof(ps2_vk_vertex) * GL_MAX_VERTS);
        slot_draws[s] = (gl_draw *)malloc(sizeof(gl_draw) * GL_MAX_DRAWS);
        if (!slot_verts[s] || !slot_draws[s]) {
            ps2_log("gl: cannot allocate the display lists");
            return -1;
        }
    }
    rec_slot = 0;
    pub_slot = LIST_SLOTS - 1;
    rec_verts = slot_verts[rec_slot];
    rec_draws = slot_draws[rec_slot];
    rep_verts = pub_verts = slot_verts[LIST_SLOTS - 1];
    rep_draws = pub_draws = slot_draws[LIST_SLOTS - 1];
    if (mesh_switch()) {
        mesh_alloc_ok = 1;
        for (int s = 0; s < LIST_SLOTS; s++) {
            slot_mesh[s].recs = (mesh_rec *)malloc(sizeof(mesh_rec) * MESH_V_MAX);
            slot_mesh[s].idx = (u16 *)malloc(sizeof(u16) * MESH_I_MAX);
            slot_mesh[s].blocks = (ps2_vk_mesh_block *)malloc(sizeof(ps2_vk_mesh_block) * MESH_B_MAX);
            if (!slot_mesh[s].recs || !slot_mesh[s].idx || !slot_mesh[s].blocks) mesh_alloc_ok = 0;
        }
        if (!mesh_alloc_ok) ps2_log("gl: no memory for the native mesh arenas; meshes go the GS way");
    }
    rec_mesh = &slot_mesh[rec_slot];
    rep_mesh = pub_mesh = &slot_mesh[LIST_SLOTS - 1];
    {
        const char *e = getenv("PS2_LOCKSTEP");
        if (e && *e && *e != '0') ps2_vk_lockstep = 1;
    }
    {   /* The renderer gets a big stack: shader compiles are deep. */
        pthread_attr_t a;
        pthread_attr_init(&a);
        pthread_attr_setstacksize(&a, 2 * 1024 * 1024);
        int rc = pthread_create(&renderer_thread, &a, renderer_main, NULL);
        pthread_attr_destroy(&a);
        if (rc != 0) return -1;
    }
    pthread_mutex_lock(&list_lock);
    while (renderer_running == 0)
        pthread_cond_wait(&list_cv, &list_lock);
    pthread_mutex_unlock(&list_lock);
    return renderer_running > 0 ? 0 : -1;
}

void ps2_video_hidden(int on) { (void)on; }

void ps2_video_shutdown(void) {
    pthread_mutex_lock(&list_lock);
    if (renderer_running > 0) renderer_running = 0;
    pthread_cond_broadcast(&list_cv);
    pthread_mutex_unlock(&list_lock);
}

void ps2_vk_stats(u64 *frames, u64 *draws, u64 *verts, u64 *tex) {
    if (frames) *frames = stat_frames;
    if (draws) *draws = stat_draws;
    if (verts) *verts = stat_verts;
    if (tex) *tex = ntextures;
}

void ps2_video_report(void) {
    ps2_log("gl: %llu frames, %llu draws, %llu verts, %u textures (%.1f MB), "
            "%llu uploads, %d shader variants, %llu fields dropped",
            (unsigned long long)stat_frames, (unsigned long long)stat_draws,
            (unsigned long long)stat_verts, ntextures, (double)tex_bytes / 1048576.0,
            (unsigned long long)stat_uploads, nvariants,
            (unsigned long long)stat_dropped);
}

/* Per-interval numbers for the Vita speed line (vita_display.c). */
void vita_gl_detail(char *buf, size_t n) {
    u64 f = t_frames_timed ? t_frames_timed : 1;
    snprintf(buf, n, "per frame: draw submit %.1f ms, present+swap %.1f ms | drawn tri %llu line %llu point %llu | "
             "skipped: shuffle/depth-read %llu, native %llu, no target %llu, no snapshot %llu, no shader %llu, empty scissor %llu | "
             "mesh: %llu draws (%llu GPU runs of pre-transformed ones), %llu stores, %llu verts, %llu overflows",
             (double)t_submit_us / 1000.0 / (double)f, (double)t_swap_us / 1000.0 / (double)f,
             (unsigned long long)stat_drawn_tri, (unsigned long long)stat_drawn_line,
             (unsigned long long)stat_drawn_point,
             (unsigned long long)stat_skip[SK_SHUFFLE], (unsigned long long)stat_skip[SK_NATIVE],
             (unsigned long long)stat_skip[SK_NORT], (unsigned long long)stat_skip[SK_NOSNAP],
             (unsigned long long)stat_skip[SK_NOPROG], (unsigned long long)stat_skip[SK_SCISSOR],
             (unsigned long long)stat_mesh_draws, (unsigned long long)stat_mesh_runs,
             (unsigned long long)stat_mesh_stores,
             (unsigned long long)stat_mesh_verts, (unsigned long long)stat_mesh_overflow);
    stat_mesh_draws = stat_mesh_stores = stat_mesh_verts = stat_mesh_overflow = stat_mesh_runs = 0;
    {   size_t o = strlen(buf);
        o += (size_t)snprintf(buf + o, o < n ? n - o : 0, " | 3D tris by target:");
        for (int r = 0; r < RT_SLOTS && o < n; r++)
            if (stat_3d_rt[r]) o += (size_t)snprintf(buf + o, n - o, " rt%d=%llu", r, (unsigned long long)stat_3d_rt[r]);
        if (o < n) o += (size_t)snprintf(buf + o, n - o, " | ztst never/always/gequal/greater %llu/%llu/%llu/%llu | shown:",
                                         (unsigned long long)stat_3d_ztst[0], (unsigned long long)stat_3d_ztst[1],
                                         (unsigned long long)stat_3d_ztst[2], (unsigned long long)stat_3d_ztst[3]);
        for (int r = 0; r < RT_SLOTS && o < n; r++)
            if (stat_present_rt[r]) o += (size_t)snprintf(buf + o, n - o, " rt%d=%llu", r, (unsigned long long)stat_present_rt[r]);
        memset(stat_3d_rt, 0, sizeof stat_3d_rt);
        memset(stat_3d_ztst, 0, sizeof stat_3d_ztst);
        memset(stat_present_rt, 0, sizeof stat_present_rt);
    }
    t_submit_us = t_swap_us = t_frames_timed = 0;
    stat_drawn_tri = stat_drawn_line = stat_drawn_point = 0;
    memset(stat_skip, 0, sizeof stat_skip);
}

/* Renderer time by category and why mesh draws were not merged (second line). */
void vita_gl_timing(char *buf, size_t n) {
    u64 f = tm_frames ? tm_frames : 1;
    {   size_t o = 0;
        buf[0] = 0;
        static const char *tn[TM_N] = { "uploads", "texture pick+snapshots", "target binds", "state",
                                        "GS draws", "mesh runs", "mesh GPU draws", "other" };
        static const char *bn[BR_N] = { "next not mesh", "texture", "blend", "shader", "depth",
                                        "target", "region/scissor", "big mesh", "self-read" };
        o += (size_t)snprintf(buf + o, o < n ? n - o : 0, "ms/frame:");
        for (int c = 0; c < TM_N && o < n; c++)
            o += (size_t)snprintf(buf + o, n - o, " %s %.1f", tn[c], (double)tm_us[c] / 1000.0 / (double)f);
        if (o < n) o += (size_t)snprintf(buf + o, n - o, " | target runs per frame %.1f (was %.1f before regrouping), %llu draws moved",
                                         (double)stat_segs / (double)f, (double)stat_segs_before / (double)f,
                                         (unsigned long long)stat_moved);
        stat_segs = stat_segs_before = stat_moved = 0;
        if (o < n && rl_frames) o += (size_t)snprintf(buf + o, n - o,
            " | renderer loop %.1f ms/frame wall (%llu frames): waiting for a frame %.1f, list lock %.1f;"
            " %llu frames published (%llu replaced an unrendered one), longest gap between publishes %.1f ms",
            rl_wall_us / 1000.0 / (double)rl_frames, (unsigned long long)rl_frames,
            rl_idle_us / 1000.0 / (double)rl_frames, rl_lock_us / 1000.0 / (double)rl_frames,
            (unsigned long long)(stat_presents - rl_pres0), (unsigned long long)(stat_dropped - rl_drop0),
            rl_gap_max / 1000.0);
        u64 rl_frames_seen = rl_frames;
        rl_wall_us = rl_idle_us = rl_lock_us = rl_frames = 0;
        rl_pres0 = stat_presents; rl_drop0 = stat_dropped; rl_gap_max = 0;
        if (o < n) o += (size_t)snprintf(buf + o, n - o,
            " | publish gaps <10ms %llu <20 %llu <40 %llu <80 %llu >=80 %llu; picked-up frame age %.1f ms;"
            " after a wait, woke %.1f ms after the publish (%llu waits)",
            (unsigned long long)rl_gap_hist[0], (unsigned long long)rl_gap_hist[1], (unsigned long long)rl_gap_hist[2],
            (unsigned long long)rl_gap_hist[3], (unsigned long long)rl_gap_hist[4],
            rl_frames_seen ? rl_age_us / 1000.0 / (double)rl_frames_seen : 0.0,
            rl_wakes ? rl_wake_us / 1000.0 / (double)rl_wakes : 0.0, (unsigned long long)rl_wakes);
        memset(rl_gap_hist, 0, sizeof rl_gap_hist);
        rl_age_us = rl_wake_us = rl_wakes = 0;
        if (o < n) o += (size_t)snprintf(buf + o, n - o, " | GS draws merged into their neighbour %llu; not merged because"
                                         " vertices apart %llu, texture %llu, shader %llu, blend %llu, depth %llu, scissor %llu,"
                                         " UV clamp %llu, target %llu, other %llu", (unsigned long long)stat_gs_merged,
                                         (unsigned long long)stat_gs_break[0], (unsigned long long)stat_gs_break[1],
                                         (unsigned long long)stat_gs_break[2], (unsigned long long)stat_gs_break[3],
                                         (unsigned long long)stat_gs_break[4], (unsigned long long)stat_gs_break[5],
                                         (unsigned long long)stat_gs_break[6], (unsigned long long)stat_gs_break[7],
                                         (unsigned long long)stat_gs_break[8]);
        stat_gs_merged = 0;
        memset(stat_gs_break, 0, sizeof stat_gs_break);
        if (o < n) o += (size_t)snprintf(buf + o, n - o, " | renderer 1 ms samples while drawing, by segment after mark (running/held/waiting):");
        for (u32 k = 0; k < 16 && o < n; k++)
            if (rn_smp[k][0] + rn_smp[k][1] + rn_smp[k][2])
                o += (size_t)snprintf(buf + o, n - o, " %s %u/%u/%u", rn_ph_name[k], rn_smp[k][0], rn_smp[k][1], rn_smp[k][2]);
        memset(rn_smp, 0, sizeof rn_smp);
        if (o < n) o += (size_t)snprintf(buf + o, n - o, " | renderer gave way to the GS %llu ms", (unsigned long long)stat_defer_ms);
        stat_defer_ms = 0;
        if (o < n) o += (size_t)snprintf(buf + o, n - o, " | paletted (P8) uploads %llu", (unsigned long long)stat_uploads_p8);
        stat_uploads_p8 = 0;
        if (o < n) o += (size_t)snprintf(buf + o, n - o, " | texture decodes %llu (%.1f ms; %llu on core 0, %llu by the renderer), renderer waited for them %.1f ms",
                                         (unsigned long long)stat_tj_jobs, stat_tj_us / 1000.0,
                                         (unsigned long long)stat_tj_jobs2, (unsigned long long)stat_tj_stolen,
                                         stat_tj_waits_us / 1000.0);
        stat_tj_jobs = stat_tj_us = stat_tj_waits_us = stat_tj_jobs2 = stat_tj_stolen = 0;
        if (o < n) o += (size_t)snprintf(buf + o, n - o, " | circuit-1 overlays %llu | scenes cover %.0f%% of their targets",
                                         (unsigned long long)stat_overlays,
                                         stat_region_full_px ? 100.0 * (double)stat_region_px / (double)stat_region_full_px : 100.0);
        stat_region_px = stat_region_full_px = stat_region_scenes = 0;
        stat_overlays = 0;
        if (o < n) o += (size_t)snprintf(buf + o, n - o,
            " | draw calls <50us %llu (%.1f ms) <250us %llu (%.1f ms) <1ms %llu (%.1f ms) >=1ms %llu (%.1f ms);"
            " first draw after a target change %llu (%.1f ms) | %llu solid 3D draws sorted, %llu overlay passes grouped after them",
            (unsigned long long)dh_n[0], dh_us[0] / 1000.0 / (double)f, (unsigned long long)dh_n[1], dh_us[1] / 1000.0 / (double)f,
            (unsigned long long)dh_n[2], dh_us[2] / 1000.0 / (double)f, (unsigned long long)dh_n[3], dh_us[3] / 1000.0 / (double)f,
            (unsigned long long)dh_first_n, dh_first_us / 1000.0 / (double)f, (unsigned long long)stat_sorted,
            (unsigned long long)stat_overlays_moved);
        memset(dh_n, 0, sizeof dh_n); memset(dh_us, 0, sizeof dh_us);
        dh_first_us = dh_first_n = stat_sorted = stat_overlays_moved = 0;
        if (o < n) o += (size_t)snprintf(buf + o, n - o, " | mesh merge broken by:");
        for (int c = 0; c < BR_N && o < n; c++)
            if (stat_break[c]) o += (size_t)snprintf(buf + o, n - o, " %s %llu", bn[c], (unsigned long long)stat_break[c]);
        memset(tm_us, 0, sizeof tm_us);
        memset(stat_break, 0, sizeof stat_break);
    }
    tm_frames = 0;
}

void vita_gl_mem(char *buf, size_t n) {
    snprintf(buf, n, "gl pools free: CDRAM %.1f MB, main %.1f MB, phycont %.1f MB; "
             "textures on GPU %.1f MB, CPU store %.1f MB; snapshots %llu (%llu over budget), "
             "target binds %llu, texture deletes %llu, drawn %.1f Mpx (GS pixels, bbox estimate), "
             "depth-target swaps %llu, depth stamps %llu",
             stat_vgl_vram_kb / 1024.0, stat_vgl_ram_kb / 1024.0, stat_vgl_phy_kb / 1024.0,
             stat_gpu_tex_kb / 1024.0, (double)tex_bytes / 1048576.0,
             (unsigned long long)stat_snapshots, (unsigned long long)stat_snap_skips,
             (unsigned long long)stat_rt_binds, (unsigned long long)stat_tex_deletes,
             (double)stat_fill / 1e6, (unsigned long long)stat_zswaps,
             (unsigned long long)stat_zclears);
}
void vita_gl_interval(u64 *frames, u64 *draws, u64 *uploads, int *variants_out) {
    if (frames) *frames = stat_frames;
    if (draws) *draws = stat_draws;
    if (uploads) *uploads = stat_uploads;
    if (variants_out) *variants_out = nvariants;
}

int  ps2_vk_screenshot(const char *path) { (void)path; return -1; }
void ps2_vk_state_save(ps2_state_put put, void *ud) { (void)put; (void)ud; }
void ps2_vk_statecap(const char *dir) { (void)dir; }
