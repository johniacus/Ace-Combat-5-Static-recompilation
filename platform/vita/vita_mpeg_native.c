/* Native (NEON) versions of the hottest loops in the game's MPEG library
 * (libmpeg's motion-compensation helpers, guest code in ps2_code_0033.c).
 *
 * The recompiled versions run one PS2 MMI instruction at a time through the
 * guest register file and guest-memory helpers; during movies they took
 * about a third of the EE thread. Each function here does exactly what the
 * guest code does -- same memory writes, same final register values -- and
 * returns 0 without touching anything if an address is not plain memory,
 * in which case the recompiled code runs as before.
 *
 * Hooked from generated/ps2_code_0033.c (search for "vita_mc_"). If the
 * code is regenerated, those few lines need adding back. */
#include "ps2_runtime.h"
#include "ps2_vu0.h"
#include <arm_neon.h>
#include <string.h>

u64 vita_mc_calls[4], vita_mc_fallbacks[4];

static inline u8 *gp(u32 a) {
    u8 *p = ps2_pt[a >> PS2_PAGE_BITS];
    return p ? p + (a & PS2_PAGE_MASK) : NULL;
}
/* host pointer to the aligned quadword holding a (as ps2_r128 reads it) */
static inline u8 *gq(u32 a) { return gp(a & ~15u); }
/* host pointer for n bytes at a, only if they sit in one mapped page */
static inline u8 *gn(u32 a, u32 n) {
    if (((a & PS2_PAGE_MASK) + n) > (1u << PS2_PAGE_BITS)) return NULL;
    return gp(a);
}
static inline void set32(ps2_ctx *ctx, int r, u32 v) { ctx->r[r].sd[0] = (s64)(s32)v; }
static inline u32 lw(u32 a) { u8 *p = gn(a, 4); u32 v = 0; if (p) memcpy(&v, p, 4); return v; }

static inline void loop_ticks(u32 n) { while (n--) PS2_LOOP(); }

/* ------------------------------------------------------------ 00342148 */
/* Rows of 16 source bytes (16-byte unaligned, via QFSRV) widened to
 * halfwords into a 32-byte destination row; a second pass if *(r4+0xC). */
int vita_mc_00342148(ps2_ctx *ctx) {
    const u32 r4 = ctx->r[4].uw[0];
    u8 *pa = gn(r4, 0x1C);
    u32 r5, r6, r7, r14, r13, r12, r11, n1, n2, pass, total = 0;
    ps2_reg128 R8, R9;
    u64 last_hi = 0;
    if (!pa) { vita_mc_fallbacks[0]++; return 0; }
    r14 = lw(r4); r13 = lw(r4 + 4); r7 = lw(r4 + 8);
    n2 = lw(r4 + 0xC); r12 = lw(r4 + 0x10);
    r5 = lw(r4 + 0x14); r6 = lw(r4 + 0x18);
    r11 = r12 << 1;
    n1 = (s32)r7 > 1 ? r7 : 1u;
    /* check every address first: nothing is written unless all are memory */
    {
        u32 a5 = r5, a6 = r6, a14 = r14, p, k, n = n1;
        for (p = 0; p < 2; p++) {
            for (k = 0; k < n; k++) {
                if (!gq(a5) || !gq(a6) || !gq(a14) || !gq(a14 + 16u)) {
                    vita_mc_fallbacks[0]++;
                    return 0;
                }
                a5 += r12; a6 += r12; a14 += r11;
            }
            a5 += 0x80u; a6 += 0x80u;
            if (!n2 || p) break;
            n = (s32)n2 > 1 ? n2 : 1u;
        }
    }
    {
        const u32 off = r13 & 15u;
        u32 n = n1;
        for (pass = 0; pass < 2; pass++) {
            u32 k;
            for (k = 0; k < n; k++) {
                u8 tmp[32];
                uint8x16_t v;
                memcpy(tmp, gq(r5), 16);
                memcpy(tmp + 16, gq(r6), 16);
                v = vld1q_u8(tmp + off);
                memcpy(&last_hi, tmp + off + 8, 8);
                {
                    uint16x8_t lo = vmovl_u8(vget_low_u8(v)), hi = vmovl_u8(vget_high_u8(v));
                    vst1q_u16((u16 *)(void *)R8.uh, lo);
                    vst1q_u16((u16 *)(void *)R9.uh, hi);
                }
                memcpy(gq(r14), &R8, 16);
                memcpy(gq(r14 + 16u), &R9, 16);
                r5 += r12; r6 += r12; r14 += r11;
                total++;
            }
            r5 += 0x80u; r6 += 0x80u;
            if (!n2 || pass) break;
            n = (s32)n2 > 1 ? n2 : 1u;
        }
    }
    /* final register state, as the guest loop leaves it */
    ctx->r[8] = R8; ctx->r[9] = R9;
    set32(ctx, 5, r5); set32(ctx, 6, r6); set32(ctx, 7, n2);
    set32(ctx, 14, r14); set32(ctx, 13, r13); set32(ctx, 12, r12);
    set32(ctx, 11, r11);
    /* QFSRV left r10 = the last row's 16 bytes; the closing AND rewrites
     * only its low doubleword (to 0) */
    ctx->r[10].ud[0] = 0; ctx->r[10].ud[1] = last_hi;
    ctx->r[15].ud[0] = 0;
    ctx->sa = (r13 & 0xFu) << 3;
    loop_ticks(total);
    ps2_vu0_advance(ctx, 5u);
    vita_mc_calls[0]++;
    return 1;
}

/* ------------------------------------------------------------ 003421C0 */
/* 8-byte rows (two doubleword loads, byte-rotated by SA) widened to 16
 * bytes; done twice (luma then chroma half), each with an optional second
 * pass. */
int vita_mc_003421C0(ps2_ctx *ctx) {
    const u32 r4 = ctx->r[4].uw[0];
    u32 b14, b18, b0, r13, r12, r11, n8, nC, off, total = 0;
    u32 r5, r6, r14, outer;
    ps2_reg128 R8 = ctx->r[8], R9 = ctx->r[9];
    if (!gn(r4, 0x1C)) { vita_mc_fallbacks[1]++; return 0; }
    b0 = lw(r4); r13 = lw(r4 + 4); n8 = lw(r4 + 8); nC = lw(r4 + 0xC);
    r12 = lw(r4 + 0x10); b14 = lw(r4 + 0x14); b18 = lw(r4 + 0x18);
    r11 = r12 << 1;
    off = r13 & 15u;
    /* validate */
    r5 = b14; r6 = b18; r14 = b0;
    for (outer = 0; outer < 2; outer++) {
        u32 pass, n = (s32)n8 > 1 ? n8 : 1u;
        for (pass = 0; pass < 2; pass++) {
            u32 k;
            for (k = 0; k < n; k++) {
                if (!gn(r5, 8) || !gn(r6, 8) || !gq(r14)) { vita_mc_fallbacks[1]++; return 0; }
                r5 += r12; r6 += r12; r14 += r11;
            }
            r5 += 0x140u; r6 += 0x140u;
            if (!nC || pass) break;
            n = (s32)nC > 1 ? nC : 1u;
        }
        r5 = b14 + 0x40u; r6 = b18 + 0x40u; r14 = b0 + 0x80u;
    }
    /* run */
    r5 = b14; r6 = b18; r14 = b0;
    for (outer = 0; outer < 2; outer++) {
        u32 pass, n = (s32)n8 > 1 ? n8 : 1u;
        for (pass = 0; pass < 2; pass++) {
            u32 k;
            for (k = 0; k < n; k++) {
                u8 cat[16], rot[8];
                u32 i;
                memcpy(cat, gn(r5, 8), 8);
                memcpy(cat + 8, gn(r6, 8), 8);
                memcpy(&R8.ud[0], cat, 8);          /* PCPYLD */
                memcpy(&R8.ud[1], cat + 8, 8);
                for (i = 0; i < 8; i++) rot[i] = cat[(off + i) & 15u];   /* QFSRV r8,r8 */
                {   /* r9 = whole rotated quadword; r8 = its low 8 bytes widened */
                    for (i = 0; i < 16; i++) R9.ub[i] = cat[(off + i) & 15u];
                    vst1q_u16((u16 *)(void *)R8.uh, vmovl_u8(vld1_u8(rot)));
                }
                memcpy(gq(r14), &R8, 16);
                r5 += r12; r6 += r12; r14 += r11;
                total++;
            }
            r5 += 0x140u; r6 += 0x140u;
            if (!nC || pass) break;
            n = (s32)nC > 1 ? nC : 1u;
        }
        r5 = b14 + 0x40u; r6 = b18 + 0x40u; r14 = b0 + 0x80u;
    }
    ctx->r[8] = R8; ctx->r[9] = R9;
    set32(ctx, 5, r5); set32(ctx, 6, r6); set32(ctx, 7, nC);
    set32(ctx, 14, r14); set32(ctx, 13, r13); set32(ctx, 12, r12);
    set32(ctx, 11, r11);
    ctx->r[10].ud[0] = 0; ctx->r[15].ud[0] = 0; ctx->r[24].ud[0] = 0;
    ctx->sa = (r13 & 0xFu) << 3;
    loop_ticks(total);
    ps2_vu0_advance(ctx, 5u);
    vita_mc_calls[1]++;
    return 1;
}

/* ---------------------------------------- 0034203C tail loops (L_342E38) */
/* 24 rows: dst bytes = clamp(src5 + src6, 0, limit) (halfword add,
 * signed min/max, then the low bytes packed). */
int vita_mc_addclamp(ps2_ctx *ctx) {
    u32 r4 = ctx->r[4].uw[0], r5 = ctx->r[5].uw[0], r6 = ctx->r[6].uw[0], k;
    u8 *lim = gq(0x00342EE0u);
    int16x8_t L, Z = vdupq_n_s16(0);
    ps2_reg128 R8, R9, R10, R13, R2;
    if (!lim) { vita_mc_fallbacks[2]++; return 0; }
    for (k = 0; k < 24; k++)
        if (!gq(r5 + 32u * k) || !gq(r5 + 32u * k + 16u) || !gq(r6 + 32u * k)
            || !gq(r6 + 32u * k + 16u) || !gq(r4 + 16u * k)) { vita_mc_fallbacks[2]++; return 0; }
    memcpy(&ctx->r[11], lim, 16);
    L = vld1q_s16((const s16 *)(const void *)ctx->r[11].sh);
    for (k = 0; k < 24; k++) {
        int16x8_t a = vld1q_s16((const s16 *)(const void *)gq(r5));
        int16x8_t b = vld1q_s16((const s16 *)(const void *)gq(r5 + 16u));
        int16x8_t c = vld1q_s16((const s16 *)(const void *)gq(r6));
        int16x8_t d = vld1q_s16((const s16 *)(const void *)gq(r6 + 16u));
        int16x8_t x = vmaxq_s16(vminq_s16(vaddq_s16(a, c), L), Z);
        int16x8_t y = vmaxq_s16(vminq_s16(vaddq_s16(b, d), L), Z);
        uint8x16_t o = vcombine_u8(vmovn_u16(vreinterpretq_u16_s16(x)),
                                   vmovn_u16(vreinterpretq_u16_s16(y)));
        vst1q_u8(gq(r4), o);
        if (k == 23) {
            vst1q_s16(R8.sh, x); vst1q_s16(R9.sh, y); vst1q_u8(R10.ub, o);
            vst1q_s16(R13.sh, c); vst1q_s16(R2.sh, d);
        }
        r4 += 16u; r5 += 32u; r6 += 32u;
    }
    ctx->r[8] = R8; ctx->r[9] = R9; ctx->r[10] = R10; ctx->r[13] = R13; ctx->r[2] = R2;
    set32(ctx, 4, r4); set32(ctx, 5, r5); set32(ctx, 6, r6);
    ctx->r[12].ud[0] = 0;
    loop_ticks(23);
    ps2_vu0_advance(ctx, 5u);
    vita_mc_calls[2]++;
    return 1;
}

/* ---------------------------------------- 0034203C tail loops (L_342E98) */
/* 24 rows: dst bytes = clamp(src5, 0, limit). */
int vita_mc_clamp(ps2_ctx *ctx) {
    u32 r4 = ctx->r[4].uw[0], r5 = ctx->r[5].uw[0], k;
    u8 *lim = gq(0x00342EE0u);
    int16x8_t L, Z = vdupq_n_s16(0);
    ps2_reg128 R8, R9, R10;
    if (!lim) { vita_mc_fallbacks[3]++; return 0; }
    for (k = 0; k < 24; k++)
        if (!gq(r5 + 32u * k) || !gq(r5 + 32u * k + 16u) || !gq(r4 + 16u * k)) {
            vita_mc_fallbacks[3]++;
            return 0;
        }
    memcpy(&ctx->r[11], lim, 16);
    L = vld1q_s16((const s16 *)(const void *)ctx->r[11].sh);
    for (k = 0; k < 24; k++) {
        int16x8_t a = vld1q_s16((const s16 *)(const void *)gq(r5));
        int16x8_t b = vld1q_s16((const s16 *)(const void *)gq(r5 + 16u));
        int16x8_t x = vmaxq_s16(vminq_s16(a, L), Z);
        int16x8_t y = vmaxq_s16(vminq_s16(b, L), Z);
        uint8x16_t o = vcombine_u8(vmovn_u16(vreinterpretq_u16_s16(x)),
                                   vmovn_u16(vreinterpretq_u16_s16(y)));
        vst1q_u8(gq(r4), o);
        if (k == 23) { vst1q_s16(R8.sh, x); vst1q_s16(R9.sh, y); vst1q_u8(R10.ub, o); }
        r4 += 16u; r5 += 32u;
    }
    ctx->r[8] = R8; ctx->r[9] = R9; ctx->r[10] = R10;
    set32(ctx, 4, r4); set32(ctx, 5, r5);
    ctx->r[12].ud[0] = 0;
    loop_ticks(23);
    ps2_vu0_advance(ctx, 5u);
    vita_mc_calls[3]++;
    return 1;
}
