/* NEON versions of the IPU's two hot loops, for the Vita's Cortex-A9.
 * Both are bit-exact with the scalar code in ps2_ipu.c (checked against it
 * over millions of random inputs under qemu-arm).
 *
 * idct_8x8_neon: the same Chen-Wang integer IDCT as idct_8x8_i32, run on
 * eight rows (then eight columns) at once in 32-bit lanes. The scalar
 * version's "all AC zero" shortcuts give the same numbers as the full
 * formula, so the vector version simply always runs the full formula.
 * Only valid where idct_8x8_i32 is (every AC coefficient below 256). */
#include <arm_neon.h>

typedef struct { int32x4_t l, h; } ipu_v8;

static inline void ipu_tr4(int32x4_t a, int32x4_t b, int32x4_t c, int32x4_t d,
                           int32x4_t *o0, int32x4_t *o1, int32x4_t *o2,
                           int32x4_t *o3) {
    int32x4x2_t ab = vtrnq_s32(a, b), cd = vtrnq_s32(c, d);
    *o0 = vcombine_s32(vget_low_s32(ab.val[0]),  vget_low_s32(cd.val[0]));
    *o1 = vcombine_s32(vget_low_s32(ab.val[1]),  vget_low_s32(cd.val[1]));
    *o2 = vcombine_s32(vget_high_s32(ab.val[0]), vget_high_s32(cd.val[0]));
    *o3 = vcombine_s32(vget_high_s32(ab.val[1]), vget_high_s32(cd.val[1]));
}

/* o[j] lane i = in[i] lane j, for an 8x8 matrix held as 8 x (l,h). */
static inline void ipu_tr8(const ipu_v8 *in, ipu_v8 *o) {
    ipu_tr4(in[0].l, in[1].l, in[2].l, in[3].l, &o[0].l, &o[1].l, &o[2].l, &o[3].l);
    ipu_tr4(in[4].l, in[5].l, in[6].l, in[7].l, &o[0].h, &o[1].h, &o[2].h, &o[3].h);
    ipu_tr4(in[0].h, in[1].h, in[2].h, in[3].h, &o[4].l, &o[5].l, &o[6].l, &o[7].l);
    ipu_tr4(in[4].h, in[5].h, in[6].h, in[7].h, &o[4].h, &o[5].h, &o[6].h, &o[7].h);
}

#define VM(a, k) vmulq_n_s32((a), (k))
static inline void ipu_idct_row4(const int32x4_t *X, int32x4_t *o) {
    int32x4_t x0, x1, x2, x3, x4, x5, x6, x7, x8;
    x0 = vaddq_s32(vshlq_n_s32(X[0], 11), vdupq_n_s32(128));
    x1 = vshlq_n_s32(X[4], 11); x2 = X[6]; x3 = X[2]; x4 = X[1];
    x5 = X[7]; x6 = X[5]; x7 = X[3];
    x8 = VM(vaddq_s32(x4, x5), IW7);
    x4 = vaddq_s32(x8, VM(x4, IW1 - IW7)); x5 = vsubq_s32(x8, VM(x5, IW1 + IW7));
    x8 = VM(vaddq_s32(x6, x7), IW3);
    x6 = vsubq_s32(x8, VM(x6, IW3 - IW5)); x7 = vsubq_s32(x8, VM(x7, IW3 + IW5));
    x8 = vaddq_s32(x0, x1); x0 = vsubq_s32(x0, x1);
    x1 = VM(vaddq_s32(x3, x2), IW6);
    x2 = vsubq_s32(x1, VM(x2, IW2 + IW6)); x3 = vaddq_s32(x1, VM(x3, IW2 - IW6));
    x1 = vaddq_s32(x4, x6); x4 = vsubq_s32(x4, x6);
    x6 = vaddq_s32(x5, x7); x5 = vsubq_s32(x5, x7);
    x7 = vaddq_s32(x8, x3); x8 = vsubq_s32(x8, x3);
    x3 = vaddq_s32(x0, x2); x0 = vsubq_s32(x0, x2);
    x2 = vshrq_n_s32(vaddq_s32(VM(vaddq_s32(x4, x5), 181), vdupq_n_s32(128)), 8);
    x4 = vshrq_n_s32(vaddq_s32(VM(vsubq_s32(x4, x5), 181), vdupq_n_s32(128)), 8);
    o[0] = vshrq_n_s32(vaddq_s32(x7, x1), 8); o[1] = vshrq_n_s32(vaddq_s32(x3, x2), 8);
    o[2] = vshrq_n_s32(vaddq_s32(x0, x4), 8); o[3] = vshrq_n_s32(vaddq_s32(x8, x6), 8);
    o[4] = vshrq_n_s32(vsubq_s32(x8, x6), 8); o[5] = vshrq_n_s32(vsubq_s32(x0, x4), 8);
    o[6] = vshrq_n_s32(vsubq_s32(x3, x2), 8); o[7] = vshrq_n_s32(vsubq_s32(x7, x1), 8);
}

static inline void ipu_idct_col4(const int32x4_t *X, int32x4_t *o) {
    int32x4_t x0, x1, x2, x3, x4, x5, x6, x7, x8;
    const int32x4_t four = vdupq_n_s32(4);
    x0 = vaddq_s32(vshlq_n_s32(X[0], 8), vdupq_n_s32(8192));
    x1 = vshlq_n_s32(X[4], 8); x2 = X[6]; x3 = X[2]; x4 = X[1];
    x5 = X[7]; x6 = X[5]; x7 = X[3];
    x8 = vaddq_s32(VM(vaddq_s32(x4, x5), IW7), four);
    x4 = vshrq_n_s32(vaddq_s32(x8, VM(x4, IW1 - IW7)), 3);
    x5 = vshrq_n_s32(vsubq_s32(x8, VM(x5, IW1 + IW7)), 3);
    x8 = vaddq_s32(VM(vaddq_s32(x6, x7), IW3), four);
    x6 = vshrq_n_s32(vsubq_s32(x8, VM(x6, IW3 - IW5)), 3);
    x7 = vshrq_n_s32(vsubq_s32(x8, VM(x7, IW3 + IW5)), 3);
    x8 = vaddq_s32(x0, x1); x0 = vsubq_s32(x0, x1);
    x1 = vaddq_s32(VM(vaddq_s32(x3, x2), IW6), four);
    x2 = vshrq_n_s32(vsubq_s32(x1, VM(x2, IW2 + IW6)), 3);
    x3 = vshrq_n_s32(vaddq_s32(x1, VM(x3, IW2 - IW6)), 3);
    x1 = vaddq_s32(x4, x6); x4 = vsubq_s32(x4, x6);
    x6 = vaddq_s32(x5, x7); x5 = vsubq_s32(x5, x7);
    x7 = vaddq_s32(x8, x3); x8 = vsubq_s32(x8, x3);
    x3 = vaddq_s32(x0, x2); x0 = vsubq_s32(x0, x2);
    x2 = vshrq_n_s32(vaddq_s32(VM(vaddq_s32(x4, x5), 181), vdupq_n_s32(128)), 8);
    x4 = vshrq_n_s32(vaddq_s32(VM(vsubq_s32(x4, x5), 181), vdupq_n_s32(128)), 8);
    o[0] = vshrq_n_s32(vaddq_s32(x7, x1), 14); o[1] = vshrq_n_s32(vaddq_s32(x3, x2), 14);
    o[2] = vshrq_n_s32(vaddq_s32(x0, x4), 14); o[3] = vshrq_n_s32(vaddq_s32(x8, x6), 14);
    o[4] = vshrq_n_s32(vsubq_s32(x8, x6), 14); o[5] = vshrq_n_s32(vsubq_s32(x0, x4), 14);
    o[6] = vshrq_n_s32(vsubq_s32(x3, x2), 14); o[7] = vshrq_n_s32(vsubq_s32(x7, x1), 14);
}
#undef VM

static void idct_8x8_neon(const s16 *in, s16 *out, int lo, int hi) {
    ipu_v8 r[8], c[8];
    int32x4_t a[8], b[8];
    const int32x4_t vlo = vdupq_n_s32(lo), vhi = vdupq_n_s32(hi);
    int i;
    for (i = 0; i < 8; i++) {
        int16x8_t v = vld1q_s16(in + i * 8);
        r[i].l = vmovl_s16(vget_low_s16(v));
        r[i].h = vmovl_s16(vget_high_s16(v));
    }
    ipu_tr8(r, c);                       /* c[j] = coefficient j of each row */
    for (i = 0; i < 8; i++) a[i] = c[i].l;
    ipu_idct_row4(a, b);
    for (i = 0; i < 8; i++) { r[i].l = b[i]; a[i] = c[i].h; }
    ipu_idct_row4(a, b);
    for (i = 0; i < 8; i++) r[i].h = b[i];   /* r[k] = output k of each row */
    ipu_tr8(r, c);                       /* c[j] = row j of the row pass */
    for (i = 0; i < 8; i++) a[i] = c[i].l;
    ipu_idct_col4(a, b);
    for (i = 0; i < 8; i++) { r[i].l = b[i]; a[i] = c[i].h; }
    ipu_idct_col4(a, b);
    for (i = 0; i < 8; i++) {
        int32x4_t l = vminq_s32(vmaxq_s32(r[i].l, vlo), vhi);
        int32x4_t h = vminq_s32(vmaxq_s32(b[i], vlo), vhi);
        vst1q_s16(out + i * 8, vcombine_s16(vmovn_s32(l), vmovn_s32(h)));
    }
}

/* csc_rgb_neon: csc_pixel() for a whole 16x16 macroblock, four pixels at a
 * time. Y/Cb/Cr are the 0..255 samples (as s16) the scalar code takes. */
static inline uint32x4_t ipu_csc4(int32x4_t Y, int32x4_t Cb, int32x4_t Cr,
                                  int32x4_t th0, int32x4_t th1, int sgn) {
    const int32x4_t one = vdupq_n_s32(1), zero = vdupq_n_s32(0);
    const int32x4_t c255 = vdupq_n_s32(255);
    int32x4_t Cr1 = vsubq_s32(Cr, vdupq_n_s32(128));
    int32x4_t Cb1 = vsubq_s32(Cb, vdupq_n_s32(128));
    int32x4_t Cr4 = vshrq_n_s32(vmulq_n_s32(Cr1, 0x0cc), 6);
    int32x4_t Cr5 = vshrq_n_s32(vmulq_n_s32(Cr1, 0x068), 6);
    int32x4_t Cb4 = vshrq_n_s32(vmulq_n_s32(Cb1, 0x032), 6);
    int32x4_t Cb5 = vshrq_n_s32(vmulq_n_s32(Cb1, 0x102), 6);
    int32x4_t Y3  = vshrq_n_s32(vmulq_n_s32(vsubq_s32(Y, vdupq_n_s32(16)), 0x095), 6);
    int32x4_t r = vshrq_n_s32(vaddq_s32(vaddq_s32(Y3, Cr4), one), 1);
    int32x4_t g = vshrq_n_s32(vaddq_s32(vsubq_s32(vsubq_s32(Y3, Cb4), Cr5), one), 1);
    int32x4_t b = vshrq_n_s32(vaddq_s32(vaddq_s32(Y3, Cb5), one), 1);
    uint32x4_t m0, m1, a, p;
    r = vminq_s32(vmaxq_s32(r, zero), c255);
    g = vminq_s32(vmaxq_s32(g, zero), c255);
    b = vminq_s32(vmaxq_s32(b, zero), c255);
    m0 = vandq_u32(vandq_u32(vcltq_s32(r, th0), vcltq_s32(g, th0)), vcltq_s32(b, th0));
    m1 = vandq_u32(vandq_u32(vcltq_s32(r, th1), vcltq_s32(g, th1)), vcltq_s32(b, th1));
    a = vbslq_u32(m1, vdupq_n_u32(0x40000000u), vdupq_n_u32(0x80000000u));
    p = vorrq_u32(vorrq_u32(vreinterpretq_u32_s32(r),
                            vshlq_n_u32(vreinterpretq_u32_s32(g), 8)),
                  vshlq_n_u32(vreinterpretq_u32_s32(b), 16));
    p = vbicq_u32(vorrq_u32(p, a), m0);     /* below th0: black, alpha 0 */
    if (sgn) p = veorq_u32(p, vdupq_n_u32(0x00808080u));
    return p;
}

static void csc_rgb_neon(const s16 *y, const s16 *cb, const s16 *cr,
                         u32 *rgb, int sgn) {
    const int32x4_t th0 = vdupq_n_s32((int)ipu_th0), th1 = vdupq_n_s32((int)ipu_th1);
    int j, k;
    for (j = 0; j < 16; j++) {
        int16x8_t cbr = vld1q_s16(cb + (j >> 1) * 8);
        int16x8_t crr = vld1q_s16(cr + (j >> 1) * 8);
        int16x8x2_t cbz = vzipq_s16(cbr, cbr), crz = vzipq_s16(crr, crr);
        for (k = 0; k < 2; k++) {
            int16x8_t yv = vld1q_s16(y + j * 16 + k * 8);
            int16x8_t cbv = cbz.val[k], crv = crz.val[k];
            vst1q_u32(rgb + j * 16 + k * 8,
                      ipu_csc4(vmovl_s16(vget_low_s16(yv)),
                               vmovl_s16(vget_low_s16(cbv)),
                               vmovl_s16(vget_low_s16(crv)), th0, th1, sgn));
            vst1q_u32(rgb + j * 16 + k * 8 + 4,
                      ipu_csc4(vmovl_s16(vget_high_s16(yv)),
                               vmovl_s16(vget_high_s16(cbv)),
                               vmovl_s16(vget_high_s16(crv)), th0, th1, sgn));
        }
    }
}

/* 384 bytes of Y/Cb/Cr samples (the CSC command's input) to s16. */
static void csc_unpack_neon(const u8 *p, s16 *y, s16 *cb, s16 *cr) {
    int i;
    for (i = 0; i < 256; i += 16) {
        uint8x16_t v = vld1q_u8(p + i);
        vst1q_s16(y + i,     vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(v))));
        vst1q_s16(y + i + 8, vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(v))));
    }
    for (i = 0; i < 64; i += 16) {
        uint8x16_t v = vld1q_u8(p + 256 + i), w = vld1q_u8(p + 320 + i);
        vst1q_s16(cb + i,     vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(v))));
        vst1q_s16(cb + i + 8, vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(v))));
        vst1q_s16(cr + i,     vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(w))));
        vst1q_s16(cr + i + 8, vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(w))));
    }
}

/* csc_rgb_neon16: the same conversion in 16-bit lanes, eight pixels at a
 * time, with saturating narrows doing the 0..255 clamp and an interleaved
 * store building the RGBA words. Products that can leave the 16-bit range
 * (149*(Y-16), 258*(Cb-128)) go through widening multiplies. */
static inline void ipu_csc8(int16x8_t Y, int16x8_t Cb, int16x8_t Cr,
                            uint16x8_t th0, uint16x8_t th1, int sgn, u8 *dst) {
    int16x8_t Cr1 = vsubq_s16(Cr, vdupq_n_s16(128));
    int16x8_t Cb1 = vsubq_s16(Cb, vdupq_n_s16(128));
    int16x8_t Ym = vsubq_s16(Y, vdupq_n_s16(16));
    int16x8_t Cr4 = vshrq_n_s16(vmulq_n_s16(Cr1, 0x0cc), 6);
    int16x8_t Cr5 = vshrq_n_s16(vmulq_n_s16(Cr1, 0x068), 6);
    int16x8_t Cb4 = vshrq_n_s16(vmulq_n_s16(Cb1, 0x032), 6);
    int16x8_t Cb5 = vcombine_s16(
        vshrn_n_s32(vmull_n_s16(vget_low_s16(Cb1), 0x102), 6),
        vshrn_n_s32(vmull_n_s16(vget_high_s16(Cb1), 0x102), 6));
    int16x8_t Y3 = vcombine_s16(
        vshrn_n_s32(vmull_n_s16(vget_low_s16(Ym), 0x095), 6),
        vshrn_n_s32(vmull_n_s16(vget_high_s16(Ym), 0x095), 6));
    const int16x8_t one = vdupq_n_s16(1);
    int16x8_t r = vshrq_n_s16(vaddq_s16(vaddq_s16(Y3, Cr4), one), 1);
    int16x8_t g = vshrq_n_s16(vaddq_s16(vsubq_s16(vsubq_s16(Y3, Cb4), Cr5), one), 1);
    int16x8_t b = vshrq_n_s16(vaddq_s16(vaddq_s16(Y3, Cb5), one), 1);
    uint8x8x4_t o;
    uint16x8_t r16, g16, b16, m0, m1;
    uint8x8_t m0b;
    o.val[0] = vqmovun_s16(r);
    o.val[1] = vqmovun_s16(g);
    o.val[2] = vqmovun_s16(b);
    r16 = vmovl_u8(o.val[0]); g16 = vmovl_u8(o.val[1]); b16 = vmovl_u8(o.val[2]);
    m0 = vandq_u16(vandq_u16(vcltq_u16(r16, th0), vcltq_u16(g16, th0)), vcltq_u16(b16, th0));
    m1 = vandq_u16(vandq_u16(vcltq_u16(r16, th1), vcltq_u16(g16, th1)), vcltq_u16(b16, th1));
    m0b = vmovn_u16(m0);
    o.val[3] = vbic_u8(vbsl_u8(vmovn_u16(m1), vdup_n_u8(0x40), vdup_n_u8(0x80)), m0b);
    o.val[0] = vbic_u8(o.val[0], m0b);
    o.val[1] = vbic_u8(o.val[1], m0b);
    o.val[2] = vbic_u8(o.val[2], m0b);
    if (sgn) {
        const uint8x8_t x = vdup_n_u8(0x80);
        o.val[0] = veor_u8(o.val[0], x);
        o.val[1] = veor_u8(o.val[1], x);
        o.val[2] = veor_u8(o.val[2], x);
    }
    vst4_u8(dst, o);
}

static void csc_rgb_neon16(const s16 *y, const s16 *cb, const s16 *cr,
                           u32 *rgb, int sgn) {
    const uint16x8_t th0 = vdupq_n_u16((u16)(ipu_th0 > 0x1FFu ? 0x1FFu : ipu_th0));
    const uint16x8_t th1 = vdupq_n_u16((u16)(ipu_th1 > 0x1FFu ? 0x1FFu : ipu_th1));
    int j;
    for (j = 0; j < 16; j++) {
        int16x8_t cbr = vld1q_s16(cb + (j >> 1) * 8);
        int16x8_t crr = vld1q_s16(cr + (j >> 1) * 8);
        int16x8x2_t cbz = vzipq_s16(cbr, cbr), crz = vzipq_s16(crr, crr);
        ipu_csc8(vld1q_s16(y + j * 16), cbz.val[0], crz.val[0], th0, th1, sgn,
                 (u8 *)(rgb + j * 16));
        ipu_csc8(vld1q_s16(y + j * 16 + 8), cbz.val[1], crz.val[1], th0, th1, sgn,
                 (u8 *)(rgb + j * 16 + 8));
    }
}
