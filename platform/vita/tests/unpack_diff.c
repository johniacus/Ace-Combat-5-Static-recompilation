#include "vifx.inc"

/* Differential test: the word-by-word unpack against the whole-element fast
 * path, from the same starting state, over random data. */
static u32 rnd(void) { static u32 s = 12345; s ^= s << 13; s ^= s >> 17; s ^= s << 5; return s; }

static void setup(ps2_vif *v, u32 vn, u32 vl, u32 num, u32 align, u32 idx, u32 cl, u32 wl, u32 usn) {
    memset(v, 0, sizeof *v);
    v->cmd = 0x60u | (vn << 2) | vl;
    v->unpack_vn = vn; v->unpack_vl = vl; v->unpack_usn = usn;
    v->unpack_num = num; v->num = num;
    v->unpack_cl = cl; v->unpack_wl = wl;
    v->unpack_align = align; v->unpack_index = idx;
    v->unpack_base = 0;
    {   u32 bits = (32u >> vl) * (vn + 1u);
        v->wait = (u32)(((u64)num * bits + 31u) / 32u); }
}

int main(void) {
    static u8 m1[16384], m2[16384];
    int bad = 0;
    for (int iter = 0; iter < 200000 && bad < 10; iter++) {
        u32 fmt = rnd() % 4;
        u32 vn = fmt == 0 ? 3 : fmt == 1 ? 1 : 2, vl = fmt == 0 ? 2 : fmt == 3 ? 1 : 0;
        u32 num = 1 + rnd() % 40, align = rnd() % 4, idx = rnd() % 8;
        u32 wl = 1 + rnd() % 4, cl = wl + (rnd() % 2) * (rnd() % 3);
        u32 usn = rnd() & 1;
        ps2_vif a, b;
        ps2_vu va = { m1, sizeof m1 }, vb = { m2, sizeof m2 };
        u32 w[256];
        u32 lane0 = rnd() % 4;        /* where the data starts in its quadword */
        setup(&a, vn, vl, num, align, idx, cl, wl, usn);
        b = a;
        u32 n = a.wait;
        for (u32 i = 0; i < n + 4; i++) w[i] = rnd();
        memset(m1, 0xCD, sizeof m1); memset(m2, 0xCD, sizeof m2);
        /* word by word */
        for (u32 i = 0; i < n; i++) { a.wait--; vif_unpack_stream(&a, &va, w[i]); }
        /* fast path the way vif_quadword_at / the transfer loop call it:
         * try the fast path at each word, else one word slowly */
        {   u32 i = 0;
            int mode = rnd() % 2;   /* 0: per quadword (in-qw), 1: flat */
            while (i < n) {
                u32 took;
                u32 lane = (lane0 + i) & 3u;
                u32 avail = mode ? n - i : (4u - lane < n - i ? 4u - lane : n - i);
                /* the real callers pass the quadword's remaining words, which
                 * can run past the unpack data into the next VIFcode */
                if (!mode) avail = 4u - lane;
                took = vif_elem_flat(&b, &vb, w + i, avail);
                if (took) { i += took; continue; }
                b.wait--; vif_unpack_stream(&b, &vb, w[i]); i++;
            }
        }
        if (memcmp(m1, m2, sizeof m1) || a.unpack_index != b.unpack_index || a.unpack_written != b.unpack_written
            || a.unpack_num != b.unpack_num || a.nbits != b.nbits || a.bits != b.bits || a.elem_n != b.elem_n) {
            u32 k;
            for (k = 0; k < sizeof m1 && m1[k] == m2[k]; k++) ;
            printf("MISMATCH fmt V%u-%u num %u align %u idx %u cl %u wl %u usn %u lane0 %u: first diff byte %u (qw %u lane %u) slow %02X fast %02X | idx %u/%u wr %u/%u nbits %u/%u bits %llx/%llx\n",
                   vn + 1, 32 >> vl, num, align, idx, cl, wl, usn, lane0, k, k / 16, (k % 16) / 4, m1[k], m2[k],
                   a.unpack_index, b.unpack_index, a.unpack_written, b.unpack_written, a.nbits, b.nbits,
                   (unsigned long long)a.bits, (unsigned long long)b.bits);
            bad++;
        }
    }
    printf("done, %d mismatches, fast words %llu\n", bad, (unsigned long long)ps2_vc_elem_words);
    return 0;
}
