#include "ps2_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
void ps2_gs_write_reg(u32 reg, u64 val);
void ps2_gs_hwreg_stream(const void *data, u32 qwc);
void ps2_gs_init(void);
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec / 1e9; }
/* one host->local transfer: CT32 w x h at (x,y) in buffer dbp/dbw */
static void upload(u32 dbp, u32 dbw, u32 psm, u32 x, u32 y, u32 w, u32 h, const u8 *data) {
    ps2_gs_write_reg(0x50, ((u64)(dbp / 256u) << 32) | ((u64)(dbw / 64u) << 48) | ((u64)psm << 56)); /* BITBLTBUF */
    ps2_gs_write_reg(0x51, ((u64)x << 32) | ((u64)y << 48));          /* TRXPOS */
    ps2_gs_write_reg(0x52, (u64)w | ((u64)h << 32));                  /* TRXREG */
    ps2_gs_write_reg(0x53, 0);                                        /* TRXDIR host->local */
    u32 bpp = psm == 0 ? 32 : psm == 0x13 ? 8 : psm == 0x14 ? 4 : 16;
    u32 qwc = (w * h * bpp / 8 + 15) / 16;
    ps2_gs_hwreg_stream(data, qwc);
}
int main(int argc, char **argv) {
    u32 pieces = 360, w = 32, h = 16, psm = 0, frames = 20;
    int change = argc > 1 ? atoi(argv[1]) : 1;
    if (argc > 2) psm = (u32)strtoul(argv[2], 0, 0);
    if (argc > 3) { w = (u32)atoi(argv[3]); h = (u32)atoi(argv[4]); pieces = (u32)atoi(argv[5]); } if (argc > 6) frames = (u32)atoi(argv[6]);
    static u8 data[4 * 1024 * 1024];
    for (u32 i = 0; i < sizeof data; i++) data[i] = (u8)rand();
    ps2_gs_init();
    double t0 = now();
    for (u32 f = 0; f < frames; f++) {
        for (u32 p = 0; p < pieces; p++) {
            u32 dbp = 0x100000u + (p % 64u) * 0x8000u;
            if (change == 1) data[(p * 131u + f) % 2048u]++;
            if (change == 2) { u32 *d32 = (u32 *)(data + (p % 16u) * 4096u); for (u32 k = 0; k < w * h; k += 7) d32[k % 1024u] += f + 1; }
            upload(dbp, 64, psm, (p / 64u) * 0, 0, w, h, data + (p % 16u) * 4096u);
        }
    }
    double dt = now() - t0;
    printf("psm %u %ux%u x%u pieces, change=%d: %.2f ms per frame\n", psm, w, h, pieces, change, dt * 1000.0 / frames);
    return 0;
}
