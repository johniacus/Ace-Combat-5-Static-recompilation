#include "ps2_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
void ps2_gs_write_reg(u32 reg, u64 val);
void ps2_gs_hwreg_stream(const void *data, u32 qwc);
void ps2_gs_init(void);
u8 *ps2_gs_vram_ptr(u32 *size);
int main(void) {
    static u8 data[1 << 20];
    srand(7);
    for (u32 i = 0; i < sizeof data; i++) data[i] = (u8)rand();
    ps2_gs_init();
    for (int t = 0; t < 3000; t++) {
        u32 psm = (u32[]){0, 0, 0, 0x13, 0x14, 2}[rand() % 6];
        u32 dbw = 64u * (1 + rand() % 8), w = 1 + rand() % 200, h = 1 + rand() % 64;
        u32 x = rand() % 64, y = rand() % 64, dbp = (rand() % 2048) * 256u;
        u32 bpp = psm == 0 ? 32 : psm == 0x13 ? 8 : psm == 0x14 ? 4 : 16;
        ps2_gs_write_reg(0x50, ((u64)(dbp / 256u) << 32) | ((u64)(dbw / 64u) << 48) | ((u64)psm << 56));
        ps2_gs_write_reg(0x51, ((u64)x << 32) | ((u64)y << 48));
        ps2_gs_write_reg(0x52, (u64)w | ((u64)h << 32));
        ps2_gs_write_reg(0x53, 0);
        u32 qwc = (w * h * bpp / 8 + 15) / 16;
        u32 off = (rand() % 4096) * 16u;
        /* in uneven pieces, as the GIF hands it over */
        while (qwc) { u32 n = 1 + rand() % 300; if (n > qwc) n = qwc; ps2_gs_hwreg_stream(data + off, n); off += n * 16u; qwc -= n; }
    }
    u32 sz; u8 *v = ps2_gs_vram_ptr(&sz);
    u64 h = 1469598103934665603ull;
    for (u32 i = 0; i < sz; i++) h = (h ^ v[i]) * 1099511628211ull;
    printf("%016llx\n", (unsigned long long)h);
    return 0;
}
