#include "ps2_gs.c"
#include <stdio.h>
int main(void){
  static u8 vram[GS_VRAM_SIZE]; srand(3); for (u32 i=0;i<sizeof vram;i++) vram[i]=rand();
  int bad=0;
  for (int t=0;t<40;t++){
    gs_geom g; gs_psm_geom(PSM_CT32,&g);
    u32 tw=32u<<(rand()%4), th=32u<<(rand()%3), tbw=64u*(1+rand()%8); if (tbw<tw) tbw=tw;
    u32 tbp=(rand()%512)*256u;
    u32 span=gs_page_span(&g,tbw,tw-1u,th-1u)*GS_PAGE_BYTES; if (span>GS_VRAM_SIZE-tbp) span=GS_VRAM_SIZE-tbp;
    tex_job *j=malloc(sizeof *j+span); j->tbp=tbp;j->span=span;j->tbw=tbw;j->psm=PSM_CT32;j->tw=tw;j->th=th;j->g=g;j->p8=0;
    memcpy(j->bytes,vram+tbp,span);
    u32 *out=malloc(tw*th*4); tex_job_fill(j,(u8*)out);
    u32 ppr=tbw/g.pw; if(!ppr)ppr=1;
    for(u32 y=0;y<th;y++)for(u32 x=0;x<tw;x++){ u64 bit=vram_bit_g(&g,tbp,ppr,x,y); u32 off=bit>>3,w=0; if(off+4<=tbp+span && off>=tbp) memcpy(&w,vram+off,4); else continue;
      u32 a=(w>>24)&255; a=a>=128?255:a*2; u32 want=(w&0xFFFFFF)|(a<<24); if(out[y*tw+x]!=want) bad++; }
    free(out); free(j);
  }
  printf("ct32 mismatches %d\n",bad); return 0; }
