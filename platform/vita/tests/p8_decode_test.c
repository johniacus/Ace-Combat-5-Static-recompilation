#include "ps2_gs.c"
#include <stdio.h>
int main(void){
  static u8 vram[GS_VRAM_SIZE]; srand(5); for (u32 i=0;i<sizeof vram;i++) vram[i]=rand();
  int bad=0; u32 psms[5]={PSM_T8,PSM_T4,PSM_T8H,PSM_T4HL,PSM_T4HH};
  for (int t=0;t<60;t++){
    u32 psm=psms[t%5]; gs_geom g; gs_psm_geom(psm,&g);
    u32 tw=32u<<(rand()%4), th=32u<<(rand()%3), tbw=64u*(1+rand()%8); if (tbw<tw) tbw=tw;
    u32 tbp=(rand()%512)*256u;
    u32 span=gs_page_span(&g,tbw,tw-1u,th-1u)*GS_PAGE_BYTES; if (span>GS_VRAM_SIZE-tbp) span=GS_VRAM_SIZE-tbp;
    tex_job *j=malloc(sizeof *j+span); j->tbp=tbp;j->span=span;j->tbw=tbw;j->psm=psm;j->tw=tw;j->th=th;j->g=g;
    for(int e=0;e<256;e++) j->pal[e]=rand();
    memcpy(j->bytes,vram+tbp,span);
    u32 *a=malloc(tw*th*4); u8 *b=malloc(tw*th*4); j->p8=0; tex_job_fill(j,(u8*)a); j->p8=1; tex_job_fill(j,b);
    u32 pal[256]; memcpy(pal,b+16,1024);
    for(u32 i=0;i<tw*th;i++) if(pal[b[1040+i]]!=a[i]) bad++;
    free(a); free(b); free(j);
  }
  printf("p8 mismatches %d\n",bad); return 0; }
