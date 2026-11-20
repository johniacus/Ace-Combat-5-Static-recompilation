/* The Vita has no dynamic loader for native mods. ps2_mod.c still runs (it
 * also handles Lua mods and archive overrides); native .dll/.so mods simply
 * report that they could not be loaded. */
#ifndef PS2_VITA_DLFCN_H
#define PS2_VITA_DLFCN_H

#define RTLD_LAZY   0x1
#define RTLD_NOW    0x2
#define RTLD_LOCAL  0x0
#define RTLD_GLOBAL 0x100

static inline void *dlopen(const char *path, int mode) {
    (void)path; (void)mode; return 0;
}
static inline void *dlsym(void *h, const char *name) {
    (void)h; (void)name; return 0;
}
static inline int dlclose(void *h) { (void)h; return 0; }
static inline char *dlerror(void) {
    return (char *)"native mods are not supported on PS Vita";
}

#endif
