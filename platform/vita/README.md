# PS Vita port (work in progress)

The full runtime (CPU code, kernel, IOP/SIF HLE, CDVD, emulated GS, VU, SPU2,
IPU, pads, memory cards, Lua) builds for the Vita. The Vulkan renderer is
replaced by `vita_video_gl.c`, a vitaGL (GXM) renderer implementing the same
`ps2_vk_*` interface: the emulated GS turns the game's drawing into GPU draws
exactly as on the PC, and this file replays them on the Vita GPU.

**Requires the Vita shader compiler**: extract `libshacccg.suprx` with the
ShaRKF00D homebrew (it ends up in `ur0:data/external/`). Without it the
renderer refuses to start and the game falls back to the (very slow) software
GS picture drawn by `vita_display.c`. Compiled shaders are cached in
`ux0:data/shader_cache/`, so the first run of each new effect stutters once.

## Build

Needs VitaSDK with GCC 15 or newer (the generated code relies on `musttail`),
the `sdl3`, `vitaGL`, `vitaShaRK`, `SceShaccCgExt`, `libmathneon` and `taihen`
packages, and `generated/` from the recompiler (main README, step 3,
same flags, same `SLUS_208.51`).

```sh
export VITASDK=/usr/local/vitasdk PATH=$VITASDK/bin:$PATH
cmake -S platform/vita -B build-vita -G Ninja
cmake --build build-vita
```

This gives `build-vita/ac5.vpk`.

## Install and run

1. Install `ac5.vpk` with VitaShell.
2. Make `ux0:data/ac5/` and copy in:
   - `generated/ps2_image.bin`
   - your disc as `ac5.iso`, or its extracted files in a `disc/` folder
3. Launch. The log goes to `ux0:data/ac5/log.txt`.

Extra runtime flags go in `ux0:data/ac5/args.txt`, whitespace separated, for
example `--verbose` or `--vblanks 600`.

## What differs from the PC build

| Area | Vita |
|---|---|
| Renderer | vitaGL: GS draws, blending, depth, alpha test, render targets. Not yet: native mesh path, sun/sky passes, HUD overlay, DATE, RG shuffles, depth sampling, colclip, upscaling |
| ImGui overlay | not built |
| Runtime detours (`ps2_hook_*`) | unavailable: they patch x86 code. The recompile-time hooks in `config/hooks.json` still work. Affects `rn_fixes`, `rn_tap`, `ps2_params` and native mods. |
| Native mods (DLL/.so) | not supported; Lua mods are still loaded |
| Threads | every pthread created without attributes gets a 1 MB stack (`vita_compat.h`) |
| Clocks | CPU 444 MHz, bus 222, GPU 222 |
| Heap | 192 MB (`-DPS2_VITA_HEAP_MB=`) |
