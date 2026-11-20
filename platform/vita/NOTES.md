# Ace Combat 5 — PS Vita port: state of the work (build 108, final for now, 20 Nov 2026)

Base: github.com/sal063/ace-combat-5-static-recompilation at commit 625ac19
("Performence overhaul"). This zip holds the changed `runtime/` and the
`platform/vita/` port. It contains no game data. These are not included:

- the generated code (`ps2_code_*.c`, which the recompiler makes from your own disc);
- the lifted VU1 programs (`runtime/src/ps2_vu1_progs.inc`; regenerate them with
  `tools/vurecomp`, though the build works without them);
- the LiveArea art.

## Building

1. Install VitaSDK. Build vitaGL from source (commit 3dddc43) with the patches
   listed below:
   `make -j4 STORE_DEPTH_STENCIL=1 HAVE_SHADER_CACHE=1`. Install it into
   `$VITASDK/arm-vita-eabi/lib`.
2. Generate the recompiled sources from your disc (main README, step 3).
3. Run `cmake -S platform/vita -B build-vita -DCMAKE_BUILD_TYPE=Release`, then
   `cmake --build build-vita -j2`. This gives `build-vita/ac5.vpk`.
   A change to `ps2_vu.c` takes about 10 minutes to build; other files take 2–3.
4. On the Vita, put your disc image at `ux0:data/ac5/ac5.iso`. Optionally add
   `ux0:data/ac5/args.txt` with one `NAME=value` setting per line.
   The log goes to `ux0:data/ac5/log.txt`.

vitaGL patches (in `source/`):
- per-framebuffer depth persistence: `vglSetFramebufferDepthPersistence(fb, on)`;
- `vglSetNextSceneRegion(on, xmax, ymax)`: a one-shot valid region for the next scene;
- `_glTexSubImage2D` skips copying old pixels when a whole single-level
  texture is replaced.

## Thread and core layout (build 106 defaults)

| Core | Thread | Priority (lower = more urgent) |
|---|---|---|
| 0 | game (EE) | 150 |
| 0 | texture decoder | 145 (`AC5_TEXHELPER_CORE`, `AC5_TEXHELPER_PRIO`) |
| 1 | graphics worker | 140 |
| 1 | second texture decoder | 142 |
| 2 | renderer (vitaGL) | 140 (`AC5_RENDER_CORE`, `AC5_RENDER_PRIO`) |
| 2 | GS thread | 141 |
| 2 | vblank timer | 112 |
| — | SDL audio thread | 110 |
| — | guest threads (cores 0 and 1 only) | 160 |

The default render scale is 0.625 (400x280), which divides evenly.

## Results so far (in-mission, resolution scale 0.75)

| Build | Average fps | Freezes of 80 ms or more |
|---|---|---|
| 82 | 10–14 | — |
| 88 | 31–40 | about 20 per 5 s |
| 91 | 46–50 | about 20 per 5 s |
| 96 | 50+ flying level, 24–37 staring at the ground | 0–3 per 5 s |

The game runs at a steady 60 fields/s throughout.

## What each fix was (builds 87–99)

- **87:** the audio thread at priority 110 (`AC5_AUDIO_PRIO`), which fixed the audio stutter.
- **88:** the V3-32 fast unpack is back on (`AC5_VIF_FAST`, default 15).
- **89:** texture decoding on three decoders: helper core 1, helper core 0, and
  the renderer while it waits (`AC5_TEX_STEAL=0`, `AC5_TEXHELPER2_CORE=-1`).
  Also the 1 kHz GS-thread sampler.
- **90:** `ps2_log` writes through a buffer and a low-priority thread. A per-line
  flush to the memory card froze the thread that logged for 40–60 ms
  (`AC5_LOG_SYNC=1` writes directly). Also a NEON VRAM page hash (470 MB/s, up from 307).
- **91:** paletted textures. Indexed textures (T8, T4, T8H, T4HL, T4HH) go to the
  GPU as P8 (`GL_PALETTE8_RGBA8_OES`) instead of being expanded to RGBA. That
  is a quarter of the bytes to write and upload (`AC5_TEX_P8=0` turns it off).
- **94:** 8-bit uploads written a whole 16-row band of blocks at a time from a
  byte table. The output is byte-identical to the old path, checked with the host
  verifier (`AC5_TRX_BAND=0` turns it off).
- **96:** removed `fopen("capture.on")` on every field on the GS thread. It
  waited behind disc streaming on the memory card, about 30 ms in heavy fields,
  and was the largest single freeze cause. It now only runs with `AC5_CAPTURE` set.
- **97/99:** samplers for the GS thread and the renderer, which report thread
  state (running / held off its core / waiting).

## Builds 100–108

- **100–101:** the renderer and the texture recognition were split into steps
  in the log.
- **102/104/106:** widescreen detection. One 3D frame switches to 16:9. It goes
  back to 4:3 only after 5 s without 3D. Missions at 512x448 had stayed 4:3
  and later flipped.
- **104:** the renderer moved to core 1 with the worker. Heavy scenes went from
  about 20.6 to about 29 fps, with the same freeze count.
- **105:** 32-bit textures decoded on the helpers. It felt slower, so it is now
  off (`AC5_TEX_ASYNC32=1` turns it on).
- **106:** the texture decoders moved to cores 0 (priority 145) and 1. Below the
  GS on core 2 they were starved and the renderer waited for decodes. The
  samplers are off by default (`AC5_GS_SAMPLER=1` turns them on).
- **107:** the renderer is back on core 2, keeping 106's decoder placement. It
  felt much better than 104–106 in normal flight.
- **108:** the renderer gives way to the GS thread. When a frame arrives while
  the GS still has queued records, the renderer waits up to 8 ms
  (`AC5_RENDER_DEFER_MS`, 0 = build 107 behaviour). In build 107 the renderer
  held the GS thread off core 2 for 26–33 ms in every slow field. Not yet
  measured.

## Tried and ruled out

- **GS thread above the renderer** (`AC5_GSQ_PRIO=139`): the renderer starved, 14–24 fps.
- **GS thread on core 1** (`AC5_GSQ_CORE=1`): the worker held it off its core, 29–35 fps.
- **Renderer on core 0 below the game thread** (build 98): the game thread
  spins while it waits, so the renderer got 2–10 fps and textures dropped out.
- **`AC5_FPS_CAP=30`**: a flat 20 fps.
- **Renderer below the GS thread** (`AC5_RENDER_PRIO=142`): frames arrived
  empty and nothing was drawn.
- **`AC5_GL_NOFX=1`**: everything but movies went black. The game draws its
  scene to another target first.

Open bugs:
- Pre-rendered movies show black vertical bars about every sixth of the width.
  This appeared around builds 91–99 and is not yet isolated; try `AC5_TEX_P8=0`
  and then `AC5_TRX_BAND=0`.
- A flashing white bar at the bottom was seen at scale 0.6.

## What is left (in order)

1. **The renderer holds the GS thread off core 2 for about 25 ms in heavy fields**
   (build 97 samples: GS "ready but not running"). Make the renderer's frame
   cheaper (about 12 ms of draw submission per frame). Build 99's renderer
   samples show where that time goes: the log line `renderer 1 ms samples while
   drawing`. The other option is to give the GS thread a core of its own.
2. **GS work in heavy terrain fields, about 60 ms:**
   - register writes 10–15 ms;
   - texture recognition (hash) 5–10 ms;
   - copying raw pages for decode 5–9 ms;
   - image writes 6–10 ms;
   - dirty marking 2–5 ms.
3. **The briefing map** runs the software vertex path (`nvertex`), at about 20 fps.
4. **In-engine cutscenes** show at 4:3. FMVs are not at full speed.
5. **UI fixes**, which need comparison photos from a PS2.

## args.txt settings

| Setting | Effect |
|---|---|
| `AC5_GL_SCALE=0.75` | render resolution scale (recommended) |
| `AC5_GSQ=0` | GS back on the graphics worker (no GS thread) |
| `AC5_GSQ_CORE`, `AC5_GSQ_PRIO` | GS thread core (default 2) and priority (default 141) |
| `AC5_RENDER_CORE`, `AC5_RENDER_PRIO` | renderer core (default 2) and priority (default 140) |
| `AC5_TEXHELPER_CORE`, `AC5_TEXHELPER2_CORE` | texture decoder cores (default 1 and 0; -1 drops the second) |
| `AC5_TEX_ASYNC=0` | decode textures on the GS thread |
| `AC5_TEX_STEAL=0` | renderer only waits for decodes instead of doing them itself |
| `AC5_TEX_P8=0` | expand indexed textures to RGBA (as builds 90 and earlier) |
| `AC5_TRX_BAND=0` | 8-bit uploads one block row at a time (as builds 93 and earlier) |
| `AC5_TEX_UPID=1` | texture naming by upload records (off: made ground textures switch) |
| `AC5_VIF_FAST` | fast unpack format mask: 1 V4-8, 2 V2-32, 4 V3-32, 8 V3-16; default 15 |
| `AC5_VIF_CHECK=1` | compare the fast unpack with the word-by-word one and log differences |
| `AC5_AUDIO_PRIO` | audio thread priority (default 110) |
| `AC5_GUEST_ALLCORES=1` | let guest threads use core 2 as well |
| `AC5_LOG_SYNC=1` | write the log directly (slow; for crash hunting) |
| `AC5_GS_SAMPLER=0` | turn off the 1 kHz sampler |
| `AC5_SAMPLER_CORE` | sampler core (default 0) |
| `AC5_CAPTURE=1` | poll for `capture.on` (debug captures) |
| `AC5_PROF=1` | detailed timers; costs speed |
| `AC5_FPS_CAP` | frame cap (off; 30 made it worse) |

## Reading the log

- `vita gl: N frames/s`: frames the renderer drew.
- `speed: ... fields=`: game speed (60 = full).
- `renderer loop ... publish gaps <10ms .. >=80 N`: how often a new frame took
  80 ms or more. These are the freezes.
- `gsq: slow GS field, X ms ... | 1 ms samples (running/held off core/waiting)`:
  what the GS thread was doing in a field that took over 50 ms.

## Host test tools (in `tests/`)

- **`gs_verify.c`, `gs_verify_aligned.c`:** upload thousands of random
  transfers and print a hash of VRAM. Run with and without a switch (for
  example `AC5_TRX_BAND=0`); the hashes must match.
- **`gs_bench.c`:** upload timing.
- **`gsq_*.c`:** stress tests of the GS queue.
- **`unpack_diff.c`:** fast against word-by-word VIF unpack.
