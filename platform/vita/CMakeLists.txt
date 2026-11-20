# PS Vita build of the Ace Combat 5 recompilation.
#
#   cmake -S platform/vita -B build-vita -G Ninja \
#         -DCMAKE_TOOLCHAIN_FILE=$VITASDK/share/vita.toolchain.cmake
#   cmake --build build-vita
#
# Needs VitaSDK (GCC 15+) with the sdl3 package, and generated/ from the
# recompiler, exactly as for the PC build.
cmake_minimum_required(VERSION 3.20)

if(NOT DEFINED CMAKE_TOOLCHAIN_FILE)
  if(DEFINED ENV{VITASDK})
    set(CMAKE_TOOLCHAIN_FILE "$ENV{VITASDK}/share/vita.toolchain.cmake" CACHE PATH "toolchain file")
  else()
    message(FATAL_ERROR "Define the VITASDK environment variable")
  endif()
endif()

project(ac5_vita C CXX)
include("$ENV{VITASDK}/share/vita.cmake" REQUIRED)

set(VITA_APP_NAME "Ace Combat 5")
set(VITA_TITLEID  "ACEC00005")
set(VITA_VERSION  "00.01")
set(PS2_VITA_HEAP_MB 192 CACHE STRING "newlib heap size in MB")

set(CMAKE_C_STANDARD 23)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_C_FLAGS_RELEASE "-O2")
set(CMAKE_CXX_FLAGS_RELEASE "-O2")

get_filename_component(ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
set(VITA_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

set(PS2_GENERATED_DIR "generated" CACHE STRING
    "recompiler output directory, relative to the repository root")
if(IS_ABSOLUTE "${PS2_GENERATED_DIR}")
  set(GEN_DIR "${PS2_GENERATED_DIR}")
else()
  set(GEN_DIR "${ROOT}/${PS2_GENERATED_DIR}")
endif()
file(GLOB GENERATED_SOURCES CONFIGURE_DEPENDS "${GEN_DIR}/ps2_code_*.c")
list(APPEND GENERATED_SOURCES
     "${GEN_DIR}/ps2_func_table.c"
     "${GEN_DIR}/ps2_image.c"
     "${GEN_DIR}/ps2_symbols.c")
if(NOT EXISTS "${GEN_DIR}/ps2_func_table.c")
  message(FATAL_ERROR "No generated sources in ${GEN_DIR}. Run the recompiler "
                      "first (see the main README, step 3).")
endif()

# Everything from the PC runtime except the Vulkan renderer and its ImGui
# overlay, which the null renderer stands in for until the GXM one exists.
set(RUNTIME_SOURCES
    ps2_core.c ps2_hostprof.c ps2_hw.c ps2_gfxq.c ps2_gs.c ps2_vu.c ps2_ipu.c
    ps2_kernel.c ps2_spu2.c ps2_nustream.c ps2_memcard.c ps2_audio.c ps2_sif.c
    ps2_vfs.c ps2_hle_test.c ps2_hle_sif.c ps2_hle_cdvd.c ps2_hle_pad.c
    ps2_hle_iop.c ps2_state.c ps2_capture.c ps2_statecap.c ps2_settings.c
    ps2_mod.c ps2_hook.c ps2_modapi.c ps2_patch.c ps2_params.c ps2_lua.c
    ps2_main.c
    rn/rn_census.c rn/rn_2d.c rn/rn_dump.c rn/rn_vp.c rn/rn_vp_model.c
    rn/rn_vp_terrain.c rn/rn_vp_shadow.c rn/rn_vp_sprite.c rn/rn_vp_load.c
    rn/rn_vp_lit.c rn/rn_vp_fx.c rn/rn_vp_flat.c rn/rn_vpfuzz.c
    rn/rn_intent.c rn/rn_fixes.c rn/rn_sun.c rn/rn_sky.c rn/rn_tri3d.c
    rn/rn_billboard.c rn/rn_worldtri.c rn/rn_screen.c rn/rn_tap.c)
list(TRANSFORM RUNTIME_SOURCES PREPEND "${ROOT}/runtime/src/")
list(APPEND RUNTIME_SOURCES
     "${VITA_DIR}/vita_main.c"
     "${VITA_DIR}/vita_video_gl.c"
     "${VITA_DIR}/vita_shims.c"
     "${VITA_DIR}/vita_emutls.c"
     "${VITA_DIR}/vita_display.c"
     "${VITA_DIR}/vita_threads.c"
     "${VITA_DIR}/vita_mpeg_native.c")

# The runtime's main() becomes ps2_main(); vita_main.c provides main().
set_source_files_properties("${ROOT}/runtime/src/ps2_main.c" PROPERTIES
                            COMPILE_DEFINITIONS "main=ps2_main")

set(LUA_DIR "${ROOT}/third_party/lua")
file(GLOB LUA_SOURCES CONFIGURE_DEPENDS "${LUA_DIR}/*.c")
list(FILTER LUA_SOURCES EXCLUDE REGEX "/(lua|luac|onelua)\\.c$")
add_library(lua54 STATIC ${LUA_SOURCES})
target_include_directories(lua54 PUBLIC "${LUA_DIR}")
target_compile_options(lua54 PRIVATE -O2 -w)

set(CPU_FLAGS -mcpu=cortex-a9 -mfpu=neon -mfloat-abi=hard -mthumb)

add_executable(ac5 ${RUNTIME_SOURCES} ${GENERATED_SOURCES})
target_include_directories(ac5 PRIVATE
    "${VITA_DIR}/include" "${ROOT}/runtime/include" "${GEN_DIR}" "${LUA_DIR}")
target_compile_definitions(ac5 PRIVATE PS2_VITA=1
                                       PS2_VITA_HEAP_MB=${PS2_VITA_HEAP_MB})
target_compile_options(ac5 PRIVATE
    ${CPU_FLAGS}
    -include "${VITA_DIR}/vita_compat.h"
    -fno-strict-aliasing -fwrapv
    -Wall -Wno-unused-parameter -Wno-unused-variable
    -Wno-unused-but-set-variable -Wno-unused-function)

# Generated code: the same -O1 as the PC build, but no patchable entry sleds --
# runtime detours patch x86 machine code and are off on ARM.
set_source_files_properties(${GENERATED_SOURCES} PROPERTIES
                            COMPILE_OPTIONS "-O1;-DPS2_GENERATED_CODE=1")
# ...except the files holding measured hot spots, at -O2 (all 43 at -O2
# takes hours to build). ps2_code_0033: the game's MPEG/IPU library (FMVs).
foreach(hot ps2_code_0033)
    set_source_files_properties("${GEN_DIR}/${hot}.c" PROPERTIES
                                COMPILE_OPTIONS "-O2;-DPS2_GENERATED_CODE=1")
endforeach()
set_source_files_properties(
    "${ROOT}/runtime/src/ps2_vu.c" "${ROOT}/runtime/src/ps2_gs.c"
    "${ROOT}/runtime/src/ps2_hw.c" PROPERTIES COMPILE_OPTIONS "-O3")

target_link_libraries(ac5 PRIVATE
    lua54
    vitaGL vitashark SceShaccCgExt mathneon taihen_stub SceShaccCg_stub
    SceAppMgr_stub SceIofilemgr_stub stdc++
    SDL3
    pthread
    m
    SceAppUtil_stub SceAudio_stub SceAudioIn_stub SceCtrl_stub SceCommonDialog_stub
    SceDisplay_stub SceGxm_stub SceHid_stub SceMotion_stub ScePower_stub
    SceSysmodule_stub SceTouch_stub SceIme_stub SceKernelDmacMgr_stub SceCamera_stub)

# ATTRIBUTE2=12: ask for the extended memory budget.
set(VITA_MKSFOEX_FLAGS "${VITA_MKSFOEX_FLAGS} -d ATTRIBUTE2=12")
vita_create_self(eboot.bin ac5 UNSAFE)
set(ASSETS "${CMAKE_CURRENT_SOURCE_DIR}/assets/sce_sys")
vita_create_vpk(ac5.vpk ${VITA_TITLEID} eboot.bin
    VERSION ${VITA_VERSION}
    NAME ${VITA_APP_NAME}
    FILE ${ASSETS}/icon0.png sce_sys/icon0.png
    FILE ${ASSETS}/livearea/contents/bg.png sce_sys/livearea/contents/bg.png
    FILE ${ASSETS}/livearea/contents/startup.png sce_sys/livearea/contents/startup.png
    FILE ${ASSETS}/livearea/contents/template.xml sce_sys/livearea/contents/template.xml)
