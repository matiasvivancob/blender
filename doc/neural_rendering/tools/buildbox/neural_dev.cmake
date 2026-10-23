# SPDX-FileCopyrightText: 2026 Blender Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

# Developer configuration for the neural-rendering fork: NVIDIA-only Cycles for
# an RTX 3090 (sm_86) and an RTX 4070 Ti Super (sm_89), on Linux and on Windows
# (MSVC + Ninja). Platform specific parts are guarded with if(UNIX) / if(WIN32):
# inside a -C script UNIX and WIN32 reflect the host, as in the files under
# build_files/cmake/config/.
#
# Style follows build_files/cmake/config/blender_{lite,release,developer}.cmake.
# Every option below was checked against the tree; the file that defines it is
# named in the comment next to it.
#
# Example usage (absolute path to this file, build dir outside the source tree):
#   cmake -G Ninja -C /path/to/blender/doc/neural_rendering/tools/buildbox/neural_dev.cmake \
#         -S /path/to/blender -B /path/to/build_linux_neural -DCMAKE_BUILD_TYPE=Release
#
# Or through the Blender GNUmakefile (Linux; BUILD_CMAKE_ARGS goes to CMAKE_CONFIG_ARGS):
#   make ninja BUILD_DIR=../build_linux_neural \
#        BUILD_CMAKE_ARGS="-C$PWD/doc/neural_rendering/tools/buildbox/neural_dev.cmake"
#
# Windows: make.bat has NO pass-through for extra CMake arguments (it resets
# BUILD_CMAKE_ARGS and only knows its own switches), so configure with cmake
# directly from a Visual Studio developer shell (see README.md, Windows section):
#   cmake -G Ninja -C D:\path\to\blender\doc\neural_rendering\tools\buildbox\neural_dev.cmake ^
#         -S D:\path\to\blender -B D:\path\to\build_neural -DCMAKE_BUILD_TYPE=Release
#
# Notes:
# - "-C" only pre-populates the cache. Every value is FORCEd, so to change one
#   later run `cmake -D<NAME>=<VALUE> <build dir>` WITHOUT "-C" (a plain -D
#   edit sticks) or edit CMakeCache.txt / run ccmake.
# - Environment variable OPTIX_ROOT_DIR is honoured below (see OptiX section).
#   On Windows the default installer location is searched too.
# - Keep the path to this file free of spaces if you go through make.bat or a
#   shell that re-splits arguments; CMake itself does not mind.

# -----------------------------------------------------------------------------
# Fail loudly instead of silently dropping a GPU backend.
#
# Without this, a missing OptiX SDK or nvcc only prints a STATUS line such as
# "OptiX not found, disabling WITH_CYCLES_DEVICE_OPTIX" and configures fine
# (set_and_warn_library_found() in build_files/cmake/macros.cmake; the option is
# defined in CMakeLists.txt). On this box that would produce a Blender that
# renders on CPU only. With it ON the same situations are a configure error.
# If it trips on an unrelated optional dependency, set it OFF with a plain
# `cmake -DWITH_STRICT_BUILD_OPTIONS=OFF <build dir>` and grep the configure
# log for "not found, disabling" instead.
set(WITH_STRICT_BUILD_OPTIONS         ON  CACHE BOOL "" FORCE)

# -----------------------------------------------------------------------------
# Cycles: NVIDIA only.

# CMakeLists.txt: option(WITH_CYCLES ...) defaults ON; spelled out for clarity.
set(WITH_CYCLES                       ON  CACHE BOOL "" FORCE)

# CMakeLists.txt, `if(NOT APPLE)` block: CUDA + OptiX devices and the
# ahead-of-time CUDA binaries (WITH_CYCLES_CUDA_BINARIES defaults OFF, which
# means no cubin AND no OptiX PTX is built; see
# intern/cycles/kernel/device/{cuda,optix}/CMakeLists.txt).
set(WITH_CYCLES_DEVICE_CUDA           ON  CACHE BOOL "" FORCE)
set(WITH_CYCLES_DEVICE_OPTIX          ON  CACHE BOOL "" FORCE)
set(WITH_CYCLES_CUDA_BINARIES         ON  CACHE BOOL "" FORCE)

# 3090 = compute capability 8.6, 4070 Ti Super = 8.9. The runtime picks
# kernel_sm_<major><minor>.cubin.zst, falling back to lower minor versions
# (intern/cycles/device/cuda/device_impl.cpp, `compile_kernel`), so both cards
# get an exact match. No compute_XX PTX entry is listed on purpose: it only
# serves GPUs we do not own. The OptiX PTX is built separately for compute_50 /
# compute_75 (intern/cycles/kernel/device/optix/CMakeLists.txt) and is not
# controlled by this variable.
# NOTE: the value is a ;-separated CMake list.
set(CYCLES_CUDA_BINARIES_ARCH         "sm_86;sm_89" CACHE STRING "" FORCE)

# Other vendors off. HIP/oneAPI/HIPRT are defined in CMakeLists.txt inside
# `if(NOT APPLE AND NOT (WIN32 AND ...ARM64))`; Metal only exists on Apple.
set(WITH_CYCLES_DEVICE_HIP            OFF CACHE BOOL "" FORCE)
set(WITH_CYCLES_HIP_BINARIES          OFF CACHE BOOL "" FORCE)
set(WITH_CYCLES_DEVICE_HIPRT          OFF CACHE BOOL "" FORCE)
set(WITH_CYCLES_DEVICE_ONEAPI         OFF CACHE BOOL "" FORCE)
set(WITH_CYCLES_ONEAPI_BINARIES       OFF CACHE BOOL "" FORCE)
if(APPLE)
  set(WITH_CYCLES_DEVICE_METAL        OFF CACHE BOOL "" FORCE)
endif()

# DLSS Ray Reconstruction needs the NVIDIA DLSS SDK (FindDLSS.cmake, DLSS_SDK_ROOT)
# whose licence is not compatible with this tree (see PROTOCOL.md). Keep OFF.
set(WITH_DLSS                         OFF CACHE BOOL "" FORCE)

# -----------------------------------------------------------------------------
# Cycles: debugging and iteration speed.

# intern/cycles/kernel/device/cuda/CMakeLists.txt, cuda_add_common_flags():
# with this ON nvcc gets `-D WITH_CYCLES_DEBUG --ptxas-options="-v"`, so every
# kernel build prints per-kernel register / spill / stack numbers. It also
# enables the developer sampling patterns, the guiding debug passes and
# `_cycles.with_debug` (intern/cycles/blender/python.cpp). It does NOT make
# kernel_assert() live on GPU (no-op in kernel/device/{cuda,optix}/compat.h).
set(WITH_CYCLES_DEBUG                 ON  CACHE BOOL "" FORCE)

# Only compile the CPU kernel for this machine's instruction set (CMakeLists.txt,
# "for development only"; also set by blender_developer.cmake). The CPU kernel
# is the scalar fallback / parity reference, not a production target here.
set(WITH_CYCLES_NATIVE_ONLY           ON  CACHE BOOL "" FORCE)

# OSL multiplies the OptiX work: kernel/device/optix/CMakeLists.txt adds eight
# relocatable-device-code PTX modules when WITH_CYCLES_OSL is ON. The neural
# closures are SVM-only for now, so turn OSL off for the fast loop. Turn it back
# on (`cmake -DWITH_CYCLES_OSL=ON <build dir>`) before touching code that is
# shared with OSL (closure registration, kernel/osl/).
set(WITH_CYCLES_OSL                   OFF CACHE BOOL "" FORCE)

# Path guiding stays ON (CMake default): the NRC plan refactors the OpenPGL
# vertex-recording calls, and a refactor must not silently break that build.
# It is CPU-only (tests/python/cycles_render_tests.py: "No path guiding on GPU").
set(WITH_CYCLES_PATH_GUIDING          ON  CACHE BOOL "" FORCE)

# Regression-test devices, used by tests/python/CMakeLists.txt to generate the
# ctest names cycles_<subject>_<device>.
set(CYCLES_TEST_DEVICES               "CPU;CUDA;OPTIX" CACHE STRING "" FORCE)

# GTest unit tests (intern/cycles/test/*; kernel_camera_projection_test.cpp is
# the pattern for the golden-vector tests of the neural stage).
set(WITH_GTESTS                       ON  CACHE BOOL "" FORCE)

# -----------------------------------------------------------------------------
# Build-system conveniences.

# Linux/macOS: platform_unix.cmake sets CMAKE_C/CXX_COMPILER_LAUNCHER to ccache
# AND intern/cycles/kernel/device/cuda/CMakeLists.txt prefixes the nvcc cubin
# commands with ccache when this is ON. Setting only the compiler launchers
# would leave the (dominant) nvcc cost uncached. Falls back to OFF with a
# warning if ccache is not installed. The option only exists under if(UNIX)
# in CMakeLists.txt, so it must not be set on Windows.
if(UNIX)
  set(WITH_COMPILER_CCACHE            ON  CACHE BOOL "" FORCE)
endif()

# Windows: the equivalent is WITH_WINDOWS_SCCACHE (CMakeLists.txt, inside the
# if(WIN32) option block). It ONLY applies with the Ninja generator:
# build_files/cmake/platform/platform_win32.cmake turns it off with a warning
# when CMAKE_VS_MSBUILD_COMMAND is set (Visual Studio generators), and when ON it
# sets the C/C++ compiler launchers to `sccache` and switches debug info to /Z7
# (sccache cannot cache /Zi). Unlike ccache above it does NOT wrap the nvcc
# cubin / OptiX PTX commands (those test WITH_COMPILER_CCACHE AND CCACHE_PROGRAM,
# which only unix/apple platform files define), so on Windows only the C/C++
# part of the build is cached. Enabled only if sccache is on PATH; otherwise the
# launcher would make every compile fail.
if(WIN32)
  find_program(_neural_sccache sccache)
  if(_neural_sccache)
    set(WITH_WINDOWS_SCCACHE          ON  CACHE BOOL "" FORCE)
  else()
    message(WARNING "neural_dev.cmake: sccache not found on PATH, leaving WITH_WINDOWS_SCCACHE OFF "
                    "(winget install Mozilla.sccache, then reconfigure)")
  endif()
  unset(_neural_sccache CACHE)
endif()

# Do not relink on every build just to refresh the build timestamp/hash
# (also set by blender_developer.cmake).
set(WITH_BUILDINFO                    OFF CACHE BOOL "" FORCE)
set(WITH_DOC_MANPAGE                  OFF CACHE BOOL "" FORCE)

# compile_commands.json for clangd (blender_developer.cmake does the same).
set(CMAKE_EXPORT_COMPILE_COMMANDS     ON  CACHE BOOL "" FORCE)

# -----------------------------------------------------------------------------
# Toolkit locations.

# OptiX SDK. build_files/cmake/Modules/FindOptiX.cmake searches
# ${OPTIX_ROOT_DIR}/include/optix.h and reads $ENV{OPTIX_ROOT_DIR} when the
# CMake variable is empty; intern/cycles/CMakeLists.txt requires >= 8.0.0
# (the Blender buildbot pins 9.0.0, build_files/buildbot/config/blender_version.cmake).
# Copy the environment value into the cache so CMakeCache.txt records it.
if(NOT "$ENV{OPTIX_ROOT_DIR}" STREQUAL "")
  set(_neural_optix_root "$ENV{OPTIX_ROOT_DIR}")
  if(WIN32)
    file(TO_CMAKE_PATH "${_neural_optix_root}" _neural_optix_root)  # C:\x\y -> C:/x/y
  endif()
  set(OPTIX_ROOT_DIR "${_neural_optix_root}" CACHE PATH "" FORCE)
elseif(WIN32)
  # FindOptiX.cmake has NO default install location: it only looks below
  # OPTIX_ROOT_DIR (CMake variable or environment). The NVIDIA installer puts
  # the SDK in "%ProgramData%\NVIDIA Corporation\OptiX SDK <version>" (the buildbot
  # config build_files/buildbot/config/blender_windows.cmake uses the same
  # "OptiX SDK <version>" folder name). Take the newest one that has include/optix.h.
  file(TO_CMAKE_PATH "$ENV{ProgramData}" _neural_programdata)
  if(_neural_programdata)
    file(GLOB _neural_optix_dirs "${_neural_programdata}/NVIDIA Corporation/OptiX SDK *")
    list(SORT _neural_optix_dirs COMPARE NATURAL ORDER DESCENDING)
    foreach(_neural_dir IN LISTS _neural_optix_dirs)
      if(EXISTS "${_neural_dir}/include/optix.h")
        set(OPTIX_ROOT_DIR "${_neural_dir}" CACHE PATH "" FORCE)
        message(STATUS "neural_dev.cmake: using OptiX SDK ${_neural_dir}")
        break()
      endif()
    endforeach()
  endif()
endif()

# CUDA toolkit. Left to find_package(CUDA) (intern/cycles/cmake/external_libs.cmake)
# which uses the nvcc on PATH or CUDA_TOOLKIT_ROOT_DIR. To pin a toolkit pass
#   -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12.8 -DCUDA_NVCC_EXECUTABLE=/usr/local/cuda-12.8/bin/nvcc
# and, when the default gcc is newer than that nvcc accepts,
#   -DCUDA_HOST_COMPILER=/usr/bin/gcc-13
# (CUDA_HOST_COMPILER is consumed as `-ccbin` by cuda_add_common_flags()).
# Per-major overrides CUDA12_/CUDA13_{TOOLKIT_ROOT_DIR,NVCC_EXECUTABLE} are read
# by cuda_get_toolkit_for_arch() in kernel/device/cuda/CMakeLists.txt.
#
# Windows: find_package(CUDA) looks for nvcc.exe in $ENV{CUDA_PATH}/bin (set by the
# CUDA installer, FindCUDA.cmake) and then on PATH. With several toolkits
# installed pin one:
#   -DCUDA_TOOLKIT_ROOT_DIR="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9"
# The nvcc host compiler (-ccbin) is the cl.exe CMake found (CMAKE_C_COMPILER);
# platform_win32.cmake only overrides CUDA_HOST_COMPILER for clang-cl. Cycles also
# passes -D _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH to nvcc, which relaxes the
# STL version check but not nvcc's own check of the cl.exe version.

# -----------------------------------------------------------------------------
# Optional trims, deliberately NOT applied (uncomment to cut cold-build time;
# none of them is needed by the Cycles render tests).
#
# set(WITH_XR_OPENXR   OFF CACHE BOOL "" FORCE)
# set(WITH_MOD_FLUID   OFF CACHE BOOL "" FORCE)
# set(WITH_USD         OFF CACHE BOOL "" FORCE)
# set(WITH_MATERIALX   OFF CACHE BOOL "" FORCE)
# set(WITH_HEADLESS    ON  CACHE BOOL "" FORCE)  # blender_headless.cmake: no GUI at all
#
# On a machine with plenty of RAM (>= 64 GB), kernels of different devices can
# build concurrently (CMakeLists.txt: advanced option, default OFF):
# set(WITH_CYCLES_PARALLEL_DEVICE_KERNEL_BUILD ON CACHE BOOL "" FORCE)
