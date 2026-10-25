cmake_minimum_required(VERSION 3.26)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (NOT DEFINED CMAKE_BUILD_TYPE AND NOT DEFINED ENV{CMAKE_BUILD_TYPE})
    set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "Build type")
endif ()

project(pathlight_gi VERSION 1.0.0 LANGUAGES CXX)

# Dusklight version to build against (a release tag or commit SHA). v2.0.3 provides GfxService 1.4.
if (NOT DUSKLIGHT_VERSION)
    set(DUSKLIGHT_VERSION "v2.0.3")
endif ()
# Fetches the Dusklight sources into ./dusklight and provides DUSKLIGHT_DIR. Pass -DDUSKLIGHT_DIR=
# to use an existing checkout instead.
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/FetchDusklight.cmake")
add_subdirectory("${DUSKLIGHT_DIR}/sdk" dusklight-sdk EXCLUDE_FROM_ALL)

add_mod(pathlight_gi
        FEATURES webgpu
        SOURCES src/mod.cpp
        MOD_JSON mod.json
        RES_DIR res
)
