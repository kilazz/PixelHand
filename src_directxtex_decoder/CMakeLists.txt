# Modern CMake minimum version
cmake_minimum_required(VERSION 4.0)

project(directxtex_decoder_project)

# Find Python first
find_package(Python 3.13 COMPONENTS Interpreter Development.Module REQUIRED)

# Include the FetchContent module to download dependencies automatically
include(FetchContent)

# 1. Declare where to fetch nanobind from
FetchContent_Declare(
  nanobind
  GIT_REPOSITORY https://github.com/wjakob/nanobind.git
  GIT_TAG        v2.12.0
)

# 2. Declare where to fetch DirectXTex from (official Microsoft repository)
FetchContent_Declare(
  DirectXTex
  GIT_REPOSITORY https://github.com/microsoft/DirectXTex.git
  GIT_TAG        main
)

# --- Subproject Build Optimizations ---
# Disable building parts of DirectXTex that we don't need (tools and tests)
# This must be set BEFORE calling FetchContent_MakeAvailable
set(BUILD_TOOLS OFF CACHE BOOL "Build command line tools" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "Build unit tests" FORCE)

# Download, configure, and include both dependencies automatically!
# This completely replaces the need for add_subdirectory() and manual git submodules.
FetchContent_MakeAvailable(nanobind DirectXTex)

# Set the C++ standard (Updated to C++20 for modern features)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Main Target Definition ---
# Create the Python module "directxtex_decoder" from its source file.
nanobind_add_module(directxtex_decoder src/decoder.cpp)

# --- Linking ---
# Link our Python module against the DirectXTex static library.
target_link_libraries(directxtex_decoder PRIVATE DirectXTex)

# --- Installation ---
# scikit-build-core requires an install command to know what to package
install(TARGETS directxtex_decoder DESTINATION .)