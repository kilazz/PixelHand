# Modern CMake minimum version
cmake_minimum_required(VERSION 4.0)

project(directxtex_decoder_project)

# Find Python first
find_package(Python 3.13 COMPONENTS Interpreter Development.Module REQUIRED)

# Find or download nanobind
include(FetchContent)
FetchContent_Declare(
  nanobind
  GIT_REPOSITORY https://github.com/wjakob/nanobind.git
  GIT_TAG        v2.12.0
)
FetchContent_MakeAvailable(nanobind)

# Set the C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Subproject Build Optimizations ---
# Disable building parts of DirectXTex that we don't need (tools and tests).
set(BUILD_TOOLS OFF CACHE BOOL "Build command line tools" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "Build unit tests" FORCE)

# Add the DirectXTex library as a subproject from the external folder.
add_subdirectory(external/DirectXTex)

# --- Main Target Definition ---
# Create the Python module "directxtex_decoder" from its source file.
nanobind_add_module(directxtex_decoder directxtex_decoder/decoder.cpp)

# --- Linking ---
# Link our Python module against the DirectXTex static library.
target_link_libraries(directxtex_decoder PRIVATE DirectXTex)

# --- Installation ---
# scikit-build-core requires an install command to know what to package
install(TARGETS directxtex_decoder DESTINATION .)
