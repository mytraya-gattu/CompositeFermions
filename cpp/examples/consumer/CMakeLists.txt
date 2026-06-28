# Example: a standalone downstream project that consumes the header-only cfsonsphere library.
#
# This is NOT built by the main cpp/ build; it is a self-contained template you can copy.
# Two common ways to pull cfsonsphere in are shown below.
#
# Build (from this directory):
#   cmake -B build -DCMAKE_PREFIX_PATH=/opt/homebrew   # so Eigen3 is found
#   cmake --build build
#   ./build/consumer
cmake_minimum_required(VERSION 3.16)
project(cfsonsphere_consumer CXX)

# --- Option A: vendor the library and add_subdirectory it ---------------------------------
# Point this at wherever the cfsonsphere cpp/ tree lives. Its tests/examples auto-disable
# when added as a subproject, so only the INTERFACE target is configured.
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../.. cfsonsphere_build)

# --- Option B: FetchContent (uncomment to pull from a git remote instead) -----------------
# include(FetchContent)
# FetchContent_Declare(cfsonsphere
#   GIT_REPOSITORY https://github.com/mytraya-gattu/CompositeFermions.git
#   GIT_TAG        main
#   SOURCE_SUBDIR  cpp)
# FetchContent_MakeAvailable(cfsonsphere)

add_executable(consumer main.cpp)
target_link_libraries(consumer PRIVATE cfsonsphere::cfsonsphere)
