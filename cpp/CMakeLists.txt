cmake_minimum_required(VERSION 3.16)
project(cfsonsphere CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Eigen3 REQUIRED)

# Header-only core library.
add_library(cfsonsphere INTERFACE)
add_library(cfsonsphere::cfsonsphere ALIAS cfsonsphere)
target_include_directories(cfsonsphere INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(cfsonsphere INTERFACE Eigen3::Eigen)
target_compile_features(cfsonsphere INTERFACE cxx_std_17)

# A downstream consumer (add_subdirectory / FetchContent) only needs the INTERFACE target
# above, so the tests and example samplers default ON for a standalone build but OFF when
# this project is pulled in as a subproject. See examples/consumer/ for a minimal consumer.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  set(_cfsonsphere_top ON)
else()
  set(_cfsonsphere_top OFF)
endif()
option(CFSONSPHERE_BUILD_TESTS    "Build the cfsonsphere test suite"       ${_cfsonsphere_top})
option(CFSONSPHERE_BUILD_EXAMPLES "Build the cfsonsphere example samplers" ${_cfsonsphere_top})

# Tests.
if(CFSONSPHERE_BUILD_TESTS)
  enable_testing()
  add_executable(test_cfsonsphere test/test_cfsonsphere.cpp)
  target_link_libraries(test_cfsonsphere PRIVATE cfsonsphere)
  add_test(NAME test_cfsonsphere COMMAND test_cfsonsphere)

  # Cross-check against the Julia reference CSVs in reference/ (regenerate with the
  # dump_*.jl scripts there if numerics change).
  add_executable(test_reference test/test_reference.cpp)
  target_link_libraries(test_reference PRIVATE cfsonsphere)
  add_test(NAME test_reference
           COMMAND test_reference ${CMAKE_CURRENT_SOURCE_DIR}/reference)

  add_executable(test_logpsi test/test_logpsi.cpp)
  target_link_libraries(test_logpsi PRIVATE cfsonsphere)
  add_test(NAME test_logpsi
           COMMAND test_logpsi ${CMAKE_CURRENT_SOURCE_DIR}/reference)
endif()

# Example samplers.
if(CFSONSPHERE_BUILD_EXAMPLES)
  add_executable(sampler_single_state examples/sampler_single_state.cpp)
  target_link_libraries(sampler_single_state PRIVATE cfsonsphere)

  add_executable(sampler_unprojected examples/sampler_unprojected.cpp)
  target_link_libraries(sampler_unprojected PRIVATE cfsonsphere)
endif()
