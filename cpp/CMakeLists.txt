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
target_include_directories(cfsonsphere INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(cfsonsphere INTERFACE Eigen3::Eigen)
target_compile_features(cfsonsphere INTERFACE cxx_std_17)

# Tests.
enable_testing()
add_executable(test_cfsonsphere test/test_cfsonsphere.cpp)
target_link_libraries(test_cfsonsphere PRIVATE cfsonsphere)
add_test(NAME test_cfsonsphere COMMAND test_cfsonsphere)

# Optional: cross-check against Julia reference CSVs if they have been generated.
add_executable(test_reference test/test_reference.cpp)
target_link_libraries(test_reference PRIVATE cfsonsphere)
add_test(NAME test_reference
         COMMAND test_reference ${CMAKE_CURRENT_SOURCE_DIR}/reference)

add_executable(test_logpsi test/test_logpsi.cpp)
target_link_libraries(test_logpsi PRIVATE cfsonsphere)
add_test(NAME test_logpsi
         COMMAND test_logpsi ${CMAKE_CURRENT_SOURCE_DIR}/reference)

# Example samplers.
add_executable(sampler_single_state examples/sampler_single_state.cpp)
target_link_libraries(sampler_single_state PRIVATE cfsonsphere)

add_executable(sampler_unprojected examples/sampler_unprojected.cpp)
target_link_libraries(sampler_unprojected PRIVATE cfsonsphere)
