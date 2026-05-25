cmake_minimum_required(VERSION 3.28)

project(
    neural_cpp
    VERSION 1.0.0
    LANGUAGES CXX
)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ------------------------------------------------------------------------------
# Dependencies
# ------------------------------------------------------------------------------

include(FetchContent)

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
)

FetchContent_Declare(
    benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.8.3
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(googletest benchmark)

# ------------------------------------------------------------------------------
# Testing
# ------------------------------------------------------------------------------

enable_testing()
include(GoogleTest)

# ------------------------------------------------------------------------------
# Source collection
# ------------------------------------------------------------------------------

file(GLOB_RECURSE NEURAL_CORE_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
list(FILTER NEURAL_CORE_SOURCES EXCLUDE REGEX ".*/main\\.cpp$")

file(GLOB_RECURSE TEST_SOURCES       CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp")
file(GLOB_RECURSE BENCHMARK_SOURCES  CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/*.cpp")

# ------------------------------------------------------------------------------
# Core library
# ------------------------------------------------------------------------------

add_library(neural_core)

target_sources(neural_core PRIVATE ${NEURAL_CORE_SOURCES})

target_compile_features(neural_core PUBLIC cxx_std_23)

target_include_directories(
    neural_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_compile_options(
    neural_core
    PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>
        $<$<CXX_COMPILER_ID:MSVC>:/W4>
)

find_package(OpenMP REQUIRED)
target_link_libraries(neural_core PRIVATE OpenMP::OpenMP_CXX)

# ------------------------------------------------------------------------------
# Main executable
# ------------------------------------------------------------------------------

add_executable(main src/main.cpp)
target_link_libraries(main PRIVATE neural_core)

# ------------------------------------------------------------------------------
# Tests  (always Debug — inherits CMAKE_BUILD_TYPE)
# ------------------------------------------------------------------------------

add_executable(run_tests)
target_sources(run_tests PRIVATE ${TEST_SOURCES})
target_link_libraries(run_tests PRIVATE neural_core GTest::gtest GTest::gtest_main)
gtest_discover_tests(run_tests)

# ------------------------------------------------------------------------------
# Benchmarks  (always fully optimised, regardless of CMAKE_BUILD_TYPE)
# ------------------------------------------------------------------------------

add_executable(run_benchmarks)
target_sources(run_benchmarks PRIVATE ${BENCHMARK_SOURCES})
target_link_libraries(run_benchmarks PRIVATE neural_core benchmark::benchmark benchmark::benchmark_main)

# Strip any inherited debug flags and force Release-level optimisation
# Works on GCC/Clang; for MSVC swap with /O2 /DNDEBUG
target_compile_options(
    run_benchmarks
    PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang>:-O3 -march=native -DNDEBUG>
        $<$<CXX_COMPILER_ID:MSVC>:/O2 /DNDEBUG>
)

# Also tell the linker to strip debug info for the benchmark binary
set_target_properties(
    run_benchmarks
    PROPERTIES
        INTERPROCEDURAL_OPTIMIZATION TRUE   # enables LTO for this target only
)