cmake_minimum_required(VERSION 3.28)

project(
    neural_cpp
    VERSION 1.0.0
    LANGUAGES CXX
)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ------------------------------------------------------------------------------
# Global build configuration (IMPORTANT FIXES)
# ------------------------------------------------------------------------------

# Ensure Release builds are truly optimized everywhere
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add architecture + high optimization globally for Release
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native -DNDEBUG")

# Enable LTO globally in Release (important for inlining across libs)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)

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

file(GLOB_RECURSE NEURAL_CORE_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
)
list(FILTER NEURAL_CORE_SOURCES EXCLUDE REGEX ".*/main\\.cpp$")

file(GLOB_RECURSE TEST_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp"
)

file(GLOB_RECURSE BENCHMARK_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/*.cpp"
)

# ------------------------------------------------------------------------------
# Core library
# ------------------------------------------------------------------------------

add_library(neural_core STATIC)

target_sources(neural_core PRIVATE ${NEURAL_CORE_SOURCES})

target_compile_features(neural_core PUBLIC cxx_std_23)

target_include_directories(
    neural_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_compile_options(
    neural_core PRIVATE
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
# Tests
# ------------------------------------------------------------------------------

add_executable(run_tests ${TEST_SOURCES})

target_link_libraries(run_tests PRIVATE
    neural_core
    GTest::gtest
    GTest::gtest_main
)

gtest_discover_tests(run_tests)

# ------------------------------------------------------------------------------
# Benchmarks (IMPORTANT: clean + fully optimized)
# ------------------------------------------------------------------------------

add_executable(run_benchmarks ${BENCHMARK_SOURCES})

target_link_libraries(run_benchmarks PRIVATE
    neural_core
    benchmark::benchmark
    benchmark::benchmark_main
)

# Ensure benchmarks are always optimized and inlined aggressively
set_target_properties(run_benchmarks PROPERTIES
    INTERPROCEDURAL_OPTIMIZATION TRUE
)

# IMPORTANT: DO NOT manually add -O3 here anymore
# CMake already handles Release flags globally