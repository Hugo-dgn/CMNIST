cmake_minimum_required(VERSION 3.28)

project(
    neural_cpp
    VERSION 1.0.0
    LANGUAGES CXX
)

# ------------------------------------------------------------------------------
# Dependencies
# ------------------------------------------------------------------------------

include(FetchContent)

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
)

# Windows/MSVC runtime consistency
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(googletest)

# ------------------------------------------------------------------------------
# Testing
# ------------------------------------------------------------------------------

enable_testing()

include(GoogleTest)

# ------------------------------------------------------------------------------
# Source collection
# ------------------------------------------------------------------------------

file(
    GLOB_RECURSE NEURAL_CORE_SOURCES
    CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
)

list(
    FILTER NEURAL_CORE_SOURCES
    EXCLUDE REGEX ".*/main\\.cpp$"
)

file(
    GLOB TEST_SOURCES
    CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp"
)

# ------------------------------------------------------------------------------
# Core library
# ------------------------------------------------------------------------------

add_library(neural_core)

target_sources(
    neural_core
    PRIVATE
        ${NEURAL_CORE_SOURCES}
)

target_compile_features(
    neural_core
    PUBLIC
        cxx_std_23
)

target_include_directories(
    neural_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# Optional but recommended warnings
target_compile_options(
    neural_core
    PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>
        $<$<CXX_COMPILER_ID:MSVC>:/W4>
)

# ------------------------------------------------------------------------------
# Main executable
# ------------------------------------------------------------------------------

add_executable(main)

target_sources(
    main
    PRIVATE
        src/main.cpp
)

target_link_libraries(
    main
    PRIVATE
        neural_core
)

# ------------------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------------------

add_executable(run_tests)

target_sources(
    run_tests
    PRIVATE
        ${TEST_SOURCES}
)

target_link_libraries(
    run_tests
    PRIVATE
        neural_core
        GTest::gtest
        GTest::gtest_main
)

gtest_discover_tests(run_tests)