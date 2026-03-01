cmake_minimum_required(VERSION 3.15)

include(FetchContent)

# Fetch GoogleTest from its official repository
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.zip
)

# For Windows/MSVC; harmless elsewhere
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(googletest)

add_executable(hadronis_gtest
    cpp/test_sanity.cpp
)

target_compile_features(hadronis_gtest PRIVATE cxx_std_23)

target_link_libraries(hadronis_gtest
    PRIVATE
        GTest::gtest_main
)

include(GoogleTest)

gtest_discover_tests(hadronis_gtest)
