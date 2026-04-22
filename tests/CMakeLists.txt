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
    cpp/test_painn.cpp
    cpp/test_linear_layer.cpp
    cpp/test_silu.cpp
    cpp/test_update.cpp
    cpp/test_message.cpp
    cpp/test_edgegraph.cpp
    cpp/test_rbf.cpp
    cpp/test_cell_list.cpp
    cpp/test_graph_builder.cpp
)

target_compile_features(hadronis_gtest PRIVATE cxx_std_23)

target_include_directories(hadronis_gtest
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(hadronis_gtest
    PRIVATE
        GTest::gtest_main
        Threads::Threads
)

# When built with Clang, enable coverage instrumentation on the C++ test
# binary so that LLVM_PROFILE_FILE is honoured and .profraw files are
# produced for llvm-profdata/llvm-cov.
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(hadronis_gtest PRIVATE
        -fprofile-instr-generate -fcoverage-mapping)
    target_link_options(hadronis_gtest PRIVATE
        -fprofile-instr-generate)
endif()

include(GoogleTest)

gtest_discover_tests(hadronis_gtest)
