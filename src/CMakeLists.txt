
cmake_minimum_required(VERSION 3.15)
project(HadronisCpp)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(pybind11 CONFIG REQUIRED)
find_package(benchmark)


# Use the helper that produces a correctly named Python extension module
pybind11_add_module(_lowlevel hadronis.cpp safetensors/safetensors_impl.cpp)
find_package(Eigen3 REQUIRED)
target_include_directories(_lowlevel PRIVATE ${EIGEN3_INCLUDE_DIRS})
target_link_libraries(_lowlevel PRIVATE Eigen3::Eigen)

target_include_directories(_lowlevel PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${safetensors_cpp_SOURCE_DIR}
)


target_compile_options(_lowlevel
    PRIVATE
        $<$<CXX_COMPILER_ID:MSVC>:/W4 /O2>
        $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -O3 -DNDEBUG -march=native>
)

option(HADRONIS_PROFILE "Build with debug symbols and frame pointers for profiling" OFF)

# When building wheels via scikit-build-core, install the extension
# at the top level so Python can import `_lowlevel`.
install(TARGETS _lowlevel LIBRARY DESTINATION .)


# Only build benchmarks if benchmark is found
if(benchmark_FOUND)
    # Standalone C++ benchmarks for graph build and PaINN only
    add_executable(hadronis_bench_neighbors
        ../benchmarks/cpp/bench_neighbors.cpp
    )
    add_executable(hadronis_bench_painn
        ../benchmarks/cpp/bench_painn.cpp
    )

    target_compile_features(hadronis_bench_neighbors PRIVATE cxx_std_23)
    target_compile_features(hadronis_bench_painn PRIVATE cxx_std_23)

    target_include_directories(hadronis_bench_neighbors PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_include_directories(hadronis_bench_painn PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

    target_compile_options(hadronis_bench_neighbors
        PRIVATE
            $<$<CXX_COMPILER_ID:MSVC>:/W4 /O2>
            $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -O3 -DNDEBUG -march=native>
    )
    target_compile_options(hadronis_bench_painn
        PRIVATE
            $<$<CXX_COMPILER_ID:MSVC>:/W4 /O2>
            $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -O3 -DNDEBUG -march=native>
    )

    if(HADRONIS_PROFILE AND NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        target_compile_options(_lowlevel PRIVATE -g -fno-omit-frame-pointer)
        target_compile_options(hadronis_bench_neighbors PRIVATE -g -fno-omit-frame-pointer)
        target_compile_options(hadronis_bench_painn PRIVATE -g -fno-omit-frame-pointer)
    endif()

    target_link_libraries(hadronis_bench_neighbors PRIVATE Threads::Threads benchmark::benchmark Eigen3::Eigen)
    target_link_libraries(hadronis_bench_painn PRIVATE Threads::Threads benchmark::benchmark Eigen3::Eigen)
endif()

target_link_libraries(_lowlevel PRIVATE Threads::Threads)
