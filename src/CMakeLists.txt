
cmake_minimum_required(VERSION 3.15)
project(HadronisCpp)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(pybind11 CONFIG REQUIRED)

# Use the helper that produces a correctly named Python extension module
pybind11_add_module(_lowlevel hadronis.cpp)

target_include_directories(_lowlevel PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${safetensors_cpp_SOURCE_DIR}
)

target_compile_options(_lowlevel
	PRIVATE
		$<$<CXX_COMPILER_ID:MSVC>:/W4 /O2>
		$<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -O2>
)

option(HADRONIS_ENABLE_SIMD "Enable SIMD optimizations (may reduce portability)" OFF)
option(HADRONIS_PROFILE "Build with debug symbols and frame pointers for profiling" OFF)

if(HADRONIS_ENABLE_SIMD AND NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
	target_compile_options(_lowlevel PRIVATE -march=native)
endif()

# When building wheels via scikit-build-core, install the extension
# at the top level so Python can import `_lowlevel`.
install(TARGETS _lowlevel LIBRARY DESTINATION .)

# Standalone C++ benchmarks for geometry and neighbor list routines
add_executable(hadronis_bench_geometry
	../benchmarks/cpp/bench_geometry.cpp
)

target_compile_features(hadronis_bench_geometry PRIVATE cxx_std_23)

target_include_directories(hadronis_bench_geometry
	PRIVATE
		${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_options(hadronis_bench_geometry
	PRIVATE
		$<$<CXX_COMPILER_ID:MSVC>:/W4 /O2>
		$<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -O2>
)

if(HADRONIS_PROFILE AND NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
	target_compile_options(_lowlevel PRIVATE -g -fno-omit-frame-pointer)
	target_compile_options(hadronis_bench_geometry PRIVATE -g -fno-omit-frame-pointer)
endif()

target_link_libraries(_lowlevel PRIVATE Threads::Threads)
target_link_libraries(hadronis_bench_geometry PRIVATE Threads::Threads)
