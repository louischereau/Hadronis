
cmake_minimum_required(VERSION 3.15)
project(HadronisCpp)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(pybind11 CONFIG REQUIRED)

# Use the helper that produces a correctly named Python extension module
pybind11_add_module(_lowlevel hadronis.cpp)

target_compile_options(_lowlevel
	PRIVATE
		$<$<CXX_COMPILER_ID:MSVC>:/W4 /O2>
		$<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -O2>
)

option(HADRONIS_ENABLE_SIMD "Enable SIMD optimizations (may reduce portability)" OFF)

if(HADRONIS_ENABLE_SIMD AND NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
	target_compile_options(_lowlevel PRIVATE -march=native)
endif()

# When building wheels via scikit-build-core, install the extension
# at the top level so Python can import `_lowlevel`.
install(TARGETS _lowlevel LIBRARY DESTINATION .)
