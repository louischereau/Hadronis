cmake_minimum_required(VERSION 3.15)
project(Hadronis)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set global release flags before any subdirectory
if(NOT MSVC)
    add_compile_options(-O3 -DNDEBUG -march=native)
endif()

find_package(Threads REQUIRED)
find_package(pybind11 REQUIRED)

include(FetchContent)
FetchContent_Declare(
    safetensors_cpp
    GIT_REPOSITORY https://github.com/syoyo/safetensors-cpp
    GIT_TAG        main
)
FetchContent_MakeAvailable(safetensors_cpp)

add_subdirectory(src)

enable_testing()
add_subdirectory(tests)
