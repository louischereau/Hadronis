cmake_minimum_required(VERSION 3.15)
project(Hadronis)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find pybind11
find_package(pybind11 REQUIRED)

add_subdirectory(src)

enable_testing()
add_subdirectory(tests)
