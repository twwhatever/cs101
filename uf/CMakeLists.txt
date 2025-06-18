cmake_minimum_required(VERSION 3.15)

include(${CMAKE_BINARY_DIR}/conan_toolchain.cmake OPTIONAL)
include(${CMAKE_BINARY_DIR}/conan_deps.cmake OPTIONAL)

project(MyProject)

set(CMAKE_CXX_STANDARD 20)

add_subdirectory(src/lib)
add_subdirectory(src/exe)
add_subdirectory(src/unittest)
