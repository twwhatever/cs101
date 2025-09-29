cmake_minimum_required(VERSION 3.15)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(${CMAKE_BINARY_DIR}/conan_toolchain.cmake OPTIONAL)
include(${CMAKE_BINARY_DIR}/conan_deps.cmake OPTIONAL)

project(MyProject)

add_subdirectory(src/lib)
add_subdirectory(src/exe)
add_subdirectory(src/unittest)
