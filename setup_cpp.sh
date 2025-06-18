#!/bin/bash

set -e

# 1. Check if Conan is installed
if ! command -v conan &> /dev/null; then
  echo "❌ Conan is not installed. Please run: pip install conan"
  exit 1
fi

echo "✅ Conan found."

# 2. Setup directory structure
mkdir -p src/lib src/exe src/unittest build

# 3. Create conanfile.txt
cat > conanfile.txt <<EOF
[requires]
gtest/1.14.0

[generators]
CMakeToolchain
CMakeDeps

[layout]
cmake_layout
EOF

# 4. Top-level CMakeLists.txt
cat > CMakeLists.txt <<EOF
cmake_minimum_required(VERSION 3.15)

include(\${CMAKE_BINARY_DIR}/conan_toolchain.cmake OPTIONAL)
include(\${CMAKE_BINARY_DIR}/conan_deps.cmake OPTIONAL)

project(MyProject)

add_subdirectory(src/lib)
add_subdirectory(src/exe)
add_subdirectory(src/unittest)
EOF

# 5. lib/
cat > src/lib/CMakeLists.txt <<EOF
add_library(mylib mylib.cpp)
target_include_directories(mylib PUBLIC \${CMAKE_CURRENT_SOURCE_DIR})
EOF

cat > src/lib/mylib.cpp <<EOF
#include "mylib.h"

int add(int a, int b) {
    return a + b;
}
EOF

cat > src/lib/mylib.h <<EOF
#pragma once

int add(int a, int b);
EOF

# 6. exe/
cat > src/exe/CMakeLists.txt <<EOF
add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE mylib)
EOF

cat > src/exe/main.cpp <<EOF
#include <iostream>
#include "mylib.h"

int main() {
    std::cout << "2 + 3 = " << add(2, 3) << std::endl;
    return 0;
}
EOF

# 7. unittest/
cat > src/unittest/CMakeLists.txt <<EOF
enable_testing()

find_package(GTest REQUIRED)

add_executable(unit_tests test_mylib.cpp)
target_link_libraries(unit_tests PRIVATE mylib GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(unit_tests)
EOF

cat > src/unittest/test_mylib.cpp <<EOF
#include "mylib.h"
#include <gtest/gtest.h>

TEST(MyLibTest, Add) {
    EXPECT_EQ(add(1, 2), 3);
    EXPECT_EQ(add(-1, 1), 0);
}
EOF

echo "✅ Project scaffolded successfully."
echo "👉 Next steps:"
echo "   cd build"
echo "   conan install .. --build=missing"
echo "   cmake .. -DCMAKE_TOOLCHAIN_FILE=Release/generators/conan_toolchain.cmake -DCMAKE_BUILD_TYPE=Release"
echo "   cmake --build ."
echo "   ./src/exe/my_app"
echo "   ./src/unittest/unit_tests"
