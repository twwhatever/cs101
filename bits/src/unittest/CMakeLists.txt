enable_testing()

find_package(GTest REQUIRED)

add_executable(unit_tests test_mylib.cpp)
target_link_libraries(unit_tests PRIVATE mylib GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(unit_tests)
