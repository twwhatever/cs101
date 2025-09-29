add_library(mylib mylib.cpp)

find_package(boost REQUIRED)
target_include_directories(mylib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${Boost_INCLUDE_DIR})
