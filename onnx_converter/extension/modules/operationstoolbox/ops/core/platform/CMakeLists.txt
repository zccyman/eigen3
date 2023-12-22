# cmake_minimum_required(VERSION <specify CMake version here>)
set(CMAKE_CXX_STANDARD 17)

include_directories(/usr/local/include/eigen3)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/platform)

file(GLOB_RECURSE platform_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.h ${CMAKE_CURRENT_SOURCE_DIR}/*.cc)

set(platform_FILES ${platform_FILES} PARENT_SCOPE)