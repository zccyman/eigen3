cmake_minimum_required(VERSION 3.12.0)
set(SOLUTION_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(OP_NAME extra_op)
set(OpDomain "") #"test.customop"

option(WITH_TEST ON)
option(WITH_CUDA ON)
set(LIBRARY_DIR ${SOLUTION_DIR}/../../libs)
set(BINARY_DIR ${SOLUTION_DIR}/../../bins)
include_directories(${SOLUTION_DIR}/ops/core/include)
include_directories(/usr/local/include/eigen3)
### gdb
# SET(CMAKE_BUILD_TYPE "Debug")
# SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb")
# SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")

add_subdirectory(${SOLUTION_DIR}/ops/core)
add_subdirectory(${SOLUTION_DIR}/ops/pyapi)

if(WITH_TEST)
    include(cmake/examples.cmake)
    add_subdirectory(test/cpp/ops/example)
    add_subdirectory(test/cpp/ops)
endif(WITH_TEST)

add_subdirectory(ops/ort_extension)
