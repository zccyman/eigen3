cmake_minimum_required(VERSION 3.12)
project(ops_test)

enable_testing()
include(GoogleTest)
#find_package(GTest 1.10.0)
find_package(GTest)

# include_directories(${SOLUTION_DIR}/ops/core/include)
link_directories(${LIBRARY_DIR})

if(WITH_CUDA)
    link_directories(${CUDNN_LIBRARY_PATH})
endif()

add_executable(ops_test splice_test.cc)

if(WITH_CUDA)
    target_link_libraries(ops_test ${CUDA_LIBRARIES})
endif()

target_link_libraries(ops_test pthread ${OpenCV_LIBS} ${OP_NAME})
target_link_libraries(ops_test gtest gtest_main gmock gmock_main)
gtest_discover_tests(ops_test)

install(
    FILES ${SOLUTION_DIR}/build/test/cpp/ops/ops_test
    DESTINATION ${BINARY_DIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ GROUP_EXECUTE GROUP_READ)
