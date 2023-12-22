cmake_minimum_required(VERSION 3.12)
project(pyops)
set(PYOPS_VERSION 0.1.0)

if(${WITH_CUDA} STREQUAL ON)
    add_compile_definitions(WITH_CUDA)
endif()

find_package(pybind11 REQUIRED)
include_directories(/usr/local/include/eigen3)
# include_directories(${SOLUTION_DIR}/ops/core/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../core/)
link_directories(${OpenCV_LIB_DIR})
if(WITH_CUDA)
    link_directories(${CUDNN_LIBRARY_PATH})
endif()

pybind11_add_module(${PROJECT_NAME} SHARED py_ops.cc pyapi.cc py_eigen_ops.cc py_def.cc)

target_link_libraries(${PROJECT_NAME} PRIVATE ${OP_NAME} pthread ${OpenCV_LIBS})
if(WITH_CUDA)
    target_link_libraries(${PROJECT_NAME} PRIVATE ${CUDA_LIBRARIES} cudart
                                                  cudadevrt cublas curand cudnn)
endif()

set_target_properties(${PROJECT_NAME} PROPERTIES VERSION ${PYOPS_VERSION})
install(
    FILES
        ${SOLUTION_DIR}/build/ops/pyapi/${PROJECT_NAME}.cpython-37m-x86_64-linux-gnu.so
    DESTINATION ${LIBRARY_DIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ GROUP_EXECUTE GROUP_READ)
install(
    FILES
        ${SOLUTION_DIR}/build/ops/pyapi/${PROJECT_NAME}.cpython-37m-x86_64-linux-gnu.so.${PYOPS_VERSION}
    DESTINATION ${LIBRARY_DIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ GROUP_EXECUTE GROUP_READ)
