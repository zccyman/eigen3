cmake_minimum_required(VERSION 3.5.1)

project(kld) 

set(QUANTIZE_NAME kld)
include_directories(/usr/local/include/eigen3)
set(SOLUTION_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(LIBRARY_DIR ${SOLUTION_DIR}/../../libs)
set(BINARY_DIR ${SOLUTION_DIR}/../../bins)
include_directories(/usr/local/include/eigen3)
# SET(pybind11_DIR /home/ts300026/anaconda3/envs/converter_py37/lib/python3.7/site-packages/pybind11/share/cmake/pybind11)

find_package(pybind11)
pybind11_add_module(${QUANTIZE_NAME} src/kld.cpp)
SET( CMAKE_CXX_FLAGS "-std=c++11 -O3")

install(
    TARGETS ${QUANTIZE_NAME}
    LIBRARY
    RUNTIME DESTINATION ${LIBRARY_DIR}
    LIBRARY DESTINATION ${LIBRARY_DIR}
    ARCHIVE DESTINATION ${LIBRARY_DIR})