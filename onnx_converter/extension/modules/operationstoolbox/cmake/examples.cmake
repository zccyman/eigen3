cmake_minimum_required(VERSION 3.12)
set(LIBRARY_OUTPUT_PATH ${LIBRARY_DIR})
aux_source_directory(${PROJECT_SOURCE_DIR}/test/cpp/ops/example/module
                     module_cpp)
include_directories(${PROJECT_SOURCE_DIR}/test/cpp/ops/example/module)
add_library(examples SHARED ${module_cpp})
