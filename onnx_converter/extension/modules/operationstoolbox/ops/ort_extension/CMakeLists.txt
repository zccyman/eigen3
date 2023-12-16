# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# Minimum CMake required
cmake_minimum_required(VERSION 3.12)
project(ort_custom)

set(ORT_LIB_NAME "ort_package")
set(ORT_LIB_UNZIP_DIR_NAME ${ORT_LIB_NAME})
set(LOCAL_ORT_LIB_ZIP_DL_DIR "ort_package")
set(LOCAL_ORT_INCLUDE_DIR_NAME "ort_include")
set(ORT_LIB_ZIP_NAME "ort_lib.tar.gz")

if(CMAKE_SYSTEM_NAME MATCHES "Linux")
    set(ORT_LIB_DOWNLOAD_URL
        "https://gitee.com/zccyman/onnxruntime-public/releases/download/v1.12.1/onnxruntime-linux-x64-1.12.1.tgz"
        )
elseif(CMAKE_SYSTEM_NAME MATCHES "Windows")
    set(ORT_LIB_DOWNLOAD_URL
    "https://gitee.com/zccyman/onnxruntime-public/releases/download/v1.12.1/onnxruntime-win-x64-1.12.1.zip"
    )
endif()
set(DOWNLOAD_ORT_LIB_TIMEOUT
    1000
    CACHE STRING "Timeout in seconds when downloading ORT_lib.")

include(${SOLUTION_DIR}/cmake/download_onnx_lib.cmake)

set(CUSTOMOPLIB_ROOT ${PROJECT_SOURCE_DIR}/custom_op_library)
# set(ORTPACKAGE_ROOT ${SOLUTION_DIR}/build/ort_package)
# set(ORTPACKAGE_INCLUDE ${ORTPACKAGE_ROOT}/build/native/include)
#set(ORTPACKAGE_ROOT ${LIBRARY_DIR}/onnxruntime)
set(ORTPACKAGE_ROOT ${LIBRARY_DIR})
set(ORTPACKAGE_INCLUDE ${ORTPACKAGE_ROOT}/include)
# set(ORTPACKAGE_INCLUDE /workspace/ssy/onnxruntime/include/onnxruntime/core/session)
set(ORT_HEADER_SRC
    "${ORTPACKAGE_INCLUDE}/onnxruntime_c_api.h"
    "${ORTPACKAGE_INCLUDE}/onnxruntime_cxx_api.h"
    "${ORTPACKAGE_INCLUDE}/onnxruntime_cxx_inline.h")

# file(GLOB CUSTOMOPLIB_SRC
#     "${CUSTOMOPLIB_ROOT}/*.h"
#     "${CUSTOMOPLIB_ROOT}/*.cc"
#     "${CUSTOMOPLIB_ROOT}/*.cu"
#      "${CUSTOMOPLIB_ROOT}/*.lds"
#      "${CUSTOMOPLIB_ROOT}/*.def"
# )
if(WITH_CUDA)
    find_package(CUDA REQUIRED)
    include(${SOLUTION_DIR}/cmake/cuda.cmake)
    link_directories(${CUDNN_LIBRARY_PATH})
endif()

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIRS})

include_directories(${ORTPACKAGE_ROOT})
include_directories(${ORTPACKAGE_INCLUDE})
include_directories(${CUSTOMOPLIB_ROOT})
include_directories(/usr/local/include/eigen3)

if(CMAKE_SYSTEM_NAME MATCHES "Linux")
    # message(${ORTPACKAGE_ROOT})
    link_directories(${ORTPACKAGE_ROOT})
    link_directories(/usr/lib/x86_64-linux-gnu/)
elseif(CMAKE_SYSTEM_NAME MATCHES "Windows")
    link_directories(${ORTPACKAGE_ROOT})
endif()

#set(CMAKE_SKIP_BUILD_RPATH FALSE)                 # 编译时加上RPATH
#set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)         # 编译时RPATH不使用安装的RPATH
#set(CMAKE_INSTALL_RPATH "./onnxruntime/lib")                       # 安装RPATH为当前目录
#set(CMAKE_INSTALL_RPATH_USE_LINK_PATH FALSE)      # 安装的执行文件不加上RPATH

add_subdirectory(custom_op_library)
link_directories(${CMAKE_CURRENT_BINARY_DIR}/custom_op_library)
add_executable(${PROJECT_NAME} ${SOLUTION_DIR}/test/cpp/ort_ops/test_add.cc)
if(CMAKE_SYSTEM_NAME MATCHES "Linux")
    target_include_directories(
        ${PROJECT_NAME} PRIVATE ${ORTPACKAGE_ROOT}/include)
elseif(CMAKE_SYSTEM_NAME MATCHES "Windows")
    target_include_directories(
        ${PROJECT_NAME} PRIVATE ${ORTPACKAGE_ROOT}/include)
endif()

if(WITH_CUDA)
    target_link_libraries(${PROJECT_NAME} PUBLIC custom_op_library_cuda gtest
                                                 onnxruntime)
else()
    target_link_libraries(${PROJECT_NAME} PUBLIC custom_op_library_cpu gtest
                                                 onnxruntime)
endif()

set_target_properties(${PROJECT_NAME}  PROPERTIES LINK_FLAGS "-Wl,-rpath=./onnxruntime/lib")

install(
    FILES ${SOLUTION_DIR}/build/ops/ort_extension/ort_custom
    DESTINATION ${BINARY_DIR}
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ GROUP_EXECUTE GROUP_READ)
