# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# add_definitions(-DOpDomain="${OpDomain}")
include_directories(/usr/local/include/eigen3)
add_library(
    custom_op_library_cpu SHARED
    # ${CUSTOMOPLIB_ROOT}/custom_op_library_cpu.cc
    ${CUSTOMOPLIB_ROOT}/custom_op_library.cc
    ${CUSTOMOPLIB_ROOT}/custom_deformconv.cc
    ${CUSTOMOPLIB_ROOT}/custom_splice.cc
    ${CUSTOMOPLIB_ROOT}/register_custom_ops.cc
    ${CUSTOMOPLIB_ROOT}/register_custom_ops.h)

if(WITH_CUDA)
    find_package(CUDA REQUIRED)
    include_directories(${CUDA_INCLUDE_DIRS})
    link_directories(${CUDNN_LIBRARY_PATH})
    cuda_add_library(
        custom_op_library_cuda
        SHARED
        # ${CUSTOMOPLIB_ROOT}/custom_op_library_cpu.cc
        ${CUSTOMOPLIB_ROOT}/custom_op_library.cc
        ${CUSTOMOPLIB_ROOT}/custom_deformconv.cc
        ${CUSTOMOPLIB_ROOT}/custom_splice.cc
        ${CUSTOMOPLIB_ROOT}/register_custom_ops.cc
        ${CUSTOMOPLIB_ROOT}/register_custom_ops.h)
endif()

# set_target_properties(
#   custom_op_library_fp32 custom_op_library_fp16
#   PROPERTIES CUDA_SEPARABLE_COMPILATION ON
# )

if(UNIX)
    if(APPLE)
        set(ONNXRUNTIME_CUSTOM_OP_LIB_LINK_FLAG "-Xlinker -dead_strip")
    else()
        set(ONNXRUNTIME_CUSTOM_OP_LIB_LINK_FLAG
            "-Xlinker --version-script=${CUSTOMOPLIB_ROOT}/custom_op_library.lds -Xlinker --no-undefined -Xlinker --gc-sections -z noexecstack"
        )
    endif()
else()
    set(ONNXRUNTIME_CUSTOM_OP_LIB_LINK_FLAG
        "-DEF:${CUSTOMOPLIB_ROOT}/custom_op_library.def")
endif()
message(${ONNXRUNTIME_CUSTOM_OP_LIB_LINK_FLAG})
set_property(
    TARGET custom_op_library_cpu
    APPEND_STRING
    PROPERTY LINK_FLAGS ${ONNXRUNTIME_CUSTOM_OP_LIB_LINK_FLAG})

target_link_libraries(custom_op_library_cpu pthread onnxruntime)
# install(FILES ${SOLUTION_DIR}/build/ops/ort_extension/custom_op_library/libcustom_op_library_cpu.so
#         DESTINATION ${SOLUTION_DIR}/libs)
if(WITH_CUDA)
    target_link_libraries(
        custom_op_library_cuda
        ${CUDA_LIBRARIES}
        cudart
        cudadevrt
        cublas
        curand
        cudnn)
    # install(FILES ${SOLUTION_DIR}/build/ops/ort_extension/custom_op_library/libcustom_op_library_cuda.so
    #         DESTINATION ${SOLUTION_DIR}/libs)
endif()

# INSTALL(DIRECTORY ${SOLUTION_DIR}/build/ort_package/runtimes/linux-x64/native/
#         DESTINATION ${SOLUTION_DIR}/libs)
