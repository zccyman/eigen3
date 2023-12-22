cmake_minimum_required(VERSION 3.12.0)
set(MY_VERSION, 0.1.0)

SET(CMAKE_BUILD_TYPE "Release")
#SET(CMAKE_BUILD_TYPE "Debug")

SET(CMAKE_CXX_STANDARD 17)
SET(CMAKE_CXX_FLAGS "-std=c++17 -O3")

if(WITH_CUDA)
    find_package(CUDA REQUIRED)

    #find_library(glog REQUIRED)
    if(CUDA_NVCC_EXECUTABLE AND NOT CUDA_VERSION)
        # Compute the version.
        execute_process(
            COMMAND ${CUDA_NVCC_EXECUTABLE} "--version"
            OUTPUT_VARIABLE NVCC_OUT
            RESULT_VARIABLE NVCC_RC)
        if(NOT (${NVCC_RC} EQUAL 0))
            message(
                WARNING "Failed to execute '${CUDA_NVCC_EXECUTABLE} --version'")
            set(CUDA_FOUND FALSE)
            return()
        endif()
        string(REGEX REPLACE ".*release ([0-9]+)\\.([0-9]+).*" "\\1"
                             CUDA_VERSION_MAJOR ${NVCC_OUT})
        string(REGEX REPLACE ".*release ([0-9]+)\\.([0-9]+).*" "\\2"
                             CUDA_VERSION_MINOR ${NVCC_OUT})
        set(CUDA_VERSION
            "${CUDA_VERSION_MAJOR}.${CUDA_VERSION_MINOR}"
            CACHE STRING "Version of CUDA as computed from nvcc.")
        mark_as_advanced(CUDA_VERSION)
    else()
        # Need to set these based off of the cached value
        string(REGEX REPLACE "([0-9]+)\\.([0-9]+).*" "\\1" CUDA_VERSION_MAJOR
                             "${CUDA_VERSION}")
        string(REGEX REPLACE "([0-9]+)\\.([0-9]+).*" "\\2" CUDA_VERSION_MINOR
                             "${CUDA_VERSION}")
    endif()

    # Always set this convenience variable
    set(CUDA_VERSION_STRING "${CUDA_VERSION}")

    if(USE_CUDNN)
        # Get cuDNN version
        if(EXISTS ${CUDNN_INCLUDE_PATH}/cudnn_version.h)
            file(READ ${CUDNN_INCLUDE_PATH}/cudnn_version.h
                 CUDNN_HEADER_CONTENTS)
        else()
            file(READ ${CUDNN_INCLUDE_PATH}/cudnn.h CUDNN_HEADER_CONTENTS)
        endif()
        string(REGEX MATCH "define CUDNN_MAJOR * +([0-9]+)" CUDNN_VERSION_MAJOR
                     "${CUDNN_HEADER_CONTENTS}")
        string(REGEX REPLACE "define CUDNN_MAJOR * +([0-9]+)" "\\1"
                             CUDNN_VERSION_MAJOR "${CUDNN_VERSION_MAJOR}")
        string(REGEX MATCH "define CUDNN_MINOR * +([0-9]+)" CUDNN_VERSION_MINOR
                     "${CUDNN_HEADER_CONTENTS}")
        string(REGEX REPLACE "define CUDNN_MINOR * +([0-9]+)" "\\1"
                             CUDNN_VERSION_MINOR "${CUDNN_VERSION_MINOR}")
        string(REGEX MATCH "define CUDNN_PATCHLEVEL * +([0-9]+)"
                     CUDNN_VERSION_PATCH "${CUDNN_HEADER_CONTENTS}")
        string(REGEX REPLACE "define CUDNN_PATCHLEVEL * +([0-9]+)" "\\1"
                             CUDNN_VERSION_PATCH "${CUDNN_VERSION_PATCH}")
        # Assemble cuDNN version
        if(NOT CUDNN_VERSION_MAJOR)
            set(CUDNN_VERSION "?")
        else()
            set(CUDNN_VERSION
                "${CUDNN_VERSION_MAJOR}.${CUDNN_VERSION_MINOR}.${CUDNN_VERSION_PATCH}"
            )
        endif()
        message(
            STATUS
                "Found cuDNN: v${CUDNN_VERSION}  (include: ${CUDNN_INCLUDE_PATH}, library: ${CUDNN_LIBRARY_PATH})"
        )
        if(CUDNN_VERSION VERSION_LESS "7.0.0")
            message(FATAL_ERROR "requires cuDNN 7 and above.")
        endif()
    endif()

    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
        message("embed_platform on")
        include_directories(/usr/local/cuda/targets/aarch64-linux/include)
        link_directories(/usr/local/cuda/targets/aarch64-linux/lib)
    else()
        message("embed_platform off")
        include_directories(/usr/local/cuda/include/)
        include_directories(/usr/include/)
        link_directories(/usr/local/cuda/lib64/)
        link_directories(/usr/lib/x86_64-linux-gnu/)
    endif()

    link_directories(${CUDNN_LIBRARY_PATH})
    set(CMAKE_CUDA_FLAGS
        "${CMAKE_CUDA_FLAGS}
        -gencode arch=compute_35,code=sm_35
        -gencode arch=compute_50,code=sm_50
        -gencode arch=compute_52,code=sm_52
        -gencode arch=compute_60,code=sm_60
        -gencode arch=compute_61,code=sm_61
        -gencode arch=compute_70,code=sm_70
        -gencode arch=compute_75,code=sm_75
        -gencode arch=compute_80,code=sm_80
        -gencode arch=compute_86,code=sm_86")
    set(CMAKE_CUDA_FLAGS
        "${CMAKE_CUDA_FLAGS} -Xcudafe \"--diag_suppress=bad_friend_decl\"")
    set(CMAKE_CUDA_FLAGS
        "${CMAKE_CUDA_FLAGS} -Xcudafe \"--diag_suppress=unsigned_compare_with_zero\""
    )
    set(CMAKE_CUDA_FLAGS
        "${CMAKE_CUDA_FLAGS} -Xcudafe \"--diag_suppress=expr_has_no_effect\"")
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
# message(%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%, ${CMAKE_CURRENT_SOURCE_DIR}/include)
# find_package(OpenCV REQUIRED)
# link_directories(${OpenCV_LIB_DIR})
add_compile_options(-std=c++17 -fPIC)
find_package(Eigen3 REQUIRED)
find_package(pybind11 REQUIRED)
# include_directories(${EIGEN3_INCLUDE_DIRS})
# message(${CMAKE_CURRENT_SOURCE_DIR})
# message(${EIGEN3_INCLUDE_DIRS})
include_directories(/usr/local/include/eigen3)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/platform)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY, ${LIBRARY_OUTPUT_PATH})

if(WITH_CUDA)
    link_directories(${CUDNN_LIBRARY_PATH})
    aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/src OP_SRCS)
    aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/platform OP_SRCS)
    cuda_add_library(${OP_NAME} SHARED ${OP_SRCS} ${platform_FILES})
    target_link_libraries(
        ${OP_NAME}
        pthread
        # ${OpenCV_LIBS}
        ${CUDA_LIBRARIES}
        cudart
        cudadevrt
        cublas
        curand
        cudnn
        nsync nsync_cpp)
else()
    file(GLOB OP_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc ${CMAKE_CURRENT_SOURCE_DIR}/platform/*.cc)
    add_library(${OP_NAME} SHARED ${OP_SRCS})# ${platform_FILES})
    target_link_libraries(${OP_NAME} 
    pthread nsync nsync_cpp
    # ${OpenCV_LIBS}
)
endif()

# message(${OP_SRCS})

if(${OPS_VERSION})
    set_target_properties(
        ${OP_NAME}
        PROPERTIES VERSION ${OP_VERSION}
                   SOVERSION "${OP_VERSION_MAJOR}.${OP_VERSION_MINOR}")
endif()
set_target_properties(${OP_NAME} PROPERTIES VERSION 0.1.0 SOVERSION 0)
# install(TARGETS ${OP_NAME} DESTINATION ${SOLUTION_DIR}/libs
#     LIBRARY OPTIONAL NAMELINK_SKIP DESTINATION ${SOLUTION_DIR}/libs COMPONENT RUNTIME
#     ARCHIVE DESTINATION libstatic)
install(
    TARGETS ${OP_NAME}
    LIBRARY
    RUNTIME DESTINATION ${LIBRARY_DIR}
    LIBRARY DESTINATION ${LIBRARY_DIR}
    ARCHIVE DESTINATION ${LIBRARY_DIR})
