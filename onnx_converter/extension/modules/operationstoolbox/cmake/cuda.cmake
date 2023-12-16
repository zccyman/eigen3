add_compile_options(-std=c++17 -fPIC)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -rdc=true)
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
