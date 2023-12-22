#pragma once

#include "py_def.h"

template <typename T>
py::array_t<T, py::array::c_style | py::array::forcecast> py_cpu_kernel_splice(
    py::array_t<T, py::array::c_style | py::array::forcecast> array,
    py_int32_t context,
    py_int32_t forward_indexs);

#ifdef WITH_CUDA
template <typename T>
py::array_t<T, py::array::c_style | py::array::forcecast> py_gpu_kernel_splice(
    py::array_t<T, py::array::c_style | py::array::forcecast> array,
    py_int32_t context,
    py_int32_t forward_indexs);
#endif