#pragma once

#include "py_def.h"

template <typename T>
void py_fast_fc_kernel(
    const py::array_t<T, py::array::c_style | py::array::forcecast> ma,
    const py::array_t<T, py::array::c_style | py::array::forcecast> mb,
    const py::array_t<T, py::array::c_style | py::array::forcecast> mc,
    uint32_t K, 
    uint32_t M, 
    uint32_t N,
    py::array_t<T, py::array::c_style | py::array::forcecast> result
);