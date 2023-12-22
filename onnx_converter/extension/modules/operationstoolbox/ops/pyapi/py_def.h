#pragma once

//#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/eval.h>
#include <Python.h>
#include <typeinfo>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <iostream>
#include <vector>

namespace py = pybind11;

using py_int8_t = py::array_t<int8_t, py::array::c_style | py::array::forcecast>;
using py_uint8_t = py::array_t<uint8_t, py::array::c_style | py::array::forcecast>;
using py_int16_t =
    py::array_t<int16_t, py::array::c_style | py::array::forcecast>;
using py_int32_t =
    py::array_t<int32_t, py::array::c_style | py::array::forcecast>;
using py_int64_t =
    py::array_t<int64_t, py::array::c_style | py::array::forcecast>;
using py_float_t = py::array_t<float, py::array::c_style | py::array::forcecast>;
using py_double_t =
    py::array_t<double, py::array::c_style | py::array::forcecast>;

template<typename T>
using py_vector = py::array_t<T, py::array::c_style | py::array::forcecast>;

template<typename T>
void copy_py_to_cpp(const py_vector<T>& input, std::vector<T>& output){
    py::buffer_info bufA = input.request();
    const int sizeA = bufA.size;
    const int sizeC = sizeA;
    output = std::vector<T>((T *)bufA.ptr, (T *)bufA.ptr + sizeA);
}

template<typename T>
void copy_cpp_to_py(const std::vector<T>& input, py_vector<T>& output){
    // std::vector<ssize_t> shape;
    output(input.size());
    size_t length = 0;
    for (size_t i=0; i<input.size(); i++){
        output[i] = input[i];
        length += input[i];
    }
}

/*template<typename T>
size_t mem_size(const py_vector<T>& shape){
    size_t length = 0;
    for (size_t i=0; i<shape.size(); i++){
        length += static_cast<size_t>(shape[i]);
    }
    return length * sizeof(T);
}*/

template<typename T>
size_t mem_size(const std::vector<T>& shape){
    size_t length = 1;
    for (size_t i=0; i<shape.size(); i++){
        length *= static_cast<size_t>(shape[i]);
    }
    return length;// * sizeof(T);
}