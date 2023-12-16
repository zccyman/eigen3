#include "splice.h"
#include "py_splice.h"

template py::array_t<int8_t, py::array::c_style | py::array::forcecast>
py_cpu_kernel_splice(
    py::array_t<int8_t, py::array::c_style | py::array::forcecast> array,
    py_int32_t context,
    py_int32_t forward_indexs);
template py::array_t<int16_t, py::array::c_style | py::array::forcecast>
py_cpu_kernel_splice(
    py::array_t<int16_t, py::array::c_style | py::array::forcecast> array,
    py_int32_t context,
    py_int32_t forward_indexs);
template py::array_t<int32_t, py::array::c_style | py::array::forcecast>
py_cpu_kernel_splice(
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> array,
    py_int32_t context,
    py_int32_t forward_indexs);
template py::array_t<int64_t, py::array::c_style | py::array::forcecast>
py_cpu_kernel_splice(
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> array,
    py_int32_t context,
    py_int32_t forward_indexs);
template py::array_t<float, py::array::c_style | py::array::forcecast>
py_cpu_kernel_splice(
    py::array_t<float, py::array::c_style | py::array::forcecast> array,
    py_int32_t context,
    py_int32_t forward_indexs);
template py::array_t<double, py::array::c_style | py::array::forcecast>
py_cpu_kernel_splice(
    py::array_t<double, py::array::c_style | py::array::forcecast> array,
    py_int32_t context,
    py_int32_t forward_indexs);
template <typename T>
py::array_t<T, py::array::c_style | py::array::forcecast> py_cpu_kernel_splice(
    py::array_t<T, py::array::c_style | py::array::forcecast> array,
    py_int32_t context,
    py_int32_t forward_indexs)
{
    if (array.size() != 3) {
        pybind11::index_error("array must be equal three dims!");
    }
    std::vector<int> input_shape{(int)array.shape(0), (int)array.shape(1)},
        output_shape;
    int64_t alloc_t = array.size() * sizeof(T);
    T *input_data = (T *)malloc(alloc_t);
    std::memcpy(input_data, array.data(), alloc_t);
    std::vector<int32_t> out_shape, context_, forward_indexs_,
        forward_const_indexes_;
    for (auto i = 0; i < context.size(); i++)
        context_.push_back(int32_t(context.at(i)));
    for (auto i = 0; i < forward_indexs.size(); i++)
        forward_indexs_.push_back(int32_t(forward_indexs.at(i)));
    // std::cout << "forward_indexs_ size " << forward_indexs_.size() << std::endl;
    T *out = cpu_splice(input_shape, context_, forward_indexs_,
        forward_const_indexes_, out_shape, input_data, 0);
    // for(auto i=0; i<out_shape.size(); i++){
    //     std::cout << "in out shape: " << input_shape.at(i) << " " << out_shape.at(i) << std::endl;
    // }
    std::vector<ssize_t> shape = {out_shape[0], out_shape[1]};
    std::vector<ssize_t> strides = {
        // static_cast<int64_t>(out_shape[1] * out_shape[2] * sizeof(T)),
        static_cast<int64_t>(out_shape[1] * sizeof(T)),
        static_cast<int64_t>(sizeof(T))};

    py::array_t<T, py::array::c_style | py::array::forcecast> res =
        py::array_t<T, py::array::c_style | py::array::forcecast>(
            py::buffer_info(out, sizeof(T), py::format_descriptor<T>::format(),
                out_shape.size(), shape, strides));
    free(input_data);
    free(out);

    return res;
}

#ifdef WITH_CUDA
template py::array_t<int8_t, py::array::c_style | py::array::forcecast>
py_gpu_kernel_splice(
    py::array_t<int8_t, py::array::c_style | py::array::forcecast> array,
    py_int32_t context,
    py_int32_t forward_indexs);
template py::array_t<int16_t, py::array::c_style | py::array::forcecast>
py_gpu_kernel_splice(
    py::array_t<int16_t, py::array::c_style | py::array::forcecast> array,
    py_int32_t context,
    py_int32_t forward_indexs);
template py::array_t<int32_t, py::array::c_style | py::array::forcecast>
py_gpu_kernel_splice(
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> array,
    py_int32_t context,
    py_int32_t forward_indexs);
template py::array_t<int64_t, py::array::c_style | py::array::forcecast>
py_gpu_kernel_splice(
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> array,
    py_int32_t context,
    py_int32_t forward_indexs);
template py::array_t<float, py::array::c_style | py::array::forcecast>
py_gpu_kernel_splice(
    py::array_t<float, py::array::c_style | py::array::forcecast> array,
    py_int32_t context,
    py_int32_t forward_indexs);
template py::array_t<double, py::array::c_style | py::array::forcecast>
py_gpu_kernel_splice(
    py::array_t<double, py::array::c_style | py::array::forcecast> array,
    py_int32_t context,
    py_int32_t forward_indexs);
template <typename T>
py::array_t<T, py::array::c_style | py::array::forcecast> py_gpu_kernel_splice(
    py::array_t<T, py::array::c_style | py::array::forcecast> array,
    py_int32_t context,
    py_int32_t forward_indexs)
{
    if (array.size() != 3) {
        pybind11::index_error("array must be equal three dims!");
    }
    std::vector<int> input_shape{(int)array.shape(0), (int)array.shape(1)},
        output_shape;
    int64_t alloc_t = array.size() * sizeof(T);
    T *input_data = (T *)malloc(alloc_t);
    std::memcpy(input_data, array.data(), alloc_t);
    std::vector<int32_t> out_shape, context_, forward_indexs_,
        forward_const_indexes_;
    for (auto i = 0; i < context.size(); i++)
        context_.push_back(int32_t(context.at(i)));
    for (auto i = 0; i < forward_indexs.size(); i++)
        forward_indexs_.push_back(int32_t(forward_indexs.at(i)));
    // std::cout << "forward_indexs_ size " << forward_indexs_.size() << std::endl;
    T *out = gpu_splice(input_shape, context_, forward_indexs_,
        forward_const_indexes_, out_shape, input_data, 0);
    // for(auto i=0; i<out_shape.size(); i++){
    //     std::cout << "in out shape: " << input_shape.at(i) << " " << out_shape.at(i) << std::endl;
    // }
    std::vector<ssize_t> shape = {out_shape[0], out_shape[1]};
    std::vector<ssize_t> strides = {
        static_cast<int64_t>(out_shape[1] * sizeof(T)),
        static_cast<int64_t>(sizeof(T))};
    return py::array_t<T, py::array::c_style | py::array::forcecast>(
        py::buffer_info(out, sizeof(T), py::format_descriptor<T>::format(),
            out_shape.size(), shape, strides));
}
#endif