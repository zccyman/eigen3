#include "py_ops.h"
#include "py_eigen_ops.h"

template py::array_t<int> pyops::add<int>(const py::array_t<int> &a,
    const py::array_t<int> &b,
    const std::vector<int> &size);
template py::array_t<float> pyops::add<float>(const py::array_t<float> &a,
    const py::array_t<float> &b,
    const std::vector<int> &size);
template py::array_t<double> pyops::add<double>(const py::array_t<double> &a,
    const py::array_t<double> &b,
    const std::vector<int> &size);
template <typename T>  // test template funcation
py::array_t<T> pyops::add(const py::array_t<T> &a,
    const py::array_t<T> &b,
    const std::vector<int> &size)
{
    py::buffer_info bufA = a.request();
    py::buffer_info bufB = b.request();

    const int sizeA = bufA.size;
    const int sizeB = bufB.size;
    const int sizeC = sizeA;
    std::vector<T> A((T *)bufA.ptr, (T *)bufA.ptr + sizeA);
    std::vector<T> B((T *)bufB.ptr, (T *)bufB.ptr + sizeB);
    std::vector<T> C(A.begin(), A.end());

    for (size_t i = 0; i < sizeA; ++i) {
        C[i] = A[i] + B[i];
    }

    std::vector<int> strides;
    strides.insert(strides.begin(), sizeof(T));
    int stride = 1;
    for (size_t i = 0; i < size.size() - 1; ++i) {
        stride *= size[i];
        strides.insert(strides.begin(), stride * sizeof(T));
    }

    return py::array_t<T>(size,  // {100, 1000, 1000}
        strides,  // {1000 * 1000 * sizeof(T), 1000 * sizeof(T), sizeof(T)},
        C.data());
}

template py::array_t<int8_t> pyops::py_cpu_splice(
    py::array_t<int8_t> array, py_int32_t context, py_int32_t forward_indexs);
template py::array_t<int16_t> pyops::py_cpu_splice(
    py::array_t<int16_t> array, py_int32_t context, py_int32_t forward_indexs);
template py::array_t<int32_t> pyops::py_cpu_splice(
    py::array_t<int32_t> array, py_int32_t context, py_int32_t forward_indexs);
template py::array_t<int64_t> pyops::py_cpu_splice(
    py::array_t<int64_t> array, py_int32_t context, py_int32_t forward_indexs);
template py::array_t<float> pyops::py_cpu_splice(
    py::array_t<float> array, py_int32_t context, py_int32_t forward_indexs);
template py::array_t<double> pyops::py_cpu_splice(
    py::array_t<double> array, py_int32_t context, py_int32_t forward_indexs);
template <typename T>  // test template funcation
py::array_t<T> pyops::py_cpu_splice(
    py::array_t<T> array, py_int32_t context, py_int32_t forward_indexs)
{
    return py_cpu_kernel_splice<T>(array, context, forward_indexs);
}

#ifdef WITH_CUDA
template py::array_t<int8_t> pyops::py_gpu_splice(
    py::array_t<int8_t> array, py_int32_t context, py_int32_t forward_indexs);
template py::array_t<int16_t> pyops::py_gpu_splice(
    py::array_t<int16_t> array, py_int32_t context, py_int32_t forward_indexs);
template py::array_t<int32_t> pyops::py_gpu_splice(
    py::array_t<int32_t> array, py_int32_t context, py_int32_t forward_indexs);
template py::array_t<int64_t> pyops::py_gpu_splice(
    py::array_t<int64_t> array, py_int32_t context, py_int32_t forward_indexs);
template py::array_t<float> pyops::py_gpu_splice(
    py::array_t<float> array, py_int32_t context, py_int32_t forward_indexs);
template py::array_t<double> pyops::py_gpu_splice(
    py::array_t<double> array, py_int32_t context, py_int32_t forward_indexs);
template <typename T>  // test template funcation
py::array_t<T> pyops::py_gpu_splice(
    py::array_t<T> array, py_int32_t context, py_int32_t forward_indexs)
{
    return py_gpu_kernel_splice<T>(array, context, forward_indexs);
}
#endif


template void pyops::py_eigen_fc(
        const py::array_t<float, py::array::c_style | py::array::forcecast> ma,
    const py::array_t<float, py::array::c_style | py::array::forcecast> mb,
    const py::array_t<float, py::array::c_style | py::array::forcecast> mc,
    uint32_t K, 
    uint32_t M, 
    uint32_t N,
    py::array_t<float, py::array::c_style | py::array::forcecast> result);

template void pyops::py_eigen_fc(
        const py::array_t<double, py::array::c_style | py::array::forcecast> ma,
    const py::array_t<double, py::array::c_style | py::array::forcecast> mb,
    const py::array_t<double, py::array::c_style | py::array::forcecast> mc,
    uint32_t K, 
    uint32_t M, 
    uint32_t N,
    py::array_t<double, py::array::c_style | py::array::forcecast> result);
   
template <typename T>  // test template funcation
void pyops::py_eigen_fc(
    const py::array_t<T, py::array::c_style | py::array::forcecast> ma,
    const py::array_t<T, py::array::c_style | py::array::forcecast> mb,
    const py::array_t<T, py::array::c_style | py::array::forcecast> mc,
    uint32_t K, 
    uint32_t M, 
    uint32_t N,
    py::array_t<T, py::array::c_style | py::array::forcecast> result)
{
    py_fast_fc_kernel<T>(ma, mb, mc, K, M, N, result);
}

/*template<typename T>
void copy_py_to_cpp(py_vector<T>& input, std::vector<T>& output){
    py::buffer_info bufA = input.request();
    const int sizeA = bufA.size;
    const int sizeC = sizeA;
    output = std::vector<T>((T *)bufA.ptr, (T *)bufA.ptr + sizeA);
}*/

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

#ifdef WITH_CUDA

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

#endif

py_resize_op_int8::py_resize_op_int8(const py_int32_t& in_shape, const py_int32_t& out_shape, const py_float_t& scale, int bilinear_mode, bool is_nchw, int coord_mode){
    //std:vector<int32_t> in_shape_v, out_shape_v;
    //std::vector<float> scale_v;
    copy_py_to_cpp<int32_t>(in_shape, this->in_shape_v);
    // std::cout << "1RRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR\n";
    copy_py_to_cpp<int32_t>(out_shape, this->out_shape_v);
    // std::cout << "2RRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR\n";
    copy_py_to_cpp<float>(scale, this->scale_v);
    // std::cout << "3RRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR\n";
    bilinear_mode_ = bilinear_mode;
    // std::cout << "4RRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR\n";
    is_nchw_ = is_nchw;
    coord_mode_ = coord_mode;
    // std::cout << "5RRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR\n";
    UpsampleMode upsample_mode_ = UpsampleMode::LINEAR;
    // std::cout << "6RRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR\n";
    ResizeCoordinateTransformationMode tmp = ResizeCoordinateTransformationMode(coord_mode_);
    // std::cout << "7RRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR\n";

    this->resize_op_ = new TimesIntelliResize<int8_t>(this->in_shape_v, this->out_shape_v, this->scale_v, upsample_mode_, tmp, nullptr);
    // std::cout << "8RRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR\n";
    this->resize_op_->TimesIntelliInit((BilinearMode)this->bilinear_mode_, this->is_nchw_);
    // std::cout << "9RRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR\n";

}

py_resize_op_int8::~py_resize_op_int8(){
    if (resize_op_){
        delete resize_op_;
        resize_op_ = nullptr;
    }
}

void py_resize_op_int8::forward(const py_int8_t& X, py_int8_t& Y, py_int32_t& in_shape){
    //std::cout << "test#############################!\n";
    std::vector<int32_t> input_shape;
    // for(size_t i=0; i<out_shape_v.size(); i++){
    //     std::cout << "i is: " << i << ", shape is: "  << out_shape_v[i] << std::endl; 
    // }
    
    // for(size_t i=0; i<out_shape_v.size(); i++){
    //     std::cout << "i is: " << i << ", shape is: "  << out_shape_v[i] << std::endl; 
    // }
    copy_py_to_cpp(in_shape, input_shape);
    out_shape_v[0] = input_shape[0];
    // for(size_t i=0; i<input_shape.size(); i++){
    //     std::cout << "i is: " << i << ", shape is: "  << input_shape[i] << std::endl; 
    // }
    // std::cout << "test!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n";
    size_t in_size = mem_size(input_shape) * sizeof(int8_t);
    size_t out_size = mem_size(out_shape_v) * sizeof(int8_t);
    // std::cout << "in_size is: " << in_size << " out_size is: " << out_size << std::endl;
    int8_t* src = (int8_t*)malloc(in_size);
    int8_t* dst = (int8_t*)malloc(out_size);
    memcpy((void*)src, (void*)X.data(), in_size);
    this->resize_op_->TimesIntelliForward(src, input_shape, dst);
    memcpy((void*)Y.data(), dst, out_size);
    free(src);
    free(dst);
    //std::cout << "test done $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$!\n";
}

void py_resize_op_int8::operator()(const py_int8_t& X, py_int8_t& Y, py_int32_t& in_shape){
    this->forward(X, Y, in_shape);
}

py_resize_op_uint8::py_resize_op_uint8(const py_int32_t& in_shape, const py_int32_t& out_shape, const py_float_t& scale, int bilinear_mode, bool is_nchw, int coord_mode){
    //std:vector<int32_t> in_shape_v, out_shape_v;
    //std::vector<float> scale_v;
    copy_py_to_cpp<int32_t>(in_shape, this->in_shape_v);
    // std::cout << "1RRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR\n";
    copy_py_to_cpp<int32_t>(out_shape, this->out_shape_v);
    // std::cout << "2RRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR\n";
    copy_py_to_cpp<float>(scale, this->scale_v);
    //std::cout << "3RRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR\n";
    bilinear_mode_ = bilinear_mode;
    //std::cout << "4RRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR\n";
    is_nchw_ = is_nchw;
    coord_mode_ = coord_mode;
    //std::cout << "5RRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR\n";
    UpsampleMode upsample_mode_ = UpsampleMode::LINEAR;
    //std::cout << "6RRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR\n";
    ResizeCoordinateTransformationMode tmp = ResizeCoordinateTransformationMode(coord_mode_);
    //std::cout << "7RRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR\n";

    this->resize_op_ = new TimesIntelliResize<uint8_t>(this->in_shape_v, this->out_shape_v, this->scale_v, upsample_mode_, tmp, nullptr);
    this->resize_op_->TimesIntelliInit((BilinearMode)this->bilinear_mode_, this->is_nchw_);

}

py_resize_op_uint8::~py_resize_op_uint8(){
    std::cout << "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n";
    std::cout << "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n";
    std::cout << "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n";
    std::cout << "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n";
    std::cout << "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n";
    std::cout << "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n";
    std::cout << "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n";
    std::cout << "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n";
    std::cout << "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n";
    std::cout << "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n";
    std::cout << "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n";
    std::cout << "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n";
    std::cout << "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n";
    std::cout << "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n";
    std::cout << "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n";
    std::cout << "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n";
    std::cout << "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n";
    std::cout << "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n";
    std::cout << "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n";
    std::cout << "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n";
    std::cout << "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n";
    std::cout << "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n";
    std::cout << "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n";
    if (resize_op_){
        delete resize_op_;
        resize_op_ = nullptr;
    }
}

void py_resize_op_uint8::forward(const py_uint8_t& X, py_uint8_t& Y, py_int32_t& in_shape){
    //std::cout << "test#############################!\n";
    std::vector<int32_t> input_shape;
    // for(size_t i=0; i<out_shape_v.size(); i++){
    //     std::cout << "i is: " << i << ", shape is: "  << out_shape_v[i] << std::endl; 
    // }
    
    // for(size_t i=0; i<out_shape_v.size(); i++){
    //     std::cout << "i is: " << i << ", shape is: "  << out_shape_v[i] << std::endl; 
    // }
    copy_py_to_cpp(in_shape, input_shape);
    out_shape_v[0] = input_shape[0];
    // for(size_t i=0; i<input_shape.size(); i++){
    //     std::cout << "i is: " << i << ", shape is: "  << input_shape[i] << std::endl; 
    // }
    // std::cout << "test!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n";
    size_t in_size = mem_size(input_shape) * sizeof(uint8_t);
    size_t out_size = mem_size(out_shape_v) * sizeof(uint8_t);
    //std::cout << "in_size is: " << in_size << " out_size is: " << out_size << std::endl;
    uint8_t* src = (uint8_t*)malloc(in_size);
    uint8_t* dst = (uint8_t*)malloc(out_size);
    memcpy((void*)src, (void*)X.data(), in_size);
    this->resize_op_->TimesIntelliForward(src, input_shape, dst);
    memcpy((void*)Y.data(), dst, out_size);
    free(src);
    free(dst);
    // std::cout << "test done $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$!\n";
}

void py_resize_op_uint8::operator()(const py_uint8_t& X, py_uint8_t& Y, py_int32_t& in_shape){
    this->forward(X, Y, in_shape);
}

py_resize_op_int16::py_resize_op_int16(const py_int32_t& in_shape, const py_int32_t& out_shape, const py_float_t& scale, int bilinear_mode, bool is_nchw, int coord_mode){
    //std:vector<int32_t> in_shape_v, out_shape_v;
    //std::vector<float> scale_v;
    copy_py_to_cpp<int32_t>(in_shape, this->in_shape_v);
    copy_py_to_cpp<int32_t>(out_shape, this->out_shape_v);
    copy_py_to_cpp<float>(scale, this->scale_v);
    bilinear_mode_ = bilinear_mode;
    is_nchw_ = is_nchw;
    coord_mode_ = coord_mode;
    UpsampleMode upsample_mode_ = UpsampleMode::LINEAR;
    ResizeCoordinateTransformationMode tmp = ResizeCoordinateTransformationMode(coord_mode_);

    this->resize_op_ = new TimesIntelliResize<int16_t>(this->in_shape_v, this->out_shape_v, this->scale_v, upsample_mode_, tmp, nullptr);
    this->resize_op_->TimesIntelliInit((BilinearMode)this->bilinear_mode_, this->is_nchw_);

}

py_resize_op_int16::~py_resize_op_int16(){
    if (resize_op_){
        delete resize_op_;
        resize_op_ = nullptr;
    }
}

void py_resize_op_int16::forward(const py_int16_t& X, py_int16_t& Y, py_int32_t& in_shape){
    
    //std::cout << "test#############################!\n";
    std::vector<int32_t> input_shape;
    // for(size_t i=0; i<out_shape_v.size(); i++){
    //     std::cout << "i is: " << i << ", shape is: "  << out_shape_v[i] << std::endl; 
    // }
    
    // for(size_t i=0; i<out_shape_v.size(); i++){
    //     std::cout << "i is: " << i << ", shape is: "  << out_shape_v[i] << std::endl; 
    // }
    copy_py_to_cpp(in_shape, input_shape);
    out_shape_v[0] = input_shape[0];
    // for(size_t i=0; i<input_shape.size(); i++){
    //     std::cout << "i is: " << i << ", shape is: "  << input_shape[i] << std::endl; 
    // }
    // std::cout << "test!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n";
    size_t in_size = mem_size(input_shape) * sizeof(int16_t);
    size_t out_size = mem_size(out_shape_v) * sizeof(int16_t);
    // std::cout << "in_size is: " << in_size << " out_size is: " << out_size << std::endl;
    int16_t* src = (int16_t*)malloc(in_size);
    int16_t* dst = (int16_t*)malloc(out_size);
    memcpy((void*)src, (void*)X.data(), in_size);
    this->resize_op_->TimesIntelliForward(src, input_shape, dst);
    memcpy((void*)Y.data(), (void*)dst, out_size);
    free(src);
    free(dst);
    // std::cout << "test done $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$!\n";
}

void py_resize_op_int16::operator()(const py_int16_t& X, py_int16_t& Y, py_int32_t& in_shape){
    this->forward(X, Y, in_shape);
}

// template<typename T>
// using py_reize_op_type = class py_resize_op<T>;

// template class py_resize_op<int8_t>;
// template class py_resize_op<uint8_t>;
// template class py_resize_op<int16_t>;
// template class py_resize_op<uint16_t>;