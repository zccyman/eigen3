
#include "py_def.h"

/*template<typename T>
void copy_py_to_cpp(const py_vector<T>& input, std::vector<T>& output){
    py::buffer_info bufA = input.request();
    const int sizeA = bufA.size;
    const int sizeC = sizeA;
    output = std::vector<T>((T *)bufA.ptr, (T *)bufA.ptr + sizeA);
}*/

/*template<typename T>
void copy_cpp_to_py(const std::vector<T>& input, py_vector<T>& output){
    
    std::vector<ssize_t> shape;
    for (size_t i=0; i<output.size(); i++){
        shape.push_back(output)
    }
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
}*/