#include "py_ops.h"

PYBIND11_MODULE(pyops, m)
{
    m.doc() = "pybind11 pyops";

    py::class_<pyops>(m, "pyops")
        .def(py::init<>())
#ifdef WITH_CUDA
        .def("py_gpu_splice", &pyops::py_gpu_splice<int8_t>)
        .def("py_gpu_splice", &pyops::py_gpu_splice<int16_t>)
        .def("py_gpu_splice", &pyops::py_gpu_splice<int32_t>)
        .def("py_gpu_splice", &pyops::py_gpu_splice<int64_t>)
        .def("py_gpu_splice", &pyops::py_gpu_splice<float>)
        .def("py_gpu_splice", &pyops::py_gpu_splice<double>)
#endif  // WITH_CUDA
        .def("add", &pyops::add<int>)
        .def("add", &pyops::add<float>)
        .def("add", &pyops::add<double>)
        .def("py_cpu_splice", &pyops::py_cpu_splice<int8_t>)
        .def("py_cpu_splice", &pyops::py_cpu_splice<int16_t>)
        .def("py_cpu_splice", &pyops::py_cpu_splice<int32_t>)
        .def("py_cpu_splice", &pyops::py_cpu_splice<int64_t>)
        .def("py_cpu_splice", &pyops::py_cpu_splice<float>)
        .def("py_cpu_splice", &pyops::py_cpu_splice<double>)
        .def("py_eigen_fc", &pyops::py_eigen_fc<float>)
        .def("py_eigen_fc", &pyops::py_eigen_fc<double>);

    py::class_<py_resize_op_int8>(m, "py_resize_op_int8")
        //.def(py::init<py_int32_t, py_int32_t, py_float_t>())
        .def(py::init<py_int32_t, py_int32_t, py_float_t, int, bool, int>())
        .def("forward", &py_resize_op_int8::forward);

    py::class_<py_resize_op_uint8>(m, "py_resize_op_uint8")
        //.def(py::init<py_int32_t, py_int32_t, py_float_t>())
        .def(py::init<py_int32_t, py_int32_t, py_float_t, int, bool, int>())
        .def("forward", &py_resize_op_uint8::forward);

    py::class_<py_resize_op_int16>(m, "py_resize_op_int16")
        //.def(py::init<py_int32_t, py_int32_t, py_float_t>())
        .def(py::init<py_int32_t, py_int32_t, py_float_t, int, bool, int>())
        .def("forward", &py_resize_op_int16::forward);
}