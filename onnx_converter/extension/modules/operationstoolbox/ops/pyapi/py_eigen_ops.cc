#include "py_eigen_ops.h"
#include "eigen_ops.h"


template <typename T>
void py_fast_fc_kernel(
    const py::array_t<T, py::array::c_style | py::array::forcecast> ma,
    const py::array_t<T, py::array::c_style | py::array::forcecast> mb,
    const py::array_t<T, py::array::c_style | py::array::forcecast> mc,
    uint32_t K, 
    uint32_t M, 
    uint32_t N,
    py::array_t<T, py::array::c_style | py::array::forcecast> result)
{
    T* a = (T*)malloc(K*M*sizeof(T));
    T* b = (T*)malloc(M*N*sizeof(T));
    T* c = (T*)malloc(N*sizeof(T));
    T* res = (T*)malloc(K*N*sizeof(T));
    memcpy(a, ma.data(),K*M*sizeof(T));
    memcpy(b, mb.data(),M*N*sizeof(T));
    memcpy(c, mc.data(),N*sizeof(T));
    memcpy(res, result.data(),K*N*sizeof(T));
    fast_fc(a, b, c, K, M, N, res);
    memcpy(const_cast<T*>(result.data()), res, K*N*sizeof(T));
    free(a);
    free(b);
    free(c);
    free(res);
}
template void py_fast_fc_kernel(    
    const py::array_t<float, py::array::c_style | py::array::forcecast> ma,
    const py::array_t<float, py::array::c_style | py::array::forcecast> mb,
    const py::array_t<float, py::array::c_style | py::array::forcecast> mc,
    uint32_t K, 
    uint32_t M, 
    uint32_t N,
    py::array_t<float, py::array::c_style | py::array::forcecast> result);

template void py_fast_fc_kernel(    
    const py::array_t<double, py::array::c_style | py::array::forcecast> ma,
    const py::array_t<double, py::array::c_style | py::array::forcecast> mb,
    const py::array_t<double, py::array::c_style | py::array::forcecast> mc,
    uint32_t K, 
    uint32_t M, 
    uint32_t N,
    py::array_t<double, py::array::c_style | py::array::forcecast> result);
