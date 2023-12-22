
#include "env.h"
#include "eigen_ops.h"

template <typename T>
void fast_fc(const T* ma, const T* mb, const T* mc, uint32_t K, uint32_t M, uint32_t N, T* result)
{
    Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> a(ma, K, M);
    Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> b(mb, M, N);
    Eigen::Map<const Eigen::Matrix<T, 1, Eigen::Dynamic>> c(mc, N);
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> output(result, K, N);
    output = a * b;
    output.rowwise()+=c;
}

template void fast_fc(const float* ma, const float* mb, const float* mc, uint32_t K, uint32_t M, uint32_t N, float* result);
template void fast_fc(const double* ma, const double* mb, const double* mc, uint32_t K, uint32_t M, uint32_t N, double* result);

