#include "custom_op_library.h"

#ifdef WITH_CUDA
#include <cuda_runtime.h>
template <typename T1, typename T2, typename T3>
void cuda_add(
    int64_t, T3 *, const T1 *, const T2 *, cudaStream_t compute_stream);
#endif
