#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
// #include "device_functions.h"
#include "cublas_v2.h"
#include "splice.h"

void TimeIntelliAssertFailure_(const char *func, const char *file, int32_t line,
                         const char *cond_str) {
  std::cout << "Assertion failed: (" << cond_str << ")" << std::endl;
  fflush(NULL); // Flush all pending buffers, abort() may not flush stderr.
  std::abort();
}

#ifndef NDEBUG
#define TIMEINTELLI_ASSERT(cond)                                                     \
  do {                                                                         \
    if (cond)                                                                  \
        (void)0;                                                                 \
    else                                                                       \
        TimeIntelliAssertFailure_(__func__, __FILE__, __LINE__, #cond);       \
  } while (0)
#else
#define TIMEINTELLI_ASSERT(cond) (void)0
#endif

inline index_t n_blocks(index_t size, index_t block_size) {
  return size / block_size + ((size % block_size == 0)? 0 : 1);
}

void GetBlockSizesForSimpleMatrixOperation(index_t num_rows,
                                           index_t num_cols,
                                           dim3 *dimGrid,
                                           dim3 *dimBlock) {
    TIMEINTELLI_ASSERT(num_rows > 0 && num_cols > 0);
    index_t col_blocksize = 64, row_blocksize = 4;
    while (
        col_blocksize > 1 && 
        (num_cols + (num_cols / 2) <= col_blocksize || 
        num_rows > 65535 * row_blocksize)) {
            col_blocksize /= 2;
            row_blocksize *= 2;
    }

    dimBlock->x = col_blocksize;
    dimBlock->y = row_blocksize;
    dimBlock->z = 1;
    dimGrid->x = n_blocks(num_cols, col_blocksize);
    dimGrid->y = n_blocks(num_rows, row_blocksize);
    TIMEINTELLI_ASSERT(dimGrid->y <= 65535 && "Matrix has too many rows to process");
    dimGrid->z = 1;
}

template<typename T>
__global__ void copy_channel(
    T* dst, 
    const T *src, 
    index_t *reorder, 
    index_t rows, 
    index_t cols, 
    index_t src_stride, 
    index_t dst_stride) {
    index_t i = blockIdx.x * blockDim.x + threadIdx.x;  // col index
    index_t j = blockIdx.y * blockDim.y + threadIdx.y;  // row index
    // stride: [src stride, dst  stride]
    if (i < cols && j < rows) {
            index_t index = reorder[j];
            index_t dst_index = j * dst_stride + i;
            if (index >= 0) {
                index_t src_index = reorder[j] * src_stride + i;
            // T val = src[src_index];
            dst[dst_index] = src[src_index];
        } else {
            dst[dst_index] = 0;
        }
    }
}

#define CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N)      \
  int32_t id = blockDim.x * blockIdx.x + threadIdx.x;   \
  if (id >= N)                                          \
    return;


template <typename T>
__global__ void splice_kernel(const T * input, T * output, index_t in_offset, index_t out_offset, index_t countNum) {
    CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, countNum);
    output[out_offset + id] = input[in_offset + id];
}

template
__global__ void splice_kernel<int8_t>(const int8_t * input, int8_t * output, index_t in_offset, index_t out_offset, index_t countNum);

template
__global__ void splice_kernel<int16_t>(const int16_t * input, int16_t * output, index_t in_offset, index_t out_offset, index_t countNum);

template
__global__ void splice_kernel<int32_t>(const int32_t * input, int32_t * output, index_t in_offset, index_t out_offset, index_t countNum);

template
__global__ void splice_kernel<int64_t>(const int64_t * input, int64_t * output, index_t in_offset, index_t out_offset, index_t countNum);

template
__global__ void splice_kernel<float>(const float * input, float * output, index_t in_offset, index_t out_offset, index_t countNum);

template
__global__ void splice_kernel<double>(const double * input, double * output, index_t in_offset, index_t out_offset, index_t countNum);

template<typename T>
T * gpu_splice(
    std::vector<int> &in_shape,
    std::vector<int> &context_, 
    std::vector<int> &forward_indexs, 
    std::vector<int> &forward_const_indexes_,
    std::vector<int> &output_shape,  
    T *input, /*T *output,*/ 
    int const_dim_){
    const index_t rank = in_shape.size();
    const index_t in_chunk = in_shape[rank - 2];
    const index_t input_dim = in_shape[rank - 1];
    const index_t input_stride = in_chunk * input_dim;

    const index_t num_splice = static_cast<index_t>(context_.size());
    const index_t dim = input_dim - const_dim_;

    const index_t out_chunk = forward_indexs.size() / num_splice;
    const index_t output_dim = dim * num_splice + const_dim_;
    const index_t output_stride = out_chunk * output_dim;

    output_shape = in_shape;
    output_shape[rank - 2] = in_chunk;
    output_shape[rank - 1] = output_dim;
    // std::cout << forward_indexs.size() << " " << output_shape[0] << " " << output_shape[1] << " " << output_shape[2] << std::endl;
    index_t output_size = std::accumulate(output_shape.begin(), output_shape.end(), 1, std::multiplies<index_t>());
    T *output_data = (T *)malloc(output_size * sizeof(T));
    memset(output_data + output_stride, 0, (in_chunk - out_chunk) * output_dim * sizeof(T));
    T *device_in, *device_out;
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaMalloc(&device_in, input_stride * sizeof(T));
    cudaMalloc(&device_out, output_stride * sizeof(T));
    
    cudaMemcpyAsync((void *)device_in, (void *)input, input_stride * sizeof(T), cudaMemcpyHostToDevice, stream);
    {
        #pragma omp parallel for num_threads(out_chunk)                   
        for (index_t out_index = 0; out_index < out_chunk; out_index++) {
            for (index_t c = 0; c < num_splice; c++) {   
                const index_t pos = forward_indexs[out_index * num_splice + c];
                splice_kernel<T><<<n_blocks(dim, 256), 256, 0, stream>>>
                    (device_in, device_out, pos * input_dim, out_index * output_dim + c * dim, dim);
            }
        }
    }

    cudaMemcpyAsync((void *)output_data, (void *)device_out, output_stride * sizeof(T), cudaMemcpyDeviceToHost, stream);
    cudaFree(device_in);
    cudaFree(device_out);
    cudaStreamDestroy(stream);
    return output_data;
}

template
int8_t * gpu_splice<int8_t>(
    std::vector<int> &in_shape,
    std::vector<int> &context_, 
    std::vector<int> &forward_indexs, 
    std::vector<int> &forward_const_indexes_,
    std::vector<int> &output_shape, 
    int8_t *input, /*int8_t *output,*/ 
    int const_dim_);

template
int16_t * gpu_splice<int16_t>(
    std::vector<int> &in_shape, 
    std::vector<int> &context_, 
    std::vector<int> &forward_indexs, 
    std::vector<int> &forward_const_indexes_,
    std::vector<int> &output_shape, 
    int16_t *input, /*int16_t *output,*/ 
    int const_dim_);

template
int32_t * gpu_splice<int32_t>(
    std::vector<int> &in_shape, 
    std::vector<int> &context_, 
    std::vector<int> &forward_indexs, 
    std::vector<int> &forward_const_indexes_,
    std::vector<int> &output_shape, 
    int32_t *input, /*int32_t *output,*/ 
    int const_dim_);

template
int64_t * gpu_splice<int64_t>(
    std::vector<int> &in_shape, 
    std::vector<int> &context_, 
    std::vector<int> &forward_indexs, 
    std::vector<int> &forward_const_indexes_,
    std::vector<int> &output_shape, 
    int64_t *input, /*int64_t *output,*/ 
    int const_dim_);

template
float * gpu_splice<float>(
    std::vector<int> &in_shape, 
    std::vector<int> &context_, 
    std::vector<int> &forward_indexs, 
    std::vector<int> &forward_const_indexes_,
    std::vector<int> &output_shape, 
    float *input, /*float *output,*/ 
    int const_dim_);

template
double * gpu_splice<double>(
    std::vector<int> &in_shape, 
    std::vector<int> &context_,
    std::vector<int> &forward_indexs, 
    std::vector<int> &forward_const_indexes_,
    std::vector<int> &output_shape, 
    double *input, /*double *output,*/ 
    int const_dim_);