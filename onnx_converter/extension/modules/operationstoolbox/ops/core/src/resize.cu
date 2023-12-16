//
// Created by shengyuan.shen on 2023/3/30.
//

#include "resize.h"
#include "common.h"

#define CUDA_1D_KERNEL_LOOP(i, n)                                 \
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

#define CUDA_2D_KERNEL_LOOP(i, n, j, m)                             \
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (n);   \
       i += blockDim.x * gridDim.x)                                 \
    for (size_t j = blockIdx.y * blockDim.y + threadIdx.y; j < (m); \
         j += blockDim.y * gridDim.y)

inline __device__ int idx(
        const int n,
        const int num_channels,
        const int c,
        const int height,
        const int width,
        const int y,
        const int x) {
    return ((n * num_channels + c) * height + y) * width + x;
}

// input is X, output is Y
template<typename T>
__global__ void UpsampleBilinearKernel(
        const int num_batch,
        const int num_channels,
        const int input_height,
        const int input_width,
        const int output_height,
        const int output_width,
        const T* __restrict__ X,
        T* __restrict__ Y) {


    const int size = output_height * output_width;
    CUDA_1D_KERNEL_LOOP(index, size) {
        int indexTemp = index;
        const int out_x = indexTemp % output_width;
        indexTemp /= output_width;
        const int out_y = indexTemp % output_height;
        indexTemp /= output_height;
        indexTemp /= num_channels;

        const float rheight =
                output_height > 1 ? (input_height - 1.f) / (output_height - 1.f) : 0.f;
        const float rwidth =
                output_width > 1 ? (input_width - 1.f) / (output_width - 1.f) : 0.f;

        // Compute Y axis lambdas
        const float h1r = rheight * out_y;
        const int h1 = (int)h1r;
        const int h1p = (h1 < input_height - 1) ? 1 : 0;
        const float h1lambda = h1r - h1;
        const float h0lambda = 1.f - h1lambda;

        // Compute X axis lambdas
        const float w1r = rwidth * out_x;
        const int w1 = (int)w1r;
        const int w1p = (w1 < input_width - 1) ? 1 : 0;
        const float w1lambda = w1r - w1;
        const float w0lambda = 1.f - w1lambda;

        for (int n = 0; n < num_batch; n++){
            for (int c = 0; c < num_channels; c++) {

                float X0 = X[idx(n, num_channels, c, input_height, input_width, h1, w1)];
                float X1 = X[idx(n, num_channels, c, input_height, input_width, h1, w1 + w1p)];
                float X2 = X[idx(n, num_channels, c, input_height, input_width, h1 + h1p, w1)];
                float X3 = X[idx(n, num_channels, c, input_height, input_width, h1 + h1p, w1 + w1p)];

                Y[idx(n, num_channels, c, output_height, output_width, out_y, out_x)] =
                        h0lambda * (w0lambda * X0 + w1lambda * X1) +
                        h1lambda * (w0lambda * X2 + w1lambda * X3);
            }
        }
    }
}

// input is X, output is Y
template<typename T>
__global__ void UpsampleBilinearIntegerKernel(
        const int num_batch,
        const int num_channels,
        const int input_height,
        const int input_width,
        const int output_height,
        const int output_width,
        const T* __restrict__ dx1,
        const T* __restrict__ dx2,
        const T* __restrict__ dy1,
        const T* __restrict__ dy2,
        const T* __restrict__ in_x1,
        const T* __restrict__ in_x2,
        const T* __restrict__ input_width_mul_y1,
        const T* __restrict__ input_width_mul_y2,
        const T* __restrict__ X,
        T* __restrict__ Y) {

    int32_t in_plan = input_height * input_width;
    int32_t out_plan = output_height * output_width;
    CUDA_1D_KERNEL_LOOP(index, out_plan) {
        int indexTemp = index;
        const int out_x = indexTemp % output_width;
        //indexTemp /= output_width;
        const int out_y = indexTemp % output_height;
        //indexTemp /= output_height;
        //indexTemp /= num_channels;


        for (int n = 0; n < num_batch; n++){
            for (int c = 0; c < num_channels; c++) {
                const int32_t y_w = n * c *  out_plan;
                const int32_t offset = n * c * in_plan;
                T* const Ydata = Y + y_w;

                const int32_t X11_offset = input_width_mul_y1[out_y] + in_x1[out_x] + offset;
                const int32_t X21_offset = input_width_mul_y1[out_y] + in_x2[out_x] + offset;
                const int32_t X12_offset = input_width_mul_y2[out_y] + in_x1[out_x] + offset;
                const int32_t X22_offset = input_width_mul_y2[out_y] + in_x2[out_x] + offset;
                const int32_t X11_coef_scale_20 = dx2[out_x] * dy2[out_y];
                const int32_t X21_coef_scale_20 = dx1[out_x] * dy2[out_y];
                const int32_t X12_coef_scale_20 = dx2[out_x] * dy1[out_y];
                const int32_t X22_coef_scale_20 = dx1[out_x] * dy1[out_y];

                const T X11 = X[X11_offset];
                const T X21 = X[X21_offset];
                const T X12 = X[X12_offset];
                const T X22 = X[X22_offset];

                const T X11_coef = dx2[out_x] * dy2[out_y];
                const T X21_coef = dx1[out_x] * dy2[out_y];
                const T X12_coef = dx2[out_x] * dy1[out_y];
                const T X22_coef = dx1[out_x] * dy1[out_y];

                T sum1 = (X11_coef_scale_20 * X11)>>RSHIFT_NUM;
                T sum2 = (X21_coef_scale_20 * X21)>>RSHIFT_NUM;
                T sum3 = (X12_coef_scale_20 * X12)>>RSHIFT_NUM;
                T sum4 = (X22_coef_scale_20 * X22)>>RSHIFT_NUM;

                Ydata[idx(n, num_channels, c, output_height, output_width, out_y, out_x)] = static_cast<T>(sum1) + static_cast<T>(sum2) + static_cast<T>(sum3) + static_cast<T>(sum4);
            }
        }
    }
}