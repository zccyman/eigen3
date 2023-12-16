#include <algorithm>
#include <atomic>
#include <functional>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <atomic>
#include <Eigen/Dense>

#include "custom_splice.h"

typedef int64_t index_t;

void Compute3D(
    const std::function<void(const index_t, const index_t, const index_t,
                             const index_t, const index_t, const index_t,
                             const index_t, const index_t, const index_t)>
        &func,
    const index_t start0, const index_t end0, const index_t step0,
    const index_t start1, const index_t end1, const index_t step1,
    const index_t start2, const index_t end2, const index_t step2,
    index_t tile_size0 = 0, index_t tile_size1 = 0, index_t tile_size2 = 0,
    const int cost_per_item = -1) {
  if (start0 >= end0 || start1 >= end1 || start2 >= end1) {
    return;
  }

  const index_t items0 = 1 + (end0 - start0 - 1) / step0;
  const index_t items1 = 1 + (end1 - start1 - 1) / step1;
  const index_t items2 = 1 + (end2 - start2 - 1) / step2;
  func(start0, end0, step0, start1, end1, step1, start2, end2, step2);
}

static std::vector<index_t>
ComputeOutOfShape(std::vector<index_t> &in_shape,
                  std::vector<index_t> &context_,
                  std::vector<index_t> &forward_indexs, index_t const_dim_) {
  const index_t batch = std::accumulate(in_shape.begin(), in_shape.end() - 2, 1,
                                        std::multiplies<index_t>());
  const index_t rank = in_shape.size();
  const index_t chunk = in_shape[rank - 2];
  const index_t input_dim = in_shape[rank - 1];
  const index_t input_stride = chunk * input_dim;

  const index_t num_splice = static_cast<index_t>(context_.size());
  const index_t dim = input_dim - const_dim_;

  const index_t out_chunk = forward_indexs.size() / num_splice;
  const index_t output_dim = dim * num_splice + const_dim_;
  const index_t output_stride = out_chunk * output_dim;

  std::vector<index_t> output_shape = in_shape;
  output_shape[rank - 2] = out_chunk;
  output_shape[rank - 1] = output_dim;
  return output_shape;
}

template <typename T>
void fast_fc(const T* x, const T* w, const T* b, uint32_t K, uint32_t M, uint32_t N, T* out)
{
    Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> weight(w, M, N);
    Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> input(x, K, M);
    Eigen::Map<const Eigen::Matrix<T, 1, Eigen::Dynamic>> bias(b, N);
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> output(out, K, N);
    output = input * weight;
    output.rowwise()+=bias;
}

template <typename T>
void cpu_splice(std::vector<index_t> &in_shape,
    std::vector<index_t> &context_,
    std::vector<index_t> &forward_indexs,
    // std::vector<index_t> &forward_const_indexes_,
    int64_t has_fc,
    std::vector<T> &weight, 
    std::vector<T> &bias, 
    std::vector<index_t> &output_shape,
    const T *input_data,
    T *output_data,
    index_t const_dim_)
{
    const index_t batch = std::accumulate(
        in_shape.begin(), in_shape.end() - 2, 1, std::multiplies<index_t>());
    const index_t rank = in_shape.size();
    const index_t chunk = in_shape[rank - 2];
    const index_t input_dim = in_shape[rank - 1];
    const index_t input_stride = chunk * input_dim;

    const index_t num_splice = static_cast<index_t>(context_.size());
    const index_t dim = input_dim - const_dim_;

    const index_t out_chunk = forward_indexs.size() / num_splice;
    const index_t output_dim = dim * num_splice + const_dim_;
    const index_t output_stride = out_chunk * output_dim;
    const index_t offset_context = context_[0];

    output_shape = in_shape;
    output_shape[rank - 2] = out_chunk;
    output_shape[rank - 1] = output_dim;
    // std::cout << forward_indexs.size() << " " << output_shape[0] << " " << output_shape[1] << " " << output_shape[2] << std::endl;
    index_t output_size = std::accumulate(output_shape.begin(),
        output_shape.end(), 1, std::multiplies<index_t>());
    // T *output_data = (T *)malloc(output_size * sizeof(T));
    auto Calc3DIndex = [=](index_t start0, index_t end0, index_t step0,
                           index_t start1, index_t end1, index_t step1,
                           index_t start2, index_t end2, index_t step2) {
        for (index_t b = start0; b < end0; b += step0) {
            for (index_t i = start1; i < end1; i += step1) {
                for (index_t c = start2; c < end2; c += step2) {
                    // index_t pos = forward_indexs[i * num_splice + c];
                    index_t pos = forward_indexs[i * num_splice + c] + offset_context;
                    pos = pos > 0 ? pos : 0;
                    T *output_base = output_data + b * output_stride +
                        i * output_dim + c * dim;
                    const T *input_base =
                        input_data + b * input_stride + pos * input_dim;
                    memcpy(output_base, input_base, dim * sizeof(T));
                }
            }
        }
    };
    const index_t output_offset = output_dim - const_dim_;
    // auto Calc2DIndex = [=](index_t start0, index_t end0, index_t step0, index_t start1, index_t end1, index_t step1) {
    //     for (index_t b = start0; b < end0; b += step0) {
    //         for (index_t i = start1; i < end1; i += step1) {
    //             T *output_base = output_data + b * output_stride + i * output_dim + output_offset;
    //             const T *input_base = input_data + b * input_stride + forward_const_indexes_[i] * input_dim + dim;
    //             memcpy(output_base, input_base, const_dim_ * sizeof(T));
    //         }
    //     }
    // };
    Compute3D(Calc3DIndex, 0, batch, 1, 0, out_chunk, 1, 0, num_splice, 1);
    // if (const_dim_ > 0) {
    //     Compute2D(Calc2DIndex, 0, batch, 1, 0, out_chunk, 1);
    // }

    if (has_fc==1)
    {
        // T *output_tmp = (T *)malloc(output_stride * sizeof(T));
        // memcpy(output_tmp, output_data, output_stride * sizeof(T));

        // fast_fc(output_tmp, weight.data(), bias.data(), (uint32_t)out_chunk, (uint32_t)output_dim, (uint32_t)output_dim, output_data);
        // free(output_tmp);
    }
    
    // return output_data;
}

template void cpu_splice<int8_t>(std::vector<index_t> &in_shape,
    std::vector<index_t> &context_,
    std::vector<index_t> &forward_indexs,
    // std::vector<index_t> &forward_const_indexes_,
    int64_t has_fc,
    std::vector<int8_t> &weight, 
    std::vector<int8_t> &bias, 
    std::vector<index_t> &output_shape,
    const int8_t *input,
    int8_t *output,
    index_t const_dim_);

template void cpu_splice<int16_t>(std::vector<index_t> &in_shape,
    std::vector<index_t> &context_,
    std::vector<index_t> &forward_indexs,
    // std::vector<index_t> &forward_const_indexes_,
    int64_t has_fc,
    std::vector<int16_t> &weight, 
    std::vector<int16_t> &bias, 
    std::vector<index_t> &output_shape,
    const int16_t *input,
    int16_t *output,
    index_t const_dim_);

template void cpu_splice<int32_t>(std::vector<index_t> &in_shape,
    std::vector<index_t> &context_,
    std::vector<index_t> &forward_indexs,
    // std::vector<index_t> &forward_const_indexes_,
    int64_t has_fc,
    std::vector<int32_t> &weight, 
    std::vector<int32_t> &bias, 
    std::vector<index_t> &output_shape,
    const int32_t *input,
    int32_t *output,
    index_t const_dim_);

template void cpu_splice<int64_t>(std::vector<index_t> &in_shape,
    std::vector<index_t> &context_,
    std::vector<index_t> &forward_indexs,
    // std::vector<index_t> &forward_const_indexes_,
    int64_t has_fc,
    std::vector<int64_t> &weight, 
    std::vector<int64_t> &bias, 
    std::vector<index_t> &output_shape,
    const int64_t *input,
    int64_t *output,
    index_t const_dim_);

template void cpu_splice<float>(std::vector<index_t> &in_shape,
    std::vector<index_t> &context_,
    std::vector<index_t> &forward_indexs,
    int64_t has_fc,
    std::vector<float> &weight, 
    std::vector<float> &bias, 
    // std::vector<index_t> &forward_const_indexes_,
    std::vector<index_t> &output_shape,
    const float *input,
    float *output,
    index_t const_dim_);

template void cpu_splice<double>(std::vector<index_t> &in_shape,
    std::vector<index_t> &context_,
    std::vector<index_t> &forward_indexs,
    // std::vector<index_t> &forward_const_indexes_,
    int64_t has_fc,
    std::vector<double> &weight, 
    std::vector<double> &bias, 
    std::vector<index_t> &output_shape,
    const double *input,
    double *output,
    index_t const_dim_);

struct OrtTensorDimensions : std::vector<int64_t> {
  OrtTensorDimensions(Ort::CustomOpApi ort, const OrtValue *value) {
    OrtTensorTypeAndShapeInfo *info = ort.GetTensorTypeAndShape(value);
    std::vector<int64_t>::operator=(ort.GetTensorShape(info));
    ort.ReleaseTensorTypeAndShapeInfo(info);
  }
};

TIMESINTELLISpliceKernel::TIMESINTELLISpliceKernel(
    OrtApi api, const OrtKernelInfo *info)
    : api_(api), ort_(api_), info_(info)
{
    context_ =
        ort_.KernelInfoGetAttribute<std::vector<int64_t>>(info, "context");
    forward_indexes = ort_.KernelInfoGetAttribute<std::vector<int64_t>>(
        info, "forward_indexes");
    // output_dim = ort_.KernelInfoGetAttribute<int64_t>(info, "output_dim");

    has_fc = ort_.KernelInfoGetAttribute<int64_t>(info, "has_fc");
    if (has_fc == 1)
    {
        //weight = ort_.KernelInfoGetAttribute<std::vector<float>>(info, "weight");
        //bias = ort_.KernelInfoGetAttribute<std::vector<float>>(info, "bias");    
    }
    // create allocator
    allocator_ = Ort::AllocatorWithDefaultOptions();
}

void TIMESINTELLISpliceKernel::Compute(OrtKernelContext *context) {
  const OrtValue *input = ort_.KernelContext_GetInput(context, 0);
  const float *input_data =
      reinterpret_cast<const float *>(ort_.GetTensorData<float>(input));

  OrtTensorDimensions input_dims(ort_, input);

  // int64_t batch_size = input_dims[0];
  // int64_t in_channels = input_dims[1];
  std::vector<index_t> in_shape{input_dims[0], input_dims[1]};

  std::vector<int64_t> output_dims =
      ComputeOutOfShape(in_shape, context_, forward_indexes, 0);

  OrtValue *output = ort_.KernelContext_GetOutput(
      context, 0, output_dims.data(), output_dims.size());
  float *out_ptr = ort_.GetTensorMutableData<float>(output);

    // allocate tmp memory
    // int64_t column_len = 256;
    // float *columns = (float *)allocator_.Alloc(sizeof(float) * column_len);
    // launch kernel
    cpu_splice<float>(in_shape, context_, forward_indexes, has_fc, weight, bias, output_dims,
        input_data, out_ptr, 0);
}