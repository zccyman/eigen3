
#include "env.h"
#include "splice.h"

void Compute3D(const std::function<void(const index_t,
                   const index_t,
                   const index_t,
                   const index_t,
                   const index_t,
                   const index_t)> &func,
    const index_t start1,
    const index_t end1,
    const index_t step1,
    const index_t start2,
    const index_t end2,
    const index_t step2,
    index_t tile_size1 = 0,
    index_t tile_size2 = 0,
    const int cost_per_item = -1)
{
    if (start1 >= end1 || start2 >= end1) {
        return;
    }

    const index_t items1 = 1 + (end1 - start1 - 1) / step1;
    const index_t items2 = 1 + (end2 - start2 - 1) / step2;
    func(start1, end1, step1, start2, end2, step2);
}

template <typename T>
T *cpu_splice(std::vector<int> &in_shape,
    std::vector<int> &context_,
    std::vector<int> &forward_indexs,
    std::vector<int> &forward_const_indexes_,
    std::vector<int> &output_shape,
    T *input_data,
    //T *output_data,
    int const_dim_)
{
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
    // std::cout << "input_dim: " << input_dim << std::endl;
    // std::cout << "output_dim: " << output_dim << std::endl;
    // std::cout << "chunk: " << chunk << std::endl;
    // std::cout << "out_chunk: " << out_chunk << std::endl;
    // std::cout << "num_splice: " << num_splice << std::endl;

    output_shape = in_shape;
    output_shape[rank - 2] = out_chunk;
    output_shape[rank - 1] = output_dim;
    // std::cout << forward_indexs.size() << " " << output_shape[0] << " " << output_shape[1] << " " << output_shape[2] << std::endl;
    int32_t output_size = std::accumulate(
        output_shape.begin(), output_shape.end(), 1, std::multiplies<int>());
    T *output_data = (T *)malloc(output_size * sizeof(T));
    
    auto Calc3DIndex = [=](index_t start1, index_t end1, index_t step1,
                           index_t start2, index_t end2, index_t step2) {
        for (index_t i = start1; i < end1; i += step1) {
            for (index_t c = start2; c < end2; c += step2) {
                index_t pos = forward_indexs[i * num_splice + c] + offset_context;
                pos = pos > 0 ? pos : 0;
                // std::cout << pos << " ";
                T *output_base = output_data + i * output_dim + c * dim;
                if (i < out_chunk) {
                    const T *input_base = input_data + pos * input_dim;
                    memcpy(output_base, input_base, dim * sizeof(T));
                } //else {
                    // memset(output_base, 0, dim * sizeof(T));
                //}
            }
        }
    };
    const index_t output_offset = output_dim - const_dim_;
    auto Calc2DIndex = [=](index_t start0, index_t end0, index_t step0,
                           index_t start1, index_t end1, index_t step1) {
        for (index_t b = start0; b < end0; b += step0) {
            for (index_t i = start1; i < end1; i += step1) {
                T *output_base = output_data + b * output_stride +
                    i * output_dim + output_offset;
                const T *input_base = input_data + b * input_stride +
                    forward_const_indexes_[i] * input_dim + dim;
                memcpy(output_base, input_base, const_dim_ * sizeof(T));
            }
        }
    };
    Compute3D(Calc3DIndex, 0, out_chunk, 1, 0, num_splice, 1);
    
    return output_data;
}

template int8_t *cpu_splice<int8_t>(std::vector<int> &in_shape,
    std::vector<int> &context_,
    std::vector<int> &forward_indexs,
    std::vector<int> &forward_const_indexes_,
    std::vector<int> &output_shape,
    int8_t *input, /*int8_t *output,*/
    int const_dim_);

template int16_t *cpu_splice<int16_t>(std::vector<int> &in_shape,
    std::vector<int> &context_,
    std::vector<int> &forward_indexs,
    std::vector<int> &forward_const_indexes_,
    std::vector<int> &output_shape,
    int16_t *input, /*int16_t *output,*/
    int const_dim_);

template int32_t *cpu_splice<int32_t>(std::vector<int> &in_shape,
    std::vector<int> &context_,
    std::vector<int> &forward_indexs,
    std::vector<int> &forward_const_indexes_,
    std::vector<int> &output_shape,
    int32_t *input, /*int32_t *output,*/
    int const_dim_);

template int64_t *cpu_splice<int64_t>(std::vector<int> &in_shape,
    std::vector<int> &context_,
    std::vector<int> &forward_indexs,
    std::vector<int> &forward_const_indexes_,
    std::vector<int> &output_shape,
    int64_t *input, /*int64_t *output,*/
    int const_dim_);

template float *cpu_splice<float>(std::vector<int> &in_shape,
    std::vector<int> &context_,
    std::vector<int> &forward_indexs,
    std::vector<int> &forward_const_indexes_,
    std::vector<int> &output_shape,
    float *input, /*float *output,*/
    int const_dim_);

template double *cpu_splice<double>(std::vector<int> &in_shape,
    std::vector<int> &context_,
    std::vector<int> &forward_indexs,
    std::vector<int> &forward_const_indexes_,
    std::vector<int> &output_shape,
    double *input, /*double *output,*/
    int const_dim_);