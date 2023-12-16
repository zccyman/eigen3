#include <cstring>
#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "env.h"
#include "splice.h"

#define generate_random(low, high) \
    (low + (high - low) * rand() / float(RAND_MAX))

template <typename T>
T add(T a, T b)
{
    return a + b;
}

TEST(TEST_ADD_FLOAT, add)
{
    float res = add(1.2f, 2.2f);
    EXPECT_FLOAT_EQ(3.4f, res);
}

TEST(TEST_ADD_INTER, add)
{
    int res = add(2, 3);
    EXPECT_EQ(5, res);
}

TEST(TEST_CPU_SPLICE_INT8, cpu_slice)
{
    typedef int8_t datatype;
    std::vector<int32_t> input_shape{74, 43}, output_shape;
    std::vector<std::vector<datatype>> array(
        input_shape[0], std::vector<datatype>(input_shape[1], 0));
    for (size_t i = 0; i < input_shape[0]; i++) {
        for (size_t j = 0; j < input_shape[1]; j++) {
            srand((int32_t)time(0));
            array[i][j] = (datatype)generate_random(-128.0f, 127.0f);
        }
    }

    int64_t alloc_t = array.size() * array[0].size() * sizeof(datatype);
    datatype *input_data = (datatype *)malloc(alloc_t);
    std::memcpy(input_data, array.data(), alloc_t);
    std::vector<int32_t> out_shape, forward_const_indexes_;
    std::vector<int32_t> context_{-1, 0, 1};
    std::vector<int32_t> forward_indexs_{0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4, 5, 4,
        5, 6, 5, 6, 7, 6, 7, 8, 7, 8, 9, 8, 9, 10, 9, 10, 11, 10, 11, 12, 11,
        12, 13, 12, 13, 14, 13, 14, 15, 14, 15, 16, 15, 16, 17, 16, 17, 18, 17,
        18, 19, 18, 19, 20, 19, 20, 21, 20, 21, 22, 21, 22, 23, 22, 23, 24, 23,
        24, 25, 24, 25, 26, 25, 26, 27, 26, 27, 28, 27, 28, 29, 28, 29, 30, 29,
        30, 31, 30, 31, 32, 31, 32, 33, 32, 33, 34, 33, 34, 35, 34, 35, 36, 35,
        36, 37, 36, 37, 38, 37, 38, 39, 38, 39, 40, 39, 40, 41, 40, 41, 42, 41,
        42, 43, 42, 43, 44, 43, 44, 45, 44, 45, 46, 45, 46, 47, 46, 47, 48, 47,
        48, 49, 48, 49, 50, 49, 50, 51, 50, 51, 52, 51, 52, 53, 52, 53, 54, 53,
        54, 55, 54, 55, 56, 55, 56, 57, 56, 57, 58, 57, 58, 59, 58, 59, 60, 59,
        60, 61, 60, 61, 62, 61, 62, 63, 62, 63, 64, 63, 64, 65, 64, 65, 66, 65,
        66, 67, 66, 67, 68, 67, 68, 69, 68, 69, 70, 69, 70, 71, 70, 71, 72, 71,
        72, 73};

    datatype *out = cpu_splice(input_shape, context_, forward_indexs_,
        forward_const_indexes_, out_shape, input_data, 0);

    int32_t d0, d1, d2;
    d0 = out_shape[0];
    d1 = context_.size();
    d2 = out_shape[1] / d1;

    for (size_t i = 0; i < d0; i++) {
        size_t offset_0 = i * d1 * d2 + context_[context_.size() - 1] * d2;
        size_t offset_1 = i * d2;
        for (size_t j = 0; j < d2; j++) {
            datatype tmp0 = (datatype)out[offset_0 + j];
            datatype tmp1 = (datatype)input_data[offset_1 + j];
            EXPECT_EQ(tmp0 - tmp1, 0);
        }
    }
}

#ifdef WITH_CUDA
TEST(TEST_GPU_SPLICE_INT8, gpu_slice)
{
    typedef int8_t datatype;
    std::vector<int32_t> input_shape{74, 43}, output_shape;
    std::vector<std::vector<datatype>> array(
        input_shape[0], std::vector<datatype>(input_shape[1], 0));
    for (size_t i = 0; i < input_shape[0]; ++i) {
        for (size_t j = 0; j < input_shape[1]; ++j) {
            srand((int32_t)time(0));
            array[i][j] = (datatype)generate_random(-128.0f, 127.0f);
        }
    }

    int64_t alloc_t = array.size() * sizeof(datatype);
    datatype *input_data = (datatype *)malloc(alloc_t);
    std::memcpy(input_data, array.data(), alloc_t);
    std::vector<int32_t> out_shape, forward_const_indexes_;
    std::vector<int32_t> context_{-1, 0, 1};
    std::vector<int32_t> forward_indexs_{0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4, 5, 4,
        5, 6, 5, 6, 7, 6, 7, 8, 7, 8, 9, 8, 9, 10, 9, 10, 11, 10, 11, 12, 11,
        12, 13, 12, 13, 14, 13, 14, 15, 14, 15, 16, 15, 16, 17, 16, 17, 18, 17,
        18, 19, 18, 19, 20, 19, 20, 21, 20, 21, 22, 21, 22, 23, 22, 23, 24, 23,
        24, 25, 24, 25, 26, 25, 26, 27, 26, 27, 28, 27, 28, 29, 28, 29, 30, 29,
        30, 31, 30, 31, 32, 31, 32, 33, 32, 33, 34, 33, 34, 35, 34, 35, 36, 35,
        36, 37, 36, 37, 38, 37, 38, 39, 38, 39, 40, 39, 40, 41, 40, 41, 42, 41,
        42, 43, 42, 43, 44, 43, 44, 45, 44, 45, 46, 45, 46, 47, 46, 47, 48, 47,
        48, 49, 48, 49, 50, 49, 50, 51, 50, 51, 52, 51, 52, 53, 52, 53, 54, 53,
        54, 55, 54, 55, 56, 55, 56, 57, 56, 57, 58, 57, 58, 59, 58, 59, 60, 59,
        60, 61, 60, 61, 62, 61, 62, 63, 62, 63, 64, 63, 64, 65, 64, 65, 66, 65,
        66, 67, 66, 67, 68, 67, 68, 69, 68, 69, 70, 69, 70, 71, 70, 71, 72, 71,
        72, 73};

    datatype *out = gpu_splice(input_shape, context_, forward_indexs_,
        forward_const_indexes_, out_shape, input_data, 0);

    int32_t d0, d1, d2;
    d0 = out_shape[0];
    d1 = context_.size();
    d2 = out_shape[1] / d1;

    for (size_t i = 0; i < d0; i++) {
        size_t offset_0 = i * d1 * d2 + context_[context_.size() - 1] * d2;
        size_t offset_1 = i * d2;
        for (size_t j = 0; j < d2; j++) {
            datatype tmp0 = (datatype)out[offset_0 + j];
            datatype tmp1 = (datatype)input_data[offset_1 + j];
            EXPECT_EQ(tmp0 - tmp1, 0);
        }
    }
}
#endif  // WITH_CUDA

int main(int argc, char *argv[])
{
    ::testing::InitGoogleMock(&argc, argv);
    return RUN_ALL_TESTS();
}
