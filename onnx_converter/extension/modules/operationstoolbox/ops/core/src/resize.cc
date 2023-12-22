//
// Created by shengyuan.shen on 2023/3/30.
//

#include "resize.h"

typedef int64_t (*shift_fun_ptr)(int);
typedef float (*fp_fun_ptr)(int);

std::vector<std::string> s_types{"Pl", "Pi", "Ps", "Pa", "Ph", "Pt", "Pj"};

GetOriginalCoordinateFunc GetOriginalCoordinateFromResizedCoordinate(
        ResizeCoordinateTransformationMode coordinate_transform_mode) {
    switch (coordinate_transform_mode) {
        case ASYMMETRIC:
            return [](float x_resized, float x_scale, float, float, float, float) {
                return x_resized / x_scale;
            };
        case PYTORCH_HALF_PIXEL:
            return [](float x_resized, float x_scale, float length_resized, float, float, float) {
                return length_resized > 1 ? (x_resized + 0.5f) / x_scale - 0.5f : 0.0f;
            };
        case TF_HALF_PIXEL_FOR_NN:
            return [](float x_resized, float x_scale, float, float, float, float) {
                return (x_resized + 0.5f) / x_scale;
            };
        case ALIGN_CORNERS:
            return [](float x_resized, float, float length_resized, float length_original, float, float) {
                return length_resized == 1 ? 0 : x_resized * (length_original - 1) / (length_resized - 1);
            };
        case TF_CROP_AND_RESIZE:
            return [](float x_resized, float, float length_resized, float length_original, float roi_start, float roi_end) {
                auto orig = length_resized > 1
                            ? roi_start * (length_original - 1) +
                              (x_resized * (roi_end - roi_start) * (length_original - 1)) / (length_resized - 1)
                            : 0.5 * (roi_start + roi_end) * (length_original - 1);
                return static_cast<float>(orig);
            };
        default:  // "half_pixel"
            return [](float x_resized, float x_scale, float, float, float, float) {
                return ((x_resized + 0.5f) / x_scale) - 0.5f;
            };
    }
}

// Same as above, but doesn't use any floating-point for the coefficient (i.e., d*_scale_10) computation
//template<>
BilinearParamsInteger SetupUpsampleBilinearInteger(const int32_t input_height,
                                                   const int32_t input_width,
                                                   const int32_t output_height,
                                                   const int32_t output_width,
                                                   const float height_scale,
                                                   const float width_scale,
                                                   const std::vector<float>& roi,
                                                   const GetOriginalCoordinateFunc& get_original_coordinate) {
    BilinearParamsInteger p;

    p.x_original.reserve(output_width);
    p.y_original.reserve(output_height);

    // For each index in the output height and output width, cache its corresponding indices in the input
    // while multiplying it with the input stride for that dimension (cache because we don't have to re-compute
    // each time we come across the output width/ output height value while iterating the output image tensor
    const size_t idx_buffer_size = static_cast<size_t>(2) * sizeof(int32_t) * (output_height + output_width);

    // For each index in the output height and output width, cache its corresponding "weights/scales" for its
    // corresponding indices in the input which proportionately indicates how much they will influence the final
    // pixel value in the output
    // (cache because we don't have to re-compute each time we come across the output width/output height
    // value while iterating the output image tensor
    const size_t scale_buffer_size = static_cast<size_t>(2) * sizeof(int32_t) * (output_height + output_width);

    // Limit number of allocations to just 1
    // const auto inx_scale_data_buffer = alloc->Alloc(idx_buffer_size + scale_buffer_size);
    // p.idx_scale_data_buffer_holder = BufferUniquePtr(inx_scale_data_buffer, BufferDeleter(alloc));
    p.idx_scale_data_buffer_holder = static_cast<int32_t*>(malloc(idx_buffer_size + scale_buffer_size));

    // Get pointers to appropriate memory locations in the scratch buffer
    // auto* const idx_data = static_cast<int32_t*>(p.idx_scale_data_buffer_holder.get());
    auto* const idx_data = static_cast<int32_t*>(p.idx_scale_data_buffer_holder);

    // input_width is the stride for the height dimension
    p.input_width_mul_y1 = idx_data;
    p.input_width_mul_y2 = p.input_width_mul_y1 + output_height;

    // stride for width is 1 (no multiplication needed)
    p.in_x1 = p.input_width_mul_y1 + 2 * output_height;
    p.in_x2 = p.in_x1 + output_width;

    auto* const scale_data = reinterpret_cast<int32_t*>(p.in_x2 + output_width);

    p.dy1_scale_10 = scale_data;
    p.dy2_scale_10 = p.dy1_scale_10 + output_height;

    p.dx1_scale_10 = p.dy1_scale_10 + 2 * output_height;
    p.dx2_scale_10 = p.dx1_scale_10 + output_width;

    // Start processing
    for (int32_t y = 0; y < output_height; ++y) {
        float in_y = height_scale == 1 ? static_cast<float>(y)
                                       : get_original_coordinate(static_cast<float>(y), height_scale,
                                                                 static_cast<float>(output_height),
                                                                 static_cast<float>(input_height),
                                                                 1.0f, 1.0f);
        p.y_original.emplace_back(in_y);
        in_y = std::max(0.0f, std::min(in_y, static_cast<float>(input_height - 1)));
        int32_t in_y_scale_10 = static_cast<int32_t>(in_y * LSHIFT_NUM);

        const int32_t in_y1 = std::min(static_cast<int32_t>(in_y), input_height - 1);
        const int32_t in_y2 = std::min(in_y1 + 1, input_height - 1);
        p.dy1_scale_10[y] = static_cast<int8_t>(std::abs(in_y_scale_10 - in_y1 * LSHIFT_NUM));
        p.dy2_scale_10[y] = static_cast<int8_t>(std::abs(in_y_scale_10 - in_y2 * LSHIFT_NUM));

        if (in_y1 == in_y2) {
            p.dy1_scale_10[y] = static_cast<int8_t>(0.5f * LSHIFT_NUM);
            p.dy2_scale_10[y] = static_cast<int8_t>(0.5f * LSHIFT_NUM);
        }

        p.input_width_mul_y1[y] = input_width * in_y1;
        p.input_width_mul_y2[y] = input_width * in_y2;
    }

    for (int32_t x = 0; x < output_width; ++x) {
        float in_x = width_scale == 1 ? static_cast<float>(x)
                                      : get_original_coordinate(static_cast<float>(x),
                                                                width_scale,
                                                                static_cast<float>(output_width),
                                                                static_cast<float>(input_width),
                                                                1.0f, 1.0f);
        p.x_original.emplace_back(in_x);
        in_x = std::max(0.0f, std::min(in_x, static_cast<float>(input_width - 1)));
        int32_t in_x_scale_10 = static_cast<int32_t>(in_x * LSHIFT_NUM);

        p.in_x1[x] = std::min(static_cast<int32_t>(in_x), input_width - 1);
        p.in_x2[x] = std::min(p.in_x1[x] + 1, input_width - 1);

        p.dx1_scale_10[x] = static_cast<int8_t>(std::abs(in_x_scale_10 - p.in_x1[x] * LSHIFT_NUM));
        p.dx2_scale_10[x] = static_cast<int8_t>(std::abs(in_x_scale_10 - p.in_x2[x] * LSHIFT_NUM));
        if (p.in_x1[x] == p.in_x2[x]) {
            p.dx1_scale_10[x] = static_cast<int8_t>(0.5f * LSHIFT_NUM);
            p.dx2_scale_10[x] = static_cast<int8_t>(0.5f * LSHIFT_NUM);
        }
    }

    return p;
}

template <typename T>
inline void UpsampleBilinearInteger(const int32_t batch_size,
                                 const int32_t num_channels,
                                 const int32_t input_height,
                                 const int32_t input_width,
                                 const int32_t output_height,
                                 const int32_t output_width,
                                 const float height_scale,
                                 const float width_scale,
                                 const float extrapolation_value, /* always equal zero, but asym equal to zero_point */
                                 const T* const XdataBase,
                                 T* const YdataBase,
                                 bool UseExtrapolation,
                                 const bool is_nchw,
                                 BilinearParamsInteger* p,
                                 MaceThreadPool *tp) {

    for (int32_t n = 0; n < batch_size; ++n) {
        if (is_nchw){
            int32_t in_plan = input_height * input_width;
            int32_t out_plan = output_height * output_width;
            const T* const Xdata = XdataBase + n * in_plan * num_channels;
            T* const Ydata = YdataBase + n * out_plan * num_channels;
            auto entity = [&](std::ptrdiff_t first, std::ptrdiff_t last, std::ptrdiff_t step=1) {
                for (std::ptrdiff_t i = first; i < last; ++i) {
                    const int32_t c = static_cast<int32_t>(i / out_plan);
                    const int32_t plan = static_cast<int32_t>(i % out_plan);
                    const int32_t x = static_cast<int32_t>(plan % output_width);
                    const int32_t y = static_cast<int32_t>(plan / output_width);
                    const int32_t offset = c * in_plan;

                    const int32_t X11_offset = (p->input_width_mul_y1[y] + p->in_x1[x]) + offset;
                    const int32_t X21_offset = (p->input_width_mul_y1[y] + p->in_x2[x]) + offset;
                    const int32_t X12_offset = (p->input_width_mul_y2[y] + p->in_x1[x]) + offset;
                    const int32_t X22_offset = (p->input_width_mul_y2[y] + p->in_x2[x]) + offset;
                    const int16_t X11_coef_scale_20 = p->dx2_scale_10[x] * p->dy2_scale_10[y];
                    const int16_t X21_coef_scale_20 = p->dx1_scale_10[x] * p->dy2_scale_10[y];
                    const int16_t X12_coef_scale_20 = p->dx2_scale_10[x] * p->dy1_scale_10[y];
                    const int16_t X22_coef_scale_20 = p->dx1_scale_10[x] * p->dy1_scale_10[y];

                    const T X11 = Xdata[X11_offset];
                    const T X21 = Xdata[X21_offset];
                    const T X12 = Xdata[X12_offset];
                    const T X22 = Xdata[X22_offset];

                    T sum1 = (X11_coef_scale_20 * X11)>>RSHIFT_NUM;
                    T sum2 = (X21_coef_scale_20 * X21)>>RSHIFT_NUM;
                    T sum3 = (X12_coef_scale_20 * X12)>>RSHIFT_NUM;
                    T sum4 = (X22_coef_scale_20 * X22)>>RSHIFT_NUM;
                    //T sum = sum1 + sum2 + sum3 + sum4;

                    // when use_extrapolation is set and original index of x or y is out of the dim range
                    // then use extrapolation_value as the output value.
                    if (UseExtrapolation) {
                        if ((p->y_original[y] < 0 || p->y_original[y] > static_cast<float>(input_height - 1)) ||
                            (p->x_original[x] < 0 || p->x_original[x] > static_cast<float>(input_width - 1))) {
                            Ydata[i] = static_cast<T>(extrapolation_value);
                        } else {
                            //Ydata[i] = static_cast<T>(sum >> RSHIFT_NUM);
                            //Ydata[i] = static_cast<T>(sum1 >> RSHIFT_NUM) + static_cast<T>(sum2 >> RSHIFT_NUM) + static_cast<T>(sum3 >> RSHIFT_NUM) + static_cast<T>(sum4 >> RSHIFT_NUM);
                            Ydata[i] = static_cast<T>(sum1) + static_cast<T>(sum2) + static_cast<T>(sum3) + static_cast<T>(sum4);
                        }
                    } else {
                        //Ydata[i] = static_cast<T>(sum1 >> RSHIFT_NUM) + static_cast<T>(sum2 >> RSHIFT_NUM) + static_cast<T>(sum3 >> RSHIFT_NUM) + static_cast<T>(sum4 >> RSHIFT_NUM);
                        Ydata[i] = static_cast<T>(sum1) + static_cast<T>(sum2) + static_cast<T>(sum3) + static_cast<T>(sum4);
                        //Ydata[i] = static_cast<T>(sum);
                        //Ydata[i] = static_cast<T>(sum1) + static_cast<T>(sum2) + static_cast<T>(sum3) + static_cast<T>(sum4);
                    }
                }
            };

            if (!tp)
                entity(0, num_channels * output_height * output_width);
            else
                tp->Compute1D(entity, 0, num_channels * output_height * output_width, 1);

        }else {
            const T* const Xdata = XdataBase + n * (input_height * input_width) * num_channels;
            T* const Ydata = YdataBase + n * (output_height * output_width) * num_channels;
            auto entity = [&](std::ptrdiff_t first, std::ptrdiff_t last, std::ptrdiff_t step=1) {
                for (std::ptrdiff_t i = first; i < last; ++i) {
                    const int32_t x = static_cast<int32_t>(i % output_width);
                    const int32_t y = static_cast<int32_t>(i / output_width);
                    const int32_t output_offset = (output_width * y + x) * num_channels;

                    const int32_t X11_offset = (p->input_width_mul_y1[y] + p->in_x1[x]) * num_channels;
                    const int32_t X21_offset = (p->input_width_mul_y1[y] + p->in_x2[x]) * num_channels;
                    const int32_t X12_offset = (p->input_width_mul_y2[y] + p->in_x1[x]) * num_channels;
                    const int32_t X22_offset = (p->input_width_mul_y2[y] + p->in_x2[x]) * num_channels;
                    const int32_t X11_coef_scale_20 = p->dx2_scale_10[x] * p->dy2_scale_10[y];
                    const int32_t X21_coef_scale_20 = p->dx1_scale_10[x] * p->dy2_scale_10[y];
                    const int32_t X12_coef_scale_20 = p->dx2_scale_10[x] * p->dy1_scale_10[y];
                    const int32_t X22_coef_scale_20 = p->dx1_scale_10[x] * p->dy1_scale_10[y];

                    // when use_extrapolation is set and original index of x or y is out of the dim range
                    // then use extrapolation_value as the output value.
                    if (UseExtrapolation) {
                        if ((p->y_original[y] < 0 || p->y_original[y] > static_cast<float>(input_height - 1)) ||
                            (p->x_original[x] < 0 || p->x_original[x] > static_cast<float>(input_width - 1))) {
                            for (int32_t c = 0; c < num_channels; ++c) {
                                Ydata[output_offset + c] = static_cast<T>(extrapolation_value);
                            }
                        } else {

                            for (int32_t c = 0; c < num_channels; ++c) {
                                const T X11 = Xdata[X11_offset + c];
                                const T X21 = Xdata[X21_offset + c];
                                const T X12 = Xdata[X12_offset + c];
                                const T X22 = Xdata[X22_offset + c];

                                T sum1 = (X11_coef_scale_20 * X11)>>RSHIFT_NUM;
                                T sum2 = (X21_coef_scale_20 * X21)>>RSHIFT_NUM;
                                T sum3 = (X12_coef_scale_20 * X12)>>RSHIFT_NUM;
                                T sum4 = (X22_coef_scale_20 * X22)>>RSHIFT_NUM;

                                Ydata[output_offset + c] = static_cast<T>(sum1) + static_cast<T>(sum2) + static_cast<T>(sum3) + static_cast<T>(sum4);
                            }
                        }
                    } else {

                        for (int32_t c = 0; c < num_channels; ++c) {
                            const T X11 = Xdata[X11_offset + c];
                            const T X21 = Xdata[X21_offset + c];
                            const T X12 = Xdata[X12_offset + c];
                            const T X22 = Xdata[X22_offset + c];

                            T sum1 = (X11_coef_scale_20 * X11)>>RSHIFT_NUM;
                            T sum2 = (X21_coef_scale_20 * X21)>>RSHIFT_NUM;
                            T sum3 = (X12_coef_scale_20 * X12)>>RSHIFT_NUM;
                            T sum4 = (X22_coef_scale_20 * X22)>>RSHIFT_NUM;

                            Ydata[output_offset + c] = static_cast<T>(sum1) + static_cast<T>(sum2) + static_cast<T>(sum3) + static_cast<T>(sum4);
                        }
                    }
                }
            };
            if (!tp)
                entity(0, output_height * output_width);
            else
                tp->Compute1D(entity, 0, output_height * output_width, 1);
        }
    }
}

template <typename T>
inline void UpsampleBilinearInteger(const int32_t batch_size,
                                    const int32_t num_channels,
                                    const int32_t input_height,
                                    const int32_t input_width,
                                    const int32_t output_height,
                                    const int32_t output_width,
                                    const float height_scale,
                                    const float width_scale,
                                    const float extrapolation_value, /* always equal zero, but asym equal to zero_point */
                                    const T* const XdataBase,
                                    T* const YdataBase,
                                    bool UseExtrapolation,
                                    const bool is_nchw,
                                    BilinearParamsInteger* p,
                                    OrtThreadPool *tp) {

    for (int32_t n = 0; n < batch_size; ++n) {
        if (is_nchw){
            int32_t in_plan = input_height * input_width;
            int32_t out_plan = output_height * output_width;
            const T* const Xdata = XdataBase + n * in_plan * num_channels;
            T* const Ydata = YdataBase + n * out_plan * num_channels;
            auto entity = [&](std::ptrdiff_t first, std::ptrdiff_t last) {
                for (std::ptrdiff_t i = first; i < last; ++i) {
                    const int32_t c = static_cast<int32_t>(i / output_height);
                    const int32_t y = static_cast<int32_t>(i % output_height);
                    const int32_t y_w = c *  out_plan + y * output_width;
                    const int32_t offset = c * in_plan;
                    for (size_t j = 0; j < output_width; j++) {
                        const int32_t x = static_cast<int32_t>(j);
                        const int32_t output_offset = y_w + j;

                        const int32_t X11_offset = (p->input_width_mul_y1[y] + p->in_x1[x]) + offset;
                        const int32_t X21_offset = (p->input_width_mul_y1[y] + p->in_x2[x]) + offset;
                        const int32_t X12_offset = (p->input_width_mul_y2[y] + p->in_x1[x]) + offset;
                        const int32_t X22_offset = (p->input_width_mul_y2[y] + p->in_x2[x]) + offset;
                        const int32_t X11_coef_scale_20 = p->dx2_scale_10[x] * p->dy2_scale_10[y];
                        const int32_t X21_coef_scale_20 = p->dx1_scale_10[x] * p->dy2_scale_10[y];
                        const int32_t X12_coef_scale_20 = p->dx2_scale_10[x] * p->dy1_scale_10[y];
                        const int32_t X22_coef_scale_20 = p->dx1_scale_10[x] * p->dy1_scale_10[y];

                        const T X11 = Xdata[X11_offset];
                        const T X21 = Xdata[X21_offset];
                        const T X12 = Xdata[X12_offset];
                        const T X22 = Xdata[X22_offset];

                        T sum1 = (X11_coef_scale_20 * X11)>>RSHIFT_NUM;
                        T sum2 = (X21_coef_scale_20 * X21)>>RSHIFT_NUM;
                        T sum3 = (X12_coef_scale_20 * X12)>>RSHIFT_NUM;
                        T sum4 = (X22_coef_scale_20 * X22)>>RSHIFT_NUM;
                        // when use_extrapolation is set and original index of x or y is out of the dim range
                        // then use extrapolation_value as the output value.
                        if (UseExtrapolation) {
                            if ((p->y_original[y] < 0 || p->y_original[y] > static_cast<float>(input_height - 1)) ||
                                (p->x_original[x] < 0 || p->x_original[x] > static_cast<float>(input_width - 1))) {
                                Ydata[output_offset] = static_cast<T>(extrapolation_value);
                            } else {
                                //Ydata[i] = static_cast<T>(sum >> RSHIFT_NUM);
                                Ydata[output_offset] = static_cast<T>(sum1) + static_cast<T>(sum2) + static_cast<T>(sum3) + static_cast<T>(sum4);
                            }
                        } else {
                            //Ydata[i] = static_cast<T>(sum);
                            Ydata[output_offset] = static_cast<T>(sum1) + static_cast<T>(sum2) + static_cast<T>(sum3) + static_cast<T>(sum4);
                        }
                    }
                }
            };

            if (!tp)
                entity(0, num_channels * output_height);
            else
                tp->TryParallelFor(
                        tp, static_cast<std::ptrdiff_t>(num_channels) *output_height,
                        static_cast<double>(output_width), entity
                );

        }else {
            const T* const Xdata = XdataBase + n * (input_height * input_width) * num_channels;
            T* const Ydata = YdataBase + n * (output_height * output_width) * num_channels;
            auto entity = [&](std::ptrdiff_t first, std::ptrdiff_t last) {
                for (std::ptrdiff_t i = first; i < last; ++i) {
                    const int32_t x = static_cast<int32_t>(i % output_width);
                    const int32_t y = static_cast<int32_t>(i / output_width);
                    const int32_t output_offset = i * num_channels;

                    // when use_extrapolation is set and original index of x or y is out of the dim range
                    // then use extrapolation_value as the output value.
                    if (UseExtrapolation) {
                        if ((p->y_original[y] < 0 || p->y_original[y] > static_cast<float>(input_height - 1)) ||
                            (p->x_original[x] < 0 || p->x_original[x] > static_cast<float>(input_width - 1))) {
                            for (int32_t c = 0; c < num_channels; ++c) {
                                Ydata[output_offset + c] = static_cast<T>(extrapolation_value);
                            }
                        } else {
                            const int32_t X11_offset = (p->input_width_mul_y1[y] + p->in_x1[x]);
                            const int32_t X21_offset = (p->input_width_mul_y1[y] + p->in_x2[x]);
                            const int32_t X12_offset = (p->input_width_mul_y2[y] + p->in_x1[x]);
                            const int32_t X22_offset = (p->input_width_mul_y2[y] + p->in_x2[x]);
                            const int32_t X11_coef_scale_20 = p->dx2_scale_10[x] * p->dy2_scale_10[y];
                            const int32_t X21_coef_scale_20 = p->dx1_scale_10[x] * p->dy2_scale_10[y];
                            const int32_t X12_coef_scale_20 = p->dx2_scale_10[x] * p->dy1_scale_10[y];
                            const int32_t X22_coef_scale_20 = p->dx1_scale_10[x] * p->dy1_scale_10[y];

                            for (int32_t c = 0; c < num_channels; ++c) {
                                const T X11 = Xdata[X11_offset + c];
                                const T X21 = Xdata[X21_offset + c];
                                const T X12 = Xdata[X12_offset + c];
                                const T X22 = Xdata[X22_offset + c];

                                T sum1 = (X11_coef_scale_20 * X11)>>RSHIFT_NUM;
                                T sum2 = (X21_coef_scale_20 * X21)>>RSHIFT_NUM;
                                T sum3 = (X12_coef_scale_20 * X12)>>RSHIFT_NUM;
                                T sum4 = (X22_coef_scale_20 * X22)>>RSHIFT_NUM;

                                Ydata[output_offset + c] = static_cast<T>(sum1) + static_cast<T>(sum2) + static_cast<T>(sum3) + static_cast<T>(sum4);
                            }
                        }
                    } else {
                        const int32_t X11_offset = (p->input_width_mul_y1[y] + p->in_x1[x]) * num_channels;
                        const int32_t X21_offset = (p->input_width_mul_y1[y] + p->in_x2[x]) * num_channels;
                        const int32_t X12_offset = (p->input_width_mul_y2[y] + p->in_x1[x]) * num_channels;
                        const int32_t X22_offset = (p->input_width_mul_y2[y] + p->in_x2[x]) * num_channels;
                        const int32_t X11_coef_scale_20 = p->dx2_scale_10[x] * p->dy2_scale_10[y];
                        const int32_t X21_coef_scale_20 = p->dx1_scale_10[x] * p->dy2_scale_10[y];
                        const int32_t X12_coef_scale_20 = p->dx2_scale_10[x] * p->dy1_scale_10[y];
                        const int32_t X22_coef_scale_20 = p->dx1_scale_10[x] * p->dy1_scale_10[y];

                        for (int32_t c = 0; c < num_channels; ++c) {
                            const T X11 = Xdata[X11_offset + c];
                            const T X21 = Xdata[X21_offset + c];
                            const T X12 = Xdata[X12_offset + c];
                            const T X22 = Xdata[X22_offset + c];

                            T sum1 = (X11_coef_scale_20 * X11)>>RSHIFT_NUM;
                            T sum2 = (X21_coef_scale_20 * X21)>>RSHIFT_NUM;
                            T sum3 = (X12_coef_scale_20 * X12)>>RSHIFT_NUM;
                            T sum4 = (X22_coef_scale_20 * X22)>>RSHIFT_NUM;

                            Ydata[output_offset + c] = static_cast<T>(sum1) + static_cast<T>(sum2) + static_cast<T>(sum3) + static_cast<T>(sum4);
                        }
                    }
                }
            };
            if (!tp)
                entity(0, output_height * output_width);
            else
                tp->TryParallelFor(
                        tp, static_cast<std::ptrdiff_t>(output_height) * output_width,
                        static_cast<double>(num_channels * 2), entity
                );
        }
    }
}

BilinearParams SetupUpsampleBilinear(const int32_t input_height,
                                     const int32_t input_width,
                                     const int32_t output_height,
                                     const int32_t output_width,
                                     const float height_scale,
                                     const float width_scale,
                                     const std::vector<float>& roi,
                                     const GetOriginalCoordinateFunc& get_original_coordinate) {
    BilinearParams p;
    std::cout << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n";
    std::cout << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n";
    std::cout << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n";
    std::cout << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n";
    std::cout << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n";

    p.x_original.reserve(output_width);
    p.y_original.reserve(output_height);

    // For each index in the output height and output width, cache its corresponding indices in the input
    // while multiplying it with the input stride for that dimension (cache because we don't have to re-compute
    // each time we come across the output width/ output height value while iterating the output image tensor
    const size_t idx_buffer_size = static_cast<size_t>(2) * sizeof(int32_t) * (output_height + output_width);

    // For each index in the output height and output width, cache its corresponding "weights/scales" for its
    // corresponding indices in the input which proportionately indicates how much they will influence the final
    // pixel value in the output
    // (cache because we don't have to re-compute each time we come across the output width/output height
    // value while iterating the output image tensor
    const size_t scale_buffer_size = static_cast<size_t>(2) * sizeof(float) * (output_height + output_width);

    // Limit number of allocations to just 1
    p.idx_scale_data_buffer_holder = static_cast<int32_t*>(malloc(idx_buffer_size + scale_buffer_size));

    // Get pointers to appropriate memory locations in the scratch buffer
    // auto* const idx_data = static_cast<int32_t*>(p.idx_scale_data_buffer_holder.get());
    auto* const idx_data = static_cast<int32_t*>(p.idx_scale_data_buffer_holder);

    // input_width is the stride for the height dimension
    p.input_width_mul_y1 = idx_data;
    p.input_width_mul_y2 = p.input_width_mul_y1 + output_height;

    // stride for width is 1 (no multiplication needed)
    p.in_x1 = p.input_width_mul_y1 + 2 * output_height;
    p.in_x2 = p.in_x1 + output_width;

    auto* const scale_data = reinterpret_cast<float*>(p.in_x2 + output_width);

    p.dy1 = scale_data;
    p.dy2 = p.dy1 + output_height;

    p.dx1 = p.dy1 + 2 * output_height;
    p.dx2 = p.dx1 + output_width;

    // Start processing
    for (int32_t y = 0; y < output_height; ++y) {
        float in_y = height_scale == 1 ? static_cast<float>(y)
                                       : get_original_coordinate(static_cast<float>(y), height_scale,
                                                                 static_cast<float>(output_height),
                                                                 static_cast<float>(input_height),
                                                                 1.0f, 1.0f);
        p.y_original.emplace_back(in_y);
        in_y = std::max(0.0f, std::min(in_y, static_cast<float>(input_height - 1)));

        const int32_t in_y1 = std::min(static_cast<int32_t>(in_y), input_height - 1);
        const int32_t in_y2 = std::min(in_y1 + 1, input_height - 1);
        p.dy1[y] = std::fabs(in_y - in_y1);
        p.dy2[y] = std::fabs(in_y - in_y2);

        if (in_y1 == in_y2) {
            p.dy1[y] = 0.5f;
            p.dy2[y] = 0.5f;
        }

        p.input_width_mul_y1[y] = input_width * in_y1;
        p.input_width_mul_y2[y] = input_width * in_y2;
    }

    for (int32_t x = 0; x < output_width; ++x) {
        float in_x = width_scale == 1 ? static_cast<float>(x)
                                      : get_original_coordinate(static_cast<float>(x),
                                                                width_scale,
                                                                static_cast<float>(output_width),
                                                                static_cast<float>(input_width),
                                                                1.0f, 1.0f);
        p.x_original.emplace_back(in_x);
        in_x = std::max(0.0f, std::min(in_x, static_cast<float>(input_width - 1)));

        p.in_x1[x] = std::min(static_cast<int32_t>(in_x), input_width - 1);
        p.in_x2[x] = std::min(p.in_x1[x] + 1, input_width - 1);

        p.dx1[x] = std::fabs(in_x - p.in_x1[x]);
        p.dx2[x] = std::fabs(in_x - p.in_x2[x]);
        if (p.in_x1[x] == p.in_x2[x]) {
            p.dx1[x] = 0.5f;
            p.dx2[x] = 0.5f;
        }
    }

    return p;
}

template <typename T>
inline void UpsampleBilinear(const int32_t batch_size,
                      const int32_t num_channels,
                      const int32_t input_height,
                      const int32_t input_width,
                      const int32_t output_height,
                      const int32_t output_width,
                      const float height_scale,
                      const float width_scale,
                      const float extrapolation_value,
                      const T* const XdataBase,
                      T* const YdataBase,
                      const bool UseExtrapolation,
                      const bool is_nchw,
                      BilinearParams *p,
                      MaceThreadPool* tp) {
    int32_t in_plan = input_height * input_width;
    int32_t out_plan = output_height * output_width;
    for (int32_t n = 0; n < batch_size; ++n) {
        if (is_nchw){
            const T* const Xdata = XdataBase + n * in_plan * num_channels;
            T* const Ydata = YdataBase + n * out_plan * num_channels;
            auto entity = [&](std::ptrdiff_t first, std::ptrdiff_t last, std::ptrdiff_t step=1) {
                for (std::ptrdiff_t i = first; i < last; ++i) {
                    const int32_t c = static_cast<int32_t>(i / out_plan);
                    const int32_t plan = static_cast<int32_t>(i % out_plan);
                    const int32_t x = static_cast<int32_t>(plan % output_width);
                    const int32_t y = static_cast<int32_t>(plan / output_width);
                    const int32_t offset = c * in_plan;

                    const int32_t X11_offset = p->input_width_mul_y1[y] + p->in_x1[x] + offset;
                    const int32_t X21_offset = p->input_width_mul_y1[y] + p->in_x2[x] + offset;
                    const int32_t X12_offset = p->input_width_mul_y2[y] + p->in_x1[x] + offset;
                    const int32_t X22_offset = p->input_width_mul_y2[y] + p->in_x2[x] + offset;
                    const float X11_coef = p->dx2[x] * p->dy2[y];
                    const float X21_coef = p->dx1[x] * p->dy2[y];
                    const float X12_coef = p->dx2[x] * p->dy1[y];
                    const float X22_coef = p->dx1[x] * p->dy1[y];
                    const T X11 = Xdata[X11_offset];
                    const T X21 = Xdata[X21_offset];
                    const T X12 = Xdata[X12_offset];
                    const T X22 = Xdata[X22_offset];

                    // when use_extrapolation is set and original index of x or y is out of the dim range
                    // then use extrapolation_value as the output value.
                    if (UseExtrapolation) {
                        if ((p->y_original[y] < 0 || p->y_original[y] > static_cast<float>(input_height - 1)) ||
                            (p->x_original[x] < 0 || p->x_original[x] > static_cast<float>(input_width - 1))) {
                            Ydata[i] = static_cast<T>(extrapolation_value);
                        } else {
                            Ydata[i] = static_cast<T>(X11_coef * X11 + X21_coef * X21 + X12_coef * X12 + X22_coef * X22);
                        }
                    } else {
                        Ydata[i] = static_cast<T>(X11_coef * X11 + X21_coef * X21 + X12_coef * X12 + X22_coef * X22);
                    }
                }
            };

            if (!tp)
                entity(0, num_channels * output_height * output_width);
            else
                tp->Compute1D(entity, 0, num_channels * output_height * output_width, 1);
        }else {
            const T* const Xdata = XdataBase + n * (input_height * input_width) * num_channels;
            T* const Ydata = YdataBase + n * (output_height * output_width) * num_channels;
            auto entity = [&](std::ptrdiff_t first, std::ptrdiff_t last, std::ptrdiff_t step=1) {
                for (std::ptrdiff_t i = first; i < last; ++i) {
                    const int32_t x = static_cast<int32_t>(i % output_width);
                    const int32_t y = static_cast<int32_t>(i / output_width);
                    const int32_t output_offset = (output_width * y + x) *  num_channels;
                    const int32_t X11_offset = (p->input_width_mul_y1[y] + p->in_x1[x]) * num_channels;
                    const int32_t X21_offset = (p->input_width_mul_y1[y] + p->in_x2[x]) * num_channels;
                    const int32_t X12_offset = (p->input_width_mul_y2[y] + p->in_x1[x]) * num_channels;
                    const int32_t X22_offset = (p->input_width_mul_y2[y] + p->in_x2[x]) * num_channels;
                    const float X11_coef = p->dx2[x] * p->dy2[y];
                    const float X21_coef = p->dx1[x] * p->dy2[y];
                    const float X12_coef = p->dx2[x] * p->dy1[y];
                    const float X22_coef = p->dx1[x] * p->dy1[y];
                    //printf("output width is: %d, output height is: %d, num channels is: %d, output offset is: %d\n", output_width, output_height, num_channels, output_offset);
                    //std::cout << output_width << output_height << num_channels << output_offset << std::endl;

                    // when use_extrapolation is set and original index of x or y is out of the dim range
                    // then use extrapolation_value as the output value.
                    if (UseExtrapolation) {
                        if ((p->y_original[y] < 0 || p->y_original[y] > static_cast<float>(input_height - 1)) ||
                            (p->x_original[x] < 0 || p->x_original[x] > static_cast<float>(input_width - 1))) {
                            for (int32_t c = 0; c < num_channels; ++c) {
                                Ydata[output_offset + c] = static_cast<T>(extrapolation_value);
                            }
                        } else {
                            for (int32_t c = 0; c < num_channels; ++c) {
                                const T X11 = Xdata[X11_offset + c];
                                const T X21 = Xdata[X21_offset + c];
                                const T X12 = Xdata[X12_offset + c];
                                const T X22 = Xdata[X22_offset + c];
                                //int32_t sum = X11_coef * X11 + X21_coef * X21 + X12_coef * X12 + X22_coef * X22;
                                //std::cout << sum << std::endl;

                                Ydata[output_offset + c] = static_cast<T>(X11_coef * X11 +
                                                                          X21_coef * X21 +
                                                                          X12_coef * X12 +
                                                                          X22_coef * X22);
                            }
                        }
                    } else {
                        for (int32_t c = 0; c < num_channels; ++c) {
                            const T X11 = Xdata[X11_offset + c];
                            const T X21 = Xdata[X21_offset + c];
                            const T X12 = Xdata[X12_offset + c];
                            const T X22 = Xdata[X22_offset + c];
                            //int32_t sum = X11_coef * X11 + X21_coef * X21 + X12_coef * X12 + X22_coef * X22;
                            //std::cout << sum << std::endl;

                            Ydata[output_offset + c] = static_cast<T>(X11_coef * X11 +
                                                                      X21_coef * X21 +
                                                                      X12_coef * X12 +
                                                                      X22_coef * X22);
                        }
                    }
                }
            };

            if (!tp)
                entity(0, output_height * output_width);
            else
                tp->Compute1D(entity, 0, output_height * output_width, 1);
        }
    }
}

template <typename T>
inline void UpsampleBilinear(const int32_t batch_size,
                             const int32_t num_channels,
                             const int32_t input_height,
                             const int32_t input_width,
                             const int32_t output_height,
                             const int32_t output_width,
                             const float height_scale,
                             const float width_scale,
                             const float extrapolation_value,
                             const T* const XdataBase,
                             T* const YdataBase,
                             const bool UseExtrapolation,
                             const bool is_nchw,
                             BilinearParams *p,
                             OrtThreadPool* tp) {
    int32_t in_plan = input_height * input_width;
    int32_t out_plan = output_height * output_width;
    for (int32_t n = 0; n < batch_size; ++n) {
        if (is_nchw){
            const T* const Xdata = XdataBase + n * in_plan * num_channels;
            T* const Ydata = YdataBase + n * out_plan * num_channels;
            auto entity = [&](std::ptrdiff_t first, std::ptrdiff_t last) {
                for (std::ptrdiff_t i = first; i < last; ++i) {
                    const int32_t c = static_cast<int32_t>(i / output_height);
                    const int32_t y = static_cast<int32_t>(i % output_height);
                    const int32_t y_w = c *  out_plan + y * output_width;
                    const int32_t offset = c * in_plan;
                    for (size_t j = 0; j < output_width; j++) {
                        const int32_t x = static_cast<int32_t>(j);
                        const int32_t output_offset = y_w + j;
                        const int32_t X11_offset = (p->input_width_mul_y1[y] + p->in_x1[x]) + offset;
                        const int32_t X21_offset = (p->input_width_mul_y1[y] + p->in_x2[x]) + offset;
                        const int32_t X12_offset = (p->input_width_mul_y2[y] + p->in_x1[x]) + offset;
                        const int32_t X22_offset = (p->input_width_mul_y2[y] + p->in_x2[x]) + offset;
                        const float X11_coef = p->dx2[x] * p->dy2[y];
                        const float X21_coef = p->dx1[x] * p->dy2[y];
                        const float X12_coef = p->dx2[x] * p->dy1[y];
                        const float X22_coef = p->dx1[x] * p->dy1[y];

                        const T X11 = Xdata[X11_offset];
                        const T X21 = Xdata[X21_offset];
                        const T X12 = Xdata[X12_offset];
                        const T X22 = Xdata[X22_offset];

                        // when use_extrapolation is set and original index of x or y is out of the dim range
                        // then use extrapolation_value as the output value.
                        if (UseExtrapolation) {
                            if ((p->y_original[y] < 0 || p->y_original[y] > static_cast<float>(input_height - 1)) ||
                                (p->x_original[x] < 0 || p->x_original[x] > static_cast<float>(input_width - 1))) {
                                Ydata[output_offset] = static_cast<T>(extrapolation_value);
                            } else {
                                Ydata[output_offset] = static_cast<T>(X11_coef * X11 +
                                                                      X21_coef * X21 +
                                                                      X12_coef * X12 +
                                                                      X22_coef * X22);
                            }
                        } else {
                            Ydata[output_offset] = static_cast<T>(X11_coef * X11 +
                                                                  X21_coef * X21 +
                                                                  X12_coef * X12 +
                                                                  X22_coef * X22);
                        }
                    }
                }
            };

            if (!tp)
                entity(0, num_channels * output_height);
            else
                tp->TryParallelFor(
                        tp, static_cast<std::ptrdiff_t>(num_channels) * output_height,
                        static_cast<double>(output_width * 2), entity
                );
        }else {
            const T* const Xdata = XdataBase + n * (input_height * input_width) * num_channels;
            T* const Ydata = YdataBase + n * (output_height * output_width) * num_channels;
            auto entity = [&](std::ptrdiff_t first, std::ptrdiff_t last) {
                for (std::ptrdiff_t i = first; i < last; ++i) {
                    const int32_t x = static_cast<int32_t>(i % output_width);
                    const int32_t y = static_cast<int32_t>(i / output_width);
                    const int32_t output_offset = i * num_channels;

                    const int32_t X11_offset = (p->input_width_mul_y1[y] + p->in_x1[x]) * num_channels;
                    const int32_t X21_offset = (p->input_width_mul_y1[y] + p->in_x2[x]) * num_channels;
                    const int32_t X12_offset = (p->input_width_mul_y2[y] + p->in_x1[x]) * num_channels;
                    const int32_t X22_offset = (p->input_width_mul_y2[y] + p->in_x2[x]) * num_channels;
                    const float X11_coef = p->dx2[x] * p->dy2[y];
                    const float X21_coef = p->dx1[x] * p->dy2[y];
                    const float X12_coef = p->dx2[x] * p->dy1[y];
                    const float X22_coef = p->dx1[x] * p->dy1[y];

                    // when use_extrapolation is set and original index of x or y is out of the dim range
                    // then use extrapolation_value as the output value.
                    if (UseExtrapolation) {
                        if ((p->y_original[y] < 0 || p->y_original[y] > static_cast<float>(input_height - 1)) ||
                            (p->x_original[x] < 0 || p->x_original[x] > static_cast<float>(input_width - 1))) {
                            for (int32_t c = 0; c < num_channels; ++c) {
                                Ydata[output_offset + c] = static_cast<T>(extrapolation_value);
                            }
                        } else {

                            for (int32_t c = 0; c < num_channels; ++c) {
                                const T X11 = Xdata[X11_offset + c];
                                const T X21 = Xdata[X21_offset + c];
                                const T X12 = Xdata[X12_offset + c];
                                const T X22 = Xdata[X22_offset + c];
                                //int32_t sum = X11_coef * X11 + X21_coef * X21 + X12_coef * X12 + X22_coef * X22;
                                //std::cout << sum << std::endl;

                                Ydata[output_offset + c] = static_cast<T>(X11_coef * X11 +
                                                                          X21_coef * X21 +
                                                                          X12_coef * X12 +
                                                                          X22_coef * X22);
                            }
                        }
                    } else {
                        for (int32_t c = 0; c < num_channels; ++c) {
                            const T X11 = Xdata[X11_offset + c];
                            const T X21 = Xdata[X21_offset + c];
                            const T X12 = Xdata[X12_offset + c];
                            const T X22 = Xdata[X22_offset + c];
                            //int32_t sum = X11_coef * X11 + X21_coef * X21 + X12_coef * X12 + X22_coef * X22;
                            //std::cout << sum << std::endl;

                            Ydata[output_offset + c] = static_cast<T>(X11_coef * X11 +
                                                                      X21_coef * X21 +
                                                                      X12_coef * X12 +
                                                                      X22_coef * X22);
                        }
                    }
                }
            };
            if (!tp)
                entity(0, output_height * output_width);
            else
                tp->TryParallelFor(
                        tp, static_cast<std::ptrdiff_t>(output_height) * output_width,
                        static_cast<double>(num_channels), entity
                );
        }
    }
}


template<typename itype>
void TimesIntelliResize<itype>::TimesIntelliForward(const itype* const Xdata, const std::vector<int32_t>& in_shape, itype* Ydata) {
    if (n != in_shape[0]) n= in_shape[0];
    float scale_h = scales_[0], scale_w = scales_[1];
    if(exectue_mode_==BilinearMode::FLOAT){
        UpsampleBilinear<itype>(n, c, in_h, in_w, out_h, out_w, scale_h, scale_w, extrapolation_value,
                                Xdata, Ydata, UseExtrapolation_, is_nchw_, &Biliner_, thread_pool_);
    }else if(exectue_mode_==BilinearMode::Integer){
        UpsampleBilinearInteger<itype>(n, c, in_h, in_w, out_h, out_w, scale_h, scale_w, extrapolation_value,
                                       Xdata, Ydata, UseExtrapolation_, is_nchw_, &BilinerInteger_, thread_pool_);
    }else{
        std::cout << "invalid BilinearMode!\n";
        exit(-1);
    }
}

template<typename itype>
void TimesIntelliResize<itype>::TimesIntelliInit(BilinearMode exectue_mode, bool is_nchw, bool UseExtrapolation, std::vector<float> roi){
    coord_func_ = GetOriginalCoordinateFromResizedCoordinate(coord_mode_);
    exectue_mode_ = exectue_mode;
    n = in_shape_[0];
    c = in_shape_[1];
    in_h = in_shape_[2];
    in_w = in_shape_[3];
    out_h = out_shape_[2];
    out_w = out_shape_[3];
    is_nchw_ = is_nchw;
    UseExtrapolation_ = UseExtrapolation;
    roi_ = roi;
    this->TimesIntelliBuildTable();
}

template<typename itype>
void TimesIntelliResize<itype>::TimesIntelliBuildTable() {
    float scale_h = scales_[0], scale_w = scales_[1];
    if(exectue_mode_==BilinearMode::FLOAT){
        Biliner_ = SetupUpsampleBilinear(in_h, in_w, out_h, out_w, scale_h, scale_w, roi_, coord_func_);
    }else if(exectue_mode_==BilinearMode::Integer){
        BilinerInteger_ = SetupUpsampleBilinearInteger(in_h, in_w, out_h, out_w, scale_h, scale_w, roi_, coord_func_);
    }else{
        std::cout << "invalid table!\n";
        exit(-1);
    }
}

template<typename itype>
TimesIntelliResize<itype>::~TimesIntelliResize() {
    if (this->exectue_mode_ == BilinearMode::Integer){
        free(BilinerInteger_.idx_scale_data_buffer_holder);
    }
    if (this->exectue_mode_ == BilinearMode::FLOAT){
        free(Biliner_.idx_scale_data_buffer_holder);
    }
}

INSTANTIATE_CLASS(TimesIntelliResize);

/*cv::Mat quzntize_mat(cv::Mat& img, float* scale){
    cv::Mat quantize_;
    img.convertTo(quantize_, CV_32F);
    double min_v, max_v;
    cv::minMaxLoc(quantize_, &min_v, &max_v);
    *scale = 127/max_v;
    for (size_t i=0; i<quantize_.rows; i++){
        for(size_t j=0; j<quantize_.cols; j++){
            quantize_.at<cv::Vec3f>(i, j)[0] *= *scale;
            quantize_.at<cv::Vec3f>(i, j)[1] *= *scale;
            quantize_.at<cv::Vec3f>(i, j)[2] *= *scale;
        }
    }
    return quantize_;
}

#define TOTAL 100

#ifdef USE_EIGEN_THREAD_POOL
void test_resize_nhwc(float scale, OrtThreadPool* thread_pool, bool save_img) {
#else
void test_resize_nhwc(float scale, MaceThreadPool* thread_pool, bool save_img){
#endif
    cv::Mat img = cv::imread("../resize-rgb888.png");
    //float scale = 0.5;
    GetOriginalCoordinateFunc mode = GetOriginalCoordinateFromResizedCoordinate(ResizeCoordinateTransformationMode::ALIGN_CORNERS);
    std::vector<float> roi{0};
    size_t src_buffer_size = static_cast<size_t>(sizeof(float)*img.rows*img.cols*img.channels());
    size_t dst_buffer_size = static_cast<size_t>(sizeof(float)*img.rows*img.cols*img.channels()*scale*scale);
    float* src = (float*)malloc(src_buffer_size);
    float* resizeed_img = (float*)malloc(dst_buffer_size);
    float scale_ = 0;
    cv::Mat q_mat = quzntize_mat(img, &scale_);
    for (size_t i=0; i<q_mat.rows; i++){
        for(size_t j=0; j<q_mat.cols; j++){
            float a = q_mat.at<cv::Vec3f>(i, j)[0];
            float b = q_mat.at<cv::Vec3f>(i, j)[1];
            float c = q_mat.at<cv::Vec3f>(i, j)[2];
            size_t buffer = (i * q_mat.cols + j) * q_mat.channels();
            *(src + buffer) = a;
            *(src + buffer + 1) = b;
            *(src + buffer + 2) = c;
        }
    }
    int32_t output_width;
    int32_t output_height;
    output_width = static_cast<int32_t>(img.cols * scale);
    output_height = static_cast<int32_t>(img.rows * scale);
    //printf("image channel is: %d rows is: %d, cols is: %d, first element is: %d\n", img.channels(), img.rows, img.cols, img.data[1000]);
    //MaceThreadPool* threadPool = nullptr;

    BilinearParams p = SetupUpsampleBilinear(img.rows, img.cols, output_height, output_width,
                                             scale, scale, roi, mode);
    //printf("output hieght is: %d, output width is: %d\n", output_height, output_width);
    clock_t start_t, end_t;
    start_t = clock();
    int total = TOTAL;
    for (size_t i=0; i<total; i++)
        UpsampleBilinear<float>(1, img.channels(), img.rows, img.cols,
                                  output_height, output_width, scale, scale,
                                  0.0f, (float*)src, (float*)resizeed_img,
                                  false, false, &p, thread_pool);
    end_t = clock();
    std::cout << "test_resize_nhwc time consume is: " << float(end_t - start_t) / total / CLOCKS_PER_SEC * 1000 * 1000 << " us" << std::endl;
    cv::Mat dst(output_height, output_width, CV_8UC3);
    for (size_t i=0; i<dst.rows; i++){
        for(size_t j=0; j<dst.cols; j++){
            size_t buffer = (i * dst.cols + j) * dst.channels();
            dst.at<cv::Vec3b>(i, j)[0] = *(resizeed_img + buffer) / scale_;
            dst.at<cv::Vec3b>(i, j)[1] = *(resizeed_img + buffer + 1) / scale_;
            dst.at<cv::Vec3b>(i, j)[2] = *(resizeed_img + buffer + 2) / scale_;

        }
    }
    if(save_img)
        cv::imwrite("../dst_nhwc.jpg", dst);
    free(p.idx_scale_data_buffer_holder);
}

#ifdef USE_EIGEN_THREAD_POOL
void test_resize_nchw(float scale, OrtThreadPool* thread_pool, bool save_img){
#else
void test_resize_nchw(float scale, MaceThreadPool* thread_pool, bool save_img){
#endif
    cv::Mat img = cv::imread("../resize-rgb888.png");
    //float scale = 0.5;
    GetOriginalCoordinateFunc mode = GetOriginalCoordinateFromResizedCoordinate(ResizeCoordinateTransformationMode::ALIGN_CORNERS);
    std::vector<float> roi{0};
    size_t src_buffer_size = static_cast<size_t>(sizeof(uint8_t)*img.rows*img.cols*img.channels());
    size_t dst_buffer_size = static_cast<size_t>(sizeof(uint8_t)*img.rows*img.cols*img.channels()*scale*scale);
    unsigned char* src = (unsigned char*)malloc(src_buffer_size);
    unsigned char* resizeed_img = (unsigned char*)malloc(dst_buffer_size);
    size_t buffer = img.rows * img.cols;
    for (size_t i=0; i<img.rows; i++){
        for(size_t j=0; j<img.cols; j++){
            size_t tmp = i * img.cols + j;
            uint8_t a = img.at<cv::Vec3b>(i, j)[0];
            uint8_t b = img.at<cv::Vec3b>(i, j)[1];
            uint8_t c = img.at<cv::Vec3b>(i, j)[2];
            *(src + tmp) = a;
            *(src + tmp + buffer) = b;
            *(src + tmp + buffer * 2) = c;
        }
    }
    int32_t output_width = static_cast<int32_t>(img.cols * scale);
    int32_t output_height = static_cast<int32_t>(img.rows * scale);
    cv::Mat dst(output_height, output_width, CV_8UC3);
    size_t buffer_dst = dst.rows * dst.cols;
    //MaceThreadPool* threadpool = nullptr;
    //printf("image channel is: %d rows is: %d, cols is: %d, first element is: %d\n", img.channels(), img.rows, img.cols, img.data[1000]);
    //printf("output hieght is: %d, output width is: %d\n", output_height, output_width);
    BilinearParams p = SetupUpsampleBilinear(img.rows, img.cols, output_height, output_width,
                                             scale, scale, roi, mode);
    clock_t start_t, end_t;
    start_t = clock();
    int total = TOTAL;
    for (size_t i=0; i<total; i++)
        UpsampleBilinear<uint8_t>(1, img.channels(), img.rows, img.cols,
                                  output_height, output_width, scale, scale,
                                  0.0f, (uint8_t*)src, (uint8_t*)resizeed_img,
                                  false, true, &p, thread_pool);
    end_t = clock();
    std::cout << "test_resize_nchw time consume is: " << float(end_t - start_t) / total / CLOCKS_PER_SEC * 1000 * 1000 << " us" << std::endl;

    for (size_t i=0; i<dst.rows; i++){
        for(size_t j=0; j<dst.cols; j++){
            size_t tmp = i * dst.cols + j;
            dst.at<cv::Vec3b>(i, j)[0] = *(resizeed_img + tmp) ;
            dst.at<cv::Vec3b>(i, j)[1] = *(resizeed_img + tmp + buffer_dst);
            dst.at<cv::Vec3b>(i, j)[2] = *(resizeed_img + tmp + buffer_dst * 2);

        }
    }
    if(save_img)
        cv::imwrite("../dst_nchw.jpg", dst);
    free(p.idx_scale_data_buffer_holder);
}

#ifdef USE_EIGEN_THREAD_POOL
void test_resize_nhwc_integer(float scale, OrtThreadPool* thread_pool, bool save_img){
#else
void test_resize_nhwc_integer(float scale, MaceThreadPool* thread_pool, bool save_img){
#endif
    cv::Mat img = cv::imread("../resize-rgb888.png");
    //float scale = 0.5;
    GetOriginalCoordinateFunc mode = GetOriginalCoordinateFromResizedCoordinate(ResizeCoordinateTransformationMode::ALIGN_CORNERS);
    std::vector<float> roi{0};
    size_t src_buffer_size = static_cast<size_t>(sizeof(uint8_t)*img.rows*img.cols*img.channels());
    size_t dst_buffer_size = static_cast<size_t>(sizeof(uint8_t)*img.rows*img.cols*img.channels()*scale*scale);
    unsigned char* src = (unsigned char*)malloc(src_buffer_size);
    unsigned char* resizeed_img = (unsigned char*)malloc(dst_buffer_size);
    for (size_t i=0; i<img.rows; i++){
        for(size_t j=0; j<img.cols; j++){
            uint8_t a = img.at<cv::Vec3b>(i, j)[0];
            uint8_t b = img.at<cv::Vec3b>(i, j)[1];
            uint8_t c = img.at<cv::Vec3b>(i, j)[2];
            size_t buffer = (i * img.cols + j) * img.channels();
            *(src + buffer) = a;
            *(src + buffer + 1) = b;
            *(src + buffer + 2) = c;
        }
    }
    int32_t output_width;
    int32_t output_height;
    output_width = static_cast<int32_t>(img.cols * scale);
    output_height = static_cast<int32_t>(img.rows * scale);
    //printf("image channel is: %d rows is: %d, cols is: %d, first element is: %d\n", img.channels(), img.rows, img.cols, img.data[1000]);
    //printf("output hieght is: %d, output width is: %d\n", output_height, output_width);
    //MaceThreadPool* threadpool = nullptr;
    BilinearParamsInteger p = SetupUpsampleBilinearInteger(img.rows, img.cols, output_height, output_width, scale, scale, roi, mode);
    clock_t start_t, end_t;
    start_t = clock();
    int total = TOTAL;
    for (size_t i=0; i<total; i++)
        UpsampleBilinearInteger<uint8_t>(1, img.channels(), img.rows, img.cols,
                                         output_height, output_width, scale, scale,
                                         0.0f, (uint8_t*)src, (uint8_t*)resizeed_img,
                                         false, false, &p, thread_pool);
    end_t = clock();
    std::cout << "test_resize_nhwc_integer time consume is: " << float(end_t - start_t) / total / CLOCKS_PER_SEC * 1000 * 1000 << " us" << std::endl;
    cv::Mat dst(output_height, output_width, CV_8UC3);
    for (size_t i=0; i<dst.rows; i++){
        for(size_t j=0; j<dst.cols; j++){
            size_t buffer = (i * dst.cols + j) * 3;
            dst.at<cv::Vec3b>(i, j)[0] = *(resizeed_img + buffer) ;
            dst.at<cv::Vec3b>(i, j)[1] = *(resizeed_img + buffer + 1);
            dst.at<cv::Vec3b>(i, j)[2] = *(resizeed_img + buffer + 2);

        }
    }
    if(save_img)
        cv::imwrite("../dst_nhwc_int.jpg", dst);
    //free(p.idx_scale_data_buffer_holder);
}

#ifdef USE_EIGEN_THREAD_POOL
void test_resize_nchw_integer(float scale, OrtThreadPool* thread_pool, bool save_img){
#else
void test_resize_nchw_integer(float scale, MaceThreadPool* thread_pool, bool save_img){
#endif
    cv::Mat img = cv::imread("../resize-rgb888.png");
    //float scale = 0.5;
    GetOriginalCoordinateFunc mode = GetOriginalCoordinateFromResizedCoordinate(ResizeCoordinateTransformationMode::ALIGN_CORNERS);
    std::vector<float> roi{0};
    size_t src_buffer_size = static_cast<size_t>(sizeof(uint8_t)*img.rows*img.cols*img.channels());
    size_t dst_buffer_size = static_cast<size_t>(sizeof(uint8_t)*img.rows*img.cols*img.channels()*scale*scale);
    unsigned char* src = (unsigned char*)malloc(src_buffer_size);
    unsigned char* resizeed_img = (unsigned char*)malloc(dst_buffer_size);
    size_t buffer = img.rows * img.cols;
    for (size_t i=0; i<img.rows; i++){
        for(size_t j=0; j<img.cols; j++){
            size_t tmp = i * img.cols + j;
            uint8_t a = img.at<cv::Vec3b>(i, j)[0];
            uint8_t b = img.at<cv::Vec3b>(i, j)[1];
            uint8_t c = img.at<cv::Vec3b>(i, j)[2];
            *(src + tmp) = a;
            *(src + tmp + buffer) = b;
            *(src + tmp + buffer * 2) = c;
        }
    }
    int32_t output_width;
    int32_t output_height;
    output_width = static_cast<int32_t>(img.cols * scale);
    output_height = static_cast<int32_t>(img.rows * scale);
    cv::Mat dst(output_height, output_width, CV_8UC3);
    size_t buffer_dst = dst.rows * dst.cols;
    //printf("image channel is: %d rows is: %d, cols is: %d, first element is: %d\n", img.channels(), img.rows, img.cols, img.data[1000]);
    //printf("output hieght is: %d, output width is: %d\n", output_height, output_width);
    //MaceThreadPool* threadpool = nullptr;
    BilinearParamsInteger p = SetupUpsampleBilinearInteger(img.rows, img.cols, output_height, output_width, scale, scale, roi, mode);
    clock_t start_t, end_t;
    start_t = clock();
    int total = TOTAL;
    for (size_t i=0; i<total; i++)
        UpsampleBilinearInteger<uint8_t>(1, img.channels(), img.rows, img.cols,
                                         output_height, output_width, scale, scale,
                                         0.0f, (uint8_t*)src, (uint8_t*)resizeed_img,
                                         false, true, &p, thread_pool);
    end_t = clock();
    std::cout << "test_resize_nchw_integer time consume is: " << float(end_t - start_t) / total / CLOCKS_PER_SEC * 1000 * 1000 << " us" << std::endl;
    for (size_t i=0; i<dst.rows; i++){
        for(size_t j=0; j<dst.cols; j++){
            int32_t tmp = i * dst.cols + j;
            dst.at<cv::Vec3b>(i, j)[0] = *(resizeed_img + tmp) ;
            dst.at<cv::Vec3b>(i, j)[1] = *(resizeed_img + tmp + buffer_dst);
            dst.at<cv::Vec3b>(i, j)[2] = *(resizeed_img + tmp + buffer_dst * 2);

        }
    }
    if(save_img)
        cv::imwrite("../dst_nchw_int.jpg", dst);
    free(p.idx_scale_data_buffer_holder);
}*/