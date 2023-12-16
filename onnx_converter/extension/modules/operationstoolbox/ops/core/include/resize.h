//
// Created by shengyuan.shen on 2023/3/30.
//

#ifndef OPERATIONS_OPERATION_H
#define OPERATIONS_OPERATION_H

#include "common.h"
#include "eigen_utils.h"
#include "platform/eigen_thread_pool.h"
#include "platform/mace_thread_pool.h"
#include "platform/context_engine.h"

#include <stdio.h>
#include <iostream>
#include <vector>
#include <memory>
#include <math.h>
#include <numeric>
#include <limits>
#include <type_traits>

/*#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>*/

#define SHIFT_NUM  (6)
#define LSHIFT_NUM (1 << (SHIFT_NUM))
#define RSHIFT_NUM ((SHIFT_NUM) * 2)
#define DIV_RSHIFT_NUM (1 << (RSHIFT_NUM))

#define INSTANTIATE_CLASS(classname) \
  template class classname<int8_t>; \
  template class classname<uint8_t>; \
  template class classname<int16_t>; \
  template class classname<uint16_t>;
  //template class classname<float>

/*template<typename T>
int32_t SafeInt(T in_data){
    return static_cast<T>(in_data);
}*/

enum UpsampleMode {
    NN = 0,      // nearest neighbor
    LINEAR = 1,  // linear interpolation
    CUBIC = 2,   // cubic interpolation
};

enum BilinearMode{
    FLOAT=0,
    Integer,
};

enum DataType{
    FP32=0,
    INT8,
    INT16
};

enum ResizeCoordinateTransformationMode {
    HALF_PIXEL = 0,
    ASYMMETRIC = 1,
    PYTORCH_HALF_PIXEL = 2,
    TF_HALF_PIXEL_FOR_NN = 3,
    ALIGN_CORNERS = 4,
    TF_CROP_AND_RESIZE = 5,
    CoordinateTransformationModeCount = 6,
};

enum ResizeNearestMode {
    SIMPLE = 0,  // For resize op 10
    ROUND_PREFER_FLOOR = 1,
    ROUND_PREFER_CEIL = 2,
    FLOOR = 3,
    CEIL = 4,
    NearestModeCount = 5,
};

struct BilinearParams {
    std::vector<float> x_original;
    std::vector<float> y_original;

    int32_t* idx_scale_data_buffer_holder;

    int32_t* input_width_mul_y1;
    int32_t* input_width_mul_y2;

    int32_t* in_x1;
    int32_t* in_x2;

    float* dx1;
    float* dx2;

    float* dy1;
    float* dy2;
};

// Same as above, but doesn't use any floating-point for the coefficient (i.e., d*_scale_10)
struct BilinearParamsInteger {
    std::vector<float> x_original;
    std::vector<float> y_original;

    int32_t* idx_scale_data_buffer_holder;

    int32_t* input_width_mul_y1;
    int32_t* input_width_mul_y2;

    int32_t* in_x1;
    int32_t* in_x2;

    int32_t* dx1_scale_10;
    int32_t* dx2_scale_10;

    int32_t* dy1_scale_10;
    int32_t* dy2_scale_10;
};

using GetNearestPixelFunc = int64_t (*)(float, bool);

using GetOriginalCoordinateFunc = float (*)(float, float, float, float, float, float);

GetOriginalCoordinateFunc GetOriginalCoordinateFromResizedCoordinate(
        ResizeCoordinateTransformationMode coordinate_transform_mode);

BilinearParams SetupUpsampleBilinear(const int32_t input_height,
                                     const int32_t input_width,
                                     const int32_t output_height,
                                     const int32_t output_width,
                                     const float height_scale,
                                     const float width_scale,
                                     const std::vector<float>& roi,
                                     const GetOriginalCoordinateFunc& get_original_coordinate);

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
                      BilinearParams* p,
                      OrtThreadPool* tp);

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
                             BilinearParams* p,
                             MaceThreadPool* tp);

BilinearParamsInteger SetupUpsampleBilinearInteger(const int32_t input_height,
                                                   const int32_t input_width,
                                                   const int32_t output_height,
                                                   const int32_t output_width,
                                                   const float height_scale,
                                                   const float width_scale,
                                                   const std::vector<float>& roi,
                                                   const GetOriginalCoordinateFunc& get_original_coordinate);

template <typename T>
inline void UpsampleBilinearInteger(const int32_t batch_size,
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
                                 bool UseExtrapolation,
                                 const bool is_nchw,
                                 BilinearParamsInteger* p,
                                 MaceThreadPool* tp);

template <typename T>
inline void UpsampleBilinearInteger(const int32_t batch_size,
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
                                    bool UseExtrapolation,
                                    const bool is_nchw,
                                    BilinearParamsInteger* p,
                                    OrtThreadPool* tp);

template<typename itype>
class TimesIntelliResize{

public:

#ifdef USE_EIGEN_THREAD_POOL
    TimesIntelliResize(OrtThreadPool* thread_pool) : thread_pool_(thread_pool) {
        coord_mode_ = ResizeCoordinateTransformationMode::ALIGN_CORNERS;
        scales_ = std::vector<float>{2.0, 2.0};
    };

    TimesIntelliResize(std::vector<int> in_shape,
                       std::vector<int> out_shape,
                       std::vector<float> scales,
                       OrtThreadPool* thread_pool) :
            in_shape_(in_shape), out_shape_(out_shape), scales_(scales),
            thread_pool_(thread_pool){

    };

    TimesIntelliResize(std::vector<int> in_shape,
                       std::vector<int> out_shape,
                       std::vector<float> scales,
                       UpsampleMode calc_mode,
                       ResizeCoordinateTransformationMode coord_mode,
                       OrtThreadPool* thread_pool)
            :in_shape_(in_shape), out_shape_(out_shape), scales_(scales), calc_mode_(calc_mode), coord_mode_(coord_mode), thread_pool_(thread_pool){

    };
#else
    TimesIntelliResize(MaceThreadPool* thread_pool) : thread_pool_(thread_pool) {
        coord_mode_ = ResizeCoordinateTransformationMode::ALIGN_CORNERS;
        scales_ = std::vector<float>{2.0, 2.0};
    };

    TimesIntelliResize(std::vector<int> in_shape,
                       std::vector<int> out_shape,
                       std::vector<float> scales,
                       MaceThreadPool* thread_pool) :
            in_shape_(in_shape), out_shape_(out_shape), scales_(scales),
            thread_pool_(thread_pool){

    };

    TimesIntelliResize(std::vector<int> in_shape,
                       std::vector<int> out_shape,
                       std::vector<float> scales,
                       UpsampleMode calc_mode,
                       ResizeCoordinateTransformationMode coord_mode,
                       MaceThreadPool* thread_pool)
            : in_shape_(in_shape), out_shape_(out_shape), scales_(scales), calc_mode_(calc_mode), coord_mode_(coord_mode), thread_pool_(thread_pool){

    };
#endif

    TimesIntelliResize(const TimesIntelliResize& resize_op){
        this->coord_mode_ = resize_op.coord_mode_;
        this->calc_mode_ = resize_op.calc_mode_;
        this->exectue_mode_ = resize_op.exectue_mode_;

        this->coord_func_ = resize_op.coord_func_;

        this->n = resize_op.n;
        this->c = resize_op.c;
        this->in_h = resize_op.in_h;
        this->in_w = resize_op.in_w;
        this->out_h = resize_op.out_h;
        this->out_w = resize_op.out_w;

        this->in_shape_ = resize_op.in_shape_;
        this->out_shape_ = resize_op.out_shape_;
        this->scales_ = resize_op.scales_;
        this->sizes_ = resize_op.sizes_;
        this->roi_ = resize_op.roi_;
        this->is_nchw_ = resize_op.is_nchw_;
        this->UseExtrapolation_ = resize_op.UseExtrapolation_;
        this->extrapolation_value = resize_op.extrapolation_value;
        this->TimesIntelliBuildTable();
        this->thread_pool_ = resize_op.thread_pool_;
    }

    TimesIntelliResize& operator=(TimesIntelliResize& resize_op){
        if (this != &resize_op) {
            this->coord_mode_ = resize_op.coord_mode_;
            this->calc_mode_ = resize_op.calc_mode_;
            this->exectue_mode_ = resize_op.exectue_mode_;

            this->coord_func_ = resize_op.coord_func_;

            this->n = resize_op.n;
            this->c = resize_op.c;
            this->in_h = resize_op.in_h;
            this->in_w = resize_op.in_w;
            this->out_h = resize_op.out_h;
            this->out_w = resize_op.out_w;

            this->in_shape_ = resize_op.in_shape_;
            this->out_shape_ = resize_op.out_shape_;
            this->scales_ = resize_op.scales_;
            this->sizes_ = resize_op.sizes_;
            this->roi_ = resize_op.roi_;
            this->is_nchw_ = resize_op.is_nchw_;
            this->UseExtrapolation_ = resize_op.UseExtrapolation_;
            this->extrapolation_value = resize_op.extrapolation_value;
            this->TimesIntelliBuildTable();
            this->thread_pool_ = resize_op.thread_pool_;
        }

        return *this;
    }

    ~TimesIntelliResize();

public:

    void TimesIntelliInit(BilinearMode exectue_mode=BilinearMode::Integer, bool is_nchw=true, bool UseExtrapolation=false, std::vector<float> roi=std::vector<float>{});

    void TimesIntelliForward(const itype* const Xdata, const std::vector<int32_t>& in_shape, itype* Ydata);

    void TimesIntelliBuildTable();

public:
    inline void set_in_shape(std::vector<int> in_shape){
        in_shape_ = in_shape;
    }

    inline std::vector<int> get_in_shape(){
        return in_shape_;
    }

    inline void set_out_shape(std::vector<int> out_shape){
        out_shape_ = out_shape;
    }

    inline std::vector<int> get_out_shape(){
        return out_shape_;
    }

    inline void set_scales(std::vector<float> scale){
        scales_ = scale;
    }

    inline std::vector<float> get_scales(){
        return scales_;
    }

    inline void set_sizes(std::vector<float> sizes){
        sizes_ = sizes;
    }

    inline std::vector<float> get_sizes(){
        return sizes_;
    }

    inline BilinearMode get_execute(){
        return exectue_mode_;
    }

    inline void set_execute_mode(BilinearMode exectue_mode){
        exectue_mode_ = exectue_mode;
    }

    std::string get_coord_mode(){
        /*HALF_PIXEL = 0,
        ASYMMETRIC = 1,
        PYTORCH_HALF_PIXEL = 2,
        TF_HALF_PIXEL_FOR_NN = 3,
        ALIGN_CORNERS = 4,
        TF_CROP_AND_RESIZE = 5,
        CoordinateTransformationModeCount = 6,*/
        std::string name;
        switch (coord_mode_) {
            case ResizeCoordinateTransformationMode::HALF_PIXEL:{
                name = "half_pixel";
                break;
            }
            case ResizeCoordinateTransformationMode::TF_CROP_AND_RESIZE:{
                name = "tf_crop_and_resize";
                break;
            }
            case ResizeCoordinateTransformationMode::ALIGN_CORNERS:{
                name = "align corners";
                break;
            }
            case ResizeCoordinateTransformationMode::PYTORCH_HALF_PIXEL:{
                name = "pytorch half pixel";
                break;
            }
            case ResizeCoordinateTransformationMode::CoordinateTransformationModeCount:{
                name = "CoordinateTransformationModeCount";
                break;
            }
            default: {
                name = "nothing";
                break;
            }
        }
        return name;
    }


private:
    ResizeCoordinateTransformationMode coord_mode_;
    UpsampleMode calc_mode_;
    BilinearMode exectue_mode_;
    BilinearParamsInteger BilinerInteger_;
    BilinearParams Biliner_;

    GetOriginalCoordinateFunc coord_func_;

    int32_t n,c,in_h,in_w,out_h,out_w;

    std::vector<int> in_shape_, out_shape_;
    std::vector<float> scales_;
    std::vector<float> sizes_;
    std::vector<float> roi_=std::vector<float>{1.0f,1.0f};
    bool is_nchw_;
    bool UseExtrapolation_;
    float extrapolation_value=0;

#ifdef USE_EIGEN_THREAD_POOL
    OrtThreadPool *thread_pool_;
#else
    MaceThreadPool *thread_pool_;
#endif

};

/*#ifdef USE_EIGEN_THREAD_POOL

void test_resize_nhwc(float scale=0.5, OrtThreadPool* thread_pool=nullptr, bool save_img=false);

void test_resize_nchw(float scale=0.5, OrtThreadPool* thread_pool=nullptr, bool save_img=false);

void test_resize_nhwc_integer(float scale=0.5, OrtThreadPool* thread_pool=nullptr, bool save_img=false);

void test_resize_nchw_integer(float scale=0.5, OrtThreadPool* thread_pool=nullptr, bool save_img=false);

#else

void test_resize_nhwc(float scale=0.5, MaceThreadPool* thread_pool=nullptr, bool save_img=false);

void test_resize_nchw(float scale=0.5, MaceThreadPool* thread_pool=nullptr, bool save_img=false);

void test_resize_nhwc_integer(float scale=0.5, MaceThreadPool* thread_pool=nullptr, bool save_img=false);

void test_resize_nchw_integer(float scale=0.5, MaceThreadPool* thread_pool=nullptr, bool save_img=false);

#endif*/

#endif //OPERATIONS_OPERATION_H
