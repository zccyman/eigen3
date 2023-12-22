#pragma once

#include "py_def.h"
#include "splice.h"
#include "resize.h"

class pyops {
public:
    pyops(){};
    ~pyops(){};

    template <typename T>  // test template funcation -> add
    static py::array_t<T> add(const py::array_t<T> &a,
        const py::array_t<T> &b,
        const std::vector<int> &size);

    template <typename T>  // test template funcation -> cpu_splice
    static py::array_t<T> py_cpu_splice(
        py::array_t<T> array, py_int32_t context, py_int32_t forward_indexs);
    
    template <typename T>
    static void py_eigen_fc(
        const py::array_t<T, py::array::c_style | py::array::forcecast> ma,
        const py::array_t<T, py::array::c_style | py::array::forcecast> mb,
        const py::array_t<T, py::array::c_style | py::array::forcecast> mc,
        uint32_t K, 
        uint32_t M, 
        uint32_t N,
        py::array_t<T, py::array::c_style | py::array::forcecast> result);

#ifdef WITH_CUDA
    template <typename T>  // test template funcation -> gpu_splice
    static py::array_t<T> py_gpu_splice(
        py::array_t<T> array, py_int32_t context, py_int32_t forward_indexs);
#endif
};

template <typename T>
py::array_t<T, py::array::c_style | py::array::forcecast> py_cpu_kernel_splice(
    py::array_t<T, py::array::c_style | py::array::forcecast> array,
    py_int32_t context,
    py_int32_t forward_indexs);

#ifdef WITH_CUDA
template <typename T>
py::array_t<T, py::array::c_style | py::array::forcecast> py_gpu_kernel_splice(
    py::array_t<T, py::array::c_style | py::array::forcecast> array,
    py_int32_t context,
    py_int32_t forward_indexs);
#endif

class py_resize_op_int8{

public:

    py_resize_op_int8() = delete;

    py_resize_op_int8(const py_int32_t& in_shape, const py_int32_t& out_shape, const py_float_t& scale, int bilinear_mode=0, bool is_nchw=true, int coord_mode=4);

    void forward(const py_int8_t& X, py_int8_t& Y, py_int32_t& in_shape);

    void operator()(const py_int8_t& X, py_int8_t& Y, py_int32_t& in_shape);

    py_resize_op_int8(const py_resize_op_int8& resize_op){
        this->in_shape_v = resize_op.in_shape_v;
        this->out_shape_v = resize_op.out_shape_v;
        this->scale_v = resize_op.scale_v;
        this->bilinear_mode_ = resize_op.bilinear_mode_;
        this->is_nchw_ = resize_op.is_nchw_;

        this->resize_op_ = new TimesIntelliResize<int8_t>(this->in_shape_v, this->out_shape_v, this->scale_v, nullptr);
        this->resize_op_->TimesIntelliInit((BilinearMode)this->bilinear_mode_, this->is_nchw_);
    }

    py_resize_op_int8& operator=(const py_resize_op_int8& resize_op){
        if (this != &resize_op){
            this->in_shape_v = resize_op.in_shape_v;
            this->out_shape_v = resize_op.out_shape_v;
            this->scale_v = resize_op.scale_v;
            this->bilinear_mode_ = resize_op.bilinear_mode_;
            this->is_nchw_ = resize_op.is_nchw_;

            this->resize_op_ = new TimesIntelliResize<int8_t>(this->in_shape_v, this->out_shape_v, this->scale_v, nullptr);
            this->resize_op_->TimesIntelliInit((BilinearMode)this->bilinear_mode_, this->is_nchw_);
        }
        //print_copy_assigned(this);
        return *this;
    }

    /*py_resize_op_int8& operator=(const py_resize_op_int8&& resize_op) noexcept {
        if (this != &resize_op){
            this->in_shape_v = resize_op.in_shape_v;
            this->out_shape_v = resize_op.out_shape_v;
            this->scale_v = resize_op.scale_v;
            this->bilinear_mode_ = resize_op.bilinear_mode_;
            this->is_nchw_ = resize_op.is_nchw_;

            this->resize_op_ = new TimesIntelliResize<int8_t>(this->in_shape_v, this->out_shape_v, this->scale_v, nullptr);
            this->resize_op_->TimesIntelliInit();
        }
        // print_copy_assigned(this);
        return *this;
    }*/

    ~py_resize_op_int8();

private:
    TimesIntelliResize<int8_t> *resize_op_;
    int coord_mode_;
    int8_t* output_data;
    std::vector<int32_t> in_shape_v, out_shape_v;
    std::vector<float> scale_v;
    int bilinear_mode_;
    bool is_nchw_;
};

class py_resize_op_uint8{

public:

    py_resize_op_uint8() = delete;

    py_resize_op_uint8(const py_int32_t& in_shape, const py_int32_t& out_shape, const py_float_t& scale, int bilinear_mode=0, bool is_nchw=true, int coord_mode=4);

    void forward(const py_uint8_t& X, py_uint8_t& Y, py_int32_t& in_shape);

    void operator()(const py_uint8_t& X, py_uint8_t& Y, py_int32_t& in_shape);

    py_resize_op_uint8(const py_resize_op_uint8& resize_op){
        this->in_shape_v = resize_op.in_shape_v;
        this->out_shape_v = resize_op.out_shape_v;
        this->scale_v = resize_op.scale_v;
        this->bilinear_mode_ = resize_op.bilinear_mode_;
        this->is_nchw_ = resize_op.is_nchw_;

        this->resize_op_ = new TimesIntelliResize<uint8_t>(this->in_shape_v, this->out_shape_v, this->scale_v, nullptr);
        this->resize_op_->TimesIntelliInit((BilinearMode)this->bilinear_mode_, this->is_nchw_);
    }

    py_resize_op_uint8& operator=(const py_resize_op_uint8& resize_op){
        if (this != &resize_op){
            this->in_shape_v = resize_op.in_shape_v;
            this->out_shape_v = resize_op.out_shape_v;
            this->scale_v = resize_op.scale_v;
            this->bilinear_mode_ = resize_op.bilinear_mode_;
            this->is_nchw_ = resize_op.is_nchw_;

            this->resize_op_ = new TimesIntelliResize<uint8_t>(this->in_shape_v, this->out_shape_v, this->scale_v, nullptr);
            this->resize_op_->TimesIntelliInit((BilinearMode)this->bilinear_mode_, this->is_nchw_);
        }
        //print_copy_assigned(this);
        return *this;
    }

    ~py_resize_op_uint8();

private:
    TimesIntelliResize<uint8_t> *resize_op_;
    int coord_mode_;
    int8_t* output_data;
    std::vector<int32_t> in_shape_v, out_shape_v;
    std::vector<float> scale_v;
    int bilinear_mode_;
    bool is_nchw_;
};

class py_resize_op_int16{

public:

    py_resize_op_int16() = delete;

    py_resize_op_int16(const py_int32_t& in_shape, const py_int32_t& out_shape, const py_float_t& scale, int bilinear_mode=0, bool is_nchw=true, int coord_mode=4);

    void forward(const py_int16_t& X, py_int16_t& Y, py_int32_t& in_shape);

    void operator()(const py_int16_t& X, py_int16_t& Y, py_int32_t& in_shape);

    py_resize_op_int16(const py_resize_op_int16& resize_op){
        this->in_shape_v = resize_op.in_shape_v;
        this->out_shape_v = resize_op.out_shape_v;
        this->scale_v = resize_op.scale_v;
        this->bilinear_mode_ = resize_op.bilinear_mode_;
        this->is_nchw_ = resize_op.is_nchw_;

        this->resize_op_ = new TimesIntelliResize<int16_t>(this->in_shape_v, this->out_shape_v, this->scale_v, nullptr);
        this->resize_op_->TimesIntelliInit((BilinearMode)this->bilinear_mode_, this->is_nchw_);
    }

    py_resize_op_int16& operator=(const py_resize_op_int16& resize_op){
        if (this != &resize_op){
            this->in_shape_v = resize_op.in_shape_v;
            this->out_shape_v = resize_op.out_shape_v;
            this->scale_v = resize_op.scale_v;
            this->bilinear_mode_ = resize_op.bilinear_mode_;
            this->is_nchw_ = resize_op.is_nchw_;

            this->resize_op_ = new TimesIntelliResize<int16_t>(this->in_shape_v, this->out_shape_v, this->scale_v, nullptr);
            this->resize_op_->TimesIntelliInit((BilinearMode)this->bilinear_mode_, this->is_nchw_);
        }
        //print_copy_assigned(this);
        return *this;
    }

    /*py_resize_op_int16& operator=(const py_resize_op_int16&& resize_op) noexcept {
        if (this != &resize_op){
            this->in_shape_v = resize_op.in_shape_v;
            this->out_shape_v = resize_op.out_shape_v;
            this->scale_v = resize_op.scale_v;
            this->bilinear_mode_ = resize_op.bilinear_mode_;
            this->is_nchw_ = resize_op.is_nchw_;

            this->resize_op_ = new TimesIntelliResize<int16_t>(this->in_shape_v, this->out_shape_v, this->scale_v, nullptr);
            this->resize_op_->TimesIntelliInit();
        }
        // print_copy_assigned(this);
        return *this;
    }*/

    ~py_resize_op_int16();

private:
    TimesIntelliResize<int16_t> *resize_op_;
    int coord_mode_;
    int16_t* output_data;
    std::vector<int32_t> in_shape_v, out_shape_v;
    std::vector<float> scale_v;
    int bilinear_mode_;
    bool is_nchw_;
};

template class TimesIntelliResize<int8_t>;
template class TimesIntelliResize<int16_t>;

