//
// Created by shengyuan.shen on 2023/4/3.
//

#ifndef C__EIGEN_UTILS_H
#define C__EIGEN_UTILS_H

#include "Eigen/Core"
#include "Eigen/Dense"
#include <iostream>
#include <vector>

template <
        typename I,
        typename std::enable_if<std::is_integral<I>::value, int>::type = 0>
struct integer_iterator : std::iterator<std::input_iterator_tag, I> {
    explicit integer_iterator(I value) : value(value) {}

    I operator*() const {
        return value;
    }

    I const* operator->() const {
        return &value;
    }

    integer_iterator& operator++() {
        ++value;
        return *this;
    }

    integer_iterator operator++(int) {
        const auto copy = *this;
        ++*this;
        return copy;
    }

    bool operator==(const integer_iterator& other) const {
        return value == other.value;
    }

    bool operator!=(const integer_iterator& other) const {
        return value != other.value;
    }

protected:
    I value;
};

template <
        typename I,
        typename std::enable_if<std::is_integral<I>::value, bool>::type = true>
struct integer_range {
public:
    integer_range(I begin, I end) : begin_(begin), end_(end) {}
    ::integer_iterator<I> begin() const {
        return begin_;
    }
    ::integer_iterator<I> end() const {
        return end_;
    }

private:
    ::integer_iterator<I> begin_;
    ::integer_iterator<I> end_;
};

/// Creates an integer range for the half-open interval [begin, end)
/// If end<=begin, then the range is empty.
/// The range has the type of the `end` integer; `begin` integer is
/// cast to this type.
template <
        typename Integer1,
        typename Integer2,
        typename std::enable_if<std::is_integral<Integer1>::value, bool>::type =
        true,
        typename std::enable_if<std::is_integral<Integer2>::value, bool>::type =
        true>
integer_range<Integer2> irange(Integer1 begin, Integer2 end) {
    // If end<=begin then the range is empty; we can achieve this effect by
    // choosing the larger of {begin, end} as the loop terminator
    return {
            static_cast<Integer2>(begin),
            std::max(static_cast<Integer2>(begin), end)};
}

/// Creates an integer range for the half-open interval [0, end)
/// If end<=begin, then the range is empty
template <
        typename Integer,
        typename std::enable_if<std::is_integral<Integer>::value, bool>::type =
        true>
integer_range<Integer> irange(Integer end) {
    // If end<=begin then the range is empty; we can achieve this effect by
    // choosing the larger of {0, end} as the loop terminator
    // Handles the case where end<0. irange only works for ranges >=0
    return {Integer(), std::max(Integer(), end)};
}

// Common Eigen types that we will often use
template <typename T>
using EigenMatrixMap =
Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>;
template <typename T>
using EigenArrayMap =
Eigen::Map<Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>>;
template <typename T>
using EigenVectorMap = Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>>;
template <typename T>
using EigenVectorArrayMap = Eigen::Map<Eigen::Array<T, Eigen::Dynamic, 1>>;
template <typename T>
using ConstEigenMatrixMap =
Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>;
template <typename T>
using ConstEigenArrayMap =
Eigen::Map<const Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>>;
template <typename T>
using ConstEigenVectorMap =
Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>>;
template <typename T>
using ConstEigenVectorArrayMap =
Eigen::Map<const Eigen::Array<T, Eigen::Dynamic, 1>>;

using EigenOuterStride = Eigen::OuterStride<Eigen::Dynamic>;
using EigenInnerStride = Eigen::InnerStride<Eigen::Dynamic>;
using EigenStride = Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>;
template <typename T>
using EigenOuterStridedMatrixMap = Eigen::
Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>, 0, EigenOuterStride>;
template <typename T>
using EigenOuterStridedArrayMap = Eigen::
Map<Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>, 0, EigenOuterStride>;
template <typename T>
using ConstEigenOuterStridedMatrixMap = Eigen::Map<
        const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>,
        0,
        EigenOuterStride>;
template <typename T>
using ConstEigenOuterStridedArrayMap = Eigen::Map<
        const Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>,
        0,
        EigenOuterStride>;
template <typename T>
using EigenStridedMatrixMap = Eigen::
Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>, 0, EigenStride>;
template <typename T>
using EigenStridedArrayMap =
Eigen::Map<Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>, 0, EigenStride>;
template <typename T>
using ConstEigenStridedMatrixMap = Eigen::
Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>, 0, EigenStride>;
template <typename T>
using ConstEigenStridedArrayMap = Eigen::
Map<const Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>, 0, EigenStride>;

// 1-d array
template <typename T>
using EArrXt = Eigen::Array<T, Eigen::Dynamic, 1>;
using EArrXf = Eigen::ArrayXf;
using EArrXd = Eigen::ArrayXd;
using EArrXi = Eigen::ArrayXi;
using EArrXb = EArrXt<bool>;
using EArrXI32 = EArrXt<int32_t>;
using EArrXU16 = EArrXt<uint16_t>;
using EArrXU8 = EArrXt<uint8_t>;
using EArr3U8 = Eigen::Array<uint8_t, 3, 1>;

// 2-d array, column major
template <typename T>
using EArrXXt = Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>;
using EArrXXf = Eigen::ArrayXXf;
using EArrXXI32 = EArrXXt<int32_t>;
using EArrXXU16 = EArrXXt<uint16_t>;
using EArrXXU8 = EArrXXt<uint8_t>;
using EArrXXi = EArrXXt<int>;

// 2-d array, row major
template <typename T>
using ERArrXXt =
Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using ERArrXXf = ERArrXXt<float>;
using ERArrXXI32t = ERArrXXt<int32_t>;
using ERArrXXU16t = ERArrXXt<uint16_t>;
using ERArrXXU8t = ERArrXXt<uint8_t>;
using ERArrXXi = ERArrXXt<int>;
using ERArrXXi64t = ERArrXXt<int64_t>;
using ERArrXXi32t = ERArrXXt<int32_t>;

// 1-d vector
template <typename T>
using EVecXt = Eigen::Matrix<T, Eigen::Dynamic, 1>;
using EVecXd = Eigen::VectorXd;
using EVecXf = Eigen::VectorXf;

// 1-d row vector
using ERVecXd = Eigen::RowVectorXd;
using ERVecXf = Eigen::RowVectorXf;

// 2-d matrix, column major
template <typename T>
using EMatXt = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
using EMatXd = Eigen::MatrixXd;
using EMatXf = Eigen::MatrixXf;
using EMatXU8 = EMatXt<uint8_t>;
using EMatXU16 = EMatXt<uint16_t>;

// 2-d matrix, row major
template <typename T>
using ERMatXt =
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using ERMatXd = ERMatXt<double>;
using ERMatXf = ERMatXt<float>;
using ERMatXU8 = ERMatXt<uint8_t>;


template <typename T>
Eigen::Map<const EArrXt<T>> AsEArrXt(const std::vector<T>& arr) {
    return {arr.data(), static_cast<int>(arr.size())};
}
template <typename T>
Eigen::Map<EArrXt<T>> AsEArrXt(std::vector<T>& arr) {
    return {arr.data(), static_cast<int>(arr.size())};
}

// return a sub array of 'array' based on indices 'indices'
template <class Derived, class Derived1, class Derived2>
void GetSubArray(
        const Eigen::ArrayBase<Derived>& array,
        const Eigen::ArrayBase<Derived1>& indices,
        Eigen::ArrayBase<Derived2>* out_array) {
    CAFFE_ENFORCE_EQ(array.cols(), 1);
    // using T = typename Derived::Scalar;

    out_array->derived().resize(indices.size());
    for (const auto i : ::irange(indices.size())) {
        DCHECK_LT(indices[i], array.size());
        (*out_array)[i] = array[indices[i]];
    }
}

// return a sub array of 'array' based on indices 'indices'
template <class Derived, class Derived1>
EArrXt<typename Derived::Scalar> GetSubArray(
        const Eigen::ArrayBase<Derived>& array,
        const Eigen::ArrayBase<Derived1>& indices) {
    using T = typename Derived::Scalar;
    EArrXt<T> ret(indices.size());
    GetSubArray(array, indices, &ret);
    return ret;
}

// return a sub array of 'array' based on indices 'indices'
template <class Derived>
EArrXt<typename Derived::Scalar> GetSubArray(
        const Eigen::ArrayBase<Derived>& array,
        const std::vector<int>& indices) {
    return GetSubArray(array, AsEArrXt(indices));
}

// return 2d sub array of 'array' based on row indices 'row_indices'
template <class Derived, class Derived1, class Derived2>
void GetSubArrayRows(
        const Eigen::ArrayBase<Derived>& array2d,
        const Eigen::ArrayBase<Derived1>& row_indices,
        Eigen::ArrayBase<Derived2>* out_array) {
    out_array->derived().resize(row_indices.size(), array2d.cols());

    for (const auto i : ::irange(row_indices.size())) {
        DCHECK_LT(row_indices[i], array2d.size());
        out_array->row(i) =
                array2d.row(row_indices[i]).template cast<typename Derived2::Scalar>();
    }
}

// return indices of 1d array for elements evaluated to true
template <class Derived>
std::vector<int> GetArrayIndices(const Eigen::ArrayBase<Derived>& array) {
    std::vector<int> ret;
    for (const auto i : ::irange(array.size())) {
        if (array[i]) {
            ret.push_back(i);
        }
    }
    return ret;
}

#endif //C__EIGEN_UTILS_H
