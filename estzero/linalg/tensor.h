#ifndef TENSOR_H
#define TENSOR_H

#include <memory>
#include <vector>
#include <cstddef>
#include <stdexcept>
#include <algorithm>
#include <Eigen/Dense>
#include "odometer.h"

// A plain row-major dense array. Previously Eigen's Tensor was only used to
// hold data, so this reimplements just enough shape/indexing/slicing to feed
// a general contraction (see einsum.h) without pulling in Eigen math.

// Use row major Eigen Matrix for easy parsing to Tensor row major
// All other files should reference this Matrix alias
using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using Vector = Eigen::VectorXd;

// For slicing
struct Sli {
    int start = 0;
    int stop = -1;
    int step = 1;
};

// For power operator
inline auto power = [](double base, int exp) {
        double result = 1;
        for (int i = 0; i < exp; ++i) {
            result *= base;
        }
        return result;
    };

// Row-major storage: the last axis has stride one, sweeps fastest.
class Tensor {
    std::shared_ptr<std::vector<double>> data;
    size_t offset = 0; // For slicing

    // Private constructor for creating zero-copy views
    Tensor(std::shared_ptr<std::vector<double>> d, std::vector<int> sh, std::vector<int> st, size_t off)
        : data(d), shape(sh), strides(st), offset(off) {}

    // == ARITHMETIC HELPERS ================================================================

    // Private shape and stride checking method for arithmetic operators.
    bool _match_check(const Tensor& other) const {
        if (shape != other.shape) {
            throw std::invalid_argument(
                "Two tensors must have matching shape for arithmetic operators.");
        }
        return _is_contiguous() && other._is_contiguous();
    }

    // A view is contiguous only when its strides match row-major strides.
    bool _is_contiguous() const {
        size_t expected_stride = 1;
        for (int axis = static_cast<int>(shape.size()) - 1; axis >= 0; --axis) {
            if (strides[axis] != expected_stride) {
                return false;
            }
            expected_stride *= static_cast<size_t>(shape[axis]);
        }
        return true;
    }

    // If a view is contigiuous, a fast path is used that walks directly along the flat index
    // Otherwise, the odometer is used to correctly account for the modified strides
    template <typename Operation>
    Tensor _binary_op(const Tensor& other, Operation operation) const {
        bool fast = _match_check(other);
        std::vector<double> values(this->size());

        if (fast) {
            for (int i = 0; i != this->size(); ++i) {
                values[i] = operation(this->flat_index(i), other.flat_index(i));
            }
        } else {
            std::vector<int> result_indices(this->shape.size(), 0);
            std::vector<int> term_flats {0,0}; //this_flat, term_flat

            for (int flat = 0; flat != this->size(); ++flat) {
                values[flat] = operation(
                    this->flat_index(term_flats[0]), other.flat_index(term_flats[1]));

                odometer(result_indices, term_flats, shape, {this->strides, other.strides});
            }
        }
        return Tensor(shape, values);
    }

    // Same fast/general split as _binary_op, but for binary assignments
    template <typename Operation>
    Tensor& _binary_assign_op(const Tensor& other, Operation operation) {
        bool fast = _match_check(other);

        if (fast) {
            for (int i = 0; i != this->size(); ++i) {
                this->flat_index(i) = operation(
                    this->flat_index(i), other.flat_index(i));
            }
        } else {
            std::vector<int> result_indices(this->shape.size(), 0);
            std::vector<int> term_flats {0, 0}; // this_flat, other_flat

            for (int flat = 0; flat != this->size(); ++flat) {
                this->flat_index(term_flats[0]) = operation(
                    this->flat_index(term_flats[0]), other.flat_index(term_flats[1]));
                odometer(result_indices, term_flats, shape, {this->strides, other.strides});
            }
        }
        return *this;
    }

    // Same fast/general split as _binary_op, but for a single tensor
    template <typename Operation>
    Tensor _unary_op(Operation operation) const {
        std::vector<double> values(this->size());

        if (_is_contiguous()) {
            for (int i = 0; i != this->size(); ++i) {
                values[i] = operation(this->flat_index(i));
            }
        } else {
            std::vector<int> indices(this->shape.size(), 0);
            std::vector<int> this_flat = {0};

            for (int flat = 0; flat != this->size(); ++flat) {
                values[flat] = operation(this->flat_index(this_flat[0]));
                odometer(indices, this_flat, shape, {this->strides});
            }
        }
        return Tensor(shape, values);
    }

    template <typename Operation>
    Tensor& _unary_assign_op(Operation operation) {
        if (_is_contiguous()) {
            for (int i = 0; i != this->size(); ++i) {
                this->flat_index(i) = operation(this->flat_index(i));
            }
        } else {
            std::vector<int> indices(this->shape.size(), 0);
            std::vector<int> this_flat = {0};

            for (int flat = 0; flat != this->size(); ++flat) {
                this->flat_index(this_flat[0]) = operation(this->flat_index(this_flat[0]));
                odometer(indices, this_flat, shape, {this->strides});
            }
        }
        return *this;
    }

public:
    std::vector<int> shape;
    std::vector<int> strides;

    void fill(double value) { std::fill(data->begin(), data->end(), value); }
    void clear() { data->clear(); data->shrink_to_fit(); }
    int ndim() const { return static_cast<int>(shape.size()); }
    int size() const { 
        int size = 1; 
        for (const int& dim : shape) size *= dim; 
        return size;
    }

    // == CONSTRUCTORS =======================================================================

    Tensor() = default;

    explicit Tensor(const std::vector<int>& shape_) : shape(shape_), strides(shape_.size()) {
        size_t total_size = 1;
        for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
            strides[i] = static_cast<int>(total_size);
            total_size *= static_cast<size_t>(shape[i]);
        }
        data = std::make_shared<std::vector<double>>(total_size);
    }

    explicit Tensor(const std::vector<int>& shape_, const std::vector<double>& values)
        : shape(shape_), strides(shape_.size()) {
        size_t total_size = 1;
        for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
            strides[i] = static_cast<int>(total_size);
            total_size *= static_cast<size_t>(shape[i]);
        }
        if (values.size() != total_size) {
            throw std::invalid_argument("Tensor data size does not match shape.");
        }
        data = std::make_shared<std::vector<double>>(values);
    }

    // == ELEMENT ACCESS ==================================================================

    // WARNING: Be careful when using on a strided slice or an axis-permuting swapaxes
    // views walk the buffer in a different order than operator() would, because the 
    // flat index of underlying contiguous data is still the original
    double flat_index(size_t index) const { return (*data)[offset + index]; }
    double& flat_index(size_t index) { return (*data)[offset + index]; }

    // The number of indices must match ndim()
    template <typename... Idx>
    double& operator()(Idx... idx) {
        if (sizeof...(idx) != ndim()) {
            throw std::invalid_argument("Number of indices must be equal to ndim");
        }
        const size_t ind[] = { static_cast<size_t>(idx)... };
        size_t flat = offset;
        for (size_t i = 0; i < ndim(); ++i) {
            flat += ind[i] * strides[i];
        }
        return (*data)[flat];
    }

    template <typename... Idx>
    double operator()(Idx... idx) const {
        if (sizeof...(idx) != ndim()) {
            throw std::invalid_argument("Number of indices must be equal to ndim");
        }
        const size_t ind[] = { static_cast<size_t>(idx)... };
        size_t flat = offset;
        for (size_t i = 0; i < ndim(); ++i) {
            flat += ind[i] * strides[i];
        }
        return (*data)[flat];
    }

    // == VIEW & COPY METHODS =============================================================

    Tensor swapaxes(size_t ax1, size_t ax2) const {
        auto new_shape = shape;
        auto new_strides = strides;
        std::swap(new_shape[ax1], new_shape[ax2]);
        std::swap(new_strides[ax1], new_strides[ax2]);
        return Tensor(data, new_shape, new_strides, offset);
    }

    template <typename... Slices>
    Tensor slice(Slices... slices) const {
        if (sizeof...(slices) != ndim()) {
            throw std::invalid_argument("Number of slices must be equal to ndim");
        }
        Sli slis[] = { slices... }; // Removed 'const' so we can modify default values
        auto new_offset = offset;
        auto new_shape = shape;
        auto new_strides = strides;

        for (size_t i = 0; i < ndim(); ++i) {
            int start = slis[i].start;
            int stop = slis[i].stop;
            int step = slis[i].step;

            if (step == 0) { throw std::invalid_argument("Slice step cannot be zero."); }
            if (stop < 0) { stop = shape[i] + stop + 1; }

            if (step > 0) {
                if (stop < start) stop = start;
                new_shape[i] = (stop - start + step - 1) / step;
            } else {
                if (stop > start) stop = start;
                new_shape[i] = (start - stop - step - 1) / (-step);
            }

            new_offset += start * strides[i];
            new_strides[i] = strides[i] * step;
        }
        return Tensor(data, new_shape, new_strides, new_offset);
    }

    Tensor copy() const {
        return _unary_op([](double v) { return v; });
    }

    Tensor& assign(const Tensor& other) {
        return _binary_assign_op(other, [](double, double new_val) { return new_val; });
    }

    // == ARITHMETIC OPERATIONS =============================================================

    Tensor operator+(const Tensor& other) const {
        return _binary_op(other, [](double lhs, double rhs) { return lhs + rhs; });
    }

    Tensor operator-(const Tensor& other) const {
        return _binary_op(other, [](double lhs, double rhs) { return lhs - rhs; });
    }

    Tensor& operator+=(const Tensor& other) {
        return _binary_assign_op(other, [](double lhs, double rhs) { return lhs + rhs; });
    }

    Tensor& operator-=(const Tensor& other) {
        return _binary_assign_op(other, [](double lhs, double rhs) { return lhs - rhs; });
    }

    Tensor operator*(double scalar) const {
        return _unary_op([scalar](double v) { return v * scalar; });
    }

    friend Tensor operator*(double scalar, const Tensor& tensor) {
        return tensor * scalar;
    }

    Tensor& operator*=(double scalar) {
        return _unary_assign_op([scalar](double v) { return v * scalar; });
    }

    Tensor operator/(double scalar) const {
        return _unary_op([scalar](double v) { return v / scalar; });
    }

    friend Tensor operator/(double scalar, const Tensor& tensor) {
        return tensor._unary_op([scalar](double v) { return scalar / v; });
    }

    Tensor& operator/=(double scalar) {
        return _unary_assign_op([scalar](double v) { return v / scalar; });
    }

    Tensor operator*(const Tensor& other) const {
        return _binary_op(other, [](double lhs, double rhs) { return lhs * rhs; });
    }

    Tensor operator/(const Tensor& other) const {
        return _binary_op(other, [](double lhs, double rhs) { return lhs / rhs; });
    }


    Tensor operator^(int exp) const {
        return _unary_op([exp](double v) { return power(v, exp); });
    }

    double sum() const {
        double total = 0.0;
        if (_is_contiguous()) {
            for (int i = 0; i != size(); ++i) {
                total += flat_index(i);
            }
            return total;
        }

        std::vector<int> indices(shape.size(), 0);
        std::vector<int> tensor_flat {0};
        for (int i = 0; i != size(); ++i) {
            total += flat_index(tensor_flat[0]);
            odometer(indices, tensor_flat, shape, {strides});
        }
        return total;
    }

    // LEGACY: Access raw data pointer, used by TensorView
    const double* data_ptr() const { return data->data(); }
};


// Copy a row-major Eigen Matrix into an owning row-major Tensor.
inline Tensor mat_to_tensor(const Matrix& M) {
    const int rows = static_cast<int>(M.rows());
    const int cols = static_cast<int>(M.cols());

    return Tensor({rows, cols},
        std::vector<double>(M.data(), M.data() + rows * cols));
}


inline Tensor vec_to_tensor(const Vector& V) {
    return Tensor({static_cast<int>(V.size())},
        std::vector<double>(V.data(), V.data() + V.size()));
}


// LEGACY: TensorView was introduced as a reference wrapper in v1.3.1 to fix
// `einsum` memory bug. In v1.6.0, `Tensor` now uses a std::shared_ptr pointing
// to the data vector, so constructing the std::vector<Tensor> in `einsum` would
// just copy the pointer instead of the entire data vector.

// Intermediate `TensorView` class that points directly to `Tensor` data,
// and this lets std::vector owns this lightweight object.
// Another alternative to TensorView is std::vector<std::reference_wrapper<const Tensor>>
class TensorView {
    const double* data_ptr;
public:
    std::vector<int> shape; // Lightweight so just let it copies
    std::vector<int> strides;
    // Constructor: Takes in the `Tensor` class
    TensorView(const Tensor& t) : data_ptr(t.data_ptr()), shape(t.shape), strides(t.strides) {}

    int ndim() const { return static_cast<int>(shape.size()); }

    // Tensor VIEW, so no non-const overload (reference to data) methods
    double flat_index(size_t index) const { return data_ptr[index]; }

    template <typename... Idx>
    double operator()(Idx... idx) const {
        size_t ind[] = { static_cast<size_t>(idx)... };
        size_t flat = 0;
        for (size_t i = 0; i < shape.size(); ++i) {
            flat += ind[i] * static_cast<size_t>(strides[i]);
        }
        return data_ptr[flat];
    }
};

#endif
