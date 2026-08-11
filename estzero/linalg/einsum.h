#ifndef EINSUM_H
#define EINSUM_H

#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <cstddef>
#include <stdexcept>
#include <algorithm>
#include <Eigen/Dense>

// Use row major Eigen Matrix for easy parsing to Tensor row major
// All other files should reference this Matrix alias
using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using Vector = Eigen::VectorXd;

// General-purpose einsum, implemented from scratch (no Eigen math). The Tensor
// below is a plain row-major dense container. Previously Eigen's Tensor was only used
// to hold data, so this reimplements just enough shape/indexing to feed a general contraction

// For slicing
struct Sli {
    int start = 0;
    int stop = -1;
    int step = 1;
};

// Row-major storage: the last axis has stride one, sweeps fastest.
class Tensor {
    std::shared_ptr<std::vector<double>> data;
    size_t offset = 0; // For slicing

    // Private constructor for creating zero-copy views
    Tensor(std::shared_ptr<std::vector<double>> d, std::vector<int> sh, std::vector<int> st, size_t off)
        : data(d), shape(sh), strides(st), offset(off) {}

public:
    std::vector<int> shape;
    std::vector<int> strides;

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

    void fill(double value) { std::fill(data->begin(), data->end(), value); }
    void clear() { data->clear(); data->shrink_to_fit(); }
    int ndim() const { return static_cast<int>(shape.size()); }

    double flat_index(size_t index) const { return data->at(index); }
    double& flat_index(size_t index) { return data->at(index); }

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
        return data->at(flat);
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
        return data->at(flat);
    }

    Tensor swapaxes(size_t ax1, size_t ax2) {
        auto new_shape = shape;
        auto new_strides = strides;
        std::swap(new_shape[ax1], new_shape[ax2]);
        std::swap(new_strides[ax1], new_strides[ax2]);
        return Tensor(data, new_shape, new_strides, offset);
    }

    template <typename... Slices>
    Tensor slice(Slices... slices) {
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
            if (stop == -1) { stop = shape[i]; }

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

    // LEGACY: Access raw data pointer, used by TensorView
    const double* data_ptr() const { return data->data(); }
};


Tensor einsum(const std::string& subscripts, const std::vector<Tensor>& operands);

// Copy a row-major Eigen Matrix into an owning row-major Tensor.
inline Tensor mat_to_tensor(const Matrix& M) {
    const int rows = static_cast<int>(M.rows());
    const int cols = static_cast<int>(M.cols());

    return Tensor({rows, cols}, 
        std::vector<double>(M.data(), M.data() + rows * cols));
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

