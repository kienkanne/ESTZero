#ifndef EINSUM_H
#define EINSUM_H

#include <string>
#include <vector>
#include <cstddef>
#include <stdexcept>
#include <Eigen/Dense>

using Matrix = Eigen::MatrixXd;

// General-purpose einsum, implemented from scratch (no Eigen math). The Tensor
// below is a plain row-major dense container. Previously Eigen's Tensor was only used
// to hold data, so this reimplements just enough shape/indexing to feed a general contraction

// Row-major storage: the last axis has stride one, sweeps fastest.
class Tensor {
    std::vector<double> data;
public:
    std::vector<int> shape;

    Tensor() = default;

    explicit Tensor(const std::vector<int>& shape_) : shape(shape_) {
        size_t total_size = 1;
        for (int dim : shape) total_size *= static_cast<size_t>(dim);
        data.resize(total_size);
    }

    explicit Tensor(const std::vector<int>& shape_, const std::vector<double>& values)
        : data(values), shape(shape_) {
        size_t total_size = 1;
        for (int dim : shape) total_size *= static_cast<size_t>(dim);
        if (data.size() != total_size) {
            throw std::invalid_argument("Tensor data size does not match shape.");
        }
    }

    void fill(double value) { std::fill(data.begin(), data.end(), value); }

    int ndim() const { return static_cast<int>(shape.size()); }

    // The number of indices must match ndim()
    template <typename... Idx>
    double& operator()(Idx... idx) {
        size_t ind[] = { static_cast<size_t>(idx)... };
        size_t flat = 0, stride = 1;
        for (int i = ndim() - 1; i >= 0; --i) { 
            flat += static_cast<size_t>(ind[i]) * stride; stride *= static_cast<size_t>(shape[i]); }
        return data[flat];
    }
    template <typename... Idx>
    double operator()(Idx... idx) const {
        size_t ind[] = { static_cast<size_t>(idx)... };
        size_t flat = 0, stride = 1;
        for (int i = ndim() - 1; i >= 0; --i) { 
            flat += static_cast<size_t>(ind[i]) * stride; stride *= static_cast<size_t>(shape[i]); }
        return data[flat];
    }

    double flat_index(size_t index) const { return data[index]; }
    double& flat_index(size_t index) { return data[index]; }

    // Access raw data pointer, used by TensorView
    const double* data_ptr() const { return data.data(); }
};


// Fix `einsum` memory bug
class TensorView {
    const double* data_ptr;
public:
    std::vector<int> shape; // Lightweight so just let it copies
    // Constructor: Takes in the `Tensor` class
    TensorView(const Tensor& t) : data_ptr(t.data_ptr()), shape(t.shape) {}

    int ndim() const { return static_cast<int>(shape.size()); }

    // Tensor VIEW, so no non-const overload (reference to data) methods
    double flat_index(size_t index) const { return data_ptr[index]; }

    template <typename... Idx>
    double operator()(Idx... idx) const {
        size_t ind[] = { static_cast<size_t>(idx)... };
        size_t flat = 0, stride = 1;
        for (int i = ndim() - 1; i >= 0; --i) { 
            flat += static_cast<size_t>(ind[i]) * stride; stride *= static_cast<size_t>(shape[i]); }
        return data_ptr[flat];
    }
};

Tensor einsum(const std::string& subscripts, const std::vector<TensorView>& operands);

// Copy an (rows x cols) MO-coefficient block into a row-major Tensor.
// Called by transmo.cc, inefficient as for now (TODO)
inline static Tensor mat_to_tensor(const Matrix& M) {
    const int rows = static_cast<int>(M.rows());
    const int cols = static_cast<int>(M.cols());
    std::vector<double> data(static_cast<size_t>(rows) * cols);
    size_t idx = 0;
    for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
        data[idx++] = M(i, j);
    }}
    return Tensor({rows, cols}, data);
}

#endif
