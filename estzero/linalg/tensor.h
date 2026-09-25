#ifndef TENSOR_H
#define TENSOR_H

#include <memory>
#include <vector>
#include <cstddef>
#include <stdexcept>
#include <Eigen/Dense>


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

// A plain row-major dense array (the last axis has stride one, sweeps fastest.). Previously Eigen's Tensor 
// was only used to hold data, so this reimplements just enough shape/indexing/slicing to feed a general contraction 
// (see einsum.h) without pulling in Eigen math.
//
// The per-element accessors are kept inline instead of moving to `.cc`. Those are called in hot loops and are very 
// cheap, so they are inlined here for compiler visbility to avoid any per-call overhead. 
class Tensor {
    std::shared_ptr<std::vector<double>> data;
    size_t offset = 0; // For slicing

    // Private constructor for creating zero-copy views
    Tensor(std::shared_ptr<std::vector<double>> d, std::vector<int> sh, std::vector<int> st, size_t off)
        : data(d), shape(sh), strides(st), offset(off) {}

    // == ARITHMETIC HELPERS (defined in tensor.cc) =========================================

    bool _match_check(const Tensor& other) const;

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

    template <typename Operation> Tensor _binary_op(const Tensor& other, Operation operation) const;
    template <typename Operation> Tensor& _binary_assign_op(const Tensor& other, Operation operation);
    template <typename Operation> Tensor _unary_op(Operation operation) const;
    template <typename Operation> Tensor& _unary_assign_op(Operation operation);

public:
    std::vector<int> shape;
    std::vector<int> strides;

    void fill(double value) { std::fill(data->begin(), data->end(), value); };
    void clear() { data->clear(); data->shrink_to_fit(); };
    int ndim() const { return static_cast<int>(shape.size()); }
    int size() const {
        int size = 1;
        for (const int& dim : shape) size *= dim;
        return size;
    }

    // == CONSTRUCTORS =======================================================================

    Tensor() = default;
    explicit Tensor(const std::vector<int>& shape_);
    explicit Tensor(const std::vector<int>& shape_, const std::vector<double>& values);

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

    Tensor contiguous() const;

    template <typename... Idx>
    Tensor permute(Idx... idx) const { return permute(std::vector<int>{ static_cast<int>(idx)... }); }
    Tensor permute(const std::vector<int>& perm) const;

    Tensor swapaxes(size_t ax1, size_t ax2) const;

    // Fix one axis to a single index, dropping that axis entirely, which is similar to numpy's fancy
    // indexing (t[..., idx, ...]), returning a smaller zero-copy view `Tensor`.
    Tensor select(size_t axis, int index) const;

    template <typename... Idx>
    Tensor reshape(Idx... idx) const { return reshape(std::vector<int>{ static_cast<int>(idx)... }); }
    Tensor reshape(const std::vector<int>& new_shape) const;

    template <typename... Slices>
    Tensor slice(Slices... slices) const { return slice(std::vector<Sli>{ slices... }); };
    Tensor slice(const std::vector<Sli>& slices) const;

    Tensor copy() const;
    Tensor& assign(const Tensor& other);

    // Zero-copy view of a contiguous Tensor's buffer as an row-major Eigen matrix, for GEMM.
    // WARNING: Only safe to use when this this is already packed row-major (straight out of reshape())
    Eigen::Map<const Matrix> as_matrix(int rows, int cols) const;

    // Get raw data pointer (+ offset). Contiguous is still checked for safety
    double* data_ptr() const {
        if (!_is_contiguous()) {
            throw std::runtime_error("Current Tensor view is not contiguous! Data might mismatch!");
        }
        return data->data() + offset; }

    // == ARITHMETIC OPERATIONS (defined in tensor.cc) ======================================

    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor& operator+=(const Tensor& other);
    Tensor& operator-=(const Tensor& other);

    Tensor operator*(double scalar) const;
    friend Tensor operator*(double scalar, const Tensor& tensor);
    Tensor& operator*=(double scalar);
    Tensor operator*(const Tensor& other) const;
    Tensor& operator*=(const Tensor& other);

    Tensor operator/(double scalar) const;
    friend Tensor operator/(double scalar, const Tensor& tensor);
    Tensor& operator/=(double scalar);
    Tensor operator/(const Tensor& other) const;
    Tensor& operator/=(const Tensor& other);

    Tensor operator^(int exp) const;

    double sum() const;
};


// Copy a row-major Eigen Matrix into an owning row-major Tensor.
Tensor mat_to_tensor(const Matrix& M);
Tensor vec_to_tensor(const Vector& V);


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
