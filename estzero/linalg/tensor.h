#ifndef TENSOR_H
#define TENSOR_H

#include <array>
#include <cstddef>
#include <iterator>
#include <memory>
#include <span>
#include <stdexcept>
#include <vector>
#include <Eigen/Dense>


// Use row major Eigen Matrix for easy parsing to Tensor row major
// All other files should reference this Matrix alias
using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using Vector = Eigen::VectorXd;

// One signed 64-bit type for everything that measures or addresses Tensor data: extents, strides, offsets,
// flat indices, element counts, and axis numbers.
using Index = std::ptrdiff_t;

// For slicing
struct Sli {
    Index start = 0;
    Index stop = -1;
    Index step = 1;
};

// A plain row-major dense array (the last axis has stride one, sweeps fastest.). Previously Eigen's Tensor 
// was only used to hold data, so this reimplements just enough shape/indexing/slicing to feed a general contraction 
// (see einsum.h) without pulling in Eigen math.
//
// The per-element accessors are kept inline instead of moving to `.cc`. Those are called in hot loops and are very 
// cheap, so they are inlined here for compiler visbility to avoid any per-call overhead. 
class Tensor {
    // All state is private and read through the const getters below, so `_size` can never go stale: `_shape`
    // only changes by building a new Tensor (every constructor sets `_size`).
    std::vector<Index> _shape;
    std::vector<Index> _strides;
    Index _size = 0;   // Cached product of `_shape`. 0 only for an empty (default-constructed or cleared) Tensor
    std::shared_ptr<std::vector<double>> _data;
    Index _offset = 0; // For slicing

    // Private constructor for creating zero-copy views. 
    // View ops (permute/swapaxes/select/reshape/slice) comes through here
    Tensor(std::shared_ptr<std::vector<double>> d, std::vector<Index> sh, std::vector<Index> st, Index off)
        : _shape(std::move(sh)), _strides(std::move(st)), _size(_numel(_shape)), _data(std::move(d)), _offset(off) {}

    // Compute size (num element) from new shape for view ops
    static Index _numel(std::span<const Index> sh) {
        Index n = 1;
        for (Index dim : sh) n *= dim;
        return n;
    }

    // Flat buffer position of element (idx...), shared by both operator() overloads.
    template <typename... Idx>
    Index _offset_of(Idx... idx) const {
        if (sizeof...(idx) != _shape.size()) {
            throw std::invalid_argument("Number of indices must be equal to ndim");
        }
        const Index ind[] = { static_cast<Index>(idx)... };
        Index flat = _offset;
        for (Index i = 0; i < ndim(); ++i) {
            flat += ind[i] * _strides[i];
        }
        return flat;
    }

    // == ARITHMETIC HELPERS (defined in tensor.cc) =========================================

    bool _match_check(const Tensor& other) const;

    // A view is contiguous only when its strides match row-major strides.
    bool _is_contiguous() const {
        Index expected_stride = 1;
        for (Index axis = ndim() - 1; axis >= 0; --axis) {
            if (_strides[axis] != expected_stride) {
                return false;
            }
            expected_stride *= _shape[axis];
        }
        return true;
    }

    template <typename Operation> Tensor _binary_op(const Tensor& other, Operation operation) const;
    template <typename Operation> Tensor& _binary_assign_op(const Tensor& other, Operation operation);
    template <typename Operation> Tensor _unary_op(Operation operation) const;
    template <typename Operation> Tensor& _unary_assign_op(Operation operation);

public:
    const std::vector<Index>& shape() const { return _shape; }
    const std::vector<Index>& strides() const { return _strides; }
    Index ndim() const { return std::ssize(_shape); }
    Index size() const { return _size; }

    // Sets only the elements this view covers (a slice shares its parent's buffer)
    void fill(double value);
    // Drops this Tensor's reference to the buffer, leaving it empty (size 0, ndim 0).
    void clear() { *this = Tensor(); }

    // == CONSTRUCTORS =======================================================================

    Tensor() = default;
    // Both by value in combination with std::move to avoid unnecessary copies
    explicit Tensor(std::vector<Index> shape_);
    explicit Tensor(std::vector<Index> shape_, std::vector<double> values);

    // == ELEMENT ACCESS ==================================================================

    // WARNING: Be careful when using on a strided slice or an axis-permuting swapaxes
    // views walk the buffer in a different order than operator() would, because the
    // flat index of underlying contiguous data is still the original
    double flat_index(Index index) const { return (*_data)[_offset + index]; }
    double& flat_index(Index index) { return (*_data)[_offset + index]; }

    // The number of indices must match ndim()
    template <typename... Idx>
    double& operator()(Idx... idx) { return (*_data)[_offset_of(idx...)]; }

    template <typename... Idx>
    double operator()(Idx... idx) const { return (*_data)[_offset_of(idx...)]; }

    // == VIEW & COPY METHODS =============================================================

    Tensor contiguous() const;

    // permute/reshape/slice: call sites pass the list directly (t.permute(0, 2, 1, 3), t.slice(O, V)). The
    // variadic front end packs it into a stack array, so no additional heap allocation.
    template <typename... I>
    Tensor permute(I... axes) const {
        const std::array<Index, sizeof...(I)> perm{ static_cast<Index>(axes)... };
        return permute(std::span<const Index>(perm));
    }
    Tensor permute(std::span<const Index> perm) const;

    Tensor swapaxes(Index ax1, Index ax2) const;

    // Fix one axis to a single index, dropping that axis entirely, which is similar to numpy's fancy
    // indexing (t[..., idx, ...]), returning a smaller zero-copy view `Tensor`.
    Tensor select(Index axis, Index index) const;

    template <typename... I>
    Tensor reshape(I... extents) const {
        const std::array<Index, sizeof...(I)> new_shape{ static_cast<Index>(extents)... };
        return reshape(std::span<const Index>(new_shape));
    }
    Tensor reshape(std::span<const Index> new_shape) const;

    template <typename... S>
    Tensor slice(S... slices) const {
        const std::array<Sli, sizeof...(S)> all{ slices... };
        return slice(std::span<const Sli>(all));
    }
    Tensor slice(std::span<const Sli> slices) const;

    Tensor copy() const;
    Tensor& assign(const Tensor& other);

    // Get raw data pointer (+ offset). Contiguous is still checked for safety
    double* data_ptr() const {
        if (!_is_contiguous()) {
            throw std::runtime_error("Current Tensor view is not contiguous! Data might mismatch!");
        }
        return _data->data() + _offset; }

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

#endif
