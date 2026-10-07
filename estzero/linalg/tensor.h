#ifndef TENSOR_H
#define TENSOR_H

#include <algorithm>
#include <array>
#include <cstddef>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>
#include <Eigen/Dense>


// Use row major Eigen Matrix for easy parsing to Tensor row major
// All other files should reference this Matrix alias
using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using Vector = Eigen::VectorXd;

// One signed 64-bit type for everything that measures or addresses Tensor data: extents, strides, offsets,
// flat indices, element counts, and axis numbers.
using Index = std::ptrdiff_t;

// Maximum rank of a Tensor. shape/strides live in fixed-size arrays (no heap allocation per view or copy)
// 6 is the highest rank used anywhere (the full T3 tensor in calc_pert_triples_noddy).
constexpr Index MAX_DIM = 6;
using Dims = std::array<Index, MAX_DIM>;

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
    // All state is private and read through the const getters below.
    //
    // `_shape`/`_strides` are fixed-size: only the first `_ndim` entries are meaningful, the rest stay 0. 
    // They must go through ndim(), or the shape()/strides() spans, which are exactly ndim long, to avoid errors.
    Dims _shape{};
    Dims _strides{};
    Index _ndim = 0;
    Index _size = 0;
    std::shared_ptr<double[]> _data;
    Index _offset = 0; // For slicing

    // Private constructor for creating zero-copy views.
    // View ops (permute/swapaxes/select/reshape/slice) comes through here
    Tensor(std::shared_ptr<double[]> d, const Dims& sh, const Dims& st, Index nd, Index off)
        : _shape(sh), _strides(st), _ndim(nd), _size(_numel(shape())), _data(std::move(d)), _offset(off) {}

    // Copies a shape into the fixed-size array, rejecting a rank above MAX_DIM
    static Dims _to_dims(std::span<const Index> sh) {
        if (std::ssize(sh) > MAX_DIM) {
            throw std::invalid_argument("Tensor rank " + std::to_string(sh.size()) + " exceeds MAX_DIM (" +
                                        std::to_string(MAX_DIM) + ")");
        }
        Dims dims{};
        std::copy(sh.begin(), sh.end(), dims.begin());
        return dims;
    }

    // Compute size (num element) from new shape for view ops
    static Index _numel(std::span<const Index> sh) {
        Index n = 1;
        for (Index dim : sh) n *= dim;
        return n;
    }

    // Flat buffer position of element (idx...), shared by both operator() overloads.
    template <typename... Idx>
    Index _offset_of(Idx... idx) const {
        if (static_cast<Index>(sizeof...(idx)) != _ndim) {
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
    // Strides that view (shape, strides) as new_shape without copying, or nullopt when no view exists
    static std::optional<Dims> _view_strides(std::span<const Index> shape, std::span<const Index> strides,
                                             std::span<const Index> new_shape);

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
    // std::span to return ndim() long
    std::span<const Index> shape() const { return { _shape.data(), static_cast<size_t>(_ndim) }; }
    std::span<const Index> strides() const { return { _strides.data(), static_cast<size_t>(_ndim) }; }
    Index ndim() const { return _ndim; }
    Index size() const { return _size; }

    // Sets only the elements this view covers (a slice shares its parent's buffer)
    void fill(double value);
    // Drops this Tensor's reference to the buffer, leaving it empty (size 0, ndim 0).
    void clear() { *this = Tensor(); }

    // == CONSTRUCTORS =======================================================================

    Tensor() = default;
    // The span overloads take any runtime shape (another tensor's shape(), a std::vector)
    // The initializer_list overloads take a braced shape, Tensor({o, v}); span can't bind to before C++26.
    // These are similar to np.zeros (default initialize to fill zero)
    explicit Tensor(std::span<const Index> shape_);
    explicit Tensor(std::initializer_list<Index> shape_)
        : Tensor(std::span<const Index>(shape_.begin(), shape_.size())) {}

    explicit Tensor(std::span<const Index> shape_, std::span<const double> values);
    explicit Tensor(std::initializer_list<Index> shape_, std::span<const double> values)
        : Tensor(std::span<const Index>(shape_.begin(), shape_.size()), values) {}

    // Tests generate the values manually via brace lists
    explicit Tensor(std::initializer_list<Index> shape_, std::initializer_list<double> values)
        : Tensor(std::span<const Index>(shape_.begin(), shape_.size()),
                 std::span<const double>(values.begin(), values.size())) {}

    // Use std::make_shared_for_overwrite<double[]>(n) to left uninitilized (only safe when overwrite right away)
    // Use static member function so .empty() can't be called on existing Tensor
    // Similar to np.empty
    static Tensor empty(std::span<const Index> shape_);
    static Tensor empty(std::initializer_list<Index> shape_) {
        return Tensor::empty(std::span<const Index>(shape_.begin(), shape_.size()));
    }

    // == ELEMENT ACCESS ==================================================================

    // WARNING: Be careful when using on a strided slice or an axis-permuting swapaxes
    // views walk the buffer in a different order than operator() would, because the
    // flat index of underlying contiguous data is still the original
    double flat_index(Index index) const { return _data.get()[_offset + index]; }
    double& flat_index(Index index) { return _data.get()[_offset + index]; }

    // The number of indices must match ndim()
    template <typename... Idx>
    double& operator()(Idx... idx) { return _data.get()[_offset_of(idx...)]; }

    template <typename... Idx>
    double operator()(Idx... idx) const { return _data.get()[_offset_of(idx...)]; }

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

    // Raw pointer to the view's first element (buffer + offset). Not checked for contiguity: the caller reads
    // the layout from strides() (e.g. a BLAS leading dimension), or calls contiguous() first.
    double* data_ptr() const { return _data.get() + _offset; }

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
