#include <algorithm>
#include <iterator> // for std::size to return signed type, replace static_cast
#include "tensor.h"
#include "odometer.h"

namespace {

// For power operator
inline auto power = [](double base, int exp) {
        double result = 1;
        for (int i = 0; i < exp; ++i) {
            result *= base;
        }
        return result;
    };

// Natural row-major strides for `shape` (last axis contiguous)
Dims row_major_strides(std::span<const Index> shape) {
    if (std::ssize(shape) > MAX_DIM) { // checked here too
        throw std::invalid_argument("Tensor rank " + std::to_string(shape.size()) + " exceeds MAX_DIM");
    }
    Dims strides{};
    Index running = 1;
    for (Index axis = std::ssize(shape) - 1; axis >= 0; --axis) {
        strides[axis] = running;
        running *= shape[axis];
    }
    return strides;
}

} // namespace

// == ARITHMETIC HELPERS ================================================================

// Private shape and stride checking method for arithmetic operators.
bool Tensor::_match_check(const Tensor& other) const {
    // C++20 feature for safe checking between two ranges
    if (!std::ranges::equal(shape(), other.shape())) {
        throw std::invalid_argument(
            "Two tensors must have matching shape for arithmetic operators.");
    }
    return _is_contiguous() && other._is_contiguous();
}

// If a view is contigiuous, a fast path is used that walks directly along the flat index
// Otherwise, the odometer is used to correctly account for the modified strides
template <typename Operation>
Tensor Tensor::_binary_op(const Tensor& other, Operation operation) const {
    bool fast = _match_check(other);
    const Index n = this->size();

    Tensor result = Tensor::empty(shape());

    if (fast) {
        for (Index i = 0; i != n; ++i) {
            result.flat_index(i) = operation(this->flat_index(i), other.flat_index(i));
        }
    } else {
        Odometer<2> odo(shape(), {strides(), other.strides()}); // terms: this, other

        for (Index i = 0; i != n; ++i) {
            result.flat_index(i) = operation(this->flat_index(odo.flat[0]), other.flat_index(odo.flat[1]));
            odo.next();
        }
    }
    return result;
}

// Same fast/general split as _binary_op, but for binary assignments
template <typename Operation>
Tensor& Tensor::_binary_assign_op(const Tensor& other, Operation operation) {
    bool fast = _match_check(other);
    const Index n = this->size();

    if (fast) {
        for (Index i = 0; i != n; ++i) {
            this->flat_index(i) = operation(
                this->flat_index(i), other.flat_index(i));
        }
    } else {
        Odometer<2> odo(shape(), {strides(), other.strides()}); // terms: this, other

        for (Index i = 0; i != n; ++i) {
            this->flat_index(odo.flat[0]) = operation(
                this->flat_index(odo.flat[0]), other.flat_index(odo.flat[1]));
            odo.next();
        }
    }
    return *this;
}

// Same fast/general split as _binary_op, but for a single tensor
template <typename Operation>
Tensor Tensor::_unary_op(Operation operation) const {
    const Index n = this->size();
    
    Tensor result = Tensor::empty(shape());

    if (_is_contiguous()) {
        for (Index i = 0; i != n; ++i) {
            result.flat_index(i) = operation(this->flat_index(i));
        }
    } else {
        Odometer<1> odo(shape(), {strides()});

        for (Index i = 0; i != n; ++i) {
            result.flat_index(i) = operation(this->flat_index(odo.flat[0]));
            odo.next();
        }
    }
    return result;
}

template <typename Operation>
Tensor& Tensor::_unary_assign_op(Operation operation) {
    const Index n = this->size();

    if (_is_contiguous()) {
        for (Index i = 0; i != n; ++i) {
            this->flat_index(i) = operation(this->flat_index(i));
        }
    } else {
        Odometer<1> odo(shape(), {strides()});

        for (Index i = 0; i != n; ++i) {
            this->flat_index(odo.flat[0]) = operation(this->flat_index(odo.flat[0]));
            odo.next();
        }
    }
    return *this;
}

// == CONSTRUCTORS =======================================================================

// _to_dims runs first (declaration order), so a rank above MAX_DIM throws before anything else is computed
Tensor::Tensor(std::span<const Index> shape_)
    : _shape(_to_dims(shape_)), _strides(row_major_strides(shape_)), _ndim(std::ssize(shape_)),
      _size(_numel(shape_)), _data(std::make_shared<double[]>(_size)) {}

Tensor::Tensor(std::span<const Index> shape_, std::span<const double> values)
    : _shape(_to_dims(shape_)), _strides(row_major_strides(shape_)), _ndim(std::ssize(shape_)),
      _size(_numel(shape_)) {
    if (std::ssize(values) != _size) {
        throw std::invalid_argument("Tensor data size does not match shape.");
    }
    _data = std::make_shared_for_overwrite<double[]>(_size); // every element is written by the copy below
    std::copy(values.begin(), values.end(), _data.get());
}

Tensor Tensor::empty(std::span<const Index> shape_) {
    return Tensor(std::make_shared_for_overwrite<double[]>(_numel(shape_)), 
                _to_dims(shape_), 
                row_major_strides(shape_), 
                std::ssize(shape_), 0);
}

// == VIEW & COPY METHODS =============================================================

Tensor Tensor::contiguous() const { return _is_contiguous() ? *this : copy(); }

Tensor Tensor::permute(std::span<const Index> perm) const {
    if (std::ssize(perm) != ndim()) {
        throw std::invalid_argument("Number of indices must be equal to ndim");
    }
    Dims new_shape{}, new_strides{};
    for (Index i = 0; i < ndim(); ++i) {
        new_shape[i] = _shape[perm[i]];
        new_strides[i] = _strides[perm[i]];
    }
    return Tensor(_data, new_shape, new_strides, _ndim, _offset);
}

Tensor Tensor::swapaxes(Index ax1, Index ax2) const {
    auto new_shape = _shape;
    auto new_strides = _strides;
    std::swap(new_shape[ax1], new_shape[ax2]);
    std::swap(new_strides[ax1], new_strides[ax2]);
    return Tensor(_data, new_shape, new_strides, _ndim, _offset);
}

Tensor Tensor::select(Index axis, Index index) const {
    auto new_shape = _shape;
    auto new_strides = _strides;
    Index new_offset = _offset + index * _strides[axis];
    // Drop the axis: shift the later axes left by one, then clear the freed last slot
    for (Index i = axis; i + 1 < ndim(); ++i) {
        new_shape[i] = new_shape[i + 1];
        new_strides[i] = new_strides[i + 1];
    }
    new_shape[ndim() - 1] = 0;
    new_strides[ndim() - 1] = 0;
    return Tensor(_data, new_shape, new_strides, _ndim - 1, new_offset);
}

Tensor Tensor::reshape(std::span<const Index> new_shape) const {
    if (_numel(new_shape) != this->size()) {
        throw std::invalid_argument("New shape is not compatible with current tensor!");
    }
    Tensor t = contiguous();
    return Tensor(t._data, _to_dims(new_shape), row_major_strides(new_shape), std::ssize(new_shape), t._offset);
}

Tensor Tensor::slice(std::span<const Sli> slices) const {
    if (std::ssize(slices) != ndim()) {
        throw std::invalid_argument("Number of slices must be equal to ndim");
    }
    auto new_offset = _offset;
    auto new_shape = _shape;
    auto new_strides = _strides;

    for (Index i = 0; i < ndim(); ++i) {
        auto [start, stop, step] = slices[i];

        if (step == 0) { throw std::invalid_argument("Slice step cannot be zero."); }
        if (stop < 0) { stop = _shape[i] + stop + 1; }

        if (step > 0) {
            if (stop < start) stop = start;
            new_shape[i] = (stop - start + step - 1) / step;
        } else {
            if (stop > start) stop = start;
            new_shape[i] = (start - stop - step - 1) / (-step);
        }

        new_offset += start * _strides[i];
        new_strides[i] = _strides[i] * step;
    }
    return Tensor(_data, new_shape, new_strides, _ndim, new_offset);
}

Tensor Tensor::copy() const {
    return _unary_op([](double v) { return v; });
}

Tensor& Tensor::assign(const Tensor& other) {
    return _binary_assign_op(other, [](double, double new_val) { return new_val; });
}

void Tensor::fill(double value) {
    _unary_assign_op([value](double) { return value; });
}

// == ARITHMETIC OPERATIONS =============================================================

Tensor Tensor::operator+(const Tensor& other) const {
    return _binary_op(other, [](double lhs, double rhs) { return lhs + rhs; });
}

Tensor Tensor::operator-(const Tensor& other) const {
    return _binary_op(other, [](double lhs, double rhs) { return lhs - rhs; });
}

Tensor& Tensor::operator+=(const Tensor& other) {
    return _binary_assign_op(other, [](double lhs, double rhs) { return lhs + rhs; });
}

Tensor& Tensor::operator-=(const Tensor& other) {
    return _binary_assign_op(other, [](double lhs, double rhs) { return lhs - rhs; });
}

Tensor Tensor::operator*(double scalar) const {
    return _unary_op([scalar](double v) { return v * scalar; });
}

Tensor operator*(double scalar, const Tensor& tensor) {
    return tensor * scalar;
}

Tensor& Tensor::operator*=(double scalar) {
    return _unary_assign_op([scalar](double v) { return v * scalar; });
}

Tensor Tensor::operator*(const Tensor& other) const {
    return _binary_op(other, [](double lhs, double rhs) { return lhs * rhs; });
}

Tensor& Tensor::operator*=(const Tensor& other) {
    return _binary_assign_op(other, [](double lhs, double rhs) { return lhs * rhs; });
}

Tensor Tensor::operator/(double scalar) const {
    return _unary_op([scalar](double v) { return v / scalar; });
}

Tensor operator/(double scalar, const Tensor& tensor) {
    return tensor._unary_op([scalar](double v) { return scalar / v; });
}

Tensor& Tensor::operator/=(double scalar) {
    return _unary_assign_op([scalar](double v) { return v / scalar; });
}

Tensor Tensor::operator/(const Tensor& other) const {
    return _binary_op(other, [](double lhs, double rhs) { return lhs / rhs; });
}

Tensor& Tensor::operator/=(const Tensor& other) {
    return _binary_assign_op(other, [](double lhs, double rhs) { return lhs / rhs; });
}

Tensor Tensor::operator^(int exp) const {
    return _unary_op([exp](double v) { return power(v, exp); });
}

double Tensor::sum() const {
    double total = 0.0;
    const Index n = size();
    if (_is_contiguous()) {
        for (Index i = 0; i != n; ++i) {
            total += flat_index(i);
        }
        return total;
    }

    Odometer<1> odo(shape(), {strides()});
    for (Index i = 0; i != n; ++i) {
        total += flat_index(odo.flat[0]);
        odo.next();
    }
    return total;
}

// == EIGEN CONVERSIONS ====================================================================

// Copies (an Eigen matrix never gives up its buffer). Matrix is row-major, so its data is already in Tensor order.
Tensor mat_to_tensor(const Matrix& M) {
    return Tensor({M.rows(), M.cols()}, std::span<const double>(M.data(), M.size()));
}

Tensor vec_to_tensor(const Vector& V) {
    return Tensor({V.size()}, std::span<const double>(V.data(), V.size()));
}
