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
std::vector<Index> row_major_strides(const std::vector<Index>& shape) {
    std::vector<Index> strides(shape.size());
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
    if (_shape != other._shape) {
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
    std::vector<double> values(n);

    if (fast) {
        for (Index i = 0; i != n; ++i) {
            values[i] = operation(this->flat_index(i), other.flat_index(i));
        }
    } else {
        Odometer<2> odo(_shape, {_strides, other._strides}); // terms: this, other

        for (Index i = 0; i != n; ++i) {
            values[i] = operation(this->flat_index(odo.flat[0]), other.flat_index(odo.flat[1]));
            odo.next();
        }
    }
    return Tensor(_shape, std::move(values));
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
        Odometer<2> odo(_shape, {_strides, other._strides}); // terms: this, other

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
    std::vector<double> values(n);

    if (_is_contiguous()) {
        for (Index i = 0; i != n; ++i) {
            values[i] = operation(this->flat_index(i));
        }
    } else {
        Odometer<1> odo(_shape, {_strides});

        for (Index i = 0; i != n; ++i) {
            values[i] = operation(this->flat_index(odo.flat[0]));
            odo.next();
        }
    }
    return Tensor(_shape, std::move(values));
}

template <typename Operation>
Tensor& Tensor::_unary_assign_op(Operation operation) {
    const Index n = this->size();

    if (_is_contiguous()) {
        for (Index i = 0; i != n; ++i) {
            this->flat_index(i) = operation(this->flat_index(i));
        }
    } else {
        Odometer<1> odo(_shape, {_strides});

        for (Index i = 0; i != n; ++i) {
            this->flat_index(odo.flat[0]) = operation(this->flat_index(odo.flat[0]));
            odo.next();
        }
    }
    return *this;
}

// == CONSTRUCTORS =======================================================================

Tensor::Tensor(const std::vector<Index>& shape_)
    : _shape(shape_), _strides(row_major_strides(_shape)), _size(_numel(_shape)),
      _data(std::make_shared<std::vector<double>>(_size)) {}

Tensor::Tensor(const std::vector<Index>& shape_, std::vector<double> values)
    : _shape(shape_), _strides(row_major_strides(_shape)), _size(_numel(_shape)) {
    if (std::ssize(values) != _size) {
        throw std::invalid_argument("Tensor data size does not match shape.");
    }
    _data = std::make_shared<std::vector<double>>(std::move(values));
}

// == VIEW & COPY METHODS =============================================================

Tensor Tensor::contiguous() const { return _is_contiguous() ? *this : copy(); }

Tensor Tensor::permute(const std::vector<Index>& perm) const {
    if (std::ssize(perm) != ndim()) {
        throw std::invalid_argument("Number of indices must be equal to ndim");
    }
    std::vector<Index> new_shape(ndim());
    std::vector<Index> new_strides(ndim());
    for (Index i = 0; i < ndim(); ++i) {
        new_shape[i] = _shape[perm[i]];
        new_strides[i] = _strides[perm[i]];
    }
    return Tensor(_data, new_shape, new_strides, _offset);
}

Tensor Tensor::swapaxes(Index ax1, Index ax2) const {
    auto new_shape = _shape;
    auto new_strides = _strides;
    std::swap(new_shape[ax1], new_shape[ax2]);
    std::swap(new_strides[ax1], new_strides[ax2]);
    return Tensor(_data, new_shape, new_strides, _offset);
}

Tensor Tensor::select(Index axis, Index index) const {
    auto new_shape = _shape;
    auto new_strides = _strides;
    Index new_offset = _offset + index * _strides[axis];
    new_shape.erase(new_shape.begin() + axis);
    new_strides.erase(new_strides.begin() + axis);
    return Tensor(_data, new_shape, new_strides, new_offset);
}

Tensor Tensor::reshape(const std::vector<Index>& new_shape) const {
    if (_numel(new_shape) != this->size()) {
        throw std::invalid_argument("New shape is not compatible with current tensor!");
    }
    Tensor t = contiguous();
    return Tensor(t._data, new_shape, row_major_strides(new_shape), t._offset);
}

Tensor Tensor::slice(const std::vector<Sli>& slices) const {
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
    return Tensor(_data, new_shape, new_strides, new_offset);
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

Eigen::Map<const Matrix> Tensor::as_matrix(Index rows, Index cols) const {
    if (!_is_contiguous()) {
        throw std::runtime_error("Current Tensor view is not contiguous! Data might mismatch!");
    }
    if (rows * cols != size()) {
        throw std::runtime_error("Requested rows and cols do not match Tensor size!");
    }
    return Eigen::Map<const Matrix>(_data->data() + _offset, rows, cols);
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

    Odometer<1> odo(_shape, {_strides});
    for (Index i = 0; i != n; ++i) {
        total += flat_index(odo.flat[0]);
        odo.next();
    }
    return total;
}

// == EIGEN CONVERSIONS ====================================================================

Tensor mat_to_tensor(const Matrix& M) {
    return Tensor({M.rows(), M.cols()}, std::vector<double>(M.data(), M.data() + M.size()));
}

Tensor vec_to_tensor(const Vector& V) {
    return Tensor({V.size()}, std::vector<double>(V.data(), V.data() + V.size()));
}
