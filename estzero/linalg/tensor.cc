#include <algorithm>
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

} // namespace

// == ARITHMETIC HELPERS ================================================================

// Private shape and stride checking method for arithmetic operators.
bool Tensor::_match_check(const Tensor& other) const {
    if (shape != other.shape) {
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
    const int n = this->size();
    std::vector<double> values(n);

    if (fast) {
        for (int i = 0; i != n; ++i) {
            values[i] = operation(this->flat_index(i), other.flat_index(i));
        }
    } else {
        Odometer<2> odo(shape, {this->strides, other.strides}); // terms: this, other

        for (int i = 0; i != n; ++i) {
            values[i] = operation(this->flat_index(odo.flat[0]), other.flat_index(odo.flat[1]));
            odo.next();
        }
    }
    return Tensor(shape, values);
}

// Same fast/general split as _binary_op, but for binary assignments
template <typename Operation>
Tensor& Tensor::_binary_assign_op(const Tensor& other, Operation operation) {
    bool fast = _match_check(other);
    const int n = this->size();

    if (fast) {
        for (int i = 0; i != n; ++i) {
            this->flat_index(i) = operation(
                this->flat_index(i), other.flat_index(i));
        }
    } else {
        Odometer<2> odo(shape, {this->strides, other.strides}); // terms: this, other

        for (int i = 0; i != n; ++i) {
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
    const int n = this->size();
    std::vector<double> values(n);

    if (_is_contiguous()) {
        for (int i = 0; i != n; ++i) {
            values[i] = operation(this->flat_index(i));
        }
    } else {
        Odometer<1> odo(shape, {this->strides});

        for (int i = 0; i != n; ++i) {
            values[i] = operation(this->flat_index(odo.flat[0]));
            odo.next();
        }
    }
    return Tensor(shape, values);
}

template <typename Operation>
Tensor& Tensor::_unary_assign_op(Operation operation) {
    const int n = this->size();

    if (_is_contiguous()) {
        for (int i = 0; i != n; ++i) {
            this->flat_index(i) = operation(this->flat_index(i));
        }
    } else {
        Odometer<1> odo(shape, {this->strides});

        for (int i = 0; i != n; ++i) {
            this->flat_index(odo.flat[0]) = operation(this->flat_index(odo.flat[0]));
            odo.next();
        }
    }
    return *this;
}

// == CONSTRUCTORS =======================================================================

Tensor::Tensor(const std::vector<int>& shape_) : shape(shape_), strides(shape_.size()) {
    size_t total_size = 1;
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
        strides[i] = static_cast<int>(total_size);
        total_size *= static_cast<size_t>(shape[i]);
    }
    data = std::make_shared<std::vector<double>>(total_size);
}

Tensor::Tensor(const std::vector<int>& shape_, const std::vector<double>& values)
    : shape(shape_), strides(shape_.size()) {
    size_t total_size = 1;
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
        strides[i] = static_cast<int>(total_size);
        total_size *= static_cast<size_t>(shape[i]);
    }
    if (values.size() != total_size) {
        throw std::invalid_argument("Tensor data size does not match shape.");
    }
    data = std::make_shared<std::vector<double>>(std::move(values));
}

// == VIEW & COPY METHODS =============================================================

Tensor Tensor::contiguous() const { return _is_contiguous() ? *this : copy(); }

Tensor Tensor::permute(const std::vector<int>& perm) const {
    if (perm.size() != static_cast<size_t>(ndim())) {
        throw std::invalid_argument("Number of indices must be equal to ndim");
    }
    std::vector<int> new_shape(ndim()), new_strides(ndim());
    for (int i = 0; i < ndim(); ++i) {
        new_shape[i] = shape[perm[i]];
        new_strides[i] = strides[perm[i]];
    }
    return Tensor(data, new_shape, new_strides, offset);
}

Tensor Tensor::swapaxes(size_t ax1, size_t ax2) const {
    auto new_shape = shape;
    auto new_strides = strides;
    std::swap(new_shape[ax1], new_shape[ax2]);
    std::swap(new_strides[ax1], new_strides[ax2]);
    return Tensor(data, new_shape, new_strides, offset);
}

Tensor Tensor::select(size_t axis, int index) const {
    auto new_shape = shape;
    auto new_strides = strides;
    size_t new_offset = offset + static_cast<size_t>(index) * static_cast<size_t>(strides[axis]);
    new_shape.erase(new_shape.begin() + axis);
    new_strides.erase(new_strides.begin() + axis);
    return Tensor(data, new_shape, new_strides, new_offset);
}

Tensor Tensor::reshape(const std::vector<int>& new_shape) const {
    int new_size = 1;
    for (int i=0; i < new_shape.size(); ++i) {
        new_size *= new_shape[i];
    }
    if (new_size != this->size()) {
        throw std::invalid_argument("New shape is not compatible with current tensor!");
    }
    Tensor t = contiguous();

    std::vector<int> new_strides(new_shape.size());
    size_t total_size = 1;
    for (int i = static_cast<int>(new_shape.size()) - 1; i >= 0; --i) {
        new_strides[i] = static_cast<int>(total_size);
        total_size *= static_cast<size_t>(new_shape[i]);
    }
    return Tensor(t.data, new_shape, new_strides, t.offset);
}

Tensor Tensor::slice(const std::vector<Sli>& slices) const {
    if (slices.size() != ndim()) {
        throw std::invalid_argument("Number of slices must be equal to ndim");
    }
    auto new_offset = offset;
    auto new_shape = shape;
    auto new_strides = strides;

    for (size_t i = 0; i < ndim(); ++i) {
        int start = slices[i].start;
        int stop = slices[i].stop;
        int step = slices[i].step;

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

Tensor Tensor::copy() const {
    return _unary_op([](double v) { return v; });
}

Tensor& Tensor::assign(const Tensor& other) {
    return _binary_assign_op(other, [](double, double new_val) { return new_val; });
}

Eigen::Map<const Matrix> Tensor::as_matrix(int rows, int cols) const {
    if (!_is_contiguous()) {
        throw std::runtime_error("Current Tensor view is not contiguous! Data might mismatch!");
    }
    if (rows * cols != size()) {
        throw std::runtime_error("Requested rows and cols do not match Tensor size!");
    }
    return Eigen::Map<const Matrix>(data->data() + offset, rows, cols);
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
    const int n = size();
    if (_is_contiguous()) {
        for (int i = 0; i != n; ++i) {
            total += flat_index(i);
        }
        return total;
    }

    Odometer<1> odo(shape, {strides});
    for (int i = 0; i != n; ++i) {
        total += flat_index(odo.flat[0]);
        odo.next();
    }
    return total;
}

// == EIGEN CONVERSIONS ====================================================================

Tensor mat_to_tensor(const Matrix& M) {
    const int rows = static_cast<int>(M.rows());
    const int cols = static_cast<int>(M.cols());

    return Tensor({rows, cols},
        std::vector<double>(M.data(), M.data() + rows * cols));
}

Tensor vec_to_tensor(const Vector& V) {
    return Tensor({static_cast<int>(V.size())},
        std::vector<double>(V.data(), V.data() + V.size()));
}
