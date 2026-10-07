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

// Row-wise walk of a non-contiguous view (shared by all element-wise ops): the odometer advances only over the
// outer ndim-1 axes, once per row, and `row(flat, inner)` runs the innermost axis as a plain loop starting at each
// term's offset `flat[t]`. Rows come in row-major order, i.e. the same element order as a contiguous walk.
template <int NT, typename Row>
void for_each_row(std::span<const Index> shape, const std::array<std::span<const Index>, NT>& strides, Row /*lambda*/ row) {
    const Index last = std::ssize(shape) - 1;
    std::array<std::span<const Index>, NT> outer_strides;
    for (int t = 0; t < NT; ++t) outer_strides[t] = strides[t].first(last);
    Odometer<NT> odo(shape.first(last), outer_strides);

    Index rows = 1;
    for (Index axis = 0; axis < last; ++axis) rows *= shape[axis];
    for (Index r = 0; r != rows; ++r) {
        row(odo.flat, shape[last]); /* perform operation across row here */
        odo.next();
    }
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
    } else if (n != 0) { // a zero-extent view has nothing to do
        // Row-wise (see for_each_row): the innermost axis is a plain loop with fixed strides, no odometer logic
        const Index sa = _strides[ndim() - 1], sb = other._strides[ndim() - 1];
        const double* a = _data.get() + _offset;
        const double* b = other._data.get() + other._offset;
        double* out = result.data_ptr(); // fresh and contiguous: rows are consecutive (last stride = 1)

        for_each_row<2>(shape(), {strides(), other.strides()}, 
            [&](const std::array<Index, 2>& flat, Index inner) {
            const double* a_row = a + flat[0];
            const double* b_row = b + flat[1];
            for (Index j = 0; j != inner; ++j) out[j] = operation(a_row[j * sa], b_row[j * sb]);
            out += inner; // manually walk the result flat index (already contiguous)
        });
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
    } else if (n != 0) {
        const Index sa = _strides[ndim() - 1], sb = other._strides[ndim() - 1];
        double* a = _data.get() + _offset;
        const double* b = other._data.get() + other._offset;

        for_each_row<2>(shape(), {strides(), other.strides()},
            [&](const std::array<Index, 2>& flat, Index inner) {
            double* a_row = a + flat[0];
            const double* b_row = b + flat[1];
            for (Index j = 0; j != inner; ++j) a_row[j * sa] = operation(a_row[j * sa], b_row[j * sb]);
        });
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
    } else if (n != 0) {
        const Index sa = _strides[ndim() - 1];
        const double* a = _data.get() + _offset;
        double* out = result.data_ptr();

        for_each_row<1>(shape(), {strides()},
            [&](const std::array<Index, 1>& flat, Index inner) {
            const double* a_row = a + flat[0];
            for (Index j = 0; j != inner; ++j) out[j] = operation(a_row[j * sa]);
            out += inner; // manually walk the result flat index (already contiguous)
        });
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
    } else if (n != 0) {
        const Index sa = _strides[ndim() - 1];
        double* a = _data.get() + _offset;

        for_each_row<1>(shape(), {strides()},
            [&](const std::array<Index, 1>& flat, Index inner) {
            double* a_row = a + flat[0];
            for (Index j = 0; j != inner; ++j) a_row[j * sa] = operation(a_row[j * sa]);
        });
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

std::optional<Dims> Tensor::_view_strides(std::span<const Index> shape, std::span<const Index> strides,
                                          std::span<const Index> new_shape) {
    if (_numel(shape) == 0) return row_major_strides(new_shape); // nothing to address, any strides work

    // Size-1 axes are never stepped along, so their strides mean nothing: drop them first
    Dims sh{}, st{};
    Index nd = 0;
    for (Index k = 0; k < std::ssize(shape); ++k) {
        if (shape[k] == 1) continue;
        sh[nd] = shape[k];
        st[nd] = strides[k];
        ++nd;
    }

    Dims out{};
    const Index new_nd = std::ssize(new_shape);
    Index o = 0, n = 0;
    while (o < nd && n < new_nd) {
        // One run: old axes [o0, o] and new axes [n0, n] holding the same number of elements. Stays in bounds
        // because both shapes hold the same nonzero element count (check outside this function)
        const Index o0 = o, n0 = n;
        Index old_prod = sh[o], new_prod = new_shape[n];
        while (old_prod != new_prod) {
            if (old_prod < new_prod) old_prod *= sh[++o];
            else                     new_prod *= new_shape[++n];
        }

        // Merge check condition: each old axis must step exactly over the whole next one (locally contiguous)
        for (Index k = o0; k < o; ++k) {
            if (st[k] != st[k + 1] * sh[k + 1]) return std::nullopt;
        }

        // New strides: start from the run's innermost old stride, multiply outward (like row_major_strides)
        Index s = st[o];
        for (Index k = n; k >= n0; --k) {
            out[k] = s;
            s *= new_shape[k];
        }
        ++o; ++n;
    }
    for (; n < new_nd; ++n) out[n] = 1; // leftover new axes can only be size 1
    return out;
}

// A view whenever one exists (splitting any axis, or merging axes whose strides line up), so a non-contiguous
// tensor no longer has to be copied just because it isn't contiguous as a whole. Otherwise copies into row-major order.
Tensor Tensor::reshape(std::span<const Index> new_shape) const {
    if (_numel(new_shape) != this->size()) {
        throw std::invalid_argument("New shape is not compatible with current tensor!");
    }
    const Dims dims = _to_dims(new_shape); // throws above MAX_DIM, before _view_strides writes into a Dims

    if (const auto view = _view_strides(shape(), strides(), new_shape)) {
        return Tensor(_data, dims, *view, std::ssize(new_shape), _offset);
    }
    Tensor t = copy(); // not viewable, new copy
    return Tensor(t._data, dims, row_major_strides(new_shape), std::ssize(new_shape), t._offset);
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

    if (n == 0) return total;
    const Index sa = _strides[ndim() - 1];
    const double* a = _data.get() + _offset;
    for_each_row<1>(shape(), {strides()}, [&](const std::array<Index, 1>& flat, Index inner) {
        const double* a_row = a + flat[0];
        for (Index j = 0; j != inner; ++j) total += a_row[j * sa];
    });
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
