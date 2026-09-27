#ifndef ODOMETER
#define ODOMETER

#include <array>
#include <iterator>
#include <span>
#include <stdexcept>
#include <string>
#include "tensor.h" // Index

// Maximum number of axes one odometer can sweep (set to use std::array)
constexpr Index MAX_AXES = 12;

// Walks every combination of a shared set of axes (last axis fastest, matching Tensor's row-major layout),
// and alongside the counter keeps the flat offset of NT "terms" up to date. The swept axes can be einsum's
// distinct index letters, or the common logical shape two Tensors walk together in an elementwise op.
//
// NT (the number of terms) is a template parameter, so it is known at compile time (1-2 for
// Tensor ops, 2-3 for pairwise einsum) to avoid using std::vector
template <int NT>
struct Odometer {
    Index naxes = 0;
    std::array<Index, MAX_AXES> idx{};  // current value of each axis
    std::array<Index, MAX_AXES> dims{}; // extent of each axis

    // stride_table[axis][term]: stride (how much to move when that axis +1) per axis per term. It is 0 if that
    // term doesn't depend on the axis (important for einsum). Stored axis-major, so next() reads every term's 
    // stride for one fixed axis at a time contiguously.
    std::array<std::array<Index, NT>, MAX_AXES> stride_table{};

    std::array<Index, NT> flat{}; // current flat offset of each term

    // einsum: the caller fills `dims` and `stride_table` directly (it relabels each operand's strides onto
    // the global index letters, some of which are zer, so the table isn't a copy of the tensor's strides.
    explicit Odometer(Index naxes_) : naxes(naxes_) {
        if (naxes > MAX_AXES) {
            throw std::invalid_argument(
                "Odometer: " + std::to_string(naxes) + " axes exceeds MAX_AXES ("+ std::to_string(MAX_AXES) + ")");
        }
    }

    // Tensor elementwise ops: use the strides directly. std::span reads both std::vector and std::array
    Odometer(std::span<const Index> shape, const std::array<std::span<const Index>, NT>& term_strides)
        : Odometer(std::ssize(shape)) {
        for (Index axis = 0; axis < naxes; ++axis) {
            dims[axis] = shape[axis];
            for (int term = 0; term < NT; ++term) {
                stride_table[axis][term] = term_strides[term][axis];
            }
        }
    }

    // Advance to the next combination
    void next() {
        for (Index axis = naxes - 1; axis >= 0; --axis) {
            if (++idx[axis] < dims[axis]) {
                for (int term = 0; term < NT; ++term) {
                    flat[term] += stride_table[axis][term];
                }
                return;
            }

            // Reset and move to the next if the bound (axis dimension) is hit
            idx[axis] = 0;
            for (int term = 0; term < NT; ++term) {
                flat[term] -= stride_table[axis][term] * (dims[axis] - 1);
            }
        }
    }
};

#endif
