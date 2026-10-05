#ifndef EINSUM_H
#define EINSUM_H

#include <array>
#include <span>
#include <string_view>
#include "tensor.h"

// einsum's operand list: pointers to the caller's Tensors, no copying.
using Operands = std::span<const Tensor* const>;

// General-purpose einsum, implemented from scratch (no Eigen math), operating on the row-major Tensor container
// defined in tensor.h. Subscripts are parsed once per distinct string and cached (see plan_for in einsum.cc).
Tensor einsum(std::string_view subscripts, Operands operands);

// numpy-style call sites: einsum("ij,jk->ik", A, B). Packs pointers to the operands into a stack array, so
// the operand list costs no heap allocation and no Tensor copies.
template <typename... Ts>
Tensor einsum(std::string_view subscripts, const Ts&... operands) {
    const std::array<const Tensor*, sizeof...(Ts)> ptrs{ &operands... };
    return einsum(subscripts, Operands(ptrs));
}

#endif
