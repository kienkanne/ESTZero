#ifndef EINSUM_H
#define EINSUM_H

#include <string>
#include <vector>

#include "tensor.h"

enum class GEMMScheme { EIGEN, BLAS };
inline GEMMScheme gs;

inline void set_gemm_scheme(std::string g) {
    if (g == "eigen") gs = GEMMScheme::EIGEN;
    else if (g == "blas") gs = GEMMScheme::BLAS;
    else throw std::runtime_error("Unknown gemm-scheme: " + g);
}

// General-purpose einsum, implemented from scratch (no Eigen math), operating
// on the row-major Tensor container defined in tensor.h.
Tensor einsum(const std::string& subscripts, const std::vector<Tensor>& operands);

#endif
