#ifndef EINSUM_H
#define EINSUM_H

#include <string>
#include <vector>
#include <cblas.h>
#include "tensor.h"

bool try_cutensor(const std::vector<std::string>& terms_strings, const std::vector<Tensor>& operands, Tensor& result);

// General-purpose einsum, implemented from scratch (no Eigen math), operating
// on the row-major Tensor container defined in tensor.h.
Tensor einsum(const std::string& subscripts, const std::vector<Tensor>& operands);

#endif
