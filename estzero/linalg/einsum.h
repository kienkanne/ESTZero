#ifndef EINSUM_H
#define EINSUM_H

#include <string>
#include <vector>

#include "tensor.h"

// General-purpose einsum, implemented from scratch (no Eigen math), operating
// on the row-major Tensor container defined in tensor.h.
Tensor einsum(const std::string& subscripts, const std::vector<Tensor>& operands);

#endif
