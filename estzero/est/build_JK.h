#ifndef BUILD_JK
#define BUILD_JK

#include <integrals.h>
#include <Eigen/Dense>
#include <libint2.hpp>
#include "einsum.h"

struct JK {Matrix J; Matrix K;};

template <typename ERI>
JK build_JK(const ERI &eri, const Matrix &D);

#endif
