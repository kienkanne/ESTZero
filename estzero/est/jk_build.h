#ifndef BUILD_JK
#define BUILD_JK

#include <libint2.hpp>
#include "tensor.h"

struct JK {Matrix J; Matrix K;};

template <typename ERI>
JK build_JK(const ERI &eri, const Matrix &D);

#endif
