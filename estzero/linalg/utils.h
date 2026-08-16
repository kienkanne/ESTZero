#ifndef RHF_UTILS
#define RHF_UTILS

#include <Eigen/Dense>
#include <libint2.hpp>
#include <tensor.h>

Matrix sym_ortho(const Matrix& S, double tol = 1e-8);
Vector diis(const std::vector<Matrix>& err_list);

#endif