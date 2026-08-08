#ifndef RHF_UTILS
#define RHF_UTILS

#include <Eigen/Dense>
#include <libint2.hpp>
#include <vector>

using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;

struct Result {
    double E0;
    Matrix Ca;
    Matrix Cb;
    Vector epsa;
    Vector epsb;
};

Matrix sym_ortho(const Matrix& S, double tol = 1e-8);
Vector diis(const std::vector<Matrix>& err_list);


#endif