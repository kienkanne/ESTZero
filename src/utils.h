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

double compute_nuclear_repulsion(const std::vector<libint2::Atom>& atoms);
Matrix sym_ortho(const Matrix& S, double tol = 1e-8);
Vector diis(const std::vector<Matrix>& err_list);

inline int pair_index(int i, int j) { return i >= j ? i*(i+1)/2 + j : j*(j+1)/2 + i; }
inline int eri_index(int i, int j, int k, int l) { return pair_index(pair_index(i,j), pair_index(k,l)); }

#endif