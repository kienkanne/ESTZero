#ifndef RHF_UTILS
#define RHF_UTILS

#include <Eigen/Dense>
#include <libint2.hpp>
#include <vector>

double compute_nuclear_repulsion(const std::vector<libint2::Atom>& atoms);
Eigen::MatrixXd sym_ortho(const Eigen::MatrixXd& S, double tol = 1e-8);
Eigen::VectorXd diis(const std::vector<Eigen::MatrixXd>& err_list);

inline int pair_index(int i, int j) { return i >= j ? i*(i+1)/2 + j : j*(j+1)/2 + i; }
inline int eri_index(int i, int j, int k, int l) { return pair_index(pair_index(i,j), pair_index(k,l)); }

#endif