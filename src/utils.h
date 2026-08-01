#ifndef UTILS
#define UTILS

#include <Eigen/Dense>
#include <libint2.hpp>
#include <vector>

double compute_nuclear_repulsion(const std::vector<libint2::Atom>& atoms);
Eigen::MatrixXd sym_ortho(const Eigen::MatrixXd& S, double tol = 1e-8);

int pair_index(int i, int j);
int eri_index(int i, int j, int k, int l);

#endif