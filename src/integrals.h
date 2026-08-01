#ifndef INTEGRALS
#define INTEGRALS

#include <vector>
#include <Eigen/Dense>
#include <libint2.hpp>

Eigen::MatrixXd build_one_center(libint2::Engine& engine, const libint2::BasisSet& obs);
std::vector<double> build_canonical_eris(libint2::Engine& engine, const libint2::BasisSet& obs);

#endif