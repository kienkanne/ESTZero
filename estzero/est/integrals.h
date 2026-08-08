#ifndef INTEGRALS
#define INTEGRALS

#include <vector>
#include <Eigen/Dense>
#include <libint2.hpp>

inline int pair_index(int i, int j) { return i >= j ? i*(i+1)/2 + j : j*(j+1)/2 + i; }
inline int eri_index(int i, int j, int k, int l) { return pair_index(pair_index(i,j), pair_index(k,l)); }

Eigen::MatrixXd build_one_center(libint2::Engine& engine, const libint2::BasisSet& obs);
std::vector<double> build_canonical_eris(libint2::Engine& engine, const libint2::BasisSet& obs);

#endif