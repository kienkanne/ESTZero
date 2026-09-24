#ifndef INTEGRALS
#define INTEGRALS

#include <vector>
#include <libint2.hpp>
#include "tensor.h"

inline int pair_index(int i, int j) { return i >= j ? i*(i+1)/2 + j : j*(j+1)/2 + i; }
inline int eri_index(int i, int j, int k, int l) { return pair_index(pair_index(i,j), pair_index(k,l)); }

// Which in-core ERI representation to build/use. Both are first-class and kept
// side by side (vector = compact canonical-index storage, tensor = dense
// nbf^4, DGEMM-ready) for benchmarking wall time and cache behavior.
enum class ERIScheme { Vector, Tensor };

Matrix build_one_center(libint2::Engine& engine, const libint2::BasisSet& obs);

// Hardcoded tolerance, matching Psi4's
inline constexpr double SCHWARZ_TOL = 1e-12;

// Shell-pair upper bound Q(s1,s2) = sqrt(max value of |(f1 f2|f1 f2)|, f1 & f2 looping through basis func).
// Used by build_eri_tensor/build_eri_vector to skip negligible shell quartets via the Cauchy-Schwarz inequality 
// |(s1 s2|s3 s4)| <= Q(s1,s2)*Q(s3,s4). 
Matrix compute_schwarz_bounds(libint2::Engine& engine, const libint2::BasisSet& obs);

Tensor build_eri_tensor(libint2::Engine& engine, const libint2::BasisSet& obs, int verbose = 0);
std::vector<double> build_eri_vector(libint2::Engine& engine, const libint2::BasisSet& obs, int verbose = 0);
Tensor eri_vector_to_tensor(const std::vector<double>& eri_vector, int nbf);

#endif