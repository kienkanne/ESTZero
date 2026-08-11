#ifndef UHF
#define UHF

#include <Eigen/Dense>
#include <utils.h>
#include <tensor.h>

template <typename ERI>
Result uhf_scf(double econv, int max_iter, int verbose, double V_nn, int n_alpha, int n_beta,
                const Matrix& S, const Matrix& T,
                const Matrix& V, const ERI& eri);

#endif
