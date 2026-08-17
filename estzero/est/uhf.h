#ifndef UHF
#define UHF

#include <tensor.h>
#include "wfn.h"

template <typename ERI>
SCFResult uhf_scf(double econv, int max_iter, int verbose, double V_nn, int n_alpha, int n_beta,
                const Matrix& S, const Matrix& T,
                const Matrix& V, const ERI& eri);

#endif
