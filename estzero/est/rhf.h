#ifndef RHF
#define RHF

#include <tensor.h>
#include "wfn.h"

template <typename ERI>
SCFResult rhf_scf(double econv, int max_iter, int verbose, double V_nn, int ndocc,
                const Matrix& S, const Matrix& T,
                const Matrix& V, const ERI& eri);

#endif
