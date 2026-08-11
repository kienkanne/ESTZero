#ifndef RHF
#define RHF

#include <Eigen/Dense>
#include <utils.h>
#include <tensor.h>


template <typename ERI>
Result rhf_scf(double econv, int max_iter, int verbose, double V_nn, int ndocc,
                const Matrix& S, const Matrix& T,
                const Matrix& V, const ERI& eri);

#endif
