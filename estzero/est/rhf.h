#ifndef RHF
#define RHF

#include <Eigen/Dense>
#include <utils.h>
#include <einsum.h>

using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;


template <typename ERI>
Result rhf_scf(double econv, int max_iter, int verbose, double V_nn, int ndocc,
                const Matrix& S, const Matrix& T,
                const Matrix& V, const ERI& eri);

#endif
