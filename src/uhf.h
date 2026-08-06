#ifndef UHF
#define UHF

#include <vector>
#include <Eigen/Dense>
#include <utils.h>

using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;


Result uhf_scf(double econv, int max_iter, int verbose, double V_nn, int n_alpha, int n_beta, 
                const Matrix& S, const Matrix& T,
                const Matrix& V, const std::vector<double>& eri_vector);

#endif