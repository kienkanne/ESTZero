#ifndef RHF
#define RHF

#include <vector>
#include <Eigen/Dense>
#include <utils.h>

using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;

struct Result {
    double E0;
    Matrix C;
    Vector eps;
};

Result get_rhf_energy(double tol, int max_iter, int verbose, double V_nn, int ndocc, 
                    const Matrix& S, const Matrix& T,
                    const Matrix& V, const std::vector<double>& eri_vector);

#endif