#ifndef RHF
#define RHF

#include <vector>
#include <Eigen/Dense>
#include <utils.h>

double roothan_solve(const Eigen::MatrixXd& S, const Eigen::MatrixXd& T,
                     const Eigen::MatrixXd& V, const std::vector<double>& ERI,
                     double V_nn, int ndocc, int max_iter = 100,
                     double tol = 1e-6);

#endif