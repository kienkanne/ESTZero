#ifndef RHF
#define RHF

#include <vector>
#include <Eigen/Dense>
#include <utils.h>

double get_rhf_energy(double tol, int max_iter, int verbose, double V_nn, int ndocc, 
                    const Eigen::MatrixXd& S, const Eigen::MatrixXd& T,
                    const Eigen::MatrixXd& V, const std::vector<double>& ERI);

#endif