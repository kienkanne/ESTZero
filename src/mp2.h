#ifndef MP2
#define MP2

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

using Vector = Eigen::VectorXd;
using Tensor4D = Eigen::Tensor<double, 4>;

double calc_rhf_MP2_E(const Tensor4D& I_mo, const Vector& eps, int n_occ, double E0);
double calc_uhf_MP2_E(const Tensor4D& Iaa_mo, const Tensor4D& Ibb_mo, const Tensor4D& Iab_mo, 
                    const Vector& epsa, const Vector& epsb, int n_alpha, int n_beta, double E0);
                    
#endif
