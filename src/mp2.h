#ifndef MP2
#define MP2

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;
using Tensor4D = Eigen::Tensor<double, 4>;

Tensor4D transform_MO(const std::vector<double>& eri_vector, const Matrix& C, int n_occ);
double calc_MP2_E(const Tensor4D& I_mo, int n_occ, const Vector& eps, double E0);

#endif
