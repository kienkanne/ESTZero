#ifndef TRANSMO
#define TRANSMO

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;
using Tensor4D = Eigen::Tensor<double, 4>;

struct SpinMO {
    Tensor4D Iaa_mo; Tensor4D Ibb_mo; Tensor4D Iab_mo;
};

Tensor4D transform_rhf_MO(const std::vector<double>& eri_vector, const Matrix& C, int n_occ);
SpinMO transform_uhf_MO(const std::vector<double>& eri_vector, 
                          const Matrix& Ca, int n_alpha, const Matrix& Cb, int n_beta);

#endif
