#ifndef TRANSMO
#define TRANSMO

#include <vector>
#include <Eigen/Dense>
#include "tensor.h"


struct SpinMO {
    Tensor Iaa_mo; Tensor Ibb_mo; Tensor Iab_mo;
};

Tensor  transform_rhf_MO(const std::vector<double>& eri_vector, const Matrix& C, int n_occ);
Tensor  transform_rhf_MO(const Tensor& eri_tensor, const Matrix& C, int n_occ);
SpinMO  transform_uhf_MO(const std::vector<double>& eri_vector,
                          const Matrix& Ca, int n_alpha, const Matrix& Cb, int n_beta);
SpinMO  transform_uhf_MO(const Tensor& eri_tensor,
                          const Matrix& Ca, int n_alpha, const Matrix& Cb, int n_beta);

#endif
