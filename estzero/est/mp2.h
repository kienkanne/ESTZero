#ifndef MP2
#define MP2

#include <Eigen/Dense>
#include "tensor.h"


double calc_rhf_MP2_E(const Tensor& Iiajb, const Vector& eps, int n_occ, double E0);
double calc_uhf_MP2_E(const Tensor& Iiajb_AA, const Tensor& Iiajb_BB, const Tensor& Iiajb_AB,
                    const Vector& epsa, const Vector& epsb, int n_alpha, int n_beta, double E0);

#endif
