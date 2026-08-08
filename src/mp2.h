#ifndef MP2
#define MP2

#include <Eigen/Dense>
#include "einsum.h"

using Vector = Eigen::VectorXd;

double calc_rhf_MP2_E(const Tensor& I_mo, const Vector& eps, int n_occ, double E0);
double calc_uhf_MP2_E(const Tensor& Iaa_mo, const Tensor& Ibb_mo, const Tensor& Iab_mo,
                    const Vector& epsa, const Vector& epsb, int n_alpha, int n_beta, double E0);

#endif
