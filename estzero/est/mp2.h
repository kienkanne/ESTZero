#ifndef MP2
#define MP2

#include "tensor.h"

double calc_rhf_mp2_E(const Tensor& Iiajb, const Vector& eps, int n_occ);
double calc_uhf_mp2_E(const Tensor& Iiajb_AA, const Tensor& Iiajb_BB, const Tensor& Iiajb_AB,
                    const Vector& epsa, const Vector& epsb, int n_alpha, int n_beta);

#endif
