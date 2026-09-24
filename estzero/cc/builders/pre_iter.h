#ifndef PRE_ITER
#define PRE_ITER

#include "tensor.h"

struct Dia_Dijab { Tensor Dia; Tensor Dijab; };

Dia_Dijab build_Dia_Dijab(const Tensor& FN, Sli O, Sli V);

Tensor build_spin_Iprqs(const Tensor& Ipqrs_AA, const Tensor& Ipqrs_BB, const Tensor& Ipqrs_AB,
                    const int n_alpha, const int n_beta);
                    
Tensor build_spin_FN(const Tensor& HA, const Tensor& HB, const Tensor& DEprqs,
                    const int n_alpha, const int n_beta);

Tensor build_diag_spin_FN(const Vector& epsA_active, const int n_alpha,
                          const Vector& epsB_active, const int n_beta);

Tensor build_diag_spat_FN(const Vector& eps_diag);

#endif