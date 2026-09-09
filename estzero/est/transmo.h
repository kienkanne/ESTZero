#ifndef TRANSMO
#define TRANSMO

#include <Eigen/Dense>
#include "tensor.h"

struct SpinMO {
    Tensor I_AA; Tensor I_BB; Tensor I_AB;
};

Tensor  transmo_spat_iajb(const Tensor& eri_tensor, const Matrix& C, int n_occ, int n_frozen = 0);
SpinMO  transmo_spin_iajb(const Tensor& eri_tensor, const Matrix& CA, int n_alpha, const Matrix& CB, int n_beta);

// Full-space (p,q,r,s all spanning the whole MO range) versions, needed for
// CCSD, which indexes MO[o,o,o,o]/[v,v,v,v]/etc. -- not just the [o,v,o,v]
// slice MP2 needs.
Tensor  transmo_spat_pqrs(const Tensor& eri_tensor, const Matrix& C);
SpinMO  transmo_spin_pqrs(const Tensor& eri_tensor, const Matrix& CA, int n_alpha, const Matrix& CB, int n_beta);

#endif
