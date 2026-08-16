#ifndef SPIN
#define SPIN

#include "tensor.h"

struct Slices_AB_OV {
    // Initial, final
    Sli OA;
    Sli OB;
    Sli VA;
    Sli VB;
};

inline Slices_AB_OV get_slices_ab_ov(const int n_alpha, const int n_beta, const int nbf) {
    const int n_occ = n_alpha + n_beta;

    Sli OA = Sli {0, n_alpha};
    Sli OB = Sli {n_alpha, n_occ};
    Sli VA = Sli {n_occ, n_occ+(nbf-n_alpha)};
    Sli VB = Sli {n_occ+(nbf-n_alpha), nbf*2};

    return {OA, OB, VA, VB};
}

struct Dia_Diajb {
    Tensor Dia; Tensor Diajb;
};

Tensor build_spin_MO(const Tensor& Iaa, const Tensor& Ibb, const Tensor& Iab, const int n_alpha, const int n_beta, const int nbf);
Dia_Diajb build_Dia_Diajb(const Vector& epsa, const Vector& epsb, const int n_alpha, const int n_beta);
double test_mp2_energy(const Tensor& Ismo, const Tensor& Dia, const Tensor& Diajb, const int n_occ);

#endif