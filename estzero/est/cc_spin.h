#ifndef SPIN
#define SPIN

#include "tensor.h"

struct Slices_AB_OV {
    // Initial, final
    Sli OA_des; Sli OB_des; Sli VA_des; Sli VB_des;
    Sli OA_src; Sli OB_src; Sli VA_src; Sli VB_src;
};

// Provide slicing for AB-OV combinations in occupancy-major form
inline Slices_AB_OV get_slices_ab_ov(const int n_alpha, const int n_beta, const int nbf) {
    const int n_occ = n_alpha + n_beta;

    const Sli OA_des = Sli {0, n_alpha};
    const Sli OB_des = Sli {n_alpha, n_occ};
    const Sli VA_des = Sli {n_occ, n_occ+(nbf-n_alpha)};
    const Sli VB_des = Sli {n_occ+(nbf-n_alpha), nbf*2};

    const Sli OA_src{0, n_alpha}; 
    const Sli VA_src{n_alpha, nbf};
    const Sli OB_src{0, n_beta};
    const Sli VB_src{n_beta, nbf};

    return {OA_des, OB_des, VA_des, VB_des, 
            OA_src, VA_src, OB_src, VB_src};
}

struct Dia_Dijab {
    Tensor Dia; Tensor Dijab;
};

Tensor build_spin_ASprqs(const Tensor& Ipqrs_AA, const Tensor& Ipqrs_BB, const Tensor& Ipqrs_AB,
                    const int n_alpha, const int n_beta, const int nbf);
                    
Tensor build_spin_FN(const Tensor& Ha, const Tensor& Hb, const Tensor& ASprqs, const int n_alpha, const int n_beta);
Dia_Dijab build_spin_Dia_Dijab(const Tensor& FN, const int n_alpha, const int n_beta);

double test_mp2_energy(const Tensor& ASprqs, const Tensor& Dia, const Tensor& Dijab, const int n_occ);

#endif
