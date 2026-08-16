
#include "tensor.h"
#include "cc_uhf.h"

Tensor build_spin_MO(const Tensor& Iaa, const Tensor& Ibb, const Tensor& Iab, 
                    const int n_alpha, const int n_beta, const int nbf) {
    const auto [OA, OB, VA, VB] = get_slices_ab_ov(n_alpha, n_beta, nbf);
    
    Tensor Ismo({2*nbf, 2*nbf, 2*nbf, 2*nbf});
    // Fill the occupied-virtual blocks used by the MP2 contraction.
    Ismo.slice(OA, VA, OA, VA).assign(Iaa);
    Ismo.slice(OB, VB, OB, VB).assign(Ibb);

    // Iab (or Iaabb) is not a symmetric block, but it represent Ibbaa also, so
    // swap axes to fill bbaa (final) takes from aabb (initial)
    Ismo.slice(OA, VA, OB, VB).assign(Iab);
    Ismo.slice(OB, VB, OA, VA).assign(Iab.swapaxes(0, 2).swapaxes(1, 3));

    return Ismo;
}


Dia_Diajb build_Dia_Diajb(const Vector& epsa, const Vector& epsb, const int n_alpha, const int n_beta) {
    const int nbf = epsa.size();
    const int n_occ = n_alpha + n_beta;
    const auto [OA, OB, VA, VB] = get_slices_ab_ov(n_alpha, n_beta, nbf);
    Tensor epsa_t = vec_to_tensor(epsa);
    Tensor epsb_t = vec_to_tensor(epsb);

    // Concatenate in O/V form:
    Tensor eps({nbf*2});
    eps.slice(OA).assign(epsa_t.slice(Sli{0, n_alpha}));
    eps.slice(OB).assign(epsb_t.slice(Sli{0, n_beta}));
    eps.slice(VA).assign(epsa_t.slice(Sli{n_alpha, nbf}));
    eps.slice(VB).assign(epsb_t.slice(Sli{n_beta, nbf}));

    const int n_virt = nbf*2 - n_occ;
    Tensor Dia({n_occ, n_virt});
    Tensor Diajb({n_occ, n_virt, n_occ, n_virt});

    for (int i = 0; i < n_occ; ++i) {
    for (int a = 0; a < n_virt; ++a) {
        Dia(i,a) = eps(i) - eps(n_occ + a);

        for (int j = 0; j < n_occ; ++j) {
        for (int b = 0; b < n_virt; ++b) {
            Diajb(i, a, j, b) = eps(i) - eps(n_occ + a) + eps(j) - eps(n_occ + b);
    }}}}

    return {Dia, Diajb};
}

double test_mp2_energy(const Tensor& Ismo, const Tensor& Dia, const Tensor& Diajb, const int n_occ) {
    const Sli O{0, n_occ};
    const Sli V{n_occ, Ismo.shape[0]};
    const Tensor Iovov = Ismo.slice(O, V, O, V);
    const Tensor Iantisym = Iovov - Iovov.swapaxes(1, 3);
    const Tensor t2 = Iantisym / Diajb;

    return 0.25 * (t2 * Iantisym).sum();
}