#include "tensor.h"
#include "cc_spin.h"
#include "einsum.h"


namespace {

// Scatters one SAME-spin chemist-notation (pq|rs) block into its four occupancy-major
// destination blocks in the MIXED-spin Iprqs (physicist notation). Converts chemist -> 
// physicist notation of the source tensor (<pq|rs> = (pr|qs), swap the middle two axes), 
// then walks every combination of {occ,virt} on each of the 4 axes, so 16 in total.
void scatter_same_spin(Tensor& Iprqs, const Tensor& I_chem,
                        Sli O_des, Sli V_des, Sli O_src, Sli V_src) {
    const Sli dest[2] = {O_des, V_des};
    const Sli src[2]  = {O_src, V_src};
    const Tensor I_phys = I_chem.swapaxes(1, 2); // (<pq|rs>)

    for (int c0 = 0; c0 < 2; ++c0) {
    for (int c1 = 0; c1 < 2; ++c1) {
    for (int c2 = 0; c2 < 2; ++c2) {
    for (int c3 = 0; c3 < 2; ++c3) {
        Iprqs.slice(dest[c0], dest[c1], dest[c2], dest[c3])
          .assign(I_phys.slice(src[c0], src[c1], src[c2], src[c3]));
    }}}}
}

// Same idea for the opposite-spin block. First convert to physicist notation. Axes 0,2 are 
// one spin (A), axes 1,3 are the (B). Also fills the "BA" block from the same source via 
// <p_B q_A|r_B s_A> = <q_A p_B|s_A r_B>, meaning swap axis pairs (0,1) and (2,3) together.
void scatter_mixed_spin(Tensor& IS, const Tensor& I_chem_AB,
                         Sli OA_des, Sli VA_des, Sli OA_src, Sli VA_src,
                         Sli OB_des, Sli VB_des, Sli OB_src, Sli VB_src) {
    const Sli destA[2] = {OA_des, VA_des}, srcA[2] = {OA_src, VA_src};
    const Sli destB[2] = {OB_des, VB_des}, srcB[2] = {OB_src, VB_src};

    const Tensor I_ab = I_chem_AB.swapaxes(1, 2); // <pA qB|rA sB>
    for (int c0 = 0; c0 < 2; ++c0) {
    for (int c1 = 0; c1 < 2; ++c1) {
    for (int c2 = 0; c2 < 2; ++c2) {
    for (int c3 = 0; c3 < 2; ++c3) {
        IS.slice(destA[c0], destB[c1], destA[c2], destB[c3])
          .assign(I_ab.slice(srcA[c0], srcB[c1], srcA[c2], srcB[c3]));
    }}}}

    const Tensor I_ba = I_ab.swapaxes(0, 1).swapaxes(2, 3); // <p_B q_A|r_B s_A>
    for (int c0 = 0; c0 < 2; ++c0) {
    for (int c1 = 0; c1 < 2; ++c1) {
    for (int c2 = 0; c2 < 2; ++c2) {
    for (int c3 = 0; c3 < 2; ++c3) {
        IS.slice(destB[c0], destA[c1], destB[c2], destA[c3])
          .assign(I_ba.slice(srcB[c0], srcA[c1], srcB[c2], srcA[c3]));
    }}}}
}

} // namespace

// The convention used here is occupancy-major: along each axis (nbf*2), the ordering is
// OA, OB, VA, VB. Because in previous UHF code, alpha and beta are separate (spin-major),
// the helpers below scatter them correctly into the occupancy-major format, which is a 
// requirment for CCSD as only occupied and virtual slices are needed without knowing about spin.
//
// This function takes in the spin blocks (AA, BB, AB) built in chemist-notation, convert them 
// into physicist notation, and scatter them correctly into the occupied/virtual - alpha/beta
// blocks. This is transforming a "spin-major" format to "occupancy-format", which requires the
// helpers above to avoid writing out all possible combinations. The final Tensor is then anti-symmetrized,
// which is used by all other intermediate-tensor-builders in CCSD.
Tensor build_spin_ASprqs(const Tensor& Ipqrs_AA, const Tensor& Ipqrs_BB, const Tensor& Ipqrs_AB,
                    const int n_alpha, const int n_beta) {

    const int nbf = Ipqrs_AA.shape[0];
    const auto [OA_des, OB_des, VA_des, VB_des,
                OA_src, OB_src, VA_src, VB_src] = get_slices_ab_ov(n_alpha, n_beta, nbf);
    
    Tensor Iprqs({2*nbf, 2*nbf, 2*nbf, 2*nbf});

    scatter_same_spin(Iprqs, Ipqrs_AA, OA_des, VA_des, OA_src, VA_src);
    scatter_same_spin(Iprqs, Ipqrs_BB, OB_des, VB_des, OB_src, VB_src);
    scatter_mixed_spin(Iprqs, Ipqrs_AB, OA_des, VA_des, OA_src, VA_src, OB_des, VB_des, OB_src, VB_src);

    // Antisymmetrize once here ASprqs = <pr||qs> = <pr|qs> - <pq|rs>
    return Iprqs - Iprqs.swapaxes(2, 3);
}


// Normal-ordered Fock matrix. FN = HN + sum_m <pi||qi>.
// Ha/Hb must each be the one-electron Hamiltonian already transformed into that spin's OWN MO basis 
// (Ha = Ca^T (T+V) Ca, Hb = Cb^T (T+V) Cb)
Tensor build_spin_FN(const Tensor& Ha, const Tensor& Hb, const Tensor& ASprqs, const int n_alpha, const int n_beta) {
    const int nbf = Ha.shape[0];
    Tensor HN({2 * nbf, 2 * nbf});

    const auto [OA_des, OB_des, VA_des, VB_des,
                OA_src, OB_src, VA_src, VB_src] = get_slices_ab_ov(n_alpha, n_beta, nbf);

    // Ha and Hb are stored in spin-major order. Reorder each spin block into shared
    // occupancy-major order Oa, Ob, Va, Vb. Cross-spin blocks stay zero. This is similar
    // idea above, but a matrix is smaller so rolling all out is feasible
    HN.slice(OA_des, OA_des).assign(Ha.slice(OA_src, OA_src));
    HN.slice(OA_des, VA_des).assign(Ha.slice(OA_src, VA_src));
    HN.slice(VA_des, OA_des).assign(Ha.slice(VA_src, OA_src));
    HN.slice(VA_des, VA_des).assign(Ha.slice(VA_src, VA_src));

    HN.slice(OB_des, OB_des).assign(Hb.slice(OB_src, OB_src));
    HN.slice(OB_des, VB_des).assign(Hb.slice(OB_src, VB_src));
    HN.slice(VB_des, OB_des).assign(Hb.slice(VB_src, OB_src));
    HN.slice(VB_des, VB_des).assign(Hb.slice(VB_src, VB_src));

    const Sli All{};
    const Sli O{0, n_alpha + n_beta};
    Tensor two_electron = einsum("piqi->pq", {ASprqs.slice(All, O, All, O)});

    return HN + two_electron;
}


Dia_Dijab build_spin_Dia_Dijab(const Tensor& FN, const int n_alpha, const int n_beta) {
    const int nbf = static_cast<int>(FN.shape[0]/2);
    const int n_occ = n_alpha + n_beta;
    const int n_virt = 2*nbf - n_occ;
        const auto [OA_des, OB_des, VA_des, VB_des,
                    OA_src, OB_src, VA_src, VB_src] = get_slices_ab_ov(n_alpha, n_beta, nbf);
    
    Tensor FN_diag({2 * nbf});
    for (int i=0; i != nbf*2; ++i) { FN_diag(i) = FN(i, i); }

    Tensor Dia({n_occ, n_virt});
    Tensor Dijab({n_occ, n_occ, n_virt, n_virt});

    for (int i = 0; i < n_occ; ++i) {
    for (int a = 0; a < n_virt; ++a) {
        Dia(i, a) = FN_diag(i) - FN_diag(n_occ + a);

        for (int j = 0; j < n_occ; ++j) {
        for (int b = 0; b < n_virt; ++b) {
            Dijab(i, j, a, b) = FN_diag(i) + FN_diag(j) - FN_diag(n_occ + a) - FN_diag(n_occ + b);
    }}}}

    return {Dia, Dijab};
}

double test_mp2_energy(const Tensor& ASprqs, const Tensor& Dia, const Tensor& Dijab, const int n_occ) {
    const Sli O{0, n_occ};
    const Sli V{n_occ, ASprqs.shape[0]};
    const Tensor Ioovv = ASprqs.slice(O, O, V, V);
    const Tensor t2 = Ioovv / Dijab;

    return 0.25 * (t2 * Ioovv).sum();
}

double calc_ccsd_E(const Tensor& ASpqrs, const Tensor& FN,
                    const Tensor& t1, const Tensor& t2, Sli O, Sli V) {
    double E = einsum("ia,ia->", {FN.slice(O, V), t1})();
    E += 0.25 * einsum("ijab,ijab->", {ASpqrs.slice(O, O, V, V), t2})();
    E += 0.5 * einsum("ijab,ia,jb->", {ASpqrs.slice(O, O, V, V), t1, t1})();
    return E;
}
