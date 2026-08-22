#include "tensor.h"
#include "einsum.h"
#include "cc_spin.h"

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
Tensor build_spin_Iprqs(const Tensor& Ipqrs_AA, const Tensor& Ipqrs_BB, const Tensor& Ipqrs_AB,
                    const int n_alpha, const int n_beta) {

    const int nbf = Ipqrs_AA.shape[0];
    const auto [OA_des, OB_des, VA_des, VB_des,
                OA_src, OB_src, VA_src, VB_src] = get_slices_ab_ov(n_alpha, n_beta, nbf);
    
    Tensor Iprqs({2*nbf, 2*nbf, 2*nbf, 2*nbf});

    scatter_same_spin(Iprqs, Ipqrs_AA, OA_des, VA_des, OA_src, VA_src);
    scatter_same_spin(Iprqs, Ipqrs_BB, OB_des, VB_des, OB_src, VB_src);
    scatter_mixed_spin(Iprqs, Ipqrs_AB, OA_des, VA_des, OA_src, VA_src, OB_des, VB_des, OB_src, VB_src);

    return Iprqs;
}


// Normal-ordered Fock matrix. FN = HN + sum_m <pi||qi>.
// Ha/Hb must each be the one-electron Hamiltonian already transformed into that spin's OWN MO basis 
// (Ha = Ca^T (T+V) Ca, Hb = Cb^T (T+V) Cb)
Tensor build_spin_FN(const Tensor& HA, const Tensor& HB, const Tensor& DEprqs, 
                    const int n_alpha, const int n_beta) {
    const int nbf = HA.shape[0];
    Tensor HN({2 * nbf, 2 * nbf});

    const auto [OA_des, OB_des, VA_des, VB_des,
                OA_src, OB_src, VA_src, VB_src] = get_slices_ab_ov(n_alpha, n_beta, nbf);

    // Ha and Hb are stored in spin-major order. Reorder each spin block into shared
    // occupancy-major order Oa, Ob, Va, Vb. Cross-spin blocks stay zero. This is similar
    // idea above, but a matrix is smaller so rolling all out is feasible
    HN.slice(OA_des, OA_des).assign(HA.slice(OA_src, OA_src));
    HN.slice(OA_des, VA_des).assign(HA.slice(OA_src, VA_src));
    HN.slice(VA_des, OA_des).assign(HA.slice(VA_src, OA_src));
    HN.slice(VA_des, VA_des).assign(HA.slice(VA_src, VA_src));

    HN.slice(OB_des, OB_des).assign(HB.slice(OB_src, OB_src));
    HN.slice(OB_des, VB_des).assign(HB.slice(OB_src, VB_src));
    HN.slice(VB_des, OB_des).assign(HB.slice(VB_src, OB_src));
    HN.slice(VB_des, VB_des).assign(HB.slice(VB_src, VB_src));

    const Sli All{};
    const Sli O{0, n_alpha + n_beta};
    Tensor two_electron = einsum("piqi->pq", {DEprqs.slice(All, O, All, O)});

    return HN + two_electron;
}


// Stanton Eqn 1
Tensor build_spin_T1(const Tensor& DEprqs, const Tensor& FN,
                const Tensor& Fae, const Tensor& Fmi, const Tensor& Fme,
                const Tensor& t1, const Tensor& t2, Sli O, Sli V, bool spin) {
    Tensor t1_new = FN.slice(O, V).copy();

    t1_new += einsum("ie,ae->ia", {t1, Fae});
    t1_new -= einsum("ma,mi->ia", {t1, Fmi});
    t1_new += einsum("imae,me->ia", {t2, Fme});
    t1_new -= einsum("nf,naif->ia", {t1, DEprqs.slice(O, V, O, V)});
    t1_new -= 0.5 * einsum("imef,maef->ia", {t2, DEprqs.slice(O, V, V, V)});
    t1_new -= 0.5 * einsum("mnae,nmei->ia", {t2, DEprqs.slice(O, O, V, O)});

    return t1_new;
}

// Stanton Eqn 2
Tensor build_spin_T2(const Tensor& DEprqs, const Tensor& tau, // (takes tau, not FN)
                const Tensor& Fae, const Tensor& Fmi, const Tensor& Fme,
                const Tensor& Wmnij, const Tensor& Wabef, const Tensor& Wmbej,
                const Tensor& t1, const Tensor& t2, Sli O, Sli V, bool spin) {
    Tensor t2_new = DEprqs.slice(O, O, V, V).copy();
    
    // P(ij) = (Pij - Pji)
    Tensor tmp = Fae - 0.5 * einsum("mb,me->be", {t1, Fme});
    Tensor Pab = einsum("ijae,be->ijab", {t2, tmp});
    t2_new += Pab;
    t2_new -= Pab.swapaxes(2, 3);

    // Most costly step here
    tmp = Fmi + 0.5 * einsum("je,me->mj", {t1, Fme});
    Tensor Pij = einsum("imab,mj->ijab", {t2, tmp});
    t2_new -= Pij;
    t2_new += Pij.swapaxes(0, 1);
    t2_new += 0.5 * einsum("mnab,mnij->ijab", {tau, Wmnij});
    t2_new += 0.5 * einsum("ijef,abef->ijab", {tau, Wabef});

    // And here
    Tensor Pijab = einsum("imae,mbej->ijab", {t2, Wmbej});

    tmp = einsum("ie,mbej,ma->ijab", {t1, DEprqs.slice(O, V, V, O), t1,});
    Pijab -= tmp;

    // P(ij) * P(ab): (ij - ji) * (ab - ba) = ijab - ijba - jiab + jiba
    t2_new += Pijab;
    t2_new -= Pijab.swapaxes(2, 3);
    t2_new -= Pijab.swapaxes(0, 1);
    t2_new += Pijab.swapaxes(0, 1).swapaxes(2, 3);
    
    Pij = einsum("ie,abej->ijab", {t1, DEprqs.slice(V, V, V, O)});
    t2_new += Pij;
    t2_new -= Pij.swapaxes(0, 1);
    
    Pab = einsum("ma,mbij->ijab", {t1, DEprqs.slice(O, V, O, O)});
    t2_new -= Pab;
    t2_new += Pab.swapaxes(2, 3);
    
    return t2_new;
}

double test_mp2_energy(const Tensor& DEprqs, const Tensor& Dia, const Tensor& Dijab, const int n_occ) {
    const Sli O{0, n_occ};
    const Sli V{n_occ, DEprqs.shape[0]};
    const Tensor Ioovv = DEprqs.slice(O, O, V, V);
    const Tensor t2 = Ioovv / Dijab;

    return 0.25 * (t2 * Ioovv).sum();
}

double calc_spin_cc_E(const Tensor& DEprqs, const Tensor& FN, const Tensor& tau,
                const Tensor& t1, const Tensor& t2, Sli O, Sli V) {
    const Tensor I_oovv = DEprqs.slice(O, O, V, V);
    double E_cc = 0.25 * einsum("ijab,ijab->", {I_oovv, t2})();
    
    E_cc += einsum("ia,ia->", {FN.slice(O, V), t1})();
    E_cc += 0.5 * einsum("ijab,ia,jb->", {I_oovv, t1, t1})();

    return E_cc;
}
