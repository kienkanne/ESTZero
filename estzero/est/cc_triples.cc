#include <vector>
#include <cblas.h>
#include "tensor.h"
#include "einsum.h"
#include "timer.h"


// Vikings' connected-T3 batch for fixed occupied (i,j,k), spin-orbital/UHF form, called once per
// (i,j,k) by calc_pert_triples below.
// This is a direct line-by-line port of `t3c_ijk_so` in `cctriples.py` in PyCC.
// Source: https://github.com/CrawfordGroup/pycc/blob/main/pycc/cctriples.py
//
// Legacy version wired through einsum. The route routed into live code is below where GEMM is called
// directly. The repeated einsum calls in PertT makes the overhead cost the dominate.
Tensor t3c_ijk_spin_old(const Tensor& t2, const Tensor& Wvvvo, const Tensor& Wovoo,
               const Tensor& F_occ, const Tensor& F_virt, int i, int j, int k) {
    Tensor abc = einsum("ad,bcd->abc", {t2.select(0, i).select(0, j), Wvvvo.select(3, k)});
    abc -= einsum("ad,bcd->abc", {t2.select(0, k).select(0, j), Wvvvo.select(3, i)});
    abc -= einsum("ad,bcd->abc", {t2.select(0, i).select(0, k), Wvvvo.select(3, j)});
    Tensor t3 = abc - abc.swapaxes(0, 1) - abc.swapaxes(0, 2);

    abc = einsum("lab,lc->abc", {t2.select(0, i), Wovoo.select(2, j).select(2, k)});
    abc -= einsum("lab,lc->abc", {t2.select(0, j), Wovoo.select(2, i).select(2, k)});
    abc -= einsum("lab,lc->abc", {t2.select(0, k), Wovoo.select(2, j).select(2, i)});
    t3 -= (abc - abc.swapaxes(0, 2) - abc.swapaxes(1, 2));

    const int n_virt = t3.shape[0];
    Tensor denom({n_virt, n_virt, n_virt});
    const double e_ijk = F_occ(i) + F_occ(j) + F_occ(k);
    for (int a = 0; a < n_virt; ++a) {
    for (int b = 0; b < n_virt; ++b) {
    for (int c = 0; c < n_virt; ++c) {
        denom(a, b, c) = e_ijk - (F_virt(a) + F_virt(b) + F_virt(c));
    }}}
    return t3 / denom;
}


// Directly call GEMM for contractions, bypassing einsum.
Tensor t3c_ijk_spin(const Tensor& t2, const Tensor& Wvvvo, const Tensor& Wovoo,
               const Tensor& F_occ, const Tensor& F_virt, int i, int j, int k) {
    const int v = Wvvvo.shape[0];
    const int o = Wovoo.shape[0];
    const int v2 = v * v;
    const size_t v3 = static_cast<size_t>(v2) * v;

    // t2[idx,idx,:,:] (v,v) is already contiguous because the it's the first index being sliced (largest stride), so no 
    // packing is needed (just pass in the data_ptr that is already offsetted). Wvvvo[:,:,:,idx] is a strided  slice (last 
    // index) and it non-contiguous, so use .copy() to pack it for GEMM.
    //
    // The GEMM notation is t2(a,d) * W(bc,d).T = (abc)
    // The sizes and leading dimension are passed into cblas, so the shape can be recognized (v^2 is bc). The transpose then
    // guarentess the output shape is abc, no permutation needed.
    std::vector<double> abc(v3, 0.0);
    auto add_vvvo_term = [&](double sign, const Tensor& t2_slice, int w_occ_idx, bool first) {
        Tensor Wpack = Wvvvo.select(3, w_occ_idx).copy();
        const double* A = t2_slice.data_ptr();
        const double* B = Wpack.data_ptr();
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    v, v2, v,
                    sign, A, v, B, v,
                    first ? 0.0 : 1.0, abc.data(), v2);
    };
    add_vvvo_term(1.0,  t2.select(0, i).select(0, j), k, true);  // "ad,bcd->abc" (a,b,c)
    add_vvvo_term(-1.0, t2.select(0, k).select(0, j), i, false); // "ad,bcd->abc" (a,b,c)
    add_vvvo_term(-1.0, t2.select(0, i).select(0, k), j, false); // "ad,bcd->abc" (a,b,c)

    // Very much similar idea here, but t2 is sliced once and W is sliced twice instead. The transpose trick is the
    // same to get the desired output shape (abc)
    std::vector<double> abc2(v3, 0.0);
    auto add_ovoo_term = [&](double sign, const Tensor& t2_slice, int w_idx1, int w_idx2, bool first) {
        Tensor Wpack = Wovoo.select(2, w_idx1).select(2, w_idx2).copy();
        const double* A = t2_slice.data_ptr();
        const double* B = Wpack.data_ptr();
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    v2, v, o,
                    sign, A, v2, B, v,
                    first ? 0.0 : 1.0, abc2.data(), v);
    };
    add_ovoo_term(1.0,  t2.select(0, i), j, k, true);  // "lab,lc->abc" (a,b,c)
    add_ovoo_term(-1.0, t2.select(0, j), i, k, false); // "lab,lc->abc" (a,b,c)
    add_ovoo_term(-1.0, t2.select(0, k), j, i, false); // "lab,lc->abc" (a,b,c)

    Tensor abc_t({v, v, v}, abc);
    Tensor abc2_t({v, v, v}, abc2);

    Tensor t3 = abc_t.copy();                   // + abc[a,b,c]
    t3 -= abc_t.swapaxes(0, 1).copy(); // - abc[b,a,c]
    t3 -= abc_t.swapaxes(0, 2).copy(); // - abc[c,b,a]
    t3 -= abc2_t;                               // - abc2[a,b,c]
    t3 += abc2_t.swapaxes(0, 2).copy();// + abc2[c,b,a]
    t3 += abc2_t.swapaxes(1, 2).copy();// + abc2[a,c,b]

    Tensor denom({v, v, v});
    const double e_ijk = F_occ(i) + F_occ(j) + F_occ(k);
    for (int a = 0; a < v; ++a) {
    for (int b = 0; b < v; ++b) {
    for (int c = 0; c < v; ++c) {
        denom(a, b, c) = e_ijk - (F_virt(a) + F_virt(b) + F_virt(c));
    }}}

    return t3 / denom;
}


// Vikings' connected-T3 batch for fixed occupied (i,j,k), spin-adapted/RHF form, where the six permutations 
// are written out explicitly. Wvvvo/Wovoo are raw Iprqs slices, unlike the UHF version which uses DEprqs
// This is a direct line-by-line port of `t3c_ijk` in `cctriples.py` in PyCC.
// Source: https://github.com/CrawfordGroup/pycc/blob/main/pycc/cctriples.py
Tensor t3c_ijk_spat_old(const Tensor& t2, const Tensor& Wvvvo, const Tensor& Wovoo,
               const Tensor& F_occ, const Tensor& F_virt, int i, int j, int k) {
    Tensor t3 = einsum("bae,ce->abc", {Wvvvo.select(3, i), t2.select(0, k).select(0, j)});
    t3 += einsum("cae,be->abc", {Wvvvo.select(3, i), t2.select(0, j).select(0, k)});
    t3 += einsum("ace,be->abc", {Wvvvo.select(3, k), t2.select(0, j).select(0, i)});
    t3 += einsum("bce,ae->abc", {Wvvvo.select(3, k), t2.select(0, i).select(0, j)});
    t3 += einsum("cbe,ae->abc", {Wvvvo.select(3, j), t2.select(0, i).select(0, k)});
    t3 += einsum("abe,ce->abc", {Wvvvo.select(3, j), t2.select(0, k).select(0, i)});

    t3 -= einsum("mc,mab->abc", {Wovoo.select(2, j).select(2, k), t2.select(0, i)});
    t3 -= einsum("mb,mac->abc", {Wovoo.select(2, k).select(2, j), t2.select(0, i)});
    t3 -= einsum("mb,mca->abc", {Wovoo.select(2, i).select(2, j), t2.select(0, k)});
    t3 -= einsum("ma,mcb->abc", {Wovoo.select(2, j).select(2, i), t2.select(0, k)});
    t3 -= einsum("ma,mbc->abc", {Wovoo.select(2, k).select(2, i), t2.select(0, j)});
    t3 -= einsum("mc,mba->abc", {Wovoo.select(2, i).select(2, k), t2.select(0, j)});

    const int n_virt = t3.shape[0];
    Tensor denom({n_virt, n_virt, n_virt});
    const double e_ijk = F_occ(i) + F_occ(j) + F_occ(k);
    for (int a = 0; a < n_virt; ++a) {
    for (int b = 0; b < n_virt; ++b) {
    for (int c = 0; c < n_virt; ++c) {
        denom(a, b, c) = e_ijk - (F_virt(a) + F_virt(b) + F_virt(c));
    }}}
    return t3 / denom;
}


Tensor t3c_ijk_spat(const Tensor& t2, const Tensor& Wvvvo, const Tensor& Wovoo,
               const Tensor& F_occ, const Tensor& F_virt, int i, int j, int k) {
    const int v = Wvvvo.shape[0];
    const int o = Wovoo.shape[0];
    const int v2 = v * v;
    const size_t v3 = static_cast<size_t>(v2) * v;

    Tensor t3({v, v, v});
    t3.fill(0.0);

    // Similar idea in UHF where t2 is already contiguous and W needs packing. However, we don't accumulate directly into 
    // t3 because the output shape varies in the RHF scheme, unlike uhf. For example, t2(c,e) * W(ba,e).T = (cba), not
    // abc, and this varies with all contractions. Therefore, a raw term must be permutated.
    auto vvvo_term = [&](const Tensor& t2_slice, int w_idx1) {
        std::vector<double> raw(v3);
        const double* A = t2_slice.data_ptr();
        Tensor Wpack = Wvvvo.select(3, w_idx1).copy();
        const double* B = Wpack.data_ptr();
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    v, v2, v,
                    1.0, A, v, B, v,
                    0.0, raw.data(), v2);
        return Tensor({v, v, v}, raw);
    };

    auto ovoo_term = [&](const Tensor& t2_slice, int w_idx1, int w_idx2) {
        std::vector<double> raw(v3);
        const double* A = t2_slice.data_ptr();
        Tensor Wpack = Wovoo.select(2, w_idx1).select(2, w_idx2).copy();
        const double* B = Wpack.data_ptr();
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    v2, v, o,
                    1.0, A, v2, B, v,
                    0.0, raw.data(), v);
        return Tensor({v, v, v}, raw);
    };

    // Permutation logic to accumulate the raw output GEMM into t3. (bac) to (abc) would be (1, 0, 2)
    auto add_term = [&](const Tensor& raw, double sign, int pos_a, int pos_b, int pos_c) {
        t3 += sign * raw.permute({pos_a, pos_b, pos_c});
    };

    // Comment shows the old einsum string, then the raw (physical) axis order the GEMM actually produces
    add_term(vvvo_term(t2.select(0, k).select(0, j), i), 1.0, 2, 1, 0); // "bae,ce->abc" (c,b,a)
    add_term(vvvo_term(t2.select(0, j).select(0, k), i), 1.0, 2, 0, 1); // "cae,be->abc" (b,c,a)
    add_term(vvvo_term(t2.select(0, j).select(0, i), k), 1.0, 1, 0, 2); // "ace,be->abc" (b,a,c)
    add_term(vvvo_term(t2.select(0, i).select(0, j), k), 1.0, 0, 1, 2); // "bce,ae->abc" (a,b,c)
    add_term(vvvo_term(t2.select(0, i).select(0, k), j), 1.0, 0, 2, 1); // "cbe,ae->abc" (a,c,b)
    add_term(vvvo_term(t2.select(0, k).select(0, i), j), 1.0, 1, 2, 0); // "abe,ce->abc" (c,a,b)

    add_term(ovoo_term(t2.select(0, i), j, k), -1.0, 0, 1, 2); // "mc,mab->abc" (a,b,c)
    add_term(ovoo_term(t2.select(0, i), k, j), -1.0, 0, 2, 1); // "mb,mac->abc" (a,c,b)
    add_term(ovoo_term(t2.select(0, k), i, j), -1.0, 1, 2, 0); // "mb,mca->abc" (c,a,b)
    add_term(ovoo_term(t2.select(0, k), j, i), -1.0, 2, 1, 0); // "ma,mcb->abc" (c,b,a)
    add_term(ovoo_term(t2.select(0, j), k, i), -1.0, 2, 0, 1); // "ma,mbc->abc" (b,c,a)
    add_term(ovoo_term(t2.select(0, j), i, k), -1.0, 1, 0, 2); // "mc,mba->abc" (b,a,c)

    Tensor denom({v, v, v});
    const double e_ijk = F_occ(i) + F_occ(j) + F_occ(k);
    for (int a = 0; a < v; ++a) {
    for (int b = 0; b < v; ++b) {
    for (int c = 0; c < v; ++c) {
        denom(a, b, c) = e_ijk - (F_virt(a) + F_virt(b) + F_virt(c));
    }}}
    return t3 / denom;
}

// Loops over occupied (i,j,k), builds one (v,v,v) connected-T3 batch (t3c_ijk_so/t3c_ijk_spat above) at a time, 
// and contracts it into the x1/x2 accumulators, meaning that the full O(o^3 v^3) T3 is never created, unlike 
// calc_pert_triples_noddy below. RHF needs both Iprqs and DEprqs, while UHF only uses DEprqs.
// This is a line-by-line direct port from `t_vikings_so` / `t_vikings` in `cctriples.py` in PyCC.
// Source: https://github.com/CrawfordGroup/pycc/blob/main/pycc/cctriples.py
double calc_pert_triples(const Tensor& Iprqs, const Tensor& DEprqs, const Tensor& t1, const Tensor& t2,
                        const Tensor& FN, Sli O, Sli V, bool spin) {
    timing::Scoped _t("Perbutative Triples");

    const int n_occ = O.stop - O.start;
    Tensor F_diag({FN.shape[0]});
    for (int i = 0; i != FN.shape[0]; ++i) { F_diag(i) = FN(i, i); }
    Tensor F_occ  = F_diag.slice(O);
    Tensor F_virt = F_diag.slice(V);
    Tensor Fov    = FN.slice(O, V);

    Tensor x1(t1.shape);
    Tensor x2(t2.shape);

    if (spin) {
    Tensor Wvvvo = DEprqs.slice(V, V, V, O);
    Tensor Wovoo = DEprqs.slice(O, V, O, O);
    
    Tensor Woovv = DEprqs.slice(O, O, V, V);
    Tensor Wvovv = DEprqs.slice(V, O, V, V);
    Tensor Wooov = DEprqs.slice(O, O, O, V);

    for (int i = 0; i < n_occ; ++i) {
    for (int j = 0; j < n_occ; ++j) {
    for (int k = 0; k < n_occ; ++k) {
        timing::Scoped _t("  Build T3");
        
        Tensor t3 = t3c_ijk_spin(t2, Wvvvo, Wovoo, F_occ, F_virt, i, j, k);

        x1.select(0, i) += 0.25 * einsum("bc,abc->a", {Woovv.select(0, j).select(0, k), t3});
        x2.select(0, i).select(0, j) += 0.25 * einsum("c,abc->ab", {Fov.select(0, k), t3});
        x2.select(0, i).select(0, j) += 0.25 * einsum("dbc,abc->ad", {Wvovv.select(1, k), t3});
        x2.select(0, i) -= 0.25 * einsum("md,abd->mab", {Wooov.select(0, j).select(0, k), t3});
    }}}

    x2 = x2 - x2.swapaxes(0, 1) - x2.swapaxes(2, 3) + x2.swapaxes(0, 1).swapaxes(2, 3);

    double et = einsum("ia,ia->", {t1, x1})();
    et += 0.25 * einsum("ijab,ijab->", {t2, x2})();
    return et;
    } else {

    Tensor Wvvvo = Iprqs.slice(V, V, V, O);
    Tensor Wovoo = Iprqs.slice(O, V, O, O);

    Tensor Wvovv = Iprqs.slice(V, O, V, V);
    Tensor Wooov = Iprqs.slice(O, O, O, V);
    Tensor DEoovv = DEprqs.slice(O, O, V, V);

    for (int i = 0; i < n_occ; ++i) {
    for (int j = 0; j < n_occ; ++j) {
    for (int k = 0; k < n_occ; ++k) {
        timing::Scoped _t("  Build T3");

        Tensor t3 = t3c_ijk_spat(t2, Wvvvo, Wovoo, F_occ, F_virt, i, j, k);
        Tensor t3_diff  = t3 - t3.swapaxes(0, 2);
        Tensor t3_comb1 = 2.0 * t3 - t3.swapaxes(1, 2) - t3.swapaxes(0, 2);

        x1.select(0, i) += einsum("abc,bc->a", {t3_diff, DEoovv.select(0, j).select(0, k)});
        x2.select(0, i).select(0, j) += einsum("abc,dbc->ad", {t3_comb1, Wvovv.select(1, k)});
        x2.select(0, i) -= einsum("abc,lc->lab", {t3_comb1, Wooov.select(0, j).select(0, k)});
        x2.select(0, i).select(0, j) += einsum("abc,c->ab", {t3_diff, Fov.select(0, k)});
    }}}

    double et = 2.0 * einsum("ia,ia->", {t1, x1})();
    Tensor t2_comb = 4.0 * t2 - 2.0 * t2.swapaxes(2, 3);
    et += einsum("ijab,ijab->", {t2_comb, x2})();
    return et;
    }
}


// "Noddy" implementation for the perbutative triples in CCSD(T). This is a direct line-by-line
// port from Psi4Numpy's spin-orbital CCSD_T module, which cites Crawford's programming project #6.
// It creates the full O(o^3 v^3) T3 tensor. It's kept for A-B comparison against calc_pert_triples
// above, not called from cc_driver.h. UHF/spin-orbital only. References:
//
// https://github.com/CrawfordGroup/ProgrammingProjects/tree/master/Project%2306
// https://github.com/psi4/psi4numpy/blob/master/Coupled-Cluster/Spin_Orbitals/CCSD/CCSD_T.py
double calc_pert_triples_noddy(const Tensor& Iprqs, const Tensor& t1, const Tensor&t2,
                        const Tensor& FN, const int nbf, const int n_occ, Sli O, Sli V) {
    timing::Scoped _t("Perbutative Triples");

    // Disconnected triples
    Tensor tmp = einsum("ia,jkbc->ijkabc", {t1, Iprqs.slice(O, O, V, V)});
    Tensor t3d = tmp.copy();

    t3d -= tmp.swapaxes(3, 4); // ijkbac
    t3d -= tmp.swapaxes(3, 5); // ijkcba
    t3d -= tmp.swapaxes(0, 1); // jikabc
    t3d -= tmp.swapaxes(0, 2); // kjiabc

    t3d += tmp.swapaxes(0, 1).swapaxes(3, 4); // jikbac
    t3d += tmp.swapaxes(0, 1).swapaxes(3, 5); // jikcba
    t3d += tmp.swapaxes(0, 2).swapaxes(3, 4); // kjibac
    t3d += tmp.swapaxes(0, 2).swapaxes(3, 5); // kjicba

    // Connected triples
    tmp =  einsum("jkae,eibc->ijkabc", {t2, Iprqs.slice(V, O, V, V)});
    tmp -= einsum("imbc,majk->ijkabc", {t2, Iprqs.slice(O, V, O, O)});
    Tensor t3c = tmp.copy();

    t3c -= tmp.swapaxes(3, 4); // ijkbac
    t3c -= tmp.swapaxes(3, 5); // ijkcba
    t3c -= tmp.swapaxes(0, 1); // jikabc
    t3c -= tmp.swapaxes(0, 2); // kjiabc

    t3c += tmp.swapaxes(0, 1).swapaxes(3, 4); // jikbac
    t3c += tmp.swapaxes(0, 1).swapaxes(3, 5); // jikcba
    t3c += tmp.swapaxes(0, 2).swapaxes(3, 4); // kjibac
    t3c += tmp.swapaxes(0, 2).swapaxes(3, 5); // kjicba

    tmp = t3d + t3c;
    Tensor F_diag({FN.shape[0]});
    for (int i=0; i != FN.shape[0]; ++i) {
        F_diag(i) = FN(i, i);
    }
    Tensor F_occ = F_diag.slice(O); Tensor F_virt = F_diag.slice(V);

    const int n_virt = nbf * 2 - n_occ;
    Tensor Dijkabc({n_occ, n_occ, n_occ, n_virt, n_virt, n_virt});

    for (int i=0; i<n_occ; ++i)  for (int j=0; j<n_occ; ++j)  for (int k=0; k<n_occ; ++k)
    for (int a=0; a<n_virt; ++a) for (int b=0; b<n_virt; ++b) for (int c=0; c<n_virt; ++c) {
        Dijkabc(i,j,k,a,b,c) = F_occ(i) + F_occ(j) + F_occ(k)
                            - F_virt(a) - F_virt(b) - F_virt(c);
    }
    tmp /= Dijkabc;

    return (1.0/36) * (t3c * tmp).sum();
}
