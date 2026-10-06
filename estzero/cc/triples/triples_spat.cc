#include "tensor.h"
#include "einsum.h"
#include "timer.h"
#include "buffers.h"
#include "dispatch.h"

// Vikings' connected-T3 batch for fixed occupied (i,j,k), spin-adapted/RHF form, where the six
// permutations are written out explicitly. Wvvvo/Wovoo are raw Iprqs slices, unlike the UHF
// version (triples_spin.cc) which uses DEprqs.
// This is a direct line-by-line port of `t3c_ijk` in `cctriples.py` in PyCC.
// Source: https://github.com/CrawfordGroup/pycc/blob/main/pycc/cctriples.py
//
// The explicit-GEMM version of this batch (v3.7.x) is t3c_ijk_spat_gemm in legacy/triples_gemm.cc.
Tensor t3c_ijk_spat_einsum(const Tensor& t2, TriplesBufferSpat& tb, const int o, const int v,
               const Tensor& F_occ, const Tensor& F_virt, int i, int j, int k) {
    Tensor t3 = einsum("bae,ce->abc", tb.Wvvvo_i, t2.select(0, k).select(0, j));
    t3 += einsum("cae,be->abc", tb.Wvvvo_i, t2.select(0, j).select(0, k));
    t3 += einsum("ace,be->abc", tb.Wvvvo_k, t2.select(0, j).select(0, i));
    t3 += einsum("bce,ae->abc", tb.Wvvvo_k, t2.select(0, i).select(0, j));
    t3 += einsum("cbe,ae->abc", tb.Wvvvo_j, t2.select(0, i).select(0, k));
    t3 += einsum("abe,ce->abc", tb.Wvvvo_j, t2.select(0, k).select(0, i));

    t3 -= einsum("mc,mab->abc", tb.Wovoo_jk, t2.select(0, i));
    t3 -= einsum("mb,mac->abc", tb.Wovoo_kj, t2.select(0, i));
    t3 -= einsum("mb,mca->abc", tb.Wovoo_ij, t2.select(0, k));
    t3 -= einsum("ma,mcb->abc", tb.Wovoo_ji, t2.select(0, k));
    t3 -= einsum("ma,mbc->abc", tb.Wovoo_ki, t2.select(0, j));
    t3 -= einsum("mc,mba->abc", tb.Wovoo_ik, t2.select(0, j));

    Tensor denom({v, v, v});
    const double e_ijk = F_occ(i) + F_occ(j) + F_occ(k);
    for (int a = 0; a < v; ++a) {
    for (int b = 0; b < v; ++b) {
    for (int c = 0; c < v; ++c) {
        denom(a, b, c) = e_ijk - (F_virt(a) + F_virt(b) + F_virt(c));
    }}}
    return t3 / denom;
}

// Loops over occupied (i,j,k), builds one (v,v,v) connected-T3 batch (t3c_ijk_spat_einsum above) at a
// time, and contracts it into the x1/x2 accumulators. RHF/spatial-orbital formulation, needs both
// Iprqs and DEprqs (unlike triples_spin.cc's UHF formulation, which only needs DEprqs).
// This is a line-by-line direct port from `t_vikings` in `cctriples.py` in PyCC.
// Source: https://github.com/CrawfordGroup/pycc/blob/main/pycc/cctriples.py
double calc_pert_triples_spat(const Tensor& Iprqs, const Tensor& DEprqs, const Tensor& t1, const Tensor& t2,
                        const Tensor& FN, Sli O, Sli V) {
    const int o = O.stop - O.start;
    const int v = V.stop - V.start;

    Tensor F_diag({FN.shape()[0]});
    for (int i = 0; i != FN.shape()[0]; ++i) { F_diag(i) = FN(i, i); }
    Tensor F_occ  = F_diag.slice(O);
    Tensor F_virt = F_diag.slice(V);
    Tensor Fov    = FN.slice(O, V);

    auto tb  = TriplesBufferSpat(t1, t2, o, v);
    Tensor& x1 = tb.x1; Tensor& x2 = tb.x2; Tensor& t3 = tb.t3;
    Tensor& t3_diff = tb.t3_diff; Tensor& t3_comb1 = tb.t3_comb1;

    // Non-copying views
    Tensor Wvvvo = Iprqs.slice(V, V, V, O);
    Tensor Wovoo = Iprqs.slice(O, V, O, O);

    Tensor Wvovv = Iprqs.slice(V, O, V, V);
    Tensor Wooov = Iprqs.slice(O, O, O, V);
    Tensor DEoovv = DEprqs.slice(O, O, V, V);

    for (int i = 0; i < o; ++i) {
        tb.Wvvvo_i.assign(Wvvvo.select(3, i));

    for (int j = 0; j < o; ++j) {
        tb.Wvvvo_j.assign(Wvvvo.select(3, j));
        tb.Wovoo_ij.assign(Wovoo.select(2, i).select(2, j));
        tb.Wovoo_ji.assign(Wovoo.select(2, j).select(2, i));

    for (int k = 0; k < o; ++k) {
        tb.Wvvvo_k.assign(Wvvvo.select(3, k));
        tb.Wovoo_jk.assign(Wovoo.select(2, j).select(2, k));
        tb.Wovoo_kj.assign(Wovoo.select(2, k).select(2, j));
        tb.Wovoo_ik.assign(Wovoo.select(2, i).select(2, k));
        tb.Wovoo_ki.assign(Wovoo.select(2, k).select(2, i));

        timing::Scoped _t("  Build T3");

        t3.assign(t3c_ijk_spat_einsum(t2, tb, o, v, F_occ, F_virt, i, j, k));
        t3_diff.assign(t3 - t3.swapaxes(0, 2));
        t3_comb1.assign(2.0 * t3 - t3.swapaxes(1, 2) - t3.swapaxes(0, 2));

        // Same contractions as the UHF version, just no 0.25 factor here
        x1.select(0, i) += einsum("abc,bc->a", t3_diff, DEoovv.select(0, j).select(0, k));
        x2.select(0, i).select(0, j) += einsum("abc,dbc->ad", t3_comb1, Wvovv.select(1, k));
        x2.select(0, i) -= einsum("mc,abc->mab", Wooov.select(0, j).select(0, k), t3_comb1);
        x2.select(0, i).select(0, j) += einsum("abc,c->ab", t3_diff, Fov.select(0, k));
    }}}

    double et = 2.0 * (t1 * x1).sum();
    Tensor t2_comb = 4.0 * t2 - 2.0 * t2.swapaxes(2, 3);
    et += (t2_comb * x2).sum();
    return et;
}
