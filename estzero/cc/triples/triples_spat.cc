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
Tensor t3c_ijk_spat_einsum(const Tensor& t2, const TriplesBlocks& W, const int o, const int v,
               const Tensor& F_occ, const Tensor& F_virt, int i, int j, int k) {
    const auto Wvvvo = [&](int x) { return W.vvvo.select(0, x); };                 // (b, a, e)
    const auto Wovoo = [&](int x, int y) { return W.ovoo.select(0, x).select(0, y); }; // (m, c)

    Tensor t3 = einsum("bae,ce->abc", Wvvvo(i), t2.select(0, k).select(0, j));
    t3 += einsum("cae,be->abc", Wvvvo(i), t2.select(0, j).select(0, k));
    t3 += einsum("ace,be->abc", Wvvvo(k), t2.select(0, j).select(0, i));
    t3 += einsum("bce,ae->abc", Wvvvo(k), t2.select(0, i).select(0, j));
    t3 += einsum("cbe,ae->abc", Wvvvo(j), t2.select(0, i).select(0, k));
    t3 += einsum("abe,ce->abc", Wvvvo(j), t2.select(0, k).select(0, i));

    t3 -= einsum("mc,mab->abc", Wovoo(j, k), t2.select(0, i));
    t3 -= einsum("mb,mac->abc", Wovoo(k, j), t2.select(0, i));
    t3 -= einsum("mb,mca->abc", Wovoo(i, j), t2.select(0, k));
    t3 -= einsum("ma,mcb->abc", Wovoo(j, i), t2.select(0, k));
    t3 -= einsum("ma,mbc->abc", Wovoo(k, i), t2.select(0, j));
    t3 -= einsum("mc,mba->abc", Wovoo(i, k), t2.select(0, j));

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

    Tensor x1(t1.shape()), x2(t2.shape());

    // Each block copied once, contiguous, loop indices leading (see TriplesBlocks)
    const TriplesBlocks W{
        .vvvo = Iprqs.slice(V, V, V, O).permute(3, 0, 1, 2).copy(),
        .ovoo = Iprqs.slice(O, V, O, O).permute(2, 3, 0, 1).copy(),
        .oovv = DEprqs.slice(O, O, V, V).copy(),
        .vovv = Iprqs.slice(V, O, V, V).permute(1, 0, 2, 3).copy(),
        .ooov = Iprqs.slice(O, O, O, V).copy(),
    };

    for (int i = 0; i < o; ++i) {
    for (int j = 0; j < o; ++j) {
    for (int k = 0; k < o; ++k) {
        timing::Scoped _t("  Build T3");

        const Tensor t3 = t3c_ijk_spat_einsum(t2, W, o, v, F_occ, F_virt, i, j, k);
        const Tensor t3_diff = t3 - t3.swapaxes(0, 2);
        const Tensor t3_comb1 = 2.0 * t3 - t3.swapaxes(1, 2) - t3.swapaxes(0, 2);

        // Same contractions as the UHF version, just no 0.25 factor here
        x1.select(0, i) += einsum("abc,bc->a", t3_diff, W.oovv.select(0, j).select(0, k));
        x2.select(0, i).select(0, j) += einsum("abc,dbc->ad", t3_comb1, W.vovv.select(0, k));
        x2.select(0, i) -= einsum("mc,abc->mab", W.ooov.select(0, j).select(0, k), t3_comb1);
        x2.select(0, i).select(0, j) += einsum("abc,c->ab", t3_diff, Fov.select(0, k));
    }}}

    double et = 2.0 * (t1 * x1).sum();
    Tensor t2_comb = 4.0 * t2 - 2.0 * t2.swapaxes(2, 3);
    et += (t2_comb * x2).sum();
    return et;
}
