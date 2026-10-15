#include "tensor.h"
#include "einsum.h"
#include "timer.h"
#include "buffers.h"
#include "dispatch.h"

// Vikings' connected-T3 batch for fixed occupied (i,j,k), spin-orbital/UHF form, called once per
// (i,j,k) by calc_pert_triples_spin below.
// This is a direct line-by-line port of `t3c_ijk_so` in `cctriples.py` in PyCC.
// Source: https://github.com/CrawfordGroup/pycc/blob/main/pycc/cctriples.py
//
// The explicit-GEMM version of this batch (v3.7.x) is t3c_ijk_spin_gemm in legacy/triples_gemm.cc.
Tensor t3c_ijk_spin_einsum(const Tensor& t2, const TriplesBlocks& W, const Tensor& F_occ, 
                            const Tensor& eps_abc, int i, int j, int k) {
    // Helpers for slicing
    const auto Wvvvo = [&](int x) { return W.vvvo.select(0, x); };
    const auto Wovoo = [&](int x, int y) { return W.ovoo.select(0, x).select(0, y); };

    Tensor abc = einsum("ad,bcd->abc", t2.select(0, i).select(0, j), Wvvvo(k));
    abc -= einsum("ad,bcd->abc", t2.select(0, k).select(0, j), Wvvvo(i));
    abc -= einsum("ad,bcd->abc", t2.select(0, i).select(0, k), Wvvvo(j));
    Tensor t3 = abc - abc.swapaxes(0, 1) - abc.swapaxes(0, 2);

    abc = einsum("lab,lc->abc", t2.select(0, i), Wovoo(j, k));
    abc -= einsum("lab,lc->abc", t2.select(0, j), Wovoo(i, k));
    abc -= einsum("lab,lc->abc", t2.select(0, k), Wovoo(j, i));
    t3 -= (abc - abc.swapaxes(0, 2) - abc.swapaxes(1, 2));

    const double e_ijk = F_occ(i) + F_occ(j) + F_occ(k);
    divide_by_denominator(t3, e_ijk, eps_abc);

    return t3;
}

// This is a line-by-line direct port from `t_vikings_so` in `cctriples.py` in PyCC.
// Source: https://github.com/CrawfordGroup/pycc/blob/main/pycc/cctriples.py
double calc_pert_triples_spin(const Tensor& DEprqs, const Tensor& t1, const Tensor& t2,
                        const Tensor& FN, Sli O, Sli V) {
    const int o = O.stop - O.start;
    const int v = V.stop - V.start;

    Tensor F_diag({FN.shape()[0]});
    for (int i = 0; i != FN.shape()[0]; ++i) { F_diag(i) = FN(i, i); }
    const Tensor F_occ  = F_diag.slice(O);
    const Tensor F_virt = F_diag.slice(V);
    const Tensor Fov    = FN.slice(O, V);

    const Tensor eps_abc = virtual_denominator(F_virt, v);
    Tensor x1(t1.shape()), x2(t2.shape());

    // Each block copied once, contiguous, loop indices leading (see TriplesBlocks)
    const TriplesBlocks W{
        .vvvo = DEprqs.slice(V, V, V, O).permute(3, 0, 1, 2).copy(),
        .ovoo = DEprqs.slice(O, V, O, O).permute(2, 3, 0, 1).copy(),
        .oovv = DEprqs.slice(O, O, V, V).copy(),
        .vovv = DEprqs.slice(V, O, V, V).permute(1, 0, 2, 3).copy(),
        .ooov = DEprqs.slice(O, O, O, V).copy(),
    };

    for (int i = 0; i < o; ++i) {
    for (int j = 0; j < o; ++j) {
    for (int k = 0; k < o; ++k) {
        timing::Scoped _t("  Build T3");

        const Tensor t3 = t3c_ijk_spin_einsum(t2, W, F_occ, eps_abc, i, j, k);

        // x1/x2 slices are views, so += / -= accumulate straight into the buffers
        x1.select(0, i) += einsum("abc,bc->a", t3, W.oovv.select(0, j).select(0, k));
        x2.select(0, i).select(0, j) += einsum("abc,c->ab", t3, Fov.select(0, k));
        x2.select(0, i).select(0, j) += einsum("abc,dbc->ad", t3, W.vovv.select(0, k)); // axis 0 instead of 1 with permute trick
        x2.select(0, i) -= einsum("mc,abc->mab", W.ooov.select(0, j).select(0, k), t3);
    }}}

    // The 0.25 of every x1/x2 contribution, scaled once
    x1 *= 0.25;
    x2 *= 0.25;

    x2 = x2 - x2.swapaxes(0, 1) - x2.swapaxes(2, 3) + x2.swapaxes(0, 1).swapaxes(2, 3);

    double et = (t1 * x1).sum();
    et += 0.25 * (t2 * x2).sum();
    return et;
}
