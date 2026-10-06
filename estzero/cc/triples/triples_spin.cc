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
Tensor t3c_ijk_spin_einsum(const Tensor& t2, TriplesBufferSpin& tb, const int o, const int v,
               const Tensor& F_occ, const Tensor& F_virt, int i, int j, int k) {

    Tensor abc = einsum("ad,bcd->abc", t2.select(0, i).select(0, j), tb.Wvvvo_k);
    abc -= einsum("ad,bcd->abc", t2.select(0, k).select(0, j), tb.Wvvvo_i);
    abc -= einsum("ad,bcd->abc", t2.select(0, i).select(0, k), tb.Wvvvo_j);
    Tensor t3 = abc - abc.swapaxes(0, 1) - abc.swapaxes(0, 2);

    abc = einsum("lab,lc->abc", t2.select(0, i), tb.Wovoo_jk);
    abc -= einsum("lab,lc->abc", t2.select(0, j), tb.Wovoo_ik);
    abc -= einsum("lab,lc->abc", t2.select(0, k), tb.Wovoo_ji);
    t3 -= (abc - abc.swapaxes(0, 2) - abc.swapaxes(1, 2));

    Tensor denom({v, v, v});
    const double e_ijk = F_occ(i) + F_occ(j) + F_occ(k);
    for (int a = 0; a < v; ++a) {
    for (int b = 0; b < v; ++b) {
    for (int c = 0; c < v; ++c) {
        denom(a, b, c) = e_ijk - (F_virt(a) + F_virt(b) + F_virt(c));
    }}}
    return t3 / denom;
}

// This is a line-by-line direct port from `t_vikings_so` in `cctriples.py` in PyCC.
// Source: https://github.com/CrawfordGroup/pycc/blob/main/pycc/cctriples.py
double calc_pert_triples_spin(const Tensor& DEprqs, const Tensor& t1, const Tensor& t2,
                        const Tensor& FN, Sli O, Sli V) {
    const int o = O.stop - O.start;
    const int v = V.stop - V.start;

    Tensor F_diag({FN.shape()[0]});
    for (int i = 0; i != FN.shape()[0]; ++i) { F_diag(i) = FN(i, i); }
    Tensor F_occ  = F_diag.slice(O);
    Tensor F_virt = F_diag.slice(V);
    Tensor Fov    = FN.slice(O, V);

    auto tb  = TriplesBufferSpin(t1, t2, o, v);
    Tensor& x1 = tb.x1; Tensor& x2 = tb.x2; Tensor& t3 = tb.t3;

    // Non-copying views
    Tensor Wvvvo = DEprqs.slice(V, V, V, O);
    Tensor Wovoo = DEprqs.slice(O, V, O, O);

    Tensor Woovv = DEprqs.slice(O, O, V, V);
    Tensor Wvovv = DEprqs.slice(V, O, V, V);
    Tensor Wooov = DEprqs.slice(O, O, O, V);

    // .assign() writes values into the buffer's already allocated storage in place
    for (int i = 0; i < o; ++i) {
        tb.Wvvvo_i.assign(Wvvvo.select(3, i));

    for (int j = 0; j < o; ++j) {
        tb.Wvvvo_j.assign(Wvvvo.select(3, j));
        tb.Wovoo_ji.assign(Wovoo.select(2, j).select(2, i));

    for (int k = 0; k < o; ++k) {
        tb.Wvvvo_k.assign(Wvvvo.select(3, k));
        tb.Wovoo_jk.assign(Wovoo.select(2, j).select(2, k));
        tb.Wovoo_ik.assign(Wovoo.select(2, i).select(2, k));

        timing::Scoped _t("  Build T3");

        t3.assign(t3c_ijk_spin_einsum(t2, tb, o, v, F_occ, F_virt, i, j, k));

        // x1/x2 slices are views, so += / -= accumulate straight into the buffers
        x1.select(0, i) += 0.25 * einsum("abc,bc->a", t3, Woovv.select(0, j).select(0, k));
        x2.select(0, i).select(0, j) += 0.25 * einsum("abc,c->ab", t3, Fov.select(0, k));
        x2.select(0, i).select(0, j) += 0.25 * einsum("abc,dbc->ad", t3, Wvovv.select(1, k));
        x2.select(0, i) -= 0.25 * einsum("mc,abc->mab", Wooov.select(0, j).select(0, k), t3);
    }}}

    x2 = x2 - x2.swapaxes(0, 1) - x2.swapaxes(2, 3) + x2.swapaxes(0, 1).swapaxes(2, 3);

    double et = (t1 * x1).sum();
    et += 0.25 * (t2 * x2).sum();
    return et;
}
