#include "tensor.h"
#include "einsum.h"
#include "timer.h"


// Vikings' connected-T3 batch for fixed occupied (i,j,k), called once per (i,j,k) by calc_pert_triples below. 
// This is a direct line-by-line port of `t3c_ijk_so` in `cctriples.py` in PyCC.
// Source: https://github.com/CrawfordGroup/pycc/blob/main/pycc/cctriples.py
Tensor t3c_ijk(const Tensor& t2, const Tensor& Wvvvo, const Tensor& Wovoo,
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


// Loops over occupied (i,j,k), builds one (v,v,v) connected-T3 batch (t3c_ijk above) at a time, 
// and immediately contracts it into the x1/x2 accumulators, meaning that the full O(o^3 v^3) T3 
// is never created, unlike calc_pert_triples_noddy below.
// This is a line-by-line direct port from the `t_vikings_so` in `cctriples.py` in the PyCC project.
// Source: https://github.com/CrawfordGroup/pycc/blob/main/pycc/cctriples.py
double calc_pert_triples(const Tensor& Iprqs, const Tensor& t1, const Tensor& t2,
                        const Tensor& FN, const int nbf, const int n_occ, Sli O, Sli V) {
    timing::Scoped _t("Perbutative Triples");

    Tensor Wvvvo = Iprqs.slice(V, V, V, O);
    Tensor Wovoo = Iprqs.slice(O, V, O, O);
    Tensor Woovv = Iprqs.slice(O, O, V, V);
    Tensor Wvovv = Iprqs.slice(V, O, V, V);
    Tensor Wooov = Iprqs.slice(O, O, O, V);
    Tensor Fov   = FN.slice(O, V);

    Tensor F_diag({FN.shape[0]});
    for (int i = 0; i != FN.shape[0]; ++i) { F_diag(i) = FN(i, i); }
    Tensor F_occ  = F_diag.slice(O);
    Tensor F_virt = F_diag.slice(V);

    Tensor x1(t1.shape);
    Tensor x2(t2.shape);

    for (int i = 0; i < n_occ; ++i) {
    for (int j = 0; j < n_occ; ++j) {
    for (int k = 0; k < n_occ; ++k) {
        Tensor t3 = t3c_ijk(t2, Wvvvo, Wovoo, F_occ, F_virt, i, j, k);

        x1.select(0, i) += 0.25 * einsum("bc,abc->a", {Woovv.select(0, j).select(0, k), t3});
        x2.select(0, i).select(0, j) += 0.25 * einsum("c,abc->ab", {Fov.select(0, k), t3});
        x2.select(0, i).select(0, j) += 0.25 * einsum("dbc,abc->ad", {Wvovv.select(1, k), t3});
        x2.select(0, i) -= 0.25 * einsum("md,abd->mab", {Wooov.select(0, j).select(0, k), t3});
    }}}

    x2 = x2 - x2.swapaxes(0, 1) - x2.swapaxes(2, 3) + x2.swapaxes(0, 1).swapaxes(2, 3);

    double et = einsum("ia,ia->", {t1, x1})();
    et += 0.25 * einsum("ijab,ijab->", {t2, x2})();
    return et;
}


// "Noddy" implementation for the perbutative triples in CCSD(T). This is a direct line-by-line
// port from Psi4Numpy's spin-orbital CCSD_T module, which cites Crawford's programming project #6.
// It creates the full O(o^3 v^3) T3 tensor. It's kept for A-B comparison against calc_pert_triples
// above, not called from cc_driver.h. References:
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
