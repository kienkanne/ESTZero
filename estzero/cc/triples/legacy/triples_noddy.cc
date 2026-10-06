#include "tensor.h"
#include "einsum.h"
#include "timer.h"
#include "triples_noddy.h"

// "Noddy" implementation for the perbutative triples in CCSD(T). This is a direct line-by-line
// port from Psi4Numpy's spin-orbital CCSD_T module, which cites Crawford's programming project #6.
// It creates the full O(o^3 v^3) T3 tensor. It's kept for A-B comparison against calc_pert_triples
// (dispatch.h), not called from cc_driver.h. UHF/spin-orbital only. References:
//
// https://github.com/CrawfordGroup/ProgrammingProjects/tree/master/Project%2306
// https://github.com/psi4/psi4numpy/blob/master/Coupled-Cluster/Spin_Orbitals/CCSD/CCSD_T.py
double calc_pert_triples_noddy(const Tensor& Iprqs, const Tensor& t1, const Tensor&t2,
                        const Tensor& FN, const int nbf, const int n_occ, Sli O, Sli V) {
    timing::Scoped _t("Perbutative Triples");

    // Disconnected triples
    Tensor tmp = einsum("ia,jkbc->ijkabc", t1, Iprqs.slice(O, O, V, V));
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
    tmp =  einsum("jkae,eibc->ijkabc", t2, Iprqs.slice(V, O, V, V));
    tmp -= einsum("imbc,majk->ijkabc", t2, Iprqs.slice(O, V, O, O));
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
    Tensor F_diag({FN.shape()[0]});
    for (int i=0; i != FN.shape()[0]; ++i) {
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
