// REFERENCES. The CCSD equations are adopted from the following sources:
// - Formulation of CCSD Equations: [Stanton:1991:4334]
// - Psi4Numpy Coupled-Cluster module: https://github.com/psi4/psi4numpy/tree/master/Coupled-Cluster
// - Dr. Crawford's PyCC repository: https://github.com/CrawfordGroup/pycc/tree/main/pycc

#include "tensor.h"
#include "einsum.h"
#include "cc_build.h"


// Stanton Eqn. 9 & 10
Taus build_taus(const Tensor& t1, const Tensor& t2) {
    Tensor ttau = t2.copy();
    Tensor t1_outer = einsum("ia,jb->ijab", {t1, t1});
    Tensor t1_sub = t1_outer - t1_outer.swapaxes(2,3);

    return {t2 + t1_sub, t2 + 0.5 * t1_sub};
}

// Stanton Eqn 3
Tensor build_Fae(const Tensor& ASpqrs, const Tensor& FN, const Tensor& tlide_tau,
                const Tensor& t1, const Tensor& t2, Sli O, Sli V) {
    Tensor Fae = FN.slice(V, V).copy();
    for (int i = 0; i != Fae.shape[0]; ++i) { Fae(i, i) = 0; }
    
    Fae -= 0.5 * einsum("me,ma->ae", {FN.slice(O, V), t1});
    Fae += einsum("mf,mafe->ae", {t1, ASpqrs.slice(O, V, V, V)});
    
    Fae -= 0.5 * einsum("mnaf,mnef->ae", {tlide_tau, ASpqrs.slice(O, O, V, V)});
    return Fae;
}

// Stanton Eqn 4
Tensor build_Fmi(const Tensor& ASpqrs, const Tensor& FN, const Tensor& tlide_tau,
                const Tensor& t1, const Tensor& t2, Sli O, Sli V) {
    Tensor Fmi = FN.slice(O, O).copy();
    for (int i = 0; i != Fmi.shape[0]; ++i) { Fmi(i, i) = 0; }

    Fmi += 0.5 * einsum("ie,me->mi", {t1, FN.slice(O, V)});
    Fmi += einsum("ne,mnie->mi", {t1, ASpqrs.slice(O, O, O, V)});

    Fmi += 0.5 * einsum("inef,mnef->mi", {tlide_tau, ASpqrs.slice(O, O, V, V)});
    return Fmi;
}

// Stanton Eqn 5
Tensor build_Fme(const Tensor& ASpqrs, const Tensor& FN,
                const Tensor& t1,  Sli O, Sli V) {
    Tensor Fmi = FN.slice(O, V).copy();
    Fmi += einsum("nf,mnef->me", {t1, ASpqrs.slice(O, O, V, V)});
    return Fmi;
}

// Stanton Eqn 6
Tensor build_Wmnij(const Tensor& ASpqrs, const Tensor& FN, const Tensor& tau,
                const Tensor& t1, const Tensor& t2, Sli O, Sli V) {
    Tensor Wmnij = ASpqrs.slice(O, O, O, O).copy();

    Tensor Pij = einsum("je,mnie->mnij", {t1, ASpqrs.slice(O, O, O, V)});
    Wmnij += Pij;
    Wmnij -= Pij.swapaxes(2, 3);

    Wmnij += 0.25 * einsum("ijef,mnef->mnij", {tau, ASpqrs.slice(O, O, V, V)});
    return Wmnij;
}

// Stanton Eqn 7
Tensor build_Wabef(const Tensor& ASpqrs, const Tensor& FN, const Tensor& tau,
                const Tensor& t1, const Tensor& t2, Sli O, Sli V) {
    Tensor Wabef = ASpqrs.slice(V, V, V, V).copy();

    Tensor Pab = einsum("mb,amef->abef", {t1, ASpqrs.slice(V, O, V, V)});

    Wabef -= Pab;
    Wabef += Pab.swapaxes(0, 1);

    Wabef += 0.25 * einsum("mnab,mnef->abef", {tau, ASpqrs.slice(O, O, V, V)});
    return Wabef;
}

// Stanton Eqn 8
Tensor build_Wmbej(const Tensor& ASpqrs, const Tensor& FN, const Tensor& tau,
                const Tensor& t1, const Tensor& t2, Sli O, Sli V) {
    Tensor Wmbej = ASpqrs.slice(O, V, V, O).copy();

    Wmbej += einsum("jf,mbef->mbej", {t1, ASpqrs.slice(O, V, V, V)});
    Wmbej -= einsum("nb,mnej->mbej", {t1, ASpqrs.slice(O, O, V, O)});

    Tensor tmp = (0.5 * t2) + einsum("jf,nb->jnfb", {t1, t1});

    Wmbej -= einsum("jnfb,mnef->mbej", {tmp, ASpqrs.slice(O, O, V, V)});
    return Wmbej;
}

// Stanton Eqn 1
Tensor build_T1(const Tensor& ASpqrs, const Tensor& FN,
                const Tensor& Fae, const Tensor& Fmi, const Tensor& Fme,
                const Tensor& t1, const Tensor& t2, Sli O, Sli V) {
    Tensor t1_new = FN.slice(O, V).copy();
    t1_new += einsum("ie,ae->ia", {t1, Fae});
    t1_new -= einsum("ma,mi->ia", {t1, Fmi});
    t1_new += einsum("imae,me->ia", {t2, Fme});
    t1_new -= einsum("nf,naif->ia", {t1, ASpqrs.slice(O, V, O, V)});
    t1_new -= 0.5 * einsum("imef,maef->ia", {t2, ASpqrs.slice(O, V, V, V)});
    t1_new -= 0.5 * einsum("mnae,nmei->ia", {t2, ASpqrs.slice(O, O, V, O)});
    return t1_new;
}

// Stanton Eqn 2
Tensor build_T2(const Tensor& ASpqrs, const Tensor& tau, // (takes tau, not FN)
                const Tensor& Fae, const Tensor& Fmi, const Tensor& Fme,
                const Tensor& Wmnij, const Tensor& Wabef, const Tensor& Wmbej,
                const Tensor& t1, const Tensor& t2, Sli O, Sli V) {
    Tensor t2_new = ASpqrs.slice(O, O, V, V).copy();
    
    // P(ij) = (Pij - Pji)
    {
    Tensor tmp = Fae - 0.5 * einsum("mb,me->be", {t1, Fme});
    Tensor Pab = einsum("ijae,be->ijab", {t2, tmp});
    t2_new += Pab;
    t2_new -= Pab.swapaxes(2, 3);
    }

    { // Most costly step here
        Tensor tmp = Fmi + 0.5 * einsum("je,me->mj", {t1, Fme});
        Tensor Pij = einsum("imab,mj->ijab", {t2, tmp});
        t2_new -= Pij;
        t2_new += Pij.swapaxes(0, 1);
    }

    t2_new += 0.5 * einsum("mnab,mnij->ijab", {tau, Wmnij});
    t2_new += 0.5 * einsum("ijef,abef->ijab", {tau, Wabef});

    { // And here
        Tensor Pijab = einsum("imae,mbej->ijab", {t2, Wmbej});

        Tensor tmp = einsum("ie,ma,mbej->ijab", {t1, t1, ASpqrs.slice(O, V, V, O)});
        Pijab -= tmp;

        // P(ij) * P(ab): (ij - ji) * (ab - ba) = ijab - ijba - jiab + jiba
        t2_new += Pijab;
        t2_new -= Pijab.swapaxes(2, 3);
        t2_new -= Pijab.swapaxes(0, 1);
        t2_new += Pijab.swapaxes(0, 1).swapaxes(2, 3);
    }

    {
        Tensor Pij = einsum("ie,abej->ijab", {t1, ASpqrs.slice(V, V, V, O)});
        t2_new += Pij;
        t2_new -= Pij.swapaxes(0, 1);
    }
    
    {
        Tensor Pab = einsum("ma,mbij->ijab", {t1, ASpqrs.slice(O, V, O, O)});
        t2_new -= Pab;
        t2_new += Pab.swapaxes(2, 3);
    }

    return t2_new;
}