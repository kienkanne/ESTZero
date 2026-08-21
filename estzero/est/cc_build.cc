// REFERENCES. The CCSD equations are adopted from the following sources:
// - Formulation of CCSD Equations: [Stanton:1991:4334]
// - Psi4Numpy Coupled-Cluster module: https://github.com/psi4/psi4numpy/tree/master/Coupled-Cluster
// - Dr. Crawford's PyCC repository: https://github.com/CrawfordGroup/pycc/tree/main/pycc

#include "tensor.h"
#include "einsum.h"
#include "cc_build.h"


Dia_Dijab build_Dia_Dijab(const Tensor& FN, const int nbf, const int n_occ, bool spin) {
    const int dim = spin ? 2*nbf : nbf;
    const int n_virt = dim - n_occ;

    Tensor FN_diag({dim});
    for (int i=0; i != dim; ++i) { FN_diag(i) = FN(i, i); }

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

// Stanton Eqn. 9 & 10 (UHF, spin-orbital: t1 outer product is antisymmetrized).
// RHF's spin-adapted tau/tilde_tau use the plain (non-antisymmetrized) outer product instead
Taus build_taus(const Tensor& t1, const Tensor& t2, bool spin) {
    Tensor t1_outer = einsum("ia,jb->ijab", {t1, t1});
    if (!spin) {
        return {t2 + t1_outer, t2 + 0.5 * t1_outer};
    }

    Tensor t1_sub = t1_outer - t1_outer.swapaxes(2,3);
    return {t2 + t1_sub, t2 + 0.5 * t1_sub};
}

// Stanton Eqn 3
Tensor build_Fae(const Tensor& DEprqs, const Tensor& FN, const Tensor& tilde_tau,
                const Tensor& t1, const Tensor& t2, Sli O, Sli V, bool spin) {
    Tensor Fae = FN.slice(V, V).copy();
    for (int i = 0; i != Fae.shape[0]; ++i) { Fae(i, i) = 0; }
    
    Fae -= 0.5 * einsum("me,ma->ae", {FN.slice(O, V), t1});
    Fae += einsum("mf,mafe->ae", {t1, DEprqs.slice(O, V, V, V)});

    double fact = (spin) ? 0.5 : 1.0;
    Fae -= fact * einsum("mnaf,mnef->ae", {tilde_tau, DEprqs.slice(O, O, V, V)});
    return Fae;
}

// Stanton Eqn 4
Tensor build_Fmi(const Tensor& DEprqs, const Tensor& FN, const Tensor& tilde_tau,
                const Tensor& t1, const Tensor& t2, Sli O, Sli V, bool spin) {
    Tensor Fmi = FN.slice(O, O).copy();
    for (int i = 0; i != Fmi.shape[0]; ++i) { Fmi(i, i) = 0; }

    Fmi += 0.5 * einsum("ie,me->mi", {t1, FN.slice(O, V)});
    Fmi += einsum("ne,mnie->mi", {t1, DEprqs.slice(O, O, O, V)});
    
    double fact = (spin) ? 0.5 : 1.0;
    Fmi += fact * einsum("inef,mnef->mi", {tilde_tau, DEprqs.slice(O, O, V, V)});
    return Fmi;
}

// Stanton Eqn 5
Tensor build_Fme(const Tensor& DEprqs, const Tensor& FN, const Tensor& tau,
                const Tensor& t1, const Tensor& t2, Sli O, Sli V, bool spin) {
    Tensor Fmi = FN.slice(O, V).copy();

    Fmi += einsum("nf,mnef->me", {t1, DEprqs.slice(O, O, V, V)});
    return Fmi;
}

// Stanton Eqn 6
// UHF uses DEprqs <pr||qs>, while RHF uses Iprqs (the raw ERI tensor)
Tensor build_Wmnij(const Tensor& _prqs, const Tensor& FN, const Tensor& tau,
                const Tensor& t1, const Tensor& t2, Sli O, Sli V, bool spin) {
    Tensor Wmnij = _prqs.slice(O, O, O, O).copy();

    if (spin) { 
        Tensor Pij = einsum("je,mnie->mnij", {t1, _prqs.slice(O, O, O, V)});
        Wmnij += Pij; Wmnij -= Pij.swapaxes(2, 3);
        Wmnij += 0.25 * einsum("ijef,mnef->mnij", {tau, _prqs.slice(O, O, V, V)}); 

    } else {
        Wmnij += einsum("je,mnie->mnij", {t1, _prqs.slice(O, O, O, V)});
        Wmnij += einsum("ie,mnej->mnij", {t1, _prqs.slice(O, O, V, O)});
        Wmnij += einsum("ijef,mnef->mnij", {tau, _prqs.slice(O, O, V, V)});
    }
    return Wmnij;
}


// Stanton Eqn 8
// UHF uses DEprqs <pr||qs>, while RHF uses Iprqs (the raw ERI tensor). MP3 skips every term that depends on T2
Tensor build_Wmbej(const Tensor& _prqs, const Tensor& FN, const Tensor& tau,
                const Tensor& t1, const Tensor& t2, Sli O, Sli V, bool spin, bool is_MP3) {
    Tensor Wmbej = _prqs.slice(O, V, V, O).copy();

    Wmbej += einsum("jf,mbef->mbej", {t1, _prqs.slice(O, V, V, V)});
    Wmbej -= einsum("nb,mnej->mbej", {t1, _prqs.slice(O, O, V, O)});

    if (!is_MP3) {
        Tensor tmp = (0.5 * t2) + einsum("jf,nb->jnfb", {t1, t1});
        Wmbej -= einsum("jnfb,mnef->mbej", {tmp, _prqs.slice(O, O, V, V)});

        if (!spin) {
            Wmbej += 1.0 * einsum("njfb,mnef->mbej", {t2, _prqs.slice(O, O, V, V)});
            Wmbej -= 0.5 * einsum("njfb,mnfe->mbej", {t2, _prqs.slice(O, O, V, V)});
        }
    }

    return Wmbej;
}

// Stanton Eqn 7 (Only used by UHF)
Tensor build_Wabef(const Tensor& DEprqs, const Tensor& FN, const Tensor& tau,
                const Tensor& t1, const Tensor& t2, Sli O, Sli V) {
    Tensor Wabef = DEprqs.slice(V, V, V, V).copy();

    Tensor Pab = einsum("mb,amef->abef", {t1, DEprqs.slice(V, O, V, V)});
    Wabef -= Pab;
    Wabef += Pab.swapaxes(0, 1);

    Wabef += 0.25 * einsum("mnab,mnef->abef", {tau, DEprqs.slice(O, O, V, V)});
    return Wabef;
}

// This intermediate only appears in spatial orbitals (RHF), which is the
// spin factorization of Wmbej terms. MP3 skips the T2-quadratic self-term.
Tensor build_Wmbje(const Tensor& Iprqs, const Tensor& FN, const Tensor& tau,
                const Tensor& t1, const Tensor& t2, Sli O, Sli V, bool is_MP3) {
    Tensor Wmbje = -1.0 * Iprqs.slice(O, V, O, V).copy();
    Wmbje -= einsum("jf,mbfe->mbje", {t1, Iprqs.slice(O, V, V, V)});
    Wmbje += einsum("nb,mnje->mbje", {t1, Iprqs.slice(O, O, O, V)});

    if (!is_MP3) {
        Tensor tmp = (0.5 * t2) + einsum("jf,nb->jnfb", {t1, t1});
        Wmbje += einsum("jnfb,mnfe->mbje", {tmp, Iprqs.slice(O, O, V, V)});
    }
    return Wmbje;
}

// This intermediate only appears in spatial orbitals (RHF), and is required to 
// build second term of 0.5 * tau_ijef * Wabef
Tensor build_Zmbij(const Tensor& Iprqs, const Tensor& FN, const Tensor& tau,
                const Tensor& t1, const Tensor& t2, Sli O, Sli V) {
    return einsum("mbef,ijef->mbij", {Iprqs.slice(O, V, V, V), tau});
}


double calc_spin_cc_E(const Tensor& DEprqs, const Tensor& FN,
                const Tensor& t1, const Tensor& t2, Sli O, Sli V) {
    const Tensor I_oovv = DEprqs.slice(O, O, V, V);
    double E_cc = 0.25 * einsum("ijab,ijab->", {I_oovv, t2})();
    
    E_cc += einsum("ia,ia->", {FN.slice(O, V), t1})();
    E_cc += 0.5 * einsum("ijab,ia,jb->", {I_oovv, t1, t1})();

    return E_cc;
}


double calc_spat_cc_E(const Tensor& Iprqs, const Tensor& FN,
                const Tensor& t1, const Tensor& t2, Sli O, Sli V) {
    const Tensor tau = t2 + einsum("ia,jb->ijab", {t1, t1});
    const Tensor I_oovv = Iprqs.slice(O, O, V, V);

    double E_cc = 2.0 * einsum("ijab,ijab->", {tau, I_oovv})();
    E_cc -= einsum("ijab,ijba->", {tau, I_oovv})();
    E_cc += 2.0 * einsum("ia,ia->", {FN.slice(O, V), t1})();

    return E_cc;
}