#include "tensor.h"
#include "einsum.h"
#include "cc_spat.h"
// Stanton Eqn 1
Tensor build_spat_T1(const Tensor& Iprqs, const Tensor& FN,
                const Tensor& Fae, const Tensor& Fmi, const Tensor& Fme,
                const Tensor& t1, const Tensor& t2, Sli O, Sli V, bool spin) {
    Tensor t1_new = FN.slice(O, V).copy();

    t1_new += einsum("ie,ae->ia", {t1, Fae});
    t1_new -= einsum("ma,mi->ia", {t1, Fmi});

    t1_new += einsum("imae,me->ia", {t2, Fme}) * 2.0;
    t1_new -= einsum("imea,me->ia", {t2, Fme});

    t1_new += einsum("nf,nafi->ia", {t1, Iprqs.slice(O, V, V, O)}) * 2.0;
    t1_new -= einsum("nf,naif->ia", {t1, Iprqs.slice(O, V, O, V)});
    
    t1_new += einsum("mief,maef->ia", {t2, Iprqs.slice(O, V, V, V)}) * 2.0;
    t1_new -= einsum("mife,maef->ia", {t2,Iprqs.slice(O, V, V, V)});

    t1_new -= einsum("mnae,nmei->ia", {t2, Iprqs.slice(O, O, V, O)}) * 2.0;
    t1_new += einsum("mnae,nmie->ia", {t2, Iprqs.slice(O, O, O, V)});
    return t1_new;
}

// Stanton Eqn 2
Tensor build_spat_T2(const Tensor& Iprqs, const Tensor& tau, // (takes tau, not FN)
                const Tensor& Fae, const Tensor& Fmi, const Tensor& Fme,
                const Tensor& Wmnij, const Tensor& Wmbej, const Tensor& Wmbje, const Tensor& Zmbij,
                const Tensor& t1, const Tensor& t2, Sli O, Sli V) {
    Tensor t2_new = Iprqs.slice(O, O, V, V).copy();

    auto add_pair = [&t2_new](const Tensor& term) {
        t2_new += term;
        t2_new += term.swapaxes(0, 1).swapaxes(2, 3);
    };

    add_pair(einsum("ijae,be->ijab", {t2, Fae}));
    add_pair(-0.5 * einsum("ijae,be->ijab", {
        t2, einsum("mb,me->be", {t1, Fme})}));
    add_pair(-1.0 * einsum("imab,mj->ijab", {t2, Fmi}));
    add_pair(-0.5 * einsum("imab,jm->ijab", {
        t2, einsum("je,me->jm", {t1, Fme})}));

    t2_new += einsum("mnab,mnij->ijab", {tau, Wmnij});
    t2_new += einsum("ijef,abef->ijab", {tau, Iprqs.slice(V, V, V, V)});
    Tensor z_term = einsum("ma,mbij->ijab", {t1, Zmbij});
    add_pair(-1.0 * z_term);

    Tensor tmp = einsum("imae,mbej->ijab", {t2, Wmbej});
    tmp -= einsum("imea,mbej->ijab", {t2, Wmbej});
    add_pair(tmp);
    tmp = einsum("imae,mbej->ijab", {t2, Wmbej});
    tmp += einsum("imae,mbje->ijab", {t2, Wmbje});
    add_pair(tmp);
    tmp = einsum("mjae,mbie->ijab", {t2, Wmbje});
    add_pair(tmp);

    tmp = einsum("ie,mbej,ma->ijab", {t1, Iprqs.slice(O, V, V, O), t1});
    add_pair(-1.0 * tmp);
    tmp = einsum("ie,maje,mb->ijab", {t1, Iprqs.slice(O, V, O, V), t1});
    add_pair(-1.0 * tmp);
    add_pair(einsum("ie,abej->ijab", {t1, Iprqs.slice(V, V, V, O)}));
    add_pair(-1.0 * einsum("ma,mbij->ijab", {t1, Iprqs.slice(O, V, O, O)}));
    return t2_new;
}


double calc_spat_cc_E(const Tensor& Iprqs, const Tensor& FN, const Tensor& tau,
                const Tensor& t1, const Tensor& t2, Sli O, Sli V) {
    double E_cc = 2.0 * einsum("ijab,ijab->", {tau, Iprqs.slice(O, O, V, V)})();
    
    E_cc -= einsum("ijab,ijba->", {tau, Iprqs.slice(O, O, V, V)})();
    E_cc += 2.0 * einsum("ia,ia->", {FN.slice(O, V), t1})();

    return E_cc;
}
