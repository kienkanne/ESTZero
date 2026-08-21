#include "tensor.h"
#include "timer.h"
#include "cc_build.h"
#include "cc_spin.h"
#include "cc_spat.h"
#include "mp3.h"

MP3Result calc_mp3_E(const Tensor& Iprqs, const Tensor& DEprqs, const Tensor& FN,
                      const Tensor& Dia, const Tensor& Dijab, Sli O, Sli V, bool spin) {
    timing::Scoped _t("MP3 Energy Calculation");

    Tensor t1(Dia.shape); // no T1 at this order (first iteration approximation)
    Tensor t2 = spin ? DEprqs.slice(O, O, V, V) / Dijab
                     : Iprqs.slice(O, O, V, V) / Dijab;

    bool is_MP3 = true;

    // tau/tilde_tau is just t2 here as t1=0, so the energy evaluation equations (`calc_*_cc_E`) gets fed t2 instead.
    //
    // The intermediates that require tau is different. "MBPT(3) is obtained by performing one CC iteration where 
    // all  nonlinear terms are skipped" [Stanton:1991:4334]. The non-linear terms here are the T2 quadratic terms
    // (since tau is just t2 here). A straight forward option is just to fed a zero tau below for the functions
    // that build the intermediates. Unlike CC2, MP3 is a more well validated method, so an attempt to avoid the
    // unnecessary contractions will be attempteed next patch.
    Tensor tau_zero(t2.shape);

    double E_mp2, E_mp3_total;

    if (spin) {
        E_mp2 = calc_spin_cc_E(DEprqs, FN, t2, t1, t2, O, V);

        Tensor Fae = build_Fae(DEprqs, FN, tau_zero, t1, t2, O, V, spin);
        Tensor Fmi = build_Fmi(DEprqs, FN, tau_zero, t1, t2, O, V, spin);
        Tensor Fme = build_Fme(DEprqs, FN, tau_zero, t1, t2, O, V, spin);
        Tensor Wmnij = build_Wmnij(DEprqs, FN, tau_zero, t1, t2, O, V, spin);
        Tensor Wabef = build_Wabef(DEprqs, FN, tau_zero, t1, t2, O, V);
        Tensor Wmbej = build_Wmbej(DEprqs, FN, tau_zero, t1, t2, O, V, spin, is_MP3);

        Tensor t2_mp3 = build_spin_T2(DEprqs, t2, Fae, Fmi, Fme, Wmnij, Wabef, Wmbej, t1, t2, O, V, spin) / Dijab;
        E_mp3_total = calc_spin_cc_E(DEprqs, FN, t2_mp3, t1, t2_mp3, O, V);
    } else {
        E_mp2 = calc_spat_cc_E(Iprqs, FN, t2, t1, t2, O, V);

        Tensor Fae = build_Fae(DEprqs, FN, tau_zero, t1, t2, O, V, spin);
        Tensor Fmi = build_Fmi(DEprqs, FN, tau_zero, t1, t2, O, V, spin);
        Tensor Fme = build_Fme(DEprqs, FN, tau_zero, t1, t2, O, V, spin);
        Tensor Wmnij = build_Wmnij(Iprqs, FN, tau_zero, t1, t2, O, V, spin);
        Tensor Wmbej = build_Wmbej(Iprqs, FN, tau_zero, t1, t2, O, V, spin, is_MP3);
        Tensor Wmbje = build_Wmbje(Iprqs, FN, tau_zero, t1, t2, O, V, is_MP3);
        Tensor Zmbij = build_Zmbij(Iprqs, FN, tau_zero, t1, t2, O, V);

        Tensor t2_mp3 = build_spat_T2(Iprqs, t2, Fae, Fmi, Fme, Wmnij, Wmbej, Wmbje, Zmbij, t1, t2, O, V) / Dijab;
        E_mp3_total = calc_spat_cc_E(Iprqs, FN, t2_mp3, t1, t2_mp3, O, V);
    }

    return {E_mp2, E_mp3_total};
}
