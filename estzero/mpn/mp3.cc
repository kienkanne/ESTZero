#include "tensor.h"
#include "einsum.h"
#include "timer.h"
#include "amps.h"
#include "mp3.h"

double calc_mp3_E(const Tensor& Iprqs, const Tensor& DEprqs, const Tensor& FN,
                      const Tensor& Dia, const Tensor& Dijab, Sli O, Sli V, bool spin) {
    timing::Scoped _t("MP3 Energy Calculation");

    Tensor t1(Dia.shape);
    Tensor t2 = spin ? DEprqs.slice(O, O, V, V) / Dijab
                     : Iprqs.slice(O, O, V, V) / Dijab;

    double E_mp3;

    // The intermediates that require tau is different. "MBPT(3) is obtained by performing one CC iteration where 
    // all nonlinear terms are skipped" [Stanton:1991:4334]. The non-linear terms here are the T2 quadratic terms
    // (since tau is just t2 here because t1 is zero)
    //
    // With t1=0 and canonical HF orbitals (FN diagonal), every intermediate (Fae, Fmi, Fme, and RHF's Zmbij) goes
    // to zero W intermediate is just an integral slice, so the builders are not necessary here. The build_spin_T2
    // and build_spat_T2 also simplifies greatly. The equations below are adopted from the T2 builders but with the
    // zero terms removed to save arithmetic.
    if (spin) {
        Tensor t2_mp3 = DEprqs.slice(O, O, V, V).copy();
        t2_mp3 += 0.5 * einsum("mnab,mnij->ijab", {t2, DEprqs.slice(O, O, O, O)}); // Wmnij
        t2_mp3 += 0.5 * einsum("ijef,abef->ijab", {t2, DEprqs.slice(V, V, V, V)}); // Wabef

        // P(ij) * P(ab)
        Tensor ring = einsum("imae,mbej->ijab", {t2, DEprqs.slice(O, V, V, O)}); // Wmbej
        t2_mp3 += ring;
        t2_mp3 -= ring.swapaxes(2, 3);
        t2_mp3 -= ring.swapaxes(0, 1);
        t2_mp3 += ring.swapaxes(0, 1).swapaxes(2, 3);

        t2_mp3 = t2_mp3 / Dijab;
        E_mp3 = calc_spin_cc_E(DEprqs, FN, t2_mp3, t1, t2_mp3, O, V);
    } else {
        Tensor t2_mp3 = Iprqs.slice(O, O, V, V).copy();
        t2_mp3 += einsum("mnab,mnij->ijab", {t2, Iprqs.slice(O, O, O, O)}); // Wmnij
        t2_mp3 += einsum("ijef,abef->ijab", {t2, Iprqs.slice(V, V, V, V)}); // Wabef

        const Tensor Wmbej = Iprqs.slice(O, V, V, O);
        const Tensor Wmbje = Iprqs.slice(O, V, O, V);

        // similar to the helper in build_spat_T2.
        auto add_pair = [&t2_mp3](const Tensor& term) {
            t2_mp3 += term;
            t2_mp3 += term.swapaxes(0, 1).swapaxes(2, 3);
        };

        Tensor tmp = einsum("imae,mbej->ijab", {t2, Wmbej});
        tmp -= einsum("imea,mbej->ijab", {t2, Wmbej});
        add_pair(tmp);

        tmp = einsum("imae,mbej->ijab", {t2, Wmbej});
        tmp -= einsum("imae,mbje->ijab", {t2, Wmbje});
        add_pair(tmp);

        tmp = -1.0 * einsum("mjae,mbie->ijab", {t2, Wmbje});
        add_pair(tmp);

        t2_mp3 = t2_mp3 / Dijab;
        E_mp3 = calc_spat_cc_E(Iprqs, FN, t2_mp3, t1, t2_mp3, O, V);
    }

    return E_mp3;
}
