#include <cmath>
#include <print>
#include "tensor.h"
#include "cc_build.h"
#include "cc_spin.h"
#include "cc_spat.h"
#include "cc_iter.h"
#include "timer.h"

// Iterates the CCSD amplitude t1 and t2 equations to convergence. Currently no diis yet (TODO)
CCResult cc_iterate(double econv, int max_iter, int verbose, const Tensor& Iprqs, 
                    const Tensor& DEprqs, const Tensor& FN, const Tensor& Dia, const Tensor& Dijab, 
                    Sli O, Sli V, bool spin, bool is_CCD) {
    Tensor t1(Dia.shape); // zero initial guess
    Tensor t2 = spin ? DEprqs.slice(O, O, V, V) / Dijab   // <ij||ab>/D
                     : Iprqs.slice(O, O, V, V) / Dijab;   // <ij|ab>/D

    double E_cc = 0.0;
    double E_ccsd_last = 0.0;

    for (int iter = 0; iter < max_iter; ++iter) {
        timing::Scoped _t("CCSD Iteration");
        auto [tau, tilde_tau] = build_taus(t1, t2, spin);

        // Intermediates with the same starting tensor (DEprqs)
        Tensor Fae = build_Fae(DEprqs, FN, tilde_tau, t1, t2, O, V, spin);
        Tensor Fmi = build_Fmi(DEprqs, FN, tilde_tau, t1, t2, O, V, spin);
        Tensor Fme = build_Fme(DEprqs, FN, tau, t1, t2, O, V, spin);

        Tensor t1_update; Tensor t2_update;
        // Wmnij and Wmbej is built by both RHF and UHF, but with different starting tensors
        if (spin) {
            Tensor Wmnij = build_Wmnij(DEprqs, FN, tau, t1, t2, O, V, spin);
            Tensor Wmbej = build_Wmbej(DEprqs, FN, tau, t1, t2, O, V, spin);

            Tensor Wabef = build_Wabef(DEprqs, FN, tau, t1, t2, O, V);
            t1_update    = build_spin_T1(DEprqs, FN, Fae, Fmi, Fme, t1, t2, O, V, spin) / Dia;
            t2_update    = build_spin_T2(DEprqs, tau, Fae, Fmi, Fme, Wmnij, Wabef, Wmbej, t1, t2, O, V, spin) / Dijab;

        } else {
            Tensor Wmnij = build_Wmnij(Iprqs, FN, tau, t1, t2, O, V, spin);
            Tensor Wmbej = build_Wmbej(Iprqs, FN, tau, t1, t2, O, V, spin);

            Tensor Wmbje = build_Wmbje(Iprqs, FN, tau, t1, t2, O, V);
            Tensor Zmbij = build_Zmbij(Iprqs, FN, tau, t1, t2, O, V);
            t1_update    = build_spat_T1(Iprqs, FN, Fae, Fmi, Fme, t1, t2, O, V, spin) / Dia;
            t2_update    = build_spat_T2(Iprqs, tau, Fae, Fmi, Fme, Wmnij, Wmbej, Wmbje, Zmbij, t1, t2, O, V) / Dijab;
        }

        t1 = t1_update;
        t2 = t2_update;
        if (is_CCD) { t1.fill(0.0); }

        E_cc = spin ? calc_spin_cc_E(DEprqs, FN, tau, t1, t2, O, V)
                    : calc_spat_cc_E(Iprqs, FN, tau, t1, t2, O, V);

        double dE = E_cc - E_ccsd_last;

        if (verbose >= 2) {
            std::println("CC Iter {:3} | E_cc = {:.8f} | dE = {:.3e}", iter+1, E_cc, dE);
        }

        if (std::abs(E_cc - E_ccsd_last) < econv) {
            if (verbose >= 2) { std::println("CCSD converged!"); };
            return {E_cc, t1, t2};
        }
        E_ccsd_last = E_cc;
    }
    throw std::runtime_error("CCSD failed to converge.");
}
