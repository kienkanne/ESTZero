#include <cmath>
#include <print>
#include "tensor.h"
#include "cc_build.h"
#include "cc_spin.h"
#include "cc_iter.h"
#include "timer.h"

// Iterates the CCSD amplitude t1 and t2 equations to convergence. Currently no diis yet (TODO)
CCResult cc_iterate(double econv, int max_iter, int verbose, const Tensor& ASprqs, const Tensor& FN,
                        const Tensor& Dia, const Tensor& Dijab, Sli O, Sli V, bool is_CCD) {
    Tensor t1(Dia.shape); // zero initial guess
    Tensor t2 = ASprqs.slice(O, O, V, V) / Dijab; // MP2 initial guess

    double E_cc = 0.0;
    double E_ccsd_last = 0.0;
    double dE = 0.0;
    bool converged = false;

    for (int iter = 0; iter < max_iter; ++iter) {
        timing::Scoped _t("CCSD Iteration");
        auto [tau, tilde_tau] = build_taus(t1, t2);

        Tensor Fae = build_Fae(ASprqs, FN, tilde_tau, t1, t2, O, V);
        Tensor Fmi = build_Fmi(ASprqs, FN, tilde_tau, t1, t2, O, V);
        Tensor Fme = build_Fme(ASprqs, FN, t1, O, V);

        Tensor Wmnij = build_Wmnij(ASprqs, FN, tau, t1, t2, O, V);
        Tensor Wabef = build_Wabef(ASprqs, FN, tau, t1, t2, O, V);
        Tensor Wmbej = build_Wmbej(ASprqs, FN, tau, t1, t2, O, V);

        Tensor t1_new = build_T1(ASprqs, FN, Fae, Fmi, Fme, t1, t2, O, V);
        Tensor t2_new = build_T2(ASprqs, tau, Fae, Fmi, Fme, Wmnij, Wabef, Wmbej, t1, t2, O, V);

        if (!is_CCD) { t1 = t1_new / Dia; }
        t2 = t2_new / Dijab;

        E_cc = calc_cc_E(ASprqs, FN, t1, t2, O, V);
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
