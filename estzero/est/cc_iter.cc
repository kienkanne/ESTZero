#include <cmath>
#include <print>
#include "tensor.h"
#include "cc_build.h"
#include "cc_spin.h"
#include "cc_iter.h"
#include "timer.h"

// Iterates the CCSD amplitude t1 and t2 equations to convergence. Currently no diis yet (TODO)
CCResult ccsd_iterate(double econv, int max_iter, int verbose, const Tensor& ASpqrs, const Tensor& FN,
                        const Tensor& Dia, const Tensor& Dijab, Sli O, Sli V) {
    Tensor t1(Dia.shape); // zero initial guess
    Tensor t2 = ASpqrs.slice(O, O, V, V) / Dijab; // MP2 initial guess

    double E_ccsd = 0.0;
    double E_ccsd_last = 0.0;
    double dE = 0.0;
    bool converged = false;

    for (int iter = 0; iter < max_iter; ++iter) {
        timing::Scoped _t("CCSD Iteration");
        auto [tau, tilde_tau] = build_taus(t1, t2);

        Tensor Fae = build_Fae(ASpqrs, FN, tilde_tau, t1, t2, O, V);
        Tensor Fmi = build_Fmi(ASpqrs, FN, tilde_tau, t1, t2, O, V);
        Tensor Fme = build_Fme(ASpqrs, FN, t1, O, V);

        Tensor Wmnij = build_Wmnij(ASpqrs, FN, tau, t1, t2, O, V);
        Tensor Wabef = build_Wabef(ASpqrs, FN, tau, t1, t2, O, V);
        Tensor Wmbej = build_Wmbej(ASpqrs, FN, tau, t1, t2, O, V);

        Tensor t1_new = build_T1(ASpqrs, FN, Fae, Fmi, Fme, t1, t2, O, V);
        Tensor t2_new = build_T2(ASpqrs, tau, Fae, Fmi, Fme, Wmnij, Wabef, Wmbej, t1, t2, O, V);

        t1 = t1_new / Dia;
        t2 = t2_new / Dijab;

        E_ccsd = calc_ccsd_E(ASpqrs, FN, t1, t2, O, V);
        double dE = E_ccsd - E_ccsd_last;

        if (verbose >= 2) {
        std::println("CCSD Iter {:3} | E_ccsd = {:.8f} | dE = {:.3e}", iter+1, E_ccsd, dE);
        }

        if (std::abs(E_ccsd - E_ccsd_last) < econv) {
            if (verbose >= 2) { std::println("CCSD converged!"); };
            return {E_ccsd, t1, t2}; 
        }
    
        E_ccsd_last = E_ccsd;
    }

    throw std::runtime_error("CCSD failed to converge.");
}
