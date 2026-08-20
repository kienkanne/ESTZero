#include <cmath>
#include <print>
#include "tensor.h"
#include "cc_build.h"
#include "cc_spin.h"
#include "cc_spat.h"
#include "cc_iter.h"
#include "timer.h"
#include "diis.h"

// Flatten and concatenate the t1/t2 residuals into one column so the existing (Matrix-based)
// diis() can be reused.
Matrix concat_errors(const Tensor& err_t1, const Tensor& err_t2) {
    int n1 = err_t1.size(), n2 = err_t2.size();
    Matrix err_vec(n1 + n2, 1);
    std::copy(err_t1.data_ptr(), err_t1.data_ptr() + n1, err_vec.data());
    std::copy(err_t2.data_ptr(), err_t2.data_ptr() + n2, err_vec.data() + n1);
    return err_vec;
}


// Iterates the CCSD amplitude t1 and t2 equations to convergence, DIIS-accelerated on the
// concatenated (t1, t2) amplitude vector (error = amplitude change between iterations, the
// standard CC-DIIS residual for a fixed-point update).
CCResult cc_iterate(double econv, int max_iter, int verbose, const Tensor& Iprqs,
                    const Tensor& DEprqs, const Tensor& FN, const Tensor& Dia, const Tensor& Dijab,
                    Sli O, Sli V, bool spin, bool is_CCD) {
    Tensor t1(Dia.shape); // zero initial guess
    Tensor t2 = spin ? DEprqs.slice(O, O, V, V) / Dijab   // <ij||ab>/D
                     : Iprqs.slice(O, O, V, V) / Dijab;   // <ij|ab>/D

    std::vector<Tensor> t1_list, t2_list;
    std::vector<Matrix> err_list;

    // Beyond this many independent directions, error vectors are forced linearly dependent, which can actually
    // happen tiny test systems with only 1-2 total amplitudes like H2, making the DIIS B-matrix rank-deficient; 
    // The solver returns a degenerate combination that reuses the same couple of history entries every iteration, 
    // so the amplitudes stay at a wrong fixed point. Cap the DIIS subspace so it never exceeds the true DOF count.
    const int max_diis_dim = t1.size() + t2.size();

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

        // T1 doesn't exist for CCD, so its error is left at zero.
        Tensor err_t1 = is_CCD ? Tensor(t1.shape) : Tensor(t1_update - t1);
        Tensor err_t2 = t2_update - t2;

        t1 = t1_update;
        t2 = t2_update;
        if (is_CCD) { t1.fill(0.0); }

        t1_list.push_back(t1);
        t2_list.push_back(t2);
        err_list.push_back(concat_errors(err_t1, err_t2));

        // Use a slice of the history to limit DIIS memory, avoid old iterations polluting the subspace
        if (err_list.size() > 6) {
            err_list.erase(err_list.begin());
            t1_list.erase(t1_list.begin());
            t2_list.erase(t2_list.begin());
        }

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

        if (err_list.size() >= 3 && static_cast<int>(err_list.size()) <= max_diis_dim) {
            Vector coeffs = diis(err_list);
            // Build into fresh tensors rather than mutating t1/t2 in place: t1/t2 still alias the buffers 
            // just got pushed into t1_list/t2_list accumulating into them directly would corrupt history.
            Tensor t1_diis(t1.shape);
            Tensor t2_diis(t2.shape);
            for (std::size_t k = 0; k < err_list.size(); ++k) {
                t1_diis += coeffs[k] * t1_list[k];
                t2_diis += coeffs[k] * t2_list[k];
            }
            t1 = t1_diis;
            t2 = t2_diis;
        }
    }
    throw std::runtime_error("CCSD failed to converge.");
}
