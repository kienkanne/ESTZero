#include <vector>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <libint2.hpp>
#include <print>

#include "rhf.h"
#include "integrals.h"
#include "wfn.h"
#include "jk_build.h"
#include "timer.h"
#include "tensor.h"
#include "sym_ortho.h"
#include "diis.h"
#include "guess.h"


SCFResult rhf_scf(WaveFunction wfn, double econv, int max_iter, int verbose, GuessScheme guess) {
    const Matrix S = wfn.get_S();
    const Matrix X = sym_ortho(S);
    const Matrix H = wfn.get_T() + wfn.get_V();

    int nbf = H.rows();
    Matrix F = initial_guess(H, S, guess);

    double E_scf = 0.0;
	double E_scf_last = 0.0;
    double dE = E_scf - E_scf_last;
    double rms_err = 0.0;
    double Vnn = wfn.get_Vnn();

    std::vector<Matrix> F_list;
    std::vector<Matrix> err_list;

    Eigen::SelfAdjointEigenSolver<Matrix> solver;
    for (int i = 0; i != max_iter; ++i) {
        Matrix F_p = X.transpose() * F * X;

        solver.compute(F_p);
        auto& C_p = solver.eigenvectors();

        Matrix C = X * C_p;
        Matrix C_docc  = C.leftCols(wfn.get_n_docc());
        Matrix D = C_docc * C_docc.transpose();

        JK jk;
        {timing::Scoped _t("JK build");
            if (wfn.get_eri_scheme() == ERIScheme::Tensor) { 
                jk = build_JK(wfn.get_eri_tensor(), D); 
            } else {
                jk = build_JK(wfn.get_eri_vector(), D); 
        }}

        auto& [J, K] = jk;
        F = H + 2.0 * J - K;
        F_list.push_back(F);

        Matrix err_vec = F * D * S - S * D * F;
        err_list.push_back(err_vec);

        // Frobenius norm, then divide to get mean
        rms_err = err_vec.norm() / std::sqrt(err_vec.size());

		E_scf = D.cwiseProduct(H + F).sum() + Vnn;
		dE = E_scf - E_scf_last;

        if (verbose >= 2) {
        std::println("SCF Iter  {:3} | E = {:.8f} | dE = {:.3e} | rms_E {:.3e}", i+1, E_scf, dE, rms_err);
        }

		if ((std::abs(dE) < econv) && (rms_err < econv)) {
            if (verbose >= 2) { std::println("SCF converged!"); }
            Vector eps = solver.eigenvalues();
			return {E_scf, C, C, eps, eps};
		}

        // Use a slice of the history to limit DIIS memory, avoid old iterations from polluting the subspace
        if (err_list.size() > 6) {
            err_list.erase(err_list.begin());
            F_list.erase(F_list.begin());
        }

        if (err_list.size() >= 3) {
            Vector coeffs = diis(err_list);
            F.setZero();

            for (std::size_t i = 0; i < err_list.size(); ++i) {
                F += coeffs[i] * F_list[i];
            }
        }
        E_scf_last = E_scf;
    }

    throw std::runtime_error("SCF failed to converge.");
}
