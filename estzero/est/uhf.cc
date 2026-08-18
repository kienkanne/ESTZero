#include <vector>
#include <print>
#include <Eigen/Eigenvalues>

#include "uhf.h"
#include "tensor.h"
#include "wfn.h"
#include "jk_build.h"
#include "timer.h"
#include "sym_ortho.h"
#include "diis.h"

SCFResult uhf_scf(WaveFunction wfn, double econv, int max_iter, int verbose) {
    const Matrix S = wfn.get_S();
    const Matrix X = sym_ortho(S);
    const Matrix H = wfn.get_T() + wfn.get_V();

    int nbf = H.rows();
    Matrix FA = H;
    Matrix FB = H;

    double E_scf = 0.0;
	double E_scf_last = 0.0;
    double dE = E_scf - E_scf_last;
    double rms_err = 0.0;
    double Vnn = wfn.get_Vnn();

    std::vector<Matrix> FA_list;
    std::vector<Matrix> FB_list;
    std::vector<Matrix> err_list;

    Eigen::SelfAdjointEigenSolver<Matrix> solver;
    for (int i = 0; i != max_iter; ++i) {
        Matrix FA_p = X.transpose() * FA * X;
        Matrix FB_p = X.transpose() * FB * X;

        solver.compute(FA_p);
        auto& CA_p = solver.eigenvectors();
        Vector epsA = solver.eigenvalues(); // Not reference, keep in case of convergence

        Matrix CA = X * CA_p;
        Matrix Cocc_A  = CA.leftCols(wfn.get_n_alpha());
        Matrix DA = Cocc_A * Cocc_A.transpose();

        solver.compute(FB_p);
        auto& CB_p = solver.eigenvectors();
        Vector epsB = solver.eigenvalues(); // Not reference, keep in case of convergence

        Matrix CB = X * CB_p;
        Matrix Cocc_B  = CB.leftCols(wfn.get_n_beta());
        Matrix DB = Cocc_B * Cocc_B.transpose();

        JK jkA; JK jkB;

        {timing::Scoped _t("JK build");
            if (wfn.get_eri_scheme() == ERIScheme::Tensor) { 
                jkA = build_JK(wfn.get_eri_tensor(), DA); 
            } else {
                jkA = build_JK(wfn.get_eri_vector(), DA); 
        }}
        auto& [JA, KA] = jkA;

        {timing::Scoped _t("JK build");
            if (wfn.get_eri_scheme() == ERIScheme::Tensor) { 
                jkB = build_JK(wfn.get_eri_tensor(), DB); 
            } else {
                jkB = build_JK(wfn.get_eri_vector(), DB); 
        }}
        auto& [JB, KB] = jkB;
        
        FA = H + JA + JB - KA;
        FB = H + JA + JB - KB;

        FA_list.push_back(FA);
        FB_list.push_back(FB);

        Matrix errA_vec = FA * DA * S - S * DA * FA;
        Matrix errB_vec = FB * DB * S - S * DB * FB;

        // Concatenate the error vectors
        Matrix err_vec(errA_vec.rows() + errB_vec.rows(), errA_vec.cols());
        err_vec << errA_vec, errB_vec;

        err_list.push_back(err_vec);

        // Frobenius norm, then divide to get mean
        rms_err = err_vec.norm() / std::sqrt(err_vec.size());

		E_scf = 0.5 * ((DA + DB).cwiseProduct(H)
                    + DA.cwiseProduct(FA)
                    + DB.cwiseProduct(FB)).sum() + Vnn;
		dE = E_scf - E_scf_last;

        if (verbose >= 2) {
            std::println("SCF Iter  {:3} | E_scf = {:.8f} | dE = {:.3e} | rms_E {:.3e}", i+1, E_scf, dE, rms_err);
        }

		if ((std::abs(dE) < econv) && (rms_err < econv)) {
            if (verbose >= 2) { std::println("SCF converged!"); }
			return {E_scf, CA, CB, epsA, epsB};
		}

                // Use a slice of the history to limit DIIS memory, avoid old iterations from polluting the subspace
        if (err_list.size() > 6) {
            err_list.erase(err_list.begin());
            FA_list.erase(FA_list.begin());
            FB_list.erase(FB_list.begin());
        }

        if (err_list.size() >= 3) {
            Vector coeffs = diis(err_list);
            FA.setZero();
            FB.setZero();

            for (std::size_t i = 0; i < err_list.size(); ++i) {
                FA += coeffs[i] * FA_list[i];
                FB += coeffs[i] * FB_list[i];
            }
        }
        E_scf_last = E_scf;
    }

    throw std::runtime_error("SCF failed to converge.\n");
}

