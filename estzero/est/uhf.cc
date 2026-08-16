#include <vector>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <libint2.hpp>
#include <print>

#include "uhf.h"
#include "wfn.h"
#include "build_JK.h"
#include "timer.h"


template <typename ERI>
Result uhf_scf(double econv, int max_iter, int verbose, double V_nn, int n_alpha, int n_beta,
                const Matrix& S, const Matrix& T,
                const Matrix& V, const ERI& eri) {
    timing::Scoped _t("SCF Iterations");

    int nbf = S.rows();

    const Matrix H = T + V;
	const Matrix X = sym_ortho(S);
    Matrix FA = H;
    Matrix FB = H;

    double E0 = 0.0;
	double E0_last = 1e2;
    double dE = E0 - E0_last;
    double rms_err = 1e2;

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
        Matrix Cocc_A  = CA.leftCols(n_alpha);
        Matrix DA = Cocc_A * Cocc_A.transpose();

        solver.compute(FB_p);
        auto& CB_p = solver.eigenvectors();
        Vector epsB = solver.eigenvalues(); // Not reference, keep in case of convergence

        Matrix CB = X * CB_p;
        Matrix Cocc_B  = CB.leftCols(n_beta);
        Matrix DB = Cocc_B * Cocc_B.transpose();

        JK JKA;
        {timing::Scoped _t("  build_JKa");
            JKA = build_JK(eri, DA);}
        auto& [JA, KA] = JKA;

        JK JKB;
        {timing::Scoped _t("  build_JKb");
            JKB = build_JK(eri, DB);}
        auto& [JB, KB] = JKB;

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

		E0 = 0.5 * ((DA + DB).cwiseProduct(H)
                    + DA.cwiseProduct(FA)
                    + DB.cwiseProduct(FB)).sum() + V_nn;
		dE = E0 - E0_last;

        if (verbose >= 2) {
        std::println("Iter {:3} | E = {:.8f} | dE = {:.3e} | rms_E {:.3e}", i+1, E0, dE, rms_err);
        }

		if ((std::abs(dE) < econv) && (rms_err < econv)) {
            if (verbose >= 2) { std::println("SCF converged!"); }
			return {E0, CA, CB, epsA, epsB};
		}

        // Use a slice of the history to limit DIIS memory
        // Avoid old iterations from polluting the subspace
        if (err_list.size() > 5) {
            err_list.erase(err_list.begin());
            FA_list.erase(FA_list.begin());
            FB_list.erase(FB_list.begin());

            Vector coeffs = diis(err_list);
            FA.setZero();
            FB.setZero();

            for (std::size_t i = 0; i < err_list.size(); ++i) {
                FA += coeffs[i] * FA_list[i];
                FB += coeffs[i] * FB_list[i];
            }
        }
        E0_last = E0;
    }

    throw std::runtime_error("SCF failed to converge.\n");
}

template Result uhf_scf<std::vector<double>>(double, int, int, double, int, int,
                                             const Matrix&, const Matrix&,
                                             const Matrix&, const std::vector<double>&);
template Result uhf_scf<Tensor>(double, int, int, double, int, int,
                                const Matrix&, const Matrix&, const Matrix&,
                                const Tensor&);
