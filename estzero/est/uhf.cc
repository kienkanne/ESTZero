#include <vector>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <libint2.hpp>
#include <print>

#include "uhf.h"
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
    Matrix Fa = H;
    Matrix Fb = H;

    double E0 = 0.0;
	double E0_last = 1e2;
    double dE = E0 - E0_last;
    double rms_err = 1e2;

    std::vector<Matrix> Fa_list;
    std::vector<Matrix> Fb_list;
    std::vector<Matrix> err_list;

    Eigen::SelfAdjointEigenSolver<Matrix> solver;
    for (int i = 0; i != max_iter; ++i) {
        Matrix Fa_p = X.transpose() * Fa * X;
        Matrix Fb_p = X.transpose() * Fb * X;

        solver.compute(Fa_p);
        auto& Ca_p = solver.eigenvectors();
        Vector epsa = solver.eigenvalues(); // Not reference, keep in case of convergence

        Matrix Ca = X * Ca_p;
        Matrix Ca_occ  = Ca.leftCols(n_alpha);
        Matrix Da = Ca_occ * Ca_occ.transpose();

        solver.compute(Fb_p);
        auto& Cb_p = solver.eigenvectors();
        Vector epsb = solver.eigenvalues(); // Not reference, keep in case of convergence

        Matrix Cb = X * Cb_p;
        Matrix Cb_occ  = Cb.leftCols(n_beta);
        Matrix Db = Cb_occ * Cb_occ.transpose();

        JK jka;
        {timing::Scoped _t("  build_JKa");
            jka = build_JK(eri, Da);}
        auto& [Ja, Ka] = jka;

        JK jkb;
        {timing::Scoped _t("  build_JKb");
            jkb = build_JK(eri, Db);}
        auto& [Jb, Kb] = jkb;

        Fa = H + Ja + Jb - Ka;
        Fb = H + Ja + Jb - Kb;

        Fa_list.push_back(Fa);
        Fb_list.push_back(Fb);

        Matrix erra_vec = Fa * Da * S - S * Da * Fa;
        Matrix errb_vec = Fb * Db * S - S * Db * Fb;

        // Concatenate the error vectors
        Matrix err_vec(erra_vec.rows() + errb_vec.rows(), erra_vec.cols());
        err_vec << erra_vec, errb_vec;

        err_list.push_back(err_vec);

        // Frobenius norm, then divide to get mean
        rms_err = err_vec.norm() / std::sqrt(err_vec.size());

		E0 = 0.5 * ((Da + Db).cwiseProduct(H)
                    + Da.cwiseProduct(Fa)
                    + Db.cwiseProduct(Fb)).sum() + V_nn;
		dE = E0 - E0_last;

        if (verbose >= 2) {
        std::println("Iter {:3} | E = {:.8f} | dE = {:.3e} | rms_E {:.3e}", i+1, E0, dE, rms_err);
        }

		if ((std::abs(dE) < econv) && (rms_err < econv)) {
            if (verbose >= 2) { std::println("SCF converged!"); }
			return {E0, Ca, Cb, epsa, epsb};
		}

        // Use a slice of the history to limit DIIS memory
        // Avoid old iterations from polluting the subspace
        if (err_list.size() > 5) {
            err_list.erase(err_list.begin());
            Fa_list.erase(Fa_list.begin());
            Fb_list.erase(Fb_list.begin());

            Vector coeffs = diis(err_list);
            Fa.setZero();
            Fb.setZero();

            for (std::size_t i = 0; i < err_list.size(); ++i) {
                Fa += coeffs[i] * Fa_list[i];
                Fb += coeffs[i] * Fb_list[i];
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
