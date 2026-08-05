#include <vector>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <libint2.hpp>
#include <print>

#include "rhf.h"
#include "build_JK.h"
#include "timer.h"

using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;

Result get_rhf_energy(double tol, int max_iter, int verbose, double V_nn, int ndocc, 
                    const Matrix& S, const Matrix& T,
                    const Matrix& V, const std::vector<double>& eri_vector) {
    timing::Scoped _t("SCF Iterations");
    
    int nbf = S.rows();

    const Matrix H = T + V;
	const Matrix X = sym_ortho(S);
    Matrix F = H;

    double E0 = 0.0;
	double E0_last = 1e2;
    double dE = E0 - E0_last;
    double rms_err = 1e2;

    std::vector<Matrix> F_list;
    std::vector<Matrix> err_list;
					
    for (int i = 0; i != max_iter; ++i) {

        Matrix F_p = X.transpose() * F * X;
        Eigen::SelfAdjointEigenSolver<Matrix> solver;

        solver.compute(F_p);

        auto& C_p = solver.eigenvectors();

        Matrix C = X * C_p;
        Matrix C_docc  = C.leftCols(ndocc);

        Matrix D = C_docc * C_docc.transpose();
        
        JK jk;
        {
            timing::Scoped _t("  build_JK");
            jk = build_JK(eri_vector, D);
        }
        auto& [J, K] = jk;
        F = H + 2.0 * J - K;

        F_list.push_back(F);

        Matrix err_vec = F * D * S - S * D * F;
        err_list.push_back(err_vec);

        // Frobenius norm, then divide to get mean
        rms_err = err_vec.norm() / std::sqrt(err_vec.size());

		E0 = D.cwiseProduct(H + F).sum() + V_nn;
		dE = E0 - E0_last;

        if (verbose >= 2) {
        std::println("Iter {:3} | E = {:.8f} | dE = {:.3e} | rms_E {:.3e}", i+1, E0, dE, rms_err);
        }

		if ((std::abs(dE) < tol) && (rms_err < tol)) {
            if (verbose >= 2) { std::println("SCF converged!"); }
            auto eps = solver.eigenvalues();
			return {E0, C, eps};
		}

        // Use a slice of the history to limit DIIS memory
        // Avoid old iterations from polluting the subspace
        if (err_list.size() > 5) {
            err_list.erase(err_list.begin());
            F_list.erase(F_list.begin());

            Vector coeffs = diis(err_list);
            Matrix F_diis = Matrix::Zero(nbf, nbf);

            for (std::size_t i = 0; i < err_list.size(); ++i) {
                F_diis += coeffs[i] * F_list[i]; 
            }
            // Damped DIIS
            F = 1.0 * F_diis + 0.0 * F;
        }
        E0_last = E0;
    }

    throw std::runtime_error("SCF failed to converge.\n");
}