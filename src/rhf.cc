#include <vector>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <libint2.hpp>
#include <print>
#include "rhf.h"
#include "build_JK.h"
#include "diis.h"


double roothan_solve(const Eigen::MatrixXd& S, const Eigen::MatrixXd& T,
                     const Eigen::MatrixXd& V, const std::vector<double>& ERI,
                     double V_nn, int ndocc, int max_iter, double tol) {
    int nbf = S.rows();

    Eigen::MatrixXd H = T + V;
    Eigen::MatrixXd F = H;
	Eigen::MatrixXd X = sym_ortho(S);
    
    double E0 = 0.0;
	double E0_last = 1e2;
    double dE = E0 - E0_last;
    double rms_err = 1e2;

    std::vector<Eigen::MatrixXd> F_list;
    std::vector<Eigen::MatrixXd> err_list;
					
    for (int i = 0; i != max_iter; ++i) {

        Eigen::MatrixXd F_p = X.transpose() * F * X;
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(F_p);
        auto& C_p = solver.eigenvectors();

        Eigen::MatrixXd C = X * C_p;
        Eigen::MatrixXd C_docc = C(Eigen::placeholders::all, Eigen::seq(0, ndocc - 1)); 

        Eigen::MatrixXd D = C_docc * C_docc.transpose();
        
		auto [J, K] = build_JK(ERI, D);
        F = H + 2.0 * J - K;

        F_list.push_back(F);

        Eigen::MatrixXd err_vec = F * D * S - S * D * F;
        err_list.push_back(err_vec);

        // Frobenius norm, then divide to get mean
        rms_err = err_vec.norm() / std::sqrt(err_vec.size());

		E0 = D.cwiseProduct(H + F).sum() + V_nn;
		dE = E0 - E0_last;

        std::println("Iter {:3} | E = {:.8f} | dE = {:.8f} | rms_E {:.8f}", i, E0, dE, rms_err);
    
		if ((std::abs(dE) < tol) && (rms_err < tol)) {
            std::println("SCF converged!");
			return E0;
		}

        // Use a slice of the history to limit DIIS memory
        // Avoid old iterations from polluting the subspace
        if (err_list.size() > 5) {
            err_list.erase(err_list.begin());
            F_list.erase(F_list.begin());

            Eigen::VectorXd coeffs = diis(err_list);
            Eigen::MatrixXd F_diis = Eigen::MatrixXd::Zero(nbf, nbf);

            for (std::size_t i = 0; i < err_list.size(); ++i) {
                F_diis += coeffs[i] * F_list[i]; 
            }
            // Damped DIIS
            F = 1.0 * F_diis + 0.0 * F;
        }
        E0_last = E0;
    }

    std::cout << "SCF failed to converge.\n";
    return E0;
}