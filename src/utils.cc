#include <Eigen/Dense>
#include <libint2.hpp>

double compute_nuclear_repulsion(const std::vector<libint2::Atom>& atoms) {
    double vnn = 0.0;
    
    for (size_t i = 0; i < atoms.size(); ++i) {
        for (size_t j = 0; j < i; ++j) {
            double dx = atoms[i].x - atoms[j].x;
            double dy = atoms[i].y - atoms[j].y;
            double dz = atoms[i].z - atoms[j].z;
            double r_ab = std::sqrt(dx * dx + dy * dy + dz * dz);
            
            // r_ab is in Bohr already
            vnn += (atoms[i].atomic_number * atoms[j].atomic_number) / r_ab;
        }
    }
    
    return vnn;
}


Eigen::MatrixXd sym_ortho (const Eigen::MatrixXd &S, double tol = 1e-8) {
    // This is for symmetric matrix; use EigenSolver for general case
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(S);
    auto s = solver.eigenvalues();
    auto U = solver.eigenvectors();

    // Check for close to 0 eigen values
    for (int i = 0; i != s.size(); ++i) {
        if (s(i) > tol) {
            s(i) = 1 / std::sqrt(s(i));
        }
        else {
            s(i) = 0;
        }
    }

    return U * s.asDiagonal() * U.transpose();
}


int pair_index(int i, int j) {
    return i >= j ? i * (i + 1) / 2 + j : j * (j + 1) / 2 + i;
}


int eri_index(int i, int j, int k, int l) {
    int ij = pair_index(i, j);
    int kl = pair_index(k, l);
    return pair_index(ij, kl);
}