#include <Eigen/Dense>
#include <libint2.hpp>

using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;

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


Matrix sym_ortho (const Matrix &S, double tol = 1e-8) {
    // This is for symmetric matrix; use EigenSolver for general case
    Eigen::SelfAdjointEigenSolver<Matrix> solver(S);
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


Vector diis(const std::vector<Matrix>& err_list) {
    int size = err_list.size();

    Matrix B = Matrix::Constant(size + 1, size + 1, -1.0);
    B(size, size) = 0.0;

    int n = err_list[0].size();
    Matrix E(size, n);

    for(int i = 0; i != size; ++i) {
        Eigen::Map<const Vector> err_vec(err_list[i].data(), n); // n for vector size
        E.row(i) = err_vec;
    }

    B(Eigen::seq(0, size - 1), Eigen::seq(0, size - 1)) = E * E.transpose();

    Vector rhs = Vector::Zero(size + 1);
    rhs(size) = -1.0;

    Vector coeffs = B.colPivHouseholderQr().solve(rhs);

    // Return only the coefficients (drop the final Lagrange multiplier element)
    return coeffs.head(size);
}