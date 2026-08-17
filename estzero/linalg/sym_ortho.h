#ifndef SYMORTHO
#define SYMORTHO

#include "tensor.h"

inline Matrix sym_ortho(const Matrix &S, double tol = 1e-8) {
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

#endif
