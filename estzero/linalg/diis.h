#ifndef DIIS
#define DIIS

#include "tensor.h"

inline Vector diis(const std::vector<Matrix>& err_list) {
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

#endif