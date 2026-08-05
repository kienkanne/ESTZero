#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include "utils.h"
#include "mp2.h"
#include "timer.h"

using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;
using Tensor4D = Eigen::Tensor<double, 4>;


Tensor4D transform_MO(const std::vector<double>& eri_vector, const Matrix& C, int n_occ) {
    timing::Scoped _t("MO Transformation");

    const int nbf   = static_cast<int>(C.rows());
    const int n_virt = nbf - n_occ;

    Matrix C_occ  = C.leftCols(n_occ);
    Matrix C_virt = C.rightCols(n_virt);

    Tensor4D I1(n_occ, nbf, nbf, nbf);
    I1.setZero();

    for (int i = 0; i < n_occ; ++i) {
    for (int q = 0; q < nbf; ++q) {
    for (int r = 0; r < nbf; ++r) {
    for (int s = 0; s < nbf; ++s) {
        double val = 0.0;

        for (int p = 0; p < nbf; ++p) {
            val += C_occ(p, i) * eri_vector[eri_index(p, q, r, s)];
        }

        I1(i, q, r, s) = val;
    }
    }
    }
    }

    Tensor4D I2(n_occ, n_virt, nbf, nbf);
    I2.setZero();

    for (int i = 0; i < n_occ; ++i) {
    for (int a = 0; a < n_virt; ++a) {
    for (int r = 0; r < nbf; ++r) {
    for (int s = 0; s < nbf; ++s) {
        double val = 0.0;

        for (int q = 0; q < nbf; ++q) {
            val += C_virt(q, a) * I1(i, q, r, s);
        }
        I2(i, a, r, s) = val;
    }
    }
    }
    }
    I1.resize(0, 0, 0, 0); // Free memory

    Tensor4D I3(n_occ, n_virt, n_occ, nbf);
    I3.setZero();

    for (int i = 0; i < n_occ; ++i) {
    for (int a = 0; a < n_virt; ++a) {
    for (int j = 0; j < n_occ; ++j) {
    for (int s = 0; s < nbf; ++s) {
        double val = 0.0;

        for (int r = 0; r < nbf; ++r) {
            val += C_occ(r, j) * I2(i, a, r, s);
        }
        I3(i, a, j, s) = val;
    }
    }
    }
    }
    I2.resize(0, 0, 0, 0); // Free memory

    Tensor4D I_mo(n_occ, n_virt, n_occ, n_virt);
    I_mo.setZero();

    for (int i = 0; i < n_occ; ++i) {
    for (int a = 0; a < n_virt; ++a) {
    for (int j = 0; j < n_occ; ++j) {
    for (int b = 0; b < n_virt; ++b) {
        double val = 0.0;

        for (int s = 0; s < nbf; ++s) {
            val += C_virt(s, b) * I3(i, a, j, s);
        }
        I_mo(i, a, j, b) = val;
    }
    }
    }
    }

    return I_mo;
}


double calc_MP2_E(const Tensor4D& I_mo, int n_occ, const Vector& eps, double E0) {
    timing::Scoped _t("MP2 Energy Calculation");

    const int nbf = eps.size();
    const int n_virt = nbf - n_occ;

    Vector eps_occ  = eps.head(n_occ);
    Vector eps_virt = eps.tail(n_virt);

    double os_E = 0.0;
    double ss_E = 0.0;

    for (int i = 0; i < n_occ; ++i) {
    for (int a = 0; a < n_virt; ++a) {
    for (int j = 0; j < n_occ; ++j) {
    for (int b = 0; b < n_virt; ++b) {
        const double denominator = 1.0 / (eps_occ(i) - eps_virt(a) + eps_occ(j)  - eps_virt(b));

        const double direct = I_mo(i, a, j, b);
        const double exchange = I_mo(i, b, j, a);

        os_E += direct * direct * denominator;
        ss_E += direct * (direct - exchange) * denominator;
    }
    }
    }
    }

    return E0 + os_E + ss_E;
}
