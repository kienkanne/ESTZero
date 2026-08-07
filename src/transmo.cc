#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include "utils.h"
#include "transmo.h"
#include "timer.h"

using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;
using Tensor4D = Eigen::Tensor<double, 4>;


Tensor4D transform_index1(const std::vector<double>& eri_vector, const Matrix& C, int n_occ) {
    const int nbf   = static_cast<int>(C.rows());
    const int n_virt = nbf - n_occ;

    const Matrix& C_occ  = C.leftCols(n_occ);
    const Matrix& C_virt = C.rightCols(n_virt);

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
    return I1;
}


Tensor4D transform_index2(const Tensor4D& I1, const Matrix& C, int n_occ) {
    const int nbf   = static_cast<int>(C.rows());
    const int n_virt = nbf - n_occ;

    const Matrix& C_occ  = C.leftCols(n_occ);
    const Matrix& C_virt = C.rightCols(n_virt);

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
    return I2;
}


Tensor4D transform_index3(const Tensor4D& I2, const Matrix& C, int n_occ) {
    const Matrix& C_occ = C.leftCols(n_occ);

    const int nbf = static_cast<int>(C.rows());
    // For the alpha-beta cross term in UHF MP2, the first two dimensions of I2
    // cannot be inferred from n_occ alone, and must be read from I2 itself
    const int n_i = static_cast<int>(I2.dimension(0));
    const int n_a = static_cast<int>(I2.dimension(1));

    Tensor4D I3(n_i, n_a, n_occ, nbf);
    I3.setZero();

    for (int i = 0; i < n_i; ++i) {
    for (int a = 0; a < n_a; ++a) {
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
    return I3;
}


Tensor4D transform_index4(const Tensor4D& I3, const Matrix& C, int n_occ) {
    const int nbf   = static_cast<int>(C.rows());
    const int n_virt = nbf - n_occ;
    // Same reasoning as transform_index3
    const int n_i = static_cast<int>(I3.dimension(0));
    const int n_a = static_cast<int>(I3.dimension(1));
    const int n_j = static_cast<int>(I3.dimension(2));

    const Matrix& C_virt = C.rightCols(n_virt);

    Tensor4D I_mo(n_i, n_a, n_j, n_virt);
    I_mo.setZero();

    for (int i = 0; i < n_i; ++i) {
    for (int a = 0; a < n_a; ++a) {
    for (int j = 0; j < n_j; ++j) {
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


Tensor4D transform_rhf_MO(const std::vector<double>& eri_vector, const Matrix& C, int n_occ) {
    timing::Scoped _t("MO Transformation");

    const int nbf   = static_cast<int>(C.rows());
    const int n_virt = nbf - n_occ;

    Tensor4D I1 = transform_index1(eri_vector, C, n_occ);
    Tensor4D I2 = transform_index2(I1, C, n_occ);
    I1.resize(0, 0, 0, 0); // Free memory
    Tensor4D I3 = transform_index3(I2, C, n_occ);
    I2.resize(0, 0, 0, 0); // Free memory
    const Tensor4D& I_mo = transform_index4(I3, C, n_occ);

    return I_mo;
}


SpinMO transform_uhf_MO(const std::vector<double>& eri_vector, 
                          const Matrix& Ca, int n_alpha, const Matrix& Cb, int n_beta) {
    timing::Scoped _t("MO Transformation");

    const int nbf   = static_cast<int>(Ca.rows());
    const int n_avirt = nbf - n_alpha;
    const int n_bvirt = nbf - n_beta;

    Tensor4D I1aa = transform_index1(eri_vector, Ca, n_alpha);
    Tensor4D I1bb = transform_index1(eri_vector, Cb, n_beta);

    Tensor4D I2aa = transform_index2(I1aa, Ca, n_alpha);
    Tensor4D I2bb = transform_index2(I1bb, Cb, n_beta);
    I1aa.resize(0, 0, 0, 0); // Free memory
    I1bb.resize(0, 0, 0, 0); // Free memory

    Tensor4D I3aa = transform_index3(I2aa, Ca, n_alpha);
    Tensor4D I3bb = transform_index3(I2bb, Cb, n_beta);
    Tensor4D I3ab = transform_index3(I2aa, Cb, n_beta);
    I2aa.resize(0, 0, 0, 0); // Free memory
    I2bb.resize(0, 0, 0, 0); // Free memory

    const Tensor4D& Iaa_mo = transform_index4(I3aa, Ca, n_alpha);
    const Tensor4D& Ibb_mo = transform_index4(I3bb, Cb, n_beta);
    const Tensor4D& Iab_mo = transform_index4(I3ab, Cb, n_beta);

    return {Iaa_mo, Ibb_mo, Iab_mo};
}