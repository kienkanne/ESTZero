#include <Eigen/Dense>
#include "integrals.h"
#include "transmo.h"
#include "einsum.h"
#include "timer.h"

using Matrix = Eigen::MatrixXd;


// transmo now used the newly built einsum implementation. The data type is now the plan Tensor,
// which is not bad for the algorithm itself, but conversion of other data types to it makes it
// very inefficient and inconvenient. The einsum implementation is still a work in progress, and will be optimized.

namespace {

// Dense row-major AO ERI tensor (nbf^4) unpacked from the 8-fold canonical
// vector, so the generic contractor can index it directly. TODO: ERI should
// just be a tensor as well.
Tensor build_ao_tensor(const std::vector<double>& eri_vector, int nbf) {
    std::vector<double> data(static_cast<size_t>(nbf) * nbf * nbf * nbf);
    size_t idx = 0;
    for (int p = 0; p < nbf; ++p) {
    for (int q = 0; q < nbf; ++q) {
    for (int r = 0; r < nbf; ++r) {
    for (int s = 0; s < nbf; ++s) {
        data[idx++] = eri_vector[eri_index(p, q, r, s)];
    }}}}
    return Tensor({nbf, nbf, nbf, nbf}, data);
}


// Full four-quarter transform of the AO tensor into the MO basis (ia|jb),
// chemist notation. Index positions p,q,r,s -> i,a,j,b use:
// spin-1 occ, spin-1 virt, spin-2 occ, spin-2 virt.
Tensor transform_eri(const Tensor& ao,
                     const Tensor& Cocc1, const Tensor& Cvirt1,
                     const Tensor& Cocc2, const Tensor& Cvirt2) {
    Tensor t1 = einsum("pi,pqrs->iqrs", {Cocc1,  ao});
    Tensor t2 = einsum("qa,iqrs->iars", {Cvirt1, t1});
    Tensor t3 = einsum("rj,iars->iajs", {Cocc2,  t2});
    return      einsum("sb,iajs->iajb", {Cvirt2, t3});
}

} // namespace


Tensor transform_rhf_MO(const std::vector<double>& eri_vector, const Matrix& C, int n_occ) {
    timing::Scoped _t("MO Transformation");

    const int nbf    = static_cast<int>(C.rows());
    const int n_virt = nbf - n_occ;

    Tensor ao    = build_ao_tensor(eri_vector, nbf);
    Tensor Cocc  = mat_to_tensor(C.leftCols(n_occ));
    Tensor Cvirt = mat_to_tensor(C.rightCols(n_virt));

    return transform_eri(ao, Cocc, Cvirt, Cocc, Cvirt);
}


SpinMO transform_uhf_MO(const std::vector<double>& eri_vector,
                          const Matrix& Ca, int n_alpha, const Matrix& Cb, int n_beta) {
    timing::Scoped _t("MO Transformation");

    const int nbf     = static_cast<int>(Ca.rows());
    const int n_avirt = nbf - n_alpha;
    const int n_bvirt = nbf - n_beta;

    Tensor ao      = build_ao_tensor(eri_vector, nbf);
    Tensor Cocc_a  = mat_to_tensor(Ca.leftCols(n_alpha));
    Tensor Cvirt_a = mat_to_tensor(Ca.rightCols(n_avirt));
    Tensor Cocc_b  = mat_to_tensor(Cb.leftCols(n_beta));
    Tensor Cvirt_b = mat_to_tensor(Cb.rightCols(n_bvirt));

    Tensor Iaa = transform_eri(ao, Cocc_a, Cvirt_a, Cocc_a, Cvirt_a);
    Tensor Ibb = transform_eri(ao, Cocc_b, Cvirt_b, Cocc_b, Cvirt_b);
    Tensor Iab = transform_eri(ao, Cocc_a, Cvirt_a, Cocc_b, Cvirt_b);

    return {Iaa, Ibb, Iab};
}



// LEGACY: hand-written quarter-transform loops (the pre-einsum implementation).
// Kept for reference and for A/B benchmarking against the einsum path. Not wired
// into the live pipeline. Ported to the plain Tensor container (used to be Eigen::Tensor)


Tensor transform_index1_old(const std::vector<double>& eri_vector, const Matrix& C, int n_occ) {
    const int nbf = static_cast<int>(C.rows());
    const Matrix& C_occ = C.leftCols(n_occ);

    Tensor I1({n_occ, nbf, nbf, nbf});
    for (int i = 0; i < n_occ; ++i) {
    for (int q = 0; q < nbf; ++q) {
    for (int r = 0; r < nbf; ++r) {
    for (int s = 0; s < nbf; ++s) {
        double val = 0.0;
        for (int p = 0; p < nbf; ++p) val += C_occ(p, i) * eri_vector[eri_index(p, q, r, s)];
        I1(i, q, r, s) = val;
    }}}}
    return I1;
}

Tensor transform_index2_old(const Tensor& I1, const Matrix& C, int n_occ) {
    const int nbf    = static_cast<int>(C.rows());
    const int n_virt = nbf - n_occ;
    const Matrix& C_virt = C.rightCols(n_virt);

    Tensor I2({n_occ, n_virt, nbf, nbf});
    for (int i = 0; i < n_occ; ++i) {
    for (int a = 0; a < n_virt; ++a) {
    for (int r = 0; r < nbf; ++r) {
    for (int s = 0; s < nbf; ++s) {
        double val = 0.0;
        for (int q = 0; q < nbf; ++q) val += C_virt(q, a) * I1(i, q, r, s);
        I2(i, a, r, s) = val;
    }}}}
    return I2;
}

Tensor transform_index3_old(const Tensor& I2, const Matrix& C, int n_occ) {
    const int nbf = static_cast<int>(C.rows());
    const Matrix& C_occ = C.leftCols(n_occ);
    // I2's (i, a) dimensions come from whatever built it (a different spin's
    // occ/virt count for the UHF ab block), so read them from I2 itself.
    const int n_i = I2.shape[0];
    const int n_a = I2.shape[1];

    Tensor I3({n_i, n_a, n_occ, nbf});
    for (int i = 0; i < n_i; ++i) {
    for (int a = 0; a < n_a; ++a) {
    for (int j = 0; j < n_occ; ++j) {
    for (int s = 0; s < nbf; ++s) {
        double val = 0.0;
        for (int r = 0; r < nbf; ++r) val += C_occ(r, j) * I2(i, a, r, s);
        I3(i, a, j, s) = val;
    }}}}
    return I3;
}

Tensor transform_index4_old(const Tensor& I3, const Matrix& C, int n_occ) {
    const int nbf    = static_cast<int>(C.rows());
    const int n_virt = nbf - n_occ;
    const Matrix& C_virt = C.rightCols(n_virt);
    const int n_i = I3.shape[0];
    const int n_a = I3.shape[1];
    const int n_j = I3.shape[2];

    Tensor I_mo({n_i, n_a, n_j, n_virt});
    for (int i = 0; i < n_i; ++i) {
    for (int a = 0; a < n_a; ++a) {
    for (int j = 0; j < n_j; ++j) {
    for (int b = 0; b < n_virt; ++b) {
        double val = 0.0;
        for (int s = 0; s < nbf; ++s) val += C_virt(s, b) * I3(i, a, j, s);
        I_mo(i, a, j, b) = val;
    }}}}
    return I_mo;
}

Tensor transform_rhf_MO_old(const std::vector<double>& eri_vector, const Matrix& C, int n_occ) {
    timing::Scoped _t("MO Transformation");
    Tensor I1 = transform_index1_old(eri_vector, C, n_occ);
    Tensor I2 = transform_index2_old(I1, C, n_occ);
    Tensor I3 = transform_index3_old(I2, C, n_occ);
    return transform_index4_old(I3, C, n_occ);
}

SpinMO transform_uhf_MO_old(const std::vector<double>& eri_vector,
                            const Matrix& Ca, int n_alpha, const Matrix& Cb, int n_beta) {
    timing::Scoped _t("MO Transformation");
    Tensor I2aa = transform_index2_old(transform_index1_old(eri_vector, Ca, n_alpha), Ca, n_alpha);
    Tensor I2bb = transform_index2_old(transform_index1_old(eri_vector, Cb, n_beta),  Cb, n_beta);

    Tensor Iaa = transform_index4_old(transform_index3_old(I2aa, Ca, n_alpha), Ca, n_alpha);
    Tensor Ibb = transform_index4_old(transform_index3_old(I2bb, Cb, n_beta),  Cb, n_beta);
    Tensor Iab = transform_index4_old(transform_index3_old(I2aa, Cb, n_beta),  Cb, n_beta);

    return {Iaa, Ibb, Iab};
}
