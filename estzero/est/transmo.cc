#include <Eigen/Dense>
#include <stdexcept>
#include "integrals.h"
#include "transmo.h"
#include "einsum.h"
#include "timer.h"


// transmo now used the newly built einsum implementation. The data type is now the plan Tensor,
// which is not bad for the algorithm itself, but conversion of other data types to it makes it
// very inefficient and inconvenient. The einsum implementation is still a work in progress, and will be optimized.

namespace {

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
// Use build mode to build Iab from Iaa intermediate 2
Tensor transform_eri(const Tensor& start,
                     const Tensor& Cocc1, const Tensor& Cvirt1,
                     const Tensor& Cocc2, const Tensor& Cvirt2, 
                     std::string build_mode="full") {

    if (build_mode == "full") {
        Tensor I1 = einsum("pi,pqrs->iqrs", {Cocc1,  start});
        Tensor I2 = einsum("qa,iqrs->iars", {Cvirt1, I1}); I1.clear();
        Tensor I3 = einsum("rj,iars->iajs", {Cocc2,  I2}); I2.clear();
        return      einsum("sb,iajs->iajb", {Cvirt2, I3});
    }
    else if (build_mode == "to_I2") {
        Tensor I1 = einsum("pi,pqrs->iqrs", {Cocc1,  start});
        return      einsum("qa,iqrs->iars", {Cvirt1, I1});
    }
    else if (build_mode == "from_I2") {
        Tensor I3 = einsum("rj,iars->iajs", {Cocc2,  start});
        return      einsum("sb,iajs->iajb", {Cvirt2, I3});
    }
    else { throw std::runtime_error("Unknown build mode"); }
}

} // namespace


Tensor transform_rhf_MO(const std::vector<double>& eri_vector, const Matrix& C, int n_occ) {
    timing::Scoped _t("MO Transformation");

    const int nbf    = static_cast<int>(C.rows());
    const int n_virt = nbf - n_occ;

    Tensor eri_tensor = build_ao_tensor(eri_vector, nbf);
    Tensor Cocc = mat_to_tensor(C.leftCols(n_occ));
    Tensor Cvirt = mat_to_tensor(C.rightCols(n_virt));

    return transform_eri(eri_tensor, Cocc, Cvirt, Cocc, Cvirt);
}


SpinMO transform_uhf_MO(const std::vector<double>& eri_vector,
                          const Matrix& Ca, int n_alpha, const Matrix& Cb, int n_beta) {
    timing::Scoped _t("MO Transformation");

    const int nbf     = static_cast<int>(Ca.rows());
    const int n_avirt = nbf - n_alpha;
    const int n_bvirt = nbf - n_beta;

    Tensor eri_tensor = build_ao_tensor(eri_vector, nbf);
    Tensor Cocc_a = mat_to_tensor(Ca.leftCols(n_alpha));
    Tensor Cvirt_a = mat_to_tensor(Ca.rightCols(n_avirt));
    Tensor Cocc_b = mat_to_tensor(Cb.leftCols(n_beta));
    Tensor Cvirt_b = mat_to_tensor(Cb.rightCols(n_bvirt));

    Tensor Ibb = transform_eri(eri_tensor, Cocc_b, Cvirt_b, Cocc_b, Cvirt_b);

    Tensor I2aa = transform_eri(eri_tensor, Cocc_a, Cvirt_a, Cocc_a, Cvirt_a, "to_I2");
    Tensor Iaa = transform_eri(I2aa, Cocc_a, Cvirt_a, Cocc_a, Cvirt_a, "from_I2");
    Tensor Iab = transform_eri(I2aa, Cocc_a, Cvirt_a, Cocc_b, Cvirt_b, "from_I2");

    return {Iaa, Ibb, Iab};
}


// Tensor-scheme overloads: `eri` is already the dense AO tensor built directly
// by build_eri_tensor(), so the vector->tensor unpack (build_ao_tensor) is
// skipped entirely and it is fed straight into transform_eri.

Tensor transform_rhf_MO(const Tensor& eri_tensor, const Matrix& C, int n_occ) {
    timing::Scoped _t("MO Transformation");

    const int nbf = static_cast<int>(C.rows());
    const int n_virt = nbf - n_occ;

    Tensor Cocc = mat_to_tensor(C.leftCols(n_occ));
    Tensor Cvirt = mat_to_tensor(C.rightCols(n_virt));

    return transform_eri(eri_tensor, Cocc, Cvirt, Cocc, Cvirt);
}


SpinMO transform_uhf_MO(const Tensor& eri_tensor,
                          const Matrix& Ca, int n_alpha, const Matrix& Cb, int n_beta) {
    timing::Scoped _t("MO Transformation");

    const int nbf = static_cast<int>(Ca.rows());
    const int n_avirt = nbf - n_alpha;
    const int n_bvirt = nbf - n_beta;

    Tensor Cocc_a = mat_to_tensor(Ca.leftCols(n_alpha));
    Tensor Cvirt_a = mat_to_tensor(Ca.rightCols(n_avirt));
    Tensor Cocc_b = mat_to_tensor(Cb.leftCols(n_beta));
    Tensor Cvirt_b = mat_to_tensor(Cb.rightCols(n_bvirt));

    Tensor Ibb = transform_eri(eri_tensor, Cocc_b, Cvirt_b, Cocc_b, Cvirt_b);

    Tensor I2aa = transform_eri(eri_tensor, Cocc_a, Cvirt_a, Cocc_a, Cvirt_a, "to_I2");
    Tensor Iaa = transform_eri(I2aa, Cocc_a, Cvirt_a, Cocc_a, Cvirt_a, "from_I2");
    Tensor Iab = transform_eri(I2aa, Cocc_a, Cvirt_a, Cocc_b, Cvirt_b, "from_I2");

    return {Iaa, Ibb, Iab};
}
