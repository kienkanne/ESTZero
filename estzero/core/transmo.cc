#include <Eigen/Dense>
#include <stdexcept>
#include "transmo.h"
#include "einsum.h"
#include "timer.h"

namespace {
// Full four-quarter transform of the AO tensor into the MO basis. C1-C4 can be
// the full C, C_occ, or C_virt, depending on the caller's purpose. The build mode is 
// added for transforming Iab directly from Iaa intermediate 2.
Tensor transmo(const Tensor& start,
                const Tensor& C1, const Tensor& C2,
                const Tensor& C3, const Tensor& C4, 
                std::string build_mode="full") {

    if (build_mode == "full") {
        Tensor I1 = einsum("pi,pqrs->iqrs", C1, start);
        Tensor I2 = einsum("qa,iqrs->iars", C2, I1); I1.clear();
        Tensor I3 = einsum("rj,iars->iajs", C3, I2); I2.clear();
        return      einsum("sb,iajs->iajb", C4, I3);
    }
    else if (build_mode == "to_I2") {
        Tensor I1 = einsum("pi,pqrs->iqrs", C1, start);
        return      einsum("qa,iqrs->iars", C2, I1);
    }
    else if (build_mode == "from_I2") {
        Tensor I3 = einsum("rj,iars->iajs", C3, start);
        return      einsum("sb,iajs->iajb", C4, I3);
    }
    else { throw std::runtime_error("Unknown build mode"); }
}

} // namespace


Tensor transmo_spat_iajb(const Tensor& eri_tensor, const Matrix& C, int n_occ, int n_frozen) {
    timing::Scoped _t("MO Transformation");

    const int nbf = static_cast<int>(C.rows());
    const int n_virt = nbf - n_occ;

    // Frozen core orbitals are not transformed
    Tensor Cocc = mat_to_tensor(C.middleCols(n_frozen, n_occ - n_frozen));
    Tensor Cvirt = mat_to_tensor(C.rightCols(n_virt));

    return transmo(eri_tensor, Cocc, Cvirt, Cocc, Cvirt);
}


SpinMO transmo_spin_iajb(const Tensor& eri_tensor, const Matrix& CA, int n_alpha, const Matrix& CB, int n_beta,
                          int n_frozen) {
    timing::Scoped _t("MO Transformation");

    const int nbf = static_cast<int>(CA.rows());
    const int n_avirt = nbf - n_alpha;
    const int n_bvirt = nbf - n_beta;

    // Frozen core orbitals are not transformed
    Tensor Cocc_A = mat_to_tensor(CA.middleCols(n_frozen, n_alpha - n_frozen));
    Tensor Cvirt_A = mat_to_tensor(CA.rightCols(n_avirt));
    Tensor Cocc_B = mat_to_tensor(CB.middleCols(n_frozen, n_beta - n_frozen));
    Tensor Cvirt_B = mat_to_tensor(CB.rightCols(n_bvirt));

    Tensor Iiajb_BB = transmo(eri_tensor, Cocc_B, Cvirt_B, Cocc_B, Cvirt_B);

    Tensor Iiajb_AA2 = transmo(eri_tensor, Cocc_A, Cvirt_A, Cocc_A, Cvirt_A, "to_I2");
    Tensor Iiajb_AA = transmo(Iiajb_AA2, Cocc_A, Cvirt_A, Cocc_A, Cvirt_A, "from_I2");
    Tensor Iiajb_AB = transmo(Iiajb_AA2, Cocc_A, Cvirt_A, Cocc_B, Cvirt_B, "from_I2");

    return {Iiajb_AA, Iiajb_BB, Iiajb_AB};
}


Tensor transmo_spat_pqrs(const Tensor& eri_tensor, const Matrix& C) {
    timing::Scoped _t("MO Transformation (CC)");
    Tensor Ct = mat_to_tensor(C);

    return transmo(eri_tensor, Ct, Ct, Ct, Ct);
}


SpinMO transmo_spin_pqrs(const Tensor& eri_tensor, const Matrix& CA, int n_alpha, const Matrix& CB, int n_beta) {
    timing::Scoped _t("MO Transformation (CC)");

    const int nbf = static_cast<int>(CA.rows());
    const int n_avirt = nbf - n_alpha;
    const int n_bvirt = nbf - n_beta;

    Tensor CAt = mat_to_tensor(CA);
    Tensor CBt = mat_to_tensor(CB);

    Tensor Ipqrs_BB = transmo(eri_tensor, CBt, CBt, CBt, CBt);

    Tensor Ipqrs_AA2 = transmo(eri_tensor, CAt, CAt, CAt, CAt, "to_I2");
    Tensor Ipqrs_AA = transmo(Ipqrs_AA2, CAt, CAt, CAt, CAt, "from_I2");
    Tensor Ipqrs_AB = transmo(Ipqrs_AA2, CAt, CAt, CBt, CBt, "from_I2");

    return {Ipqrs_AA, Ipqrs_BB, Ipqrs_AB};
}
