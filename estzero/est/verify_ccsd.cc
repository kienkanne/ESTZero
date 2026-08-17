#include <print>
#include <libint2.hpp>
#include "tensor.h"
#include "wfn.h"
#include "uhf.h"
#include "transmo.h"
#include "mp2.h"
#include "cc_spin.h"

// Standalone verification: build the spin-orbital ASprqs/Dia/Dijab pipeline for an open-shell
// UHF system and check that test_mp2_energy() reproduces the correlation energy as the already
// golden-tested calc_uhf_MP2_E, from the same Ca/Cb/epsa/epsb.
int main() {
    libint2::initialize();

    WaveFunction wfn("input/ch3.xyz", "6-31g", 2, 0);
    wfn.get_integrals("tensor");

    const Matrix& S = wfn.get_S();
    const Matrix& T = wfn.get_T();
    const Matrix& V = wfn.get_V();
    const Tensor& eri_tensor = wfn.get_eri_tensor();
    const double Vnn = wfn.get_Vnn();
    const int n_alpha = wfn.get_n_alpha();
    const int n_beta = wfn.get_n_beta();
    const int nbf = static_cast<int>(S.rows());

    SCFResult r = uhf_scf(wfn, 1e-10, 200, 0);

    // Oracle: golden tested MP2 path (iajb-restricted transform).
    const auto& [Iiajb_AA, Iiajb_BB, Iiajb_AB] = 
                transmo_spin_iajb(eri_tensor, r.CA, n_alpha, r.CB, n_beta);
    double E_oracle = calc_uhf_mp2_E(Iiajb_AA, Iiajb_BB, Iiajb_AB,
                                      r.epsA, r.epsB, n_alpha, n_beta);

    // New path: full-space transform -> spin-orbital Ismo -> test_mp2_energy.
    const auto& [Ipqrs_AA, Ipqrs_BB, Ipqrs_AB] = 
                transmo_spin_pqrs(eri_tensor, r.CA, n_alpha, r.CB, n_beta);
    Tensor ASprqs = build_spin_ASprqs(Ipqrs_AA, Ipqrs_BB, Ipqrs_AB, n_alpha, n_beta);

    // Ha/Hb must be in each spin's own MO basis.
    Tensor HA = mat_to_tensor(Matrix(r.CA.transpose() * (T + V) * r.CA));
    Tensor HB = mat_to_tensor(Matrix(r.CB.transpose() * (T + V) * r.CB));
    Tensor FN = build_spin_FN(HA, HB, ASprqs, n_alpha, n_beta);
    auto [Dia, Dijab] = build_spin_Dia_Dijab(FN, n_alpha, n_beta);
    
    double E_testmp2 = r.E_scf + test_mp2_energy(ASprqs, Dia, Dijab, n_alpha + n_beta);

    // Test using newly built ccsd energy equation
    const int n_occ = n_alpha + n_beta;
    const Sli O_sli{0, n_occ};
    const Sli V_sli{n_occ, ASprqs.shape[0]};
    Tensor t1_zero(Dia.shape);
    Tensor t2_mp2 = ASprqs.slice(O_sli, O_sli, V_sli, V_sli) / Dijab;

    double E_ccsd = r.E_scf + calc_ccsd_E(ASprqs, FN, t1_zero, t2_mp2, O_sli, V_sli);

    std::println("n_alpha={} n_beta={} nbf={}", n_alpha, n_beta, nbf);
    std::println("oracle (calc_uhf_MP2_E): {:.12f}", E_oracle);
    std::println("");
    std::println("[1] test_mp2_energy(): {:.12f}", E_testmp2);
    std::println("[2] calc_ccsd_E(): {:.12f}", E_ccsd);
    std::println("");
    std::println("[1] difference: {:.3e}", E_oracle - E_testmp2);
    std::println("[1] difference: {:.3e}", E_oracle - E_ccsd);

    libint2::finalize();
    return 0;
}
