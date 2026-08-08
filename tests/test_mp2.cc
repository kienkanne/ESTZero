#include "doctest.h"
#include <Eigen/Dense>

#include "mp2.h"
#include "transmo.h"
#include "golden_io.h"

using golden::Matrix;
using Vector = Eigen::VectorXd;

// ── Property: UHF MP2 reduces to RHF MP2 on a closed shell ──────────────────
// With Ca=Cb and n_alpha=n_beta the two spin channels are identical, so
// calc_uhf_MP2_E must equal calc_rhf_MP2_E on the same data. This is the exact
// invariant that would have caught the "n_beta passed for both spins" bug. Uses
// the frozen h2o integrals/orbitals only as realistic inputs — the assertion
// itself is a code-path equality, not a golden-value comparison.

TEST_CASE("calc_uhf_MP2_E: reduces to calc_rhf_MP2_E for a real closed-shell case") {
    golden::Golden g = golden::load_golden("h2o_sto-3g");
    const std::vector<double> eri = g.list("eri");
    const Matrix C = g.mat("C");
    const Vector eps = g.vec("eps");
    const int n_occ = g.n_docc;
    const double E0 = g.scalar("E_scf");

    Tensor I_mo = transform_rhf_MO(eri, C, n_occ);
    SpinMO mo = transform_uhf_MO(eri, C, n_occ, C, n_occ);

    double rhf = calc_rhf_MP2_E(I_mo, eps, n_occ, E0);
    double uhf = calc_uhf_MP2_E(mo.Iaa_mo, mo.Ibb_mo, mo.Iab_mo, eps, eps, n_occ, n_occ, E0);

    CHECK(uhf == doctest::Approx(rhf).epsilon(1e-12));
}

// ── Oracle-golden regression ────────────────────────────────────────────────
// Feed the frozen MO integrals + orbital energies and require the frozen MP2
// energy (tight), which is itself certified against Psi4 (loose). Each function
// tested in isolation on frozen inputs.

TEST_CASE("mp2: reproduces golden correlation energy, matching the Psi4 oracle") {
    for (const auto& sys : golden::golden_systems()) {
        CAPTURE(sys.label);
        golden::Golden g = golden::load_golden(sys.name);
        const double E_scf = g.scalar("E_scf");

        double e_mp2;
        if (g.reference == "rhf") {
            e_mp2 = calc_rhf_MP2_E(g.tensor("I_mo"), g.vec("eps"), g.n_docc, E_scf);
        } else {
            e_mp2 = calc_uhf_MP2_E(g.tensor("Iaa"), g.tensor("Ibb"), g.tensor("Iab"),
                                   g.vec("epsa"), g.vec("epsb"), g.n_alpha, g.n_beta, E_scf);
        }

        CHECK(e_mp2 == doctest::Approx(g.scalar("E_mp2")).epsilon(1e-12));       // regression
        CHECK(e_mp2 == doctest::Approx(g.scalar("psi4_mp2_energy")).epsilon(1e-6)); // oracle
    }
}
