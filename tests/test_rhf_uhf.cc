#include "doctest.h"
#include <Eigen/Dense>
#include <stdexcept>

#include "rhf.h"
#include "uhf.h"
#include "golden_io.h"


// ── Property tests ──────────────────────────────────────────────────────────

TEST_CASE("uhf_scf: reproduces rhf_scf exactly on a closed-shell singlet") {
    golden::Golden g = golden::load_golden("h2o_sto-3g");
    const std::vector<double> eri = g.list("eri");
    const double v_nn = g.scalar("v_nn");

    Result rhf = rhf_scf(1e-10, 200, 0, v_nn, g.n_docc,
                          g.mat("S"), g.mat("T"), g.mat("V"), eri);
    Result uhf = uhf_scf(1e-10, 200, 0, v_nn, g.n_alpha, g.n_beta,
                         g.mat("S"), g.mat("T"), g.mat("V"), eri);
    CHECK(uhf.E0 == doctest::Approx(rhf.E0).epsilon(1e-9));
}

TEST_CASE("rhf_scf: throws on non-convergence") {
    golden::Golden g = golden::load_golden("h2o_sto-3g");
    CHECK_THROWS_AS(
        rhf_scf(1e-14, 1, 0, 0.0, g.n_docc, g.mat("S"), g.mat("T"), g.mat("V"), g.list("eri")),
        std::runtime_error);
}

TEST_CASE("uhf_scf: throws on non-convergence") {
    golden::Golden g = golden::load_golden("ch3_sto-3g");
    CHECK_THROWS_AS(
        uhf_scf(1e-14, 1, 0, 0.0, g.n_alpha, g.n_beta, g.mat("S"), g.mat("T"), g.mat("V"), g.list("eri")),
        std::runtime_error);
}

// ── Oracle-golden regression ────────────────────────────────────────────────
// Feed frozen S/T/V/ERI + the frozen nuclear repulsion, converge, and require
// the frozen total energy (tight) which is itself certified against Psi4
// (loose). Also check the gauge-invariant converged quantities (orbital
// energies and density); orbital coefficients C are NOT compared directly
// because of their arbitrary sign/phase.

namespace {
// Compute density from occupied orbitals — gauge-invariant (unlike C itself), so
// safe to compare against golden even though orbital signs/phases are arbitrary.
Matrix density(const Matrix& C, int n_occ) {
    Matrix occ = C.leftCols(n_occ);
    return occ * occ.transpose();
}
} // namespace

TEST_CASE("scf: converged energy, eps and density match golden (and the Psi4 oracle)") {
    for (const auto& sys : golden::golden_systems()) {
        CAPTURE(sys.label);
        golden::Golden g = golden::load_golden(sys.name);
        const std::vector<double> eri = g.list("eri");
        const double v_nn = g.scalar("v_nn");

        if (g.reference == "rhf") {
            Result r = rhf_scf(1e-10, 200, 0, v_nn, g.n_docc, g.mat("S"), g.mat("T"), g.mat("V"), eri);
            CHECK(r.E0 == doctest::Approx(g.scalar("E_scf")).epsilon(1e-9));           // regression
            CHECK(r.E0 == doctest::Approx(g.scalar("psi4_scf_energy")).epsilon(1e-6)); // oracle
            CHECK(r.epsA.isApprox(g.vec("eps"), 1e-8));
            CHECK(density(r.CA, g.n_docc).isApprox(g.mat("D"), 1e-8));
        } else {
            Result r = uhf_scf(1e-10, 200, 0, v_nn, g.n_alpha, g.n_beta,
                               g.mat("S"), g.mat("T"), g.mat("V"), eri);
            CHECK(r.E0 == doctest::Approx(g.scalar("E_scf")).epsilon(1e-9));           // regression
            CHECK(r.E0 == doctest::Approx(g.scalar("psi4_scf_energy")).epsilon(1e-6)); // oracle
            CHECK(r.epsA.isApprox(g.vec("epsa"), 1e-8));
            CHECK(r.epsB.isApprox(g.vec("epsb"), 1e-8));
            CHECK(density(r.CA, g.n_alpha).isApprox(g.mat("Da"), 1e-8));
            CHECK(density(r.CB, g.n_beta).isApprox(g.mat("Db"), 1e-8));
        }
    }
}
