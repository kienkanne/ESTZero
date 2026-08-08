#include "doctest.h"
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include "wfn.h"
#include "golden_io.h"

using golden::Matrix;

// ── Property tests ──────────────────────────────────────────────────────────
// Structural facts about the AO integrals that must hold for any valid
// molecule/basis, needing no reference values. Exercised on the smallest system.

TEST_CASE("integrals: overlap S is symmetric, positive-definite, unit-diagonal") {
    WaveFunction wfn(std::string(SCF_REPO_ROOT) + "/input/h2o.xyz", "sto-3g", 1, 0);
    wfn.get_integrals();
    const Matrix& S = wfn.get_S();

    CHECK(S.isApprox(S.transpose(), 1e-12));

    Eigen::SelfAdjointEigenSolver<Matrix> solver(S);
    for (int i = 0; i < solver.eigenvalues().size(); ++i)
        CHECK(solver.eigenvalues()(i) > 0.0);

    for (int i = 0; i < S.rows(); ++i)
        CHECK(S(i, i) == doctest::Approx(1.0).epsilon(1e-6)); // normalized basis
}

TEST_CASE("integrals: T symmetric PSD, V symmetric") {
    WaveFunction wfn(std::string(SCF_REPO_ROOT) + "/input/h2o.xyz", "sto-3g", 1, 0);
    wfn.get_integrals();

    CHECK(wfn.get_T().isApprox(wfn.get_T().transpose(), 1e-12));
    CHECK(wfn.get_V().isApprox(wfn.get_V().transpose(), 1e-12));

    Eigen::SelfAdjointEigenSolver<Matrix> solver(wfn.get_T());
    for (int i = 0; i < solver.eigenvalues().size(); ++i)
        CHECK(solver.eigenvalues()(i) > -1e-10);
}

// ── Oracle-golden regression ────────────────────────────────────────────────
// Rebuild S/T/V/ERI live via libint2 (build_one_center / build_canonical_eris,
// driven by WaveFunction::get_integrals) and compare to the frozen, oracle-
// certified values. Runs for every golden system.

TEST_CASE("integrals: S/T/V/ERI match golden") {
    for (const auto& sys : golden::golden_systems()) {
        CAPTURE(sys.label);
        golden::Golden g = golden::load_golden(sys.name);

        WaveFunction wfn(std::string(SCF_REPO_ROOT) + "/" + sys.xyz, sys.basis, sys.mult, sys.charge);
        wfn.get_integrals();

        CHECK(wfn.get_S().isApprox(g.mat("S"), 1e-12));
        CHECK(wfn.get_T().isApprox(g.mat("T"), 1e-12));
        CHECK(wfn.get_V().isApprox(g.mat("V"), 1e-12));

        const std::vector<double> eri = wfn.get_eri_vector();
        const std::vector<double> eri_golden = g.list("eri");
        REQUIRE(eri.size() == eri_golden.size());
        for (size_t i = 0; i < eri.size(); ++i)
            CHECK(eri[i] == doctest::Approx(eri_golden[i]).epsilon(1e-12));
    }
}
