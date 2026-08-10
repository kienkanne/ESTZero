#include "doctest.h"
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include "integrals.h"
#include "wfn.h"
#include "golden_io.h"


// ── Property tests ──────────────────────────────────────────────────────────
// Structural facts about the AO integrals that must hold for any valid
// molecule/basis, needing no reference values. Exercised on the smallest system.

TEST_CASE("integrals: overlap S is symmetric, positive-definite, unit-diagonal") {
    WaveFunction wfn(std::string(EST_REPO_ROOT) + "/input/h2o.xyz", "sto-3g", 1, 0);
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
    WaveFunction wfn(std::string(EST_REPO_ROOT) + "/input/h2o.xyz", "sto-3g", 1, 0);
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

        WaveFunction wfn(std::string(EST_REPO_ROOT) + "/" + sys.xyz, sys.basis, sys.mult, sys.charge);
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


// ── pair_index / eri_index ──────────────────────────────────────────────────
// Analytical tests: pure integer arithmetic, hand-checkable.

TEST_CASE("pair_index: exact hand-checked values") {
    CHECK(pair_index(0, 0) == 0);
    CHECK(pair_index(1, 0) == 1);
    CHECK(pair_index(0, 1) == 1); // symmetric in its arguments
    CHECK(pair_index(1, 1) == 2);
    CHECK(pair_index(2, 0) == 3);
    CHECK(pair_index(2, 1) == 4);
    CHECK(pair_index(2, 2) == 5);
}

TEST_CASE("pair_index: symmetric in its two arguments (property)") {
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j) {
            CHECK(pair_index(i, j) == pair_index(j, i));
        }
    }
}

TEST_CASE("pair_index: distinct pairs map to distinct indices (property)") {
    // A canonical index must be injective over the triangular domain i>=j,
    // otherwise eri_index would silently collide two different integrals.
    std::vector<std::pair<int,int>> seen_pairs;
    std::vector<int> seen_idx;
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j <= i; ++j) {
            int idx = pair_index(i, j);
            for (size_t k = 0; k < seen_idx.size(); ++k) {
                CHECK(seen_idx[k] != idx);
            }
            seen_idx.push_back(idx);
            seen_pairs.push_back({i, j});
        }
    }
}

TEST_CASE("eri_index: 8-fold permutational symmetry (property)") {
    // (pq|rs) = (qp|rs) = (pq|sr) = (qp|sr) = (rs|pq) = ... all 8 permutations
    // must land on the same canonical index. This is the invariant the whole
    // in-core ERI storage scheme depends on.
    for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
    for (int k = 0; k < 4; ++k) {
    for (int l = 0; l < 4; ++l) {
        int base = eri_index(i, j, k, l);
        CHECK(eri_index(j, i, k, l) == base);
        CHECK(eri_index(i, j, l, k) == base);
        CHECK(eri_index(j, i, l, k) == base);
        CHECK(eri_index(k, l, i, j) == base);
        CHECK(eri_index(l, k, i, j) == base);
        CHECK(eri_index(k, l, j, i) == base);
        CHECK(eri_index(l, k, j, i) == base);
    }
    }
    }
    }
}

TEST_CASE("eri_index: exact hand-checked value") {
    // eri_index(0,1,0,0): pair_index(0,1)=1, pair_index(0,0)=0,
    // pair_index(1,0)=1.
    CHECK(eri_index(0, 1, 0, 0) == 1);
    CHECK(eri_index(0, 0, 0, 0) == 0);
}