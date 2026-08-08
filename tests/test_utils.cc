#include "doctest.h"
#include <Eigen/Dense>
#include <libint2.hpp>

#include "integrals.h"
#include "utils.h"

double compute_nuclear_repulsion(const std::vector<libint2::Atom>& atoms);

using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;

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

// ── compute_nuclear_repulsion ───────────────────────────────────────────────
// Analytical: for two atoms it's exactly Z1*Z2/R (R in bohr, as libint2
// stores atom coordinates after read_dotxyz converts from Angstrom).

TEST_CASE("compute_nuclear_repulsion: two hydrogen atoms, hand-computed") {
    std::vector<libint2::Atom> atoms(2);
    atoms[0] = {1, 0.0, 0.0, 0.0};
    atoms[1] = {1, 0.0, 0.0, 2.0}; // 2 bohr apart

    double vnn = compute_nuclear_repulsion(atoms);
    CHECK(vnn == doctest::Approx(0.5)); // 1*1/2.0
}

TEST_CASE("compute_nuclear_repulsion: three-atom sum of unique pairs") {
    std::vector<libint2::Atom> atoms(3);
    atoms[0] = {1, 0.0, 0.0, 0.0};
    atoms[1] = {1, 0.0, 0.0, 1.0};
    atoms[2] = {1, 0.0, 0.0, 3.0};

    // pairs: (0,1) R=1 -> 1.0; (0,2) R=3 -> 1/3; (1,2) R=2 -> 0.5
    double expected = 1.0 + (1.0 / 3.0) + 0.5;
    CHECK(compute_nuclear_repulsion(atoms) == doctest::Approx(expected));
}

// ── sym_ortho ────────────────────────────────────────────────────────────
// Property test: for any symmetric positive-definite S, X = sym_ortho(S)
// must satisfy X^T S X = I. No "expected matrix" needed — this is a
// mathematical fact about a Löwdin orthogonalizer, true regardless of the
// implementation details of how sym_ortho gets there.

TEST_CASE("sym_ortho: identity input returns identity") {
    Matrix S = Matrix::Identity(4, 4);
    Matrix X = sym_ortho(S);
    CHECK(X.isApprox(Matrix::Identity(4, 4), 1e-10));
}

TEST_CASE("sym_ortho: X^T S X == I for a synthetic SPD matrix") {
    Matrix A(3, 3);
    A << 2, 1, 0,
         1, 3, 1,
         0, 1, 4;
    Matrix S = A.transpose() * A + Matrix::Identity(3, 3); // guaranteed SPD

    Matrix X = sym_ortho(S);
    Matrix should_be_I = X.transpose() * S * X;
    CHECK(should_be_I.isApprox(Matrix::Identity(3, 3), 1e-8));
}

// ── diis ─────────────────────────────────────────────────────────────────
// Property tests: the Pulay constraint (coefficients sum to 1) holds
// regardless of the specific error-matrix history, and a single-entry
// history has the unique solution [1.0].

TEST_CASE("diis: single-entry history returns exactly [1.0]") {
    Matrix e(2, 2);
    e << 0.1, 0.2, 0.3, 0.4;
    Vector coeffs = diis({e});
    REQUIRE(coeffs.size() == 1);
    CHECK(coeffs(0) == doctest::Approx(1.0));
}

TEST_CASE("diis: coefficients always sum to 1 (Pulay constraint)") {
    Matrix e1(2, 2); e1 << 0.10, 0.02, -0.02, -0.10;
    Matrix e2(2, 2); e2 << 0.05, 0.01, -0.01, -0.05;
    Matrix e3(2, 2); e3 << 0.01, 0.00,  0.00, -0.01;

    Vector coeffs = diis({e1, e2, e3});
    REQUIRE(coeffs.size() == 3);
    CHECK(coeffs.sum() == doctest::Approx(1.0));
}
