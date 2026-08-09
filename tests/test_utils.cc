#include "doctest.h"
#include <Eigen/Dense>
#include <libint2.hpp>

#include "utils.h"

using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;


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
