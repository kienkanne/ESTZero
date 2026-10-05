#include "doctest.h"
#include <Eigen/Dense>

#include "integrals.h"
#include "transmo.h"
#include "golden_io.h"

namespace {

// shape()/strides() are std::span (no operator==); copy into a vector to compare and print in CHECKs
std::vector<Index> as_vector(std::span<const Index> s) { return {s.begin(), s.end()}; }

// doctest has no Tensor comparison; compare element-wise over the flat storage.
// ADDED the new contiguous() method. flat_index already got a warning for its uses cases,
// so contiguous is required here for using flat_index() to validate
void check_tensor_approx(const Tensor& a, const Tensor& b, double eps) {
    REQUIRE(as_vector(a.shape()) == as_vector(b.shape()));
    Tensor ac = a.contiguous();
    Tensor bc = b.contiguous();

    for (int i = 0; i < ac.size(); ++i)
        CHECK(ac.flat_index(i) == doctest::Approx(bc.flat_index(i)).epsilon(eps));
}

} // namespace

// ── Property tests: identity transform = reshape ────────────────────────────
// With C = Identity the AO->MO transform must be a pure reindex of the AO
// integrals. This directly targets the dimension-inheritance class of bug once
// found in transform_index3/transform_index4. It's an invariant, not a
// frozen-value comparison — the golden eri just supplies a valid input.

TEST_CASE("transmo_spat_iajb: identity C reproduces AO integrals unchanged") {
    golden::Golden g = golden::load_golden("h2o_sto-3g");
    const std::vector<double> eri = g.list("eri");
    const int nbf = g.nbf, n_occ = g.n_docc;
    const Tensor eri_tensor = eri_vector_to_tensor(eri, nbf);
    Matrix C = Matrix::Identity(nbf, nbf);

    Tensor I_mo = transmo_spat_iajb(eri_tensor, C, n_occ);
    for (int i = 0; i < n_occ; ++i)
    for (int a = 0; a < nbf - n_occ; ++a)
    for (int j = 0; j < n_occ; ++j)
    for (int b = 0; b < nbf - n_occ; ++b)
        CHECK(I_mo(i, a, j, b) == doctest::Approx(eri[eri_index(i, n_occ + a, j, n_occ + b)]));
}

TEST_CASE("transmo_spat_iajb: frozen core is exactly a sub-slice of the unfrozen occupied axes") {
    // Freezing core orbitals only removes them from the transform (columns of Cocc), and must
    // not otherwise change the values of the orbitals that remain.
    golden::Golden g = golden::load_golden("h2o_sto-3g");
    const Tensor eri_tensor = eri_vector_to_tensor(g.list("eri"), g.nbf);
    const Matrix& C = g.mat("C");
    const int n_occ = g.n_docc;
    const int n_frozen = 1; // O (period 2) contributes 1; the two H's contribute 0 each

    Tensor I_full = transmo_spat_iajb(eri_tensor, C, n_occ);
    Tensor I_frozen = transmo_spat_iajb(eri_tensor, C, n_occ, n_frozen);

    REQUIRE(I_frozen.shape()[0] == I_full.shape()[0] - n_frozen);
    REQUIRE(I_frozen.shape()[2] == I_full.shape()[2] - n_frozen);

    for (int i = 0; i < I_frozen.shape()[0]; ++i)
    for (int a = 0; a < I_frozen.shape()[1]; ++a)
    for (int j = 0; j < I_frozen.shape()[2]; ++j)
    for (int b = 0; b < I_frozen.shape()[3]; ++b)
        CHECK(I_frozen(i, a, j, b) == doctest::Approx(I_full(i + n_frozen, a, j + n_frozen, b)));
}

TEST_CASE("transmo_spin_iajb: identity C reproduces AO integrals (all three blocks)") {
    // ch3 is open-shell (n_alpha != n_beta), so the ab block mixes two different occ/virt splits
    // eri is in the canonical vector form, so an explicit nested loop is used to check
    golden::Golden g = golden::load_golden("ch3_sto-3g");
    const std::vector<double> eri = g.list("eri");
    const int nbf = g.nbf, n_alpha = g.n_alpha, n_beta = g.n_beta;
    const Tensor eri_tensor = eri_vector_to_tensor(eri, nbf);
    Matrix C = Matrix::Identity(nbf, nbf);

    SpinMO mo = transmo_spin_iajb(eri_tensor, C, n_alpha, C, n_beta);

    for (int i = 0; i < n_alpha; ++i)
    for (int a = 0; a < nbf - n_alpha; ++a)
    for (int j = 0; j < n_alpha; ++j)
    for (int b = 0; b < nbf - n_alpha; ++b)
        CHECK(mo.I_AA(i, a, j, b) == doctest::Approx(eri[eri_index(i, n_alpha + a, j, n_alpha + b)]));

    for (int i = 0; i < n_beta; ++i)
    for (int a = 0; a < nbf - n_beta; ++a)
    for (int j = 0; j < n_beta; ++j)
    for (int b = 0; b < nbf - n_beta; ++b)
        CHECK(mo.I_BB(i, a, j, b) == doctest::Approx(eri[eri_index(i, n_beta + a, j, n_beta + b)]));

    // ab block: i,a on the alpha split, j,b on the (different) beta split.
    for (int i = 0; i < n_alpha; ++i)
    for (int a = 0; a < nbf - n_alpha; ++a)
    for (int j = 0; j < n_beta; ++j)
    for (int b = 0; b < nbf - n_beta; ++b)
        CHECK(mo.I_AB(i, a, j, b) == doctest::Approx(eri[eri_index(i, n_alpha + a, j, n_beta + b)]));
}

// ── Property: UHF transform reduces to RHF when the spins are equivalent ─────

TEST_CASE("transmo_spin_iajb: reduces to transmo_spat_iajb when Ca=Cb, n_alpha=n_beta") {
    // h2o is closed-shell, so its converged orbitals give a valid Ca=Cb case.
    golden::Golden g = golden::load_golden("h2o_sto-3g");
    const std::vector<double> eri = g.list("eri");
    const Matrix C = g.mat("C");
    const int n_occ = g.n_docc;
    const Tensor eri_tensor = eri_vector_to_tensor(eri, g.nbf);

    Tensor I_mo = transmo_spat_iajb(eri_tensor, C, n_occ);
    SpinMO mo = transmo_spin_iajb(eri_tensor, C, n_occ, C, n_occ);

    check_tensor_approx(mo.I_AA, I_mo, 1e-10);
    check_tensor_approx(mo.I_BB, I_mo, 1e-10);
    check_tensor_approx(mo.I_AB, I_mo, 1e-10);
}

// ── Oracle-golden regression ────────────────────────────────────────────────
// Feed the frozen (eri, C) and require the frozen MO tensor(s). Feeding the
// FROZEN orbitals (not freshly converged ones) makes this deterministic despite
// orbital sign/phase freedom.

TEST_CASE("transform: reproduces golden MO integrals from golden orbitals") {
    for (const auto& sys : golden::golden_systems()) {
        CAPTURE(sys.label);
        golden::Golden g = golden::load_golden(sys.name);
        const std::vector<double> eri = g.list("eri");
        const Tensor eri_tensor = eri_vector_to_tensor(eri, g.nbf);

        if (g.reference == "rhf") {
            Tensor I_mo = transmo_spat_iajb(eri_tensor, g.mat("C"), g.n_docc);
            check_tensor_approx(I_mo, g.tensor("I_mo"), 1e-12);
        } else {
            SpinMO mo = transmo_spin_iajb(eri_tensor, g.mat("Ca"), g.n_alpha, g.mat("Cb"), g.n_beta);
            check_tensor_approx(mo.I_AA, g.tensor("Iaa"), 1e-12);
            check_tensor_approx(mo.I_BB, g.tensor("Ibb"), 1e-12);
            check_tensor_approx(mo.I_AB, g.tensor("Iab"), 1e-12);
        }
    }
}
