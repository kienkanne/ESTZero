#include "doctest.h"
#include <Eigen/Dense>

#include "integrals.h"
#include "transmo.h"
#include "golden_io.h"

namespace {

// doctest has no Tensor comparison; compare element-wise over the flat storage.
void check_tensor_approx(const Tensor& a, const Tensor& b, double eps) {
    REQUIRE(a.shape == b.shape);
    int count = 1;
    for (int d : a.shape) count *= d;
    for (int i = 0; i < count; ++i)
        CHECK(a.flat_index(i) == doctest::Approx(b.flat_index(i)).epsilon(eps));
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
