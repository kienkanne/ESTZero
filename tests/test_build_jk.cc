#include "doctest.h"
#include <Eigen/Dense>

#include "jk_build.h"
#include "golden_io.h"


// ── Property test ───────────────────────────────────────────────────────────
// J and K must be symmetric whenever D is symmetric. This is a mathematical
// invariant, not a comparison against frozen values — the golden (eri, D) just
// supply a valid symmetric-density input to assert it on.

TEST_CASE("build_JK: J and K are symmetric for symmetric D") {
    golden::Golden g = golden::load_golden("h2o_sto-3g");

    JK jk = build_JK(g.list("eri"), g.mat("D"));
    CHECK(jk.J.isApprox(jk.J.transpose(), 1e-12));
    CHECK(jk.K.isApprox(jk.K.transpose(), 1e-12));
}

// ── Oracle-golden regression ────────────────────────────────────────────────
// build_JK fed the frozen (eri, density) must reproduce the frozen (J, K). RHF
// systems freeze one density D; UHF systems freeze Da/Db and their J/K blocks.

TEST_CASE("build_JK: reproduces golden J/K from golden density") {
    for (const auto& sys : golden::golden_systems()) {
        CAPTURE(sys.label);
        golden::Golden g = golden::load_golden(sys.name);
        const std::vector<double> eri = g.list("eri");

        if (g.reference == "rhf") {
            JK jk = build_JK(eri, g.mat("D"));
            CHECK(jk.J.isApprox(g.mat("J"), 1e-10));
            CHECK(jk.K.isApprox(g.mat("K"), 1e-10));
        } else {
            JK jka = build_JK(eri, g.mat("Da"));
            JK jkb = build_JK(eri, g.mat("Db"));
            CHECK(jka.J.isApprox(g.mat("Ja"), 1e-10));
            CHECK(jka.K.isApprox(g.mat("Ka"), 1e-10));
            CHECK(jkb.J.isApprox(g.mat("Jb"), 1e-10));
            CHECK(jkb.K.isApprox(g.mat("Kb"), 1e-10));
        }
    }
}
