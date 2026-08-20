#include "doctest.h"
#include <stdexcept>
#include <string>

#include "wfn.h"
#include "golden_io.h"

// End-to-end tests of the WaveFunction orchestration layer (the same path
// main.cc drives): loader -> get_integrals -> get_energy dispatch. This sits
// above the isolated leaf-function golden tests and checks the wiring between
// them.

// ── Full-pipeline energies vs golden + Psi4 oracle ──────────────────────────

TEST_CASE("WaveFunction: end-to-end scf/mp2 energies match golden and Psi4") {
    for (const auto& sys : golden::golden_systems()) {
        CAPTURE(sys.label);
        golden::Golden g = golden::load_golden(sys.name);

        WaveFunction wfn(std::string(EST_REPO_ROOT) + "/" + sys.xyz, sys.basis, sys.mult, sys.charge);
        wfn.get_integrals();

        double e_scf = wfn.get_energy(sys.reference, "scf", 1e-10, 200, 0);
        double e_mp2 = wfn.get_energy(sys.reference, "mp2", 1e-10, 200, 0);

        CHECK(e_scf == doctest::Approx(g.scalar("E_scf")).epsilon(1e-9));
        CHECK(e_scf == doctest::Approx(g.scalar("psi4_scf_energy")).epsilon(1e-6));
        CHECK(e_mp2 == doctest::Approx(g.scalar("E_mp2")).epsilon(1e-9));
        CHECK(e_mp2 == doctest::Approx(g.scalar("psi4_mp2_energy")).epsilon(1e-6));
    }
}

// ── Failure paths ────────────────────────────────────────────────────────────

TEST_CASE("WaveFunction: throws on a missing xyz file") {
    CHECK_THROWS_AS(
        WaveFunction(std::string(EST_REPO_ROOT) + "/input/does_not_exist.xyz", "sto-3g", 1, 0),
        std::runtime_error);
}

TEST_CASE("WaveFunction: throws on charge/multiplicity parity mismatch") {
    // h2o has 10 electrons; mult=2 implies 1 unpaired electron, and (10 - 1) is
    // odd -> invalid parity.
    CHECK_THROWS_AS(
        WaveFunction(std::string(EST_REPO_ROOT) + "/input/h2o.xyz", "sto-3g", 2, 0),
        std::runtime_error);
}

TEST_CASE("WaveFunction: get_energy throws when reference=rhf but multiplicity != 1") {
    WaveFunction wfn(std::string(EST_REPO_ROOT) + "/input/h2o.xyz", "sto-3g", 3, 0);
    wfn.get_integrals();
    CHECK_THROWS_AS(wfn.get_energy("rhf", "scf", 1e-10, 200, 0), std::runtime_error);
}

TEST_CASE("WaveFunction: get_energy throws on an unknown reference string") {
    WaveFunction wfn(std::string(EST_REPO_ROOT) + "/input/h2o.xyz", "sto-3g", 1, 0);
    wfn.get_integrals();
    CHECK_THROWS_AS(wfn.get_energy("xhf", "scf", 1e-10, 200, 0), std::runtime_error);
}

TEST_CASE("WaveFunction: get_energy throws on an unknown method string") {
    WaveFunction wfn(std::string(EST_REPO_ROOT) + "/input/h2o.xyz", "sto-3g", 1, 0);
    wfn.get_integrals();
    CHECK_THROWS_AS(wfn.get_energy("rhf", "unknown", 1e-10, 200, 0), std::runtime_error);
}
