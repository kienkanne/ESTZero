#include <fstream>
#include <stdexcept>
#include <libint2.hpp>
#include <stdexcept>

#include "wfn.h"
#include "libint2/basis.h"
#include "integrals.h"
#include "libint2/initialize.h"
#include "timer.h"
#include "rhf.h"
#include "mp2.h"
#include "utils.h"


void WaveFunction::loader(const std::string xyz, const std::string basis) {
    std::ifstream infile(xyz);
    if (!infile) {
        throw std::runtime_error("Could not open XYZ file: " + xyz);
    }

    auto atoms = libint2::read_dotxyz(infile);
    if (atoms.empty()) {
        throw std::runtime_error("XYZ file contains no atoms: " + xyz);
    }

    int nelectron = 0;
    for (const auto& atom : atoms) {
        nelectron += atom.atomic_number;
    }
    if (nelectron % 2 != 0) {
        throw std::runtime_error("RHF requires an even number of electrons");
    }
    
    this->obs = libint2::BasisSet(basis, atoms);
    this->atoms = atoms;
    this->ndocc = nelectron / 2;
}


void WaveFunction::get_integrals() {
    auto& obs = this->obs;
    auto& atoms = this->atoms;

    libint2::initialize();

    libint2::Engine overlap_engine(libint2::Operator::overlap, obs.max_nprim(), obs.max_l());
    libint2::Engine kinetic_engine(libint2::Operator::kinetic, obs.max_nprim(), obs.max_l());
    libint2::Engine nuclear_engine(libint2::Operator::nuclear, obs.max_nprim(), obs.max_l());

    nuclear_engine.set_params(make_point_charges(atoms));
    
    std::println("nbf: {}", obs.nbf());

    Eigen::MatrixXd S, T, V;
    {
        timing::Scoped _t("One-electron (S,T,V)");
        this->S = build_one_center(overlap_engine, obs);
        this->T = build_one_center(kinetic_engine, obs);
        this->V = build_one_center(nuclear_engine, obs);
    }

    libint2::Engine eri_engine(libint2::Operator::coulomb,
                                obs.max_nprim(), obs.max_l());
    std::vector<double> eri;
    {
        timing::Scoped _t("ERI build");
        this->eri_vector = build_canonical_eris(eri_engine, obs);
    }

    libint2::finalize();
}


double WaveFunction::get_energy(const std::string method, const double tol=1e-8, 
                                const int max_iter=200, const int verbose=2) {
    double E0;
    if (method == "rhf") {
        auto result = get_rhf_energy(tol, max_iter, verbose, compute_nuclear_repulsion(atoms),
                            this->ndocc, this->S, this->T, this->V, this->eri_vector);
        E0 = result.E0;
    }
    
    else if (method == "mp2") {
        auto result = get_rhf_energy(tol, max_iter, verbose, compute_nuclear_repulsion(atoms),
                            this->ndocc, this->S, this->T, this->V, this->eri_vector);
        auto I_mo = transform_MO(this->eri_vector, result.C, this->ndocc);
        E0 = calc_MP2_E(I_mo, this->ndocc, result.eps, result.E0);
    }

    else {
        throw std::runtime_error("Unknown method");
    }
    return E0;
}