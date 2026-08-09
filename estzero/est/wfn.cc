#include <fstream>
#include <stdexcept>
#include <libint2.hpp>
#include <stdexcept>
#include <format>

#include "wfn.h"
#include "libint2/basis.h"
#include "integrals.h"
#include "libint2/initialize.h"
#include "timer.h"
#include "rhf.h"
#include "uhf.h"
#include "mp2.h"
#include "transmo.h"
#include "utils.h"


void WaveFunction::loader(const std::string xyz, const std::string basis, 
                          const int mult, const int charge) {
    std::ifstream infile(xyz);
    if (!infile) {
        throw std::runtime_error("Could not open XYZ file: " + xyz);
    }

    auto atoms = libint2::read_dotxyz(infile);
    if (atoms.empty()) {
        throw std::runtime_error("XYZ file contains no atoms: " + xyz);
    }

    this->obs = libint2::BasisSet(basis, atoms);
    this->atoms = atoms;

    int n_electron = 0;
    for (const auto& atom : atoms) {
        n_electron += atom.atomic_number;
    }

    n_electron -= charge;
    int n_unpaired = mult - 1;
    if ((n_electron - n_unpaired) % 2 != 0) {
        throw std::runtime_error(std::format(
            "Invalid charge and multiplicity. Molecule has {} electrons and {} multiplicity", n_electron, mult));
    };

    this->n_docc = (n_electron - n_unpaired) / 2;
    this->n_alpha = n_docc + n_unpaired;
    this->n_beta = this->n_docc;
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



double WaveFunction::get_Vnn() {
    auto &atoms = this->atoms;
    double vnn = 0.0;
    
    for (size_t i = 0; i < atoms.size(); ++i) {
        for (size_t j = 0; j < i; ++j) {
            double dx = atoms[i].x - atoms[j].x;
            double dy = atoms[i].y - atoms[j].y;
            double dz = atoms[i].z - atoms[j].z;
            double r_ab = std::sqrt(dx * dx + dy * dy + dz * dz);
            
            // r_ab is in Bohr already
            vnn += (atoms[i].atomic_number * atoms[j].atomic_number) / r_ab;
        }
    }
    
    return vnn;
}

double WaveFunction::get_energy(const std::string reference, const std::string method, 
                                const double econv=1e-8, const int max_iter=200, const int verbose=2) {
    double E0;
    Result result;

    if (reference == "rhf") {
        if (this->n_docc != this-> n_alpha) {
            throw std::runtime_error("Cannot use RHF with multiplicity other than 1!");
        }

        result = rhf_scf(econv, max_iter, verbose, get_Vnn(), this->n_docc, 
                        this->S, this->T, this->V, this->eri_vector);
        E0 = result.E0;
    }

    else if (reference == "uhf") {
        result = uhf_scf(econv, max_iter, verbose, get_Vnn(), this->n_alpha, this->n_beta, 
                        this->S, this->T, this->V, this->eri_vector);
        E0 = result.E0;
    }

    else { throw std::runtime_error(std::format("Unknown reference: {}", reference)); }


    if (method == "scf") { return E0; }

    else if (method == "mp2") {
        if (reference == "rhf") {
            const auto& I_mo = transform_rhf_MO(this->eri_vector, result.Ca, this->n_docc);
            return calc_rhf_MP2_E(I_mo, result.epsa, this->n_docc, result.E0);
        }
        else if (reference == "uhf") {
            const auto& [Iaa_mo, Ibb_mo, Iab_mo] =
                transform_uhf_MO(this->eri_vector, result.Ca,
                        this->n_alpha, result.Cb, this->n_beta);
            return calc_uhf_MP2_E(Iaa_mo, Ibb_mo, Iab_mo, result.epsa, result.epsb,
                        this->n_alpha, this->n_beta, E0);
        }
    }

    else { throw std::runtime_error(std::format("Unknown method: {}", method)); }
}