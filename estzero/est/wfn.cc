#include <fstream>
#include <stdexcept>
#include <libint2.hpp>
#include <stdexcept>
#include <format>
#include <string>

#include "wfn.h"
#include "cc_iter.h"
#include "libint2/basis.h"
#include "integrals.h"
#include "timer.h"
#include "rhf.h"
#include "uhf.h"
#include "tensor.h"
#include "mp2.h"
#include "transmo.h"
#include "cc_driver.h"


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
    this->nbf = static_cast<int>(obs.nbf());

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


ERIScheme parse_eri_scheme(const std::string& s) {
    if (s == "vector") return ERIScheme::Vector;
    if (s == "tensor") return ERIScheme::Tensor;
    throw std::runtime_error("Unknown eri-scheme: " + s);
}


void WaveFunction::get_integrals(std::string eri_scheme_str) {
    this->eri_scheme_ = parse_eri_scheme(eri_scheme_str);

    if (!libint2::initialized()) { libint2::initialize(); }

    auto& obs = this->obs;
    auto& atoms = this->atoms;

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
    
    if (this->eri_scheme_ == ERIScheme::Tensor) {
        timing::Scoped _t("ERI build");
        this->eri_tensor = build_eri_tensor(eri_engine, obs);
    } else {
        timing::Scoped _t("ERI build");
        this->eri_vector = build_eri_vector(eri_engine, obs);
    }
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
    double E_scf;
    SCFResult sr;

    if (reference == "rhf") {
        if (this->n_docc != this-> n_alpha) { throw std::runtime_error("Cannot use RHF with multiplicity other than 1!"); }

        sr = rhf_scf(*this, econv, max_iter, verbose);
        E_scf = sr.E_scf;
    }

    else if (reference == "uhf") {
        sr = sr = uhf_scf(*this, econv, max_iter, verbose);
        E_scf = sr.E_scf;
    }

    else { throw std::runtime_error(std::format("Unknown reference: {}", reference)); }


    if (method == "scf") { return E_scf; }
    // Post SCF, the eri_tensor is neeeded
    if (this->eri_scheme_ == ERIScheme::Vector) {
        this->eri_tensor = eri_vector_to_tensor(eri_vector, this->nbf);
    }

    else if (method == "mp2") {
        if (reference == "rhf") {
            const Tensor Iiajb = transmo_spat_iajb(this->eri_tensor, sr.CA, this->n_docc);
            return E_scf + calc_rhf_mp2_E(Iiajb, sr.epsA, this->n_docc);
        }
        else if (reference == "uhf") {
            const auto& [Iiajb_AA, Iiajb_BB, Iiajb_AB] = transmo_spin_iajb(
                this->eri_tensor, sr.CA, this->n_alpha, sr.CB, this->n_beta);
            return E_scf + calc_uhf_mp2_E(Iiajb_AA, Iiajb_BB, Iiajb_AB, sr.epsA, sr.epsB,
                        this->n_alpha, this->n_beta);
        }
    }

    else if (method.starts_with("cc")) {
        CCResult ccresult = cc_driver(*this, sr, method, reference, econv, max_iter, verbose);
        return E_scf + ccresult.E_cc;
    }

    else { throw std::runtime_error(std::format("Unknown method: {}", method)); }
}