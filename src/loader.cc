#include <fstream>
#include <stdexcept>
#include <libint2.hpp>

#include "loader.h"

Mol loader(const std::string& xyz, const std::string& basis) {
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
    
    libint2::BasisSet obs = libint2::BasisSet(basis, atoms);

    return {obs, atoms, nelectron / 2};
}