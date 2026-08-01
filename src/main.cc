#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>

#include <libint2.hpp>

#include "integrals.h"
#include "loader.h"
#include "rhf.h"
#include "utils.h"

int main(int argc, char* argv[]) {
    if (argc < 3 || argc > 5) {
        std::cerr << "Usage: " << argv[0]
                  << " <xyz-file> <basis> [max-iterations] [energy-tolerance]\n";
        return EXIT_FAILURE;
    }

    const int max_iter = argc >= 4 ? std::stoi(argv[3]) : 100;
    const double e_conv = argc == 5 ? std::stod(argv[4]) : 1e-6;

    libint2::initialize();
    try {
        const auto molecule = loader(argv[1], argv[2]);
        const auto& basis = molecule.obs;

        libint2::Engine overlap_engine(libint2::Operator::overlap,
                                       basis.max_nprim(), basis.max_l());
        libint2::Engine kinetic_engine(libint2::Operator::kinetic,
                                       basis.max_nprim(), basis.max_l());
        libint2::Engine nuclear_engine(libint2::Operator::nuclear, 
                                       basis.max_nprim(), basis.max_l());

        nuclear_engine.set_params(make_point_charges(molecule.atoms));

        auto S = build_one_center(overlap_engine, basis);
        auto T = build_one_center(kinetic_engine, basis);
        auto V = build_one_center(nuclear_engine, basis);

        libint2::Engine eri_engine(libint2::Operator::coulomb,
                                   basis.max_nprim(), basis.max_l());
        auto eri = build_canonical_eris(eri_engine, basis);
        const double energy = roothan_solve(
            S, T, V, eri, compute_nuclear_repulsion(molecule.atoms),
            molecule.ndocc, max_iter, e_conv);

        std::cout << std::setprecision(12) << "Final SCF energy: " << energy << '\n';
    } catch (const std::exception& error) {
        std::cerr << "SCF error: " << error.what() << '\n';
        libint2::finalize();
        return EXIT_FAILURE;
    }

    libint2::finalize();
    return EXIT_SUCCESS;
}