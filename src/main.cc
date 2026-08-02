#include <cstdlib>
#include <string>

#include <Eigen/Dense>
#include <libint2.hpp>

#include "integrals.h"
#include "loader.h"
#include "rhf.h"
#include "timer.h"
#include "utils.h"

int main(int argc, char* argv[]) {
    if (argc < 3 || argc > 5) {
        std::println("Usage: {} <xyz-file> <basis> [max-iterations] [energy-tolerance]", argv[0]);
        return EXIT_FAILURE;
    }

    const int max_iter = argc >= 4 ? std::stoi(argv[3]) : 100;
    const double e_conv = argc == 5 ? std::stod(argv[4]) : 1e-6;

    try {
        timing::Scoped _t("Total time", true);

        libint2::initialize();
        const auto molecule = loader(argv[1], argv[2]);
        const auto& basis = molecule.obs;

        libint2::Engine overlap_engine(libint2::Operator::overlap,
                                       basis.max_nprim(), basis.max_l());
        libint2::Engine kinetic_engine(libint2::Operator::kinetic,
                                       basis.max_nprim(), basis.max_l());
        libint2::Engine nuclear_engine(libint2::Operator::nuclear, 
                                       basis.max_nprim(), basis.max_l());

        nuclear_engine.set_params(make_point_charges(molecule.atoms));
        
        std::println("nbf: {}", basis.nbf());

        Eigen::MatrixXd S, T, V;
        {
            timing::Scoped _t("One-electron (S,T,V)");
            S = build_one_center(overlap_engine, basis);
            T = build_one_center(kinetic_engine, basis);
            V = build_one_center(nuclear_engine, basis);
        }

        libint2::Engine eri_engine(libint2::Operator::coulomb,
                                   basis.max_nprim(), basis.max_l());
        std::vector<double> eri;
        {
            timing::Scoped _t("ERI build");
            eri = build_canonical_eris(eri_engine, basis);
        }

        double energy;
        {
            timing::Scoped _t("SCF iterations");
            energy = roothan_solve(
                S, T, V, eri, compute_nuclear_repulsion(molecule.atoms),
                molecule.ndocc, max_iter, e_conv);
        }
        std::println("Final SCF energy: {:.8f}", energy);

    } catch (const std::exception& error) {
        std::println("SCF error: {}", error.what());
        libint2::finalize();
        return EXIT_FAILURE;
    }
    timing::report();

    libint2::finalize();
    return EXIT_SUCCESS;
}