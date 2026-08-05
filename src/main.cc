#include "wfn.h"
#include "timer.h"

int main(int argc, char* argv[]) {
    if (argc < 3 || argc > 6) {
        std::println("Usage: {} <xyz-file> <basis> [energy-tolerance] [max-iterations] [verbose]", argv[0]);
        return EXIT_FAILURE;
    }

    const double e_conv = argc >= 4 ? std::stod(argv[3]) : 1e-6;
    const int max_iter = argc >= 5 ? std::stoi(argv[4]) : 100;
    const int verbose = argc >= 6 ? std::stoi(argv[5]) : 2;

    try {
        timing::Scoped _t("Total time", true);
        auto wfn = WaveFunction(argv[1], argv[2]);
        wfn.get_integrals();
        const double E0 = wfn.get_energy("rhf", e_conv, max_iter, verbose);

        std::println("Final SCF energy: {:.8f}", E0);

    } catch (const std::exception& error) {
        std::println("SCF error: {}", error.what());
        return EXIT_FAILURE;
    }
    timing::report();

    return EXIT_SUCCESS;
}