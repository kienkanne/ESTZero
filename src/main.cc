#include "wfn.h"
#include "timer.h"

int main(int argc, char* argv[]) {
    std::string xyz_file;
    std::string basis;
    std::string method = "rhf";
    double e_conv = 1e-6;
    int max_iter = 100;
    int verbose = 2;

    for (int i = 1; i < argc; ++i) {
        std::string_view arg = argv[i];

        if ((arg == "--method" || arg == "-m") && i + 1 < argc) { method = argv[++i]; } 
        else if ((arg == "--tolerance" || arg == "-t") && i + 1 < argc) { e_conv = std::stod(argv[++i]); } 
        else if ((arg == "--max-iter" || arg == "-i") && i + 1 < argc) { max_iter = std::stoi(argv[++i]); } 
        else if ((arg == "--verbose" || arg == "-v") && i + 1 < argc) { verbose = std::stoi(argv[++i]); }  
        else if (xyz_file.empty()) { xyz_file = argv[i]; } 
        else if (basis.empty()) { basis = argv[i]; } 
        else {
            std::println("Unknown or unexpected argument: {}", arg);
            return EXIT_FAILURE;
        }
    }

    // Validate required arguments
    if (xyz_file.empty() || basis.empty()) {
        std::println("Usage: {} <xyz-file> <basis> [--method <val>] [--tolerance <val>] [--max-iter <val>] [--verbose <val>]", argv[0]);
        return EXIT_FAILURE;
    }

    try {
        timing::Scoped _t("Total time", true);
        auto wfn = WaveFunction(xyz_file, basis);
        wfn.get_integrals();
        double E0;

        E0 = wfn.get_energy(method, e_conv, max_iter, verbose);

        std::println("Final SCF energy: {:.8f}", E0);

    } catch (const std::exception& error) {
        std::println("SCF error: {}", error.what());
        return EXIT_FAILURE;
    }
    timing::report();

    return EXIT_SUCCESS;
}