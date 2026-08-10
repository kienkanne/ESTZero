#include "wfn.h"
#include "timer.h"

int main(int argc, char* argv[]) {
    std::string xyz_file;
    std::string basis;

    std::string reference = "rhf";
    std::string method = "scf";
    std::string eri_scheme_str = "vector";
    int mult = 1;
    int charge = 0;
    double econv = 1e-6;
    int max_iter = 100;
    int verbose = 2;

    for (int i = 1; i < argc; ++i) {
        std::string_view arg = argv[i];

        if ((arg == "--reference") && i + 1 < argc) { reference = argv[++i]; }
        else if ((arg == "--method") && i + 1 < argc) { method = argv[++i]; }
        else if ((arg == "--eri-scheme") && i + 1 < argc) { eri_scheme_str = argv[++i]; }
        else if ((arg == "--mult") && i + 1 < argc) { mult = std::stoi(argv[++i]); }
        else if ((arg == "--charge") && i + 1 < argc) { charge = std::stoi(argv[++i]); }
        else if ((arg == "--econv") && i + 1 < argc) { econv = std::stod(argv[++i]); }
        else if ((arg == "--max-iter") && i + 1 < argc) { max_iter = std::stoi(argv[++i]); }
        else if ((arg == "--verbose") && i + 1 < argc) { verbose = std::stoi(argv[++i]); }
        else if (xyz_file.empty()) { xyz_file = argv[i]; }
        else if (basis.empty()) { basis = argv[i]; }
        else {
            std::println("Unknown or unexpected argument: {}", arg);
            return EXIT_FAILURE;
        }
    }

    // Validate required arguments
    if (xyz_file.empty() || basis.empty()) {
        std::println(R"(Usage: {} <xyz-file> <basis> [--reference <val>] [--method <val>] 
            [--eri-scheme <vector|tensor>] [--mult <val>] [--charge <val>] [--econv <val>] 
            [--max-iter <val>] [--verbose <val>])", argv[0]);
        return EXIT_FAILURE;
    }

    try {
        timing::Scoped _t("Total time", true);
        auto wfn = WaveFunction(xyz_file, basis, mult, charge);
        wfn.get_integrals(eri_scheme_str);
        double E0;

        E0 = wfn.get_energy(reference, method, econv, max_iter, verbose);

        std::println("Final SCF energy: {:.8f}", E0);

    } catch (const std::exception& error) {
        std::println("SCF error: {}", error.what());
        return EXIT_FAILURE;
    }
    timing::report();

    return EXIT_SUCCESS;
}