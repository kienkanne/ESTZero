#include "libint2/initialize.h"
#include "wfn.h"
#include "timer.h"
#include "einsum.h"
#include <cblas.h>

int main(int argc, char* argv[]) {
    std::string xyz_file;
    std::string basis;

    std::string reference = "rhf";
    std::string method = "scf";
    std::string eri_scheme_str = "tensor";
    std::string gemm_scheme_str = "blas";
    int mult = 1;
    int charge = 0;
    double econv = 1e-8;
    int max_iter = 200;
    int verbose = 2;

    for (int i = 1; i < argc; ++i) {
        std::string_view arg = argv[i];

        if ((arg == "--reference") && i + 1 < argc) { reference = argv[++i]; }
        else if ((arg == "--method") && i + 1 < argc) { method = argv[++i]; }
        else if ((arg == "--eri-scheme") && i + 1 < argc) { eri_scheme_str = argv[++i]; }
        else if ((arg == "--gemm-scheme") && i + 1 < argc) { gemm_scheme_str = argv[++i]; }
        else if ((arg == "--gemm-min") && i + 1 < argc) { gemm_threshold = std::stoul(argv[++i]); }
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
            [--mult <val>] [--charge <val>]
            [--eri-scheme <vector|tensor>]
            [--gemm-scheme <odometer|eigen|blas|cublas_dgemm|cublas_sgemm|avx_8x8|avx_16x8|avx_24x8|cutensor>] [--gemm-min <val>]
            [--econv <val>] [--max-iter <val>] [--verbose <val>])", argv[0]);
        return EXIT_FAILURE;
    }

    try {
        openblas_set_num_threads(1);
        set_gemm_scheme(gemm_scheme_str);

        timing::Scoped _t("Total time", true);
        libint2::initialize();

        auto wfn = WaveFunction(xyz_file, basis, mult, charge);
        wfn.get_integrals(eri_scheme_str, verbose);
        
        double E;
        E = wfn.get_energy(reference, method, econv, max_iter, verbose);
        std::println("Final energy: {:.8f}", E);
        
        libint2::finalize();

    } catch (const std::exception& error) {
        std::println("ESTZero error: {}", error.what());
        return EXIT_FAILURE;
    }
    timing::report();

    return EXIT_SUCCESS;
}