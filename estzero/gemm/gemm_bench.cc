// GEMM microbenchmark, independent from the main EST pipeline. It times every GEMM route on a few
// large square matrices. This is preparation for the future manually written GEMM optimizations.

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <print>
#include <random>
#include <string>
#include <vector>
#include <cblas.h>

#include "tensor.h" // Matrix (row-major Eigen alias)
#include "gemm_all.h"
#include "timer.h"

// `kernel_ms` is only not null for `is_gpu` routes only. GPU lambdas forward it straight into the
// matching gemm_all.h function so it measures kernel-only time internally (cudaEvents), not the
// cudaMalloc/H2D/D2H/free around it.
struct Route {
    std::string name;
    bool is_gpu = false;
    std::function<void(int M, int N, int K, double alpha, double beta,
                       const double* A, const double* B, double* C, double* kernel_ms)> run;
};

// GPU family: always available regardless of --cpu, since this is gemm_bench's default focus now
static void add_gpu_routes(std::vector<Route>& routes) {
    routes.push_back({"cublas_sgemm", true,
        [](int M, int N, int K, double alpha, double beta, const double* A, const double* B, double* C, double* kernel_ms) {
                  cublas_sgemm(M, N, K, alpha, beta, A, B, C, kernel_ms); }});
    routes.push_back({"cublas_dgemm", true,
        [](int M, int N, int K, double alpha, double beta, const double* A, const double* B, double* C, double* kernel_ms) {
                  cublas_dgemm(M, N, K, alpha, beta, A, B, C, kernel_ms); }});

    routes.push_back({"cu1s_naive", true,
        [](int M, int N, int K, double alpha, double beta, const double* A, const double* B, double* C, double* kernel_ms) {
                  cu_s_ladder(M, N, K, alpha, beta, A, B, C, 1, kernel_ms); }});
    routes.push_back({"cu2s_coalesce", true,
        [](int M, int N, int K, double alpha, double beta, const double* A, const double* B, double* C, double* kernel_ms) {
                  cu_s_ladder(M, N, K, alpha, beta, A, B, C, 2, kernel_ms); }});
    routes.push_back({"cu3s_smem", true,
        [](int M, int N, int K, double alpha, double beta, const double* A, const double* B, double* C, double* kernel_ms) {
                  cu_s_ladder(M, N, K, alpha, beta, A, B, C, 3, kernel_ms); }});
    routes.push_back({"cu4s_thread_mul", true,
        [](int M, int N, int K, double alpha, double beta, const double* A, const double* B, double* C, double* kernel_ms) {
                  cu_s_ladder(M, N, K, alpha, beta, A, B, C, 4, kernel_ms); }});

    routes.push_back({"cu1d_naive", true,
        [](int M, int N, int K, double alpha, double beta, const double* A, const double* B, double* C, double* kernel_ms) {
                  cu_d_ladder(M, N, K, alpha, beta, A, B, C, 1, kernel_ms); }});
    routes.push_back({"cu2d_coalesce", true,
        [](int M, int N, int K, double alpha, double beta, const double* A, const double* B, double* C, double* kernel_ms) {
                  cu_d_ladder(M, N, K, alpha, beta, A, B, C, 2, kernel_ms); }});
    routes.push_back({"cu3d_smem", true,
        [](int M, int N, int K, double alpha, double beta, const double* A, const double* B, double* C, double* kernel_ms) {
                  cu_d_ladder(M, N, K, alpha, beta, A, B, C, 3, kernel_ms); }});
}

// CPU family via --cpu
// `enable_legacy` further opens every pre-AVX hand-rolled route (naive_*/mkn_RxR/mnk_RxR).
static void add_cpu_routes(std::vector<Route>& routes, bool enable_legacy) {
    routes.push_back({"openblas", false,
        [](int M, int N, int K, double alpha, double beta, const double* A, const double* B, double* C, double*) {
                      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                      M, N, K, alpha, A, K, B, N, beta, C, N); }});
    routes.push_back({"eigen", false,
        [](int M, int N, int K, double alpha, double beta, const double* A, const double* B, double* C, double*) {
                  Eigen::Map<Matrix> C_map(C, M, N);
                  C_map = alpha * (Eigen::Map<const Matrix>(A, M, K) *
                          Eigen::Map<const Matrix>(B, K, N)) +
                      beta * C_map; }});

    routes.push_back({"avx_8x8", false,
        [](int M, int N, int K, double alpha, double beta, const double* A, const double* B, double* C, double*) {
                  avx_8x8(M, N, K, alpha, beta, A, B, C); }});
    routes.push_back({"avx_12x8", false,
        [](int M, int N, int K, double alpha, double beta, const double* A, const double* B, double* C, double*) {
                  avx_12x8(M, N, K, alpha, beta, A, B, C); }});
    routes.push_back({"avx_16x8", false,
        [](int M, int N, int K, double alpha, double beta, const double* A, const double* B, double* C, double*) {
                  avx_16x8(M, N, K, alpha, beta, A, B, C); }});
    routes.push_back({"avx_20x8", false,
        [](int M, int N, int K, double alpha, double beta, const double* A, const double* B, double* C, double*) {
                  avx_20x8(M, N, K, alpha, beta, A, B, C); }});
    routes.push_back({"avx_24x8", false,
        [](int M, int N, int K, double alpha, double beta, const double* A, const double* B, double* C, double*) {
                  avx_24x8(M, N, K, alpha, beta, A, B, C); }});

    routes.push_back({"avx_blis_8x8", false,
        [](int M, int N, int K, double alpha, double beta, const double* A, const double* B, double* C, double*) {
                  avx_blis_8x8(M, N, K, alpha, beta, A, B, C); }});
    routes.push_back({"avx_blis_12x8", false,
        [](int M, int N, int K, double alpha, double beta, const double* A, const double* B, double* C, double*) {
                  avx_blis_12x8(M, N, K, alpha, beta, A, B, C); }});
    routes.push_back({"avx_blis_16x8", false,
        [](int M, int N, int K, double alpha, double beta, const double* A, const double* B, double* C, double*) {
                  avx_blis_16x8(M, N, K, alpha, beta, A, B, C); }});
    routes.push_back({"avx_blis_20x8", false,
        [](int M, int N, int K, double alpha, double beta, const double* A, const double* B, double* C, double*) {
                  avx_blis_20x8(M, N, K, alpha, beta, A, B, C); }});
    routes.push_back({"avx_blis_24x8", false,
        [](int M, int N, int K, double alpha, double beta, const double* A, const double* B, double* C, double*) {
                  avx_blis_24x8(M, N, K, alpha, beta, A, B, C); }});

    if (enable_legacy) {
        routes.push_back({"avx_1x8", false,
            [](int M, int N, int K, double alpha, double beta, const double* A, const double* B, double* C, double*) {
                      avx_1x8(M, N, K, alpha, beta, A, B, C); }});
        routes.push_back({"naive_mnk", false,
            [](int M, int N, int K, double alpha, double beta, const double* A, const double* B, double* C, double*) {
                      naive_mnk(M, N, K, alpha, beta, A, B, C); }});
        routes.push_back({"naive_mkn", false,
            [](int M, int N, int K, double alpha, double beta, const double* A, const double* B, double* C, double*) {
                      naive_mkn(M, N, K, alpha, beta, A, B, C); }});
        routes.push_back({"mkn_4x1", false,
            [](int M, int N, int K, double alpha, double beta, const double* A, const double* B, double* C, double*) {
                  mkn_4x1(M, N, K, alpha, beta, A, B, C); }});
        routes.push_back({"mkn_8x1", false,
            [](int M, int N, int K, double alpha, double beta, const double* A, const double* B, double* C, double*) {
                  mkn_8x1(M, N, K, alpha, beta, A, B, C); }});
        routes.push_back({"mnk_4x1", false,
            [](int M, int N, int K, double alpha, double beta, const double* A, const double* B, double* C, double*) {
                  mnk_4x1(M, N, K, alpha, beta, A, B, C); }});
        routes.push_back({"mnk_1x4", false,
            [](int M, int N, int K, double alpha, double beta, const double* A, const double* B, double* C, double*) {
                  mnk_1x4(M, N, K, alpha, beta, A, B, C); }});
        routes.push_back({"mnk_8x1", false,
            [](int M, int N, int K, double alpha, double beta, const double* A, const double* B, double* C, double*) {
                  mnk_8x1(M, N, K, alpha, beta, A, B, C); }});
        routes.push_back({"mnk_1x8", false,
            [](int M, int N, int K, double alpha, double beta, const double* A, const double* B, double* C, double*) {
                  mnk_1x8(M, N, K, alpha, beta, A, B, C); }});
        routes.push_back({"mnk_4x4", false,
            [](int M, int N, int K, double alpha, double beta, const double* A, const double* B, double* C, double*) {
                  mnk_4x4(M, N, K, alpha, beta, A, B, C); }});
        routes.push_back({"mnk_8x8", false,
            [](int M, int N, int K, double alpha, double beta, const double* A, const double* B, double* C, double*) {
                  mnk_8x8(M, N, K, alpha, beta, A, B, C); }});
    }
}

// GPU routes are always included
static std::vector<Route> make_routes(bool include_cpu, bool enable_legacy) {
    std::vector<Route> routes;
    add_gpu_routes(routes);
    if (include_cpu) add_cpu_routes(routes, enable_legacy);
    return routes;
}

static std::vector<int> parse_sizes(const char* arg) {
    std::vector<int> sizes;
    std::string s(arg);
    size_t start = 0;
    while (start < s.size()) {
        size_t comma = s.find(',', start);
        if (comma == std::string::npos) comma = s.size();
        sizes.push_back(std::stoi(s.substr(start, comma - start)));
        start = comma + 1;
    }
    return sizes;
}

static std::vector<std::string> parse_names(const char* arg) {
    std::vector<std::string> names;
    std::string s(arg);
    size_t start = 0;
    while (start < s.size()) {
        size_t comma = s.find(',', start);
        if (comma == std::string::npos) comma = s.size();
        names.push_back(s.substr(start, comma - start));
        start = comma + 1;
    }
    return names;
}

// True for cublas_sgemm and any hand-written single-precision ladder route, i.e. "cu<N>s_*"
// (cu1s_naive, cu2s_coalesce, ...) as opposed to the double-precision "cu<N>d_*" ones. Matching on
// "sgemm" alone (cublas_sgemm/cublas_dgemm) isn't enough since the cuN_ ladder routes don't carry
// that suffix -- they mark precision with the s/d right after the digit instead.
static bool is_single_precision_route(const std::string& name) {
    if (name.find("sgemm") != std::string::npos) return true;
    if (name.size() < 3 || name[0] != 'c' || name[1] != 'u') return false;
    size_t i = 2;
    while (i < name.size() && std::isdigit(static_cast<unsigned char>(name[i]))) ++i;
    return i < name.size() && name[i] == 's';
}

// Default sizes are family-dependent. Larger default sizes for GPU to actually stress test
static const std::vector<int> CPU_DEFAULT_SIZES = {510, 1022, 1024, 2045};
static const std::vector<int> GPU_DEFAULT_SIZES = {510, 1022, 1024, 2045, 4096, 6144, 8192};

int main(int argc, char** argv) {
    std::vector<int> sizes;
    bool sizes_explicit = false;
    int repeats = 3;
    bool enable_legacy = false;
    bool include_cpu = false;
    std::string only;
    std::string routes_filter;
    bool list_routes = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--sizes") == 0 && i + 1 < argc) {
            sizes = parse_sizes(argv[++i]);
            sizes_explicit = true;
        } else if (std::strcmp(argv[i], "--repeats") == 0 && i + 1 < argc) {
            repeats = std::stoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--legacy") == 0) {
            enable_legacy = true;
        } else if (std::strcmp(argv[i], "--cpu") == 0) {
            include_cpu = true;
        } else if (std::strcmp(argv[i], "--only") == 0 && i + 1 < argc) {
            only = argv[++i];
        } else if (std::strcmp(argv[i], "--routes") == 0 && i + 1 < argc) {
            routes_filter = argv[++i];
        } else if (std::strcmp(argv[i], "--list-routes") == 0) {
            list_routes = true;
        }
    }

    // Lets callers (bench/gemm_bench.py) discover the current route set at runtime instead
    // of hardcoding a list. Lists the full universe (GPU + CPU + legacy) regardless of --cpu
    if (list_routes) {
        for (const auto& r : make_routes(/*include_cpu=*/true, /*enable_legacy=*/true)) {
            std::println("{}", r.name);
        }
        return 0;
    }

    openblas_set_num_threads(1);

    // Naming a route explicitly (--only/--routes) or asking for --legacy always builds the full
    // universe first, then narrows below -- same as --cpu, just implicit.
    const bool need_full = enable_legacy || !only.empty() || !routes_filter.empty();
    auto routes = make_routes(include_cpu || need_full, enable_legacy);

    if (!only.empty()) {
        std::erase_if(routes, [&](const Route& r) { return r.name != only; });
        if (routes.empty()) {
            std::println("no route named '{}'", only);
            return 1;
        }
    } else if (!routes_filter.empty()) {
        const std::vector<std::string> names = parse_names(routes_filter.c_str());
        std::erase_if(routes, [&](const Route& r) {
            return std::find(names.begin(), names.end(), r.name) == names.end();
        });
        if (routes.empty()) {
            std::println("no routes matched '{}'", routes_filter);
            return 1;
        }
    }

    if (!sizes_explicit) {
        const bool any_cpu = std::any_of(routes.begin(), routes.end(),
                                          [](const Route& r) { return !r.is_gpu; });
        sizes = any_cpu ? CPU_DEFAULT_SIZES : GPU_DEFAULT_SIZES;
    }

    // Fixed seed
    std::mt19937 rng(12345);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    std::println("GEMM microbenchmark (single-threaded, repeats={})", repeats);

    for (int n : sizes) {
        const int M = n, N = n, K = n;
        const std::string shape =
            std::to_string(M) + "x" + std::to_string(N) + "x" + std::to_string(K);

        std::vector<double> A(static_cast<size_t>(M) * K);
        std::vector<double> B(static_cast<size_t>(K) * N);
        std::vector<double> C(static_cast<size_t>(M) * N);
        for (double& a : A) a = dist(rng);
        for (double& b : B) b = dist(rng);

        // Compute with OpenBLAS once as oracle.
        // Computes C = A*B into a C with ones; alpha=1.5, beta=0.5 for potential bugs checking
        double alpha = 1.5;
        double beta = 0.5;
        std::vector<double> C_ref(static_cast<size_t>(M) * N, 1.0);
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    M, N, K, alpha, A.data(), K, B.data(), N, beta, C_ref.data(), N);

        // Correctness pass. Single-precision routes (cublas_sgemm, and any hand-written cuN_s_*
        // ladder route -- see is_single_precision_route()) accumulate in float (~1e-7 relative
        // precision vs. double's ~1e-16), so they need a much looser tolerance -- same "error
        // scales with K" shape, just the coefficient bumped by float's ~1e6 worse precision. This
        // is an EXPECTED, correct-behavior difference (the whole point of trying single
        // precision), not a bug -- the raw max_diff is still printed either way so the actual
        // accuracy tradeoff is visible.
        for (const auto& r : routes) {
            const double tol = is_single_precision_route(r.name) ? 1e-3 * K : 1e-9 * K;
            std::fill(C.begin(), C.end(), 1.0);
            r.run(M, N, K, alpha, beta, A.data(), B.data(), C.data(), nullptr);
            double max_diff = 0.0;
            for (size_t i = 0; i < C.size(); ++i) {
                max_diff = std::max(max_diff, std::abs(C[i] - C_ref[i]));
            }
            std::println("[check] {:13} {:16} {:5} max_diff={:.3e}",
                         r.name, shape, (max_diff < tol ? "PASS" : "FAIL"), max_diff);
        }

        // One warmup call, then repeats (timer module measure the total, average, and num calls).
        // GPU routes (`is_gpu`) report their own kernel-only elapsed time (cudaEvents inside
        // cublas_dgemm/cublas_sgemm/cu_s_ladder/cu_d_ladder -- see gemm_all.h's `kernel_ms`), fed straight into
        // timing::add, so cudaMalloc/H2D/D2H/free around the kernel are excluded from the number.
        // CPU routes keep the old wall-clock `timing::Scoped` wrap.
        for (const auto& r : routes) {
            std::fill(C.begin(), C.end(), 0.0);
            r.run(M, N, K, alpha, beta, A.data(), B.data(), C.data(), nullptr);  // warmup
            const std::string phase = std::format("{:13} {:16}", r.name, shape);
            for (int rep = 0; rep < repeats; ++rep) {
                std::fill(C.begin(), C.end(), 0.0);
                if (r.is_gpu) {
                    double kernel_ms = 0.0;
                    r.run(M, N, K, alpha, beta, A.data(), B.data(), C.data(), &kernel_ms);
                    timing::add(phase, kernel_ms / 1000.0, false);
                } else {
                    timing::Scoped s(phase);
                    r.run(M, N, K, alpha, beta, A.data(), B.data(), C.data(), nullptr);
                }
            }
        }
    }

    timing::report();
    return 0;
}
