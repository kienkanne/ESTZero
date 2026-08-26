// GEMM microbenchmark, independent from the main EST pipeline. It times every GEMM route on a few
// large square matrices. This is preparation for the future manually written GEMM optimizations.

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

struct Route {
    std::string name;
    std::function<void(int M, int N, int K, double alpha, double beta,
                       const double* A, const double* B, double* C)> run;
};


static std::vector<Route> make_routes(bool enable_naive) {

    std::vector<Route> routes = {
        {"openblas", [](int M, int N, int K, double alpha, double beta, const double* A, const double* B, double* C) {
                      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                      M, N, K, alpha, A, K, B, N, beta, C, N); }},
        {"eigen", [](int M, int N, int K, double alpha, double beta, const double* A, const double* B, double* C) {
                  Eigen::Map<Matrix> C_map(C, M, N);
                  C_map = alpha * (Eigen::Map<const Matrix>(A, M, K) *
                          Eigen::Map<const Matrix>(B, K, N)) +
                      beta * C_map; }},
        {"mkn_4x1", [](int M, int N, int K, double alpha, double beta, const double* A, const double* B, double* C) {
                  mkn_4x1(M, N, K, alpha, beta, A, B, C); }},
        {"mkn_8x1", [](int M, int N, int K, double alpha, double beta, const double* A, const double* B, double* C) {
                  mkn_8x1(M, N, K, alpha, beta, A, B, C); }},
        {"mnk_4x1", [](int M, int N, int K, double alpha, double beta, const double* A, const double* B, double* C) {
                  mnk_4x1(M, N, K, alpha, beta, A, B, C); }},
        {"mnk_1x4", [](int M, int N, int K, double alpha, double beta, const double* A, const double* B, double* C) {
                  mnk_1x4(M, N, K, alpha, beta, A, B, C); }},
        {"mnk_8x1", [](int M, int N, int K, double alpha, double beta, const double* A, const double* B, double* C) {
                  mnk_8x1(M, N, K, alpha, beta, A, B, C); }},
        {"mnk_1x8", [](int M, int N, int K, double alpha, double beta, const double* A, const double* B, double* C) {
                  mnk_1x8(M, N, K, alpha, beta, A, B, C); }},
        {"mnk_4x4", [](int M, int N, int K, double alpha, double beta, const double* A, const double* B, double* C) {
                  mnk_4x4(M, N, K, alpha, beta, A, B, C); }},
        {"mnk_8x8", [](int M, int N, int K, double alpha, double beta, const double* A, const double* B, double* C) {
                  mnk_8x8(M, N, K, alpha, beta, A, B, C); }},
                };

    if (enable_naive) {
        routes.push_back(
         {"naive_mnk", [](int M, int N, int K, double alpha, double beta, const double* A, const double* B, double* C) {
                      naive_mnk(M, N, K, alpha, beta, A, B, C); }});

        routes.push_back(
         {"naive_mkn", [](int M, int N, int K, double alpha, double beta, const double* A, const double* B, double* C) {
                      naive_mkn(M, N, K, alpha, beta, A, B, C); }});
    }

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

int main(int argc, char** argv) {
    std::vector<int> sizes = {256, 511, 1022, /*2045*/};
    int repeats = 3;
    bool enable_naive = false;
    std::string only;  // if set, run just this one route -- needed so `perf stat` can attribute
                        // counters to a single kernel instead of the whole route loop.

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--sizes") == 0 && i + 1 < argc) {
            sizes = parse_sizes(argv[++i]);
        } else if (std::strcmp(argv[i], "--repeats") == 0 && i + 1 < argc) {
            repeats = std::stoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--enable_naive") == 0) {
            enable_naive = true;
        } else if (std::strcmp(argv[i], "--only") == 0 && i + 1 < argc) {
            only = argv[++i];
        }
    }

    openblas_set_num_threads(1);

    auto routes = make_routes(enable_naive || !only.empty());
    if (!only.empty()) {
        std::erase_if(routes, [&](const Route& r) { return r.name != only; });
        if (routes.empty()) {
            std::println("no route named '{}'", only);
            return 1;
        }
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

        // Error scales with K
        const double tol = 1e-9 * K;

        // Correctness pass
        for (const auto& r : routes) {
            std::fill(C.begin(), C.end(), 1.0);
            r.run(M, N, K, alpha, beta, A.data(), B.data(), C.data());
            double max_diff = 0.0;
            for (size_t i = 0; i < C.size(); ++i) {
                max_diff = std::max(max_diff, std::abs(C[i] - C_ref[i]));
            }
            std::println("[check] {:13} {:16} {:5} max_diff={:.3e}",
                         r.name, shape, (max_diff < tol ? "PASS" : "FAIL"), max_diff);
        }

        // One warmup call, then repeats (timer module measure the total, average, and num calls)
        // Zeroing C outside the scoped block so the memset isn't timed, inside the timing scope
        // is just purely doing arithmetic (all route already verifed correctness above).
        for (const auto& r : routes) {
            std::fill(C.begin(), C.end(), 0.0);
            r.run(M, N, K, alpha, beta, A.data(), B.data(), C.data());  // warmup
            const std::string phase = std::format("{:13} {:16}", r.name, shape);
            for (int rep = 0; rep < repeats; ++rep) {
                std::fill(C.begin(), C.end(), 0.0);
                timing::Scoped s(phase);
                r.run(M, N, K, alpha, beta, A.data(), B.data(), C.data());
            }
        }
    }

    timing::report();
    return 0;
}
