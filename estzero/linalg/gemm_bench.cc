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
#include "gemm.h"
#include "timer.h"

struct Route {
    std::string name;
    std::function<void(int M, int N, int K,
                       const double* A, const double* B, double* C)> run;
};

// Computes C = A*B into a zeroed C (alpha=1, beta=0)
static std::vector<Route> make_routes(bool enable_naive) {
    std::vector<Route> routes = {
        {"gemm_reorder", [](int M, int N, int K, const double* A, const double* B, double* C) {
                      gemm_reorder(M, N, K, 1.0, 0.0, A, B, C); }},
        {"openblas", [](int M, int N, int K, const double* A, const double* B, double* C) {
                      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                                  M, N, K, 1.0, A, K, B, N, 0.0, C, N); }},
        {"eigen", [](int M, int N, int K, const double* A, const double* B, double* C) {
                      Eigen::Map<Matrix>(C, M, N).noalias() =
                          Eigen::Map<const Matrix>(A, M, K) *
                          Eigen::Map<const Matrix>(B, K, N); }} };

    if (enable_naive) {
        routes.push_back({"gemm_naive1", [](int M, int N, int K, const double* A, const double* B, double* C) {
                      gemm_naive1(M, N, K, 1.0, 0.0, A, B, C); }});
                      
        routes.push_back({"gemm_naive2", [](int M, int N, int K, const double* A, const double* B, double* C) {
                      gemm_naive2(M, N, K, 1.0, 0.0, A, B, C); }});
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
    std::vector<int> sizes = {256, 512, 1024};
    int repeats = 5;
    bool enable_naive = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--sizes") == 0 && i + 1 < argc) {
            sizes = parse_sizes(argv[++i]);
        } else if (std::strcmp(argv[i], "--repeats") == 0 && i + 1 < argc) {
            repeats = std::stoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--enable-naive") == 0) {
            enable_naive = true; }
    }

    openblas_set_num_threads(1);

    auto routes = make_routes(enable_naive);

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
        std::vector<double> C_ref(static_cast<size_t>(M) * N, 0.0);
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    M, N, K, 1.0, A.data(), K, B.data(), N, 0.0, C_ref.data(), N);

        // Error scales with K
        const double tol = 1e-9 * K;

        // Correctness pass
        for (const auto& r : routes) {
            std::fill(C.begin(), C.end(), 0.0);
            r.run(M, N, K, A.data(), B.data(), C.data());
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
            r.run(M, N, K, A.data(), B.data(), C.data());  // warmup
            const std::string phase = std::format("{:13} {:16}", r.name, shape);
            for (int rep = 0; rep < repeats; ++rep) {
                std::fill(C.begin(), C.end(), 0.0);
                timing::Scoped s(phase);
                r.run(M, N, K, A.data(), B.data(), C.data());
            }
        }
    }

    timing::report();
    return 0;
}
