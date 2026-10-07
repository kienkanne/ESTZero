#include <iostream>
#include <vector>
#include <chrono>
#include <immintrin.h>

// 1. Plain loop (compiler will likely auto-vectorize this with -O3)
void add_scalar(const double* A, const double* B, double* C, size_t N) {
    for (size_t i = 0; i < N; ++i) {
        C[i] = A[i] + B[i];
    }
}

// 2. Explicit AVX-512 Intrinsics loop
void add_avx512(const double* A, const double* B, double* C, size_t N) {
    size_t i = 0;
    // Process 8 doubles at a time
    for (; i + 7 < N; i += 8) {
        __m512d va = _mm512_loadu_pd(&A[i]);
        __m512d vb = _mm512_loadu_pd(&B[i]);
        __m512d vc = _mm512_add_pd(va, vb);
        _mm512_storeu_pd(&C[i], vc);
    }
    // Cleanup loop for remaining elements
    for (; i < N; ++i) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    constexpr size_t N = 64 * 1024 * 1024 * 16; 
    constexpr int ITERATIONS = 1;

    // Allocate memory aligned to 64 bytes for optimal AVX-512 loading
    double* a = static_cast<double*>(_mm_malloc(N * sizeof(double), 64));
    double* b = static_cast<double*>(_mm_malloc(N * sizeof(double), 64));
    double* c_scalar = static_cast<double*>(_mm_malloc(N * sizeof(double), 64));
    double* c_avx = static_cast<double*>(_mm_malloc(N * sizeof(double), 64));

    if (!a || !b || !c_scalar || !c_avx) {
        std::cerr << "Memory allocation failed!\n";
        return 1;
    }

    for (size_t i = 0; i < N; ++i) {
        a[i] = static_cast<double>(i);
        b[i] = static_cast<double>(i * 2);
    }

    // Warm-up caches
    add_scalar(a, b, c_scalar, N);
    add_avx512(a, b, c_avx, N);

    // --- Benchmark Plain Loop ---
    auto start = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < ITERATIONS; ++it) {
        add_scalar(a, b, c_scalar, N);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> scalar_duration = end - start;

    // --- Benchmark AVX-512 Loop ---
    start = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < ITERATIONS; ++it) {
        add_avx512(a, b, c_avx, N);
    }
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> avx_duration = end - start;

    std::cout << "Array Size: " << (N * sizeof(double)) / (1024 * 1024) << " MB\n";
    std::cout << "Iterations: " << ITERATIONS << "\n\n";
    std::cout << "Plain Loop Time: " << scalar_duration.count() << " ms\n";
    std::cout << "AVX-512 Time:    " << avx_duration.count() << " ms\n";
    std::cout << "Speedup:         " << scalar_duration.count() / avx_duration.count() << "x\n";

    // Cleanup
    _mm_free(a);
    _mm_free(b);
    _mm_free(c_scalar);
    _mm_free(c_avx);

    return 0;
}