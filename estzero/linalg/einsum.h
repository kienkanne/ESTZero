#ifndef EINSUM_H
#define EINSUM_H

#include <string>
#include <vector>
#include <cblas.h>
#include "tensor.h"
#include "gemm_all.h"

enum class GEMMScheme {
    ODO, EIGEN, BLAS, CUBLAS,
    AVX_8x8, AVX_16x8, AVX_24x8,
};
inline GEMMScheme gs;

// Minimum estimated FLOPs (left_size * summed_size * right_size) for try_gemm to route through GEMM at all. 
// Below this, the odometer scheme is used instead. use --gemm-min in main.cc to adjust this default.
inline size_t gemm_threshold = 1024;

inline void set_gemm_scheme(std::string g) {
    if      (g == "odometer")   gs = GEMMScheme::ODO;
    else if (g == "eigen")      gs = GEMMScheme::EIGEN;
    else if (g == "blas")       gs = GEMMScheme::BLAS;
    else if (g == "cublas")     gs = GEMMScheme::CUBLAS;
    else if (g == "avx_8x8")    gs = GEMMScheme::AVX_8x8;
    else if (g == "avx_16x8")   gs = GEMMScheme::AVX_16x8;
    else if (g == "avx_24x8")   gs = GEMMScheme::AVX_24x8;
    else throw std::runtime_error("Unknown gemm-scheme: " + g);
}

// Mirroring `gemm_bench.cc` to initialize non-Eigen routes for GEMM
inline auto raw_gemm =
    [](GEMMScheme gs, int M, int N, int K, const double *A, const double *B, double *C) {
        double alpha = 1.0; double beta = 0.0;

        if (gs == GEMMScheme::BLAS) {
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K, alpha, A, K /*lda*/, B, N /*ldb*/, beta, C, N /*ldc*/);
        }
        else if (gs == GEMMScheme::AVX_8x8)      avx_8x8(M, N, K, alpha, beta, A, B, C);
        else if (gs == GEMMScheme::AVX_16x8)    avx_16x8(M, N, K, alpha, beta, A, B, C);
        else if (gs == GEMMScheme::AVX_24x8)    avx_24x8(M, N, K, alpha, beta, A, B, C);
        else if (gs == GEMMScheme::CUBLAS)  cublas_dgemm(M, N, K, alpha, beta, A, B, C);
        else throw std::runtime_error("raw_gemm: unknown GEMMScheme");
    };

// General-purpose einsum, implemented from scratch (no Eigen math), operating
// on the row-major Tensor container defined in tensor.h.
Tensor einsum(const std::string& subscripts, const std::vector<Tensor>& operands);

#endif
