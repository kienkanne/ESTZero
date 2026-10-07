#ifndef EST_GEMM
#define EST_GEMM

#include <cstddef>
#include <stdexcept>
#include <string>
#include <cblas.h>
#include "gemm_all.h"
#include "einsum.h"


enum class GEMMScheme {
    ODO, EIGEN, OPENBLAS, CUBLAS_DGEMM, CUBLAS_SGEMM,
    AVX_8x8, AVX_16x8, AVX_24x8, CUTENSOR,
};
inline GEMMScheme gs;

// Minimum estimated FLOPs (left_size * summed_size * right_size) for try_gemm to route through GEMM at all. 
// Below this, the odometer scheme is used instead. use --gemm-min in main.cc to adjust this default.
inline std::ptrdiff_t gemm_threshold = 1024;

inline void set_gemm_scheme(std::string g) {
    if      (g == "odometer")   gs = GEMMScheme::ODO;
    else if (g == "eigen")      gs = GEMMScheme::EIGEN;
    else if (g == "openblas")   gs = GEMMScheme::OPENBLAS;
    else if (g == "cublas_dgemm")   gs = GEMMScheme::CUBLAS_DGEMM;
    else if (g == "cublas_sgemm")   gs = GEMMScheme::CUBLAS_SGEMM;
    else if (g == "avx_8x8")    gs = GEMMScheme::AVX_8x8;
    else if (g == "avx_16x8")   gs = GEMMScheme::AVX_16x8;
    else if (g == "avx_24x8")   gs = GEMMScheme::AVX_24x8;
    else if (g == "cutensor")   { gs = GEMMScheme::CUTENSOR; throw std::runtime_error("cuTENSOR temporarily disabled"); }
    else throw std::runtime_error("Unknown gemm-scheme: " + g);
}

bool try_cutensor(const std::vector<std::string>& terms_strings, Operands operands, Tensor& result);

// C (M x N) = alpha * op(A) * op(B) + beta * C, where op(A) is M x K and op(B) is K x N (stored transposed if trans).
// Row-major with BLAS leading dimensions: A is stored (trans ? K x M : M x K) with row stride lda, likewise B, and C
// (M x N) with row stride ldc. Eigen and OpenBLAS take any valid ld, the other backends only packed ones (ld = row length).
inline void call_gemm(GEMMScheme gs, int M, int N, int K, int lda, int ldb, int ldc, const double *A, const double *B, double *C,
                    bool transA = false, bool transB = false, double alpha = 1.0, double beta = 0.0) {
    // The odometer scheme means "no manually calling GEMM": einsum never reaches here under it, so a direct call is a caller bug.
    if (gs == GEMMScheme::ODO) throw std::runtime_error("call_gemm: the odometer scheme has no GEMM backend");

    else if (gs == GEMMScheme::EIGEN) {
        // Zero-copy row-major views of the raw buffers. For a row-major Map the outer stride is the row stride (~BLAS ld)
        using Stride = Eigen::OuterStride<>;
        Eigen::Map<const Matrix, 0, Stride> Am(A, transA ? K : M, transA ? M : K, Stride(lda));
        Eigen::Map<const Matrix, 0, Stride> Bm(B, transB ? N : K, transB ? K : N, Stride(ldb));
        Eigen::Map<Matrix, 0, Stride> Cm(C, M, N, Stride(ldc));

        // generic lambda for Eigen
        auto gemm = [&](const auto& opA, const auto& opB) {
            if (beta == 0.0) {
                Cm.noalias() = alpha * opA * opB;
            } else {
                Cm *= beta;
                Cm.noalias() += alpha * opA * opB;
            }
        };
        if      (!transA && !transB) gemm(Am, Bm);
        else if ( transA && !transB) gemm(Am.transpose(), Bm);
        else if (!transA &&  transB) gemm(Am, Bm.transpose());
        else                         gemm(Am.transpose(), Bm.transpose());
    }
    else if (gs == GEMMScheme::OPENBLAS) {
        auto TA = (transA) ? CblasTrans : CblasNoTrans;
        auto TB = (transB) ? CblasTrans : CblasNoTrans;

        cblas_dgemm(CblasRowMajor, TA, TB,
            M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    }
    else if (lda != (transA ? M : K) || ldb != (transB ? K : N) || ldc != N) {
        throw std::runtime_error("call_gemm: this GEMM scheme only takes packed operands (ld = row length)");
    }
    else if (gs == GEMMScheme::CUBLAS_DGEMM) cublas_dgemm(M, N, K, alpha, beta, A, B, C, nullptr, transA, transB);
    else if (gs == GEMMScheme::CUBLAS_SGEMM) cublas_sgemm(M, N, K, alpha, beta, A, B, C, nullptr, transA, transB);
    
    else if (transA || transB) throw std::runtime_error("Tranpose not yet implemented for AVX family!");
    else if (gs == GEMMScheme::AVX_8x8)      avx_8x8(M, N, K, alpha, beta, A, B, C);
    else if (gs == GEMMScheme::AVX_16x8)    avx_16x8(M, N, K, alpha, beta, A, B, C);
    else if (gs == GEMMScheme::AVX_24x8)    avx_24x8(M, N, K, alpha, beta, A, B, C);
    else throw std::runtime_error("call_gemm: unknown GEMMScheme");
};


#endif