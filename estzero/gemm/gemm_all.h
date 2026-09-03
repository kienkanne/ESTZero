#ifndef GEMM_H
#define GEMM_H

// Manually written GEMM optimization ladder. A(M x K) x B(K x N) = C(M x N), all row-major.
// Convention: C = alpha*(A*B) + beta*C.
//
// Three families:
// - naive_* : unoptimized reference loops, neither blocked nor register-accumulating.
//
// - mkn_RxR : loop order fixed at m-outer/k-middle/n-inner, keeping inner contiguous for B and C.
// MR blocks rows of A/C, reused against B. NR (=1 always) because it's meaningless to block along
// an already contiguous dimension for B and C. This is blocking for input data.
//
// - mnk_RxR : tiles an MR x NR block of C, held as named local accumulators across the entire k 
// reduction, written to C once, MR x NR is the tile size. This is the "canonical" loop that major
// libraries uses.
//
// - avx_RxR : utilizes <immintrin.h> AVX-512 intrinsics to vectorize the accumulators fill a zmm
// register completely with 8 entries on a C row, providing actual vectorization instead of manually
// declaring all named local variables in `mnk_RxR` and hoping the compiler does so.

void naive_mnk(const int M, const int N, const int K, const double alpha, const double beta,
                const double* A, const double* B, double* C);
void naive_mkn(const int M, const int N, const int K, const double alpha, const double beta,
                const double* A, const double* B, double* C);

void mkn_4x1(const int M, const int N, const int K, const double alpha, const double beta,
                const double* A, const double* B, double* C);
void mkn_8x1(const int M, const int N, const int K, const double alpha, const double beta,
                const double* A, const double* B, double* C);

void mnk_4x1(const int M, const int N, const int K, const double alpha, const double beta,
                const double* A, const double* B, double* C);
void mnk_1x4(const int M, const int N, const int K, const double alpha, const double beta,
                const double* A, const double* B, double* C);
void mnk_8x1(const int M, const int N, const int K, const double alpha, const double beta,
                const double* A, const double* B, double* C);
void mnk_1x8(const int M, const int N, const int K, const double alpha, const double beta,
                const double* A, const double* B, double* C);
void mnk_4x4(const int M, const int N, const int K, const double alpha, const double beta,
                const double* A, const double* B, double* C);
void mnk_8x8(const int M, const int N, const int K, const double alpha, const double beta,
                const double* A, const double* B, double* C);

void avx_1x8(const int M, const int N, const int K, const double alpha, const double beta,
                const double* A, const double* B, double* C);

void avx_8x8(const int M, const int N, const int K, const double alpha, const double beta,
                const double* A, const double* B, double* C);
void avx_12x8(const int M, const int N, const int K, const double alpha, const double beta,
                const double* A, const double* B, double* C);
void avx_16x8(const int M, const int N, const int K, const double alpha, const double beta,
                const double* A, const double* B, double* C);
void avx_20x8(const int M, const int N, const int K, const double alpha, const double beta,
                const double* A, const double* B, double* C);
void avx_24x8(const int M, const int N, const int K, const double alpha, const double beta,
                const double* A, const double* B, double* C);

void avx_blis_8x8(const int M, const int N, const int K, const double alpha, const double beta,
                const double* A, const double* B, double* C);
void avx_blis_12x8(const int M, const int N, const int K, const double alpha, const double beta,
                const double* A, const double* B, double* C);
void avx_blis_16x8(const int M, const int N, const int K, const double alpha, const double beta,
                const double* A, const double* B, double* C);
void avx_blis_20x8(const int M, const int N, const int K, const double alpha, const double beta,
                const double* A, const double* B, double* C);
void avx_blis_24x8(const int M, const int N, const int K, const double alpha, const double beta,
                const double* A, const double* B, double* C);

void cublas_dgemm(const int M, const int N, const int K, const double alpha, const double beta,
                const double* A, const double* B, double* C, double* kernel_ms = nullptr);
void cublas_sgemm(const int M, const int N, const int K, const double alpha, const double beta,
                const double* A, const double* B, double* C, double* kernel_ms = nullptr);

void cu_d_ladder(const int M, const int N, const int K, const double alpha, const double beta,
                 const double* A, const double* B, double* C, int ladder, double* kernel_ms = nullptr);
#endif
