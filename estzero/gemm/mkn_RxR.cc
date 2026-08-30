#include "gemm_all.h"

// MR = R rows of A/C are blocked together, reused against each row of B before moving on to 
// the next k. This is the pure savings on traffic by reusing the B, sweeping through it M/R
// times instead of M times. The problem it introduces, however, is that it is still writing
// to C k times, and the writing is in a 4x1 C block, which is non-contiguous. Increasing R
// is less loops overall, but the writing step just gets worse, and it still k times.


static inline void beta_init(int M, int N, double beta, double* C) {
    double* C_ptr = C;
    for (int i = 0; i < M * N; ++i) {
        *C_ptr = (beta == 0.0) ? 0.0 : *C_ptr * beta;
        ++C_ptr;
    }
}

// Single-row (MR=1) cleanup sweep, no unroll. This is used to finish off any rows 
// left over when M isn't a multiple of MR.
static inline void mkn_row_1x1(int N, int K, double alpha, const double* A_row, int lda,
                                const double* B, int ldb, double* C_row, int ldc) {
    for (int k = 0; k < K; ++k) {
        const double A_in = A_row[k];
        const int b_flat = k * ldb;
        double* C_ptr = C_row;
        for (int n = 0; n < N; ++n) {
            *C_ptr += alpha * A_in * B[b_flat + n];
            ++C_ptr;
        }
    }
}

void mkn_4x1(const int M, const int N, const int K, const double alpha, const double beta,
                const double* A, const double* B, double* C) {
    const int lda = K, ldb = N, ldc = N;
    beta_init(M, N, beta, C);

    const int M4 = (M / 4) * 4;
    for (int m = 0; m < M4; m += 4) {
    for (int k = 0; k < K; ++k) {
        double A_4x1[4]; // Create a contiguous 4x1 block of A to reuse 
        for (int i = 0; i < 4; ++i) A_4x1[i] = A[(m + i) * lda + k];
        const int b_flat = k * ldb;

        for (int i = 0; i < 4; ++i) {
            double* C_ptr = C + (m + i) * ldc;
            for (int n = 0; n < N; ++n) {
                *C_ptr += alpha * A_4x1[i] * B[b_flat + n];
                ++C_ptr;
            }
        }
    }}

    for (int m = M4; m < M; ++m) {
        mkn_row_1x1(N, K, alpha, A + m * lda, lda, B, ldb, C + m * ldc, ldc);
    }
}

void mkn_8x1(const int M, const int N, const int K, const double alpha, const double beta,
                const double* A, const double* B, double* C) {
    const int lda = K, ldb = N, ldc = N;
    beta_init(M, N, beta, C);

    const int M8 = (M / 8) * 8;
    for (int m = 0; m < M8; m += 8) {
    for (int k = 0; k < K; ++k) {
        double A_8x1[8];
        for (int i = 0; i < 8; ++i) A_8x1[i] = A[(m + i) * lda + k];
        const int b_flat = k * ldb;

        for (int i = 0; i < 8; ++i) {
            double* C_ptr = C + (m + i) * ldc;
            for (int n = 0; n < N; ++n) {
                *C_ptr += alpha * A_8x1[i] * B[b_flat + n];
                ++C_ptr;
            }
        }
    }}

    for (int m = M8; m < M; ++m) {
        mkn_row_1x1(N, K, alpha, A + m * lda, lda, B, ldb, C + m * ldc, ldc);
    }
}
