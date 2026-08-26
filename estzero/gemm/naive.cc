#include "gemm_all.h"

// The rock-bottom baseline: m outer, n middle, k inner. k is strided along B and contiguous
// along A only.
void naive_mnk(int M, int N, int K, double alpha, double beta,
               const double* A, const double* B, double* C) {
    const int lda = K, ldb = N, ldc = N;

    for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
        double* C_ptr = C + (m * ldc + n);
        *C_ptr = (beta == 0.0) ? 0.0 : *C_ptr * beta;

        const int a_flat = m * lda;
        double sum = 0.0;
        for (int k = 0; k < K; ++k) {
            sum += A[a_flat + k] * B[k * ldb + n];
        }
        *C_ptr += alpha * sum;
    }}
}

// Reordered: m outer, k middle, n inner. n is contiguous for both B (fixed k, walk n) and C (fixed m, walk n),
// and A is a scalar reused across the sweep. This approach improved performance significantly with its two
// contiguous walk in the innermost loop. It still needs to go round trip in C, however, which cannot utilize the
// register blocking for accmulators like "mnk" can.
void naive_mkn(int M, int N, int K, double alpha, double beta,
               const double* A, const double* B, double* C) {
    const int lda = K, ldb = N, ldc = N;

    { // init beta outside of main loop, avoid complex logic (similar for other mkn)
        double* C_ptr = C;
        for (int i = 0; i < M * N; ++i) {
            *C_ptr = (beta == 0.0) ? 0.0 : *C_ptr * beta;
            ++C_ptr;
        }
    }

    for (int m = 0; m < M; ++m) {
    for (int k = 0; k < K; ++k) {
        const double A_in = A[m * lda + k];
        const int b_flat = k * ldb;
        double* C_ptr = C + m * ldc;
        for (int n = 0; n < N; ++n) {
            *C_ptr += alpha * A_in * B[b_flat + n];
            ++C_ptr;
        }
    }}
}
