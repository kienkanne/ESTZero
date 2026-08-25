#include "gemm.h"

// Baseline naive version with explicit loops (precomputed a_flat before final k loop)
void gemm_naive1(const int M, const int N, const int K,
            const double alpha, const double beta,
            const double* A, const double* B, double* C) {
    const int lda = K;
    const int ldb = N;
    const int ldc = N;
    
    
    for (int m=0; m < M; ++m) {
    for (int n=0; n < N; ++n) {
        double* C_ptr = C + (m * ldc + n);
        if (beta == 0.0) { *C_ptr = 0.0;
        } else { *C_ptr *= beta; }

        int a_flat = m * lda;

        for (int k=0; k < K; ++k) {
            *C_ptr += alpha * A[a_flat + k] * B[k * ldb + n];
        }
    }}
}


// Accumulate into sum first, avoid repeatedly reading from c_ptr
void gemm_naive2(const int M, const int N, const int K, 
            const double alpha, const double beta,
            const double* A, const double* B, double* C) {
    const int lda = K; int flat_a = 0;
    const int ldb = N; int flat_b = 0;
    const int ldc = N; int flat_c = 0;
    
    for (int m=0; m < M; ++m) {
    for (int n=0; n < N; ++n) {
        double* C_ptr = C + (m * ldc + n);
        if (beta == 0.0) { *C_ptr = 0.0;
        } else { *C_ptr *= beta; }

        int a_flat = m * lda;
        double sum = 0;

        for (int k=0; k < K; ++k) {
            sum += A[a_flat + k] * B[k * ldb + n];
        }
        *C_ptr += alpha * sum;
    }}
}

// Reorder loop, keeping N in the inner most loop
void gemm_reorder(const int M, const int N, const int K, 
            const double alpha, const double beta,
            const double* A, const double* B, double* C) {
    const int lda = K; int flat_a = 0;
    const int ldb = N; int flat_b = 0;
    const int ldc = N; int flat_c = 0;

    for (int m=0; m < M; ++m) {
    for (int k=0; k < K; ++k) {
        const double& A_in = A[m * lda + k];
        int c_flat = m * ldc;
        int b_flat = k * ldb;
        
        for (int n=0; n < N; ++n) {
            double* C_ptr = C + (c_flat + n);
            if (k==0) { // Does once at the beginning
                if (beta == 0.0) { *C_ptr = 0.0;
                } else { *C_ptr *= beta; }
            }
            *C_ptr += alpha * A_in * B[b_flat + n];
        }
    }}
}
