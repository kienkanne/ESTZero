#include <immintrin.h>
#include "gemm_all.h"

static inline double mnk_dot_1x1(int K, const double* A_row, const double* B, int ldb, int n) {
    double c = 0.0;
    for (int k = 0; k < K; ++k) c += A_row[k] * B[k * ldb + n];
    return c;
}

void avx_1x8(int M, int N, int K, double alpha, double beta,
             const double* A, const double* B, double* C) {
    const int lda = K, ldb = N, ldc = N;
    const int N8 = (N / 8) * 8;

    for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N8; n += 8) {
        
        // Initialize the C accumulator vector to zero
        // i.e. C(0,0) -> C(0,7)
        __m512d c_vec = _mm512_setzero_pd(); // replacing c0-c7 declarations

        const double* A_ptr = A + m * lda;
        for (int k = 0; k < K; ++k) {
            // Set the entire vectcor to A_row[k], ready to broadcast over a row in B
            __m512d a_vec = _mm512_set1_pd(A_ptr[k]);
            // Contiguous load from a row (length 8) of B
            __m512d b_vec = _mm512_loadu_pd(&B[k * ldb + n]);
            // c_vec = (a_vec * b_vec) + c_vec
            c_vec = _mm512_fmadd_pd(a_vec, b_vec, c_vec);
        }

        double* C_ptr = C + m * ldc + n;
        __m512d c_new;
        __m512d alpha_vec = _mm512_set1_pd(alpha); // Broadcast alpha, (alpha * c_vec) + res

        if (beta == 0.0) {
            c_new = _mm512_mul_pd(alpha_vec, c_vec);
        } else {
            
            __m512d beta_vec = _mm512_set1_pd(beta); 
            __m512d c_curr = _mm512_loadu_pd(C_ptr); // Loads in the current contiguous C row
            c_new = _mm512_mul_pd(beta_vec, c_curr);
            c_new = _mm512_fmadd_pd(alpha_vec, c_vec, c_new);
        }
    
        // Store the final 8 elements back into C
        _mm512_storeu_pd(C_ptr, c_new);
    }}

    for (int m = 0; m < M; ++m) {
    for (int n = N8; n < N; ++n) {
        const double c = mnk_dot_1x1(K, A + m * lda, B, ldb, n);
        const int idx = m * ldc + n;
        C[idx] = (beta == 0.0) ? alpha * c : C[idx] * beta + alpha * c;
    }}
}


void avx_8x8(int M, int N, int K, double alpha, double beta,
             const double* A, const double* B, double* C) {
    const int lda = K, ldb = N, ldc = N;
    const int M8 = (M / 8) * 8;
    const int N8 = (N / 8) * 8;

    for (int m = 0; m < M8; m += 8) {
    for (int n = 0; n < N8; n += 8) {
        // Initialize the C accumulator vectors (across rows) to zero
        __m512d c0_vec = _mm512_setzero_pd();
        __m512d c1_vec = _mm512_setzero_pd();
        __m512d c2_vec = _mm512_setzero_pd();
        __m512d c3_vec = _mm512_setzero_pd();
        __m512d c4_vec = _mm512_setzero_pd();
        __m512d c5_vec = _mm512_setzero_pd();
        __m512d c6_vec = _mm512_setzero_pd();
        __m512d c7_vec = _mm512_setzero_pd();
            
        const double* A_ptr = A + m * lda;
        for (int k = 0; k < K; ++k) {
            __m512d b_vec = _mm512_loadu_pd(&B[k * ldb + n]);

            // ~ stride along m, get 1 value of A, broadcast multiply B to a C row
            __m512d a0_vec = _mm512_set1_pd(A_ptr[0 * lda + k]);
            c0_vec = _mm512_fmadd_pd(a0_vec, b_vec, c0_vec);

            __m512d a1_vec = _mm512_set1_pd(A_ptr[1 * lda + k]);
            c1_vec = _mm512_fmadd_pd(a1_vec, b_vec, c1_vec);

            __m512d a2_vec = _mm512_set1_pd(A_ptr[2 * lda + k]);
            c2_vec = _mm512_fmadd_pd(a2_vec, b_vec, c2_vec);

            __m512d a3_vec = _mm512_set1_pd(A_ptr[3 * lda + k]);
            c3_vec = _mm512_fmadd_pd(a3_vec, b_vec, c3_vec);

            __m512d a4_vec = _mm512_set1_pd(A_ptr[4 * lda + k]);
            c4_vec = _mm512_fmadd_pd(a4_vec, b_vec, c4_vec);

            __m512d a5_vec = _mm512_set1_pd(A_ptr[5 * lda + k]);
            c5_vec = _mm512_fmadd_pd(a5_vec, b_vec, c5_vec);

            __m512d a6_vec = _mm512_set1_pd(A_ptr[6 * lda + k]);
            c6_vec = _mm512_fmadd_pd(a6_vec, b_vec, c6_vec);

            __m512d a7_vec = _mm512_set1_pd(A_ptr[7 * lda + k]);
            c7_vec = _mm512_fmadd_pd(a7_vec, b_vec, c7_vec);
        }

        double* C_ptr = C + m * ldc + n;
        __m512d alpha_vec = _mm512_set1_pd(alpha); // Broadcast alpha, (alpha * c_vec) + res

        __m512d c0_new, c1_new, c2_new, c3_new, c4_new, c5_new, c6_new, c7_new;
        if (beta == 0.0) {
            c0_new = _mm512_mul_pd(alpha_vec, c0_vec);
            c1_new = _mm512_mul_pd(alpha_vec, c1_vec);
            c2_new = _mm512_mul_pd(alpha_vec, c2_vec);
            c3_new = _mm512_mul_pd(alpha_vec, c3_vec);
            c4_new = _mm512_mul_pd(alpha_vec, c4_vec);
            c5_new = _mm512_mul_pd(alpha_vec, c5_vec);
            c6_new = _mm512_mul_pd(alpha_vec, c6_vec);
            c7_new = _mm512_mul_pd(alpha_vec, c7_vec);
        } else { 
            __m512d beta_vec = _mm512_set1_pd(beta);

            __m512d c0_curr = _mm512_loadu_pd(C_ptr + 0 * ldc); // Loads in the current contiguous C row
            c0_new = _mm512_mul_pd(beta_vec, c0_curr);
            c0_new = _mm512_fmadd_pd(alpha_vec, c0_vec, c0_new);

            __m512d c1_curr = _mm512_loadu_pd(C_ptr + 1 * ldc); // Loads in the current contiguous C row
            c1_new = _mm512_mul_pd(beta_vec, c1_curr);
            c1_new = _mm512_fmadd_pd(alpha_vec, c1_vec, c1_new);

            __m512d c2_curr = _mm512_loadu_pd(C_ptr + 2 * ldc); // Loads in the current contiguous C row
            c2_new = _mm512_mul_pd(beta_vec, c2_curr);
            c2_new = _mm512_fmadd_pd(alpha_vec, c2_vec, c2_new);

            __m512d c3_curr = _mm512_loadu_pd(C_ptr + 3 * ldc); // Loads in the current contiguous C row
            c3_new = _mm512_mul_pd(beta_vec, c3_curr);
            c3_new = _mm512_fmadd_pd(alpha_vec, c3_vec, c3_new);

            __m512d c4_curr = _mm512_loadu_pd(C_ptr + 4 * ldc); // Loads in the current contiguous C row
            c4_new = _mm512_mul_pd(beta_vec, c4_curr);
            c4_new = _mm512_fmadd_pd(alpha_vec, c4_vec, c4_new);

            __m512d c5_curr = _mm512_loadu_pd(C_ptr + 5 * ldc); // Loads in the current contiguous C row
            c5_new = _mm512_mul_pd(beta_vec, c5_curr);
            c5_new = _mm512_fmadd_pd(alpha_vec, c5_vec, c5_new);

            __m512d c6_curr = _mm512_loadu_pd(C_ptr + 6 * ldc); // Loads in the current contiguous C row
            c6_new = _mm512_mul_pd(beta_vec, c6_curr);
            c6_new = _mm512_fmadd_pd(alpha_vec, c6_vec, c6_new);

            __m512d c7_curr = _mm512_loadu_pd(C_ptr + 7 * ldc); // Loads in the current contiguous C row
            c7_new = _mm512_mul_pd(beta_vec, c7_curr);
            c7_new = _mm512_fmadd_pd(alpha_vec, c7_vec, c7_new);
        }
    
        // Store the final 8x8 elements back into C
        _mm512_storeu_pd(C_ptr + 0 * ldc, c0_new);
        _mm512_storeu_pd(C_ptr + 1 * ldc, c1_new);
        _mm512_storeu_pd(C_ptr + 2 * ldc, c2_new);
        _mm512_storeu_pd(C_ptr + 3 * ldc, c3_new);
        _mm512_storeu_pd(C_ptr + 4 * ldc, c4_new);
        _mm512_storeu_pd(C_ptr + 5 * ldc, c5_new);
        _mm512_storeu_pd(C_ptr + 6 * ldc, c6_new);
        _mm512_storeu_pd(C_ptr + 7 * ldc, c7_new);
    }}

    for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
        if (m < M8 && n < N8) continue;
        const double c = mnk_dot_1x1(K, A + m * lda, B, ldb, n);
        const int idx = m * ldc + n;
        C[idx] = (beta == 0.0) ? alpha * c : C[idx] * beta + alpha * c;
    }}
}