#include "gemm_all.h"

// Tiles an MR x NR block of C and holds all of them as across the k reduction (innermost index),
// so C is visited exactly once instead of once per k compared to "mkn". It makes accumulators
// explicitly as named local variables (instead of an array) to hint directly that they should live
// in the register (reference: https://github.com/flame/how-to-optimize-gemm).

static inline double mnk_dot_1x1(int K, const double* A_row, const double* B, int ldb, int n) {
    double c = 0.0;
    for (int k = 0; k < K; ++k) c += A_row[k] * B[k * ldb + n];
    return c;
}

void mnk_4x1(int M, int N, int K, double alpha, double beta,
             const double* A, const double* B, double* C) {
    const int lda = K, ldb = N, ldc = N;
    const int M4 = (M / 4) * 4;

    for (int m = 0; m < M4; m += 4) {
    for (int n = 0; n < N; ++n) {
        double c0 = 0.0, c1 = 0.0, c2 = 0.0, c3 = 0.0;
        for (int k = 0; k < K; ++k) {
            const double b_val = B[k * ldb + n];
            c0 += A[(m + 0) * lda + k] * b_val;
            c1 += A[(m + 1) * lda + k] * b_val;
            c2 += A[(m + 2) * lda + k] * b_val;
            c3 += A[(m + 3) * lda + k] * b_val;
        }
        const int idx = m * ldc + n;
        C[idx + 0 * ldc] = (beta == 0.0) ? alpha * c0 : C[idx + 0 * ldc] * beta + alpha * c0;
        C[idx + 1 * ldc] = (beta == 0.0) ? alpha * c1 : C[idx + 1 * ldc] * beta + alpha * c1;
        C[idx + 2 * ldc] = (beta == 0.0) ? alpha * c2 : C[idx + 2 * ldc] * beta + alpha * c2;
        C[idx + 3 * ldc] = (beta == 0.0) ? alpha * c3 : C[idx + 3 * ldc] * beta + alpha * c3;
    }}

    for (int m = M4; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
        const double c = mnk_dot_1x1(K, A + m * lda, B, ldb, n);
        const int idx = m * ldc + n;
        C[idx] = (beta == 0.0) ? alpha * c : C[idx] * beta + alpha * c;
    }}
}

void mnk_1x4(int M, int N, int K, double alpha, double beta,
             const double* A, const double* B, double* C) {
    const int lda = K, ldb = N, ldc = N;
    const int N4 = (N / 4) * 4;

    for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N4; n += 4) {
        double c0 = 0.0, c1 = 0.0, c2 = 0.0, c3 = 0.0;
        const double* A_row = A + m * lda;
        for (int k = 0; k < K; ++k) {
            const double a_val = A_row[k];
            const int b_flat = k * ldb + n;
            c0 += a_val * B[b_flat + 0];
            c1 += a_val * B[b_flat + 1];
            c2 += a_val * B[b_flat + 2];
            c3 += a_val * B[b_flat + 3];
        }
        double* C_ptr = C + m * ldc + n;
        C_ptr[0] = (beta == 0.0) ? alpha * c0 : C_ptr[0] * beta + alpha * c0;
        C_ptr[1] = (beta == 0.0) ? alpha * c1 : C_ptr[1] * beta + alpha * c1;
        C_ptr[2] = (beta == 0.0) ? alpha * c2 : C_ptr[2] * beta + alpha * c2;
        C_ptr[3] = (beta == 0.0) ? alpha * c3 : C_ptr[3] * beta + alpha * c3;
    }}

    for (int m = 0; m < M; ++m) {
    for (int n = N4; n < N; ++n) {
        const double c = mnk_dot_1x1(K, A + m * lda, B, ldb, n);
        const int idx = m * ldc + n;
        C[idx] = (beta == 0.0) ? alpha * c : C[idx] * beta + alpha * c;
    }}
}

void mnk_8x1(int M, int N, int K, double alpha, double beta,
             const double* A, const double* B, double* C) {
    const int lda = K, ldb = N, ldc = N;
    const int M8 = (M / 8) * 8;

    for (int m = 0; m < M8; m += 8) {
    for (int n = 0; n < N; ++n) {
        double c0 = 0.0, c1 = 0.0, c2 = 0.0, c3 = 0.0, c4 = 0.0, c5 = 0.0, c6 = 0.0, c7 = 0.0;
        for (int k = 0; k < K; ++k) {
            const double b_val = B[k * ldb + n];
            c0 += A[(m + 0) * lda + k] * b_val;
            c1 += A[(m + 1) * lda + k] * b_val;
            c2 += A[(m + 2) * lda + k] * b_val;
            c3 += A[(m + 3) * lda + k] * b_val;
            c4 += A[(m + 4) * lda + k] * b_val;
            c5 += A[(m + 5) * lda + k] * b_val;
            c6 += A[(m + 6) * lda + k] * b_val;
            c7 += A[(m + 7) * lda + k] * b_val;
        }
        const int idx = m * ldc + n;
        C[idx + 0 * ldc] = (beta == 0.0) ? alpha * c0 : C[idx + 0 * ldc] * beta + alpha * c0;
        C[idx + 1 * ldc] = (beta == 0.0) ? alpha * c1 : C[idx + 1 * ldc] * beta + alpha * c1;
        C[idx + 2 * ldc] = (beta == 0.0) ? alpha * c2 : C[idx + 2 * ldc] * beta + alpha * c2;
        C[idx + 3 * ldc] = (beta == 0.0) ? alpha * c3 : C[idx + 3 * ldc] * beta + alpha * c3;
        C[idx + 4 * ldc] = (beta == 0.0) ? alpha * c4 : C[idx + 4 * ldc] * beta + alpha * c4;
        C[idx + 5 * ldc] = (beta == 0.0) ? alpha * c5 : C[idx + 5 * ldc] * beta + alpha * c5;
        C[idx + 6 * ldc] = (beta == 0.0) ? alpha * c6 : C[idx + 6 * ldc] * beta + alpha * c6;
        C[idx + 7 * ldc] = (beta == 0.0) ? alpha * c7 : C[idx + 7 * ldc] * beta + alpha * c7;
    }}

    for (int m = M8; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
        const double c = mnk_dot_1x1(K, A + m * lda, B, ldb, n);
        const int idx = m * ldc + n;
        C[idx] = (beta == 0.0) ? alpha * c : C[idx] * beta + alpha * c;
    }}
}

void mnk_1x8(int M, int N, int K, double alpha, double beta,
             const double* A, const double* B, double* C) {
    const int lda = K, ldb = N, ldc = N;
    const int N8 = (N / 8) * 8;

    for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N8; n += 8) {
        double c0 = 0.0, c1 = 0.0, c2 = 0.0, c3 = 0.0, c4 = 0.0, c5 = 0.0, c6 = 0.0, c7 = 0.0;
        const double* A_row = A + m * lda;
        for (int k = 0; k < K; ++k) {
            const double a_val = A_row[k];
            const int b_flat = k * ldb + n;
            c0 += a_val * B[b_flat + 0];
            c1 += a_val * B[b_flat + 1];
            c2 += a_val * B[b_flat + 2];
            c3 += a_val * B[b_flat + 3];
            c4 += a_val * B[b_flat + 4];
            c5 += a_val * B[b_flat + 5];
            c6 += a_val * B[b_flat + 6];
            c7 += a_val * B[b_flat + 7];
        }
        double* C_ptr = C + m * ldc + n;
        C_ptr[0] = (beta == 0.0) ? alpha * c0 : C_ptr[0] * beta + alpha * c0;
        C_ptr[1] = (beta == 0.0) ? alpha * c1 : C_ptr[1] * beta + alpha * c1;
        C_ptr[2] = (beta == 0.0) ? alpha * c2 : C_ptr[2] * beta + alpha * c2;
        C_ptr[3] = (beta == 0.0) ? alpha * c3 : C_ptr[3] * beta + alpha * c3;
        C_ptr[4] = (beta == 0.0) ? alpha * c4 : C_ptr[4] * beta + alpha * c4;
        C_ptr[5] = (beta == 0.0) ? alpha * c5 : C_ptr[5] * beta + alpha * c5;
        C_ptr[6] = (beta == 0.0) ? alpha * c6 : C_ptr[6] * beta + alpha * c6;
        C_ptr[7] = (beta == 0.0) ? alpha * c7 : C_ptr[7] * beta + alpha * c7;
    }}

    for (int m = 0; m < M; ++m) {
    for (int n = N8; n < N; ++n) {
        const double c = mnk_dot_1x1(K, A + m * lda, B, ldb, n);
        const int idx = m * ldc + n;
        C[idx] = (beta == 0.0) ? alpha * c : C[idx] * beta + alpha * c;
    }}
}

void mnk_4x4(int M, int N, int K, double alpha, double beta,
             const double* A, const double* B, double* C) {
    const int lda = K, ldb = N, ldc = N;
    const int M4 = (M / 4) * 4;
    const int N4 = (N / 4) * 4;

    for (int m = 0; m < M4; m += 4) {
    for (int n = 0; n < N4; n += 4) {
        double c00 = 0, c01 = 0, c02 = 0, c03 = 0;
        double c10 = 0, c11 = 0, c12 = 0, c13 = 0;
        double c20 = 0, c21 = 0, c22 = 0, c23 = 0;
        double c30 = 0, c31 = 0, c32 = 0, c33 = 0;

        for (int k = 0; k < K; ++k) {
            const double a0 = A[(m + 0) * lda + k], a1 = A[(m + 1) * lda + k],
                         a2 = A[(m + 2) * lda + k], a3 = A[(m + 3) * lda + k];
            const int b_flat = k * ldb + n;
            const double b0 = B[b_flat + 0], b1 = B[b_flat + 1],
                         b2 = B[b_flat + 2], b3 = B[b_flat + 3];

            c00 += a0 * b0; c01 += a0 * b1; c02 += a0 * b2; c03 += a0 * b3;
            c10 += a1 * b0; c11 += a1 * b1; c12 += a1 * b2; c13 += a1 * b3;
            c20 += a2 * b0; c21 += a2 * b1; c22 += a2 * b2; c23 += a2 * b3;
            c30 += a3 * b0; c31 += a3 * b1; c32 += a3 * b2; c33 += a3 * b3;
        }

        const double crow[4][4] = {{c00, c01, c02, c03}, {c10, c11, c12, c13},
                                    {c20, c21, c22, c23}, {c30, c31, c32, c33}};
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                const int idx = (m + i) * ldc + (n + j);
                const double val = alpha * crow[i][j];
                C[idx] = (beta == 0.0) ? val : C[idx] * beta + val;
            }
        }
    }}

    // remainder columns (N not a multiple of 4), for the fully-blocked M rows
    for (int m = 0; m < M4; m += 4) {
    for (int n = N4; n < N; ++n) {
        double c0 = 0, c1 = 0, c2 = 0, c3 = 0;
        for (int k = 0; k < K; ++k) {
            const double b_val = B[k * ldb + n];
            c0 += A[(m + 0) * lda + k] * b_val;
            c1 += A[(m + 1) * lda + k] * b_val;
            c2 += A[(m + 2) * lda + k] * b_val;
            c3 += A[(m + 3) * lda + k] * b_val;
        }
        const int idx = m * ldc + n;
        C[idx + 0 * ldc] = (beta == 0.0) ? alpha * c0 : C[idx + 0 * ldc] * beta + alpha * c0;
        C[idx + 1 * ldc] = (beta == 0.0) ? alpha * c1 : C[idx + 1 * ldc] * beta + alpha * c1;
        C[idx + 2 * ldc] = (beta == 0.0) ? alpha * c2 : C[idx + 2 * ldc] * beta + alpha * c2;
        C[idx + 3 * ldc] = (beta == 0.0) ? alpha * c3 : C[idx + 3 * ldc] * beta + alpha * c3;
    }}

    // remainder rows (M not a multiple of 4)
    for (int m = M4; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
        const double c = mnk_dot_1x1(K, A + m * lda, B, ldb, n);
        const int idx = m * ldc + n;
        C[idx] = (beta == 0.0) ? alpha * c : C[idx] * beta + alpha * c;
    }}
}

void mnk_8x8(int M, int N, int K, double alpha, double beta,
             const double* A, const double* B, double* C) {
    const int lda = K, ldb = N, ldc = N;
    const int M8 = (M / 8) * 8;
    const int N8 = (N / 8) * 8;

    for (int m = 0; m < M8; m += 8) {
    for (int n = 0; n < N8; n += 8) {
        double c00=0,c01=0,c02=0,c03=0,c04=0,c05=0,c06=0,c07=0;
        double c10=0,c11=0,c12=0,c13=0,c14=0,c15=0,c16=0,c17=0;
        double c20=0,c21=0,c22=0,c23=0,c24=0,c25=0,c26=0,c27=0;
        double c30=0,c31=0,c32=0,c33=0,c34=0,c35=0,c36=0,c37=0;
        double c40=0,c41=0,c42=0,c43=0,c44=0,c45=0,c46=0,c47=0;
        double c50=0,c51=0,c52=0,c53=0,c54=0,c55=0,c56=0,c57=0;
        double c60=0,c61=0,c62=0,c63=0,c64=0,c65=0,c66=0,c67=0;
        double c70=0,c71=0,c72=0,c73=0,c74=0,c75=0,c76=0,c77=0;

        for (int k = 0; k < K; ++k) {
            const double a0 = A[(m+0)*lda+k], a1 = A[(m+1)*lda+k], a2 = A[(m+2)*lda+k], a3 = A[(m+3)*lda+k],
                         a4 = A[(m+4)*lda+k], a5 = A[(m+5)*lda+k], a6 = A[(m+6)*lda+k], a7 = A[(m+7)*lda+k];
            const int b_flat = k * ldb + n;
            const double b0 = B[b_flat+0], b1 = B[b_flat+1], b2 = B[b_flat+2], b3 = B[b_flat+3],
                         b4 = B[b_flat+4], b5 = B[b_flat+5], b6 = B[b_flat+6], b7 = B[b_flat+7];

            c00+=a0*b0; c01+=a0*b1; c02+=a0*b2; c03+=a0*b3; c04+=a0*b4; c05+=a0*b5; c06+=a0*b6; c07+=a0*b7;
            c10+=a1*b0; c11+=a1*b1; c12+=a1*b2; c13+=a1*b3; c14+=a1*b4; c15+=a1*b5; c16+=a1*b6; c17+=a1*b7;
            c20+=a2*b0; c21+=a2*b1; c22+=a2*b2; c23+=a2*b3; c24+=a2*b4; c25+=a2*b5; c26+=a2*b6; c27+=a2*b7;
            c30+=a3*b0; c31+=a3*b1; c32+=a3*b2; c33+=a3*b3; c34+=a3*b4; c35+=a3*b5; c36+=a3*b6; c37+=a3*b7;
            c40+=a4*b0; c41+=a4*b1; c42+=a4*b2; c43+=a4*b3; c44+=a4*b4; c45+=a4*b5; c46+=a4*b6; c47+=a4*b7;
            c50+=a5*b0; c51+=a5*b1; c52+=a5*b2; c53+=a5*b3; c54+=a5*b4; c55+=a5*b5; c56+=a5*b6; c57+=a5*b7;
            c60+=a6*b0; c61+=a6*b1; c62+=a6*b2; c63+=a6*b3; c64+=a6*b4; c65+=a6*b5; c66+=a6*b6; c67+=a6*b7;
            c70+=a7*b0; c71+=a7*b1; c72+=a7*b2; c73+=a7*b3; c74+=a7*b4; c75+=a7*b5; c76+=a7*b6; c77+=a7*b7;
        }

        const double crow[8][8] = {
            {c00,c01,c02,c03,c04,c05,c06,c07}, {c10,c11,c12,c13,c14,c15,c16,c17},
            {c20,c21,c22,c23,c24,c25,c26,c27}, {c30,c31,c32,c33,c34,c35,c36,c37},
            {c40,c41,c42,c43,c44,c45,c46,c47}, {c50,c51,c52,c53,c54,c55,c56,c57},
            {c60,c61,c62,c63,c64,c65,c66,c67}, {c70,c71,c72,c73,c74,c75,c76,c77}};
        for (int i = 0; i < 8; ++i) {
            for (int j = 0; j < 8; ++j) {
                const int idx = (m + i) * ldc + (n + j);
                const double val = alpha * crow[i][j];
                C[idx] = (beta == 0.0) ? val : C[idx] * beta + val;
            }
        }
    }}

    // universal cleanup for anything outside the 8x8 grid (remainder rows and/or columns)
    for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
        if (m < M8 && n < N8) continue;
        const double c = mnk_dot_1x1(K, A + m * lda, B, ldb, n);
        const int idx = m * ldc + n;
        C[idx] = (beta == 0.0) ? alpha * c : C[idx] * beta + alpha * c;
    }}
}
