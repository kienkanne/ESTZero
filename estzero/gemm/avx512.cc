#include <immintrin.h>
#include "gemm_all.h"
#include "vector"

static inline double mnk_dot_1x1(int K, const double* A_row, const double* B, int ldb, int n) {
    double c = 0.0;
    for (int k = 0; k < K; ++k) c += A_row[k] * B[k * ldb + n];
    return c;
}

void avx_1x8(const int M, const int N, const int K, const double alpha, const double beta,
                const double* A, const double* B, double* C) {
    const int lda = K, ldb = N, ldc = N;
    const int Nblk = (N / 8) * 8;

    for (int m = 0; m < M; ++m) {
    for (int n = 0; n < Nblk; n += 8) {

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
    for (int n = Nblk; n < N; ++n) {
        const double c = mnk_dot_1x1(K, A + m * lda, B, ldb, n);
        const int idx = m * ldc + n;
        C[idx] = (beta == 0.0) ? alpha * c : C[idx] * beta + alpha * c;
    }}
}


// avx_Rx8 family: MR x 8 tile (MR in 8,12,16,20,24), NR fixed at 8 (one zmm register wide). 
// Each MR only differs by how many zmm registers they block, so the body is written once,
// and with the help of macros, the different versions can be written easily without mistake
//
// REPEATn is built by chaining onto the previous REPEATn because you can't write a loop for 
// macro, so adding additional avx_Rx8 needs to define the new REPEATn here first

#define REPEAT8(X)  X(0) X(1) X(2) X(3) X(4) X(5) X(6) X(7)
#define REPEAT12(X) REPEAT8(X)  X(8)  X(9)  X(10) X(11)
#define REPEAT16(X) REPEAT12(X) X(12) X(13) X(14) X(15)
#define REPEAT20(X) REPEAT16(X) X(16) X(17) X(18) X(19)
#define REPEAT24(X) REPEAT20(X) X(20) X(21) X(22) X(23)

// One __m512d accumulator per C row (8 entries)
#define INIT_C(i) __m512d c##i##_vec = _mm512_setzero_pd();

// Within the innermost loop k, each row's FMA against the shared b_vec loaded once. This loops
// MR times for MR registers.
#define COMPUTE_ROW(i) { \
    __m512d a_vec = _mm512_set1_pd(A_packed[a_flat++]); /* A_packed is contiguous in MR, so ++ */ \
    c##i##_vec = _mm512_fmadd_pd(a_vec, b_vec, c##i##_vec); \
}

#define STORE_ZERO_BETA(i) \
    _mm512_storeu_pd(C_ptr + (i) * ldc, _mm512_mul_pd(alpha_vec, c##i##_vec));

#define STORE_WITH_BETA(i) { \
    __m512d c_curr = _mm512_loadu_pd(C_ptr + (i) * ldc); \
    __m512d c_new = _mm512_fmadd_pd(alpha_vec, c##i##_vec, _mm512_mul_pd(beta_vec, c_curr)); \
    _mm512_storeu_pd(C_ptr + (i) * ldc, c_new); \
}

#define DEFINE_AVX_RX8(MR) \
void avx_##MR##x8(const int M, const int N, const int K, const double alpha, const double beta, \
                const double* A, const double* B, double* C) { \
    const int lda = K, ldb = N, ldc = N; \
    const int Mblk = (M / MR) * MR; \
    const int Nblk = (N / 8) * 8; \
    /* Initialize the vector buffers for packed A, B once to be reused */ \
    std::vector<double> A_packed(MR * K);\
    std::vector<double> B_packed(static_cast<size_t>(Nblk / 8) * K * 8); /* integer division */ \
\
    /* Packs K rows (row with length NR) of B (row-major, starting at n0) in full k length, into a scratch
    buffer laid out contiguously STILL in NR (~row major). AND THEN, in the next n, do the same. This basically
    means that, for example, position (k_last, 8) is connected to (0, 9) when NR=8. Goes zigzag (left, down right)
    with the width being NR, then finally jumps up and do it again, across Nblk. B_packed is more PACKED (big) than
    A_packed, the size being N_blk x 8 x K */ \
\
    for (int n = 0; n < Nblk; n += 8) { \
        double* dst = B_packed.data() + (n / 8) * K * 8; \
        for (int k = 0; k < K; ++k) { \
        for (int j = 0; j < 8; ++j) { \
            dst[k * 8 + j] = B[k * ldb + j + n]; \
        } \
    }} \
\
    /* MAIN LOOP */ \
    for (int m = 0; m < Mblk; m += MR) { \
    const double* A_ptr = A + m * lda; \
\
    /* Packs MR rows of A (row-major, starting at row m0) in full k length, into a scratch
    buffer laid out contiguously in MR (~column major), so that the innermost loop advancing in
    MR is contiguous. Done inside the m loop, needs the current m. */ \
    for (int k = 0; k < K; ++k) { \
    for (int i = 0; i < MR; ++i) { \
        A_packed[k * MR + i] = A[(m + i) * lda + k]; \
    }} \
\
        for (int n = 0; n < Nblk; n += 8) { \
            REPEAT##MR(INIT_C) \
\
            /* Get the correct B block based on current n */ \
            const double* B_ptr = B_packed.data() + (n / 8) * K * 8; \
            for (int k = 0; k < K; ++k) { \
    \
                int a_flat = k * MR; \
                __m512d b_vec = _mm512_loadu_pd(&B_ptr[k * 8]); \
                REPEAT##MR(COMPUTE_ROW) /* A_packed is used here */ \
            } \
    \
            double* C_ptr = C + m * ldc + n; \
            __m512d alpha_vec = _mm512_set1_pd(alpha); \
            if (beta == 0.0) { \
                REPEAT##MR(STORE_ZERO_BETA) \
            } else { \
                __m512d beta_vec = _mm512_set1_pd(beta); \
                REPEAT##MR(STORE_WITH_BETA) \
            } \
        } \
    } \
\
    for (int m = 0; m < M; ++m) { \
    for (int n = 0; n < N; ++n) { \
        if (m < Mblk && n < Nblk) continue; \
        const double c = mnk_dot_1x1(K, A + m * lda, B, ldb, n); \
        const int idx = m * ldc + n; \
        C[idx] = (beta == 0.0) ? alpha * c : C[idx] * beta + alpha * c; \
    }} \
}

DEFINE_AVX_RX8(8)
DEFINE_AVX_RX8(12)
DEFINE_AVX_RX8(16)
DEFINE_AVX_RX8(20)
DEFINE_AVX_RX8(24)

#undef DEFINE_AVX_RX8
#undef STORE_WITH_BETA
#undef STORE_ZERO_BETA
#undef COMPUTE_ROW
#undef INIT_C
#undef REPEAT24
#undef REPEAT20
#undef REPEAT16
#undef REPEAT12
#undef REPEAT8
