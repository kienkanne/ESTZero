// Full multi-level blocked & packed GEMM (NC/KC/MC + register-blocked micro-kernel), following exactly
// the BLIS 5-loop structure from BLISlab, and adopted to row-major storage
// Reference: https://github.com/flame/blislab/blob/master/tutorial.pdf
//
// Loop nest (outermost to innermost), matching the paper's Loop5..Loop0 in BLISlab:
//   nc (steps NC): pack Bc = B[pc:pc+kc_len, nc:nc+nc_len] (targets L3)
//     pc (steps KC): kc-chunk of the K reduction; pc>0 must ACCUMULATE into C, not overwrite
//       mc (steps MC): pack Ac = A[mc:mc+mc_len, pc:pc+kc_len] (targets L2)
//         jr (steps NR): one nr-wide micro-panel of the already-packed Bc
//           ir (steps MR): one mr-wide micro-panel of the already-packed Ac
//             k  (0..kc_len): register-resident rank-1 update (the micro-kernel)
//
// Following the targets in BLISlab tutorial:
// KC=192: B's micro-panel = KC*NR*8B = 192*8*8    = 12KB << 32KB L1
// MC=480: A's block       = MC*KC*8B = 480*192*8  = 720KB < 1MiB L2 (headroom for the rest)
// NC=2048: B's block      = KC*NC*8B = 192*2048*8 = 3MiB << 32MiB L3
// MC=480 is a common multiple of every MR in this family (8,12,16,20,24)

#include <immintrin.h>
#include <vector>
#include <algorithm>
#include "gemm_all.h"

namespace {

constexpr int NR = 8;
constexpr int KC = 192;
constexpr int MC = 480;
constexpr int NC = 2048;

inline double dot_1x1(int K, const double* A_row, const double* B, int ldb, int n) {
    double c = 0.0;
    for (int k = 0; k < K; ++k) c += A_row[k] * B[k * ldb + n];
    return c;
}

// Packs mc_len (multiple of MR) rows x kc_len cols of A, starting at (m0,k0), into Ac as
// sequential MR-wide micro-panels: Ac[(ir/MR)*kc_len*MR + k*MR + i] = A[(m0+ir+i)*lda + k0+k].
void pack_A_block(int MR, int mc_len, int kc_len, const double* A, int lda,
                   int m0, int k0, double* Ac) {
    for (int ir = 0; ir < mc_len; ir += MR) {
        double* dst = Ac + static_cast<size_t>(ir) * kc_len;
        for (int k = 0; k < kc_len; ++k)
            for (int i = 0; i < MR; ++i)
                dst[k * MR + i] = A[(m0 + ir + i) * lda + (k0 + k)];
    }
}

// Packs kc_len rows x nc_len cols (multiple of NR) of B, starting at (k0,n0), into Bc as
// sequential NR-wide micro-panels: Bc[(jr/NR)*kc_len*NR + k*NR + j] = B[(k0+k)*ldb + n0+jr+j].
void pack_B_block(int nc_len, int kc_len, const double* B, int ldb,
                   int k0, int n0, double* Bc) {
    for (int jr = 0; jr < nc_len; jr += NR) {
        double* dst = Bc + static_cast<size_t>(jr) * kc_len;
        for (int k = 0; k < kc_len; ++k)
            for (int j = 0; j < NR; ++j)
                dst[k * NR + j] = B[(k0 + k) * ldb + (n0 + jr + j)];
    }
}

}  // namespace

#define REPEAT8(X)  X(0) X(1) X(2) X(3) X(4) X(5) X(6) X(7)
#define REPEAT12(X) REPEAT8(X)  X(8)  X(9)  X(10) X(11)
#define REPEAT16(X) REPEAT12(X) X(12) X(13) X(14) X(15)
#define REPEAT20(X) REPEAT16(X) X(16) X(17) X(18) X(19)
#define REPEAT24(X) REPEAT20(X) X(20) X(21) X(22) X(23)

#define INIT_C(i) __m512d c##i##_vec = _mm512_setzero_pd();

// a_flat is flat for the Ac_ptr only, already contiguously packed in the desired order
#define COMPUTE_ROW(i) { \
    __m512d a_vec = _mm512_set1_pd(Ac_ptr[a_flat++]); \
    c##i##_vec = _mm512_fmadd_pd(a_vec, b_vec, c##i##_vec); \
}

#define STORE_FIRST_ZERO(i) \
    _mm512_storeu_pd(C_ptr + (i) * ldc, _mm512_mul_pd(alpha_vec, c##i##_vec));
#define STORE_FIRST_BETA(i) { \
    __m512d c_curr = _mm512_loadu_pd(C_ptr + (i) * ldc); \
    _mm512_storeu_pd(C_ptr + (i) * ldc, \
                      _mm512_fmadd_pd(alpha_vec, c##i##_vec, _mm512_mul_pd(beta_vec, c_curr))); \
}
// pc > 0 means that previous kc blocks already applied alpha/beta and are sitting in, so just 
// accumulate this chunk's contribution
#define STORE_ACCUM(i) { \
    __m512d c_curr = _mm512_loadu_pd(C_ptr + (i) * ldc); \
    _mm512_storeu_pd(C_ptr + (i) * ldc, _mm512_fmadd_pd(alpha_vec, c##i##_vec, c_curr)); \
}

#define DEFINE_AVX_BLOCKED(MR) \
void avx_blis_##MR##x8(const int M, const int N, const int K, const double alpha, const double beta, \
                const double* A, const double* B, double* C) { \
    const int lda = K, ldb = N, ldc = N; \
    const int Mblk = (M / MR) * MR; \
    const int Nblk = (N / NR) * NR; \
\
    std::vector<double> Ac(static_cast<size_t>(MC) * KC); \
    std::vector<double> Bc(static_cast<size_t>(NC) * KC); \
\
    for (int nc = 0; nc < Nblk; nc += NC) { /* Loop 5 */ \
        const int nc_len = std::min(NC, Nblk - nc); \
\
        for (int pc = 0; pc < K; pc += KC) { /* Loop 4 */ \
            const int kc_len = std::min(KC, K - pc); \
            const bool first_kc = (pc == 0); \
\
            pack_B_block(nc_len, kc_len, B, ldb, pc, nc, Bc.data()); \
\
            for (int mc = 0; mc < Mblk; mc += MC) { /* Loop 3 */ \
                const int mc_len = std::min(MC, Mblk - mc); \
\
                pack_A_block(MR, mc_len, kc_len, A, lda, mc, pc, Ac.data()); \
\
                /* "MAIN" triple loop nmk */ \
                for (int jr = 0; jr < nc_len; jr += NR) { /* Loop 2 */ \
                    const double* Bc_ptr = Bc.data() + static_cast<size_t>(jr) * kc_len; \
\
                    for (int ir = 0; ir < mc_len; ir += MR) { /* Loop 1 */ \
                        const double* Ac_ptr = Ac.data() + static_cast<size_t>(ir) * kc_len; \
\
                        REPEAT##MR(INIT_C) \
                        for (int k = 0; k < kc_len; ++k) { /* Loop 0 */ \
                            int a_flat = k * MR; \
                            __m512d b_vec = _mm512_loadu_pd(&Bc_ptr[k * NR]); \
                            REPEAT##MR(COMPUTE_ROW) \
                        } \
\
                        double* C_ptr = C + (mc + ir) * ldc + (nc + jr); \
                        __m512d alpha_vec = _mm512_set1_pd(alpha); \
                        if (first_kc) { \
                            if (beta == 0.0) { REPEAT##MR(STORE_FIRST_ZERO) } \
                            else { __m512d beta_vec = _mm512_set1_pd(beta); \
                                   REPEAT##MR(STORE_FIRST_BETA) } \
                        } else { \
                            REPEAT##MR(STORE_ACCUM) \
                        } \
                    } \
                } \
            } \
        } \
    } \
\
    for (int m = 0; m < M; ++m) { \
    for (int n = 0; n < N; ++n) { \
        if (m < Mblk && n < Nblk) continue; \
        const double c = dot_1x1(K, A + m * lda, B, ldb, n); \
        const int idx = m * ldc + n; \
        C[idx] = (beta == 0.0) ? alpha * c : C[idx] * beta + alpha * c; \
    }} \
}

DEFINE_AVX_BLOCKED(8)
DEFINE_AVX_BLOCKED(12)
DEFINE_AVX_BLOCKED(16)
DEFINE_AVX_BLOCKED(20)
DEFINE_AVX_BLOCKED(24)

#undef DEFINE_AVX_BLOCKED
#undef STORE_ACCUM
#undef STORE_FIRST_BETA
#undef STORE_FIRST_ZERO
#undef COMPUTE_ROW
#undef INIT_C
#undef REPEAT24
#undef REPEAT20
#undef REPEAT16
#undef REPEAT12
#undef REPEAT8
