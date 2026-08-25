#ifndef GEMM_H
#define GEMM_H

// Manually written GEMM optimization ladder. A(M x K) x B(K x N) = C(M x N), all row-major.
// Convention: C = alpha*(A*B) + beta*C.
void gemm_naive1(int M, int N, int K, double alpha, double beta,
              const double* A, const double* B, double* C);
void gemm_naive2(int M, int N, int K, double alpha, double beta,
              const double* A, const double* B, double* C);
void gemm_reorder(int M, int N, int K, double alpha, double beta,
              const double* A, const double* B, double* C);

#endif
