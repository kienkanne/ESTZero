#include "gemm_all.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>

void cublas_dgemm(const int M, const int N, const int K, const double alpha, const double beta,
                 const double* A, const double* B, double* C) {

    double *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, M * K * sizeof(double));
    cudaMalloc((void**)&d_B, K * N * sizeof(double));
    cudaMalloc((void**)&d_C, M * N * sizeof(double));

    // Contiguous linear copy from host to device 
    cudaMemcpy(d_A, A, M * K * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, K * N * sizeof(double), cudaMemcpyHostToDevice);
    
    // Copy C to device also if beta != 0 to accumulate into it
    if (beta != 0.0) {
        cudaMemcpy(d_C, C, M * N * sizeof(double), cudaMemcpyHostToDevice);
    }

    cublasHandle_t handle;
    cublasCreate(&handle);

    // Pass in as (N, M, K, B, ldb=N, A, lda=K, C, ldc=N)
    // This means computing CT = BT * AT in column-major,
    cublasDgemm(handle, 
                CUBLAS_OP_N, CUBLAS_OP_N, 
                N, M, K, 
                &alpha, 
                d_B, N, // BT
                d_A, K, // AT
                &beta, 
                d_C, N); // CT
                
    // CT in column-major is C in row-major
    cudaMemcpy(C, d_C, M * N * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cublasDestroy(handle);
}
