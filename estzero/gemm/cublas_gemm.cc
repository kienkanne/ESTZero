#include "gemm_all.h"

// Guarded by EST_HAVE_CUBLAS (set in CMakeLists.txt, alongside the CUDA::cublas link) so a clean
// CPU-only build doesn't have to load this in.
#ifdef EST_HAVE_CUBLAS

#include <stdlib.h>
#include <vector>
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

// Single-precision path to explore the performance vs accuracy tradeoff. The double to float 
// casting happens here.
void cublas_sgemm(const int M, const int N, const int K, const double alpha, const double beta,
                   const double* A, const double* B, double* C) {

    std::vector<float> hA(static_cast<size_t>(M) * K);
    std::vector<float> hB(static_cast<size_t>(K) * N);
    std::vector<float> hC(static_cast<size_t>(M) * N);
    for (size_t i = 0; i < hA.size(); ++i) hA[i] = static_cast<float>(A[i]);
    for (size_t i = 0; i < hB.size(); ++i) hB[i] = static_cast<float>(B[i]);
    if (beta != 0.0) {
        for (size_t i = 0; i < hC.size(); ++i) hC[i] = static_cast<float>(C[i]);
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, M * K * sizeof(float));
    cudaMalloc((void**)&d_B, K * N * sizeof(float));
    cudaMalloc((void**)&d_C, M * N * sizeof(float));

    cudaMemcpy(d_A, hA.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, hB.data(), K * N * sizeof(float), cudaMemcpyHostToDevice);
    if (beta != 0.0) {
        cudaMemcpy(d_C, hC.data(), M * N * sizeof(float), cudaMemcpyHostToDevice);
    }

    cublasHandle_t handle;
    cublasCreate(&handle);

    const float alpha_f = static_cast<float>(alpha);
    const float beta_f = static_cast<float>(beta);

    // Same row-major-via-column-major swap trick as cublas_dgemm.
    cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K,
                &alpha_f,
                d_B, N,
                d_A, K,
                &beta_f,
                d_C, N);

    cudaMemcpy(hC.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < hC.size(); ++i) C[i] = static_cast<double>(hC[i]);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cublasDestroy(handle);
}

#else

#include <stdexcept>

void cublas_dgemm(const int, const int, const int, const double, const double,
                  const double*, const double*, double*) {
    throw std::runtime_error(
        "gemm_cublas: this build was compiled without EST_HAVE_CUBLAS -- the cuBLAS block in "
        "CMakeLists.txt is commented out (clean CPU-only build). Un-comment it to use "
        "--gemm-scheme cublas_dgemm.");
}

void cublas_sgemm(const int, const int, const int, const double, const double,
                   const double*, const double*, double*) {
    throw std::runtime_error(
        "gemm_cublas: this build was compiled without EST_HAVE_CUBLAS -- the cuBLAS block in "
        "CMakeLists.txt is commented out (clean CPU-only build). Un-comment it to use "
        "--gemm-scheme cublas_sgemm.");
}

#endif
