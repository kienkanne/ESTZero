#include "gemm_all.h"

// Guarded by EST_HAVE_CUBLAS (set in CMakeLists.txt, alongside the CUDA::cublas link) so a clean
// CPU-only build doesn't have to load this in.
#ifdef EST_HAVE_CUBLAS

#include <stdlib.h>
#include <cstring>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// The fixes down here are similar to the idea in v3.1.7a, where a persistent buffer is allocated once
// at the beginning to reuse, not to be re-initiated every call.

namespace {

// Persistent global cuBLAS handle, created once and reused for every call instead of a fresh
// cublasCreate/cublasDestroy per GEMM.
cublasHandle_t get_handle() {
    static cublasHandle_t handle = [] {
        cublasHandle_t h;
        cublasCreate(&h);
        return h;
    }();
    return handle;
}

struct Events {
    cudaEvent_t start, stop;
    Events() { cudaEventCreate(&start); cudaEventCreate(&stop); }
};
Events& get_events() {
    static Events e;
    return e;
}

template <typename T>
struct PinnedBuffer {
    T* ptr = nullptr;
    size_t capacity = 0;

    T* get(size_t n) {
        if (n > capacity) {
            if (ptr) cudaFreeHost(ptr);
            cudaMallocHost(reinterpret_cast<void**>(&ptr), n * sizeof(T));
            capacity = n;
        }
        return ptr;
    }
};

} // namespace

void cublas_dgemm(const int M, const int N, const int K, const double alpha, const double beta,
                const double* A, const double* B, double* C, double* kernel_ms,
                bool transA, bool transB) {

    const size_t sizeA = static_cast<size_t>(M) * K;
    const size_t sizeB = static_cast<size_t>(K) * N;
    const size_t sizeC = static_cast<size_t>(M) * N;

    double *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, sizeA * sizeof(double));
    cudaMalloc((void**)&d_B, sizeB * sizeof(double));
    cudaMalloc((void**)&d_C, sizeC * sizeof(double));

    cudaMemcpy(d_A, A, sizeA * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeB * sizeof(double), cudaMemcpyHostToDevice);
    if (beta != 0.0) {
        cudaMemcpy(d_C, C, sizeC * sizeof(double), cudaMemcpyHostToDevice);
    }

    // The leading dimensions of A & B must change according to the transpose (row->column if trans)
    auto TA = (transB) ? CUBLAS_OP_T : CUBLAS_OP_N;
    auto TB = (transA) ? CUBLAS_OP_T : CUBLAS_OP_N;
    const int ldb_call = transB ? K : N;
    const int lda_call = transA ? M : K;

    cublasHandle_t handle = get_handle();

    Events& ev = get_events();
    if (kernel_ms) cudaEventRecord(ev.start);

    cublasDgemm(handle,
                TA, TB,
                N, M, K,
                &alpha,
                d_B, ldb_call, // BT
                d_A, lda_call, // AT
                &beta,
                d_C, N); // CT

    if (kernel_ms) {
        cudaEventRecord(ev.stop);
        cudaEventSynchronize(ev.stop);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, ev.start, ev.stop);
        *kernel_ms = ms;
    }

    // CT in column-major is C in row-major
    cudaMemcpy(C, d_C, sizeC * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// Single-precision path to explore the performance vs accuracy tradeoff. The double to float 
// casting happens here.
void cublas_sgemm(const int M, const int N, const int K, const double alpha, const double beta,
                const double* A, const double* B, double* C, double* kernel_ms,
                bool transA, bool transB) {

    const size_t sizeA = static_cast<size_t>(M) * K;
    const size_t sizeB = static_cast<size_t>(K) * N;
    const size_t sizeC = static_cast<size_t>(M) * N;

    static PinnedBuffer<float> pinA, pinB, pinC;
    float* hA = pinA.get(sizeA);
    float* hB = pinB.get(sizeB);
    float* hC = pinC.get(sizeC);
    for (size_t i = 0; i < sizeA; ++i) hA[i] = static_cast<float>(A[i]);
    for (size_t i = 0; i < sizeB; ++i) hB[i] = static_cast<float>(B[i]);
    if (beta != 0.0) {
        for (size_t i = 0; i < sizeC; ++i) hC[i] = static_cast<float>(C[i]);
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, sizeA * sizeof(float));
    cudaMalloc((void**)&d_B, sizeB * sizeof(float));
    cudaMalloc((void**)&d_C, sizeC * sizeof(float));

    cudaMemcpy(d_A, hA, sizeA * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, hB, sizeB * sizeof(float), cudaMemcpyHostToDevice);
    if (beta != 0.0) {
        cudaMemcpy(d_C, hC, sizeC * sizeof(float), cudaMemcpyHostToDevice);
    }

    auto TA = (transB) ? CUBLAS_OP_T : CUBLAS_OP_N;
    auto TB = (transA) ? CUBLAS_OP_T : CUBLAS_OP_N;
    const int ldb_call = transB ? K : N;
    const int lda_call = transA ? M : K;

    cublasHandle_t handle = get_handle();

    const float alpha_f = static_cast<float>(alpha);
    const float beta_f = static_cast<float>(beta);

    Events& ev = get_events();
    if (kernel_ms) cudaEventRecord(ev.start);

    cublasSgemm(handle,
                TA, TB,
                N, M, K,
                &alpha_f,
                d_B, ldb_call,
                d_A, lda_call,
                &beta_f,
                d_C, N);

    if (kernel_ms) {
        cudaEventRecord(ev.stop);
        cudaEventSynchronize(ev.stop);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, ev.start, ev.stop);
        *kernel_ms = ms;
    }

    cudaMemcpy(hC, d_C, sizeC * sizeof(float), cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < sizeC; ++i) C[i] = static_cast<double>(hC[i]);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

#else

#include <stdexcept>

void cublas_dgemm(const int, const int, const int, const double, const double,
                  const double*, const double*, double*, double*) {
    throw std::runtime_error(
        "gemm_cublas: this build was compiled without EST_HAVE_CUBLAS -- the cuBLAS block in "
        "CMakeLists.txt is commented out (clean CPU-only build). Un-comment it to use "
        "--gemm-scheme cublas_dgemm.");
}

void cublas_sgemm(const int, const int, const int, const double, const double,
                   const double*, const double*, double*, double*) {
    throw std::runtime_error(
        "gemm_cublas: this build was compiled without EST_HAVE_CUBLAS -- the cuBLAS block in "
        "CMakeLists.txt is commented out (clean CPU-only build). Un-comment it to use "
        "--gemm-scheme cublas_sgemm.");
}

#endif
