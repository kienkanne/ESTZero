#include <cstdio>
#include <cuda_runtime.h>
#include "gemm_all.h"

// Singel thread, naive baseline version
__global__ void cu1_naive(const int M, const int N, const int K, const double alpha, const double beta,
				const double* A, const double* B, double* C) {

	const unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int col = blockIdx.y * blockDim.y + threadIdx.y;

	if (row < M && col < N) {
		double tmp = 0.0;
		for (int i = 0; i < K; ++i) {
			tmp += A[row * K + i] * B[i * N + col];
		}
		C[row * N + col] = alpha * tmp + beta * C[row * N + col];
	}
}

__global__ void cu2_coalescing(const int M, const int N, const int K, const double alpha, const double beta,
                const double* A, const double* B, double* C) {
    // Along the wrap threadIdx.x moves the fastest, so we want to align that with the contiguous
    // axis in both B and C (col), so col gets the + `threadIdx.x` here so threads within a wrap
    // points to a contiguous block of memory (row in C and row in B)
    const unsigned int row = blockIdx.x * blockDim.x + threadIdx.y;
    const unsigned int col = blockIdx.y * blockDim.y + threadIdx.x;

    if (row < M && col < N) {
        double tmp = 0.0;
        for (int i = 0; i < K; ++i) {
            tmp += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = alpha * tmp + beta * C[row * N + col];
    }
}


namespace {
// Persistent cudaEvent_t pair for kernel-only timing (see `kernel_ms` in gemm_all.h), same
// pattern as cublas_gemm.cc's Events -- created once, reused across calls.
struct Events {
    cudaEvent_t start, stop;
    Events() { cudaEventCreate(&start); cudaEventCreate(&stop); }
};
Events& get_events() {
    static Events e;
    return e;
}
}  // namespace

void cuda_ladder(const int M, const int N, const int K, const double alpha, const double beta,
                 const double* A, const double* B, double* C, int ladder, double* kernel_ms) {

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

    dim3 block(32, 32);
    dim3 grid(
        (M + block.x - 1) / block.x,
        (N + block.y - 1) / block.y
    );

    Events& ev = get_events();
    if (kernel_ms) cudaEventRecord(ev.start);

	switch (ladder) {
		case 1: cu1_naive<<<grid, block>>>(M, N, K, alpha, beta, d_A, d_B, d_C); break;
        case 2: cu2_coalescing<<<grid, block>>>(M, N, K, alpha, beta, d_A, d_B, d_C); break;
	}

    if (kernel_ms) {
        cudaEventRecord(ev.stop);
        cudaEventSynchronize(ev.stop);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, ev.start, ev.stop);
        *kernel_ms = ms;
    }

    // CT in column-major is C in row-major
    cudaMemcpy(C, d_C, M * N * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
