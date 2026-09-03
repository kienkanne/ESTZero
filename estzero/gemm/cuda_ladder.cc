#include <cuda_runtime.h>
#include "gemm_all.h"

// Similar to in `cublas_gemm.cc`
namespace {

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

}  // namespace


// Singel thread, naive baseline version
__global__ void cu1s_naive(const int M, const int N, const int K, const float alpha, const float beta,
				const float* A, const float* B, float* C) {

	const unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int col = blockIdx.y * blockDim.y + threadIdx.y;

	if (row < M && col < N) {
		float tmp = 0.0;
		for (int i = 0; i < K; ++i) {
			tmp += A[row * K + i] * B[i * N + col];
		}
		C[row * N + col] = alpha * tmp + beta * C[row * N + col];
	}
}

__global__ void cu2s_coalesce(const int M, const int N, const int K, const float alpha, const float beta,
                const float* A, const float* B, float* C) {
    // Along the wrap threadIdx.x moves the fastest, so we want to align that with the contiguous
    // axis in both B and C (col), so col gets the + `threadIdx.x` here so threads within a wrap
    // points to a contiguous block of memory (row in C and row in B)
    const unsigned int row = blockIdx.x * blockDim.x + threadIdx.y;
    const unsigned int col = blockIdx.y * blockDim.y + threadIdx.x;

    if (row < M && col < N) {
        float tmp = 0.0;
        for (int i = 0; i < K; ++i) {
            tmp += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = alpha * tmp + beta * C[row * N + col];
    }
}


void cu_s_ladder(const int M, const int N, const int K, const double alpha, const double beta,
                 const double* A, const double* B, double* C, int ladder, double* kernel_ms) {
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

    dim3 block(32, 32);
    dim3 grid(
        (M + block.x - 1) / block.x,
        (N + block.y - 1) / block.y
    );

    Events& ev = get_events();
    if (kernel_ms) cudaEventRecord(ev.start);

	switch (ladder) {
		case 1: cu1s_naive<<<grid, block>>>(M, N, K, alpha, beta, d_A, d_B, d_C); break;
        case 2: cu2s_coalesce<<<grid, block>>>(M, N, K, alpha, beta, d_A, d_B, d_C); break;
	}

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


// Singel thread, naive baseline version
__global__ void cu1d_naive(const int M, const int N, const int K, const double alpha, const double beta,
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

__global__ void cu2d_coalesce(const int M, const int N, const int K, const double alpha, const double beta,
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


void cu_d_ladder(const int M, const int N, const int K, const double alpha, const double beta,
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
		case 1: cu1d_naive<<<grid, block>>>(M, N, K, alpha, beta, d_A, d_B, d_C); break;
        case 2: cu2d_coalesce<<<grid, block>>>(M, N, K, alpha, beta, d_A, d_B, d_C); break;
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
