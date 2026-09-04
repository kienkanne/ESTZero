#include <cuda_runtime.h>
#include "gemm_all.h"

#define BLOCKSIZE 32

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

	const unsigned int row = blockIdx.x * BLOCKSIZE + threadIdx.x;
	const unsigned int col = blockIdx.y * BLOCKSIZE + threadIdx.y;

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
    const unsigned int row = blockIdx.x * BLOCKSIZE + threadIdx.y;
    const unsigned int col = blockIdx.y * BLOCKSIZE + threadIdx.x;

    if (row < M && col < N) {
        float tmp = 0.0;
        for (int i = 0; i < K; ++i) {
            tmp += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = alpha * tmp + beta * C[row * N + col];
    }
}

// The idea here is basically identical to cache blocking KC in GEMM by breaking 
// up K into KC length blocks (BLOCKSIZE length in this case)
__global__ void cu3s_smem_block(const int M, const int N, const int K, const float alpha, const float beta,
                const float* A, const float* B, float* C) {
    const int lda = K; const int ldb = N; const int ldc = N;

    const unsigned int blk_row = blockIdx.x * BLOCKSIZE;
    const unsigned int blk_col = blockIdx.y * BLOCKSIZE;

    const unsigned int thr_row = threadIdx.y;
    const unsigned int thr_col = threadIdx.x;

    const unsigned int global_row = blk_row + thr_row;
    const unsigned int global_col = blk_col + thr_col;

    const float* A_start = A + blk_row * lda;
    const float* B_start = B + blk_col;
    float* C_start = C + blk_row * ldc + blk_col;
    
    // Fixed on block size, so all threads can access this. This is loaded into the
    // shared memory (SM) that doesn't get evicted automatically.
    __shared__ float A_blk[BLOCKSIZE * BLOCKSIZE];
    __shared__ float B_blk[BLOCKSIZE * BLOCKSIZE];

    float tmp = 0.0f;
    for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE) {
        // Safe load for A with boundary checks, creating a contiguous block of data that
        // follows the wrap fastest axis. Basically similar to packing.
        if (global_row < M && (bkIdx + thr_col) < K) {
            A_blk[thr_row * BLOCKSIZE + thr_col] = A_start[thr_row * lda + thr_col];
        } else {
            A_blk[thr_row * BLOCKSIZE + thr_col] = 0.0f; // at the end: the excessive threads
        }

        // Similar for B
        if ((bkIdx + thr_row) < K && global_col < N) {
            B_blk[thr_row * BLOCKSIZE + thr_col] = B_start[thr_row * ldb + thr_col];
        } else {
            B_blk[thr_row * BLOCKSIZE + thr_col] = 0.0f;
        }

        // Wait until all threads finish loading shared memory
        __syncthreads();

        A_start += BLOCKSIZE;
        B_start += BLOCKSIZE * ldb;

        // Dot product computation
        for (int i = 0; i < BLOCKSIZE; ++i) {
            tmp += A_blk[thr_row * BLOCKSIZE + i] * B_blk[i * BLOCKSIZE + thr_col];
        }

        // Wait until all threads finish reading shared memory before the next loop iteration overwrites it
        __syncthreads();
    }

    // Safe write-back to C with boundary checks
    if (global_row < M && global_col < N) {
        C_start[thr_row * ldc + thr_col] = alpha * tmp + beta * C_start[thr_row * ldc + thr_col];
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
        case 3: cu3s_smem_block<<<grid, block>>>(M, N, K, alpha, beta, d_A, d_B, d_C); break;
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

	const unsigned int row = blockIdx.x * BLOCKSIZE + threadIdx.x;
	const unsigned int col = blockIdx.y * BLOCKSIZE + threadIdx.y;

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
    const unsigned int row = blockIdx.x * BLOCKSIZE + threadIdx.y;
    const unsigned int col = blockIdx.y * BLOCKSIZE + threadIdx.x;

    if (row < M && col < N) {
        double tmp = 0.0;
        for (int i = 0; i < K; ++i) {
            tmp += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = alpha * tmp + beta * C[row * N + col];
    }
}

__global__ void cu3d_smem_block(const int M, const int N, const int K, const double alpha, const double beta,
                const double* A, const double* B, double* C) {
    const int lda = K; const int ldb = N; const int ldc = N;

    const unsigned int blk_row = blockIdx.x * BLOCKSIZE;
    const unsigned int blk_col = blockIdx.y * BLOCKSIZE;
    
    const unsigned int thr_row = threadIdx.y;
    const unsigned int thr_col = threadIdx.x;

    const unsigned int global_row = blk_row + thr_row;
    const unsigned int global_col = blk_col + thr_col;

    const double* A_start = A + blk_row * lda;
    const double* B_start = B + blk_col;
    double* C_start = C + blk_row * ldc + blk_col;

    __shared__ double A_blk[BLOCKSIZE * BLOCKSIZE];
    __shared__ double B_blk[BLOCKSIZE * BLOCKSIZE];
    
    double tmp = 0.0;
    for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE) {
        // Safe load for A with boundary checks
        if (global_row < M && (bkIdx + thr_col) < K) {
            A_blk[thr_row * BLOCKSIZE + thr_col] = A_start[thr_row * lda + thr_col];
        } else {
            A_blk[thr_row * BLOCKSIZE + thr_col] = 0.0;
        }

        // Safe load for B with boundary checks
        if ((bkIdx + thr_row) < K && global_col < N) {
            B_blk[thr_row * BLOCKSIZE + thr_col] = B_start[thr_row * ldb + thr_col];
        } else {
            B_blk[thr_row * BLOCKSIZE + thr_col] = 0.0;
        }

        // Wait until all threads finish loading shared memory
        __syncthreads();
        
        A_start += BLOCKSIZE; 
        B_start += BLOCKSIZE * ldb;

        // Dot product computation
        for (int i = 0; i < BLOCKSIZE; ++i) {
            tmp += A_blk[thr_row * BLOCKSIZE + i] * B_blk[i * BLOCKSIZE + thr_col];
        }

        // Wait until all threads finish reading shared memory before the next loop iteration overwrites it
        __syncthreads();
    }

    // Safe write-back to C with boundary checks
    if (global_row < M && global_col < N) {
        C_start[thr_row * ldc + thr_col] = alpha * tmp + beta * C_start[thr_row * ldc + thr_col];
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
        case 3: cu3d_smem_block<<<grid, block>>>(M, N, K, alpha, beta, d_A, d_B, d_C); break;
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
