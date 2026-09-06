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
    const unsigned int row = blockIdx.x * BLOCKSIZE + threadIdx.x / BLOCKSIZE;
    const unsigned int col = blockIdx.y * BLOCKSIZE + threadIdx.x % BLOCKSIZE;

    if (row < M && col < N) {
        float tmp = 0.0;
        for (int i = 0; i < K; ++i) {
            tmp += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = alpha * tmp + beta * C[row * N + col];
    }
}

#define BM 32
#define BN 32
#define BK 32

// The idea here is basically identical to cache blocking KC in GEMM by breaking 
// up K into KC length blocks (BLOCKSIZE length in this case)
__global__ void cu3s_smem_block(const int M, const int N, const int K, const float alpha, const float beta,
                const float* A, const float* B, float* C) {
    const int lda = K; const int ldb = N; const int ldc = N;

    const unsigned int blk_row = blockIdx.x * BLOCKSIZE;
    const unsigned int blk_col = blockIdx.y * BLOCKSIZE;

    const unsigned int thr_row = threadIdx.x / BLOCKSIZE;
    const unsigned int thr_col = threadIdx.x % BLOCKSIZE;

    const unsigned int global_row = blk_row + thr_row;
    const unsigned int global_col = blk_col + thr_col;

    const float* A_start = A + blk_row * lda;
    const float* B_start = B + blk_col;
    float* C_start = C + blk_row * ldc + blk_col;
    
    // Fixed on block size, so all threads can access this. This is loaded into the
    // shared memory (SM) that doesn't get evicted automatically.
    __shared__ float A_blk[BM * BK];
    __shared__ float B_blk[BK * BN];

    float tmp = 0.0f;
    for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {
        // Safe load for A with boundary checks, creating a contiguous block of data that
        // follows the wrap fastest axis. Basically similar to packing.
        if (global_row < M && (bkIdx + thr_col) < K) {
            A_blk[thr_row * BK + thr_col] = A_start[thr_row * lda + thr_col];
        } else {
            A_blk[thr_row * BK + thr_col] = 0.0f; // at the end: the excessive threads
        }

        // Similar for B
        if ((bkIdx + thr_row) < K && global_col < N) {
            B_blk[thr_row * BN + thr_col] = B_start[thr_row * ldb + thr_col];
        } else {
            B_blk[thr_row * BN + thr_col] = 0.0f;
        }

        // Wait until all threads finish loading shared memory
        __syncthreads();

        A_start += BK;
        B_start += BN * ldb;

        // Dot product computation
        for (int i = 0; i < BK; ++i) {
            tmp += A_blk[thr_row * BK + i] * B_blk[i * BN + thr_col];
        }

        // Wait until all threads finish reading shared memory before the next loop iteration overwrites it
        __syncthreads();
    }

    // Safe write-back to C with boundary checks
    if (global_row < M && global_col < N) {
        C_start[thr_row * ldc + thr_col] = alpha * tmp + beta * C_start[thr_row * ldc + thr_col];
    }
}

#undef BM
#undef BN
#undef BK

#define BM 64
#define BN 64
#define BK 8
#define TM 8

// The idea is that each thread still only populates one value in A/B, but with this choice of BLOCKSIZE
// each thread can fill 8 C titles.
__global__ void cu4s_thread_mul(const int M, const int N, const int K, const float alpha, const float beta,
                const float* A, const float* B, float* C) {
    const int lda = K; const int ldb = N; const int ldc = N;

    const unsigned int blk_row = blockIdx.x * BM;
    const unsigned int blk_col = blockIdx.y * BN;

    // Renaming so it's clear and consistent
    const unsigned int innerRowA = threadIdx.x / BK;
    const unsigned int innerColA = threadIdx.x % BK;
    
    // B and C shares the same ldb, so their inner rows/cols match
    const unsigned int innerRowBC = threadIdx.x / BN;
    const unsigned int innerColBC = threadIdx.x % BN;
    
    // Global as to C, but can be used to check bounds for both A and B
    const unsigned int global_row = blk_row + innerRowBC * TM;
    const unsigned int global_col = blk_col + innerColBC;

    const float* A_start = A + blk_row * lda;
    const float* B_start = B + blk_col;
    float* C_start = C + blk_row * ldc + blk_col;

    __shared__ float A_blk[BM * BK];
    __shared__ float B_blk[BK * BN];

    float thread_results[TM] = {};

    for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {
        if ((blk_row + innerRowA) < M && (bkIdx + innerColA) < K) {
            A_blk[innerRowA * BK + innerColA] = A_start[innerRowA * lda + innerColA];
        } else {
            A_blk[innerRowA * BK + innerColA] = 0.0f; // at the end: the excessive threads
        }

        if ((bkIdx + innerRowBC) < K && global_col < N) {
            B_blk[innerRowBC * BN + innerColBC] = B_start[innerRowBC * ldb + innerColBC];
        } else {
            B_blk[innerRowBC * BN + innerColBC] = 0.0f;
        }
        // Wait until all threads finish loading shared memory
        __syncthreads();

        A_start += BK;
        B_start += BK * ldb;

        // Dot product computation
        for (int i = 0; i < BK; ++i) {
            float B_val = B_blk[i * BN + innerColBC]; // Fetch once, reuse as move down the column
            for (int row_tm = 0; row_tm < TM; ++row_tm) {
                thread_results[row_tm] += A_blk[(innerRowBC * TM + row_tm) * BK + i] * B_val;
            }
        }
        // Wait until all threads finish reading shared memory before the next loop iteration overwrites it
        __syncthreads();
    }

    // Safe write-back to C with boundary checks
    // inner for B, but C can basically reuse
    if (global_row < M && global_col < N) {
        for (int row_tm = 0; row_tm < TM; ++row_tm) {
            if (global_row + row_tm >= M) break;

            C_start[(innerRowBC * TM + row_tm) * ldc + innerColBC] = 
                alpha * thread_results[row_tm] 
                + beta * C_start[(innerRowBC * TM + row_tm) * ldc + innerColBC];
        }
        
    }
}


// The idea is expanded, each thread now loads in MULTIPLE values.
#undef BM
#undef BN
#undef BK
#undef TM

#define BM 64
#define BN 64
#define BK 4
#define TM 4
#define TN 4

// Kernel 6 in the original tutorial is baked in here too, it is literally a single line change to
// transpose A_blk upon creation so that the regsiter loading step is contiguous. Not worth another 100 lines.
__global__ void cu5s_2d_tile(const int M, const int N, const int K, const float alpha, const float beta,
                const float* A, const float* B, float* C) {
    const int lda = K; const int ldb = N; const int ldc = N;

    // 256 (each loads 8, twice as much as before, so we only need 512/2=256 threads per block now)
    const uint numthreads = blockDim.x; 
    // these strides allow us to sweep all of A_blk and B_blk using threadIdx
    const uint strideA = blockDim.x / BK; // 32
    const uint strideB = blockDim.x / BN; // 4
    // these now mean WITHIN the assigned C block (4x4) of THIS thread
    const uint threadRow = threadIdx.x / (BN / TN); // ranges 0, BM/TM
    const uint threadCol = threadIdx.x % (BN / TN); // ranges 0, BN/TN

    const unsigned int blk_row = blockIdx.x * BM;
    const unsigned int blk_col = blockIdx.y * BN;

    const unsigned int innerRowA = threadIdx.x / BK;
    const unsigned int innerColA = threadIdx.x % BK;
    
    const unsigned int innerRowBC = threadIdx.x / BN;
    const unsigned int innerColBC = threadIdx.x % BN;
    
    const float* A_start = A + blk_row * lda;
    const float* B_start = B + blk_col;
    float* C_start = C + blk_row * ldc + blk_col;

    __shared__ float A_blk[BM * BK];
    __shared__ float B_blk[BK * BN];

    float thread_results[TM * TN] = {};
    
    float regM[TM] = {0.0};
    float regN[TN] = {0.0};

    for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {
        // populate the SMEM caches
        for (uint offset = 0; offset < BM; offset += strideA) {
            if ((blk_row + innerRowA + offset) < M && (bkIdx + innerColA) < K) {
                A_blk[(innerColA + offset) * BM + innerRowA] = // TRANSPOSE!!!!
                    A_start[(innerRowA + offset) * K + innerColA]; 
            } else {
                A_blk[(innerColA + offset) * BM + innerRowA] = 0; // and here
            }
        }
        for (uint offset = 0; offset < BK; offset += strideB) {
            if ((bkIdx + innerRowBC + offset) < K && (blk_col + innerColBC) < N) {
                B_blk[(innerRowBC + offset) * BN + innerColBC] =
                    B_start[(innerRowBC + offset) * N + innerColBC];
            } else {
                B_blk[(innerRowBC + offset) * BN + innerColBC] = 0;
            }
        }
        __syncthreads();

        A_start += BK;
        B_start += BK * ldb; 

        // calculate per-thread results
        for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
            // Similar to register blocking style in CPU, we get the two mini vectors, 
            for (uint i = 0; i < TM; ++i) {
                regM[i] = A_blk[dotIdx * BM + threadRow * TM + i]; // this right here WAS non-contiguous: (threadRow * TM + i) * BK + dotIdx
            }
            for (uint i = 0; i < TN; ++i) {
                regN[i] = B_blk[dotIdx * BN + threadCol * TN + i];   
            }
            // and then perform outer product
            for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
                for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
                thread_results[resIdxM * TN + resIdxN] +=
                    regM[resIdxM] * regN[resIdxN];
                }
            }
        }
        __syncthreads();
    }

    const unsigned int global_row = blk_row + threadRow * TM;
    const unsigned int global_col = blk_col + threadCol * TN;
    // Fill data with bounds check
    for (int resIdxM = 0; resIdxM < TM; ++resIdxM) {
        if (global_row + resIdxM >= M) break;
        for (int resIdxN = 0; resIdxN < TN; ++resIdxN) {
            if (global_col + resIdxN >= N) continue;
            int idx = (threadRow * TM + resIdxM) * ldc + (threadCol * TN + resIdxN);
            C_start[idx] = alpha * thread_results[resIdxM * TN + resIdxN] + beta * C_start[idx];
        }
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
    
    dim3 grid64((M + BM - 1) / BM,(N + BN - 1) / BN);

    Events& ev = get_events();
    if (kernel_ms) cudaEventRecord(ev.start);

	switch (ladder) {
		case 1: cu1s_naive<<<grid, block>>>(M, N, K, alpha, beta, d_A, d_B, d_C); break;
        case 2: cu2s_coalesce<<<grid, dim3(32*32)>>>(M, N, K, alpha, beta, d_A, d_B, d_C); break;
        case 3: cu3s_smem_block<<<grid, dim3(32*32)>>>(M, N, K, alpha, beta, d_A, d_B, d_C); break;
        case 4: cu4s_thread_mul<<<grid64, dim3(64*8)>>>(M, N, K, alpha, beta, d_A, d_B, d_C); break;
        case 5: cu5s_2d_tile<<<grid64, dim3((BM*BN)/(TM*TN))>>>(M, N, K, alpha, beta, d_A, d_B, d_C); break;
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
    const unsigned int row = blockIdx.x * BLOCKSIZE + threadIdx.x / BLOCKSIZE;
    const unsigned int col = blockIdx.y * BLOCKSIZE + threadIdx.x % BLOCKSIZE;

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
    
    const unsigned int thr_row = threadIdx.x / BLOCKSIZE;
    const unsigned int thr_col = threadIdx.x % BLOCKSIZE;

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
        case 2: cu2d_coalesce<<<grid, dim3(32*32)>>>(M, N, K, alpha, beta, d_A, d_B, d_C); break;
        case 3: cu3d_smem_block<<<grid, dim3(32*32)>>>(M, N, K, alpha, beta, d_A, d_B, d_C); break;
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
