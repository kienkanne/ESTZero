#ifndef TRIPLES_BUFFERS_GEMM
#define TRIPLES_BUFFERS_GEMM

#include "cuda_runtime_api.h"
#include "tensor.h"
#include "est_gemm.h"
#include <vector>

inline void to_device(void* dst, const void* src, Index size) {
    cudaMemcpy(dst, src, size * sizeof(double), cudaMemcpyHostToDevice);
}

inline void to_host(void* dst, const void* src, Index size) {
    cudaMemcpy(dst, src, size * sizeof(double), cudaMemcpyDeviceToHost);
}

// Legacy (v3.7.3-v3.7.5): the explicit-GEMM (T) path's buffers (triples_gemm.cc), host Tensors plus mirrored
// device pointers for the cuBLAS-direct route. The live einsum path uses the host-only TriplesBuffer{Spin,Spat}
// (buffers.h) instead.
//
// Allocate all the Tensor buffers that is reused frequently throughout the entire loop at the beginning,
// so no new allocation needed per loop. The allocation of tensors that depend on i,j,k can be saved further
// more by only creating them when the index change, so for outer indices (i,j) they won't needed to be resigned
// every innermost loop (see `triples_gemm.cc`)
struct VikingsBufferSpin {
    std::vector<double> gemm_out;

    Tensor x1, x2, t3,
    Wvvvo_i, Wvvvo_j, Wvvvo_k,
    Wovoo_jk, Wovoo_ik, Wovoo_ji;

    // Pointers on device, needed by t3 builder
    double *d_gemm_out, *d_t2,
        *d_Wvvvo_i, *d_Wvvvo_j, *d_Wvvvo_k,
        *d_Wovoo_jk, *d_Wovoo_ik, *d_Wovoo_ji,

        // Not needed for t3 builder, but is the final accumulation steps which needs GEMM also, so
        // move them to device also.
        *d_x1, *d_x2, *d_t3,
        *d_Woovv_jk, *d_Fov_k, *d_Wvovv_k, *d_Wooov_jk;

    VikingsBufferSpin(const Tensor& t1, const Tensor&t2, const int o, const int v)
        : gemm_out(v*v*v), x1(t1.shape()), x2(t2.shape()), t3({v,v,v}),
        Wvvvo_i({v,v,v}), Wvvvo_j({v,v,v}), Wvvvo_k({v,v,v}),
        Wovoo_jk({o,v}), Wovoo_ik({o,v}), Wovoo_ji({o,v}),
        d_gemm_out(nullptr), d_t2(nullptr),
        d_Wvvvo_i(nullptr), d_Wvvvo_j(nullptr), d_Wvvvo_k(nullptr),
        d_Wovoo_jk(nullptr), d_Wovoo_ik(nullptr), d_Wovoo_ji(nullptr),
        d_x1(nullptr), d_x2(nullptr), d_t3(nullptr),
        d_Woovv_jk(nullptr), d_Fov_k(nullptr), d_Wvovv_k(nullptr), d_Wooov_jk(nullptr) {

        if (gs != GEMMScheme::CUBLAS_DGEMM) return;

        cudaMalloc((void**)&d_t2, t2.size() * sizeof(double));
        cudaMalloc((void**)&d_gemm_out, (v*v*v) * sizeof(double));
        cudaMalloc((void**)&d_Wvvvo_i, (v*v*v) * sizeof(double));
        cudaMalloc((void**)&d_Wvvvo_j, (v*v*v) * sizeof(double));
        cudaMalloc((void**)&d_Wvvvo_k, (v*v*v) * sizeof(double));
        cudaMalloc((void**)&d_Wovoo_jk, (o*v) * sizeof(double));
        cudaMalloc((void**)&d_Wovoo_ik, (o*v) * sizeof(double));
        cudaMalloc((void**)&d_Wovoo_ji, (o*v) * sizeof(double));

        cudaMalloc((void**)&d_x1, x1.size() * sizeof(double));
        cudaMalloc((void**)&d_x2, x2.size() * sizeof(double));
        cudaMalloc((void**)&d_t3, (v*v*v) * sizeof(double));
        cudaMalloc((void**)&d_Woovv_jk, (v*v) * sizeof(double));
        cudaMalloc((void**)&d_Fov_k, v * sizeof(double));
        cudaMalloc((void**)&d_Wvovv_k, (v*v*v) * sizeof(double));
        cudaMalloc((void**)&d_Wooov_jk, (o*v) * sizeof(double));

        // t2 never changes across the (i,j,k) loop, so it's uploaded once here. It's sliced at the beginning
        // only, so it's always contiguous.
        to_device(d_t2, t2.data_ptr(), t2.size());

        // These starts at zero too, so can be ported to device directly.
        to_device(d_x1, x1.data_ptr(), x1.size());
        to_device(d_x2, x2.data_ptr(), x2.size());
    };

    // Raw owning device pointers with no copy schemes defined.
    VikingsBufferSpin(const VikingsBufferSpin&) = delete;
    VikingsBufferSpin& operator=(const VikingsBufferSpin&) = delete;

    ~VikingsBufferSpin() {
        cudaFree(d_t2);
        cudaFree(d_gemm_out);
        cudaFree(d_Wvvvo_i);
        cudaFree(d_Wvvvo_j);
        cudaFree(d_Wvvvo_k);
        cudaFree(d_Wovoo_jk);
        cudaFree(d_Wovoo_ik);
        cudaFree(d_Wovoo_ji);

        cudaFree(d_x1);
        cudaFree(d_x2);
        cudaFree(d_t3);
        cudaFree(d_Woovv_jk);
        cudaFree(d_Fov_k);
        cudaFree(d_Wvovv_k);
        cudaFree(d_Wooov_jk);
    }
};

// Similar idea but for spatial (RHF), which needs additional permuted Wovoo tensors.
struct VikingsBufferSpat {
    std::vector<double> gemm_out;

    Tensor x1, x2, t3, t3_diff, t3_comb1,
    Wvvvo_i, Wvvvo_j, Wvvvo_k,
    Wovoo_jk, Wovoo_ik, Wovoo_ji,
    Wovoo_kj, Wovoo_ki, Wovoo_ij;

    // Same device residents idea as VikingsBufferSpin
    double *d_gemm_out, *d_t2,
        *d_Wvvvo_i, *d_Wvvvo_j, *d_Wvvvo_k,
        *d_Wovoo_jk, *d_Wovoo_ik, *d_Wovoo_ji,
        *d_Wovoo_kj, *d_Wovoo_ki, *d_Wovoo_ij,

        *d_x1, *d_x2, *d_t3_diff, *d_t3_comb1,
        *d_DEoovv_jk, *d_Fov_k, *d_Wvovv_k, *d_Wooov_jk;

    VikingsBufferSpat(const Tensor& t1, const Tensor&t2, const int o, const int v)
        : gemm_out(v*v*v), x1(t1.shape()), x2(t2.shape()),
        t3({v,v,v}), t3_diff({v,v,v}), t3_comb1({v,v,v}),
        Wvvvo_i({v,v,v}), Wvvvo_j({v,v,v}), Wvvvo_k({v,v,v}),
        Wovoo_jk({o,v}), Wovoo_ik({o,v}), Wovoo_ji({o,v}),
        Wovoo_kj({o,v}), Wovoo_ki({o,v}), Wovoo_ij({o,v}),
        d_gemm_out(nullptr), d_t2(nullptr),
        d_Wvvvo_i(nullptr), d_Wvvvo_j(nullptr), d_Wvvvo_k(nullptr),
        d_Wovoo_jk(nullptr), d_Wovoo_ik(nullptr), d_Wovoo_ji(nullptr),
        d_Wovoo_kj(nullptr), d_Wovoo_ki(nullptr), d_Wovoo_ij(nullptr),
        d_x1(nullptr), d_x2(nullptr), d_t3_diff(nullptr), d_t3_comb1(nullptr),
        d_DEoovv_jk(nullptr), d_Fov_k(nullptr), d_Wvovv_k(nullptr), d_Wooov_jk(nullptr) {

        if (gs != GEMMScheme::CUBLAS_DGEMM) return;

        cudaMalloc((void**)&d_t2, t2.size() * sizeof(double));
        cudaMalloc((void**)&d_gemm_out, (v*v*v) * sizeof(double));
        cudaMalloc((void**)&d_Wvvvo_i, (v*v*v) * sizeof(double));
        cudaMalloc((void**)&d_Wvvvo_j, (v*v*v) * sizeof(double));
        cudaMalloc((void**)&d_Wvvvo_k, (v*v*v) * sizeof(double));
        cudaMalloc((void**)&d_Wovoo_jk, (o*v) * sizeof(double));
        cudaMalloc((void**)&d_Wovoo_ik, (o*v) * sizeof(double));
        cudaMalloc((void**)&d_Wovoo_ji, (o*v) * sizeof(double));
        cudaMalloc((void**)&d_Wovoo_kj, (o*v) * sizeof(double));
        cudaMalloc((void**)&d_Wovoo_ki, (o*v) * sizeof(double));
        cudaMalloc((void**)&d_Wovoo_ij, (o*v) * sizeof(double));

        cudaMalloc((void**)&d_x1, x1.size() * sizeof(double));
        cudaMalloc((void**)&d_x2, x2.size() * sizeof(double));
        cudaMalloc((void**)&d_t3_diff, (v*v*v) * sizeof(double));
        cudaMalloc((void**)&d_t3_comb1, (v*v*v) * sizeof(double));
        cudaMalloc((void**)&d_DEoovv_jk, (v*v) * sizeof(double));
        cudaMalloc((void**)&d_Fov_k, v * sizeof(double));
        cudaMalloc((void**)&d_Wvovv_k, (v*v*v) * sizeof(double));
        cudaMalloc((void**)&d_Wooov_jk, (o*v) * sizeof(double));

        to_device(d_t2, t2.data_ptr(), t2.size());
        to_device(d_x1, x1.data_ptr(), x1.size());
        to_device(d_x2, x2.data_ptr(), x2.size());
    };

    VikingsBufferSpat(const VikingsBufferSpat&) = delete;
    VikingsBufferSpat& operator=(const VikingsBufferSpat&) = delete;

    ~VikingsBufferSpat() {
        cudaFree(d_t2);
        cudaFree(d_gemm_out);
        cudaFree(d_Wvvvo_i);
        cudaFree(d_Wvvvo_j);
        cudaFree(d_Wvvvo_k);
        cudaFree(d_Wovoo_jk);
        cudaFree(d_Wovoo_ik);
        cudaFree(d_Wovoo_ji);
        cudaFree(d_Wovoo_kj);
        cudaFree(d_Wovoo_ki);
        cudaFree(d_Wovoo_ij);

        cudaFree(d_x1);
        cudaFree(d_x2);
        cudaFree(d_t3_diff);
        cudaFree(d_t3_comb1);
        cudaFree(d_DEoovv_jk);
        cudaFree(d_Fov_k);
        cudaFree(d_Wvovv_k);
        cudaFree(d_Wooov_jk);
    }
};

#endif
