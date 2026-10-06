#ifndef TRIPLES_GEMM
#define TRIPLES_GEMM

#include "tensor.h"
#include "buffers_gemm.h"

// Legacy (v3.7.1-v3.7.5) explicit-GEMM (T) path: call_gemm/cublas_direct_dgemm instead of einsum. Same signatures
// as the live calc_pert_triples_{spin,spat} (dispatch.h), kept for A-B comparison, not called from cc_driver.h.
Tensor t3c_ijk_spin_gemm(const Tensor& t2, VikingsBufferSpin& vb, const int o, const int v,
               const Tensor& F_occ, const Tensor& F_virt, int i, int j, int k);

Tensor t3c_ijk_spat_gemm(const Tensor& t2, VikingsBufferSpat& vb, const int o, const int v,
               const Tensor& F_occ, const Tensor& F_virt, int i, int j, int k);

double calc_pert_triples_spin_gemm(const Tensor& DEprqs, const Tensor& t1, const Tensor& t2,
                            const Tensor& FN, Sli O, Sli V);

double calc_pert_triples_spat_gemm(const Tensor& Iprqs, const Tensor& DEprqs, const Tensor& t1, const Tensor& t2,
                            const Tensor& FN, Sli O, Sli V);

#endif
