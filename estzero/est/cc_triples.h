#ifndef TRIPLES
#define TRIPLES

#include "tensor.h"
#include <vector>

// Allocate all the Tensor buffers that is reused frequently throughout the entire loop at the beginning,
// so no new allocation needed per loop. The allocation of tensors that depend on i,j,k can be saved further
// more by only creating them when the index change, so for outer indices (i,j) they won't needed to be resigned
// every innermost loop (see `cc_triples.cc`)
struct VikingsBufferSpin {
    std::vector<double> gemm_out;

    Tensor x1, x2, t3,
    Wvvvo_i, Wvvvo_j, Wvvvo_k,
    Wovoo_jk, Wovoo_ik, Wovoo_ji;

    VikingsBufferSpin(const Tensor& t1, const Tensor&t2, const int o, const int v) 
        : gemm_out(v*v*v), x1(t1.shape), x2(t2.shape), t3({v,v,v}),
        Wvvvo_i({v,v,v}), Wvvvo_j({v,v,v}), Wvvvo_k({v,v,v}),
        Wovoo_jk({o,v}), Wovoo_ik({o,v}), Wovoo_ji({o,v}) {};
};

// Similar idea but for spatial (RHF), which needs additional permuted Wovoo tensors.
struct VikingsBufferSpat {
    std::vector<double> gemm_out;

    Tensor x1, x2, t3, t3_diff, t3_comb1,
    Wvvvo_i, Wvvvo_j, Wvvvo_k,
    Wovoo_jk, Wovoo_ik, Wovoo_ji,
    Wovoo_kj, Wovoo_ki, Wovoo_ij;

    VikingsBufferSpat(const Tensor& t1, const Tensor&t2, const int o, const int v) 
        : gemm_out(v*v*v), x1(t1.shape), x2(t2.shape), 
        t3({v,v,v}), t3_diff({v,v,v}), t3_comb1({v,v,v}),
        Wvvvo_i({v,v,v}), Wvvvo_j({v,v,v}), Wvvvo_k({v,v,v}),
        Wovoo_jk({o,v}), Wovoo_ik({o,v}), Wovoo_ji({o,v}),
        Wovoo_kj({o,v}), Wovoo_ki({o,v}), Wovoo_ij({o,v}) {};
};

double calc_pert_triples(const Tensor& Iprqs, const Tensor& DEprqs, const Tensor& t1, const Tensor& t2,
                        const Tensor& FN, Sli O, Sli V, bool spin);

double calc_pert_triples_noddy(const Tensor& Iprqs, const Tensor& t1, const Tensor&t2,
                        const Tensor& FN, Sli O, Sli V);

#endif
