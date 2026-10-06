#ifndef TRIPLES_BUFFERS
#define TRIPLES_BUFFERS

#include "tensor.h"

// Allocate all the Tensor buffers that is reused frequently throughout the entire loop at the beginning,
// so no new allocation needed per loop. The W slices are packed only when their own index changes, so for
// the outer indices (i,j) they are not refilled every innermost loop (see `triples_spin.cc`).
struct TriplesBufferSpin {
    Tensor x1, x2, t3,
    Wvvvo_i, Wvvvo_j, Wvvvo_k,
    Wovoo_jk, Wovoo_ik, Wovoo_ji;

    TriplesBufferSpin(const Tensor& t1, const Tensor& t2, const int o, const int v)
        : x1(t1.shape()), x2(t2.shape()), t3({v,v,v}),
        Wvvvo_i({v,v,v}), Wvvvo_j({v,v,v}), Wvvvo_k({v,v,v}),
        Wovoo_jk({o,v}), Wovoo_ik({o,v}), Wovoo_ji({o,v}) {}
};

// Similar idea but for spatial (RHF), which needs additional permuted Wovoo tensors and t3 combinations.
struct TriplesBufferSpat {
    Tensor x1, x2, t3, t3_diff, t3_comb1,
    Wvvvo_i, Wvvvo_j, Wvvvo_k,
    Wovoo_jk, Wovoo_ik, Wovoo_ji,
    Wovoo_kj, Wovoo_ki, Wovoo_ij;

    TriplesBufferSpat(const Tensor& t1, const Tensor& t2, const int o, const int v)
        : x1(t1.shape()), x2(t2.shape()),
        t3({v,v,v}), t3_diff({v,v,v}), t3_comb1({v,v,v}),
        Wvvvo_i({v,v,v}), Wvvvo_j({v,v,v}), Wvvvo_k({v,v,v}),
        Wovoo_jk({o,v}), Wovoo_ik({o,v}), Wovoo_ji({o,v}),
        Wovoo_kj({o,v}), Wovoo_ki({o,v}), Wovoo_ij({o,v}) {}
};

#endif
