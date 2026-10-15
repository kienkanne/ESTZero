#ifndef TRIPLES_BUFFERS
#define TRIPLES_BUFFERS

#include "tensor.h"

// The integral blocks (T) reads, each copied once before the i/j/k loops into a contiguous tensor with the loop
// indices leading. See `triples_{spat,spin}.cc` for the permute + packing calls: the occupied indices (v: i,j,k) 
// are permuted to the front (outermost), so every per (i,j,k) slice is then a contiguous view (select on the outer axes).
// Cost more memory (pack all initially) but avoid much of the packing cost every loop.
struct TriplesBlocks {
    Tensor vvvo; // (3,i/j/k): permute to (3,0,1,2)
    Tensor ovoo; // (2,i/j/k).(2,i/j/k): permute to (2,3,0,1)

    // These are used once in accumulating into x2. The comments show their original slicing and packing trick
    Tensor oovv; // (0,j): No permute
    Tensor vovv; // (1,k): permute to (1,0,2,3)
    Tensor ooov; // (0,j).(0,k): No permute
};

// eps_a + eps_b + eps_c over the virtuals, the (i,j,k)-independent part of the denominator: built once
inline Tensor virtual_denominator(const Tensor& F_virt, const int v) {
    Tensor eps_abc({v, v, v});
    for (int a = 0; a < v; ++a) {
    for (int b = 0; b < v; ++b) {
    for (int c = 0; c < v; ++c) {
        eps_abc(a, b, c) = F_virt(a) + F_virt(b) + F_virt(c);
    }}}
    return eps_abc;
}

// t3 /= (e_ijk - eps_abc) in place, one pass. Both are freshly built (v,v,v) tensors, so contiguous.
// TODO: introduce broadcasting support for `Tensor`
// noinline: compiled on its own this loop vectorizes (vdivpd), but when LTO inlines it into the large driver loop GCC
// emits the scalar version (one vdivsd per element), measured ~0.3 s slower on parabenzyne/sto-3g.
[[gnu::noinline]] inline void divide_by_denominator(Tensor& t3, const double e_ijk, const Tensor& eps_abc) {
    double* __restrict t = t3.data_ptr();
    const double* __restrict e = eps_abc.data_ptr();
    for (Index n = 0; n < t3.size(); ++n) t[n] /= e_ijk - e[n];
}

#endif
