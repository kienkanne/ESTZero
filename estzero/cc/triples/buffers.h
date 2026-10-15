#ifndef TRIPLES_BUFFERS
#define TRIPLES_BUFFERS

#include "tensor.h"

// The integral blocks (T) reads, each copied once before the i/j/k loops into a contiguous tensor with the loop
// indices leading. See `triples_{spat,spin}.cc` for the permute + packing calls: the virtual indices (v: i,j,k) 
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
#endif
