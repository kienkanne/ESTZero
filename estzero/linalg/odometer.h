#ifndef ODOMETER
#define ODOMETER

#include <vector>

// Advances `global_indices` to the next combination of a shared set of axes. The last axis 
// is the fastest-moving, matching Tensor's row-major layout where the last axis sweeps fastest.
// The sweeped axes can be einsum's distinct index letters, or the common logical shape two
// Tensors walk together in an elementwise op. 
//
// Alongside the counter, it incrementally updates the flat offset of every "term" via 
// `terms_strides`, which is term_idx's own real per-axis stride for that axis, or 0 if 
// that term doesn't depend on the axis at all (important for einsum) Because each step only 
// adds/subtracts one axis's stride delta instead of recomputing every term's flat offset 
// from scratch every iteration, this is O(1) per combination
inline void odometer(
    std::vector<int>& global_indices,
    std::vector<int>& terms_flats,
    const std::vector<int>& dim_sizes,
    const std::vector<std::vector<int>>& terms_strides) {

    for (int axis = global_indices.size() - 1; axis >= 0; --axis) {
        if (global_indices[axis] < dim_sizes[axis] - 1) {
            ++global_indices[axis];

            for (size_t term_idx = 0;
                 term_idx < terms_flats.size();
                 ++term_idx) {
                terms_flats[term_idx] += terms_strides[term_idx][axis];
            }
            return;
        }

        global_indices[axis] = 0;

        for (size_t term_idx = 0; term_idx < terms_flats.size(); ++term_idx) {
            terms_flats[term_idx] -= terms_strides[term_idx][axis] * (dim_sizes[axis] - 1);
        }
    }
};

#endif
