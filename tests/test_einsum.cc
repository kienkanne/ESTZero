#include "doctest.h"

#include "einsum.h"

namespace {

// doctest has no Tensor comparison; compare element-wise over the flat storage.
void check_tensor(const Tensor& actual, const std::vector<int>& shape,
                   const std::vector<double>& expected) {
    REQUIRE(actual.shape == shape);
    for (size_t i = 0; i < expected.size(); ++i)
        CHECK(actual.flat_index(i) == doctest::Approx(expected[i]));
}

} // namespace

// ── Basic contractions (matrices) ───────────────────────────────────────────
// Hand-checkable against the einsum definition directly.

TEST_CASE("einsum: identity, ij->ij") {
    Tensor input({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor output = einsum("ij->ij", {input});
    check_tensor(output, {2, 3}, {1, 2, 3, 4, 5, 6});
}

TEST_CASE("einsum: transpose, ij->ji") {
    Tensor input({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor output = einsum("ij->ji", {input});
    check_tensor(output, {3, 2}, {1, 4, 2, 5, 3, 6});
}

TEST_CASE("einsum: elementwise product, ij,ij->ij") {
    Tensor left({2, 2}, {1, 2, 3, 4});
    Tensor right({2, 2}, {5, 6, 7, 8});
    Tensor output = einsum("ij,ij->ij", {left, right});
    check_tensor(output, {2, 2}, {5, 12, 21, 32});
}

TEST_CASE("einsum: matrix multiply, ik,kj->ij") {
    Tensor left({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor right({3, 2}, {7, 8, 9, 10, 11, 12});
    Tensor output = einsum("ik,kj->ij", {left, right});
    check_tensor(output, {2, 2}, {58, 64, 139, 154});
}

TEST_CASE("einsum: matrix-vector contraction, ij,j->i") {
    Tensor left({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor right({3}, {10, 20, 30});
    Tensor output = einsum("ij,j->i", {left, right});
    check_tensor(output, {2}, {140, 320});
}

TEST_CASE("einsum: full reduction to scalar, ij->") {
    Tensor input({2, 2}, {1, 2, 3, 4});
    Tensor output = einsum("ij->", {input});
    check_tensor(output, {}, {10});
}

// ── Tensors produced by view operations ─────────────────────────────────────
// These live on non-contiguous logical views, so they verify that einsum uses
// each Tensor's actual shape/strides/offset instead of reconstructing a plain
// contiguous buffer from the logical dimensions.

TEST_CASE("einsum: swapaxes view preserves logical order") {
    Tensor input({2, 3}, {1, 2, 3,
                         4, 5, 6});
    Tensor view = input.swapaxes(0, 1);

    Tensor output = einsum("ij->ij", {view});
    check_tensor(output, {3, 2}, {1, 4,
                                 2, 5,
                                 3, 6});
}

TEST_CASE("einsum: sliced view uses its offset") {
    Tensor input({4, 3}, {1, 2, 3,
                         4, 5, 6,
                         7, 8, 9,
                         10, 11, 12});
    Tensor view = input.slice(Sli{1, 4, 1}, Sli{});

    Tensor output = einsum("ij->ij", {view});
    check_tensor(output, {3, 3}, {4, 5, 6,
                                 7, 8, 9,
                                 10, 11, 12});
}

TEST_CASE("einsum: strided slice respects step") {
    Tensor input({6}, {0, 1, 2, 3, 4, 5});
    Tensor view = input.slice(Sli{0, 6, 2});

    Tensor output = einsum("i->", {view});
    check_tensor(output, {}, {6});
}

// ── Higher-dimensional tensors ──────────────────────────────────────────────
// Beyond matrices: a rank-3 batched contraction and a rank-4 <-> rank-4
// contraction, the shape transmo.cc actually exercises for AO->MO integrals.

TEST_CASE("einsum: rank-3 batched matrix multiply, bik,bkj->bij") {
    // Two independent 2x2 matrix multiplies, batched along axis b.
    // Batch 0: [[1,2],[3,4]] * [[1,0],[0,1]] = [[1,2],[3,4]]  (identity)
    // Batch 1: [[1,1],[1,1]] * [[2,0],[0,2]] = [[2,2],[2,2]]
    Tensor left({2, 2, 2}, {1, 2, 3, 4,
                            1, 1, 1, 1});
    Tensor right({2, 2, 2}, {1, 0, 0, 1,
                             2, 0, 0, 2});
    Tensor output = einsum("bik,bkj->bij", {left, right});
    check_tensor(output, {2, 2, 2}, {1, 2, 3, 4,
                                     2, 2, 2, 2});
}

TEST_CASE("einsum: rank-3 contraction over a shared axis, ijk,k->ij") {
    // 2x2x3 tensor contracted with a length-3 vector over the last axis.
    Tensor input({2, 2, 3}, {1, 2, 3, 4, 5, 6,
                             7, 8, 9, 10, 11, 12});
    Tensor vec({3}, {1, 0, -1});
    Tensor output = einsum("ijk,k->ij", {input, vec});
    // Each output element is input[i,j,0] - input[i,j,2].
    check_tensor(output, {2, 2}, {1 - 3, 4 - 6, 7 - 9, 10 - 12});
}

TEST_CASE("einsum: rank-4 contraction, pqrs,rs->pq") {
    // A 2x2x2x2 "integral-like" tensor contracted with a 2x2 "density" over
    // its last two indices, the same contraction shape as J(p,q) += (pq|rs)D(r,s).
    Tensor eri({2, 2, 2, 2}, {
        1, 2, 3, 4,   5, 6, 7, 8,
        9, 10, 11, 12,  13, 14, 15, 16
    });
    Tensor density({2, 2}, {1, 0, 0, 1}); // identity: picks out the trace over r==s
    Tensor output = einsum("pqrs,rs->pq", {eri, density});
    // output[p,q] = eri[p,q,0,0] + eri[p,q,1,1]
    check_tensor(output, {2, 2}, {1 + 4, 5 + 8, 9 + 12, 13 + 16});
}

TEST_CASE("einsum: rank-4 two-step transform matches sequential rank-2 transforms") {
    // (pq|rs) C_ri C_sj summed over r,s, done in one rank-4 einsum call, must
    // match doing the same contraction as two sequential matrix multiplies
    // reshaped by hand -- i.e. einsum's handling of higher-rank operands is
    // consistent with its handling of matrices.
    Tensor eri({2, 2, 2, 2}, {
        1, 2, 3, 4,   5, 6, 7, 8,
        9, 10, 11, 12,  13, 14, 15, 16
    });
    Tensor C({2, 2}, {1, 1, 1, -1});

    Tensor once = einsum("pqrs,ri->pqis", {eri, C});
    Tensor twice = einsum("pqis,sj->pqij", {once, C});

    // Reference: brute-force the same sum by hand for a couple of elements.
    auto ref = [&](int p, int q, int i, int j) {
        double sum = 0.0;
        for (int r = 0; r < 2; ++r)
            for (int s = 0; s < 2; ++s)
                sum += eri(p, q, r, s) * C(r, i) * C(s, j);
        return sum;
    };
    for (int p = 0; p < 2; ++p)
    for (int q = 0; q < 2; ++q)
    for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j)
        CHECK(twice(p, q, i, j) == doctest::Approx(ref(p, q, i, j)));
}

// ── Validation ───────────────────────────────────────────────────────────────

TEST_CASE("einsum: rejects subscripts with no arrow") {
    Tensor matrix({2, 2}, {1, 2, 3, 4});
    CHECK_THROWS_AS((void)einsum("ij", {matrix}), std::invalid_argument);
}

TEST_CASE("einsum: rejects subscripts with two arrows") {
    Tensor matrix({2, 2}, {1, 2, 3, 4});
    CHECK_THROWS_AS((void)einsum("ij->i->j", {matrix}), std::invalid_argument);
}

TEST_CASE("einsum: rejects an empty input term") {
    Tensor matrix({2, 2}, {1, 2, 3, 4});
    Tensor vec({2}, {1, 2});
    CHECK_THROWS_AS((void)einsum("ij,->i", {matrix, vec}), std::invalid_argument);
}

TEST_CASE("einsum: rejects a duplicate output index") {
    Tensor matrix({2, 2}, {1, 2, 3, 4});
    CHECK_THROWS_AS((void)einsum("ij->ii", {matrix}), std::invalid_argument);
}

TEST_CASE("einsum: rejects an output index absent from the input terms") {
    Tensor matrix({2, 2}, {1, 2, 3, 4});
    CHECK_THROWS_AS((void)einsum("ij->k", {matrix}), std::invalid_argument);
}

TEST_CASE("einsum: rejects an operand-count mismatch") {
    Tensor matrix({2, 2}, {1, 2, 3, 4});
    CHECK_THROWS_AS((void)einsum("ij,j->i", {matrix}), std::invalid_argument);
}

TEST_CASE("einsum: rejects an operand rank mismatch") {
    Tensor matrix({2, 2}, {1, 2, 3, 4});
    CHECK_THROWS_AS((void)einsum("i->i", {matrix}), std::invalid_argument);
}

TEST_CASE("einsum: rejects a dimension-size mismatch across a shared index") {
    Tensor matrix({2, 2}, {1, 2, 3, 4});
    Tensor vec({3}, {1, 2, 3});
    CHECK_THROWS_AS((void)einsum("ij,j->i", {matrix, vec}), std::invalid_argument);
}
