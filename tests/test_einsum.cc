#include "doctest.h"

#include "einsum.h"
#include "est_gemm.h"

namespace {

// shape()/strides() are std::span (no operator==); copy into a vector to compare and print in CHECKs
std::vector<Index> as_vector(std::span<const Index> s) { return {s.begin(), s.end()}; }

// doctest has no Tensor comparison; compare element-wise over the flat storage.
void check_tensor(const Tensor& actual, const std::vector<Index>& shape,
                   const std::vector<double>& expected) {
    REQUIRE(as_vector(actual.shape()) == shape);
    for (size_t i = 0; i < expected.size(); ++i)
        CHECK(actual.flat_index(i) == doctest::Approx(expected[i]));
}

// Deterministic, non-symmetric fill so a wrong axis pairing changes the result
Tensor filled(const std::vector<Index>& shape, double seed) {
    Tensor t(shape);
    for (Index i = 0; i < t.size(); ++i) t.flat_index(i) = seed + 0.37 * i - 0.011 * i * i;
    return t;
}

// Runs a test body under each einsum backend that matters on CPU: the odometer (the tests' default, gs is
// never set), and Eigen and OpenBLAS (try_gemm -> call_gemm). Restores the process-global scheme afterwards.
template <typename Body>
void for_each_cpu_scheme(Body body) {
    const GEMMScheme saved = gs;
    for (GEMMScheme scheme : {GEMMScheme::ODO, GEMMScheme::EIGEN, GEMMScheme::OPENBLAS}) {
        gs = scheme;
        body();
    }
    gs = saved;
}

} // namespace

// ── Repeated index within a single operand (diagonal/trace patterns) ───────
// Regression tests: terms_strides used to be built by assignment, so when a
// label appeared twice in the same operand's term, the second occurrence's
// stride silently overwrote the first's -- only one of the two tied axes
// actually moved when that label incremented, e.g. einsum("ii->", A)
// walked A's column stride only, ignoring the row stride entirely. First
// surfaced building the normal-ordered Fock matrix's two-electron term,
// F_pq += sum_i <pi||qi>, i.e. einsum("piqi->pq", ...).

TEST_CASE("einsum: trace via a doubled index, ii->") {
    Tensor A({3, 3}, {1, 2, 3,
                       4, 5, 6,
                       7, 8, 9});
    double trace = einsum("ii->", A)();
    CHECK(trace == doctest::Approx(1 + 5 + 9));
}

TEST_CASE("einsum: rank-4 trace over a doubled index, piqi->pq") {
    // Same shape as the normal-ordered Fock matrix's two-electron term:
    // contract axes 1 and 3 (tied to the same label) while axes 0 and 2
    // stay free.
    Tensor B({2, 2, 2, 2});
    for (int p = 0; p < 2; ++p)
    for (int i = 0; i < 2; ++i)
    for (int q = 0; q < 2; ++q)
    for (int j = 0; j < 2; ++j)
        B(p, i, q, j) = p * 1000 + i * 100 + q * 10 + j;

    Tensor R = einsum("piqi->pq", B);
    for (int p = 0; p < 2; ++p)
    for (int q = 0; q < 2; ++q) {
        double expected = B(p, 0, q, 0) + B(p, 1, q, 1);
        CHECK(R(p, q) == doctest::Approx(expected));
    }
}

// ── Basic contractions (matrices) ───────────────────────────────────────────
// Hand-checkable against the einsum definition directly.

TEST_CASE("einsum: identity, ij->ij") {
    Tensor input({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor output = einsum("ij->ij", input);
    check_tensor(output, {2, 3}, {1, 2, 3, 4, 5, 6});
}

TEST_CASE("einsum: transpose, ij->ji") {
    Tensor input({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor output = einsum("ij->ji", input);
    check_tensor(output, {3, 2}, {1, 4, 2, 5, 3, 6});
}

TEST_CASE("einsum: elementwise product, ij,ij->ij") {
    Tensor left({2, 2}, {1, 2, 3, 4});
    Tensor right({2, 2}, {5, 6, 7, 8});
    Tensor output = einsum("ij,ij->ij", left, right);
    check_tensor(output, {2, 2}, {5, 12, 21, 32});
}

TEST_CASE("einsum: matrix multiply, ik,kj->ij") {
    Tensor left({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor right({3, 2}, {7, 8, 9, 10, 11, 12});
    Tensor output = einsum("ik,kj->ij", left, right);
    check_tensor(output, {2, 2}, {58, 64, 139, 154});
}

TEST_CASE("einsum: matrix-vector contraction, ij,j->i") {
    Tensor left({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor right({3}, {10, 20, 30});
    Tensor output = einsum("ij,j->i", left, right);
    check_tensor(output, {2}, {140, 320});
}

TEST_CASE("einsum: full reduction to scalar, ij->") {
    Tensor input({2, 2}, {1, 2, 3, 4});
    Tensor output = einsum("ij->", input);
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

    Tensor output = einsum("ij->ij", view);
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

    Tensor output = einsum("ij->ij", view);
    check_tensor(output, {3, 3}, {4, 5, 6,
                                 7, 8, 9,
                                 10, 11, 12});
}

TEST_CASE("einsum: strided slice respects step") {
    Tensor input({6}, {0, 1, 2, 3, 4, 5});
    Tensor view = input.slice(Sli{0, 6, 2});

    Tensor output = einsum("i->", view);
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
    Tensor output = einsum("bik,bkj->bij", left, right);
    check_tensor(output, {2, 2, 2}, {1, 2, 3, 4,
                                     2, 2, 2, 2});
}

TEST_CASE("einsum: rank-3 contraction over a shared axis, ijk,k->ij") {
    // 2x2x3 tensor contracted with a length-3 vector over the last axis.
    Tensor input({2, 2, 3}, {1, 2, 3, 4, 5, 6,
                             7, 8, 9, 10, 11, 12});
    Tensor vec({3}, {1, 0, -1});
    Tensor output = einsum("ijk,k->ij", input, vec);
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
    Tensor output = einsum("pqrs,rs->pq", eri, density);
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

    Tensor once = einsum("pqrs,ri->pqis", eri, C);
    Tensor twice = einsum("pqis,sj->pqij", once, C);

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
    CHECK_THROWS_AS((void)einsum("ij", matrix), std::invalid_argument);
}

TEST_CASE("einsum: rejects subscripts with two arrows") {
    Tensor matrix({2, 2}, {1, 2, 3, 4});
    CHECK_THROWS_AS((void)einsum("ij->i->j", matrix), std::invalid_argument);
}

TEST_CASE("einsum: rejects an empty input term") {
    Tensor matrix({2, 2}, {1, 2, 3, 4});
    Tensor vec({2}, {1, 2});
    CHECK_THROWS_AS((void)einsum("ij,->i", matrix, vec), std::invalid_argument);
}

TEST_CASE("einsum: rejects a duplicate output index") {
    Tensor matrix({2, 2}, {1, 2, 3, 4});
    CHECK_THROWS_AS((void)einsum("ij->ii", matrix), std::invalid_argument);
}

TEST_CASE("einsum: rejects an output index absent from the input terms") {
    Tensor matrix({2, 2}, {1, 2, 3, 4});
    CHECK_THROWS_AS((void)einsum("ij->k", matrix), std::invalid_argument);
}

TEST_CASE("einsum: rejects an operand-count mismatch") {
    Tensor matrix({2, 2}, {1, 2, 3, 4});
    CHECK_THROWS_AS((void)einsum("ij,j->i", matrix), std::invalid_argument);
}

TEST_CASE("einsum: rejects an operand rank mismatch") {
    Tensor matrix({2, 2}, {1, 2, 3, 4});
    CHECK_THROWS_AS((void)einsum("i->i", matrix), std::invalid_argument);
}

TEST_CASE("einsum: rejects a dimension-size mismatch across a shared index") {
    Tensor matrix({2, 2}, {1, 2, 3, 4});
    Tensor vec({3}, {1, 2, 3});
    CHECK_THROWS_AS((void)einsum("ij,j->i", matrix, vec), std::invalid_argument);
}

// ── Cached subscript plans ──────────────────────────────────────────────────
// einsum parses each distinct subscript string once and caches the plan, so the plan must depend only on the
// string: reusing a subscript with different operand shapes (below and above try_gemm's FLOP guard) must still
// match the defining sum.

TEST_CASE("einsum: a cached subscript plan is reused correctly across operand shapes") {
    for_each_cpu_scheme([] {
        for (auto [m, k, n] : {std::array<Index, 3>{2, 3, 2}, {16, 8, 16}, {5, 1, 7}}) {
            const Tensor A = filled({m, k}, 0.5), B = filled({k, n}, -1.25);
            const Tensor C = einsum("ik,kj->ij", A, B);
            REQUIRE(as_vector(C.shape()) == std::vector<Index>{m, n});
            for (Index i = 0; i < m; ++i)
            for (Index j = 0; j < n; ++j) {
                double sum = 0.0;
                for (Index p = 0; p < k; ++p) sum += A(i, p) * B(p, j);
                CHECK(C(i, j) == doctest::Approx(sum));
            }
        }
    });
}

TEST_CASE("einsum: a 3-operand chain matches nested pairwise contractions") {
    for_each_cpu_scheme([] {
        const Tensor A = filled({6, 4}, 0.3), B = filled({4, 5, 3}, -0.7), C = filled({3, 6}, 1.1);
        const Tensor chained = einsum("ij,jkl,li->ik", A, B, C);
        const Tensor nested = einsum("ikl,li->ik", einsum("ij,jkl->ikl", A, B), C);
        REQUIRE(as_vector(chained.shape()) == as_vector(nested.shape()));
        for (Index i = 0; i < chained.size(); ++i)
            CHECK(chained.flat_index(i) == doctest::Approx(nested.flat_index(i)));
    });
}

TEST_CASE("einsum: a cached plan is rebound for an operand with the same shape but different strides") {
    // A transposed view has the same shape as a contiguous tensor but different strides, which the odometer's
    // stride table bakes in, so the cache must not treat it as a hit.
    for_each_cpu_scheme([] {
        const Tensor X = filled({4, 4}, 0.2), Y = filled({4, 4}, -0.4);
        for (const Tensor& A : {X, X.swapaxes(0, 1)}) {
            const Tensor C = einsum("ik,kj->ij", A, Y);
            const Tensor T = einsum("ij->ji", A);
            for (Index i = 0; i < 4; ++i)
            for (Index j = 0; j < 4; ++j) {
                double sum = 0.0;
                for (Index p = 0; p < 4; ++p) sum += A(i, p) * Y(p, j);
                CHECK(C(i, j) == doctest::Approx(sum));
                CHECK(T(i, j) == doctest::Approx(A(j, i)));
            }
        }
    });
}

TEST_CASE("einsum: GEMM path with operands whose permute+reshape is a non-contiguous view") {
    // reshape returns a strided view when one exists, e.g. lab -> (ab, l) is (1, l-stride): a transposed matrix.
    // try_gemm passes those as trans + ld (Eigen, OpenBLAS) or packs them; never as if they were packed:
    // data_ptr() doesn't check contiguity, so that would silently read the wrong elements.
    // Sizes keep every contraction above the GEMM FLOP threshold. These are the (T) einsum shapes.
    const Index l = 4, a = 6, b = 7, c = 8;
    const Tensor lab = filled({l, a, b}, 0.3), lc = filled({l, c}, -0.2);
    const Tensor ad = filled({a, c}, 0.1), bcd = filled({b, a, c}, -0.5); // "ad,bcd->abc" with d = c's extent
    const Tensor mc = filled({l, c}, 0.7), abc = filled({a, b, c}, 0.05);

    // K = 1: an outer product has no summed label, so reshape adds a size-1 K axis whose stride means nothing
    const Tensor x = filled({40}, 0.2), y = filled({40}, -0.3);
    // Padded views: lda > row length. mc_pad is (l, c) with row stride 20 (normal, lda = 20); lab_pad is (l, a, b)
    // with row stride (a+2)*b, so (ab, l) is transposed with lda = (a+2)*b > a*b
    const Tensor mc_pad = filled({l, 20}, 0.4).slice(Sli{}, Sli{2, 2 + c});
    const Tensor lab_pad = filled({l, a + 2, b}, -0.1).slice(Sli{}, Sli{1, 1 + a}, Sli{});

    // Negative strides (reversed slices): a unit stride on one axis isn't enough, the other stride must be a valid
    // leading dimension (>= the row length), or BLAS rejects the call and leaves the result uninitialized.
    const Tensor X = filled({40, 40}, 0.1), Y = filled({40, 40}, 0.5);
    const Tensor x_rev = x.slice(Sli{39, -42, -1});                          // (40,) stride -1 -> (40,1) strides (-1,1)
    const Tensor X_rev = X.slice(Sli{39, -42, -1}, Sli{});                   // strides (-40, 1)
    const Tensor X_revT = X.slice(Sli{}, Sli{39, -42, -1}).swapaxes(0, 1);   // strides (-1, 40)

    // Reference: the odometer path, computed once
    const GEMMScheme saved = gs;
    gs = GEMMScheme::ODO;
    const Tensor ref1 = einsum("lab,lc->abc", lab, lc);
    const Tensor ref2 = einsum("ad,bcd->abc", ad, bcd);
    const Tensor ref3 = einsum("mc,abc->mab", mc, abc);
    const Tensor ref4 = einsum("abc,dbc->ad", abc, bcd.permute(1, 0, 2).slice(Sli{0, a}, Sli{}, Sli{}));
    const Tensor ref5 = einsum("i,j->ij", x, y);
    const Tensor ref6 = einsum("mc,abc->mab", mc_pad, abc);
    const Tensor ref7 = einsum("lab,lc->abc", lab_pad, lc);
    const Tensor ref8 = einsum("i,j->ij", x_rev, y);
    const Tensor ref9 = einsum("ij,jk->ik", X_rev, Y);
    const Tensor ref10 = einsum("ij,jk->ik", X_revT, Y);
    gs = saved;

    for_each_cpu_scheme([&] {
        const auto same = [](const Tensor& got, const Tensor& want) {
            REQUIRE(got.size() == want.size());
            for (Index i = 0; i < want.size(); ++i) CHECK(got.flat_index(i) == doctest::Approx(want.flat_index(i)));
        };
        same(einsum("lab,lc->abc", lab, lc), ref1);
        same(einsum("ad,bcd->abc", ad, bcd), ref2);
        same(einsum("mc,abc->mab", mc, abc), ref3);
        same(einsum("abc,dbc->ad", abc, bcd.permute(1, 0, 2).slice(Sli{0, a}, Sli{}, Sli{})), ref4);
        same(einsum("i,j->ij", x, y), ref5);
        same(einsum("mc,abc->mab", mc_pad, abc), ref6);
        same(einsum("lab,lc->abc", lab_pad, lc), ref7);
        same(einsum("i,j->ij", x_rev, y), ref8);
        same(einsum("ij,jk->ik", X_rev, Y), ref9);
        same(einsum("ij,jk->ik", X_revT, Y), ref10);
    });
}

// ── call_gemm backends ──────────────────────────────────────────────────────
// Every backend implements the same BLAS contract (C = alpha * op(A) * op(B) + beta * C, row-major, op = optional
// transpose, leading dimensions lda/ldb/ldc = row strides of the stored matrices). Eigen and OpenBLAS must both match
// a naive reference for every transpose combination, beta != 0, and padded leading dimensions (ld > row length),
// without touching the padding. The odometer scheme has no GEMM, so it must throw rather than silently leave C
// untouched.

TEST_CASE("call_gemm: Eigen and OpenBLAS match a naive reference for all transposes, alpha/beta, padded ld; ODO throws") {
    const int M = 5, N = 3, K = 4;
    const double alpha = 0.7;

    for (int pad : {0, 3})
    for (bool transA : {false, true})
    for (bool transB : {false, true})
    for (double beta : {0.0, 1.3}) {
        // Stored shapes: A is (K x M) when transposed, else (M x K); likewise B. ld = stored row length + pad.
        const int a_rows = transA ? K : M, a_cols = transA ? M : K;
        const int b_rows = transB ? N : K, b_cols = transB ? K : N;
        const int lda = a_cols + pad, ldb = b_cols + pad, ldc = N + pad;

        std::vector<double> A(a_rows * lda), B(b_rows * ldb), C0(M * ldc);
        for (size_t i = 0; i < A.size(); ++i)  A[i]  = 0.3 + 0.17 * i - 0.01 * i * i;
        for (size_t i = 0; i < B.size(); ++i)  B[i]  = -0.8 + 0.23 * i;
        for (size_t i = 0; i < C0.size(); ++i) C0[i] = 0.05 * i - 0.4;

        const auto opA = [&](int i, int k) { return transA ? A[k * lda + i] : A[i * lda + k]; };
        const auto opB = [&](int k, int j) { return transB ? B[j * ldb + k] : B[k * ldb + j]; };
        std::vector<double> C_ref = C0;
        for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            double sum = 0.0;
            for (int k = 0; k < K; ++k) sum += opA(i, k) * opB(k, j);
            C_ref[i * ldc + j] = alpha * sum + beta * C0[i * ldc + j];
        }

        CAPTURE(pad); CAPTURE(transA); CAPTURE(transB); CAPTURE(beta);
        for (GEMMScheme scheme : {GEMMScheme::EIGEN, GEMMScheme::OPENBLAS}) {
            std::vector<double> C = C0;
            call_gemm(scheme, M, N, K, lda, ldb, ldc, A.data(), B.data(), C.data(), transA, transB, alpha, beta);
            for (size_t i = 0; i < C0.size(); ++i) CHECK(C[i] == doctest::Approx(C_ref[i])); // padding stays C0
        }

        std::vector<double> C_odo = C0;
        CHECK_THROWS_AS(call_gemm(GEMMScheme::ODO, M, N, K, lda, ldb, ldc, A.data(), B.data(), C_odo.data(),
                                  transA, transB, alpha, beta),
                        std::runtime_error);
        CHECK(C_odo == C0);
    }
}
