#include "doctest.h"

#include "tensor.h"

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

} // namespace

// ── Construction ─────────────────────────────────────────────────────────────

TEST_CASE("Tensor: default construction has rank zero") {
    Tensor t;
    CHECK(t.ndim() == 0);
}

TEST_CASE("Tensor: shape constructor computes row-major strides") {
    Tensor v({5});
    CHECK(as_vector(v.strides()) == std::vector<Index>{1});

    Tensor m({2, 3});
    CHECK(as_vector(m.shape()) == std::vector<Index>{2, 3});
    CHECK(as_vector(m.strides()) == std::vector<Index>{3, 1});

    Tensor t3({2, 3, 4});
    CHECK(as_vector(t3.strides()) == std::vector<Index>{12, 4, 1});
}

TEST_CASE("Tensor: shape constructor zero-initializes storage") {
    Tensor m({2, 2});
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            CHECK(m(i, j) == doctest::Approx(0.0));
}

TEST_CASE("Tensor: shape+values constructor stores values row-major") {
    Tensor m({2, 3}, {1, 2, 3, 4, 5, 6});
    check_tensor(m, {2, 3}, {1, 2, 3, 4, 5, 6});
    CHECK(m(0, 0) == doctest::Approx(1));
    CHECK(m(0, 2) == doctest::Approx(3));
    CHECK(m(1, 0) == doctest::Approx(4));
    CHECK(m(1, 2) == doctest::Approx(6));
}

TEST_CASE("Tensor: shape+values constructor rejects a size mismatch") {
    CHECK_THROWS_AS((void)Tensor({2, 3}, {1, 2, 3}), std::invalid_argument);
}

// ── fill / clear ──────────────────────────────────────────────────────────────

TEST_CASE("Tensor: fill overwrites every element") {
    Tensor m({2, 2}, {1, 2, 3, 4});
    m.fill(7.0);
    check_tensor(m, {2, 2}, {7, 7, 7, 7});
}

TEST_CASE("Tensor: fill on a view leaves the rest of the shared buffer untouched") {
    // Column 1 of a 2x3: a non-contiguous view (strides (3,1), shape (2,1)), so this also covers the
    // odometer path. Filling the whole buffer instead would zero columns 0 and 2 too.
    Tensor m({2, 3}, {1, 2, 3, 4, 5, 6});
    m.slice(Sli{0, 2}, Sli{1, 2}).fill(0.0);
    check_tensor(m, {2, 3}, {1, 0, 3, 4, 0, 6});
}

TEST_CASE("Tensor: clear empties the tensor without invalidating views that share its buffer") {
    // Accessing an element of the cleared tensor itself is still UB (indexing is unchecked, v1.5.3), but
    // its size/ndim are observable, and a view taken before clear() must keep reading valid data.
    Tensor v({3}, {1, 2, 3});
    Tensor tail = v.slice(Sli{1, 3});
    v.clear();
    CHECK(v.size() == 0);
    CHECK(v.ndim() == 0);
    check_tensor(tail, {2}, {2, 3});
}

// ── operator() / flat_index agreement (owning tensor, offset == 0) ──────────

TEST_CASE("Tensor: operator() and flat_index agree for an owning tensor") {
    Tensor m({2, 3}, {1, 2, 3, 4, 5, 6});
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 3; ++j)
            CHECK(m(i, j) == doctest::Approx(m.flat_index(static_cast<size_t>(i * 3 + j))));
}

TEST_CASE("Tensor: operator() writes through the non-const overload") {
    Tensor m({2, 2}, {1, 2, 3, 4});
    m(0, 1) = 42.0;
    CHECK(m(0, 1) == doctest::Approx(42.0));
    check_tensor(m, {2, 2}, {1, 42, 3, 4});
}

TEST_CASE("Tensor: operator() rejects an index-count mismatch") {
    Tensor m({2, 2}, {1, 2, 3, 4});
    CHECK_THROWS_AS((void)m(0), std::invalid_argument);
    CHECK_THROWS_AS((void)m(0, 0, 0), std::invalid_argument);
}

// ── swapaxes: zero-copy transpose-like views ─────────────────────────────────

TEST_CASE("Tensor: swapaxes swaps shape and strides") {
    Tensor m({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor t = m.swapaxes(0, 1);
    CHECK(as_vector(t.shape()) == std::vector<Index>{3, 2});
    CHECK(as_vector(t.strides()) == std::vector<Index>{1, 3});
}

TEST_CASE("Tensor: swapaxes(0,1) on a matrix matches its transpose") {
    Tensor m({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor t = m.swapaxes(0, 1);
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 3; ++j)
            CHECK(t(j, i) == doctest::Approx(m(i, j)));
}

TEST_CASE("Tensor: swapaxes is a zero-copy view aliasing the source") {
    Tensor m({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor t = m.swapaxes(0, 1);

    // t(1, 0) addresses the same element as m(0, 1) (flat index 1).
    t(1, 0) = 99.0;
    CHECK(m(0, 1) == doctest::Approx(99.0));

    // Mutating the source is visible through the view too.
    m(1, 2) = -5.0;
    CHECK(t(2, 1) == doctest::Approx(-5.0));
}

// ── slice: zero-copy start/stop/step views ───────────────────────────────────

TEST_CASE("Tensor: slice with an explicit range on a vector") {
    Tensor v({5}, {10, 20, 30, 40, 50});
    Tensor s = v.slice(Sli{1, 4, 1});
    check_tensor(s, {3}, {20, 30, 40});
}

TEST_CASE("Tensor: default Sli{} is a full-axis identity view") {
    Tensor m({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor s = m.slice(Sli{}, Sli{});
    check_tensor(s, {2, 3}, {1, 2, 3, 4, 5, 6});
}

TEST_CASE("Tensor: slice with a stride selects every step-th element") {
    // A strided slice is not a contiguous view, so compare via operator()
    // (stride-aware) rather than flat_index (raw buffer walk).
    Tensor v({6}, {0, 1, 2, 3, 4, 5});
    Tensor s = v.slice(Sli{0, 6, 2});
    REQUIRE(as_vector(s.shape()) == std::vector<Index>{3});
    CHECK(s(0) == doctest::Approx(0));
    CHECK(s(1) == doctest::Approx(2));
    CHECK(s(2) == doctest::Approx(4));
}

TEST_CASE("Tensor: slice with a negative step reverses a sub-range") {
    // Equivalent to Python's a[4:1:-1] on [10,20,30,40,50] -> [50,40,30].
    // Not contiguous, so again compare via operator(), not flat_index.
    Tensor v({5}, {10, 20, 30, 40, 50});
    Tensor s = v.slice(Sli{4, 1, -1});
    REQUIRE(as_vector(s.shape()) == std::vector<Index>{3});
    CHECK(s(0) == doctest::Approx(50));
    CHECK(s(1) == doctest::Approx(40));
    CHECK(s(2) == doctest::Approx(30));
}

TEST_CASE("Tensor: slice on a matrix selects independent row and column ranges") {
    Tensor m({3, 4}, {1, 2, 3, 4,
                       5, 6, 7, 8,
                       9, 10, 11, 12});
    // Rows {1,2}, columns {0,2}.
    Tensor s = m.slice(Sli{1, 3, 1}, Sli{0, 4, 2});
    CHECK(as_vector(s.shape()) == std::vector<Index>{2, 2});
    CHECK(s(0, 0) == doctest::Approx(5));
    CHECK(s(0, 1) == doctest::Approx(7));
    CHECK(s(1, 0) == doctest::Approx(9));
    CHECK(s(1, 1) == doctest::Approx(11));
}

TEST_CASE("Tensor: slice rejects a zero step") {
    Tensor v({5}, {1, 2, 3, 4, 5});
    CHECK_THROWS_AS((void)v.slice(Sli{0, -1, 0}), std::invalid_argument);
}

TEST_CASE("Tensor: slice rejects a slice-count mismatch") {
    Tensor m({2, 3}, {1, 2, 3, 4, 5, 6});
    CHECK_THROWS_AS((void)m.slice(Sli{}), std::invalid_argument);
}

TEST_CASE("Tensor: slice is a zero-copy view aliasing the source") {
    Tensor v({5}, {10, 20, 30, 40, 50});
    Tensor s = v.slice(Sli{1, 4, 1});
    s(0) = 999.0;
    CHECK(v(1) == doctest::Approx(999.0));
}

// ── Aliasing / copy semantics ─────────────────────────────────────────────────
// The whole point of Tensor holding a std::shared_ptr: copying a Tensor
// shares the underlying storage rather than deep-copying it.

TEST_CASE("Tensor: copying a Tensor aliases the same storage") {
    Tensor a({2, 2}, {1, 2, 3, 4});
    Tensor b = a;

    b(0, 0) = 99.0;
    CHECK(a(0, 0) == doctest::Approx(99.0));

    a(1, 1) = -1.0;
    CHECK(b(1, 1) == doctest::Approx(-1.0));
}

TEST_CASE("Tensor: a view built from a copy still aliases the original") {
    Tensor a({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor b = a; // shares storage with a
    Tensor view = b.slice(Sli{0, 2, 1}, Sli{1, 3, 1});

    view(0, 0) = 100.0;
    CHECK(a(0, 1) == doctest::Approx(100.0));
}

// ── flat_index must respect a view's offset ──────────────────────────────────
// Regression test: flat_index() used to index the backing vector directly,
// ignoring Tensor::offset, so it disagreed with operator() on any view with a
// nonzero offset (swapaxes/slice results). Both must now agree.

TEST_CASE("Tensor: flat_index agrees with operator() on a sliced view") {
    Tensor t({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor row1 = t.slice(Sli{1, 2, 1}, Sli{0, 3, 1}); // just row 1: {4, 5, 6}

    REQUIRE(as_vector(row1.shape()) == std::vector<Index>{1, 3});
    for (int j = 0; j < 3; ++j)
        CHECK(row1.flat_index(static_cast<size_t>(j)) == doctest::Approx(row1(0, j)));
    check_tensor(row1, {1, 3}, {4, 5, 6});
}

TEST_CASE("Tensor: flat_index on a swapaxes view walks the shared buffer, not the view's logical order") {
    // swapaxes permutes strides, so unlike a same-order contiguous slice, its
    // flat_index() traversal legitimately diverges from operator()'s -- it
    // still respects the fixed offset (the fix), but doesn't reproduce the
    // view's own row-major order. Confirmed here: flat_index(k) on the
    // swapped view returns the same values as the untransposed m, in m's
    // original flat order.
    Tensor m({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor t = m.swapaxes(0, 1);
    for (size_t k = 0; k < 6; ++k)
        CHECK(t.flat_index(k) == doctest::Approx(m.flat_index(k)));
}

// ── arithmetic ───────────────────────────────────────────────────────────────

TEST_CASE("Tensor: addition works for owning tensors") {
    Tensor a({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor b({2, 3}, {6, 5, 4, 3, 2, 1});

    Tensor sum = a + b;
    check_tensor(sum, {2, 3}, {7, 7, 7, 7, 7, 7});
    CHECK(as_vector(sum.strides()) == std::vector<Index>{3, 1});
}

TEST_CASE("Tensor: addition follows slice strides and offsets") {
    Tensor a({2, 6}, {1, 2, 3, 4, 5, 6,
                       7, 8, 9, 10, 11, 12});
    Tensor b({2, 6}, {12, 11, 10, 9, 8, 7,
                       6, 5, 4, 3, 2, 1});

    Tensor sum = a.slice(Sli{0, 2, 1}, Sli{1, 6, 2})
               + b.slice(Sli{0, 2, 1}, Sli{1, 6, 2});
    check_tensor(sum, {2, 3}, {13, 13, 13, 13, 13, 13});
}

TEST_CASE("Tensor: addition follows swapped-axis view order") {
    Tensor a({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor b({2, 3}, {10, 20, 30, 40, 50, 60});

    Tensor sum = a.swapaxes(0, 1) + b.swapaxes(0, 1);
    check_tensor(sum, {3, 2}, {11, 44, 22, 55, 33, 66});
}

TEST_CASE("Tensor: addition rejects mismatched shapes") {
    Tensor a({2, 3});
    Tensor b({3, 2});
    CHECK_THROWS_AS((void)(a + b), std::invalid_argument);
}

TEST_CASE("Tensor: addition works when only one operand is a view") {
    // "a" is a plain contiguous tensor; view_b is a non-contiguous strided
    // slice of a larger tensor with the same {2,3} logical shape -- forces
    // _match_check's fast path off even though one side IS contiguous.
    Tensor a({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor big({2, 6}, {10, 20, 30, 40, 50, 60,
                        70, 80, 90, 100, 110, 120});
    Tensor view_b = big.slice(Sli{0, 2, 1}, Sli{0, 6, 2}); // columns 0,2,4: {10,30,50 / 70,90,110}

    Tensor sum = a + view_b;
    check_tensor(sum, {2, 3}, {11, 32, 53, 74, 95, 116});
}

TEST_CASE("Tensor: addition works at rank 4") {
    // The shape CCSD intermediates actually need -- the odometer/strides
    // machinery is rank-agnostic, but every prior arithmetic test was rank 2.
    Tensor a({2, 2, 2, 2}, {
        1, 2, 3, 4,     5, 6, 7, 8,
        9, 10, 11, 12,  13, 14, 15, 16
    });
    Tensor b({2, 2, 2, 2}, {
        16, 15, 14, 13,  12, 11, 10, 9,
        8, 7, 6, 5,      4, 3, 2, 1
    });
    check_tensor(a + b, {2, 2, 2, 2}, std::vector<double>(16, 17.0));
}

TEST_CASE("Tensor: subtraction works for owning tensors") {
    Tensor a({2, 3}, {7, 8, 9, 10, 11, 12});
    Tensor b({2, 3}, {1, 2, 3, 4, 5, 6});

    check_tensor(a - b, {2, 3}, {6, 6, 6, 6, 6, 6});
}

TEST_CASE("Tensor: subtraction follows view strides") {
    Tensor a({2, 6}, {12, 11, 10, 9, 8, 7,
                       6, 5, 4, 3, 2, 1});
    Tensor b({2, 6}, {1, 2, 3, 4, 5, 6,
                       7, 8, 9, 10, 11, 12});

    Tensor difference = a.slice(Sli{0, 2, 1}, Sli{1, 6, 2})
                      - b.slice(Sli{0, 2, 1}, Sli{1, 6, 2});
    check_tensor(difference, {2, 3}, {9, 5, 1, -3, -7, -11});
}

TEST_CASE("Tensor: subtraction rejects mismatched shapes") {
    Tensor a({2, 3});
    Tensor b({3, 2});
    CHECK_THROWS_AS((void)(a - b), std::invalid_argument);
}

TEST_CASE("Tensor: compound addition updates the left tensor") {
    Tensor a({2, 2}, {1, 2, 3, 4});
    Tensor b({2, 2}, {5, 6, 7, 8});

    Tensor& result = (a += b);
    CHECK(&result == &a);
    check_tensor(a, {2, 2}, {6, 8, 10, 12});
}

TEST_CASE("Tensor: compound addition through a view mutates the source") {
    // The mirror of the subtraction case below: += taking the general
    // (non-contiguous) path, rather than +='s fast path above.
    Tensor a({2, 3}, {10, 20, 30, 40, 50, 60});
    Tensor b({3, 2}, {1, 2, 3, 4, 5, 6});
    Tensor view = a.swapaxes(0, 1);

    view += b;
    check_tensor(a, {2, 3}, {11, 23, 35, 42, 54, 66});
}

TEST_CASE("Tensor: compound subtraction updates the left tensor (fast path)") {
    // The mirror of the view case above: -= taking the fast (contiguous)
    // path, rather than -='s general path below.
    Tensor a({2, 2}, {10, 20, 30, 40});
    Tensor b({2, 2}, {1, 2, 3, 4});

    Tensor& result = (a -= b);
    CHECK(&result == &a);
    check_tensor(a, {2, 2}, {9, 18, 27, 36});
}

TEST_CASE("Tensor: compound subtraction updates a view in logical order") {
    Tensor a({2, 3}, {10, 20, 30, 40, 50, 60});
    Tensor b({3, 2}, {1, 2, 3, 4, 5, 6});
    Tensor view = a.swapaxes(0, 1);

    view -= b;
    check_tensor(a, {2, 3}, {9, 17, 25, 38, 46, 54});
}

TEST_CASE("Tensor: compound addition rejects mismatched shapes") {
    Tensor a({2, 3});
    Tensor b({3, 2});
    CHECK_THROWS_AS(a += b, std::invalid_argument);
}

TEST_CASE("Tensor: compound subtraction rejects mismatched shapes") {
    Tensor a({2, 3});
    Tensor b({3, 2});
    CHECK_THROWS_AS(a -= b, std::invalid_argument);
}

TEST_CASE("Tensor: scalar multiplication works in both operand orders") {
    Tensor a({2, 3}, {1, 2, 3, 4, 5, 6});

    check_tensor(a * 2.5, {2, 3}, {2.5, 5, 7.5, 10, 12.5, 15});
    check_tensor(2.5 * a, {2, 3}, {2.5, 5, 7.5, 10, 12.5, 15});
}

TEST_CASE("Tensor: scalar multiplication follows view strides") {
    Tensor a({2, 6}, {1, 2, 3, 4, 5, 6,
                       7, 8, 9, 10, 11, 12});
    Tensor scaled = a.slice(Sli{0, 2, 1}, Sli{1, 6, 2}) * -2.0;

    check_tensor(scaled, {2, 3}, {-4, -8, -12, -16, -20, -24});
}

TEST_CASE("Tensor: compound scalar multiplication updates the tensor in place") {
    Tensor a({2, 2}, {1, 2, 3, 4});

    Tensor& result = (a *= 2.0);
    CHECK(&result == &a);
    check_tensor(a, {2, 2}, {2, 4, 6, 8});
}

TEST_CASE("Tensor: compound scalar multiplication through a view mutates the source") {
    Tensor a({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor view = a.swapaxes(0, 1);

    view *= 10.0;
    check_tensor(a, {2, 3}, {10, 20, 30, 40, 50, 60});
}

TEST_CASE("Tensor: compound scalar multiplication respects a slice's offset and stride") {
    Tensor v({6}, {1, 2, 3, 4, 5, 6});
    Tensor s = v.slice(Sli{1, 6, 2}); // view onto elements at flat 1,3,5: {2,4,6}

    s *= 100.0;
    check_tensor(v, {6}, {1, 200, 3, 400, 5, 600});
}

// ── copy: materializing a view into an independent, contiguous tensor ───────

TEST_CASE("Tensor: copy materializes a swapaxes view into a contiguous tensor") {
    Tensor m({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor view = m.swapaxes(0, 1);
    Tensor c = view.copy();

    check_tensor(c, {3, 2}, {1, 4, 2, 5, 3, 6});
    CHECK(as_vector(c.strides()) == std::vector<Index>{2, 1}); // now naturally contiguous
}

TEST_CASE("Tensor: copy of a strided slice is contiguous and independent") {
    Tensor v({6}, {0, 1, 2, 3, 4, 5});
    Tensor s = v.slice(Sli{0, 6, 2}); // view onto {0, 2, 4}
    Tensor c = s.copy();

    check_tensor(c, {3}, {0, 2, 4});
    CHECK(as_vector(c.strides()) == std::vector<Index>{1});

    c(0) = -1.0;
    CHECK(v(0) == doctest::Approx(0.0)); // source untouched
}

TEST_CASE("Tensor: copy is independent of its source (unlike copy-construction)") {
    // Contrasts directly with "copying a Tensor aliases the same storage"
    // above: Tensor's copy CONSTRUCTOR aliases, but the copy() METHOD does
    // not -- that's the whole reason it exists (mirrors NumPy's own
    // F[v,v].copy() idiom for escaping a view before further in-place work).
    Tensor a({2, 2}, {1, 2, 3, 4});
    Tensor b = a.copy();

    b(0, 0) = 999.0;
    CHECK(a(0, 0) == doctest::Approx(1.0));
}

// ── elementwise tensor*tensor (Hadamard product) ─────────────────────────────

TEST_CASE("Tensor: elementwise multiplication works for owning tensors") {
    Tensor a({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor b({2, 3}, {6, 5, 4, 3, 2, 1});

    check_tensor(a * b, {2, 3}, {6, 10, 12, 12, 10, 6});
}

TEST_CASE("Tensor: elementwise multiplication follows view strides") {
    Tensor a({2, 6}, {1, 2, 3, 4, 5, 6,
                       7, 8, 9, 10, 11, 12});
    Tensor b({2, 3}, {1, 0, -1, 2, 0, -2});

    Tensor product = a.slice(Sli{0, 2, 1}, Sli{1, 6, 2}) * b;
    // a's slice picks columns 1,3,5 per row: {2,4,6 / 8,10,12}
    check_tensor(product, {2, 3}, {2, 0, -6, 16, 0, -24});
}

TEST_CASE("Tensor: elementwise multiplication rejects mismatched shapes") {
    Tensor a({2, 3});
    Tensor b({3, 2});
    CHECK_THROWS_AS((void)(a * b), std::invalid_argument);
}

// ── scalar division ───────────────────────────────────────────────────────────

TEST_CASE("Tensor: scalar division divides each element") {
    Tensor a({2, 3}, {2, 4, 6, 8, 10, 12});
    check_tensor(a / 2.0, {2, 3}, {1, 2, 3, 4, 5, 6});
}

TEST_CASE("Tensor: scalar-divided-by-tensor takes the reciprocal direction") {
    Tensor a({2, 2}, {1, 2, 4, 5});
    check_tensor(12.0 / a, {2, 2}, {12, 6, 3, 12.0/5.0});
}

TEST_CASE("Tensor: scalar division follows view strides") {
    Tensor a({2, 6}, {2, 4, 6, 8, 10, 12,
                       14, 16, 18, 20, 22, 24});
    Tensor divided = a.slice(Sli{0, 2, 1}, Sli{1, 6, 2}) / 2.0;
    check_tensor(divided, {2, 3}, {2, 4, 6, 8, 10, 12});
}

TEST_CASE("Tensor: compound scalar division updates the tensor in place") {
    Tensor a({2, 2}, {2, 4, 6, 8});

    Tensor& result = (a /= 2.0);
    CHECK(&result == &a);
    check_tensor(a, {2, 2}, {1, 2, 3, 4});
}

TEST_CASE("Tensor: compound scalar division through a view mutates the source") {
    Tensor a({2, 3}, {10, 20, 30, 40, 50, 60});
    Tensor view = a.swapaxes(0, 1);

    view /= 10.0;
    check_tensor(a, {2, 3}, {1, 2, 3, 4, 5, 6});
}

// ── power ────────────────────────────────────────────────────────────────────

TEST_CASE("Tensor: power raises each element to the given exponent") {
    Tensor a({2, 3}, {1, 2, 3, 4, 5, 6});

    check_tensor(a^2, {2, 3}, {1, 4, 9, 16, 25, 36});
    check_tensor(a^0, {2, 3}, {1, 1, 1, 1, 1, 1});
    check_tensor(a^1, {2, 3}, {1, 2, 3, 4, 5, 6});
}

TEST_CASE("Tensor: power follows view strides") {
    Tensor a({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor squared = a.swapaxes(0, 1)^2;

    check_tensor(squared, {3, 2}, {1, 16, 4, 25, 9, 36});
}

// ── sum ──────────────────────────────────────────────────────────────────────

TEST_CASE("Tensor: sum reduces a contiguous tensor to a scalar") {
    Tensor a({2, 3}, {1, 2, 3, 4, 5, 6});
    CHECK(a.sum() == doctest::Approx(21.0));
}

TEST_CASE("Tensor: sum follows view strides") {
    Tensor a({2, 6}, {1, 2, 3, 4, 5, 6,
                       7, 8, 9, 10, 11, 12});
    Tensor view = a.slice(Sli{0, 2, 1}, Sli{1, 6, 2}); // {2,4,6 / 8,10,12}
    CHECK(view.sum() == doctest::Approx(42.0));
}

TEST_CASE("Tensor: sum matches the MP2-style pairwise-multiply-then-sum pattern") {
    // The exact shape calc_rhf_MP2_E now uses: (A * B * C).sum().
    Tensor a({2, 2}, {1, 2, 3, 4});
    Tensor b({2, 2}, {2, 2, 2, 2});
    Tensor c({2, 2}, {1, 1, 1, 1});

    CHECK((a * b * c).sum() == doctest::Approx(20.0)); // 2*(1+2+3+4)
}

// ── Fixed-size shape/strides (MAX_DIM) ──────────────────────────────────────
// shape/strides live in MAX_DIM-long arrays; only the first ndim entries are meaningful, and shape()/strides()
// must expose exactly those. These cover the rank-changing view ops and the MAX_DIM boundary.

TEST_CASE("Tensor: rank above MAX_DIM is rejected, MAX_DIM itself works") {
    CHECK_THROWS_AS((void)Tensor({1, 1, 1, 1, 1, 1, 1}), std::invalid_argument);

    Tensor t6({2, 1, 3, 1, 2, 2}); // rank 6, the largest the code uses (noddy triples' T3)
    for (Index i = 0; i < t6.size(); ++i) t6.flat_index(i) = 0.5 * i;
    CHECK(t6.ndim() == 6);
    CHECK(std::ssize(t6.shape()) == 6);
    CHECK(t6(1, 0, 2, 0, 1, 1) == doctest::Approx(t6.flat_index(23)));

    // rank 6 -> 2 -> 6 round trip keeps every element
    const Tensor flat = t6.reshape(4, 6);
    CHECK(as_vector(flat.shape()) == std::vector<Index>{4, 6});
    CHECK(as_vector(flat.strides()) == std::vector<Index>{6, 1});
    const Tensor back = flat.reshape(2, 1, 3, 1, 2, 2);
    for (Index i = 0; i < t6.size(); ++i) CHECK(back.flat_index(i) == doctest::Approx(t6.flat_index(i)));
}

TEST_CASE("Tensor: select on a middle axis shifts the later axes and shrinks ndim") {
    Tensor t({2, 3, 4, 5});
    for (Index i = 0; i < t.size(); ++i) t.flat_index(i) = i;

    const Tensor s = t.select(1, 2); // drop axis 1 at index 2 -> shape (2, 4, 5)
    CHECK(s.ndim() == 3);
    CHECK(as_vector(s.shape()) == std::vector<Index>{2, 4, 5});
    CHECK(as_vector(s.strides()) == std::vector<Index>{60, 5, 1});
    for (Index a = 0; a < 2; ++a)
    for (Index c = 0; c < 4; ++c)
    for (Index d = 0; d < 5; ++d)
        CHECK(s(a, c, d) == doctest::Approx(t(a, 2, c, d)));

    // down to rank 0 (a scalar view) and its spans stay exactly ndim long
    const Tensor scalar = t.select(0, 1).select(0, 2).select(0, 3).select(0, 4);
    CHECK(scalar.ndim() == 0);
    CHECK(scalar.shape().empty());
    CHECK(scalar.size() == 1);
    CHECK(scalar() == doctest::Approx(t(1, 2, 3, 4)));
}
