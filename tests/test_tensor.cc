#include "doctest.h"

#include "tensor.h"

namespace {

// doctest has no Tensor comparison; compare element-wise over the flat storage.
void check_tensor(const Tensor& actual, const std::vector<int>& shape,
                   const std::vector<double>& expected) {
    REQUIRE(actual.shape == shape);
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
    CHECK(v.strides == std::vector<int>{1});

    Tensor m({2, 3});
    CHECK(m.shape == std::vector<int>{2, 3});
    CHECK(m.strides == std::vector<int>{3, 1});

    Tensor t3({2, 3, 4});
    CHECK(t3.strides == std::vector<int>{12, 4, 1});
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

TEST_CASE("Tensor: clear empties the backing storage") {
    Tensor v({3}, {1, 2, 3});
    v.clear();
    CHECK_THROWS_AS((void)v.flat_index(0), std::out_of_range);
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
    CHECK(t.shape == std::vector<int>{3, 2});
    CHECK(t.strides == std::vector<int>{1, 3});
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
    REQUIRE(s.shape == std::vector<int>{3});
    CHECK(s(0) == doctest::Approx(0));
    CHECK(s(1) == doctest::Approx(2));
    CHECK(s(2) == doctest::Approx(4));
}

TEST_CASE("Tensor: slice with a negative step reverses a sub-range") {
    // Equivalent to Python's a[4:1:-1] on [10,20,30,40,50] -> [50,40,30].
    // Not contiguous, so again compare via operator(), not flat_index.
    Tensor v({5}, {10, 20, 30, 40, 50});
    Tensor s = v.slice(Sli{4, 1, -1});
    REQUIRE(s.shape == std::vector<int>{3});
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
    CHECK(s.shape == std::vector<int>{2, 2});
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

    REQUIRE(row1.shape == std::vector<int>{1, 3});
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
