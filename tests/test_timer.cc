#include "doctest.h"
#include <algorithm>
#include <string>
#include <thread>
#include <chrono>

#include "timer.h"

// timer.h's registry() is a global singleton shared with production code
// (and with every other test in this binary), so tests here use a unique
// name per TEST_CASE and only ever assert on that one entry — never on the
// registry's total size or on entries they didn't create themselves.

namespace {

const timing::Record* find_record(const std::string& name) {
    auto& reg = timing::registry();
    auto it = std::find_if(reg.begin(), reg.end(),
                            [&](const timing::Record& r) { return r.name == name; });
    return it == reg.end() ? nullptr : &(*it);
}

} // namespace

TEST_CASE("timing::add: accumulates seconds and call count under one name") {
    const std::string name = "test_timer_add_accumulate";
    timing::add(name, 0.5, false);
    timing::add(name, 0.25, false);

    const timing::Record* rec = find_record(name);
    REQUIRE(rec != nullptr);
    CHECK(rec->seconds == doctest::Approx(0.75));
    CHECK(rec->calls == 2);
}

TEST_CASE("timing::add: distinct names do not interfere with each other") {
    timing::add("test_timer_add_distinct_a", 1.0, false);
    timing::add("test_timer_add_distinct_b", 2.0, false);

    const timing::Record* a = find_record("test_timer_add_distinct_a");
    const timing::Record* b = find_record("test_timer_add_distinct_b");
    REQUIRE(a != nullptr);
    REQUIRE(b != nullptr);
    CHECK(a->seconds == doctest::Approx(1.0));
    CHECK(b->seconds == doctest::Approx(2.0));
}

TEST_CASE("timing::Scoped: records exactly one call on scope exit") {
    const std::string name = "test_timer_scoped_single_call";
    {
        timing::Scoped _t(name);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    const timing::Record* rec = find_record(name);
    REQUIRE(rec != nullptr);
    CHECK(rec->calls == 1);
    CHECK(rec->seconds >= 0.0);
}

TEST_CASE("timing::Scoped: repeated scopes under the same name accumulate calls") {
    const std::string name = "test_timer_scoped_repeated";
    for (int i = 0; i < 3; ++i) {
        timing::Scoped _t(name);
    }

    const timing::Record* rec = find_record(name);
    REQUIRE(rec != nullptr);
    CHECK(rec->calls == 3);
}

TEST_CASE("timing::peak_rss_kb: returns a positive value") {
    CHECK(timing::peak_rss_kb() > 0);
}
