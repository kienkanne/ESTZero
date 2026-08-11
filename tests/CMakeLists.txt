add_library(doctest INTERFACE)
target_include_directories(doctest INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty)

add_executable(est_tests
    main_test.cc
    test_utils.cc
    test_build_jk.cc
    test_transmo.cc
    test_mp2.cc
    test_rhf_uhf.cc
    test_integrals.cc
    test_wfn.cc
    test_timer.cc
    test_tensor.cc
    test_einsum.cc
)

target_link_libraries(est_tests PRIVATE est_core doctest)
target_include_directories(est_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Tests/generator resolve input/*.xyz, bench/reference.json and the golden data
# files relative to the repo root, not the build dir.
target_compile_definitions(est_tests PRIVATE
    EST_REPO_ROOT="${CMAKE_CURRENT_SOURCE_DIR}/.."
)

# Precompile the heavy third-party headers (Eigen + libint2) that dominate each
# test TU's compile time. Parsed once into a PCH and reused across all test
# sources, so editing a single test file recompiles in ~1-2s instead of ~7s.
# doctest.h is deliberately NOT here: main_test.cc must #define
# DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN *before* including it, which a force-included
# PCH would pre-empt.
target_precompile_headers(est_tests PRIVATE
    <Eigen/Dense>
    <libint2.hpp>
)

add_test(NAME est_tests COMMAND est_tests)

# Developer tool (not a test): regenerates the frozen golden data from the
# current, benchmark-confirmed code. Run manually, then commit tests/golden/data.
add_executable(gen_golden gen_golden.cc)
target_link_libraries(gen_golden PRIVATE est_core)
target_include_directories(gen_golden PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(gen_golden PRIVATE
    EST_REPO_ROOT="${CMAKE_CURRENT_SOURCE_DIR}/.."
)
# Reuse the test PCH rather than compiling a second identical one.
target_precompile_headers(gen_golden REUSE_FROM est_tests)
