cmake_minimum_required(VERSION 3.14)

project(ESTZero LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(Eigen3_DIR "/localscratch/kbui/CPP/eigen_install/share/eigen3/cmake")
find_package(Eigen3 REQUIRED)

set(LIBINT_ROOT "/localscratch/kbui/CPP/libint_install")

set(OPENBLAS_ROOT "/localscratch/kbui/CPP/OpenBLAS")
find_path(OPENBLAS_INCLUDE_DIR cblas.h
    HINTS ${OPENBLAS_ROOT}/include ${OPENBLAS_ROOT}
    REQUIRED
)
find_library(OPENBLAS_LIBRARY NAMES openblas
    HINTS ${OPENBLAS_ROOT}/lib ${OPENBLAS_ROOT}
    REQUIRED
)

option(EST_BUILD_TESTS "Build unit tests" ON)

add_library(est_core STATIC
    estzero/est/jk_build.cc
    estzero/est/integrals.cc
    estzero/est/wfn.cc
    estzero/est/rhf.cc
    estzero/est/uhf.cc
    estzero/est/mp2.cc
    estzero/est/mp3.cc
    estzero/est/transmo.cc
    estzero/est/cc_spin.cc
    estzero/est/cc_spat.cc
    estzero/est/cc_build.cc
    estzero/est/cc_iter.cc
    estzero/est/cc_triples.cc
    estzero/linalg/einsum.cc
)

target_include_directories(est_core PUBLIC
    ${LIBINT_ROOT}/include
    ${OPENBLAS_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/estzero/est
    ${CMAKE_CURRENT_SOURCE_DIR}/estzero/linalg
    ${CMAKE_CURRENT_SOURCE_DIR}/estzero/utils
)
target_link_directories(est_core PUBLIC ${LIBINT_ROOT}/lib)
target_link_libraries(est_core PUBLIC Eigen3::Eigen int2 ${OPENBLAS_LIBRARY})

target_compile_options(est_core PUBLIC
    -O3
    -g
    -fno-omit-frame-pointer
    -march=native
)

add_executable(ESTZero estzero/est/main.cc)
target_link_libraries(ESTZero PRIVATE est_core)

add_executable(verify_ccsd estzero/est/verify_ccsd.cc)
target_link_libraries(verify_ccsd PRIVATE est_core)

if(EST_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()