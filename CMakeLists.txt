cmake_minimum_required(VERSION 3.14)

project(ESTZero LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(Eigen3_DIR "/localscratch/kbui/CPP/eigen_install/share/eigen3/cmake")
find_package(Eigen3 REQUIRED)

set(LIBINT_ROOT "/localscratch/kbui/CPP/libint_install")

option(EST_BUILD_TESTS "Build unit tests" ON)

add_library(est_core STATIC
    estzero/est/build_JK.cc
    estzero/est/integrals.cc
    estzero/est/wfn.cc
    estzero/est/rhf.cc
    estzero/est/uhf.cc
    estzero/est/mp2.cc
    estzero/est/transmo.cc
    estzero/est/cc_spin.cc
    estzero/linalg/einsum.cc
    estzero/linalg/utils.cc
)

target_include_directories(est_core PUBLIC
    ${LIBINT_ROOT}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/estzero/est
    ${CMAKE_CURRENT_SOURCE_DIR}/estzero/linalg
    ${CMAKE_CURRENT_SOURCE_DIR}/estzero/utils
)
target_link_directories(est_core PUBLIC ${LIBINT_ROOT}/lib)
target_link_libraries(est_core PUBLIC Eigen3::Eigen int2)

target_compile_options(est_core PUBLIC
    -O3
    -g
    -fno-omit-frame-pointer
)

add_executable(ESTZero estzero/est/main.cc)
target_link_libraries(ESTZero PRIVATE est_core)

add_executable(verify_ccsd estzero/est/verify_ccsd.cc)
target_link_libraries(verify_ccsd PRIVATE est_core)

if(EST_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()