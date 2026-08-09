cmake_minimum_required(VERSION 3.14)

project(MySCF LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(Eigen3_DIR "/localscratch/kbui/CPP/eigen_install/share/eigen3/cmake")
find_package(Eigen3 REQUIRED)

set(LIBINT_ROOT "/localscratch/kbui/CPP/libint_install")

option(SCF_BUILD_TESTS "Build unit tests" ON)

add_library(scf_core STATIC
    estzero/est/build_JK.cc
    estzero/est/integrals.cc
    estzero/est/wfn.cc
    estzero/est/rhf.cc
    estzero/est/uhf.cc
    estzero/est/mp2.cc
    estzero/est/transmo.cc
    estzero/linalg/einsum.cc
    estzero/linalg/utils.cc
)

target_include_directories(scf_core PUBLIC
    ${LIBINT_ROOT}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/estzero/est
    ${CMAKE_CURRENT_SOURCE_DIR}/estzero/linalg
    ${CMAKE_CURRENT_SOURCE_DIR}/estzero/utils
)
target_link_directories(scf_core PUBLIC ${LIBINT_ROOT}/lib)
target_link_libraries(scf_core PUBLIC Eigen3::Eigen int2)

target_compile_options(scf_core PUBLIC
    -O3
    -g
    -fno-omit-frame-pointer
)

add_executable(SCF estzero/est/main.cc)
target_link_libraries(SCF PRIVATE scf_core)

if(SCF_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

option(SCF_BUILD_LEGACY "Build legacy benchmarks" OFF)
if(SCF_BUILD_LEGACY)
    add_subdirectory(legacy)
endif()
