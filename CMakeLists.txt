cmake_minimum_required(VERSION 3.14)

# RTX 4090 = compute capability 8.9
set(CMAKE_CUDA_ARCHITECTURES 89)

project(ESTZero LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(Eigen3_DIR "/localscratch/kbui/CPP/eigen_install/share/eigen3/cmake")
find_package(Eigen3 REQUIRED)

set(CUTENSOR_ROOT "/localscratch/kbui/CPP/libcutensor")

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

    estzero/gemm/cutensor_gemm.cc
    estzero/gemm/cublas_gemm.cc
    estzero/gemm/cuda_ladder.cc
    estzero/gemm/avx512.cc
    estzero/gemm/avx_blis.cc
    estzero/gemm/mkn_RxR.cc
    estzero/gemm/mnk_RxR.cc
    estzero/gemm/naive.cc
)

target_include_directories(est_core PUBLIC
    ${LIBINT_ROOT}/include
    ${CUTENSOR_ROOT}/include
    ${OPENBLAS_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/estzero/est
    ${CMAKE_CURRENT_SOURCE_DIR}/estzero/linalg
    ${CMAKE_CURRENT_SOURCE_DIR}/estzero/utils
    ${CMAKE_CURRENT_SOURCE_DIR}/estzero/gemm
)

target_link_directories(est_core PUBLIC ${LIBINT_ROOT}/lib)
target_link_directories(est_core PUBLIC ${CUTENSOR_ROOT}/lib)
target_link_libraries(est_core PUBLIC Eigen3::Eigen int2 ${OPENBLAS_LIBRARY})

target_compile_options(est_core PUBLIC
    -O3
    -g
    -fno-omit-frame-pointer
    -march=native
)

set_source_files_properties(estzero/gemm/cutensor_gemm.cc PROPERTIES LANGUAGE CUDA)
set_source_files_properties(estzero/gemm/cublas_gemm.cc PROPERTIES LANGUAGE CUDA)
set_source_files_properties(estzero/gemm/cuda_ladder.cc PROPERTIES LANGUAGE CUDA)

# ================================================
# cuBLAS backend for einsum's --gemm-scheme cublas (estzero/gemm/cublas_gemm.cc).
#
# cublas gets linked into EVERY est_core-linked executable by this unconditionally, and just having it 
# loaded without any calls inflates process RSS by ~180MB on every single run, regardless of which
# --gemm-scheme is actually selected (see CHANGELOG). That's a tradeoff for having --gemm-scheme cublas 
# available at runtime in the same binary as the CPU schemes.
#
# For a clean CPU-only baseline run, just these two lines have to be commented out. Nothing else needs to 
# change, as cublas_gemm.cc's EST_HAVE_CUBLAS guard makes --gemm-scheme cublas throw a clear runtime error
target_compile_definitions(est_core PUBLIC EST_HAVE_CUBLAS)
target_link_libraries(est_core PRIVATE CUDA::cublas)
# ================================================
# cuTENSOR backend for einsum's --gemm-scheme cutensor (estzero/gemm/cutensor_gemm.cc).

# Same tradeoff/toggle pattern as cuBLAS above. Comment out these two lines for a clean CPU-only build. 
# cutensor_gemm.cc's EST_HAVE_CUTENSOR guard makes --gemm-scheme cutensor throw a runtime error similarly
target_compile_definitions(est_core PUBLIC EST_HAVE_CUTENSOR)
target_link_libraries(est_core PRIVATE cutensor)
# ================================================

add_executable(ESTZero estzero/est/main.cc)
target_link_libraries(ESTZero PRIVATE est_core)

add_executable(verify_ccsd estzero/est/verify_ccsd.cc)
target_link_libraries(verify_ccsd PRIVATE est_core)

add_executable(gemm_bench estzero/gemm/gemm_bench.cc)
target_link_libraries(gemm_bench PRIVATE est_core)

if(EST_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()