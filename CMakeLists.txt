cmake_minimum_required(VERSION 3.14)

# RTX 4090 = compute capability 8.9
set(CMAKE_CUDA_ARCHITECTURES 89)

project(ESTZero LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

# Release (adds -DNDEBUG) is the benchmark build; -O3 -g -march=native come from target_compile_options
# below either way. Debug keeps Eigen's bounds-check assertions on: use it for a separate dev build dir
# (cmake -S . -B build-debug -DCMAKE_BUILD_TYPE=Debug). Asserts cost JK ~14% on benzene/def2-svp.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(Eigen3_DIR "/localscratch/kbui/CPP/eigen_install/share/eigen3/cmake")
find_package(Eigen3 REQUIRED)

set(CUTENSOR_ROOT "/localscratch/kbui/CPP/libcutensor")

# libint_install_O3 is libint2 built as Release with -march=native. The old libint_install was configured
# with an empty CMAKE_BUILD_TYPE (= no -O flag at all); override with -DLIBINT_ROOT=... to A/B against it.
set(LIBINT_ROOT "/localscratch/kbui/CPP/libint_install_O3" CACHE PATH "libint2 install prefix")

set(OPENBLAS_ROOT "/localscratch/kbui/CPP/OpenBLAS")
find_path(OPENBLAS_INCLUDE_DIR cblas.h
    HINTS ${OPENBLAS_ROOT}/include ${OPENBLAS_ROOT}
    REQUIRED
)
find_library(OPENBLAS_LIBRARY NAMES openblas
    HINTS ${OPENBLAS_ROOT}/lib ${OPENBLAS_ROOT}
    REQUIRED
)

include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported OUTPUT error_message)

if(ipo_supported)
    # Enable LTO globally for all targets added after this line
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
else()
    message(WARNING "LTO/IPO is not supported by this compiler: ${error_message}")
endif()

option(EST_BUILD_TESTS "Build unit tests" ON)

add_library(est_core STATIC
    estzero/wfn.cc
    estzero/core/jk_build.cc
    estzero/core/integrals.cc
    estzero/core/rhf.cc
    estzero/core/uhf.cc
    estzero/core/transmo.cc
    estzero/mpn/mp2.cc
    estzero/mpn/mp3.cc
    estzero/cc/cc_driver.cc
    estzero/cc/cc_iter.cc
    estzero/cc/builders/amps.cc
    estzero/cc/builders/intmds.cc
    estzero/cc/builders/pre_iter.cc
    estzero/cc/triples/triples_spin.cc
    estzero/cc/triples/triples_spat.cc
    estzero/linalg/tensor.cc
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
    ${CMAKE_CURRENT_SOURCE_DIR}/estzero
    ${CMAKE_CURRENT_SOURCE_DIR}/estzero/core
    ${CMAKE_CURRENT_SOURCE_DIR}/estzero/mpn
    ${CMAKE_CURRENT_SOURCE_DIR}/estzero/cc
    ${CMAKE_CURRENT_SOURCE_DIR}/estzero/cc/builders
    ${CMAKE_CURRENT_SOURCE_DIR}/estzero/cc/triples
    ${CMAKE_CURRENT_SOURCE_DIR}/estzero/cc/triples/legacy
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

# nvcc defaults to C++17, but tensor.h/einsum.h (included by cutensor_gemm.cc) need C++20 (std::span, concepts).
# Passed as a raw flag because this CMake (3.21) predates CUDA 13 and ignores CMAKE_CUDA_STANDARD for it.
target_compile_options(est_core PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:-std=c++20>)

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

add_executable(ESTZero estzero/main.cc)
target_link_libraries(ESTZero PRIVATE est_core)

add_executable(verify_ccsd estzero/cc/verify_ccsd.cc)
target_link_libraries(verify_ccsd PRIVATE est_core)

add_executable(gemm_bench estzero/gemm/gemm_bench.cc)
target_link_libraries(gemm_bench PRIVATE est_core)

if(EST_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()