cmake_minimum_required(VERSION 3.14)

project(MySCF LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(Eigen3_DIR "/localscratch/kbui/CPP/eigen_install/share/eigen3/cmake")
find_package(Eigen3 REQUIRED)

set(LIBINT_ROOT "/localscratch/kbui/CPP/libint_install")

option(SCF_BUILD_TESTS "Build unit tests" ON)

add_library(scf_core STATIC
    src/build_JK.cc
    src/integrals.cc
    src/wfn.cc
    src/rhf.cc
    src/uhf.cc
    src/mp2.cc
    src/transmo.cc
    src/einsum.cc
    src/utils.cc
)

target_include_directories(scf_core PUBLIC
    ${LIBINT_ROOT}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_directories(scf_core PUBLIC ${LIBINT_ROOT}/lib)
target_link_libraries(scf_core PUBLIC Eigen3::Eigen int2)

target_compile_options(scf_core PUBLIC
    -O3
    -g
    -fno-omit-frame-pointer
)

add_executable(SCF src/main.cc)
target_link_libraries(SCF PRIVATE scf_core)

if(SCF_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
