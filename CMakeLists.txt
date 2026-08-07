cmake_minimum_required(VERSION 3.14)

project(MySCF LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(Eigen3_DIR "/localscratch/kbui/CPP/eigen_install/share/eigen3/cmake")
find_package(Eigen3 REQUIRED)

set(LIBINT_ROOT "/localscratch/kbui/CPP/libint_install")

add_executable(SCF
    src/main.cc
    src/build_JK.cc
    src/integrals.cc
    src/wfn.cc
    src/rhf.cc
    src/uhf.cc
    src/mp2.cc
    src/transmo.cc
    src/utils.cc
)

# Specify header and library search paths for BOTH targets
target_include_directories(SCF PRIVATE ${LIBINT_ROOT}/include)
target_include_directories(SCF PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

target_link_directories(SCF PRIVATE ${LIBINT_ROOT}/lib)
target_link_libraries(SCF PRIVATE Eigen3::Eigen int2)

# Add profiling and debug flags
target_compile_options(SCF PRIVATE 
    -O3 
    -g 
    -fno-omit-frame-pointer
)
