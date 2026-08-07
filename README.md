# Self-Consistent-Field engine using C++

An optimized restricted Hartree-Fock (RHF) Self-Consistent-Field (SCF) engine, built using C++. The integrals are evaluated using `libint2`, and the linear algebra library used is `Eigen`.

The test used to evaluate the performance of the engine is formaldehyde at the def2-tzvp basis set. The accuracy and performance is benchmarked against Psi4. `{scf_type: pk}` is used (no density-fitting).

## Benchmarking

A benchmark + accuracy suite lives in `bench/`. It runs the engine over a tiered test suite and checks every energy against a cached Psi4/PK reference.

```bash
# 1. store Psi4 reference energies:
python bench/gen_reference.py

# 2. benchmark + accuracy check (rebuilds, best-of-3 wall time, PASS/FAIL vs Psi4):
python3 bench/benchmark.py
```

## Feature log

This log tracks the update in features and their performance in the most computationally costly steps.

### RHF

For RHF, the test used to track through different versions is benzene/def2-svp (nbf=114), as it has a sufficiently large number of basis functions for testing. The baseline version, v0.1.0, uses in-core 4-index ERIs (no permutational symmetry, no screening), dense N^4 JK build, DIIS, symmetric orthogonalization, core guess.

#### SCF

| version | Total wall (s) | ERI build (s) | JK build (per iter) (ms) | N_SCF_iter | Note |
|---|---|---|---|---|---|
| v0.1.0 | 28.736 | 9.789 | 1182.5 | 16 | SCF Baseline |
| v0.1.1 | 18.882 | 9.588 | 579.2 | 16 | Moved eri index helpers defintion to header file to reduce overhead |
| v0.2.0 | 10.526 | 1.348 | 527.0 | 16 | Build ERI now loops over shell quartets (~8x speed up) |
| v0.3.0 | 7.075 | 1.346 | 356.4 | 16 | Added scattering for JK building (similar ERI looping, scatter J twice and K four times, and symmetry) |
| v0.4.0 | 7.061 | 1.343 | 355.7 | 16 | Reorganize code around wavefunction object (no algorithm optimization) |

The peak memory usage (peak resident set size) in MB for all versions here is about 177MB.

#### MP2

Version v1.0.0 now includes MP2 for RHF. The baseline is the 4 step MO transformation written via explicit loops ~ O(N^5). The calculation runs a base SCF first, so ERI build and JK build performance is similar to versions above.

| version | Total wall (s) | ERI build (s) | JK build (per iter) (ms) |  N_SCF_iter | MO transform(s) | MP2 energy (ms) | Note |
|---|---|---|---|---|---|---|---|
| v1.0.0 | 29.570| 1.346 | 407.7 | 16 | 21.666 | 6.9 | MP2 Baseline |

The peak memory usage (peak resident set size) in MB for all versions here is about 607MB.

### UHF

Version 1.1.0 reorganize the benchmark suite to separate the reference and the method; UHF is added as an additional reference to the previous RHF, with the methods being SCF and MP2. The benchmark system for UHF is •CH3 methyl radical/aug-cc-pvtz (nbf=115). The baseline for UHF SCF is similar to RHF SCF version 0.4.0, but with separate alpha and beta terms following the Pople-Nesbet equations.

#### SCF

| version | Total wall (s) | ERI build (s) | JK build** (per iter) (ms) | N_SCF_iter | Note |
|---|---|---|---|---|---|
| v1.1.0 | 13.998 | 1.020 | 862.6 | 15 | UHF SCF Baseline |

**Total time for both alpha and beta calculation time per iteration

The peak memory usage (peak resident set size) in MB for all versions here is about 188MB.

#### MP2

UHF MP2 are built on top of the Ca and Cb matrices to transform ERI into the alpha-alpha, beta-beta, and alpha-beta MO integral tensors. Baseline version (v1.1.0): the index transform function used is similar to the one used for RHF MP2 to build alpha-alpha and beta-beta tensors; the alpha-beta block is built from the intermediate tensor after index 2 transformation of the alpha-alpha tensor

| version | Total wall (s) | ERI build (s) | JK build** (per iter) (ms) |  N_SCF_iter | MO transform*** (s) | MP2 energy (ms) | Note |
|---|---|---|---|---|---|---|---|
| v1.1.0 | 21.285| 1.024 | 905.5 | 15 | 6.635 | 0.8 | UHF MP2 Baseline |

***Total time of alpha-alpha, beta-beta, and alpha-beta integrals building.

The peak memory usage (peak resident set size) in MB for all versions here is about 393MB.