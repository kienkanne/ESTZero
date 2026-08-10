# ESTZero: Electronic Structure Theory implementations from scratch

An electronic structure engine built using C++ that is optimized over time. The current repository supports RHF/UHF references and SCF/MP2 methods. Some supplemental methods are also implemented, including `einsum` and `diis`. The integrals are evaluated using `libint2`, and the linear algebra library used is `Eigen`.

The test used to evaluate the performance of the engine is formaldehyde at the def2-tzvp basis set. The accuracy and performance is benchmarked against Psi4. 

Psi4 options:
```bash
e_convergence 1e-8
d_convergence 1e-8
maxiter 200
scf_type pk # no density-fitting
guess core

mp2_type conv
freeze_core false
```

## Benchmarking

A benchmark + accuracy suite lives in `bench/`. It runs the engine over a tiered test suite and checks every energy against a cached Psi4/PK reference.

```bash
# 1. store Psi4 reference energies:
python bench/gen_reference.py

# 2. benchmark + accuracy check (rebuilds, best-of-3 wall time, PASS/FAIL vs Psi4):
python3 bench/benchmark.py
```

## Performance log

This log tracks the update in features and their performance in the most computationally costly steps. Versions without algorithm update for each section is not documented. See `bench/outputs` for full version peformance logs.

> v1.5.0 added the ERI tensor building scheme alongside the original ERI vector building, so for this version both schemes are shown for direct comparison

### RHF

For RHF, the test used to track through different versions is benzene/def2-svp (nbf=114), as it has a sufficiently large number of basis functions for testing. The baseline version, v0.1.0, uses in-core 4-index ERIs (no permutational symmetry, no screening), dense N^4 JK build, DIIS, symmetric orthogonalization, core guess.

#### SCF

| version | Total wall (s) | ERI build (s) | JK build (per iter) (ms) | N_SCF_iter | Peak RSS (MB) | Note |
|---|---|---|---|---|---|---|
| v0.1.0 | 28.736 | 9.789 | 1182.5 | 16 | 177.1 MB | SCF Baseline |
| v0.1.1 | 18.882 | 9.588 | 579.2 | 16 | 177.1 MB | Moved eri index helpers defintion to header file to reduce overhead |
| v0.2.0 | 10.526 | 1.348 | 527.0 | 16 | 176.9 MB | Build ERI now loops over shell quartets (~8x speed up) |
| v0.3.0 | 7.075 | 1.346 | 356.4 | 16 | 177.0 MB | Added scattering for JK building (similar ERI looping, scatter J twice and K four times, and symmetry) |
| v0.4.0 | 7.061 | 1.343 | 355.7 | 16 | 175.5 MB | Reorganize code around wavefunction object (no algorithm optimization) |
| v1.5.0 (vector) | 8.450 | 1.356 | 441.8 | 16 | 176.6 MB | Slightly slower JK build compared to v0.4.0 (reorganization effects) |
| v1.5.0 (tensor) | 5.727 | 2.535 | 193.7 | 16 | 1300.8 MB | ERI build ~2x slower, JK build ~2.3x faster, and RSS ~7.4x as the vector exploited the 8-fold symmetry |

#### MP2

Version v1.0.0 now includes MP2 for RHF. The baseline is the 4 step MO transformation written via explicit loops ~ O(N^5). The calculation runs a base SCF first, so ERI build and JK build performance is similar to versions above.

| version | Total wall (s) | ERI build (s) | JK build (per iter) (ms) |  N_SCF_iter | MO transform(s) | MP2 energy (ms) | Peak RSS (MB) | Note |
|---|---|---|---|---|---|---|---|---|
| v1.0.0 | 29.570| 1.346 | 407.7 | 16 | 21.666 | 6.9 | 607.4 | MP2 Baseline |
| v1.3.0 | 30.093 | 1.350 | 400.5 | 16 | 22.287 | 20.6 | 4279.4 | MO transform rewired through a from-scratch `einsum` (CCSD prep). Peak RSS ~7x (see `CHANGELOG.md`) |
| v1.3.1 | 31.412 | 1.375 | 449.5 | 16 | 22.797 | 21.3 | 2753.1 | Fix `einsum` memory bug (see `CHANGELOG.md`) |
| v1.5.0 (vector) | 32.390 | 1.359 | 459.1 | 16 | 23.635 | 21.2 | 2753.6 | Similar to v1.3.1 |
| v1.5.0 (tensor) | 27.940 | 2.528 | 193.1 | 16 | 22.212 | 20.6 | 1796.1 | Similar speed up to RHF, and RSS much more sufficient because ERI conversion from vector to tensor is not needed |

### UHF

Version 1.1.0 reorganize the benchmark suite to separate the reference and the method; UHF is added as an additional reference to the previous RHF, with the methods being SCF and MP2. The benchmark system for UHF is •CH3 methyl radical/aug-cc-pvtz (nbf=115). The baseline for UHF SCF is similar to RHF SCF version 0.4.0, but with separate alpha and beta terms following the Pople-Nesbet equations.

#### SCF

| version | Total wall (s) | ERI build (s) | JK build* (per iter) (ms) | N_SCF_iter | Peak RSS (MB) | Note |
|---|---|---|---|---|---|---|
| v1.1.0 | 13.998 | 1.020 | 862.6 | 15 | 188.2 MB | UHF SCF Baseline |
| v1.5.0 (vector) | 16.248 | 1.028 | 1012.0 | 16 | 187.9 MB | Slightly slower JK build compared to v1.1.0 (reorganization effects) |
| v1.5.0 (tensor) | 5.727 | 2.535 | 193.7 | 16 | 1300.8 MB | Similar speed up to RHF, but build JK is significantly faster because of both alpha and beta building per iteration |

> *Total time for both alpha and beta calculation time per iteration

#### MP2

UHF MP2 are built on top of the Ca and Cb matrices to transform ERI into the alpha-alpha, beta-beta, and alpha-beta MO integral tensors. Baseline version (v1.1.0): the index transform function used is similar to the one used for RHF MP2 to build alpha-alpha and beta-beta tensors; the alpha-beta block is built from the intermediate tensor after index 2 transformation of the alpha-alpha tensor

| version | Total wall (s) | ERI build (s) | JK build* (per iter) (ms) |  N_SCF_iter | MO transform* (s) | MP2 energy (ms) | Peak RSS (MB) | Note |
|---|---|---|---|---|---|---|---|---|
| v1.1.0 | 21.285 | 1.024 | 905.5 | 15 | 6.635 | 0.8 | 393.1 | UHF MP2 Baseline |
| v1.3.0 | 28.646 | 1.036 | 839.4 | 15 | 14.976 | 2.1 | 4255.5 | Same `einsum` rewrite as RHF MP2 above. Peak RSS ~11x (see `CHANGELOG.md`) |
| v1.3.1 | 29.826 | 1.043 | 939.0 | 15 | 14.655 | 2.3 | 2857.3 | Fixed `einsum` memory bug (see `CHANGELOG.md`) |
| v1.5.0 (vector) | 31.203 | 1.032 | 972.4 | 15 | 15.542 | 2.2 | 2856.6 | Similar to v1.3.1 |
| v1.5.0 (tensor) | 22.084 | 2.213 | 378.6 | 15 | 14.099 | 2.0 | 1471.4 | Similar improvement to RHF MP2 |
