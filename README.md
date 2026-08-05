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

The test used to track through different versions is benzene/def2-svp (nbf=114), as it has a sufficiently large number of basis functions for testing.

**Baseline - v0.1.0**: in-core 4-index ERIs (no permutational symmetry, no screening), dense N^4 JK build, DIIS, symmetric orthogonalization, core guess.

| version | Total wall (s) | ERI build (s) | JK build (per iter) (ms) | N_SCF_iter | Note |
|---|---|---|---|---|---|
| v0.1.0 | 30.797 | 9.752 | 1313.2 | 15 | Baseline |
| v0.1.1 | 22.379 | 9.564 | 799.2 | 16 | Moved eri index helpers defintion to header file to reduce overhead |
| v0.2.0 | 14.044 | 1.342 | 792.2 | 16 | Build ERI now loops over shell quartets (~8x speed up) |
| v0.3.0 | 7.822 | 1.390 | 400.3 | 16 | Added scattering for JK building (similar ERI looping, scatter J twice and K four times, and symmetry) |
| v0.4.0 | 7.774 | 1.336 | 400.8 | 16 | Reorganize code around wavefunction object (no algorithm optimization) |
