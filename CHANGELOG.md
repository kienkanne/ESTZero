# CHANGELOG

All version updates to this project will be documented in this file. The performance comparison between versions live in `README.md`.

## [v1.5.1] Readded build `Iab` directly from intermediate 2 of `Iaa`

- When `einsum` was introduced in `v1.3.0`, it built `Iab` explictly from `eri_tensor` unlike the previous scheme, affecting UHF MP2 performance substantially. It is recovered here, and performance increased.
- Unify alias for `Eigen::MatrixXd`, and now row-major is explicitly used so it can be copied easily into a `Tensor`.

## [v1.5.0] Added option to build `eri_tensor`

- ERI can now be directly built as a dense tensor instead of a compressed vector with smart canonical indexing that exploited the 8-fold symmetry to save memory. Both options are kept because while the tensor is much less memory efficient, it's more friendly for cache and dense matrix multiplications, which are future optimization targets. Due to the eri vector indexing look up every iteration, it's much less vectorizable and suffer from cache misses.
- For post-SCF methods (MP2/CCSD), the dense ERI tensor is needed anyways, so using the tensor actually saves memory to avoid conversion of the vector to the tensor (see `README.md`)

## [v1.4.1]: Added unit tests for einsum

- Added `tests/test_einsum.cc` covering `einsum` in isolation, on top of the existing golden/property coverage.

## [v1.4.0]: Reorganized folders and namings, renamed project to ESTZero

- Split `src/` into `estzero/{est,linalg,utils}` by scope (SCF/wavefunction, tensor math, misc utilities) and renamed the project to ESTZero. No algorithm change.

## [v1.3.1] Fixed `einsum` memory bug using an intermediate class `TensorView` pointing to `Tensor`

### Added

- Intermediate `TensorView` class that points directly to `Tensor` data, and this lets std::vector owns this lightweight object.

### Notes

- Passing `operands={Cocc1, ao}` into einsum constructs a temporary std::vector, whose members must be owned by the vector and can't be references, so tensors are copied, which heaviliy eats into allocated memory.
- Only einsum inputs are updated to `const std::vector<TensorView>& operands`, while every call site remains the same (`operands={...}`)

## [v1.3.0]: Added `einsum` implementation from scratch, rewired MO transformation

### Added

- The AO→MO integral transform (`src/transmo.cc`) now goes through a from-scratch `einsum` (`src/einsum.{h,cc}`) as four sequential contractions, instead of the hand-written quarter-transform loops. This is prep work for CCSD, where writing explicit loops for dozens of contractions is extremely impractical. 
- A new `Tensor` class is implemented to represent any generic tensor to replace `Eigen::Tensor`, whose dimension must be known at run time.
- The pre-einsum loop implementation is kept as `transform_*_old` in `transmo.cc` for reference and A/B comparison as `einsum` gets optimized.
- Measured RSS dropped as expected: `4279.4 -> 2753.1 MB` (benzene/def2-svp), `4255.5 -> 2857.3 MB` (ch3/aug-cc-pvtz).

### Notes

- **Wall time:** close to unchanged, as expected as the nested algorithm structure is essentially the same, so same FLOP count ish. The slight slowndown could be a constant-factor cost, where `einsum`'s generic per-element loop (`move_idx`, walking every index combination) is much harder for the compiler to vectorize than the old explicit `for` loop.
- **Peak RSS jumped ~7-11x:** traced with temporary peak RSS check on `ch3/aug-cc-pvtz` (nbf=115) and double checked on `benzene/def2-svp` (RHF, ruling out a UHF-specific cause):
    - **Expected, ~1.3 GB**: `build_ao_tensor` expands the compact 8-fold-symmetric ERI vector into a dense 4D Tensor so `einsum` can index it directly. The old loops read straight from the packed vector via `eri_index()` and never had a problem. This is One-time cost per transform call, unavoidable as long as `einsum` needs dense tensors: `115^4 × 8 bytes = 1.334 GB`, measured `187.6 MB -> 1522.0 MB`.
    - **Unexpected, another ~1.3-2.7 GB (actual bug):** `einsum(subscripts, {operands...})` takes `const std::vector<Tensor>&`. At every call site `einsum("pi,pqrs->iqrs", {Cocc1, ao})` the braced list initialization constructs a temporary `std::vector<Tensor>` by **copying** each operand because `ao` is a named lvalue (full copy-construction, not a move) and `std::vector` requirment is that it must owns the elements, and they cannot be references. Measured: `1522.0 MB -> 4248.8 MB` on the very first `einsum` call that consumes `ao` (+2727 MB ≈ 2× `ao`'s own size), then flat for the rest of the transform.

## [v1.2.0]: Added full test suite (oracle-golden and property)

- Added the property-test and oracle-golden test suites (`tests/`), freezing intermediates (S/T/V/ERI, densities, J/K, MO integrals) certified against Psi4.

## [v1.1.0]: Added Unrestricted Hartree-Fock reference (UHF) alongside RHF

- Added UHF (with charge/multiplicity options), benchmarked against Psi4 alongside RHF/MP2.

## [v1.0.0]: Added MP2 for RHF, update benchmark suite

- Added a working RHF MP2 module, and reworked the benchmark suite to extend past SCF energies with human-readable logs.

## [v0.4.0]: Reorganize code around a wavefunction object (no algorithm optimization)

- Reorganized around a `WaveFunction` object; pure refactor, no algorithmic change.

## [v0.3.0]: Added scattering for JK building

- Replaced the dense J/K loop with a scatter over the unique ERI list.

## [v0.2.0]: Build ERI now loops over shell quartets

- ERI build now loops over shell quartets instead of per-index quartets; ~8x faster.

## [v0.1.1]: Eri index helpers defintion moved to header file

- Moved the ERI index helpers into a header to cut call overhead.

## [v0.1.0]: Initial baseline working RHF SCF with `Eigen` and `libint2`

- Baseline working RHF SCF (Eigen + libint2), with a first oracle/baseline test.
