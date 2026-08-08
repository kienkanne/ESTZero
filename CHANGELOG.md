# CHANGELOG

All version updates to this project will be documented in this file. The performance comparison between versions live in `README.md`. This is initiated from v1.2.0, and previous version notes will be recovered later.

## [v1.2.0]: Added `einsum` implementation from scratch, rewired MO transformation

### Added

- The AO→MO integral transform (`src/transmo.cc`) now goes through a from-scratch `einsum` (`src/einsum.{h,cc}`) as four sequential contractions, instead of the hand-written quarter-transform loops. This is prep work for CCSD, where writing explicit loops for dozens of contractions is extremely impractical. 
- A new `Tensor` class is implemented to represent any generic tensor to replace `Eigen::Tensor`, whose dimension must be known at run time.
- The pre-einsum loop implementation is kept as `transform_*_old` in `transmo.cc` for reference and A/B comparison as `einsum` gets optimized.

### Notes

- **Wall time:** close to unchanged, as expected as the nested algorithm structure is essentially the same, so same FLOP count ish. The slight slowndown could be a constant-factor cost, where `einsum`'s generic per-element loop (`move_idx`, walking every index combination) is much harder for the compiler to vectorize than the old explicit `for` loop.
- **Peak RSS jumped ~7-11x:** traced with temporary peak RSS check on `ch3/aug-cc-pvtz` (nbf=115) and double checked on `benzene/def2-svp` (RHF, ruling out a UHF-specific cause):
    - **Expected, ~1.3 GB**: `build_ao_tensor` expands the compact 8-fold-symmetric ERI vector into a dense 4D Tensor so `einsum` can index it directly. The old loops read straight from the packed vector via `eri_index()` and never had a problem. This is One-time cost per transform call, unavoidable as long as `einsum` needs dense tensors: `115^4 × 8 bytes = 1.334 GB`, measured `187.6 MB → 1522.0 MB`.
    - **Unexpected, another ~1.3-2.7 GB (actual bug):** `einsum(subscripts, {operands...})` takes `const std::vector<Tensor>&`. At every call `einsum("pi,pqrs->iqrs", {Cocc1, ao})` the braced list initialization constructs a temporary `std::vector<Tensor>` by **copying** each operand because `ao` is a named lvalue (full copy-construction, not a move). Measured: `1522.0 MB → 4248.8 MB` on the very first `einsum` call that consumes `ao` (+2727 MB ≈ 2× `ao`'s own size), then flat for the rest of the transform.

## [v1.1.0]: Added Unrestricted Hartree-Fock reference (UHF) alongside RHF

## [v1.0.0]: Added MP2 for RHF, update benchmark suite

## [v0.4.0]: Reorganize code around a wavefunction object (no algorithm optimization)

## [v0.3.0]: Added scattering for JK building

## [v0.2.0]: Build ERI now loops over shell quartets

## [v0.1.1]: Eri index helpers defintion moved to header file

## [v0.1.0]: Initial baseline working RHF SCF with `Eigen` and `libint2`