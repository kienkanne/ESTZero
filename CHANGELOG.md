# CHANGELOG

All version updates to this project will be documented in this file. The performance comparison between versions lives in `README.md`.

## [v3.0.0] Added GEMM (General Matrix Multiply) via Eigen for `einsum`

- `einsum` now has a route to build Eigen matrices to call for GEMM (General Matrix Multiply), which works for most contractions. Those who don't fall back to the original odometer scheme.
- The speed up is up to 8.46x for larger systems (MP3, pyrimidine/6-31g (nbf=62)). The previous odometer, despite being optimized with the O(1) complexity to walk naturally along the strides, did not have any vectorization, and didn't come close to GEMM. The GEMM setup, however, require the contiguous buffer from Tensor, and in some cases it creates copies, so the memory footprint is higher.
- The GEMM route also have quite high memory overhead per `einsum` call as it has to set up maps, strings, the correct indices, and permutate to finally call GEMM.
- The perbutative triples calculation in CCSD(T) still takes a significant amount of time due to many small `einsum` call in an explicit loop built following the vikings algorithm to reduce memory footprint. For a large system like pyrimidine/sto-3g the performance degraded: 19.197s in v2.4.2 to 21.243s in v2.5.0 (PertT timings). For a small system (o2/sto-3g, nbf=10, n_alpha=9, n_beta=7), the degradation was even worse: 0.180s in v2.4.2 to 0.327s in v2.5.0 (PertT timings).

## [v2.4.2] Added pairwise evaluation for `einsum`

- Three CCSD contractions contain three operands (`ijab,ia,jb->`, `ie,ma,mbej->ijab`, `ie,mb,maje->ijab`), and the current implementation of `einsum` contracts all indices at the same time, but evaluating them pairwise would be less FLOPs. The pairwise summation is added for `einsum`, and it does so strictly from left to right without searching for an optimal path, which is overkill for just those three contractions. The three contractions that involved 3 operands were re-ordered manually to minimize the arithemtic.
- For RHF CCSD on propargyl/sto-3g (nbf=34, ndocc=21): Contracting `ie,ma,mbej->ijab` all at once takes $3O^3V^3$ (61,039,251) FLOPs, and doing it pairwise with reordering `ie,mbej,ma->ijab` is $2(O^2V^3 + O^3V^2)$ (5,067,972) FLOPs. This (and another triple contraction re-ordering with about the same FLOPs saving) actually saved about ~135ms per CCSD iteration for this system compared to v2.3.2 (1234.5ms to 1099.3ms).

## [v2.4.1] Remove contractions with a zero tensor for MP3

- In the previous version, MP3 was wired in directly using the CCSD intermediates and amplitudes builder. However, since `t1` and `tau` for MP3 are zero tensors, a lot of the contractions were unncessary. The equations for MP3 was written out explicitly, copying the equations from CCSD, but dropping all the zero terms to avoid those arithmetics. The MP3 energy calculation step improved signficantly, as run time dropped from to 26.0s (v2.4.0) to 16.9s for pyrimidine/6-31g.

## [v2.4.0] Added Third-order Moller-Plesset (MP3) for both RHF & UHF

- "MBPT(3) is obtained by performing one CC iteration where all nonlinear terms are skipped" [Stanton:1991:4334], which is the same paper that provided the CCSD equations. The MP3 module was added by initializing `t1` (as 0) and `t2` in a similar fashion to CCSD first iteration, but all the intermediates are built with `tau` as an 0 tensor, because `tau` here is just `t2` (since `t1` is 0), so those are the quadratic `t2` terms. The energy expression, however, still needs `tau` for evaluation, and `t2` is passed in there instead.
- The first version here just initializes `tau` as 0, so the equations that require it still does the 0 arithemtic. Unlike CC2, MP3 is a more well validated method, so an attempt to avoid the unnecessary contractions will be attempeted next patch.
- MP2 energy can also be easily calculated by just not updating `t2`, and is printed out to validate against Psi4 also.

## [v2.3.2] Added Perbutative Triples to the "vikings" algorithm for RHF CCSD(T)

- Implemented the "vikings" algorithm for perbutative triples for RHF, which is the similar formulation but spin-adapted. Peak RSS also stayed at normal levels: 62.3 MB for pyrimidine/sto-3g.
- This implementation for RHF also adapted the equations from [PyCC's `cctriples.py` module](https://github.com/CrawfordGroup/pycc/blob/main/pycc/cctriples.py), similar to the "vikings" algorithm for UHF.

## [v2.3.1] Changed Perbutative Triples to the "vikings" algorithm for UHF CCSD(T)

- Implemented the "vikings" algorithm for perbutative triples for UHF, which never explicitly store the rank 6 tensors as the "noddy" algorithm did. The loops are written explicitly, and at each index combination (i,j,k) the connected-T3 batch is created temporarily for contractions.
- Peak RSS returned to normal level at 60.7 MB for propargyl/sto-3g, which is much expected. Computation time also improved.
- The implementation is a line-by-line direct port from [PyCC's `cctriples.py` module](https://github.com/CrawfordGroup/pycc/blob/main/pycc/cctriples.py), which cites the original equations in Helgaker-Jorgensen-Olsen "vikings" formulation.

## [v2.3.0] Added Perbutative Triples for UHF, giving CCSD(T)

- Added the perbutative triples energy calculation from the converged `t1` and `t2` amplitudes, which is calculated once when CCSD converges. The first implementation is the "noddy" version, which is a direct line-by-line port from [Psi4Numpy's spin-orbital CCSD_T module](https://github.com/CrawfordGroup/ProgrammingProjects/tree/master/Project%2306), which cites [Crawford's programming project #6](https://github.com/psi4/psi4numpy/blob/master/Coupled-Cluster/Spin_Orbitals/CCSD/CCSD_T.py)
- The "noddy" implementation explicitly loops through and adds each term one by one (different permutations of the `ijkabc` rank 6 tensor), which doesn't exploit any symmetry and is very inefficient in wall time and especially memory, which blows up to ~1.5 GB for the small propargyl system. 
- The benchmark module only runs CCSD(T) because it produces the CCSD energy baseline too, and this quantity (as well as the full CCSD(T) energy) is parsed and compared with the two energies produced by Psi4.

## [v2.2.0] Wired in DIIS Acceleration for CCSD

- The `diis()` module already existed and used by the SCF algorithm to accelerate convergence, and the same machinery is wired in for the convergence of the amplitudes `t1` and `t2`. The error vectors are flatten and concatenate in a similar fashion to the two error vectors for alpha and beta in UHF SCF. An additional guard is added so that the error vector list size does not exceed the error vector size itself, which would make them linearly dependent, and the produced coefficients are degenerate and incorrect. This is possible for the tinest systems like `h2/sto-3g`.
- The DIIS addition accelerated both RHF and UHF CCSD signficantly, cutting the number of iterations by HALF, and so as the wall time (see `README.md`)

## [v2.1.0] Added Restricted HF Coupled Cluster Singles & Doubles (CCSD)

- Fully working RHF CCSD module. The RHF counterpart turned out to be much more complex to implement as there isn't a canonical contraction scheme like in UHF, which are the equations in [Stanton:1991:4334]. The version implemented here references from the scheme from Psi4Numpy and PyCC.
- Main references: [Psi4Numpy](https://github.com/psi4/psi4numpy/tree/master/Coupled-Cluster), [PyCC](https://github.com/CrawfordGroup/pycc/tree/main/pycc).

## [v2.0.1] Added UHF CCD (no singles)

- This addition is trivial because `t1` just doesn't get an update. Technically, it can be optimized by dropping unnecessary contractions on the empty `t1` tensor, but this method is limited in accuracy for UHF CCD, and Psi4 does not even implement it. For this reason the optimal CCD path is not coded, it won't be reported as well.

## [v2.0.0] Added Unrestricted HF Coupled Cluster Singles & Doubles (CCSD)

- Fully working UHF CCSD module, with spin-orbital CCSD equations adopted from [Stanton:1991:4334]. Equations all written using the `Tensor` and `einsum` underlying infrastructure built in the last few updates, which include multiple contractions and arthimetic operations on tensors (i.e. `Fae`, `Fmi`, `Fme`, `Wmnij`, `Wabef`, `Wmbej`, and the amplitude equations).
- Main references: [Psi4Numpy](https://github.com/psi4/psi4numpy/tree/master/Coupled-Cluster), [PyCC](https://github.com/CrawfordGroup/pycc/tree/main/pycc).

## [v1.6.1] Fixed `einsum` accumulate bug. Added validation for CCSD energy (first iteration) against MP2

- `einsum` had a bug which made `piqi->pq` failed in `build_FN`. When the strides of terms are calculated, they must accumualte, not overwrite, for cases of duplicate label.
- An additional executable is added: `verify_ccsd`. This initializes the t1 and t2 amplitudes (first CCSD iteration) and compute the energy, which must be equal to the conventional MP2 energy evaluation (difference 8.527e-14). This validation is important enough that it earns an executable for illustration, instead of being hidden in the `tests/` module.

## [v1.6.0] Added full spin MO tensor builder, renamed variables and functions for consistency.

- `build_spin_ASprqs()` is added to build the anti-symmetrized spin MO tensor in physicist notation (UHF-CCSD prep). New MO transformations were added to obtain the fully transform ERI tensor for both occupied and virtual, instead of selective out as in MP2 (only needs `iajb`). The chemist notation was used in `integrals.cc`, so every algorithm involved implcitily used it too, including MP2. CCSD equations are mostly written in physicist notation, and this conversion is flagged explicitly.
- Renamed variables and functions to explicitly express the notation type and spin. `I*` (i.e. `Iprqs`, `Ipqrs_AA`) now represents ERI in the MO basis for post-HF methods; `eri_tensor` or `eri_vector` is in AO basis. Spin is appended to the end and capitalized (`AA`, `BB`, `AB`). The MO transformation methods now explicitly say `spin` or `spat`, and the tensors coming from `spin` are annotated with the spin suffix.


## [v1.5.5c] Used `Tensor` operators entirely for MP2 energy evaluation

- The MP2 energy evaluation formulas are actually very specific, and they are just actually tensor pairwise mupltications with a final summation. Follow the previous version idea of using `Tensor` operators to replace `einsum`, all operations in MP2 energy evaluation now uses these operators purely to exploit fast contiguous paths whenever possible. A `Tensor::sum()` method is added for the final summation.
- However, performance is basically identical to v1.5.2. It shows that the number of arithmetic is identical and already fully exploited. It is the memory allocation steps of the built `Tensor` that cost the set up time, and it doesn't matter which tool is doing the arithemtic, whether it is `operator*` in `Tensor` or `einsum`.

## [v1.5.5b] Rerouted `einsum` for MP2 energy evaluation

- This mini-patch try to avoid using the triple contraction `einsum` (i.e. `iajb,iajb,iajb->`) and only keep it to two operands at most by explicitly creating intermediate tensors. Some of these intermediate steps can utilize `Tensor` operators (pairwise multiply or power) to optimize the calculation via the fast contiguous route offered by these methods.
- However, RHF MP2 performance did not improve, perhaps slightly worse: 227.0ms vs 215.5ms in v1.5.5a for benzene/def2-svp. It's true that the current einsum is not optimized for cases more than 2 operands, but for `iajb,iajb,iajb->` specifically, it's still 4 indices, so this is not the source of the problem. Additionally, in a previous update that replaced explicit loops for einsum in `transmo.cc` (22.564s in v1.1.0 -> 22.287s in v.1.3.0), no performance drop was observed because `einsum` performs the same amount of arithmetic for the contractions as the loops. 
- The drop for MP2 energy evaluation only showed up because the process itself is already extremely fast, and the fixed cost is setting up `einsum`, which is at the ms level. The most costly step of for setting up is actually not inside `einsum`, but allocating memory for `Tensor`, which happens repeatedly in the MP2 energy evaluation.
- UHF MP2, however, seen some noticeable improvement. It utilizes the operators by `Tensor` much better in the square operations in UHF MP2 instead of `einsum`, and also avoids additional memory allocation steps.

## [v1.5.5a] Wired in `einsum` for MP2 energy evaluation

- The only algorithm in the source code that can utilize the new `Tensor` operators and view methods is the MP2 energy evaluation, and this mini-patch is testing this new feature on live code. This is a cheapest possible place to test out a primitive CCSD needs anyways, because this step takes very little time (few ms), so overall wall time won't be hit as much.
- As expected, the unoptimized einsum heavily hits this step: 215.5ms vs 5.3ms (v1.5.5) for benzene/def2-svp. *This ~40x slow down is mainly because the current `einsum` implementation doesn't have pairwise evaluation, so the triple contractions in MP2 energy is evaluated simultaneously (i.e. `iajb,iajb,iajb->`)* Correction: (see v1.5.5b).
- The old version written explicitly in loops is kept for reference with the tag `_old`.

## [v1.5.5] Added arithmetic operators for `Tensor`, generalize `odometer`

- Basic arithmetic binary/unary operators (+/-/\*\) (assignment included: +=/-=/\*\=) are added for `Tensor`. These operations all have two paths, one being the fast path that walks directly along the flat index if the operands are contiguous (strides matching default). The general path is used for views with modified strides.
- The `odometer`, which was previously implemented as a helper for `einsum`, is moved to a separate header because of its generality. It's now used extensively by `Tensor` for the general path operations to compute the correct flat index cheaply with its O(1) complexity.
- None of the current source code uses these operations yet, again this is prepration for CCSD equations. Performance is similar to `v1.5.3`, so no update in `README.md`

## [v1.5.4] Fixed `einsum` for operating on modified views of `Tensor`

- `einsum` computed the strides for the operands on its own without caring about the `strides` attribute of each, which is added in the previous patch. This did not cause any issue as every `Tensor` in the main source code is freshly created and no `Tensor::swapaxes` or `Tensor::slice` views were used, so `strides` remained the default. Only adding specific tests show this, and this is important for CCSD later on where `Tensor` views are passed around a lot.
- Fixed: `terms_strides` (table of strides for all terms for all indices) is now computed correctly from `Tensor.strides`, and `odometer` is introduced to correctly increment the global indices and the flat indices for all terms, keeping the complexity O(1).
- The performance did not change compared to v1.5.3 and is not reported in `README.md`.

## [v1.5.3] Removed bounds-checked `.at()` from `Tensor` indexing

- `Tensor::operator()` and `flat_index` used `std::vector::at()` for every element access after the `v1.5.2` shared_ptr rewrite, adding a bounds-check branch and stride loop to every read/write, including inside the ERI tensor build, `build_JK`, and the `einsum` MO transform. Every call site already satisfies in-bounds indices, so `.at()` is replaced with unchecked `operator[]`.
- Restores MP2 (MO transform) to `v1.5.1` performance and further speeds up ERI tensor build and `build_JK` (see `README.md`).

## [v1.5.2] `Tensor` now uses `std::shared_ptr` for underlying data

- `std::shared_ptr` allows for methods involving zero-copy views like slicing or swap axes (added in this patch). This is another preparation for the tensor manipulations in CCSD.
- `TensorView` is no longer needed because the the shared pointer is copied into the temporary vector, not the entire data vector in the original bug.
- Separated `tensor.h` from `einsum.h` for consistency.

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
