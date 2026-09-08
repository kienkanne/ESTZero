# CHANGELOG

All version updates to this project will be documented in this file. The performance comparison between versions lives in `README.md`.

## [v3.6.0] Added Schwarz screening for ERI building

- `compute_schwarz_bounds` is added to help with eri building by computing over unique shell PAIRS first, returning a matrix `Q` containing the bounds of the pair. In the main ERI loop, a quartet is dropped when `|(s1 s2|s3 s4)| <= Q(s1,s2)*Q(s3,s4)`, avoiding calling `libint2` for the integrals.
- However, the selected systems for benchmarking are relatively small for SCF (RHF: benzene/def2-svp, UHF: ch3/aug-cc-pvdz), especially the radical methyl where Schwarz screening doesn't do anything because the molecule is tiny. For benzene, 10% of shell quartets were skipped, and the ERI building step is just faster by less than 10%.
- The benchmarking systems was selected based on the wall time of the first version that was introduced, which put a constraint on them being relatively small. Throughout many optimization updates, the wall time of the latests versions have shrinked significantly, so future updates will plan to reorganize benchmarking systems.

## [v3.5.0] Added `cuTENSOR` as an additional backend for `einsum`

- `cuTENSOR` supports binary tensor contractions (`A,B->C`) via `cutensorCreateContraction`/`cutensorContract`, which fits perfectly into this project where there are numerous contractions for post-HF algorithms (the reason why `einsum` was built in the first place). It's considered an `einsum` "GEMM" backend as it still runs through the same string parsing machinery. It's routed through `try_cutensor` instead of `try_gemm` as it doesn't need the flattening steps and permutating back, as the requested shape is provided. This route can be called via `--gemm-scheme cutensor`.
- In terms of raw performance, `cutensor` is expected to outperform the `try_gemm` variants. However, in full benchmarks where total wall time is counted where overhead setup cost adds up, it doesn't have an edge. For example, large systems with few contractions like MP3 `pyrimidine/6-31g` (nbf=62), `cutensor` took 2.927s, while the best `try_gemm` is `cublas_dgemm` at 4.216s for GPU and `eigen` at 4.292s for CPU.
- The massive number of repeated small `einsum` calls in the perbutative triples hurt `cutensor` the most, much worse than `cublas` which also requires moving data to device. Calculating this term for `pyrimidine/sto-3g` (nbf=34) took 72.024s for `cutensor`, 27.067s for `cublas_dgemm` (v3.2.3), and 22.262s for `eigen` (v3.2.0).
- The vikings algorithm for the perbutative triples greatly avoids large memory usage compared to the naive version, but it's currently routed through `einsum` for every contraction, creating a lot of overhead cost that is tanked by every gemm schemes (in fact, the odometer performs the best just in the calculation time of this term at 19.168s (v3.2.0)). The next goal will be rewrite this algorithm by calling GEMM directly instead of routing through `einsum`.

## [v3.3.4a] Made `A_blk` transposed so that the register loading step is contiguous

- A few simple character changes of indices at three lines of `A_blk` made it transposed so that the register loading step `regM[i] = A_blk[(threadRow * TM + i) * BK + dotIdx]` becomes `regM[i] = A_blk[dotIdx * BM + threadRow * TM + i]`.
- It's a small change, so there is no similar doubling in performance, but it still improved somewhat, from 23.36 TFLOPs (v3.3.4) to 25.1 TFLOPs, or about ~42% of `cublas_sgemm`.

## [v3.3.4] Added `cu5s_2d_title`: each thread calculates 16 C titles (4x4) block

- Making each thread calculate a 4x4 block would push arithmetic intensity even higher, and each thread fills 8 values (4 for A, 4 for B: twice as much as before, so only 256 threads per block) in the SM cache filling step. A simple `strideA`/`strideB` (dividing `numthreads` by `BK`/`BN`) helps achieve this easily. The block values here are BM=BN=64, BK=TM=TN=4.
- Peformance of `cu5s_2d_title` again doubled compared to the previous version, `cu4s_thread_mul`, as it reached 23.36 TFLOPs, or about 38% of `cublas_sgemm`.

## [v3.3.3] Added `cu4s_thread_mul`: each thread calculates 8 C titles

- To improve data reuse, each thread calculates a column of 8 rows (TM=8) in the innermost loop. Previously the block size was fixed to 32x32, but here it was set to changable macros (BM=BN=64, BK=8, TM=8). At this stage, each thread still only populates one value in A/B, but the smart choice of shapes exploit that the resulting C block (64x8 @ 8x64 -> 64x64) is 8 times larger than the number of threads per block, which is 64x8 only.
- Variables definitions and names were renamed to be more readable and consistent.
- `cu4s_thread_mul` reached 12.8 TFLOPs as it requires less memory access, doubling the peformance of `cu3s_smem`.

## [v3.3.2] Added Shared Memory Cache Blocking: `cu3s_smem`/`cu3d_smem`

- The idea here is basically similar to `KC` cache blocking implemented in the `avx_blis` family, meaning that the innermost iteration is broken into blocks (size 32, GPU block size), and going along it loops through C titles K/BLOCKSIZE times. It's also somewhat a packing idea, with each block packing the 32x32 block into the shared memory on every chunk, so that the fastest data read axis aligns with that of the wrap
- `cu3s_smem` reached 6.163 TFLOPs, ~2.67x faster than `cu2s_coalesce`, but is only 10% of `cublas_sgemm`. Caching the block is a major improvement, but the ladder can continue be extended following the tutorial.
- `cu3d_smem`, as expected, doesn't improve by any, and is basically equivalent to `cublas_dgemm` also even on large matrices due to the capped F64. In the next patches the double-precision version won't be mentioned, but they will still appear in the logs. 

## [v3.3.1] Single-precision counterparts of GPU GEMM ladder (`cu1s_naive`/`cu2s_coalesce`)

- Added `cu1s_naive`/`cu2s_coalesce` (`estzero/gemm/cuda_ladder.cc`) via `cu_s_ladder`. Basically same to `cu_d_ladder`, but with a double to float case inside the wrapper using a pinned buffer just like `cublas_sgemm`.
-Both added routes don't get anywhere near the ~64x jump FP32's capability, and both get *worse* past N~2045-4096 instead of climbing toward it the way `cublas_sgemm`.
- `cu1s_naive` is additionally ~3-8x slower than `cu2s_coalesce` at every size, and still collapses at the same power-of-two N (1024/4096/6144/8192) as its double-precision counterpart -- the same DRAM partition-camping penalty from v3.3.0, now compounding with genuine bandwidth-bound behavior.
- The benchmarking, however, will now be focusing on the single-precision version, as there is much more potential to improve, whereas the double precision performance is capped.

## [v3.3.0] Hand-written double-precision GPU GEMM kernels (`cu1d_naive`/`cu2d_coalescing`), microbenchmark moved to kernel-only timing

- Added `cu1d_naive` and `cu2d_coalesce` (`estzero/gemm/cuda_ladder.cc`), the start of a from-scratch CUDA GEMM ladder alongside cuBLAS. The ladder here follows this tutorial: https://siboehm.com/articles/22/CUDA-MMM
- `gemm_bench`'s timing moved inside the GEMM call itself, as `cublas_dgemm`/`cublas_sgemm`/`cu_s_ladder`/`cu_d_ladder` now take an optional `kernel_ms` out-param, measured with a persistent `cudaEvent_t` pair around just the kernel/cublas launch, excluding the memory allocation steps, allowing for a better measurenment on FLOPs by excluding the overhead cost.
- `gemm_bench` now defaults to the GPU route family only, the CPU family has to be called via `--cpu`. The default matrix sizes are raised up to 8192 to match the GPU's performance.
- `cu2d_coalesce` is already comparable with `cublas_dgemm` even at size 8192, reaching ~90-100% of the ~1.3 TFLOP/s ceiling for double-precision. This GPU's FP64 rate is capped to ~1:64 of FP32, so even the naive version already comparable with `cublas_dgemm`.

## [v3.2.3] Persistent `cublasHandle_t` + pinned staging for `cublas_sgemm`: why the cuBLAS routes were far from peak, and the fix

- Both `cublas_dgemm` and `cublas_sgemm` called `cublasCreate`/`cublasDestroy` on every GEMM, and these are avoidable costs by reusing a similar idea in v3.1.7a by using a persistent global buffer. A single `cublasHandle_t` is created once via a local function static and reused for every call.
- `cublas_sgemm` requires an additional `double->float` cast before going to device, so similar to the mentioned strategy, we can redirect the cast's output straight into pinned memory, no allocation cost.
- The improvement is seen most in the vikings PertT loop. Pyrimidine/sto-3g 56.397s -> 27.067s (`cublas_dgemm`, ~2.1x), 55.192s -> 26.896s (`cublas_sgemm`, ~2.1x); propargyl/sto-3g 39.540s -> 22.125s / 38.272s -> 21.370s (~1.8x both).
- For pure GEMM `cublas_dgemm` is ~537-780 GFLOP/s across 1022-2048 (unchanged from v3.2.2). `cublas_sgemm` climbs to ~1900-2300% of OpenBLAS at 2045/2048 (up from ~907-1062% pre-fix), so the pinned copying buffer helps with its dominant cost.

## [v3.2.2] Added `cublas_sgemm` as a backend for `einsum` and GEMM microbenchmark, explores the precision/perf tradeoff

- Added a single-precision cuBLAS backend (`--gemm-scheme cublas_sgemm`) with casting from double to float happening inside the call.
- Pure GEMM microbenchmark (`bench/outputs_gemm/v3.2.2.log`): `cublas_sgemm` roughly doubles `cublas_dgemm`'s throughput at every tested size (2045x2045x2045: 1305.98 vs 724.00 GFLOP/s, 906.9% vs 502.8% of OpenBLAS).
- As an `einsum` backend, that gain doesn't shows up in the vikings PertT loop: 55.192s vs `cublas_dgemm`'s 56.397s at pyrimidine/sto-3g. PertT's bottleneck is fixed per-call cost from its many small contractions, and it this the same observation v3.2.1 for `cublas_dgemm`.
- Two larger systems actually failed with the single precision: `ch3/aug-cc-pvtz` MP2/UHF (nbf=115, |dE|=1.59e-6) and `propargyl/6-31g` MP3/UHF (nbf=33, |dE|=1.35e-6).
- Renamed the existing `cublas` scheme/rows to `cublas_dgemm` throughout for symmetry with the new `cublas_sgemm` 

## [v3.2.1] Added `cublas_dgemm` as a backend for `einsum` and GEMM microbenchmark

- The `cuBLAS` library is the GPU version for GEMM, in this update it got introduced in both the microbenchmark and wired directly as a backend for `einsum`. The GPU on this machine is GeForce RTX 4090.
- Illustrated in the microbenchmark for pure GEMM task, it absolutely crushes the CPU GEMM implementations, reaching 726.18 GFLOP/s, which is 887.6% of OpenBLAS.
- As an `einsum` backend, however, it doesn't perform as its setup cost is much more for than the CPU GEMM counterparts as it has to move memory from CPU. When the task is a few large contractions, for example MP2 benzene/def2-svp (nbf=114), cuBLAS (6.294s) matcheed Eigen (6.360s). However, for tasks with a large number of small contracions, such as the perbutative triples in the vikings algorithm, its performance tanked signficantly, taking 56.434s to finish this step, compared to 22.262s of Eigen.
- Including cuBLAS made it so that the entire library got linked and always loaded unconditionally, even if `cublas_dgemm` is not called. This affect the pure-CPU workflows, increasing their peak RSS (see `bench/outputs/v3.2.1_eigen_cublas_build.log` and `bench/outputs/v3.2.1_eigen_nocublas_build.log`). To address this, an explicit note is added in `CMakeLists.txt` to comment out two lines when `cublas_dgemm` is not needed.

## [v3.2.0] Wired `avx_8x8`/`avx_16x8`/`avx_24x8` into `einsum` backends

- `einsum`'s `GEMMScheme` now covers 6 backends: `odometer`, `eigen`, `blas`, `avx_8x8`, `avx_16x8`, `avx_24x8`.
- Added `--gemm-min` (`estzero/est/main.cc`) to override `try_gemm`'s FLOP-size guard (default 1024, unchanged).
- This is the first version where all 6 schemes are correct and compared fairly, side by side, on the same systems. 6 new rows were added to each of the MP2/MP3/CCSD(T) tables in `README.md`.

## [v3.1.8] Added `avx_blis_Rx8`, a full NC/KC/MC blocked + packed GEMM following BLIS framework. Closing GEMM CPU phase

- Implemented the full BLIS 5-loop scheme (`estzero/gemm/avx_blis.cc`): NC/KC/MC blocking on top of packing, targeting L3/L1/L2 respectively, same `MRx8` micro-kernel as `avx_Rx8`. Follows exactly the [BLIS framework](https://github.com/flame/blislab/blob/master/tutorial.pdf). No `gemm_sweep.py` support added since this purely exploratory.
- The full blocking scheme did not pay off compared to the best of the `avx_Rx8` family (v3.1.7a), even at large matrices: `avx_Rx8` beats `avx_blis_Rx8` at every tested size (2048/4096/5000) (see `v3.1.8.log`). The reuse theory isn't wrong, but this implementation doesn't have enough of an edge over an optimized flat kernel to justify its extra complexity. The cache level tunning was not fully optimized as that would required more complex schemes to fit per matrix shape.
- This closes the hand-written GEMM CPU ladder. `avx_Rx8` is the final, best-performing route, and will be added as a backend for `einsum`.

## [v3.1.7a] Fix: allocate buffer for `A` packing once outside the main loop

- A packing does need to be done inside every `m` loop, but the buffer vector can be allocated outside of this to be reused. This mini patch just moves it outside the loop.
- Avoiding repeated heap allocations actually substantially improved performance in all `avx` schemes. At 2048x2048x2048, `avx_16x8` reached 89.5% OpenBLAS, essentially equal to Eigen (89.8%). For the sizes that are not power of 2, the best `avx` varies around 70-90%, and `avx_16x8` performed generally better for >2048 than <2048, meaning the cleaning up sweep can be improved.

## [v3.1.7] Added `B` packing over the `N_blk x 8 x K` block for `avx` family

- This is a similar idea to A packing, but different from A packing, when reading at a fixed ROW `k` it's already contiguous. The problem is that when `k` changes, it jumps down a row, moving by `k * ldb`. Moving to the next NR block (n+1 along N_blk) jumps all the way from k=K to k=0. This pattern is predictable, so the strategy is basically reshaping B in this special layout so that throughout the entire main loop reading from B is always contiguous.
- Performance across all `avx` schemes improved significantly at 2048x2048x2048, with `avx_16x8` reaching 74.7% OpenBLAS, compared to 54.1% last patch with only A packing. At the smaller size matrix 1024x1024x1024, performance is essentially equivalent to OpenBLAS, with `avx_8x8` and `avx_16x8` being at 97.8% and 97.5% with B_packed sitting entirely in L2 cache, not needing L3 like in size 2048.

## [v3.1.6] Added `A` packing over the `MR x K` block for `avx` family

- The innermost loop requires continuous reads from a fixed column `k` in A, so the idea here is to pack this scratch buffer that will be read inside this loop into a contiguous block in `MR`, basically converting into a column major block.
- This blocking scheme significantly improved the performance of the `avx` family as a whole. Performance varies, however, for example at 1024x1024x1024 `avx_20x8` went from 30.2% to 58.0% of OpenBLAS at 1024, and `avx_16x8` went from 47.2 to 95.2%, beating Eigen! At 2048x2048x2048, the family still loses to OpenBLAS by a decent amount, but has improved since last patch, with the best being `avx_16x8` reaching 54.1% OpenBLAS performance.

## [v3.1.5] Added `gemm_sweep.py`, sweeping around 1024 and 2048 with SVG chart

- v3.1.4 found the `avx` family performed much worse than OpenBLAS at large matrix sizes, and also a sharp drop at the power of twos (1024, 2048). This update sweeps around that range to investigate what's happening with `bench/gemm_sweep.py` that uses `gemm_bench.py` helpers to run different matrix sizes around these numbers (`avx_1x8` excluded).
- A sharp, narrow spike exactly at the power of 2** (1024 and 2048) showed `avx_12x8` (the fastest hand written route) is at 71.3%/**43.5%**/75.8% of OpenBLAS for N=1023/1024/1025, and 50.4%/**29.6%**/52.2% for N=2047/2048/2049. See `README.md`

## [v3.1.4] Generalized `avx_Rx8` via nested macros, updated benchmark matrix size.

- Added `avx_12x8`/`avx_16x8`/`avx_20x8`/`avx_24x8` (`estzero/gemm/avx512.cc`) by generalizing `avx_8x8` into a macro-generated family instead of hand writing the function for each size. `REPEATn(X)` is built by chaining onto the previous `REPEATn` (`REPEAT12` = `REPEAT8` + 4 more literal index calls). `DEFINE_AVX_RX8(MR)` then write out one whole function per size.
- `avx_12x8` reaches 85.0% of OpenBLAS at 1022x1022x1022 (69.72 GFLOP/s) (see [v3.1.4.log](bench/outputs_gemm/v3.1.4.log)). However, `avx_16x8` (44.49 GFLOP/s) is worse than both its neighbors `avx_12x8` and `avx_20x8` (see `README.md`). A possibility is that in larger `MR` schemes they run out of general purpose registers (there is only 16 of them) to store the memory address for FMA instructions, so beyond MR=12 it starts to become a tradeoff.
- **Benchmark defaults reorganized.** The `naive_*`/`mkn_RxR`/`mnk_RxR` ladder is now frozen at 3.14. and will not be run in the future by default. The new `avx_Rx8` family became the active work, requiring benchmark at bigger sizes: 510, 1022, 2045. The standard size for comparison in `README.md` is now 2045.
- At 2045x2045x2045 (three ~33MB matrices, past L3), the `avx_Rx8` family couldn't keep up with BLAS anymore: `avx_12x8` drops from 83.5% of OpenBLAS to 53.5%, while OpenBLAS/Eigen barely move (OpenBLAS 82.39 -> 82.79 GFLOP/s, Eigen 78.81 -> 80.12 GFLOP/s). The smaller matrices likely fit entirely in cache, so smaller matrices hid their the limitation of the `avx_Rx8` family. The size of matrices used to benchmark will be investigated more next patch.

## [v3.1.3] Added in AVX-512 intrinsics for GEMM (`avx_8x8`) to vectorize filling in the registers

- Added `avx_1x8`/`avx_8x8` (`estzero/gemm/avx512.cc`), using `<immintrin.h>` `__m512d` intrinsics directly instead of "praying" the auto-vectorizer packs named scalar accumulators into a register, which it probably didn't past `mnk_4x4` (v3.1.1). `avx_8x8` tiles the same 8x8 block of C as `mnk_8x8`, but as 8 named `__m512d` accumulators (one zmm register per row of 8 doubles) instead of 64 scalar doubles.
- `avx_8x8` hits 64.27 GFLOP/s at 1022x1022x1022 (~33ms, 78.0% of OpenBLAS, 44.6% of peak) vs. `mnk_8x8`'s 13.36 GFLOP/s (~160ms, 16.2%). The vectorization scheme completely beats the manually blocking ones, including the `mkn` family. It is also relatively close to OpenBLAS and Eigen already with zero packing.
- `avx_1x8`, despite also filling a full zmm register per k-step, performed far worse (19.84 GFLOP/s, 24.1% of OpenBLAS at 1022x1022x1022). `perf stat` showsed 48.83% L1-dcache-load-miss and 14.29% LLC-miss, vs. `avx_8x8`'s 21.73%/3.80%. It cannot fully utilized the loaded B block with only using one register of vectorized C accumulators.

## [v3.1.2] Added `perf stat` to microbenchmark to monitor IPC and L1 cache load misses

- `gemm_bench.py` now additionally reports IPC and L1-dcache-load-miss% per route (it runs one process per route under `perf stat`, so hardware counters attribute to a single kernel)
- `gemm_bench.py` and `benchmark.py` (EST) now prints out markdown formatted tables/rows to fill in directly into `README.md`

## [v3.1.1] Reorganized GEMM ladder into naive/mkn_RxR/mnk_RxR; register-file investigation

- Renamed different gemm versions from v3.1.0 (and newer ones) into three families, moved from `estzero/linalg/` into their own `estzero/gemm/` directory: `naive_mnk`/`naive_mkn` (unoptimized references: wrong vs. right innermost index), `mkn_RxR` (`mkn` loop, MR blocks rows of A reused against B), `mnk_RxR` (`mnk` tiles an MRxNR block of C held as NAMED local scalar accumulators across the full k reduction, written once).
- `mnk_RxR` uses named scalar accumulators (`c00, c01, ...`), as named locals give the compiler the a clear signal to keep them in the register (reference: https://github.com/flame/how-to-optimize-gemm). This alone doesn't save `mnk_8x8` from spilling (64 named accumulators is still far past what fits): confirmed via `perf stat` + `objdump` that `mnk_8x8` has the *lowest* L1-dcache-load-miss rate of the whole family (10.5%) while also having the *most* total instructions — consistent with spill-then-immediate-reload traffic (an L1 hit, not a miss, but a real extra load/store pair each time) rather than genuine memory pressure.
- `mnk_1x8` (~115ms at 1022x1022x1022) beats `mnk_8x1` (~179ms) despite both fill 8 doubles. `1x8`'s 8 values are one contiguous 64-byte fetch from B, while `8x1`'s are 8 separate strided fetches from A (row-major; if it's column major the problem just reverses)
- `naive_mkn` (~87ms) is actually the best custom GEMM scheme here even with the new blocking schemes, which all have their own tradeoffs.
- `mkn_8x1` (~93ms, 21.7 GFLOP/s) beats every `mnk_RxR` combo including `mnk_1x8` (~120ms). The `mnk_Rxr` tradeoffs register blocking for the non-contiguous k stride in the innermost loop in B, and without packing it turned out worse than `mkn_`. However, the register blocking scheme here has more optimization potential.

## [v3.1.0] Added Microbenchmark for GEMM (manual GEMM optimization prep)

- This update added a manually written GEMM with the intention of gradually walking up the optimization ladder. To measure the performance independently from the main EST pipeline (the naive baseline GEMM version would also be too slow to wire in the main runs), a microbenchmark is added: `gemm_bench` executable that runs the different size matrix multiplications on all routes, and `gemm_bench.py` to parse and format the data.
- Notation: A(M x K) x B(K x N) = C(M x N), all row-major; C = alpha\*\(A\*\B) + beta\*\C.
- Three first three versions of manually written GEMM are added: `gemm_naive1` loops in `m-n-k` and has a single accumulate operation directly to C inside the inner most loop, `gemm_naive2` accumulates into a temporary sum first inside `k` loop then accumulates to C outside the loop, and `gemm_reorder` loops through `m-k-n`, striding along a single row for B and C in the innermost loop, preserving cache locality.
- Both `gemm_naive1` and `gemm_naive2` are inferior to the rest, with the latter isn't much better. `gemm_reorder` significantly improved the performance: on the 1024x1024x1024 system, it was ~45x faster the naive implementations, and it's ~4x slower than Eigen/OpenBLAS (see `README.md`).

## [v3.0.3] Fixed the v3.0.1 GEMM size guard

- `try_gemm`'s minimum size guard (introduced in v3.0.1) declined GEMM if any operand had fewer than 1024 elements, defaulting to the odometer scheme instead. However, a contraction between a small coefficient matrix and a large tensor would still benefited from GEMM pays off most, but this would get declined.
- This was spotted on UHF MP2's quarter-transform for a system with a small occupied-orbital count: CH3/aug-cc-pvtz (nbf=115, doublet, `n_alpha=5`/`n_beta=4`). Both Eigen and OpenBLAS routes fell back to the odometer path for the previous version v3.0.1-v3.0.2a range. MO transform went from 1.429s (v3.0.0, pre-guard) to 5.5s (v3.0.2/v3.0.2a, guard present, either GEMM backend)
- Fix: checks the contraction's total work (`left_size * summed_size * right_size` < 1024). UHF MP2 (ch3/aug-cc-pvtz) MO transform is back to 1.429s (Eigen) / 1.271s (OpenBLAS). However "vikings" perturbative triples loop's many small `einsum` calls are also affected, since the fix widens GEMM criteria, but this was system-dependent: pyrimidine/sto-3g (RHF) got ~20% slower, propargyl/sto-3g (UHF) got ~10% faster.
- `bench/benchmark.py` now logs the run's CLI parameters as a final line in each run's log, including GEMM scheme (Eigen/BLAS). 
- v3.0.1 through v3.0.2a's Eigen-vs-OpenBLAS comparisons are unreliable due to the guard bug above (see `README.md`).

## [v3.0.2a] Set num threads of openblas to 1

- Disabling threading for openblas significantly improved the performance throughout, and importantly moved the peak RSS to the normal level, about the same as Eigen, even more efficient. Per CCSD iteration, pyrimidine/sto-3g (nbf=34): 263.4ms, 74.8 MB (OpenBLAS, v3.0.2a), compared to 272.1ms, 85.1 MB (Eigen, v3.0.1). 
- GEMM is declined for `Tensor` with less than 1024 elements, but slightly bigger than that the number of operations are still very small, so the threading overhead dominated, and each of them has its own buffer so the memory usage also increased substantially. The system used here are small, so disabling it here helps with performance.

## [v3.0.2] Added `cblas_dgemm` route for `einsum`

- An additional backend GEMM engine is added for `einsum` via OpenBLAS. This wiring uses the raw data pointers of `Tensor` directly to perform GEMM without calling Eigen for mapping to its matrix. Both routes, OpenBLAS and Eigen, have similar memory allocation and copy assignment steps, so performance should not differ here.
- However, the OpenBLAS route saw some slowdown and additional memory usage: per CCSD iteration, pyrimidine/sto-3g (nbf=34) went from 272.1ms, peakRSS 85.1 MB (Eigen, v3.0.1) to 342.2ms, peak RSS 203.8 MB (OpenBLAS, v3.0.2). MP2 and MP3 suffered much less, so this is attributed to the high intensity of `einsum` calls in CCSD(T). This will be investigated next patch.

## [v3.0.1] Added `Tensor` minimum size requirement to call GEMM

- A guardrail is added that declines to call GEMM if any `Tensor` operand has size (total number of elements) less that 1024, and defaults to the odometer scheme. This change targets the repeated small `einsum` calls in the vikings algorithm for perbutative triples for small systems to avoid the overhead for small contractions. The guard successfully lowered the PertT run time of the small systems, for example o2/sto-3g (nbf=10, n_alpha=9, n_beta=7) goes back to 0.182s, which is similar to v2.4.2, compared to 0.327s in v2.5.0.

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
