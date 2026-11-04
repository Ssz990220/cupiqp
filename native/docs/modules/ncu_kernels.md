# Nsight Compute analysis and optimization of the reduced KKT kernels

Brief: `native/docs/briefs/ncu_kernels.md`. Branch `perf/ncu-kernels`, from
`native-cuda-port` at bf1ba2c. Raw data: `native/bench/results/ncu-20261103/`
(`.ncu-rep`, `.nsys-rep` and `.sqlite` not committed). GPU: RTX 4070 SUPER
(sm_89, 56 SMs, FP64 1/64 of FP32), CUDA 13.2, Nsight Compute 2026.1.1,
date 2026-11-03. Every number below was measured on this host in this
session; graph-replay timings are medians of CUDA-event timed graph
launches, ncu durations are at ncu's locked clocks with cache flushes
between replays (so they are not comparable to graph-replay times one to
one).

All changes are inside `native/src/linalg/reduced_kkt/` and only affect
`Settings::kkt_backend = Reduced`. The cuDSS path, `Settings` and all
defaults are unchanged.

## 1. Method and tools

| Tool | Purpose |
|---|---|
| `cupiqp_native_reduced_kkt_bench --profile` | real states (case 0 after N/2 and N iterations); `--profile` puts a profiler range around the last state, for `ncu --profile-from-start off --graph-profiling node --set full --import-source yes` |
| `cupiqp_native_reduced_kkt_sweep --synthetic` | generality sweep, 28 synthetic MPC-like problems (section 6): factor / solve / iteration graph replay and accuracy |
| `cupiqp_native_reduced_kkt_sweep --data` | real-state accuracy: every IPM iteration of case 0, one random and one IPM-residual right-hand side |
| `tools/ncu_summary.py`, `tools/ncu_hot_lines.py` | ncu raw CSV -> per-kernel table; source CSV -> hottest CUDA lines |
| `tools/measure_step.sh`, `tools/compare_step.py` | per-change measurement (microbench, sweep, accuracy) and comparison |
| `tools/run_ladder_reduced.sh`, `tools/summarize_ladder_reduced.py` | ladder rows 3 (cuDSS default, control), 7, 7d (whole-solve graph evaluation), 8 |
| `tools/run_tests.sh` | ctest, native pytest, upstream `test_sparse_solver.py` with both backends |

Accuracy metrics (long double on the CPU, full condensed K): normwise
residual `||K x - b||_inf / (||K||_inf ||x||_inf + ||b||_inf)` and
componentwise (Oettli-Prager) backward error
`max_i |K x - b|_i / (|K| |x| + |b|)_i`.

Noise floor of the real-state accuracy (`results/ncu-20261103/noise/`):
the states come from a solver run with the default cuSPARSE SpMV, which is
not bitwise reproducible, so two runs of the same code give different
states. Same code, two runs: per-state ratio of the metrics median 0.92 -
1.08, max 1.14 - 3.59. The synthetic sweep is deterministic (identical
values and code give identical bits).

Microbenchmarks done first (scratch, numbers used below): a dependent DFMA
takes 48 cycles; the SM's FP64 pipe accepts one warp instruction every 19
cycles whatever the number of warps (one warp with 8 independent chains
reaches 22.9 cycles); predicated-off lanes cost the same as active ones (19
cycles per warp instruction with 32, 16, 8 or 1 active lanes); a dependent
shuffle + DADD is 76 cycles. Latency of 1 / sqrt(d) in a dependent loop
(including one DFMA): float seed + two FP64 Newton steps (the old code)
472 cycles, max error 2.42 ulp; CUDA `rsqrt()` (double) 282 cycles, 2.00
ulp; float seed + one second-order step 390 cycles, 2.41 ulp.

## 2. ncu findings per kernel (before)

Trajopt (n = 241, p = 64, m = 6385), last state; `results/ncu-20261103/baseline/summary_trajopt.md`
(MPC in `summary_mpc.md`; hottest lines in `hot_lines_trajopt.md`).
"FP64 pipe %" is the busiest SM's FP64 pipe activity over the kernel's
elapsed time; occupancy is achieved / theoretical; sectors/req is for
global loads.

| kernel | grid x block | us (ncu) | SM % | mem % | FP64 pipe % | occ % | regs | spills | sectors/req | smem bank conflicts ld/st | top stalls |
|---|---|---|---|---|---|---|---|---|---|---|---|
| assemble_kernel<16> | 602 x 512 | 62.2 | 40.8 | 33.2 | 51.2 | 86.8 / 100 | 40 | 0 | 4.12 | 5820 / 38523 | long_scoreboard 32%, short_scoreboard 22%, barrier 14% |
| chol_panel_kernel (kk = 0) | 8 x 32 | 26.0 | 5.0 | 1.5 | 36.8 | 2.1 / 20.8 | 164 | 0 | 6.37 | 0 / 0 | short_scoreboard 59%, wait 15%, no_instructions 13% |
| chol_update_kernel (kk = 0) | 196 x 256 | 9.6 | 34.6 | 10.5 | 70.4 | 35.0 / 100 | 36 | 0 | 11.66 | 0 / 0 | short_scoreboard 60%, long_scoreboard 19% |
| trsv_kernel<512> | 1 x 512 | 89.3 | 0.5 | 0.9 | 26.4 | 33.3 / 33.3 | 102 | 0 | 14.68 | 3648 / 0 | barrier 70%, short_scoreboard 14% |
| reduce_rhs_kernel | 241 x 128 | 14.6 | 13.3 | 39.7 | 31.9 | 30.7 / 100 | 38 | 0 | 10.66 | 0 / 10 | long_scoreboard 70%, barrier 10% |
| recover_kernel | 807 x 256 | 11.2 | 57.7 | 40.8 | 60.4 | 86.1 / 100 | 40 | 0 | 2.65 | 0 / 0 | long_scoreboard 48%, short_scoreboard 31% |
| compact_rows_kernel | 807 x 256 | 7.7 | 17.9 | 56.6 | 20.8 | 83.1 / 100 | 40 | 0 | 2.71 | 0 / 0 | long_scoreboard 78% |
| reduce_split_rows_kernel | 215 x 256 | 6.9 | 5.1 | 19.0 | 27.2 | 35.1 / 100 | 40 | 0 | 3.05 | 0 / 0 | long_scoreboard 66% |
| weights_kernel | 26 x 256 | 2.6 | 5.3 | 18.5 | 11.8 | 15.3 / 100 | 16 | 0 | 17.68 | 0 / 0 | long_scoreboard 51%, imc_miss 23% |

No kernel spills. Per kernel:

- **chol_panel_kernel** (one warp per block): latency bound. 2.1% achieved
  occupancy (one warp on 6-8 SMs), FP64 pipe 37% busy on its SM, stalls
  dominated by short_scoreboard (FP64 and shuffle dependencies; on sm_89
  FP64 instructions go through MIO). Hot lines: the panel solve
  `x[t] = fma(-x[c], Ls[t][c], x[t])` 25.5%, the shuffles 14.7%, the
  rsqrt seed and its two Newton steps 24%, the predicated rank-1 update
  of the diagonal tile 6.9%. The diagonal-tile rank-1 update issues 496
  FP64 warp instructions for 5456 useful FMAs (34% of the lanes), and the
  pivot chain (shuffle, rsqrt, scale, update of the next diagonal) is
  serial.
- **chol_update_kernel**: FP64 pipe 58-70% busy on the busiest SM,
  short_scoreboard 50-60%; 62% of the samples on the 32-step dependent
  FMA loop, 12.5% on the L tile loads. For kk = 0 the whole update is 917k
  FMAs, i.e. about 3.9 us at the GPU's FP64 peak: it is within 2.5x of the
  throughput floor. Uncoalesced Lj loads (11.7 sectors per request).
- **trsv_kernel<512>**: one block, 511 threads waiting at
  `__syncthreads` (barrier 70-74%) while warp 0 runs the 32-step
  substitution chain of each tile; the other 55 SMs idle; 2560-3648 shared
  bank conflicts on the x staging.
- **assemble_kernel**: memory-latency bound on gathers (the compacted row
  values of M through Cc / Cv), 43.6% of the samples on the accumulator
  line; the walk over the selected entries is serialized by `__syncwarp`
  and each entry waits for its gather; barrier 53% on MPC (warp imbalance
  inside a block); shared-store bank conflicts from the column-indexed
  accumulators (38.5k on Trajopt).
- **reduce_rhs_kernel**: three gathers per stored CSC entry (K[Cslot],
  w[k], b[n + k]), 10.6 sectors per request, long_scoreboard 66-70%, and
  it walks every stored entry, explicit zeros and inactive rows included.
- **recover_kernel**: walks every stored entry of a row of M, zeros
  included (MPC: about 17% of the stored entries are nonzero).
- **compact_rows / reduce_split_rows / weights**: small, long_scoreboard
  bound, each 2.5 - 8 us; not changed.

## 3. Optimizations (one commit each)

Graph-replay medians, `cupiqp_native_reduced_kkt_bench`, real states (MPC
n = 169, Trajopt n = 241; k = N/2 and N); sweep and accuracy as in section
1. Each step is compared with the previous one; files
`results/ncu-20261103/step*/compare.md`.

### 3.1 Multi-block dataflow triangular solve (b67fd63)

`trsv_flow_kernel`: one warp per 32-row tile and direction, 2N blocks
(N = ceil(n / 32), from n only). A block takes its task from a ticket
counter (atomicAdd; the ticket also gives the epoch, so the flags never
need resetting) and waits only for tasks with smaller tickets, so no
deadlock is possible whatever the residency or the block scheduling
order. Forward tile b accumulates L_bt y_t as the flags of tiles t < b
appear (tile of L prefetched into registers before the wait, 4
interleaved accumulators), then the same division-free scaled
substitution as before; backward symmetric with L^T. Flags use
st.release / ld.acquire at GPU scope, data read with `__ldcg`. Fixed
summation order: deterministic (the bitwise-replay tests pass);
graph-capturable, including the conditional bodies (tested).

| | before | after |
|---|---:|---:|
| solve MPC k12 / k25 (us) | 64.5 / 58.4 | 47.8 / 43.9 |
| solve Trajopt k12 / k25 | 85.6 / 85.8 | 56.3 / 56.3 |
| iteration (1 factor + 2 solves) MPC k25 / Trajopt k25 | 303.1 / 415.6 | 274.4 / 355.3 |
| sweep, iteration time | | 25 faster, 2 same, 1 slower |
| sweep solve n = 2016 / 2030 | 1516.5 / 1548.9 | 279.6 / 285.3 |

Slower: `edge_n33` (n = 33, N = 2) 68.6 -> 71.7 us (+4.5%). Same within
3%: `T20_x2u1_block` (n = 62), `edge_n1`. ncu after: trsv_flow_kernel
36.2 us MPC / 44.8 us Trajopt (ncu clocks and cold caches; 89.3 us for the
old kernel on Trajopt), 57% of its samples are flag polling. Accuracy:
real-state per-state ratio medians 0.91 - 1.07 (within the noise floor);
sweep max componentwise backward error per case within 1.0 - 1.8x either
way (e.g. near-limit 2.7e-12 -> 3.2e-12, ill_T80 2.9e-7 -> 3.6e-7,
global0.05 8.8e-13 -> 4.7e-13).

### 3.2 Diagonal-tile Cholesky: CUDA rsqrt and a pipelined pivot chain (22e171e)

`factor_diag_tile`: `rsqrt()` instead of the float-seed Newton iteration
(half the latency, 2.00 vs 2.42 ulp); step j first updates column j + 1,
takes the next pivot and starts its rsqrt, and only then applies the rest
of the rank-1 update, so the rsqrt latency overlaps independent FMAs (the
per-entry operations and their order are unchanged). Measured separately
on Trajopt k25: rsqrt alone 247.8 -> 225-227 us factor, with the
pipelining 216 us. ncu: chol_panel_kernel 26.0 -> 19.2 us per tile column
(FP64 pipe 36.8 -> 43.3%).

| | before | after |
|---|---:|---:|
| factor MPC k12 / k25 (us) | 207.9 / 190.5 | 181.2 / 165.9 |
| factor Trajopt k12 / k25 | 247.5 / 247.8 | 216.1 / 216.8 |
| sweep, iteration time | | 28 / 28 faster (-3.6% at n ~ 2030 to -12% at small n) |

Accuracy: real-state ratio medians 0.81 - 1.08; sweep factor status 0
everywhere, componentwise error within 0.6 - 1.4x per case.

### 3.3 rhs reduction on weighted CSC values, recovery on compacted rows (e308652)

The assembly already computes a = w_k M_ki for every CSC entry of M; it now
stores it (`Ca`, coalesced, no extra pass). `reduce_rhs_kernel` reads `Ca`
and skips the gather of b for zero entries; `recover_kernel` walks the
row's compacted nonzeros (`Cv` / `Cc` of `compact_rows_kernel`). The device
copy of `Cslot` is gone.

| | before | after |
|---|---:|---:|
| solve MPC k12 / k25 (us) | 47.7 / 44.0 | 42.0 / 38.8 |
| solve Trajopt k12 / k25 | 57.0 / 57.0 | 53.2 / 53.1 |
| assembly (us, unchanged) | 65.5 / 59.4 MPC | 65.5 / 59.4 MPC |
| sweep, iteration time | | 28 / 28 within 3% |

The gain needs explicit zeros or inactive rows (MPC states have many);
the sweep cases have few, so no change there. ncu after (MPC):
reduce_rhs 12.8 -> 9.3 us, recover 10.3 -> 6.8 us; Trajopt reduce_rhs
14.6 -> 12.6 us, recover 11.2 -> 10.8 us. Accuracy: ratio medians 0.90 -
1.08.

### 3.4 Software-pipelined row walk in the H assembly (f42841d)

In `assemble_kernel` the first 32 compacted nonzeros of the next selected
row of M are loaded before the current row's shared-accumulator update,
so the gather overlaps it. Same products and the same `+=` order per
accumulator: H is bitwise unchanged (sweep accuracy identical bit for
bit).

| | before | after |
|---|---:|---:|
| assembly MPC k12 / k25 (us) | 65.5 / 59.4 | 57.3 / 53.1 |
| assembly Trajopt k12 / k25 | 62.4 / 61.4 | 56.3 / 57.0 |
| sweep, factor time | | 28 / 28 within 3% (-0.9% .. +1.7%) |

ncu after: assemble_kernel 64.7 -> 57.8 us MPC, 62.2 -> 55.1 us Trajopt;
MPC still barrier 49% (warp imbalance), Trajopt short_scoreboard 31%.

### 3.5 Overall (baseline -> after 3.4)

`results/ncu-20261103/final/compare_baseline_to_final.md`.

| graph replay (us) | MPC k12 | MPC k25 | Trajopt k12 | Trajopt k25 |
|---|---:|---:|---:|---:|
| assembly | 65.2 -> 57.3 | 59.2 -> 53.1 | 62.1 -> 56.3 | 62.2 -> 57.0 |
| factor (assembly + Cholesky) | 208.7 -> 173.7 | 188.4 -> 160.7 | 248.8 -> 210.8 | 249.5 -> 211.6 |
| solve | 64.5 -> 42.7 | 58.4 -> 38.9 | 85.6 -> 53.2 | 85.8 -> 53.2 |
| iteration (1 factor + 2 solves) | 304.1 -> 255.6 | 303.1 -> 234.5 | 416.4 -> 314.4 | 415.6 -> 313.9 |

Synthetic sweep, iteration time: 28 / 28 faster, -7.5% (n = 33) to
-35.7% (T40, n = 366); n = 2016 / 2030: 9478 / 9811 -> 6746 / 7023 us.
Real-state accuracy baseline -> final, all iterations of case 0:

| metric | MPC before max / median | MPC after | Trajopt before | Trajopt after |
|---|---|---|---|---|
| normwise, random rhs | 2.35e-18 / 1.18e-24 | 1.94e-18 / 7.20e-25 | 3.03e-18 / 8.92e-24 | 3.03e-18 / 7.26e-24 |
| componentwise, random rhs | 7.15e-06 / 4.10e-07 | 5.51e-06 / 5.07e-07 | 8.23e-06 / 8.05e-08 | 9.05e-06 / 8.06e-08 |
| normwise, residual rhs | 1.01e-15 / 1.09e-17 | 1.14e-15 / 9.67e-18 | 3.78e-16 / 2.00e-20 | 4.49e-16 / 2.52e-20 |
| componentwise, residual rhs | 4.95e-06 / 1.58e-09 | 6.62e-06 / 2.33e-09 | 1.70e-06 / 7.53e-11 | 1.56e-06 / 6.93e-11 |

Per-state ratios after / before: medians 0.92 - 1.11, max 1.42 - 2.60,
within the run-to-run noise floor of the states (section 1: same code
medians 0.92 - 1.08, max up to 3.59). ncu after: `results/ncu-20261103/final/summary_{mpc,trajopt}.md`.

## 4. What did not help / not done

- rsqrt with a float seed and one second-order step: 390 cycles vs 282
  for CUDA `rsqrt()` (microbenchmark); not used.
- An "exact" `1 / sqrt` had been tried before (reduced_kkt_backend.md);
  `rsqrt()` is a different instruction sequence and is faster.
- Dataflow (tile-owner) persistent Cholesky: not implemented. Estimate
  from the measured FP64 rates: a 32 x 32 x 32 tile update is 1024 FP64
  warp instructions = 19.5k cycles (about 7.8 us) on one SM whatever the
  number of warps, so tile-per-block ownership puts a SYRK of the next
  diagonal tile on one SM (about 4 us) instead of the current four
  32 x 8 sub-tile blocks; the expected per-column critical path (diagonal
  factor + panel solve + one SYRK + flag latency) is no shorter than the
  current one. The saving would be the launch gaps only.
- Explicit inverses of the diagonal tiles (panel solve and triangular
  solve): not done. They would remove the 32-step substitution chains
  (about 2.4k cycles per tile in the triangular solve) but the error then
  grows with cond(L_kk); with the real-state accuracy noise at up to 3.6x
  per state, an accuracy study would need deterministic states first.
- Packing the diagonal-tile rank-1 update so that every FP64 warp
  instruction carries 32 useful FMAs (496 -> about 190 instructions): not
  done for lack of time; it would leave the arithmetic bitwise unchanged.

## 5. Whole-solve graph as the Auto choice for `kkt_backend = Reduced`

Evaluated, not switched (`GraphMode::Auto` still resolves to
`IterationGraphHostLoop`). Ladder row 7d = row 7 (default settings,
reduced backend) with `--graph-mode device`, i.e. what Auto would pick
if it chose the whole-solve graph for the reduced backend. Same session,
20 repeats, gpu_resident, end-to-end p50 / P95 (ms):

| | MPC row 7 (Auto today) | MPC row 7d | Trajopt row 7 | Trajopt row 7d |
|---|---:|---:|---:|---:|
| before this branch's kernels | 14.24 / 15.50 | 12.74 / 13.71 (-10.5%) | 19.39 / 21.95 | 17.61 / 19.95 (-9.2%) |
| after | 12.32 / 13.33 | 10.88 / 11.71 (-11.7%) | 16.22 / 18.35 | 14.41 / 16.27 (-11.2%) |

Every transition of 7d passes the KKT check at 1e-6 (1920 / 1920 and
960 / 960) with the same iteration statistics as row 7 (MPC p50 / max /
mean 24.5 / 28 / 24.58 vs 24.59; Trajopt 30.0 / 36 / 29.81 vs 29.82). With
the reduced backend the whole-solve graph needs no cuDSS memcpy patching
and allocates nothing (as in reduced_kkt_backend.md). Per-iteration
profile (section 7): the whole-solve graph removes the host round trip
(100.8 / 102.4 us per iteration in iteration-graph mode, MPC / Trajopt)
but has 79.4 / 84.5 us of GPU idle inside the graph (conditional-node
and launch gaps) against 20.8 / 21.3 us, so the net gain per iteration is
about 100 / 63 us. Things the user should weigh before making it the
Auto choice: Auto already falls back to no graph with
`iterative_refinement_always_enabled` and `verbose` (unchanged), the mode
is labelled EXPERIMENTAL in the public header, and a failed whole-solve
capture falls back to the iteration graph (`solve_graph_failed_`).

## 6. Generality sweep

`cupiqp_native_reduced_kkt_sweep --synthetic` (`reduced_kkt_sweep.cu`,
`mpc_cases()`). Variables: states x_0 .. x_T (nx each) and inputs
u_0 .. u_{T-1} (nu each), n = (T + 1) nx + T nu. P block diagonal (dense
nx x nx and nu x nu blocks). A: x_0 = x0 and the dynamics rows (identity
on x_{k+1}, dense on x_k and u_k), p = (T + 1) nx. G: `rows_per_stage`
rows per stage on that stage's variables, either dense (block), banded
(half bandwidth 2 within the stage), sparse (each stage variable with
probability `density`) or global (each of the n variables with
probability `density`, plus one stage variable). Values: P symmetric
(0.1 x uniform), diagonal 1 + 2U ("mild") or 1e-8 + 2U ("ill"); y
diagonal -1e-4 (mild) or -1e-9 (ill); z diagonal -(0.1 + 2U) (mild) or
-10^(-9 + 12U) (ill); inactive rows as in K (diagonal -1, G row zeroed);
explicit zeros in G with probability `zeros`. Cases: horizon T = 5, 10, 20,
40, 80 (nx 6, nu 3); (nx, nu) = (2,1), (4,2), (12,4), (20,8), (32,8) at
T = 20; banded, sparse 0.2 / 0.5, global 0.05 / 0.3, 120 rows per stage
(m ~ 10 n); edge cases p = 0, m = 0, box-only (p = m = 0), 90% inactive
rows, 80% explicit zeros, n = 1, n = 33; n near the limit (2016, 2030);
three ill-conditioned cases. Each case: graph replay medians of factor,
solve, one iteration, and the worst residual / backward error of two
random right-hand sides. Results per step in `step*/sweep.jsonl`; the
table baseline -> final is in `final/compare_baseline_to_final.md`.

Every kernel launch configuration is derived from n, the pattern and the
device at setup (tile count N = ceil(n / 32), triangular-solve grid 2N,
assembly tasks and warps from the CSC pattern and the shared-memory
limit, as before); nothing is specific to the two datasets.

## 7. End to end

### Ladder (same session, GPU exclusive, 20 repeats, gpu_resident, CLOCK_MONOTONIC_RAW wall)

`tools/run_ladder_reduced.sh`; every run checked with `check_solutions.py`
at 1e-6. Before: `results/ncu-20261103/baseline/ladder` (commit c0730e3,
kernels as on `native-cuda-port`); after: `final/ladder` (f42841d).

| MPC | KKT@1e-6 | iters p50/max/mean | e2e p50 / P95 before | after | change |
|---|---|---|---|---|---|
| 3-native-default (cuDSS, control) | 1920/1920 | 24.5/28/24.60 | 36.03 / 39.44 | 35.89 / 39.31 | -0.4% |
| 7-native-default-reduced | 1920/1920 | 24.5/28/24.59 | 14.24 / 15.50 | 12.32 / 13.33 | -13.5% |
| 7d-native-default-reduced-devloop | 1920/1920 | 24.5/28/24.58 | 12.74 / 13.71 | 10.88 / 11.71 | -14.6% |
| 8-staged-devloop-customspmv-reduced | 1920/1920 | 24.5/28/24.58 | 10.84 / 11.65 | 8.98 / 9.64 | -17.1% |

| Trajopt | KKT@1e-6 | iters p50/max/mean | e2e p50 / P95 before | after | change |
|---|---|---|---|---|---|
| 3-native-default (cuDSS, control) | 960/960 | 30.0/36/29.81 | 39.98 / 46.35 | 39.82 / 46.00 | -0.4% |
| 7-native-default-reduced | 960/960 | 30.0/36/29.82 | 19.39 / 21.95 | 16.22 / 18.35 | -16.3% |
| 7d-native-default-reduced-devloop | 960/960 | 30.0/36/29.81 | 17.61 / 19.95 | 14.41 / 16.27 | -18.1% |
| 8-staged-devloop-customspmv-reduced | 960/960 | 30.0/36/29.79 | 15.96 / 18.05 | 12.86 / 14.51 | -19.4% |

Iteration statistics after (before in parentheses where different): MPC
row 7d mean 24.58 (24.59); Trajopt row 7 29.82, 7d 29.81 (29.82), row 8
p50 30.0 (29.5), max 36, mean 29.79 (29.79). The cuDSS control row is
within run-to-run variation (-0.4%); its iteration statistics: MPC
identical, Trajopt mean 29.81 (29.82; cuDSS default mode is not bitwise
deterministic).

### Per-iteration profile (`profile_iteration.sh`, Nsight Systems, medians over 25 iterations, us)

| | MPC iteration graph before | after | MPC whole-solve + custom SpMV before | after | Trajopt iteration graph before | after | Trajopt whole-solve + custom SpMV before | after |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| reduced: Cholesky | 142.5 | 115.5 | 146.2 | 119.2 | 202.1 | 166.1 | 206.7 | 170.9 |
| reduced: solve (2 per iteration) | 125.2 | 91.7 | 126.1 | 83.1 | 182.0 | 112.5 | 184.6 | 114.6 |
| reduced: assembly | 63.3 | 55.6 | 64.1 | 56.5 | 66.7 | 62.4 | 67.1 | 61.5 |
| SpMV | 91.2 | 81.1 | 25.0 | 24.9 | 62.3 | 60.8 | 28.9 | 29.0 |
| IPM kernels | 45.8 | 45.9 | 50.4 | 50.6 | 47.8 | 47.9 | 52.3 | 52.3 |
| KKT system kernels | 23.6 | 23.8 | 27.2 | 27.3 | 24.5 | 24.5 | 27.9 | 28.1 |
| GPU idle inside the graph | 20.8 | 20.8 | 79.2 | 79.4 | 21.2 | 21.3 | 84.2 | 84.5 |
| host round trip | 97.4 | 100.8 | -1.3 | -1.3 | 100.1 | 102.4 | -1.4 | -1.4 |
| **total** | **616.5** | **541.8** | **519.4** | **442.2** | **713.5** | **604.8** | **652.8** | **542.0** |

(`iteration_profile.py` now counts `compact_rows_kernel` and
`reduce_split_rows_kernel` as assembly; the "before" columns were
recomputed from the same traces with that rule. The negative host round
trip of the whole-solve graph is the script's segmentation.)

### Maros-Meszaros (reduced backend, 3 runs each, same session)

`results/ncu-20261103/maros_meszaros/` (`run_mm.sh`, `summarize_mm.py`,
`summary.txt`); "baseline" is commit c0730e3 built from `git archive`.

| | run 1 | run 2 | run 3 |
|---|---|---|---|
| baseline | 89 passed, 0 failed, 49 skipped | 89 / 0 / 49 | 89 / 0 / 49 |
| final | 89 passed, 0 failed, 49 skipped | 89 / 0 / 49 | 89 / 0 / 49 |

Per-problem test call time (pytest `--durations`, 10 ms resolution,
median of 3 runs), final / baseline over the 89 problems: median 0.80; 47
faster, 41 same within 3% (mostly problems at the 10 ms resolution), 1
slower (QSHARE1B 0.07 -> 0.08 s). Sum over the 89 problems 6.78 -> 4.65 s;
largest: STADAT1 0.91 -> 0.50 s, YAO 0.53 -> 0.34, QSHELL 0.51 -> 0.27,
QSIERRA 0.35 -> 0.21. Whole-file wall time 7 - 11 s -> 6 s.

## 8. Tests

`bash native/bench/tools/run_tests.sh` after every change (last run on
f42841d):

| Suite | Result |
|---|---|
| ctest | 141 / 141 passed (4 disabled by design, as before) |
| pytest `native/tests/python` | 1246 passed |
| upstream `test_sparse_solver.py`, cuDSS | 7 passed, 3 skipped |
| upstream `test_sparse_solver.py`, `CUPIQP_NATIVE_KKT_BACKEND=reduced` | 7 passed, 3 skipped |

The reduced-module tests (CPU long-double reference, unchanged
tolerances), the bitwise eager / captured / conditional-body tests and the
failure-reporting tests all pass with the new kernels; no test was
changed.

## 9. Open issues

- The Cholesky is still the largest part (MPC 115.5 us, Trajopt 166.1 us
  per iteration in the iteration-graph profile). The panel kernel is 19 us
  per tile column (ncu): the diagonal tile's pivot chain (one warp) and
  the 528 FP64 instructions of the panel solve on one SM. Next steps, in
  expected-gain order: pack the diagonal-tile rank-1 update (bitwise
  identical arithmetic); then, with an accuracy study on deterministic
  states (custom SpMV), explicit diagonal-tile inverses to split the panel
  solve over SMs and to remove the substitution chains of the triangular
  solve.
- The triangular solve spends most of its time in the per-tile chain:
  flag latency, the last tile product and the 31-step substitution (about
  2.5 us per tile in graph replay). n = 33 (N = 2) is 1.3 us slower than
  the one-block kernel.
- The assembly is still 53-57 us; on MPC half of its warp samples wait
  at the block barrier (warps of a task get equal entry ranges but
  unequal work). A per-factorization balance of the selected entries
  would need the active pattern, which changes every iteration.
- The real-state accuracy is not reproducible run to run with the default
  cuSPARSE SpMV (noise floor up to 3.6x per state); a deterministic
  accuracy harness should produce the states with `custom_spmv`.
- One GPU type measured; on full-rate FP64 GPUs the balance changes (the
  FP64-pipe arguments of section 2 do not carry over).
- `compute-sanitizer` not run on the new kernels.
