# Reduced (Schur-complement) KKT solve vs cuDSS: feasibility probe

Brief: `native/docs/briefs/reduced_kkt.md`. This was a measurement only; the
solver is unchanged. Raw data: `native/bench/results/reduced-kkt-20261103/`.
GPU: RTX 4070 SUPER (FP64 is 1/64 of FP32), CUDA 13.2, cuDSS 0.8.0.10, date 2026-11-03.

## What was built

`native/bench/reduced_kkt_probe.cu`, CMake target `cupiqp_native_reduced_kkt_probe`
(links `CUDA::cublas`, `CUDA::cusolver`).

- **States.** For each selected snapshot case, the public `SparseSolver` (replay
  settings: eps_abs = eps_duality_gap_abs = 1e-8, relative tolerances 0, 10 Ruiz
  passes) runs once to convergence (N iterations). It then runs with
  `max_iter = k` for k = 1..N. After each run the probe copies `kkt.values`,
  `res.primal` and `res.dual` from `debug_buffers()`. The KKT pattern is rebuilt
  with `build_kkt_host_maps` from the snapshot patterns, and its nnz is checked
  against the debug buffer. The delta of the factored matrix is read from K's
  y diagonal (checked to be constant). It equals the `info().delta` of iteration
  k-1, so `rho_factored` is recorded as `info().rho` of k-1 (rho_init for k = 1).
  No factor retries occurred in any state.
- **Right-hand sides.** `random` is uniform [-1, 1] with a fixed seed.
  `residual` is the state's IPM residual `[res.x | res.y | res.z_u - res.z_l]`.
  The solver's real cuDSS rhs is not exposed by `debug_buffers()`.
- **Method A.** A `CudssSolver` with the solver's configuration
  (non-deterministic, LOWER view). It is analyzed once per pattern group, on
  the first state, then runs factor + solve.
- **Method B.** Weights are `w = -1 / diag(K)` on the y and z rows: 1/delta,
  z_reg_inv, and 1 on inactive rows, whose G row is zero. Then
  `H = P + diag(x_reg) + [A; G]^T diag(w) [A; G]` is assembled into a dense
  n x n lower-triangular matrix, factored, and used to solve
  `H dx = r_x + M^T (w * r_m)`. Finally `d_m = w * (M dx - r_m)` is recovered
  (one warp per column or row, shuffle reductions, deterministic).
  Assembly variants:
  - (a) `a_dense_syrk`: dense `diag(sqrt w) [A; G]`, (p+m) x n column-major
    (stored-entry scatter; zero entries stay zero), then cuBLAS DSYRK with beta = 1
    on top of the scattered P block.
  - (b) `b_sparse`: one block per output row i with 16 warps. Each warp takes a
    contiguous slice of column i's row list, prefetches 32 rows of metadata and
    `w_k M_ki` per round, and adds `w_k M_ki M_kj` (j <= i) into its own
    shared-memory accumulator. The accumulators are summed in a fixed order, so
    there are no atomics. The probe checks bitwise determinism on every state.
  - (c) `c_sparse_skip0`: (b), plus a warp ballot that drops rows with
    `w_k M_ki == 0` at run time. The resulting H is bitwise identical to (b) in
    every state.
  - Cholesky: cuSOLVER `cusolverDnDpotrf` and `cusolverDnDpotrs`, FP64, with a
    preallocated workspace. **Extra, not in the brief:** an exploratory tiled
    multi-block right-looking Cholesky (`tiled_chol`, tile 32, 3 kernels per
    tile column, deterministic). It replaces potrf only; the solves still use
    potrs.
- **Accuracy.** Computed on the CPU in long double.
  - Normwise (brief):
    `||K x - b||_inf / (||K||_inf ||x||_inf + ||b||_inf)`.
  - Componentwise (Oettli-Prager) backward error:
    `max_i |K x - b|_i / (|K||x| + |b|)_i`. I added it because the normwise
    value is dominated by the huge dz entries late in the solve and falls to
    1e-25 to 1e-34, which says little.
  - Also recorded: the relative inf-norm difference between the A and B
    solutions, the potrf `info`, `(max L_ii / min L_ii)^2` (a lower bound on
    cond(H)), and the dense-vs-sparse difference in H.
  - One iterative-refinement step on B (residual in FP64 on the CPU, then one
    more reduced solve).
- **Timing.** CUDA events, 5 warm-ups, then 100 reps, eager and as a replayed
  CUDA graph; the median is reported. One state in four is timed, plus the last
  state. Everything is allocated once. cuDSS FACTORIZATION/SOLVE, cuSOLVER
  potrf/potrs and cuBLAS DSYRK all captured into graphs without error.

## Commands

```bash
cd ~/Projects/cupiqp-native-wt/reduced_kkt/native
cmake -S . -B ../build/native -DPython_EXECUTABLE=$HOME/Projects/cupiqp-native/.venv/bin/python
cmake --build ../build/native -j 8 --target cupiqp_native_reduced_kkt_probe
cd .. && native/bench/results/reduced-kkt-20261103/run.sh      # 6.5 min, sequential runs
python3 native/bench/results/reduced-kkt-20261103/summarize.py > native/bench/results/reduced-kkt-20261103/summary.md
```

`run.sh` runs MPC group 0 cases 0, 21, 42, 63 (97 states) and every Trajopt
group 0..5 with cases 0 and 4, the first and last SQP iterate (359 states).
Every one of these solves reached CUPIQP_SOLVED.

## Timing (graph replay, median over the timed states, us; eager in brackets)

Cholesky alone = `copyH_potrf` - `copyH` (the 3 us copy restores H before each
in-place factorization).

| | MPC (n=169, m=2521) | Trajopt (n=241, m=6385) |
|---|---:|---:|
| cuDSS factor | 816 (883) | 758 (821) |
| cuDSS solve | 119 (178) | 137 (194) |
| **cuDSS per iteration (1 factor + 2 solves)** | **1040 (1125)** | **995 (1074)** |
| assembly (a) dense DSYRK | 574 (580) | 2773 (2778) |
| assembly (b) sparse | 191 (193) | 159 (160) |
| assembly (c) sparse, skip zeros | 124 (129) | 153 (154) |
| Cholesky, cuSOLVER potrf | 453 (462) | 658 (667) |
| Cholesky, tiled (exploratory) | 242 (267) | 336 (366) |
| solve + recovery (rhs kernel, potrs, recovery kernel) | 65 (72) | 96 (102) |
| per iteration, (a) + potrf | 1153 (1176) | 3617 (3640) |
| per iteration, (b) + potrf | 770 (790) | 1003 (1022) |
| per iteration, (c) + potrf | **701** (722) | **997** (1016) |
| per iteration, (b) + tiled Cholesky | 561 (595) | 689 (722) |
| per iteration, (c) + tiled Cholesky | **493** (527) | **677** (716) |

cuDSS per iteration divided by reduced per iteration (graph):

| | MPC | Trajopt |
|---|---:|---:|
| (a) + potrf | 0.90x | 0.28x |
| (b) + potrf | 1.35x | 0.99x |
| (c) + potrf | 1.48x | 1.00x |
| (b) + tiled | 1.85x | 1.44x |
| (c) + tiled | 2.11x | 1.47x |

Spread over the timed states was small: cuDSS per iteration 1034..1049 us on
MPC and 967..1025 us on Trajopt; the reduced variants varied by under 3%.

Where the time goes (nsys, one MPC and one Trajopt run):
- cuSOLVER potrf at these sizes is a single `getrf_wo_pivot` kernel, i.e. one
  SM. A standalone sweep gave 0.50 ms at n = 169, 0.72 ms at n = 241 and
  1.47 ms at n = 512, with no gain from `cusolverDnXpotrf`.
- potrs costs 45 us (MPC) to 60 us (Trajopt) per solve in two `trsv` kernels.
- Skipping zeros helps on MPC, where only 41,485..44,120 of the 250,721 stored
  [A; G] entries are nonzero. It does not help on Trajopt (146,007..149,062 of
  165,745 nonzero).
- The first sparse assembly version (8 warps, no prefetch) took 382 us on MPC.
  Prefetching and 16 warps brought it to 191 us.
- In the tiled Cholesky, the diagonal-tile kernel takes about 27 us per tile.
  On this GPU it is limited by FP64 issue, because a warp FP64 instruction
  occupies about 16 cycles whether or not its lanes are active. A Newton rsqrt
  variant was slower and was reverted.

## Accuracy

Criterion, per state, rhs and config, on the brief's normwise residual:
**`res_B <= 1e-10` and `res_B <= 10 * max(res_A, 2.2e-16)`**. The machine-epsilon
floor stops a cuDSS residual of 1e-30 from setting an unreachable target.

| dataset | config | rhs | states | potrf fails | max res_A | max res_B | pass | max cw_A | max cw_B | max cw_B after 1 IR step |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MPC | (b) + potrf | random | 97 | 0 | 6.1e-18 | 3.1e-18 | 97/97 | 2.7e-05 | 1.3e-05 | 5.8e-10 |
| MPC | (b) + potrf | residual | 97 | 0 | 1.0e-15 | 1.9e-15 | 97/97 | 1.6e-05 | 1.6e-05 | 5.6e-10 |
| MPC | (a) + potrf | residual | 97 | 0 | 1.0e-15 | 1.7e-15 | 97/97 | 3.1e-05 | 2.5e-05 | 3.0e-10 |
| MPC | (b) + tiled | residual | 97 | 0 | 1.3e-15 | 1.9e-15 | 97/97 | 1.8e-05 | 3.5e-05 | 3.4e-10 |
| Trajopt | (b) + potrf | random | 359 | 0 | 8.3e-18 | 4.5e-18 | 359/359 | 1.1e-02 | 1.3e-02 | 4.4e-04 |
| Trajopt | (b) + potrf | residual | 359 | 0 | 4.1e-15 | 1.7e-15 | 359/359 | 1.0e+00 | 4.2e-02 | 1.1e-03 |
| Trajopt | (a) + potrf | residual | 359 | 0 | 4.2e-15 | 6.3e-15 | 356/359 | 1.0e+00 | 3.6e-02 | 7.8e-04 |
| Trajopt | (b) + tiled | residual | 359 | 0 | 4.0e-15 | 1.6e-15 | 359/359 | 1.0e+00 | 2.3e-02 | 1.3e-03 |

All configs and both rhs are in `summary.md`.
- **potrf, and the tiled Cholesky, succeeded in every one of the 456 states.**
- The 3 misses of (a) are at 2.6e-15 to 3.0e-15 against a 10x bound of
  1.5e-15 to 2.2e-15. They are rounding-level misses of the dense DSYRK path,
  not a breakdown.
- H from (a) and (b) agrees to at most 2.0e-14 relative.

Across iterations (worst over cases, config (b) + potrf, `summary.md`):
- The normwise residual of both methods stays at 1e-15 or below at every k.
- The componentwise error of both grows with the conditioning:
  `(max L_ii / min L_ii)^2` goes from about 1e3 at k = 1 to 1e11 once
  rho = delta = 1e-10. On Trajopt solves where the solver pushes rho and delta
  down to 1e-13 (group 3 case 0, group 5 case 0), it reaches 1.3e14.
- In those last Trajopt iterations, cuDSS (no refinement, as in the solver by
  default) reaches componentwise errors up to 1.0, while the reduced solve
  reaches up to 4.2e-2. There the two solutions differ by up to 0.3 relative:
  neither is accurate componentwise, and the reduced one has the smaller
  backward error.
- On MPC both stay at or below 3.5e-5 componentwise. The A-B solution
  difference grows from 1e-13 early to at most 1.3e-4 late. This is
  consistent with forward-error growth with cond(H); I did not measure
  cuDSS-vs-cuDSS run-to-run differences.
- One refinement step on B brings its componentwise error down. The worst
  value per k is 4e-16 to 1e-12 up to k = 21 on both datasets. The overall
  worst is 5.8e-10 on MPC, and 1.1e-3 on the rho = 1e-13 Trajopt states.

## Verdict

- **Accuracy: acceptable.** The threshold was normwise condensed residual at or
  below 1e-10 and within 10x of cuDSS (floor 2.2e-16). The sparse-assembly
  configurations met it in 912/912 (state, rhs) pairs and dense DSYRK in
  909/912; potrf never failed. Componentwise, the reduced solve is as good as
  or better than cuDSS without refinement, including the last iterations.
  The margin is thinnest at rho = delta = 1e-13: cond(H) is at least 1.3e14,
  about 30x from 1/eps, so smaller regularization or larger weights could make
  potrf fail. This is where it would break down first; no failure was
  observed.
- **Speed: small win on MPC, none on Trajopt with the method as briefed**
  (cuSOLVER potrf).
  - MPC: 701 us vs 1040 us per iteration (1.48x), with variant (c).
  - Trajopt: 997 us vs 995 us (1.00x).
  - Dense DSYRK (a) is slower than cuDSS on both (0.90x, 0.28x).
- **The bottleneck is cuSOLVER potrf.** It runs on one SM: 453 us on MPC and
  658 us on Trajopt, 65 to 66% of the reduced iteration. With the exploratory
  tiled Cholesky the reduced path gets to 2.11x on MPC (493 us) and 1.47x on
  Trajopt (677 us), with the same accuracy. These gains depend on custom
  Cholesky kernels that have only been checked on these 456 states.

## What an integration would need

- A new backend: `KktSystem` would choose between cuDSS and the reduced solver
  (setting or heuristic on n, p + m and nnz). The reduced side would cover
  pattern extraction from `KktHostMaps` (P lower block, contiguous
  [A; G] rows, CSC with row offsets), the w kernel, assembly, Cholesky, and
  solve with recovery. Its interface would mirror `CudssSolver`
  (factor / solve on a stable rhs / sol). Inactive rows need no special case
  (w = 1 on a zero row reproduces dz = -r_z).
- Graph capture: cuSOLVER potrf/potrs, cuBLAS DSYRK and the custom kernels all
  captured in plain graphs here. Capture inside the conditional bodies of
  `SolveGraphDeviceLoop` was not tested. cuBLAS needs `cublasSetWorkspace`
  (done here, 32 MiB) so it does not allocate during capture.
- Workspace: H is n^2 doubles (228 KB for MPC, 465 KB for Trajopt), plus potrf
  lwork (36 and 48 doubles), w (p + m), the index arrays (about 4 nnz ints),
  and, for (a) only, the (p+m) x n dense matrix (3.5 MB / 12.4 MB).
- Factor status: potrf `info` is a real positive-definiteness signal (cuDSS's
  is ignored today). It could drive the factor-retry logic, which would be a
  behavior change to discuss.
- Expected deviations: solutions differ from cuDSS by 1e-13 to 1e-4 relative
  over the iterations (up to 0.3 at the rho = 1e-13 Trajopt states, where cuDSS
  is itself inaccurate). IPM iteration counts may therefore change. The method
  is deterministic, unlike cuDSS in non-deterministic mode. The solver's
  iterative refinement would call the reduced solve in place of the cuDSS
  solve. That path was not exercised here.

## Open issues

- The tiled Cholesky is exploratory: tile 32, untuned, not unit-tested beyond
  agreeing with potrf on these states. Next steps: a fused or persistent
  kernel to cut the 3N launches, smaller diagonal tiles, better packing of the
  FP64 work.
- potrs (two trsv kernels, 45 to 60 us per solve) is the next cost after
  Cholesky. Not optimized.
- The sparse assembly is still well above an FP64-throughput estimate (about
  45 us for MPC's roughly 12M FMAs), and balance across rows is uneven
  (row i does about i/n of the work).
- Timing bias: the cuDSS solve is timed with the rhs already in place, while
  the reduced solve includes its rhs and recovery kernels. The solver's own
  rhs assembly and splitting kernels are excluded from both.
- The `residual` rhs is a stand-in; the solver's real condensed rhs is not
  exposed by `debug_buffers()`.
- cuDSS analysis ran once per group on the first state's values. The solver
  analyzes at setup, so the reordering inputs differ slightly from what the
  solver uses.
- The IR step computed its residual on the CPU in FP64; a GPU version would
  need a symmetric SpMV of K or the condensed operator.
- Only one GPU type was measured. On GPUs with full-rate FP64 (A100, H100),
  potrf and the assembly would scale very differently.
