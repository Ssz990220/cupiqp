# Reduced (Schur-complement) KKT backend (`Settings::kkt_backend = Reduced`)

Brief: `native/docs/briefs/reduced_kkt_backend.md`. Builds on the feasibility
probe (`native/docs/modules/reduced_kkt.md`). Raw data:
`native/bench/results/reduced-kkt-backend-20261103/`. GPU: RTX 4070 SUPER
(FP64 is 1/64 of FP32), CUDA 13.2, cuDSS 0.8.0.10, date 2026-11-03.

## 1. What it is

An opt-in replacement of cuDSS inside `KktSystem`. The condensed KKT matrix of
`KktAssembly`,

    K = [[P + diag(x_reg), A^T, G^T], [A, -delta I, 0], [G, 0, -diag(z_reg)]]

(inactive z rows: diagonal -1 and a zeroed G row), is solved by eliminating
the y / z rows. With M = [A; G] and w = -1 / diag(K) on the y / z rows
(1 / delta, 1 / z_reg, and 1 on inactive rows):

    H  = P + diag(x_reg) + M^T diag(w) M      (dense n x n, SPD)
    H dx = r_x + M^T (w * r_m),    d_m = w * (M dx - r_m)

Every value is read in place from the K values buffer through index maps
built at setup, so the backend factors exactly the matrix cuDSS would
factor. Inactive rows contribute nothing (zero row, w = 1, dz = -r_z
exactly; tested).

### Files

| File | Contents |
|---|---|
| `native/src/linalg/kkt_system/kkt_linear_solver.hpp` | `KktLinearSolver` interface (rhs / sol / factor / solve / factor_status). `CudssSolver` and `ReducedKktSolver` implement it; `KktSystem` drives either. |
| `native/src/linalg/reduced_kkt/reduced_kkt_solver.{hpp,cu}` | `ReducedKktSolver`: setup-time maps, workspaces, launches. |
| `native/src/linalg/reduced_kkt/reduced_kkt_kernels.cuh` | Kernels (weights, row compaction, assembly, split-row reduce, Cholesky panel / update, rhs reduction, triangular solve, recovery). |
| `native/include/cupiqp/cupiqp.hpp` | `enum class KktBackend { Cudss, Reduced }`, `Settings::kkt_backend` (default `Cudss`, frozen after setup). |
| `native/src/ipm/solver/sparse_solver.cu` | Backend selection at setup; n limit; no cuDSS analysis / warm-up with `Reduced`. |
| `native/python/...` | `cupiqp_native.KktBackend`; `settings.kkt_backend` accepts the enum or `"cudss"` / `"reduced"`. |
| `native/tests/python/upstream_native.py` | `CUPIQP_NATIVE_KKT_BACKEND=reduced` runs the upstream tests on the reduced backend (problems above the n limit are skipped with the reason). |
| `native/bench/reduced_kkt_bench.cu` | Microbenchmark vs cuDSS on real states and an n sweep. |

### Kernels (all deterministic: no atomics, fixed summation orders)

Per factorization (`factor()`):
1. `weights_kernel`: w_k = -1 / K_diag (p + m threads).
2. `compact_rows_kernel`: one warp per row of M; the nonzero entries of the
   row (explicit zeros and zeroed inactive rows dropped) are compacted in
   place of the row's K slots, with an inclusive prefix count per stored
   entry (ballot / popc). MPC stores 250k entries in [A; G] of which about
   42k are nonzero; the compaction removes the zeros from the inner loop of
   the assembly.
3. `assemble_kernel<W>`: one block per *task*. A task is a row i of H and a
   contiguous range of column i's CSC list of M. Heavy rows are split over
   several tasks at setup (estimated work per task at most
   max(4096, total / 2n) products); this matters on Trajopt, where one
   column of M is stored in 6177 of 6449 rows and its row of H carried 14x
   the mean work. W warps per block each accumulate a chunk into their own
   shared row (W = 16 when (W + 1) n doubles fit in shared memory, else 8,
   4, 2, 1); the final sum is P, warp 0, ..., warp W - 1. Single-task rows
   write H directly.
4. `reduce_split_rows_kernel`: split rows, sum of the task partials in task
   order.
5. Blocked right-looking Cholesky, tile 32, two launches per tile column:
   - `chol_panel_kernel` (one warp per block): every block factors the
     diagonal tile in registers (lane = row, shuffles; 1 / sqrt by a float
     seed and two FP64 Newton steps); block 0 writes L_kk, 1 / L_jj and the
     status, block b >= 1 solves the panel tile X = A L_kk^-T (lane = row);
   - `chol_update_kernel`: trailing update, 32 x 8 sub-tiles.
   No cuSOLVER / cuBLAS: they are not linked by the library.

Per solve (`solve()`):
1. `reduce_rhs_kernel`: one 128-thread block per column of M.
2. `trsv_kernel`: one 512-thread block, x in shared memory. Per tile one
   warp runs the substitution chain on the diagonal block in the scaled
   form z_c = b_c - sum_t (L_ct / L_tt) z_t (one shuffle + one FMA per
   column, division-free), then 4 threads per entry apply the panel.
3. `recover_kernel`: one warp per row of M.

### Failure handling (reference dense backend semantics)

`factor_status` (device int32, B = 1) is assigned by the first panel kernel
of every factorization (so it starts from 0) and then keeps the first
failing global column + 1, as potrf's `info`: a pivot is a failure when it
is not > 0 or not finite (NaN, +inf; a zero y diagonal gives an infinite
weight and is caught the same way). `KktSystem::factor_status()` returns this
buffer for the reduced backend; the existing `factor_retry_kernel` then
raises rho and delta (x100), refactors, and ends the problem with
`NUMERICAL_ISSUES` once `max_factor_retires` is reached. With cuDSS the
status stays 0 as before (the cuDSS outcome cannot be read on the device).
After a failed factorization the solve still runs (garbage in, as with
cuDSS); the problem is inactive by then and its iterate is not updated.

### Deviations from the cuDSS path

- The factorization is a Cholesky of the Schur complement instead of a
  sparse LDL^T of the quasi-definite K: different rounding, so iterates
  differ from cuDSS in the last bits early and by more in the last,
  ill-conditioned iterations (probe: 1e-13 to 1e-4 relative per solve).
- It reports factorization failures; cuDSS does not. On an indefinite H
  (for example a non-convex P) cuDSS factors K without complaint, the
  reduced backend retries with larger regularization.
- It is deterministic (bitwise across runs and across the three graph
  modes with `custom_spmv`; tested), cuDSS in default mode is not.
- n limit: `Reduced` is rejected at setup (`ErrorCode::Unsupported`) for
  n > 2048 (`ReducedKktSolver::kMaxN`). H is dense: 32 MB at n = 2048 and an
  iteration (1 factor + 2 solves) of 9.8 ms on the synthetic sweep
  (section 4). Above that the n^3 factorization is the wrong tool; cuDSS
  scales with the fill instead. Below the limit cuDSS can still be faster
  on large, very sparse problems; the backend is opt-in.
- No cuDSS analysis or warm-up at setup, no cuDSS handle.

## 2. Tests

Commands (worktree root, `WARP_CACHE_PATH=$HOME/.cache/warp-cupiqp-native`):

```bash
(cd build/native && ctest)
CUPIQP_NATIVE_BUILD=$PWD/build/native .venv/bin/python -m pytest -q native/tests/python
for be in cudss reduced; do
  CUPIQP_NATIVE_KKT_BACKEND=$be PYTHONPATH=native/tests/python CUPIQP_NATIVE_BUILD=$PWD/build/native \
    .venv/bin/python -m pytest -p upstream_native -q tests/solver/test_sparse_solver.py
done
```

| Suite | Result |
|---|---|
| ctest | 141 / 141 passed (4 disabled by design, as before). 122 before this branch's additions; 19 new. |
| pytest `native/tests/python` | 1246 passed (3 new) |
| upstream `test_sparse_solver.py`, cuDSS | 7 passed, 3 skipped (batched, float32) |
| upstream `test_sparse_solver.py`, `CUPIQP_NATIVE_KKT_BACKEND=reduced` | 7 passed, 3 skipped (same) |

New tests:
- `test_reduced_kkt` (8): the module against a long-double CPU reference.
  Assembled H (relative 1e-14), the solution of K x = b (relative 1e-9) and
  its normwise residual (1e-12), for n = 1, 9, 31, 32, 33, 64, 65, 70, 100,
  169, 241, 300, 800 (W = 8 there), with p = 0, m = 0, p = m = 0
  (box-only), all inequalities inactive, heavy columns (split rows, checked
  to occur), mostly explicit zeros (compaction); inactive rows give
  dz = -r_z exactly; a second factor + solve is bitwise identical;
  non-positive pivots set `factor_status` to the CPU Cholesky's first
  failing column + 1 and a valid matrix resets it; NaN, inf and a zero y
  diagonal are flagged; n > kMaxN is rejected; plain graph replay and replay
  inside conditional IF and WHILE bodies are bitwise equal to eager.
- `test_kkt_system_class` (+4): the full 10-block residual of the
  un-condensed system with the reduced backend (5 seeds, and 2 with
  iterative refinement), captured = eager bitwise (with and without
  refinement), a failing factorization is reported and reset, NaN input
  sets solve_status (both backends).
- `test_solver` (+7, `SolverReduced.*`): analytic QP and random QPs in all
  three graph modes, matching cuDSS (1e-6) with iterations within 2 and,
  with `custom_spmv`, bitwise equal across modes and repeated solves;
  box-only and equality-only QPs; no allocation and no recapture across
  updates in all modes; n = 2049 rejected (cuDSS solves it); the backend is
  frozen after setup; the retry test (P = -10 I with box bounds and an
  equality block): default retries recover the factorization (5 retries)
  and the IPM runs on to a final status (`PRIMAL_INFEASIBLE` after 13
  iterations, the problem is not convex), `max_factor_retires = 2` ends
  `NUMERICAL_ISSUES` with 2 retries; x, rho, delta finite in every mode.
- `native/tests/python/linalg/reduced_kkt` (3): the settings proxy, reduced
  vs cuDSS through Python, the n limit error.

## 3. Maros-Meszaros

`tests/solver/test_maros_meszaros.py` (iterative refinement on, eps_abs
1e-6, max_iter 250), 3 runs per backend in one session, script and logs in
`native/bench/results/reduced-kkt-backend-20261103/maros_meszaros/`
(`summary.txt` from `summarize_mm.py`).

| Backend | run 1 | run 2 | run 3 |
|---|---|---|---|
| cuDSS | 107 passed, 31 failed | 108 passed, 30 failed | 109 passed, 29 failed |
| reduced | 89 passed, 0 failed, 49 skipped | 89 passed, 0 failed, 49 skipped | 89 passed, 0 failed, 49 skipped |

- No problem fails with the reduced backend in any run, so none fails
  consistently while cuDSS solves it.
- Of the 89 problems the reduced backend accepts, cuDSS fails 16 in every
  run (PRIMALC1, PRIMALC2, PRIMALC5, PRIMALC8, Q25FV47, QE226, QETAMACR,
  QFFFFF80, QISRAEL, QSC205, QSCORPIO, QSCRS8, QSCTAP1, QSCTAP2, QSHARE2B,
  STADAT1) and 6 in some runs (PRIMAL1, QADLITTL, QAFIRO, QSCFXM1,
  QSHARE1B, QSIERRA); the reduced backend solves all 22 in every run.
- Skipped by the n limit (n > 2048), 49: AUG2D, AUG2DC, AUG2DCQP, AUG2DQP,
  AUG3D, AUG3DC, AUG3DCQP, AUG3DQP, BOYD1, BOYD2, CONT-050, CONT-100,
  CONT-101, CONT-200, CONT-201, CONT-300, CVXQP1_L, CVXQP2_L, CVXQP3_L,
  DTOC3, EXDATA, HUES-MOD, HUESTIS, LISWET1 to LISWET12, MOSARQP1,
  POWELL20, QPILOTNO, QSCSD8, QSCTAP3, QSHIP04L, QSHIP08L, QSHIP08S,
  QSHIP12L, QSHIP12S, STADAT3, STCQP1, STCQP2, UBH1.
- Wall time of the whole file: cuDSS 41-42 s, reduced 7.1-7.2 s (the 49
  skipped problems include most of the large ones, so this is not a
  like-for-like comparison).

I did not investigate why cuDSS fails these problems; the reduced
backend's Cholesky failure reporting and retry path is one candidate
(cuDSS's failures are never reported), the different rounding another.

## 4. Microbenchmark (graph replay, median, us)

`cupiqp_native_reduced_kkt_bench`, raw data in `microbench/`. Real states:
kkt.values of a reduced-backend solve of case 0 after k iterations; cuDSS
analyzed and factored on the same values. "iter" = 1 factor + 2 solves in one
graph.

| State | reduced assemble | reduced factor (assemble + Cholesky) | reduced solve | reduced iter | cuDSS factor | cuDSS solve | cuDSS iter | cuDSS / reduced |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MPC k = 12 (n = 169) | 65 | 208 | 65 | 333 | 810 | 118 | 1031 | 3.09x |
| MPC k = 25 | 59 | 189 | 58 | 303 | 813 | 119 | 1035 | 3.42x |
| Trajopt k = 12 (n = 241) | 62 | 248 | 86 | 417 | 709 | 144 | 984 | 2.36x |
| Trajopt k = 25 | 61 | 246 | 85 | 417 | 716 | 144 | 984 | 2.36x |

The probe's best configuration (sparse assembly + exploratory tiled
Cholesky + cuSOLVER potrs) measured 493 us (MPC) and 677 us (Trajopt) per
iteration. Steps from the first integrated version (MPC / Trajopt iteration
417 / 565 us): row compaction and task splitting of the assembly (MPC
133 -> 65 us, Trajopt 151 -> 62 us), the division-free substitution with
512 threads and 4 threads per panel entry (Trajopt solve 116 -> 94 us), the
block-per-column rhs reduction (Trajopt solve 94 -> 85 us). Tried and kept
out: exact 1 / sqrt in the diagonal tile (n = 241 factor 216 -> 229 us).

n sweep (synthetic: P with about 9 entries per row, p = n / 4 and m = 2n rows
with 8 random entries each, so H is dense and cuDSS fills in too):

| n | reduced factor | reduced solve | reduced iter | cuDSS factor | cuDSS solve | cuDSS iter |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 90 | 33 | 151 | 351 | 81 | 504 |
| 256 | 199 | 72 | 339 | 840 | 118 | 1069 |
| 512 | 474 | 176 | 822 | 2309 | 209 | 2716 |
| 768 | 852 | 318 | 1484 | 4566 | 301 | 5164 |
| 1024 | 1416 | 493 | 2397 | 7680 | 424 | 8452 |
| 1536 | 3319 | 955 | 5224 | 17178 | 592 | 18236 |
| 2048 | 6701 | 1563 | 9822 | 32530 | 845 | 34000 |

The limit n <= 2048 is where an iteration of the reduced backend reaches
about 10 ms and H 32 MB. This sweep is the worst case for cuDSS (random
coupling, dense fill); on structured sparse problems of that size cuDSS is
expected to be much faster than here, which the limit does not try to
predict. From n = 768 the one-block triangular solve is slower than the
cuDSS solve.

## 5. End-to-end ladder (same session, GPU exclusive)

`native/bench/tools/run_ladder.sh` with two new rows (7: default + reduced,
8: staged + whole-solve graph + custom SpMV + reduced), 20 repeats, MPC and
Trajopt; every transition checked with `check_solutions.py` at 1e-6.
`DATA=~/Projects/cupiqp-native/native/bench/data`, the pre-multi-block
binary built from `62f4392` (`git archive`). Raw data in `ladder/`. For rows
7 and 8 "vs previous" compares with the same configuration on cuDSS (rows 3
and 6).

### MPC

| Step | KKT@1e-6 | iters p50/max/mean | end-to-end p50 / P95 (ms) | update p50 | solve p50 | vs previous | vs reference |
|---|---|---|---|---|---|---|---|
| Reference (Warp, harness protocol) | 1920/1920 | 24.5/28/24.59 | 41.18 / 44.72 | 3.17 | 37.82 |  | 1.00x |
| Native default, before multi-block reductions | 1920/1920 | 24.5/28/24.59 | 37.92 / 41.18 | 0.90 | 36.98 | -7.9% | 1.09x |
| + multi-block reductions (current default) | 1920/1920 | 24.5/28/24.60 | 35.39 / 38.82 | 0.90 | 34.46 | -6.7% | 1.16x |
| + staged preallocated inputs (update_staged) | 1920/1920 | 24.5/28/24.60 | 35.20 / 38.65 | 0.75 | 34.40 | -0.6% | 1.17x |
| + whole-solve CUDA graph (opt-in) | 1920/1920 | 24.5/28/24.60 | 31.60 / 34.32 | 0.75 | 30.79 | -10.2% | 1.30x |
| + custom SpMV (opt-in) | 1920/1920 | 24.5/28/24.60 | 29.93 / 32.75 | 0.75 | 29.13 | -5.3% | 1.38x |
| Native default + reduced KKT backend (opt-in) | 1920/1920 | 24.5/28/24.59 | 14.40 / 15.55 | 0.90 | 13.46 | -59.3% (vs step 3) | 2.86x |
| staged + whole-solve graph + custom SpMV + reduced KKT backend | 1920/1920 | 24.5/28/24.58 | 10.81 / 11.62 | 0.75 | 10.00 | -63.9% (vs step 6) | 3.81x |

### Trajopt

| Step | KKT@1e-6 | iters p50/max/mean | end-to-end p50 / P95 (ms) | update p50 | solve p50 | vs previous | vs reference |
|---|---|---|---|---|---|---|---|
| Reference (Warp, harness protocol) | 960/960 | 30.0/36/29.82 | 50.51 / 58.68 | 4.03 | 46.36 |  | 1.00x |
| Native default, before multi-block reductions | 960/960 | 30.0/38/29.83 | 46.48 / 54.16 | 0.61 | 45.85 | -8.0% | 1.09x |
| + multi-block reductions (current default) | 960/960 | 30.0/36/29.83 | 40.31 / 46.30 | 0.60 | 39.68 | -13.3% | 1.25x |
| + staged preallocated inputs (update_staged) | 960/960 | 30.0/36/29.82 | 40.00 / 46.05 | 0.42 | 39.56 | -0.8% | 1.26x |
| + whole-solve CUDA graph (opt-in) | 960/960 | 30.0/36/29.83 | 35.29 / 40.57 | 0.43 | 34.85 | -11.8% | 1.43x |
| + custom SpMV (opt-in) | 960/960 | 30.0/36/29.82 | 33.98 / 39.58 | 0.43 | 33.54 | -3.7% | 1.49x |
| Native default + reduced KKT backend (opt-in) | 960/960 | 30.0/36/29.81 | 19.25 / 21.81 | 0.61 | 18.60 | -52.3% (vs step 3) | 2.62x |
| staged + whole-solve graph + custom SpMV + reduced KKT backend | 960/960 | 29.5/36/29.79 | 15.88 / 18.00 | 0.42 | 15.41 | -53.3% (vs step 6) | 3.18x |

The cuDSS default row (35.39 ms MPC, 40.31 ms Trajopt) matches the previous
ladder (35.79, 39.84 ms) within run-to-run variation, with the same
iteration statistics: the default path is unchanged. Iterations with the
reduced backend: MPC p50 / max / mean 24.5 / 28 / 24.59 and 24.58 (cuDSS
24.60), Trajopt 30.0 / 36 / 29.81 and 29.5 / 36 / 29.79 (cuDSS 29.82 to
29.83). The whole-solve graph with the reduced backend runs as one graph
launch per solve with no cuDSS allocation (counters in `setup.json`).

## 6. Per-iteration profile

`native/bench/tools/profile_iteration.sh` (Nsight Systems, graph nodes
traced, one warmed update + solve, median over the 25 iterations), files in
`profile/` (`.nsys-rep` / `.sqlite` not committed). us per iteration:

| | MPC cuDSS (iteration graph) | MPC reduced (iteration graph) | MPC reduced (whole-solve graph, custom SpMV) | Trajopt cuDSS | Trajopt reduced | Trajopt reduced (whole-solve, custom SpMV) |
|---|---:|---:|---:|---:|---:|---:|
| Factorization (cuDSS / reduced assembly + Cholesky) | 794.2 | 56.5 + 140.7 | 56.7 + 144.7 | 699.3 | 57.3 + 201.1 | 57.1 + 205.7 |
| KKT solves (cuDSS / reduced, 2 per iteration) | 205.3 | 129.4 | 131.3 | 243.7 | 190.6 | 193.2 |
| SpMV | 81.8 | 89.8 | 24.5 | 56.6 | 62.1 | 28.7 |
| IPM kernels | 41.0 | 45.2 | 50.0 | 43.5 | 47.6 | 52.1 |
| KKT system kernels | 21.7 | 23.4 | 26.9 | 22.7 | 24.4 | 27.8 |
| GPU idle inside the graph | 133.0 | 23.4 | 87.5 | 123.8 | 21.1 | 87.4 |
| Host round trip | 112.3 | 113.0 | -1.3 | 108.3 | 110.7 | -1.4 |
| **Total** | **1407.4** | **627.9** | **522.7** | **1316.2** | **721.6** | **653.1** |

(The negative host round trip of the whole-solve graph is the profiler
script's segmentation; there is no host round trip in that mode.)

Within the reduced backend (Trajopt, iteration graph): `chol_panel_kernel`
is about 20 us per tile column, of which about 12 us is the diagonal-tile
factorization (measured by disabling it: n = 241 factor 216 -> 118 us);
`trsv_kernel` is the largest part of a solve; the substitution chain alone
is about 38 us of a Trajopt solve (measured by removing it).

## 7. Open issues

- The Cholesky is latency bound on this GPU: one warp per diagonal tile,
  32 dependent pivot steps per tile column, FP64 at 1/64 rate. Options not
  tried: two warps per diagonal tile, look-ahead (factor tile kk + 1 inside
  the panel kernel of kk), a cooperative single-kernel Cholesky (capturability
  inside conditional bodies would need checking), explicit inverses of the
  diagonal tiles for the panel solve and the triangular solves (faster, but
  the backward error then grows with cond(L_kk); I did not want to risk it
  without an accuracy study).
- The triangular solve is one block and dominated by the n-step
  substitution chain; it is slower than the cuDSS solve from n = 768 on.
- Only one GPU type measured; on full-rate FP64 GPUs (A100 / H100) the
  balance between assembly, Cholesky and the solves will differ, and the
  n limit should be re-measured.
- The n limit is a fixed number; a heuristic on n and the cuDSS fill (or a
  timing at setup) would pick the faster backend automatically. Not done:
  the backend stays an explicit opt-in.
- Why cuDSS fails 16 Maros-Meszaros problems that the reduced backend solves
  was not investigated.
- B = 1 only (as the rest of milestone 1).
- `compute-sanitizer` was not run on the new kernels (conditional-node
  graphs cannot run under it on WSL; the eager paths could be checked).
