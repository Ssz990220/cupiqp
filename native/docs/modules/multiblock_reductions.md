# Multi-block reductions (single-QP performance)

> **Record (2026-11-03).** Profiles below were taken with the cuDSS KKT
> solver, which was removed on 2026-11-04 (REPORT.md section 14).


Branch `perf/multiblock-reductions`, brief `native/docs/briefs/multiblock_reductions.md`.
Raw results: `native/bench/results/multiblock-20261103/{before,after}` (same
session, same machine, GPU otherwise idle, runs one at a time).

## What changed

The "one block per problem" kernels (ports of `wp.launch_tiled(dim=[B])`) ran on
a single SM when B = 1. They now split each problem over `nb` blocks when B is
small. With B large enough to fill the GPU they still run one block per problem.

Files:

| File | Change |
|---|---|
| `src/common/multiblock.hpp` (new) | `ReductionWorkspace` (non-owning view), `ReductionWorkspaceBuffers` (owner, allocated at setup, queries the SM count once), the blocks-per-problem heuristic, block size |
| `src/common/multiblock.cuh` (new) | device primitive `multiblock_reduce<BLOCK, K, KINDS>`, `MultiblockGrid`, `MultiblockStorage`, `reduce_kinds(...)` |
| `src/ipm/kernels/ipm_step_kernels.cu` | calculate_step, calculate_mu, prepare_factor_and_predictor_step on the primitive; init_guess_center and prepare_corrector_step split into reduction launches plus an element-wise launch when nb > 1 (the one-block kernels are kept for nb == 1) |
| `src/ipm/kernels/ipm_residual_kernels.cu` | update_residual_nr (K = 23) and update_residuals_r (K = 4) on the primitive; one pass over the concatenated row; finalization of update_residual_nr spread over three warps |
| `src/ipm/kernels/ipm_kernels.hpp`, `ipm_device_helpers.cuh` | launchers take `const ReductionWorkspace& ws = ReductionWorkspace{}` (last argument); `mb_launch`, `mul_rn` helpers |
| `src/linalg/kkt_system/*` | `solution_finite` on the primitive (K = 1, int max); `KktSystem` owns its own workspace |
| `src/ipm/solver/sparse_solver.cu` | owns a workspace (allocated in `allocate_state`), passes it to the 7 launch sites |
| `python/bindings/bind_ipm_kernels.cpp`, `bind_kkt_system.cpp` | workspace allocated per call (tests only); `_testing.ipm_set_blocks_per_problem(n)` and `kkt_solution_finite(..., blocks_per_problem=n)` force a split |
| `tests/cpp/common/test_multiblock.cu` (new) | primitive tests |
| `tests/cpp/ipm/kernels/test_ipm_multiblock_timing.cu` (new) | disabled timing tests (kernels vs nb; primitive alone) |
| `tests/python/ipm/kernels/*`, `tests/python/linalg/kkt_system/test_kkt_system_kernels.py` | reduction parity tests parametrized over blocks per problem (0 = heuristic, 3, 64; kkt: 0, 4, 64) |

## Design

Launch: grid `(B, nb)` (B on x so B > 65535 still works; nb on y, at most
65535), block `BLOCK`. Block `(b, j)` visits elements
`k = j * BLOCK + i, k += nb * BLOCK` of problem b. With nb == 1 this is the
reference mapping.

`multiblock_reduce` (all threads of the block must reach it):

1. Each thread holds K values. Floating min/max values are encoded as
   order-preserving integer keys (bits, with the magnitude bits flipped for
   negatives). NaN becomes the key that loses every comparison. The result is
   exactly fmin/fmax (a NaN is ignored unless all operands are NaN). For a
   +0/-0 tie, max returns +0 and min returns -0. These trees use integer
   instructions. FP64 runs at 1/64 of the FP32 rate on this GPU, and this
   encoding is what made the K = 23 reduction affordable.
2. Warp shuffle tree for all K values, level by level so that the K chains
   overlap. Lane-0 results go to shared memory, and thread k folds the warp
   results in warp order.
3. If nb == 1, the block decodes and returns: no partials, no atomics.
4. Otherwise thread k stores the encoded partial (`__stcg`), then
   `__threadfence`, `__syncthreads`, and thread 0 does `atomicAdd` on the
   problem's ticket. Only the block that draws ticket nb - 1 continues. It
   fences, stages the partials into shared memory (32 or more blocks at a
   time, coalesced `__ldcg`), and each group of R lanes (R a power of two up
   to 32) folds value k: lane r takes blocks r, r + R, ... in order, then a
   shuffle tree of width R. It decodes, resets the ticket to 0 (ready for the
   next launch or graph replay) and returns true. The caller then runs the
   scalar finalization.

The order is fixed for a fixed (BLOCK, nb), so results are bitwise
reproducible run to run (tested).

Kernels whose element-wise phase needs the reduced scalar:

- `prepare_corrector_step` (nb > 1): `calculate_step` into primal_step/dual_step,
  then `corrector_sigma` (sum, then sigma), then `corrector_rhs` (element-wise).
  `sigma * mu` is computed with `__dmul_rn`. The one-block kernel hoists it as
  a separate product, and without this the compiler contracted it into an FMA
  (46 ulp on cancelling entries, now 0).
- `init_guess_center` (nb > 1): min kernel (min s, min z into the workspace
  scratch scalars), then shift + sum kernel (writes s, z and mu), then the
  element-wise projection. The expressions are unchanged.
- `prepare_factor_and_predictor_step` needs no split. The element-wise part
  (boundary shift, `res_s = -s * z`) does not use mu, so one launch does
  everything; apply_finetune and mu run in the finalizing block.
- `update_residual_nr`: `prev_*_res` is copied in the finalizing block right
  before `*_res` is overwritten (the reference copies at kernel start; the
  values are the same).

Workspace: partials `B * nb_max * 24` doubles, tickets `B` uint32 (zeroed at
setup), scratch `4 B` doubles. A workspace is shared by kernels serialized on
one stream. The solver and `KktSystem` each own one. Graph capture works: the
iteration graph, the prelude graph and the whole-solve graph (device mode) all
pass the replay check (below), and the primitive has a graph-replay test.

## Blocks-per-problem heuristic (measured)

```
target    = 3 * SMs                       (168 on this GPU)
nb        = 1                             if B >= target
nb        = min(ceil(target / B), width / (256 + 8 K))   otherwise
BLOCK     = 128 when nb > 1, reduction_block_dim(width) when nb == 1
```

The brief suggested slices of about 1-2k elements. Measurement (disabled test
`IpmMultiblockTiming.KernelsVsBlocksPerProblem`, 200 launches per CUDA graph,
`native/bench/results/multiblock-20261103/microbench/`) favours much smaller
slices, 1-3 elements per thread at 128 threads. Large K favours fewer blocks,
because the per-block reduction and the final combine grow with K. Selected
rows (us per launch, including the in-graph launch gap):

| shape | nb | residual_nr | calculate_step | calculate_mu |
|---|---|---|---|---|
| MPC | 1 | 22.6 | 8.7 | 4.0 |
| MPC | 8 | 8.0 | 4.1 | 2.9 |
| MPC | 16 | 6.6 | 3.5 | 2.8 |
| MPC | 32 | 6.4 | 3.1 | 2.7 |
| MPC | 64 | 7.7 | 2.8 | 2.6 |
| Trajopt | 16 | 8.0 | 4.4 | 3.0 |
| Trajopt | 32 | 6.8 | 3.8 | 2.7 |
| Trajopt | 64 | 8.1 | 2.9 | 2.7 |

The nb chosen for B = 1 is:

| kernel | MPC | Trajopt |
|---|---|---|
| calculate_step, prepare_corrector_step (K = 2) | 18 | 46 |
| calculate_mu, prepare_factor_and_predictor, init_guess_center (K = 1-2) | 18-19 | 46-48 |
| update_residual_nr (K = 23, width n + p + num_ineq) | 11 | 29 |
| update_residuals_r (K = 4) | 18 | 45 |
| solution_finite (K = 1, width n + p + m) | 10 | 25 |

## Before / after

### Per kernel (Nsight, `profile_iteration.sh <dir> {mpc,trajopt}`, median us per call)

The profile table's grid column shows gridDim.x (= B = 1) only. nb is in the
table above.

| kernel | MPC before | MPC after | Trajopt before | Trajopt after |
|---|---|---|---|---|
| prepare_corrector_step (after: calculate_step + corrector_sigma + corrector_rhs) | 32.9 | 4.64 + 3.12 + 1.19 = 8.95 | 74.0 | 4.77 + 3.19 + 1.32 = 9.28 |
| update_residual_nr | 28.5 | 6.50 | 51.9 | 7.16 |
| calculate_step | 23.1 | 4.64 | 53.4 | 4.77 |
| prepare_factor_and_predictor_step | 13.4 | 3.80 | 27.7 | 3.83 |
| update_residuals_r | 12.4 | 4.28 | 22.8 | 4.58 |
| calculate_mu | 4.0 | 2.74 | 7.1 | 2.84 |
| solution_finite (x2 per iteration) | 2.23 | 2.42 | 4.03 | 2.48 |
| init_guess_center (once per solve; after: min + shift + project) | 22.7 | 2.64 + 3.28 + 1.77 = 7.69 | 51.1 | 2.68 + 3.29 + 1.81 = 7.78 |

### Per iteration (same tool, medians per iteration)

| | MPC before | MPC after | Trajopt before | Trajopt after |
|---|---|---|---|---|
| Total per iteration | 1480.3 | 1436.7 | 1526.3 | 1333.7 |
| IPM kernels | 125.2 | 42.1 | 248.1 | 43.7 |
| KKT system kernels | 21.4 | 22.3 | 25.5 | 22.9 |
| cuDSS factorization | 789.4 | 813.6 | 694.4 | 709.7 |
| cuDSS solve | 204.8 | 211.3 | 242.2 | 245.2 |

(The cuDSS rows move by +-3% between profiles. The change does not touch them.)

### End to end (`cupiqp_native_replay`, default mode, 20 repeats, `check_solutions.py`)

| | MPC before | MPC after | Trajopt before | Trajopt after |
|---|---|---|---|---|
| pass KKT@1e-6 (gpu_resident) | 960/960 | 960/960 | 480/480 | 480/480 |
| iterations p50 / max | 24.5 / 28 | 24.5 / 28 | 30 / 36 | 30 / 36 |
| wall p50 ms (gpu_resident) | 37.906 | 35.794 | 46.529 | 39.962 |
| solve p50 ms (gpu_resident) | 36.947 | 34.836 | 45.876 | 39.293 |
| wall p95 ms (gpu_resident) | 41.253 | 38.960 | 54.164 | 46.245 |
| wall p50 ms (host_values) | 37.747 | 35.815 | 46.546 | 39.959 |

Per-transition iteration counts vary run to run in the baseline itself (cuDSS
default mode is not bitwise reproducible, see `design/DEVIATIONS.md` 3). Within
the 20 repeats, 2/96 MPC and 14/48 Trajopt transitions vary before, and the
same numbers vary after. The sets of iteration counts seen per transition
differ for 2 MPC and 8 Trajopt transitions. Every one of them varies within at
least one of the two runs (for example Trajopt before {29, 30, 35}, after
{29}), so this is the existing nondeterminism. The aggregates are identical.

Other graph modes, 1-3 repeats each, all pass KKT@1e-6 with the same p50/max
iterations: `--graph-mode device` (whole-solve graph with conditional nodes)
MPC 48/48, Trajopt 24/24; `--graph-mode none` MPC 48/48, Trajopt 24/24
(Trajopt gpu_resident p50 29.5 over 24 samples; there is no baseline run of
this mode to compare against).

## Tests and parity

Commands (worktree root):

```bash
(cd build/native && ctest --output-on-failure)              # 122 run, 122 passed (4 disabled)
CUPIQP_NATIVE_BUILD=$PWD/build/native .venv/bin/python -m pytest -q native/tests/python   # 1243 passed
PYTHONPATH=native/tests/python CUPIQP_NATIVE_BUILD=$PWD/build/native \
  .venv/bin/python -m pytest -p upstream_native -q tests/solver/test_sparse_solver.py    # 7 passed, 3 skipped (as before)
```

The pytest count grows from 1075 to 1243 because of the blocks-per-problem
parametrization. The tolerances are unchanged.

New C++ tests (`test_multiblock`, 6 tests): heuristic and block size; widths
1..100003 incl. odd widths, width < blocks (w = 7 over 5 blocks, w = 31 over
64, w = 3 over 8), B = 2..7 with 3..100 blocks per problem and padded row
strides, exactly one finalizing block per problem, tickets back to 0; NaN
semantics (some NaN ignored, all-NaN gives NaN, all-positive / all-negative
rows guard the identity), for 1, 4 and 50 blocks; int32 max; bitwise
reproducibility over 50 runs (60 blocks); graph capture of two launches
sharing the workspace, 5 replays with new data.

ipm_kernels parity vs the reference Warp kernels (worst over all cases and all
forced nb, `CUPIQP_IPM_PARITY_REPORT`, file `ipm-parity-report.txt`). Every
output not listed is bitwise identical (all min/max reductions, all integer
outputs, the corrector rhs, sigma):

| kernel | output | max ulp | max abs | max rel | before (ipm_kernels.md) |
|---|---|---|---|---|---|
| calculate_mu | mu | 1 | 2.2e-16 | 2.2e-16 | 1 ulp |
| prepare_factor_and_predictor_step | mu | 2 | 2.2e-16 | 2.4e-16 | 0 |
| init_guess_center | mu / s / z | 1 / 4 / 2 | 8.9e-16 | 6.3e-16 | 1 / 4 / 2 |
| update_residual_nr | primal_obj | 11 | 2.1e-14 | 1.6e-15 | 6 ulp |
| update_residual_nr | dual_obj, gap, gap_rel | 1, 1, 2 | 2.8e-14 | 2.8e-16 | 1, 2, 1 |

The differences come only from the summation order. Sums near cancellation
(primal_obj) show more ulps at the same absolute error.

Upstream Maros-Meszaros (`-p upstream_native`, iterative refinement on), 3 runs
of the baseline build (commit 90221e1, built in a scratch directory) and 3 of
this branch, alternating: failures base 32 / 27 / 29, new 32 / 29 / 31. No
problem fails in every new run while never failing in the baseline, and the
reverse is also true. The problems that differ (QADLITTL, QE226, QSHIP04L,
QSIERRA, QBRANDY, QSHIP04S, QSHIP12S, STADAT3, PRIMAL1, QSTAIR) flip run to
run (lists in `maros-meszaros/`). This is consistent with the known
nondeterminism. With 3 runs I cannot separate a small systematic shift from
run-to-run noise.

## Deviations

1. Summation order differs from both Warp `tile_sum` and the previous cub order
   (ulp-level, table above). Min/max are exact, as before.
2. Slices are much smaller than the brief's 1-2k elements per block (measured,
   see the heuristic). The block size of split launches is 128, not
   `reduction_block_dim`.
3. `update_residual_nr` / `update_residuals_r` visit the concatenated row in one
   loop instead of one loop per segment (same per-entry expressions; fewer
   dependent memory round trips). The `nb` of `update_residual_nr` uses width
   `n + p + num_ineq` instead of `max(n, p, m)`.
4. Launcher signatures gained a trailing `const ReductionWorkspace&` (default:
   one block per problem, as before). `solution_finite` also gained one.

## Open issues / not done

- `solution_finite` gained nothing (2.2 to 2.4 us on MPC; 4.0 to 2.5 us on
  Trajopt). It could be fused into `split_sol` (which already writes every
  value): zero the status in `assemble_rhs`, set it in `split_sol`. That would
  remove 2 launches per iteration.
- The remaining per-kernel cost (2.7-7 us) is mostly latency: a launch, a
  global round trip for the partials and ticket, and the FP64 finalization.
  The corrector is still 3 launches (about 9 us). Further gains would come
  from fusing consecutive kernels (calculate_step + update_vars +
  calculate_mu), not from the reduction itself.
- FP64 throughput (1/64) dominates the per-element work on this GPU. The
  element loops still use FP64 `fmax`. Integer keys there would change nothing
  numerically but were not attempted.
- The IPM kernels are now about 3% of the iteration. cuDSS factorization and
  solve (about 70%) and the host round trip and graph idle time (about 17%)
  remain the large items.
- `ipm_kernels.md` and `kkt_system.md` still describe the one-block kernels.
  This document supersedes them for the reduction kernels.
