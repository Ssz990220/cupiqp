# Brief: multi-block reductions for single-QP (batch size 1) performance

Worktree `~/Projects/cupiqp-native-wt/multiblock_reductions`, branch
`perf/multiblock-reductions` (from `native-cuda-port`). Read `README.md` in this
folder for the shared build and test commands, conventions and commit rules.
Commit on your branch; do not push or merge.

## Why

Goal of the project from here on: accelerate a single medium-size QP. The
per-iteration kernel profile is in
`native/bench/results/kernel-profile-20261103/{mpc,trajopt}-native-iteration-iteration.md`
(raw traces next to them; tool: `native/bench/tools/profile_iteration.sh`).

The "one block per problem" kernels ported from the reference's
`wp.launch_tiled(dim=[B])` run on a single block (grid = 1, one SM of 56)
when B = 1:

| kernel (`native/src/ipm/kernels/`) | MPC us | Trajopt us |
|---|---|---|
| `prepare_corrector_step_kernel` | 34 | 74 |
| `update_residual_nr_kernel` | 29 | 52 |
| `calculate_step_kernel` | 24 | 52 |
| `prepare_factor_and_predictor_step_kernel` | 14 | 28 |
| `update_residuals_r_kernel` | 13 | 23 |
| `calculate_mu_kernel` | 4 | 7 |
| `init_guess_center_kernel` (once per solve) | 23 | |
| `solution_finite_kernel` (`linalg/kkt_system`, 2 per iteration) | 2.3 | 4.0 |

Per iteration: ~118 us (MPC, num_ineq = 5042) and ~236 us (Trajopt,
num_ineq = 12770). The target is single-digit microseconds per kernel.

## What to build

1. A reusable deterministic multi-block reduction primitive in `native/src/common/`
   (you own a new header there, e.g. `common/multiblock.cuh`):
   - grid = (blocks_per_problem, B); each block reduces its slice to K values
     (each a sum, fmin or fmax with the existing Warp semantics from
     `common/reduce.cuh`);
   - it writes the partials to a workspace, `__threadfence()`, and takes an
     atomic ticket per problem;
   - the last block combines the partials in a fixed order and resets the
     ticket for the next launch or graph replay, then runs the kernel's scalar
     finalization.
   - Results must be bitwise reproducible run to run for a fixed grid. Order
     differences vs the reference stay at the ulp level.
2. `blocks_per_problem` chosen on the host from the width and B. Keep
   blocks_per_problem = 1 (today's behavior) when B alone fills the GPU, and
   split one problem over several blocks when B is small. Aim at about 2-4 x
   the SM count in total, and stop splitting once slices get too small
   (about 1-2k elements per block at most; measure). Query the SM count once
   at setup.
3. Port the kernels above. Kernels whose element-wise phase needs the reduced
   scalar (`prepare_corrector_step`: step lengths -> sigma -> rhs update;
   `init_guess_center`, and check `prepare_factor_and_predictor_step`
   carefully) become reduction kernel(s) plus an element-wise kernel. Keep
   the order of floating-point operations inside each element-wise
   expression unchanged.
4. Workspace: partial buffers and tickets are allocated at setup (no runtime
   allocation; see DESIGN section 10). Add whatever the launchers need
   (e.g. a `ReductionWorkspace` struct argument). You may change
   `ipm_kernels.hpp` launcher signatures and update the call sites in
   `native/src/ipm/solver/sparse_solver.cu` and `native/src/linalg/kkt_system/`
   accordingly, plus `python/bindings/bind_ipm_kernels.cpp` and
   `bind_kkt_system.cpp` (the test bindings may allocate a workspace per
   call).
5. Everything must still capture into CUDA graphs: the iteration graph, the
   prelude graph, and the whole-solve graph with conditional nodes. Tickets
   must be reset in-kernel so that replays work.

## Acceptance

- Existing parity suites pass with unchanged tolerances:
  - `native/tests/python/ipm/kernels` (B up to 4096 and strided rows);
  - `native/tests/python/linalg/kkt_system`;
  - all of `ctest`, `pytest native/tests/python`, the upstream
    `tests/solver/test_sparse_solver.py` via `-p upstream_native`.
- Add tests for the primitive itself (C++): odd widths, width < blocks,
  B > 1 with several blocks per problem, NaN semantics of fmin/fmax,
  bitwise reproducibility, graph capture + several replays.
- End to end, run `build/native/bench/cupiqp_native_replay` on MPC and
  Trajopt with 3 repeats. Every transition must pass
  `native/bench/tools/check_solutions.py` at 1e-6 with unchanged iteration
  statistics (p50 / max 24.5 / 28 on MPC, 30 / 36 on Trajopt).
- Measure before/after:
  - `bash native/bench/tools/profile_iteration.sh <dir> mpc` and `trajopt`
    (per-kernel durations and per-iteration totals);
  - the replay benchmark with 20 repeats on MPC and Trajopt, default mode,
    GPU exclusive.
  Report the numbers, not estimates.

## Hand-off

Write `native/docs/modules/multiblock_reductions.md` covering: design,
blocks-per-problem heuristic, before/after tables, parity results,
deviations, and anything left undone. Commit raw results under
`native/bench/results/multiblock-<date>/` (no `.nsys-rep` or `.sqlite`). Reply
with a concise summary: what changed, test commands and pass counts, timing
before/after, and open issues.
