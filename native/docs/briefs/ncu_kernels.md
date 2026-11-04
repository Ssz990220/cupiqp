# Brief: Nsight Compute analysis and optimization of the hot kernels (single QP)

Worktree `~/Projects/cupiqp-native-wt/ncu_kernels`, branch `perf/ncu-kernels`
(from `native-cuda-port` at bf1ba2c). Shared rules: `README.md` in this
folder and `native/docs/design/DESIGN.md` section 10 (no runtime
allocation, RAII, CUB, determinism, graph-capturable; kernel changes keep
parity). Commit on your branch only.

## Goal

Make one medium-size QP solve faster on this GPU (RTX 4070 SUPER, sm_89,
56 SMs, FP64 at 1/64 of the FP32 rate). Focus on the reduced KKT backend
(`Settings::kkt_backend = Reduced`, `native/src/linalg/reduced_kkt/`) and the
remaining kernels of an IPM iteration. Use Nsight Compute to find out why
each hot kernel is slow before changing it.

Nsight Compute works on this host now (`/usr/local/cuda/bin/ncu`, 2026.1.1).
The WSL performance-counter permission is enabled. Profile the C++ executables,
not Python. Useful drivers:
- `build/native/bench/cupiqp_native_reduced_kkt_bench`
- `cupiqp_native_module_bench`
- `cupiqp_native_replay --repeats 1`
Use `--graph-profiling node` (or eager mode) so that graph kernels are
profiled individually. Collect at least: SpeedOfLight, Occupancy,
LaunchStats, WarpStateStats, SchedulerStats, MemoryWorkloadAnalysis,
SourceCounters (with `-lineinfo`, already on). Save the `.ncu-rep` files
under the results folder but do not commit them (add them to `.gitignore`).
Commit text and CSV exports.

## Current per-iteration profile (reduced backend, default graph mode)

From `native/bench/results/reduced-kkt-backend-20261103/profile/*reduced-iteration.md`.
MPC is 628 us per iteration, Trajopt 722 us:

| kernel | MPC us/iter | Trajopt us/iter | launch config | note |
|---|---|---|---|---|
| `chol_panel_kernel` | 122 (6 calls x 21 us) | 167 (8 x 22) | 1 x 32 threads, 164 regs | one warp per panel |
| `trsv_kernel` | 94 (2 x 47) | 141 (2 x 71) | 1 x 512 | single block, n-step dependency chain |
| `assemble_kernel` | 55 | 56 | 424 / 602 x 512 | H assembly |
| `chol_update_kernel` | 19 (5 x 3) | 34 (7 x 5) | 4 x 32 | trailing update, few blocks |
| `reduce_rhs` + `recover` | 30 | 40 | | |
| IPM reductions (multi-block) | ~45 | ~48 | | already split |
| SpMV (cuSPARSE by default) | 90 | 62 | | custom SpMV is opt-in |
| host round trip (iteration-graph mode) | 113 | 111 | | 0 in whole-solve graph mode |

## Tasks

1. **ncu analysis report** (before any change). For each hot kernel above:
   - limiter: latency, issue, memory or compute; with SOL percentages;
   - achieved occupancy vs theoretical;
   - top warp-stall reasons;
   - register spills;
   - shared-memory bank conflicts;
   - global access efficiency (sectors per request);
   - the source lines that dominate.
2. **Optimizations**, driven by the findings. Do them in order of expected
   gain, one commit per change, measuring each. Candidates (verify, do not
   assume):
   - Cholesky: more warps per diagonal tile, look-ahead or a fused panel and
     update, more blocks in the trailing update, a cooperative single kernel.
     Explicit inverses of the diagonal tiles only with an accuracy check
     against the current version.
   - Triangular solve: a blocked or multi-block forward/backward substitution
     (for example tile-level dependency with diagonal-tile inverses or a
     wavefront), fused with `reduce_rhs` / `recover` where possible.
   - Assembly: fix whatever ncu shows (coalescing, load balance, FP64
     issue).
   - Smaller kernels: fuse launches inside the iteration where it reduces
     graph nodes without changing results.
3. **Evaluate, do not switch.** With the reduced backend, the whole-solve
   graph mode needs no cuDSS memcpy patching, so its main risk is gone.
   Measure making it the `Auto` choice for `kkt_backend = Reduced`, and
   report it. Do not change any default; the user decides.
4. **The cuDSS path and default settings stay behaviorally unchanged.**

## Acceptance

- **Correctness per change.**
  - `ctest`, `pytest native/tests/python` and the upstream
    `tests/solver/test_sparse_solver.py` via `-p upstream_native` pass, with
    both `CUPIQP_NATIVE_KKT_BACKEND=cudss` and `=reduced`.
  - The reduced-module tests pass against the CPU reference with unchanged
    tolerances.
  - Any change to the factorization or solve arithmetic must keep the
    condensed residual and the componentwise backward error at least as good
    as now on the real KKT states (reuse `native/bench/reduced_kkt_probe.cu`).
- **End to end.**
  - `cupiqp_native_replay` with the reduced backend, MPC and Trajopt, 20
    repeats: every transition passes `check_solutions.py` at 1e-6 and
    iteration statistics are unchanged.
  - Maros-Meszaros with the reduced backend: still 0 failures among accepted
    problems (3 runs).
- **Measurements** (same session, GPU exclusive, `CLOCK_MONOTONIC_RAW` as
  now):
  - the per-kernel ncu numbers before and after;
  - `profile_iteration.sh` before and after;
  - the ladder rows `7-native-default-reduced` and
    `8-staged-devloop-customspmv-reduced`, plus the cuDSS default row as a
    control, before and after.

## Hand-off

Write `native/docs/modules/ncu_kernels.md` covering:
- the ncu findings per kernel;
- each optimization with its before/after measurements and accuracy check;
- what did not help;
- the whole-solve-graph evaluation;
- open issues.

Update `native/docs/REPORT.md`. Commit raw text results under
`native/bench/results/ncu-<date>/`. Reply with a concise summary of measured
numbers only.
