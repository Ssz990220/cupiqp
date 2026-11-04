# Brief: integrate the reduced (Schur-complement) KKT solve as an opt-in backend

Worktree `~/Projects/cupiqp-native-wt/reduced_kkt_backend`, branch
`feature/reduced-kkt-backend`. It starts from `native-cuda-port`, with the
feasibility probe branch `exp/reduced-kkt` already merged in. Shared rules:
`README.md` in this folder (build/test commands, conventions) and
`native/docs/design/DESIGN.md` section 10 (no runtime allocation, RAII,
CUB, determinism, graph-capturable). Commit on your branch only.

## Starting point (read first)

- `native/docs/modules/reduced_kkt.md` and
  `native/bench/results/reduced-kkt-20261103/summary.md`: the probe, the
  formula, the timing and accuracy evidence, and the integration notes.
- `native/bench/reduced_kkt_probe.cu`: working kernels to lift out:
  - sparse assembly of H with run-time skipping of zeros (variant c);
  - the tiled multi-block Cholesky;
  - rhs reduction and recovery.
- `native/src/linalg/kkt_system/` (KktSystem: regularization, condensation,
  rhs assembly into the cuDSS rhs buffer, cuDSS factor and solve, split,
  recovery, finite check, iterative refinement) and
  `native/src/linalg/cudss/cudss_solver.hpp`. The new backend must be a
  drop-in for `CudssSolver` inside KktSystem.
- `native/src/linalg/kkt_assembly/` (the K values buffer and its slot maps:
  the diagonal slots, P/A/G slots, P_diag) and `kkt_pattern.hpp`.
- Reference semantics for failures: the dense backend
  (`cupiqp/dense/dense_kkt_solver.py`, `cupiqp/dense/dense_cholesky.py`)
  writes the Cholesky outcome to `factor_status`. The existing factor-retry
  kernel then raises rho/delta, retries, and reports NUMERICAL_ISSUES when
  `max_factor_retires` is exhausted.

## What to build

1. **Module `native/src/linalg/reduced_kkt/`.** A class with the same role
   and interface as `CudssSolver`:
   - `init` from the K pattern/slot maps (setup time; build all index maps
     needed to read P + x_reg, A, G, delta, z_reg straight from the K values
     buffer, the CSC/row structures for the assembly, and every workspace);
   - `factor(s)`: assemble H, then Cholesky, then write the per-problem
     outcome to a device `factor_status` (nonzero on a non-positive or
     non-finite pivot);
   - `solve(s)`: same rhs/sol buffers and [x | y | z] layout as `CudssSolver`
     (reduce rhs, triangular solves, recover dy and dz);
   - `rhs()` / `sol()` at stable addresses; no allocation after init;
     deterministic; capturable, including inside the conditional bodies of
     the whole-solve graph mode (verify this).
   Inactive inequality rows (z diagonal -1, zeroed G row) must contribute
   nothing, exactly as in K.
2. **Backend selection.** Add `Settings::kkt_backend` (public enum `KktBackend
   { Cudss, Reduced }`, default `Cudss`, frozen after setup) to
   `include/cupiqp/cupiqp.hpp`. Plumb it through the solver
   (`native/src/ipm/solver/sparse_solver.cu`), the nanobind binding and the
   Python settings proxy, and the benchmarks (`native/bench/mpc_replay.cpp`
   `--kkt-backend reduced`, `native/bench/harness/replay_updates.py`).
   - Introduce a small interface (or template) so that KktSystem drives
     either solver.
   - With `Reduced`, KktSystem must report the real `factor_status` so the
     retry path works (reference dense semantics). With `Cudss` behavior must
     not change at all.
   - Reject `Reduced` at setup with a clear error when n exceeds a documented
     limit (choose it from measurements; H is dense n x n).
   - Skip cuDSS ANALYSIS and its warm-up when the reduced backend is
     selected.
3. **Speed (do after correctness).**
   - Tune the tiled Cholesky.
   - Replace cuSOLVER potrs if a custom triangular solve is faster (the probe
     measured potrs at 45-60 us per solve).
   - Keep the assembly deterministic.
   - Remove cuBLAS/cuSOLVER from the hot path entirely if your kernels win;
     otherwise keep them with workspaces set before capture.

## Acceptance

- **Tests**: all existing tests pass (`ctest`, `pytest native/tests/python`,
  the upstream `tests/solver/test_sparse_solver.py` via `-p upstream_native`).
- **New tests**:
  - C++ tests of the module against a dense CPU reference (including inactive
    rows, p = 0, m = 0, box-only problems, a failing-pivot case that must set
    factor_status);
  - the KktSystem class test (full 10-block residual) with the reduced
    backend;
  - solver tests in all three graph modes with the reduced backend;
  - a retry test: a case that forces a Cholesky failure must raise rho/delta
    and either recover or end NUMERICAL_ISSUES, never NaN;
  - make the upstream plugin able to select the backend (for example env
    `CUPIQP_NATIVE_KKT_BACKEND=reduced`) and run the upstream sparse tests
    with it.
- **Maros-Meszaros** (`tests/solver/test_maros_meszaros.py`, iterative
  refinement on): 3 runs with each backend. Report the failure sets and any
  problem that the reduced backend fails consistently but cuDSS solves.
  Problems with large n may be skipped by the n limit; list them.
- **End-to-end** with `cupiqp_native_replay`, 20 repeats, MPC and Trajopt.
  Every transition must pass `native/bench/tools/check_solutions.py` at 1e-6.
  Iteration statistics must be comparable to cuDSS (report p50/max and
  mean).
  - Configurations: default + reduced, and staged + whole-solve graph +
    custom SpMV + reduced.
  - Extend `native/bench/tools/run_ladder.sh` / `summarize_ladder.py` with
    these rows and rerun the whole ladder in one session (GPU exclusive), so
    that all steps are comparable.
- **Profile**: per-iteration profile with
  `native/bench/tools/profile_iteration.sh` for the reduced backend, MPC and
  Trajopt.

## Hand-off

Write `native/docs/modules/reduced_kkt_backend.md` covering:
- design, the n limit, failure handling and how it deviates from the cuDSS
  path;
- tests and pass counts;
- the Maros-Meszaros comparison, the ladder table and the profile;
- open issues.

Update `native/docs/design/DEVIATIONS.md` and `native/docs/REPORT.md`. Commit
raw results under `native/bench/results/` (no `.nsys-rep` or `.sqlite`).
Reply with a concise summary of measured numbers only.
