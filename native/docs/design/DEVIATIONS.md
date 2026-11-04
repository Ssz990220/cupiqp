# Deviations from the pinned reference, unsupported features, open issues

Reference: `PREDICT-EPFL/cupiqp` @ `d538dda` (Python/Warp). Port: branch
`native-cuda-port`. This file lists every known difference. Each one is either
numerically neutral (verified by tests) or is reported with its measured effect.

Revision 2026-11-04: cuDSS and cuSPARSE were removed (REPORT.md section 14).
The rows and sections that described them are kept below, marked as removed,
because the measurements in REPORT.md sections 1 to 13 were taken with them.
Section 4 lists the current differences of the KKT solve and the SpMV.

## 1. Numerically neutral changes (verified)

| Change | Where | Evidence |
|---|---|---|
| Ruiz early exit decided on the device (flag makes later passes no-ops) instead of a host read + sync after each pass | `problem/ruiz` | 488 parity cases bitwise equal, identical pass counts |
| `update()` skips unscaling blocks that are about to be overwritten | `problem/ruiz`, `ipm/solver` | retained blocks bitwise equal after `unscale_data` |
| Finite-bound masks recomputed once per `update()` (reference: after each bound setter); block-reduction kernel | `problem/data` | bitwise equal masks |
| Fused rhs assembly / solution split (one kernel instead of three copies each) | `linalg/kkt_system` | 283 kernel cases bitwise equal |
| Bound sides and sizes passed as runtime kernel arguments instead of `wp.static` specializations (no per-shape JIT) | all kernels | bitwise equal |
| `scale_bounds` / `unscale_bounds` share one kernel; `reset` is one kernel | `problem/ruiz` | bitwise equal |
| (removed 2026-11-04) cuDSS phases forked/joined onto a solver-owned stream | `linalg/cudss` | results within cuDSS run-to-run spread |
| (removed 2026-11-04) Warm-up FACTORIZATION + SOLVE at setup | `ipm/solver` | result discarded |
| (removed 2026-11-04) One cuSPARSE handle shared by the five SpMV operators | `linalg/spmv` | op = N, beta = 0 bitwise equal |
| Conditional loop bodies captured on the same stream (enclosing capture suspended and resumed) | `common/graph` | captured and eager runs agree |

## 2. Floating-point order differences (small, measured)

- Block sums use `cub::BlockReduce` (fixed tree) instead of Warp `tile_sum`:
  `mu`, objective terms and duality gap differ by at most 6 ulp
  (max relative 8.7e-16). Min/max reductions are order independent and use
  Warp's NaN-ignoring `fmax`/`fmin` semantics: bitwise equal.

## 3. Non-reproducibility of the reference (and of the port before 2026-11-04)

Since 2026-11-04 the port is bitwise reproducible from run to run and across
graph modes (fixed summation orders in the KKT solve and the SpMV). The
reference is not, and the measurements below were taken when the port still
used cuDSS and cuSPARSE:

- cuDSS 0.8.0.10 default mode is not bitwise reproducible from run to run
  (spread up to ~6e-13 on test systems). cuSPARSE transposed SpMV
  (`A^T y`, `G^T z`) uses atomics (~3e-14). Both apply to the reference.
- Consequence for hard problems: trajectories agree to 3-4 digits until the
  regularization reaches its floor (1e-13) and the KKT system becomes very
  ill-conditioned, then they diverge. Example QSIERRA (Maros-Meszaros,
  n = 2036): native and reference iterates agree through iteration 20
  (`max|dx|` 1e-11 to 1e-9) and diverge from about iteration 21. Both runs end
  `MAX_ITER_REACHED` in the same session.
- Upstream Maros-Meszaros suite (iterative refinement on, eps_abs 1e-6),
  native with cuDSS, 5 runs vs reference 1 full run plus 3 reruns of
  borderline problems (all 138 problems; today the native solver runs the 89
  with n <= 2048 and solves all of them, section 4):
  - 26 problems fail in every native run; 24 of them fail in the reference too.
    The other two (QSIERRA, STADAT1) also fail in reference reruns.
  - No problem fails in every native run while always passing in the reference.
  - No reference failure is a problem native always solves.
  - 12 problems flip between runs in both implementations (for example
    QSCFXM1, QSHIP04S, QSHIP12S, STADAT1: reference fails 2 of 3 reruns).

## 4. KKT solve, SpMV and graph mode (the native design since 2026-11-04)

- KKT linear solver (`linalg/reduced_kkt`, `native/docs/modules/reduced_kkt_backend.md`):
  dense Schur complement H = P + x_reg + [A; G]^T W [A; G] with custom
  deterministic assembly, Cholesky and triangular-solve kernels, instead of
  the reference sparse LDL^T (cuDSS). Differences from the reference sparse
  backend:
  - different factorization, so different rounding: iterates agree with the
    reference early and diverge in the last bits once the system is
    ill-conditioned; iteration counts stay comparable;
  - Cholesky failures (non-positive or non-finite pivot) are written to
    `factor_status` and drive the factor-retry kernel (rho, delta x100,
    `NUMERICAL_ISSUES` after `max_factor_retires`), as in the reference dense
    backend; the reference sparse backend never reads its factorization status;
  - bitwise deterministic;
  - **problems with n > 2048 are rejected at setup (`Unsupported`)**: H is a
    dense n x n matrix. The limit applies to the whole solver. Upstream tests
    and the Maros-Meszaros suite skip such problems with the reason (49 of
    138 Maros-Meszaros problems).
  Maros-Meszaros: the solver passes every one of the 89 problems with
  n <= 2048 (3 runs on 2026-11-03 with cuSPARSE SpMV, 3 runs on 2026-11-04
  without). With cuDSS the port failed 16 of those 89 in every run and 6 more
  in some runs.
- SpMV (`linalg/spmv`): the module's own CSR kernels (op = N, a fixed number
  of lanes per row) and CSC gathers (op = T), instead of cuSPARSE
  `CUSPARSE_SPMV_ALG_DEFAULT`. Bitwise reproducible for every op and beta;
  within 1e-14 of the reference relative to |op(M)| |x| (parity tests).
- `GraphMode::Auto` = `SolveGraphDeviceLoop`: the whole solve is one CUDA
  graph with a device-controlled IPM loop, captured once. The reference
  sparse backend uses one graph per iteration with a host loop (available as
  `IterationGraphHostLoop`). The same operations run in the same order in
  every mode, and the results are bitwise equal across modes (test
  `Solver.AutoModeRunsTheWholeSolveAsOneGraph`). Verbose and
  `iterative_refinement_always_enabled` still select no graph, as in the
  reference.
- Reference settings field `use_deterministic_mode_for_cudss`: not in the C++
  `Settings`. The Python shim (`cupiqp_native.Settings`) accepts it, stores
  it (default `False`) and ignores it, so upstream code that sets it keeps
  working. Both values are harmless: there is no cuDSS, and the solve is
  deterministic either way (test
  `test_deterministic_mode_setting_is_accepted_and_ignored`). Before
  2026-11-04, `True` was rejected because cuDSS 0.8.0.10 DETERMINISTIC_MODE
  faulted intermittently.
- Removed native-only settings: `kkt_backend` (with `KktBackend`) and
  `custom_spmv`. Setting them on the Python shim raises `AttributeError`.

## 5. Unsupported in milestone 1 (rejected with an error)

- Batch size > 1, float32, dense and multistage backends, gradients
  (`enable_grad`, smoothing, `backward`), the transposed KKT solve.
- Problems with more than 2048 variables (section 4).
- Capturing `solve()` into a user's CUDA graph (the reference does not support
  this for its sparse backend either).
- Verbose printing is supported only in `GraphMode::None`, which `verbose`
  selects automatically.

## 6. Open issues

- Conditional-node graphs cannot run under compute-sanitizer on this machine
  (WSL). Memcheck covered the non-conditional paths only.
- KKT solve: the Cholesky and the triangular solve are latency bound on this
  GPU (FP64 1/64 rate); the n limit (2048) was measured on one GPU type only.
  Larger problems need a sparse factorization (none is linked any more). See
  `native/docs/modules/reduced_kkt_backend.md` section 7.
- `iterative_refinement_always_enabled` still disables CUDA graphs (reference
  rule). The reference needed it because its sparse solve could not run
  inside the refinement's conditional node; the native solve can, so this
  could be relaxed after measuring.
