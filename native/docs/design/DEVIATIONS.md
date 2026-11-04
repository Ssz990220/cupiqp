# Deviations from the pinned reference, unsupported features, open issues

Reference: `PREDICT-EPFL/cupiqp` @ `d538dda` (Python/Warp). Port: branch
`native-cuda-port`. This file lists every known difference. Each one is either
numerically neutral (verified by tests) or is reported with its measured effect.

## 1. Numerically neutral changes (verified)

| Change | Where | Evidence |
|---|---|---|
| Ruiz early exit decided on the device (flag makes later passes no-ops) instead of a host read + sync after each pass | `problem/ruiz` | 488 parity cases bitwise equal, identical pass counts |
| `update()` skips unscaling blocks that are about to be overwritten | `problem/ruiz`, `ipm/solver` | retained blocks bitwise equal after `unscale_data` |
| Finite-bound masks recomputed once per `update()` (reference: after each bound setter); block-reduction kernel | `problem/data` | bitwise equal masks |
| Fused rhs assembly / solution split (one kernel instead of three copies each) | `linalg/kkt_system` | 283 kernel cases bitwise equal |
| Bound sides and sizes passed as runtime kernel arguments instead of `wp.static` specializations (no per-shape JIT) | all kernels | bitwise equal |
| `scale_bounds` / `unscale_bounds` share one kernel; `reset` is one kernel | `problem/ruiz` | bitwise equal |
| cuDSS phases forked/joined onto a solver-owned stream (cuDSS binds its data object to the ANALYSIS stream) | `linalg/cudss` | results within cuDSS run-to-run spread |
| Warm-up FACTORIZATION + SOLVE at setup (cuDSS allocates its factor workspace lazily) | `ipm/solver` | result discarded; removes the only post-setup allocation |
| One cuSPARSE handle shared by the five SpMV operators | `linalg/spmv` | op = N, beta = 0 bitwise equal |
| Conditional loop bodies captured on the same stream (enclosing capture suspended and resumed) | `common/graph` | captured and eager runs agree |

## 2. Floating-point order differences (small, measured)

- Block sums use `cub::BlockReduce` (fixed tree) instead of Warp `tile_sum`:
  `mu`, objective terms and duality gap differ by at most 6 ulp
  (max relative 8.7e-16). Min/max reductions are order independent and use
  Warp's NaN-ignoring `fmax`/`fmin` semantics: bitwise equal.

## 3. Non-reproducibility shared with the reference

- cuDSS 0.8.0.10 default mode is not bitwise reproducible from run to run
  (spread up to ~6e-13 on test systems). cuSPARSE transposed SpMV
  (`A^T y`, `G^T z`) uses atomics (~3e-14). Both apply to the reference too.
- Consequence for hard problems: trajectories agree to 3-4 digits until the
  regularization reaches its floor (1e-13) and the KKT system becomes very
  ill-conditioned, then they diverge. Example QSIERRA (Maros-Meszaros,
  n = 2036): native and reference iterates agree through iteration 20
  (`max|dx|` 1e-11 to 1e-9) and diverge from about iteration 21. Both runs end
  `MAX_ITER_REACHED` in the same session.
- Upstream Maros-Meszaros suite (iterative refinement on, eps_abs 1e-6),
  native 5 runs vs reference 1 full run plus 3 reruns of borderline problems:
  - 26 problems fail in every native run; 24 of them fail in the reference too.
    The other two (QSIERRA, STADAT1) also fail in reference reruns.
  - No problem fails in every native run while always passing in the reference.
  - No reference failure is a problem native always solves.
  - 12 problems flip between runs in both implementations (for example
    QSCFXM1, QSHIP04S, QSHIP12S, STADAT1: reference fails 2 of 3 reruns).

## 4. Native-only options (off by default)

- `Settings::custom_spmv`: deterministic custom CSR/CSC SpMV kernels
  (2.4-7.7x faster than cuSPARSE in graph replay, within 1e-14 relative).
- `GraphMode::SolveGraphDeviceLoop`: whole solve in one CUDA graph.
  - **Default (`Auto`) for `kkt_backend = Reduced`** since 2026-11-03. No cuDSS is
    involved, so there are no undocumented internals and no patching.
  - Opt-in with cuDSS: cuDSS's two pageable 8-byte SOLVE memcpy nodes are
    re-pointed to pinned snapshots. A capture-time self-test (two captures must
    agree) is required, as is cuDSS 0.8; otherwise the solver falls back to the
    per-iteration graph mode. This relies on undocumented cuDSS internals.
  - The reference has no whole-solve graph for its sparse backend (it uses one
    graph per iteration with a host loop). The same operations run in the same
    order in every mode; with the deterministic custom SpMV the results are bitwise
    equal across modes (test `SolverReduced.AutoModeRunsTheWholeSolveAsOneGraph`).

- `Settings::kkt_backend = KktBackend::Reduced` (added 2026-11-03,
  `native/docs/modules/reduced_kkt_backend.md`): dense Schur complement
  H = P + x_reg + [A; G]^T W [A; G] with custom deterministic assembly,
  Cholesky and triangular-solve kernels instead of cuDSS. Differences from
  the default (cuDSS) path:
  - different factorization, so different rounding: iterates agree with
    cuDSS early and diverge in the last bits once the system is
    ill-conditioned; iteration counts stay comparable (ladder in the module
    note);
  - Cholesky failures (non-positive or non-finite pivot) are written to
    `factor_status` and drive the factor-retry kernel (rho, delta x100,
    `NUMERICAL_ISSUES` after `max_factor_retires`), as in the reference
    dense backend; with cuDSS the status stays 0, as in the reference sparse
    backend;
  - bitwise deterministic (cuDSS default mode is not);
  - rejected at setup (`Unsupported`) for n > 2048;
  - no cuDSS analysis or warm-up.
  Maros-Meszaros (3 runs each): the reduced backend solves every one of the
  89 problems with n <= 2048 in every run; cuDSS fails 16 of those in every
  run and 6 more in some runs.

## 5. Unsupported in milestone 1 (rejected with an error)

- Batch size > 1, float32, dense and multistage backends, gradients
  (`enable_grad`, smoothing, `backward`), the transposed KKT solve.
- `use_deterministic_mode_for_cudss = true`: cuDSS 0.8.0.10
  DETERMINISTIC_MODE faults intermittently (`cudaErrorIllegalAddress`,
  out-of-bounds shared read in `cudss::fwd_dtmn_ker`). The reference crashes
  too.
- Capturing `solve()` into a user's CUDA graph (the reference does not support
  this for its sparse backend either).
- Verbose printing is supported only in `GraphMode::None`, which `verbose`
  selects automatically.

## 6. Open issues

- No bitwise capture-vs-eager test is possible for cuDSS in default mode.
- Conditional-node graphs cannot run under compute-sanitizer on this machine
  (WSL). Memcheck covered the non-conditional paths only.
- `cudaPointerGetAttributes` reports `cudaMalloc` memory as "unregistered" on
  WSL. The memcpy patch therefore decides by copy direction first.
- Reduced KKT backend: the Cholesky and the triangular solve are latency
  bound on this GPU (FP64 1/64 rate); the n limit (2048) was measured on one
  GPU type only and is not a heuristic between the backends. See
  `native/docs/modules/reduced_kkt_backend.md` section 7.
