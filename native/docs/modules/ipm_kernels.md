# M7 ipm_kernels: hand-off note

## What is implemented

Every forward interior-point kernel of `cupiqp/solver_kernels.py` used by
`cupiqp/solver.py` has a native CUDA launcher. Version 1 mirrors the Warp code:
the same thread mapping (one block per problem for `wp.launch_tiled`,
`reduction_block_dim(width)` threads; flattened `(B, T)` grids of 256 threads for
`wp.launch`), the same helpers, masks, `wp.where` selects, integer counters,
status codes, and the same order of floating-point operations inside each expression.
Kernels are templated on `T`; only `double` is instantiated. Out of scope (as briefed):
smoothing, backward and gradient kernels, and `create_solution_finite_kernel`
(owned by `kkt_system`).

## Files

| File | Contents |
|---|---|
| `src/ipm/kernels/ipm_kernels.hpp` | public interface: `RowsView<T>`, `rows_*()` Variables slice helpers, `*Args` structs, launchers, convenience builders from `InfoView` |
| `src/ipm/kernels/ipm_device_helpers.cuh` | ports of `finite_value`, `step_candidate(_masked)`, `calculate_step`, `calculate_mu`, `calculate_sigma`, `apply_finetune`, `boundary_shift`; Warp float min/max semantics (`FminOp`/`FmaxOp`, `wmin`/`wmax`/`wclamp`) |
| `src/ipm/kernels/ipm_step_kernels.cu` | step, mu, centering, predictor/corrector, variable updates |
| `src/ipm/kernels/ipm_residual_kernels.cu` | initial-guess rhs, `z_u - z_l` prep, residuals nr / r |
| `src/ipm/kernels/ipm_control_kernels.cu` | reset, termination, continue flag, retry, mark failed, any flag, terminate, advance, finetune, rho/delta, prox |
| `src/ipm/kernels/sources.cmake` | the three `.cu` files |
| `python/bindings/bind_ipm_kernels.cpp` | `_testing.ipm_<launcher>(args: dict, sizes...)` |
| `tests/cpp/ipm/kernels/{CMakeLists.txt,test_ipm_kernels.cu}` | GoogleTest, hand-computed cases |
| `tests/python/ipm/kernels/` | `ipm_kernels_harness.py`, `conftest.py`, `test_ipm_{step,residual,control}_kernels.py` |

## Public API

Conventions (header comment has the details): `(B,)` arrays are plain device
pointers; `(B, k)` arrays are `RowsView<T>{data, stride}` so that the
column slices of the Variables buffers (strided when `B > 1`) are passed exactly
as in the reference; Warp bool arrays are `bool*`; absent blocks may be null.
Widths that Warp reads from array shapes are explicit (`int` or `Dims`). Every
launcher takes `cudaStream_t s` first, enqueues only on it, allocates nothing,
does not synchronize, and is graph capturable (tested). `B == 0` or an empty domain
enqueues nothing.

| Launcher | Reference kernel | Geometry |
|---|---|---|
| `launch_calculate_step<T>(s, B, num_ineq, s_all, z_all, finite_mask_all, step_s_all, step_z_all, settings, alpha_s, alpha_z)` | `create_calculate_step_kernel` | tiled, num_ineq |
| `launch_calculate_mu<T>(s, B, num_ineq, s_all, z_all, finite_mask_all, num_finite_bounds, mu)` | `create_calculate_mu_kernel` | tiled, num_ineq |
| `launch_init_guess_center<T>(s, B, num_ineq, finite_mask_all, num_finite_bounds, s_all, z_all, mu)` | `create_init_guess_center_kernel` | tiled, num_ineq |
| `launch_prepare_factor_and_predictor_step<T>(s, B, num_ineq, PrepareFactorAndPredictorStepArgs)` | `create_prepare_factor_and_predictor_step_kernel` | tiled, num_ineq |
| `launch_prepare_corrector_step<T>(s, B, num_ineq, PrepareCorrectorStepArgs)` | `create_prepare_corrector_step_kernel` | tiled, num_ineq |
| `launch_update_vars_after_corrector_step<T>(s, B, n, p, num_ineq, active_mask, finite_mask_all, primal_step, dual_step, step_primals_all, step_duals_all, primals_all, duals_all)` | `create_update_vars_after_corrector_step_kernel` | (B, n+2 num_ineq+p) |
| `launch_run_full_newton_step<T>(s, B, n, p, active_mask, step_x, step_y, result_x, result_y, primal_step, dual_step)` | `create_run_full_newton_step_kernel` | (B, n+p) |
| `launch_init_guess_rhs<T>(s, Dims, InitGuessRhsArgs)` | `create_init_guess_rhs_kernel` | (B, n+p+num_ineq) |
| `launch_prepare_zu_minus_zl_and_zbu_minus_zbl<T>(s, Dims, z_u, z_l, z_bl, z_bu, x_b_scaling, zu_minus_zl, zbu_minus_zbl)` | `create_prepare_zu_minus_zl_and_zbu_minus_zbl_kernel` | (B, m+n) |
| `launch_update_residual_nr<T>(s, Dims, UpdateResidualNrArgs)` | `create_update_residual_nr_kernel` | tiled, max(n, p, m) |
| `launch_update_residuals_r<T>(s, B, n, num_duals, UpdateResidualsRArgs)` | `create_update_residuals_r_kernel` | tiled, max(n, num_duals) |
| `launch_update_rho_delta_with_ineq<T>(s, B, UpdateRhoDeltaArgs, dual_improved_out, primal_improved_out, settings, current_iter)` | `create_update_rho_delta_with_ineq_kernel` | (B,) |
| `launch_update_prox_vars<T>(s, B, n, num_duals, active_mask, dual_improved, primal_improved, result_x, prox_x, result_duals, prox_duals)` | `create_update_prox_vars_kernel` | (B, n+num_duals) |
| `launch_update_rho_delta_without_ineq<T>(s, B, n, p, UpdateRhoDeltaArgs, result_x, prox_x, result_y, prox_y, settings, current_iter)` | `create_update_rho_delta_without_ineq_kernel` | (B, n+p) |
| `launch_apply_finetune<T>(s, B, mask, reg_limit, no_primal_update, no_dual_update, settings)` | `create_apply_finetune_kernel` | (B,) |
| `launch_reset_solve_state<T>(s, B, ResetSolveStateArgs)` | `create_reset_solve_state_kernel` | (B,) |
| `launch_update_termination<T>(s, B, UpdateTerminationArgs)` | `create_update_termination_kernel` | (B,) |
| `launch_update_continue_flag(s, B, active, current_iter, settings_i, continue_flag)` | `update_continue_flag_kernel` | 1 block, B |
| `launch_factor_retry<T>(s, B, FactorRetryArgs)` | `create_factor_retry_kernel` | (B,) |
| `launch_mark_failed(s, B, failed, current_iter, active, status, iteration)` | `mark_failed_kernel` | (B,) |
| `launch_any_flag(s, B, flags, out)` | `any_flag_kernel` | 1 block, B |
| `launch_terminate_active(s, B, active, current_iter, new_status, status, iteration)` | `terminate_active_kernel` | (B,) |
| `launch_advance_iteration(s, current_iter)` | `advance_iteration_kernel` | 1 thread |

The `*Args` structs list the reference parameter names in the reference order.
Builders for the solver: `reset_solve_state_args(settings_floats, info, active)`,
`update_termination_args(info, current_iter, settings, active, finetune)`,
`update_rho_delta_args(active, info)`, and `rows_x(v)`, `rows_s_all(v)`,
`rows_z_l(v)`, `rows_duals_all(v)`, ... for `VariablesView` slices.

## Tests (run on this worktree, RTX 4070 SUPER, shared GPU)

```bash
cd ~/Projects/cupiqp-native-wt/ipm_kernels/native
cmake -S . -B ../build/native -DPython_EXECUTABLE=$HOME/Projects/cupiqp-native/.venv/bin/python
cmake --build ../build/native -j 8
(cd ../build/native && ctest --output-on-failure -R IpmKernels)      # 10/10 passed
(cd ../build/native && ctest)                                         # 17/17 passed (incl. Core)
CUPIQP_NATIVE_BUILD=$HOME/Projects/cupiqp-native-wt/ipm_kernels/build/native \
  WARP_CACHE_PATH=$HOME/.cache/warp-cupiqp-native \
  $HOME/Projects/cupiqp-native/.venv/bin/python -m pytest -q \
  $HOME/Projects/cupiqp-native-wt/ipm_kernels/native/tests/python/ipm/kernels   # 96 passed
```

Set `CUPIQP_IPM_PARITY_REPORT=<file>` to write the per-array worst differences.

pytest parity: each test builds the inputs twice from one seed (strided
Variables slices stay strided), launches the reference Warp kernel from its
factory with the reference launch geometry, launches the native one through
`_testing`, and compares every output. Integer and bool outputs must match exactly.
Element-wise floats must be within 4 ulp. Outputs that depend on a sum must be
within 4 ulp or 1e-14 relative. Coverage: B = 1, 2, 3 for row kernels
(strided slices), B = 4096 with values drawn from discrete sets around every
threshold for per-problem kernels (asserting that every status / branch occurs), masks
0/1, infinite bounds, absent sides (p = 0, m = 0, no box, G without bounds), all
four block widths (32/64/128/256), a problem without finite bounds, mu = 0
(sigma branch), duals below eps (boundary shift), finetune, retries exhausted,
max iter, primal and dual infeasibility, duality-gap check on/off, iteration
below/above 5 in the rho/delta updates, and NaN-poisoned iterates / directions.

Harness sanity checks I ran:
1. With every native call replaced by a no-op, all 23 test functions fail.
2. With the block and element min/max switched to ordered compares
   (`a < b ? a : b`), the NaN tests fail (2 of 4 cases); restored, they pass.

C++ tests (hand-computed): step lengths (limited and capped at tau), mu (batch,
no finite bounds, 3000-wide row on 256 threads), sigma and the corrector rhs,
sigma = 0 for mu = 0, all termination outcomes incl. inactive / finetune /
duality-gap on-off, factor retry + any flag, loop control (continue flag,
advance, terminate active), CUDA-graph capture of advance + continue flag.

## Parity numbers (worst over all pytest cases)

All outputs not listed are bitwise identical (0 ulp), including every
integer/bool output, every min/max reduction (step lengths, residual norms,
prox infeasibility), and every element-wise output.

| Kernel | Output | max ulp | max abs | max rel |
|---|---|---|---|---|
| calculate_mu | mu | 1 | 2.2e-16 | 2.0e-16 |
| init_guess_center | mu | 1 | 8.9e-16 | 1.4e-16 |
| init_guess_center | s (primals) | 4 | 8.9e-16 | 6.2e-16 |
| init_guess_center | z (duals) | 2 | 8.9e-16 | 4.1e-16 |
| update_residual_nr | primal_obj | 6 | 4.3e-14 | 8.7e-16 |
| update_residual_nr | dual_obj | 1 | 1.4e-14 | 1.5e-16 |
| update_residual_nr | duality_gap | 2 | 1.4e-14 | 3.2e-16 |
| update_residual_nr | duality_gap_rel | 1 | 5.6e-17 | 2.1e-16 |

The only source of difference is the summation order of block sums (cub
`BLOCK_REDUCE_WARP_REDUCTIONS` vs Warp `tile_sum`). The s/z differences of
`init_guess_center` come from the summed mu.

## Deviations from the reference

1. Block sums use cub (fixed tree, run-to-run reproducible) instead of the Warp
   tile reduction order: last-bit differences in sums (table above). Min/max
   reductions use `fmin`/`fmax` like Warp, so they are exact, including NaN handling.
2. Widths are explicit arguments (`Dims` or ints) instead of array shapes. The
   presence flags of `prepare_zu_minus_zl_and_zbu_minus_zbl` are runtime
   (`d.num_hl > 0` ...) instead of `wp.static`; the results are the same.
3. `update_residuals_r`/`update_residual_nr`: thread 0 reuses the values it just
   wrote (local variable) instead of re-reading them from global memory. The
   values are identical, and no other thread writes them.
4. Warp bool arrays are `bool*` (1 byte). The solver must allocate the active /
   finetune / improved masks as 1-byte bools.

No change to formulas, thresholds, masks, status codes or branch structure.

## Optimization assessment (DESIGN 10.6)

Nothing optimized is merged; version 1 only. I measured no performance numbers.

- Launch count. Per iteration with inequalities, this module issues 15
  launches plus 2 per factorization attempt: prepare_factor_and_predictor,
  (factor_retry + any_flag) x attempts, prepare_corrector, mark_failed,
  calculate_step, update_vars, calculate_mu, prepare_zu_minus_zl,
  residual_nr, rho_delta_with_ineq, prox_vars, residuals_r, advance,
  termination, continue_flag. Fusion candidates that keep the formulas
  (all are per-problem work, so one block per problem suffices):
  1. `rho_delta_with_ineq + prox_vars + residuals_r`: the flags are decided in
     registers by thread 0 and broadcast via shared memory before rho/delta are
     overwritten. That removes the two-launch split the reference needs and
     saves 2 launches. Could also absorb `residual_nr` (its outputs feed
     rho/delta directly), which saves 3.
  2. `advance_iteration + update_termination + update_continue_flag` in one
     block (B <= 1024 per block, or a block loop): saves 2 launches.
  3. `factor_retry + any_flag` in one block: saves 1 per attempt (inside the
     retry WHILE node).
  4. `calculate_step + update_vars_after_corrector_step + calculate_mu` in one
     block per problem (the step must be known before the update). This saves 2 but
     moves the n-sized update onto one block per problem; worthwhile for B=1 MPC
     sizes, not for very wide problems.
  For the host-loop and eager modes each saved launch is about one launch latency;
  inside the iteration graph the gain is smaller (node overhead).
- Registers vs shared memory: per-problem scalars (mu, tau, rho, delta, alpha,
  sigma) already live in registers; block reductions broadcast through one
  shared slot. The residual_nr kernel performs up to 22 back-to-back block
  reductions with a `__syncthreads` pair each. Combining the reductions of a
  segment into one multi-value cub reduction (struct of 3-6 doubles) would cut
  barriers by about 4x; it changes nothing numerically for max, and the sum order
  stays fixed.
- Tiling: rows are walked with a block-strided loop (coalesced). For B = 1 and
  very wide rows (n or num_ineq >> 2048) a single 256-thread block per problem
  limits bandwidth. A grid-wide two-pass reduction would be the next step
  for large problems. It is not needed for the MPC sizes.
- Tensor cores: not applicable (no matrix products in these kernels; sm_89 has no
  FP64 tensor path).

## Requested changes to shared code

1. `common/reduce.cuh`: `MaxOp`/`MinOp` use ordered compares (`a > b ? a : b`),
   which differ from Warp `wp.max`/`tile_max` (`fmax`, NaN ignored) when a NaN
   is present. I added `FminOp`/`FmaxOp` and `block_fmin`/`block_fmax` locally in
   `ipm_device_helpers.cuh` (the NaN tests show the difference). Suggest moving
   them into `reduce.cuh` so other modules that port `wp.max`/`tile_max` get
   the Warp semantics.
2. `common/views.hpp` has no strided 2-D view. I defined `RowsView<T>` (pointer
   + row stride) in `ipm_kernels.hpp`; it could move to `views.hpp` if other
   modules need Variables slices for B > 1.
3. `python/cupiqp_native/_interop.dev_ptr` rejects `(B, k)` arrays with B > 1
   and has no stride output. The ipm tests use their own `encode()` (address,
   row stride) in `ipm_kernels_harness.py`. No change is needed for B = 1.

## Known gaps

- No optimized variants (fusions above) and no timing measurements.
- Only FP64 instantiated (helpers carry the FP32 epsilon for later).
- The kernels are tested individually against the reference, not in a full IPM
  iteration sequence (that is M8's iterate-parity test).
- Row kernels tested with B <= 3 (strided slices); per-problem kernels with
  B up to 4096. Index math uses 64-bit for element offsets; widths are int32.
- Bool masks rely on 1-byte `bool`, matching Warp; values other than 0/1 in a
  bool buffer are treated as true by both.
- Work is left uncommitted in the worktree (see the report to the integration owner).
