# Brief: `ipm_kernels` module (interior-point step, residual, termination kernels)

Worktree `~/Projects/cupiqp-native-wt/ipm_kernels`, branch `task/ipm_kernels`. Read `../AGENT_BRIEF.md`
first, then DESIGN.md, DEPENDENCY_GRAPH.md (row "M7 ipm_kernels"), shared headers (`InfoView`,
`VariablesView`, `DeviceSettingsView`, settings index enums and status codes in `src/core/views.hpp`).
Your code lives in `native/src/ipm/`, tests in `native/tests/cpp/ipm/` and `native/tests/python/ipm/`,
bindings in `native/python/src/bind_ipm_kernels.cpp`. You design the header `native/src/ipm/ipm_kernels.hpp`.

## Start here (reference, pinned)
- `cupiqp/solver_kernels.py`: every forward kernel factory and helper used by `cupiqp/solver.py`
  (`_run_device_solve`, `_run_one_iteration`, `_initial_guess`, `_update_residuals_nr`,
  `_update_residuals_r`, rho/delta updates, `_update_termination`, `_reset_solve_state`, `_factor_once`,
  `_mark_solve_failures`, `_terminate_active`). Read the call sites to get the exact input lists.
- Out of scope: backward, gradient and smoothing kernels, and `create_solution_finite_kernel` (owned by
  `kkt_system`).

## Scope (FP64, B = 1, kernels take B)
- One launcher per reference kernel: `launch_<reference name without create_ / _kernel>(cudaStream_t s,
  ...)`, arguments in the order of the reference `wp.launch` inputs (use `InfoView<double>` in place of
  the individual info columns where convenient). Tiled kernels (`wp.launch_tiled`, dim=[B]) become one
  block per problem with `reduction_block_dim` of the same width the reference uses, via
  `core/reduce.cuh` (cub).
- Version 1 mirrors the Warp code exactly: helper functions (`step_candidate`, `calculate_step`,
  `calculate_sigma`, `apply_finetune`, `boundary_shift`, ...), the masks, `wp.where` semantics, integer
  counters, status codes, and the order of floating-point operations inside expressions.
- Optimization assessment (DESIGN 10.6): kernel fusion opportunities along the iteration sequence,
  registers vs shared memory for the per-problem scalars, and launch-count reduction (it matters for
  the eager and host-loop modes).

## Tests
- pytest parity (`native/tests/python/ipm/`, main oracle): for each kernel, random inputs (with masks
  0/1, infinite bounds, edge cases that hit every branch: finetune, retries exhausted, max iter, primal
  and dual infeasibility detection, duality-gap check on/off), launch the reference Warp kernel from its
  factory and the native one on copies; compare all outputs (bitwise or <= 4 ulp for element-wise;
  <= 1e-14 relative for sums; status/iteration integers exact).
- C++ (`native/tests/cpp/ipm/`): hand-computed cases for termination, step length, mu and sigma.

Hand-off note: `native/docs/modules/ipm_kernels.md`.
