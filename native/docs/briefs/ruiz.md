# Brief: `ruiz` module (sparse Ruiz equilibration)

Worktree `~/Projects/cupiqp-native-wt/ruiz`, branch `task/ruiz`. Read `README.md` (this folder) first, then
DESIGN.md, DEPENDENCY_GRAPH.md (row "M3 ruiz"), FEASIBILITY.md section 3.5, shared headers (note
`SparseDataView`, `ScalingView`, `VariablesView`, `UpdateMask` in `src/common/views.hpp`), and your
contract `native/src/problem/ruiz/ruiz.hpp`.

## Start here (reference, pinned)
- `cupiqp/preconditioner.py` `RuizEquilibration.scale_data` (the pass loop, lines ~358-422), then
  `unscale_data`, `reuse_scaling`, `reset`, `refresh_x_b_scaling`, `unscale_solution`, `scale_solution`,
  `compute_constraints_rhs_inf_norm_unscaled`, `_scale_bounds`, `_unscale_bounds`.
- Kernels: `cupiqp/preconditioner_kernels.py`, `cupiqp/sparse/sparse_preconditioner.py`,
  `cupiqp/sparse/sparse_preconditioner_kernels.py`. Callers: `cupiqp/solver.py` `_setup_impl`,
  `_update_impl`, `_solve_impl`.

## Scope (FP64, B = 1)
- Every header method, with kernels ported 1:1 from Warp (constants min_scaling 1e-4, max_scaling 1e4,
  convergence_tol 1e-3).
- Required difference 1, device early exit: the convergence kernel sets a device `done` flag, and every
  kernel of later passes is a no-op. The same passes run as in the reference, with no host sync; count
  them in `passes_run`. `calc_scaling_inv_and_scale_bounds` runs after the loop regardless.
- Required difference 2: `unscale_data(data, replaced, s)` skips replaced blocks. Retained blocks must
  match the reference exactly. Work out which per-block operations are independent (P and c share cost
  scaling; A/b; G/h_l/h_u; box bounds). The `x_b_scaling` update always runs.
- `x_b_scaling_init` aliases `data.active_x_bound` when any box side exists, else a ones buffer.
  Handle absent A, G, and bound sides.
- Reductions through `core/reduce.cuh` (cub). Max-norms are order-independent, so aim for bitwise
  parity. For CSR column norms, find how the reference computes them and reproduce deterministically.
- The `data` module is built in parallel: your tests build `SparseDataView` inputs with a small fixture
  (upload CSR + vectors into `CudaBuffer`s, masks on the host with the same rule).

## Tests
- pytest parity (`native/tests/python/problem/ruiz/`, main oracle): several problems (`tests.helpers.random_feasible_qp`,
  SciPy random with explicit zeros, with/without A, G and each bound side, infinite bounds, scale_cost
  on/off, preconditioner_iter 0/1/10). Compare scaled data, delta, delta_b, cost_scaling, x_b_scaling,
  unscale factors, RHS norm, pass count, (un)scale_solution, and the solver's update sequence
  (unscale_data -> overwrite blocks -> reset + scale_data, and the reuse path) against the reference.
- C++ (`native/tests/cpp/problem/ruiz/`): iter 0 is identity; scale -> unscale round trip ~1e-15; device early exit
  pass count equals a host-driven loop in the test; no allocation; scale_data captures and replays identically.

Hand-off note: `native/docs/modules/ruiz.md` (optimization assessment on the MPC shape: pass fusion,
shared-memory/tile CSR norms).
