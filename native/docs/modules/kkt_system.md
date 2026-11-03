# M6 kkt_system: hand-off note

Branch `task/kkt_system`. Reference (pinned): `cupiqp/kkt_systems.py` `KKTSystem`
(forward direction), `cupiqp/kkt_systems_kernels.py`, `cupiqp/kkt_solver.py`
(`_write_solve_status`), `cupiqp/solver_kernels.py` (`create_solution_finite_kernel`),
`cupiqp/sparse/sparse_kkt_solver.py` `solve`.

## What is implemented

- All forward kernels, version 1 (thread mapping, formula order, masks as in Warp):
  `update_regularizations`, `eliminate_slacks`, `eliminate_duals`, `recover_duals`
  (`box_rhs_eliminated=False`), `recover_slacks`, `condensed_kkt_combine`, the five
  refinement kernels (`begin`, `decide`, `accept`, `rhs_norm`, `add`), the
  `wp.map(wp.mul)` mask product, the solution finite check (one block per problem,
  `cub::BlockReduce` max, block size `reduction_block_dim(n + p + m)`), and the fused
  rhs assembly / solution split.
- `KktSystem` (`init`, `update_scalings_and_factor`, `solve`) with the reference
  call sequence: regularization kernel -> `KktAssembly::update_diagonal(x_reg, delta
  copy, z_reg)` -> `CudssSolver::factor`; solve = eliminate slacks -> eliminate duals ->
  [assemble -> cuDSS solve -> split -> finite check] -> optional IR -> `G lhs.x` ->
  recover duals -> recover slacks.
- Iterative refinement exactly as the reference: begin, rhs norm, `capture_if(loop,
  first_step)`, `capture_while(loop, later_step)`, with `capture_if(correct, correct)`
  inside each step. The loop bodies are `std::function`s built once in `init()`.
- `capture_if` (warp.capture_if semantics): conditional IF node under capture, host
  read + sync otherwise. Lives in this module (see requested changes).

## Files

- `src/linalg/kkt_system/kkt_system.hpp` (contract; private members/helpers added,
  public comments made precise, no public signature changed)
- `src/linalg/kkt_system/kkt_system.cu` (class)
- `src/linalg/kkt_system/kkt_system_kernels.hpp/.cu` (kernels + launchers, `RowMat`
  pointer+row-stride views)
- `src/linalg/kkt_system/capture_if.hpp/.cu`
- `src/linalg/kkt_system/sources.cmake`
- `python/bindings/bind_kkt_system.cpp` (`_testing.kkt_*`, kernel launchers only)
- `tests/python/linalg/kkt_system/test_kkt_system_kernels.py`
- `tests/cpp/linalg/kkt_system/CMakeLists.txt`, `test_kkt_system_kernels.cu`,
  `test_kkt_system_class.cu` (guarded)

## Public API as implemented

As in the contract header. `init` assumes `kkt` is initialized, `cudss` is
initialized on `kkt`'s pattern/values and analyzed, `ops` is initialized on the same
data (the reference `KKTSystem.init` builds these itself; in the port the solver owns
them). `settings` is unused. Launchers: `cupiqp::kkt_kernels::*` in
`kkt_system_kernels.hpp`.

## Tests (exact commands, results)

```
cd native && cmake -S . -B ../build/native -DPython_EXECUTABLE=$HOME/Projects/cupiqp-native/.venv/bin/python
cmake --build ../build/native -j 8
(cd ../build/native && ctest --output-on-failure -R KktSystem)        # 3/3 passed
CUPIQP_KKT_MAX_ULP=0 CUPIQP_NATIVE_BUILD=<wt>/build/native WARP_CACHE_PATH=$HOME/.cache/warp-cupiqp-native \
  $HOME/Projects/cupiqp-native/.venv/bin/python -m pytest -q <wt>/native/tests/python/linalg/kkt_system
                                                                       # 283 passed (5 s warm, ~9 min cold Warp cache)
```

- ctest (this branch): `KktSystemCaptureIf.HostPath`, `.ConditionalNodeNestedInWhile`
  (IF node nested in a WHILE body, graph launched twice), `KktSystemKernels.FiniteCheckAndAssembly`.
- pytest: every kernel vs its Warp factory on random inputs over the 16 bound-side
  layouts x 4 shapes (incl. p = 0, m = 0, n = m = p = 1, n = 300), zero masks, G rows with
  both sides infinite (z_reg == 0), NaN/+inf/-inf in each block for the finite check,
  NaN in the combine residual (atomic max semantics), all decide branches.
- Class-level test `test_kkt_system_class` (added by CMake only when the three
  dependency implementations exist; the guard globs `.cu` or `.cpp` because cudss uses
  `cudss_solver.cpp`, not the `.cu` named in the brief). Run in a scratch copy of this
  branch overlaid with the current (uncommitted) sibling sources of spmv, kkt_assembly
  and cudss: 4/4 passed, 25/25 repeated runs: full 10-block system residual <= 1e-9
  (3 seeds, IR off; 2 seeds, IR on), infinite-bound entries exactly 0, solve_status 0;
  captured graph (update_scalings_and_factor + solve) vs eager within 1e-10 relative;
  NaN rhs -> solve_status 1.

## Parity numbers

- Kernels: bitwise identical to Warp on all 283 cases (`CUPIQP_KKT_MAX_ULP=0`).
- Class level (scratch driver binding, not committed; same SparseData, variables,
  rhs, rho, delta, device settings as the reference KKTSystem with identity Ruiz):
  shapes (10,3,5) (10,0,5) (10,3,0) (10,0,0) (20,8,9) (200,40,150), IR off/on,
  (rho, delta) in {(1,1), (1e-6,1e-4), (1e-9,1e-9)}, 2 seeds: most small cases bitwise;
  max relative difference 6.7e-13 at (1e-6, 1e-4), 8.6e-8 at (1e-9, 1e-9) without IR,
  <= 1.4e-15 with IR. The reference against itself (two KKTSystem instances, same
  inputs) differs by 7.8e-13 and 5.1e-8 on the same cases: the spread is cuDSS
  run-to-run variation (non-deterministic mode), not this module.
  solve_status equal (0) everywhere.

## Deviations from the reference

1. Fused rhs assembly and solution split (one kernel each instead of three `wp.copy`),
   DESIGN section 6 item 3. No numerical effect.
2. `has_*` / sizes are runtime kernel arguments instead of `wp.static` specializations.
   No numerical effect (bitwise parity measured).
3. Atomic max reproduces Warp's float64 `atomic_max` (CAS + `fmax`), so, as in the
   reference, a NaN residual entry is ignored by the IR norm (a NaN residual is only
   seen as non-finite if every entry is NaN). Faithful port; noted as a reference quirk.
4. Reference quirk kept: at IR step 0 the candidate `lhs_z` is `work_z`, and the mask
   product writes `work_z` in place, so inactive rows of the condensed z are zeroed
   before the combine. Identical behavior (same aliasing).
5. `solve_status` is written by the finite check of every condensed solve, including
   each IR correction (as in the reference), not after recovery.

## Optimization assessment (DESIGN 10.6)

- Tiling: all kernels are element-wise over (B, k); no per-row loops, so tiling brings
  nothing. The finite check is a single strided pass + block reduce.
- Tensor cores: none applicable (FP64 element-wise; sm_89 has no FP64 MMA).
- Fusion / registers: the solve issues eliminate_slacks + eliminate_duals + assemble
  (3 launches) and split + finite check (2) plus recover_duals + recover_slacks (2).
  Candidates: fuse eliminate_duals into the assembly kernel (it already writes
  rhs_x_bar/rhs_z_bar that are then copied), fuse split + finite check (finite check
  needs a block reduce per problem; for B = 1 a single-block grid-stride kernel or a
  flag with atomicOr would do), fuse recover_duals + recover_slacks (same thread owns
  the same inequality index). Each saves a launch and one pass over n + m or num_ineq
  doubles; not done in version 1, no measurement yet.

## Requested changes to shared code

- Move `capture_if` (`src/linalg/kkt_system/capture_if.*`) into `common/graph.*`
  next to `capture_while`; it shares `LoopContext` depth/helper streams. It takes a
  caller pinned flag for the host path; `LoopContext` could own one per depth.
- Integration (M8): with IR enabled the IR loop contains cuDSS SOLVE inside conditional
  nodes, which the cuDSS solve phase does not support (pageable memcpy). The reference
  avoids this by using NO_GRAPH when `iterative_refinement_always_enabled` on the
  sparse backend (`solver.py _cuda_graph_mode`); M8 must do the same (or rely on the
  M5 pageable-memcpy patch).
- M5 / cudss (observed in the scratch integration build, not in this module): with
  `deterministic = true`, factor + solve of a small KKT intermittently fails with
  `cudaErrorIllegalAddress` after SOLVE (3/30 and 1/40 runs of a test that only calls
  `CudssSolver::factor` / `solve`; 0/40 in non-deterministic mode). Also: destroying
  `CudssSolver` after the stream it last ran on segfaults in `cudaFreeAsync` (the memory
  handler frees on that stream); owners must destroy the stream last.

## Known gaps

- Class-level C++ test and class-level reference parity only run outside this branch
  (scratch overlay of uncommitted sibling sources); the integration owner should run
  `test_kkt_system_class` after merging M2/M4/M5. A pytest class-level parity test
  needs a `KktSystem` binding, which cannot link on this branch.
- B > 1 is not supported in `KktSystem` itself (rhs.y / lhs.x / lhs.y are used as
  contiguous rows; kernels and launchers do support row strides).
- IR under graph capture is tested only through the IF-in-WHILE capture_if test; the
  full IR path was run in host-loop mode (eager), matching the reference's NO_GRAPH use.
- Transposed solve (`transpose=True`), `mul_condensed_kkt`, `get_refinement_error` are
  out of scope and not ported.
