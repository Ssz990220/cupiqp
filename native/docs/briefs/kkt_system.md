# Brief: `kkt_system` module (condensed KKT system)

Worktree `~/Projects/cupiqp-native-wt/kkt_system`, branch `task/kkt_system`. Read `README.md` (this folder)
first, then DESIGN.md, DEPENDENCY_GRAPH.md (row "M6 kkt_system"), shared headers, and your contract
`native/src/linalg/kkt_system/kkt_system.hpp` (it depends on the headers of `spmv`, `kkt_assembly`, `cudss`,
whose implementations are built in parallel by other agents).

## Start here (reference, pinned)
- `cupiqp/kkt_systems.py` `KKTSystem`: `init`, `update_scalings_and_factor`, `_update_regularization`,
  `solve`, `_prepare_rhs`, `_recover_lhs`, `iterative_refinement` (forward direction only;
  transpose/backward is out of scope).
- `cupiqp/kkt_systems_kernels.py` (forward kernels and refinement kernels).
- `cupiqp/sparse/sparse_kkt_solver.py` `solve` (rhs assembly, cuDSS solve, split, `_write_solve_status`)
  and `cupiqp/kkt_solver.py` (`_write_solve_status`, `create_solution_finite_kernel` from
  `cupiqp/solver_kernels.py`: this module owns the native finite-check kernel).

## Scope (FP64, B = 1)
- All forward kernels ported 1:1 from Warp (version 1), with launchers in a separate
  `kkt_system_kernels.cu`/`.hpp` so they are testable on their own. The `KktSystem` class in
  `kkt_system.cu` per the header. Fuse the three rhs copies into one assembly kernel and the three split
  copies into one (documented deviation without numerical effect).
- Iterative refinement (`iterative_refinement_always_enabled`): implement with `capture_while`
  (`core/graph.hpp`) as the reference uses `wp.capture_while`. Default settings keep it off.
- Linking: until the dependency modules are merged, an executable that references `KktSystem` will not
  link. Keep the kernel parity tests and `bind_kkt_system.cpp` limited to the kernel launchers. Put the
  class-level C++ tests in a separate test target added only when
  `${CMAKE_SOURCE_DIR}/src/linalg/cudss/cudss_solver.cu`, `src/linalg/spmv/spmv.cu` and `src/linalg/kkt_assembly/kkt_assembly.cu`
  exist (CMake `if(EXISTS ...)`). The integration owner runs them after merging.

## Tests
- pytest parity (`native/tests/python/linalg/kkt_system/`): each reference Warp kernel (via its factory in
  `cupiqp/kkt_systems_kernels.py` / `solver_kernels.py`) on random inputs, including zero and infinite
  masks and z_reg == 0 rows, vs the native launchers; bitwise or <= 4 ulp.
- Class-level (guarded) C++ test: assemble a small problem, run `update_scalings_and_factor` + `solve`,
  check the full un-condensed 4x4 block system residual on the CPU (formula in the `KKTSystem` docstring).

Hand-off note: `native/docs/modules/kkt_system.md`.
