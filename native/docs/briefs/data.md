# Brief: `data` module (problem storage)

Worktree `~/Projects/cupiqp-native-wt/data`, branch `task/data`. Read `README.md` (this folder) first (rules,
build/test commands, ownership, commit format), then DESIGN.md, DEPENDENCY_GRAPH.md (row "M1 data"),
the shared headers in `native/include/cupiqp/` and `native/src/common/`, and your contract
`native/src/problem/data/sparse_data.hpp`.

## Start here (reference, pinned)
- `cupiqp/sparse/sparse_solver.py`: `_check_csr`, `_check_dense_vector`, `setup`, `update`.
- `cupiqp/data.py` (`init`, `set_*`, `_update_finite_bound_masks`), `cupiqp/data_kernels.py`
  (`create_finite_bound_masks_kernel`), `cupiqp/sparse/sparse_data.py`, `batched_csr.py`, `csr_helpers.py`.
- Expected behavior: `tests/solver/test_input_validation.py`, `tests/problem_data/test_sparse_data.py`.

## Scope (FP64, B = 1)
- `validate_csr`: indptr starts at 0, non-decreasing, ends at nnz; indices in [0, cols); strictly
  increasing within each row; nnz < 2^31. Messages close to the reference.
- `SparseData::init`: the reference's block/vector consistency rules (P and c required; A <-> b; G
  needs h_l or h_u; bounds without their block; lengths). Absent A/G give p = 0 / m = 0 with valid
  empty views. Bound sides given at setup fix num_hl/num_hu/num_xl/num_xu (full length or 0).
  Allocate everything here; upload patterns plus the COO row index of every stored entry (`row_idx`).
- `set_values` / `set_values_from_host`: stream-ordered copies, return the `UpdateMask`, reject a bound
  side absent at setup, refresh masks when a bound changed. No allocation, no sync.
- `update_finite_masks`: port the Warp kernel (lower finite iff v > -1e20, upper finite iff v < 1e20; layout [hl|hu|xl|xu];
  active_G_row = max(fl, fu); active_x_bound; num_finite_bounds). Match the reference on NaN and on
  +-inf vs +-1e20 exactly.

## Tests
- C++ (`native/tests/cpp/problem/data/`): validation accept/reject; explicit zeros kept in value order; masks
  for all-finite, one-sided, mixed, and finite<->infinite switches; repeated `set_values` keeps all
  device addresses and allocates nothing (`cudaMemGetInfo` before/after a loop).
- pytest parity (`native/tests/python/problem/data/`): problems from `tests.helpers` and SciPy; reference
  `SparseData` (e.g. `cupiqp.SparseSolver` with `preconditioner_iter = 0`, read `solver.data`) vs native
  via `_testing`; values and masks bitwise equal, also after bound switches and explicit zeros
  becoming nonzero.
- `bind_data.cpp`: a small `_testing` wrapper class (init from pattern arrays + device addresses,
  set_values, view pointers as ints, dims).

Hand-off note: `native/docs/modules/data.md`.
