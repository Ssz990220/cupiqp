# Brief: `kkt_assembly` module (KKT pattern, slot maps, value updates)

Worktree `~/Projects/cupiqp-native-wt/kkt_assembly`, branch `task/kkt_assembly`. Read `README.md` (this folder)
first, then DESIGN.md, DEPENDENCY_GRAPH.md (row "M4 kkt_assembly"), shared headers, and your contract
`native/src/linalg/kkt_assembly/kkt_assembly.hpp`.

## Start here (reference, pinned)
- `cupiqp/sparse/sparse_kkt_solver.py`: `__init__`, `_initialize_kkt_csr`, `_refresh_P_diag_buffer`,
  `_scatter_P_A_G`, `update_data`, and the diagonal part of `update_and_factor_kkt`.
- `cupiqp/sparse/sparse_kkt_solver_kernels.py` (all four kernels), `cupiqp/sparse/csr_helpers.py`
  (`csr_diag_indices`, `csr_row_indices`, `csr_subblock_indices`).

## Scope (FP64, B = 1)
- Build on the host in C++ (no SciPy) exactly the pattern of
  `bmat([[P + I_n, None, None], [A, I_p, None], [G, None, I_m]])` after `sum_duplicates()` (sorted
  columns, union pattern). The reference builds it from all-ones patterns, so the initial KKT values are
  the bmat sums: 2 on a P diagonal stored by P, 1 elsewhere. Reproduce these template values bitwise
  before the scatter; slots that are never scattered keep them (e.g. identity diagonals before
  `update_diagonal`).
- Slot maps for P, A, G entries (explicit zeros keep slots), diagonal slots for the x/y/z blocks,
  P-diagonal source/destination maps, `P_diag` refresh, masked G scatter (`G * active_G_row[row]`), and
  `update_diagonal` (x: P_diag + x_reg; y: -delta; z: -z_reg if z_reg > 0 else -1).
- Kernels mirror Warp version 1. Optimization assessment: fusing scatter + diagonal update, and
  coalescing (sorted slot maps) on the MPC shape (n = 169, p = 64, m = 2521, nnz(G) = 249801).
- Your tests build `SparseDataView` inputs with a small fixture (the `data` module is built in parallel).

## Tests
- C++ (`native/tests/cpp/linalg/kkt_assembly/`): pattern vs an independent dense construction in the test;
  slot maps land every entry where expected; explicit zeros that later become nonzero land in their
  slot; inactive G rows are zeroed; `update_diagonal`; addresses stable and no allocation across updates.
- pytest parity (`native/tests/python/linalg/kkt_assembly/`): reference `SparseKKTSolver` (built from a reference
  `SparseData`; its `_kkt_mats` holds indptr/indices/data). Compare pattern and values bitwise after
  init, after `update_data` with new values and bound switches, and after the diagonal update with random
  x_reg/delta/z_reg (z_reg including zeros).

Hand-off note: `native/docs/modules/kkt_assembly.md`.
