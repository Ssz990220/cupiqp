# Brief: `spmv` module (cuSPARSE sparse mat-vec operators)

Worktree `~/Projects/cupiqp-native-wt/spmv`, branch `task/spmv`. Read `README.md` (this folder) first, then
DESIGN.md, DEPENDENCY_GRAPH.md (row "M2 spmv"), shared headers, and your contract `native/src/linalg/spmv/spmv.hpp`.

## Start here (reference, pinned)
- `cupiqp/sparse/sparse_matvec.py` (`SparseMatVecProduct`: cuSPARSE SpMV, `CUSPARSE_SPMV_ALG_DEFAULT`,
  32-bit indices, base 0, compute type = value type, host scalars, buffer from `cusparseSpMV_bufferSize`).
- `cupiqp/sparse/sparse_kkt_solver.py` `eval_*`, and their callers `cupiqp/solver.py::_update_residuals_nr`
  and `cupiqp/kkt_systems.py` (zero-fill when p == 0 or m == 0). Tests: `tests/linalg/test_sparse_matvec.py`.

## Scope (FP64, B = 1)
- `CusparseHandle`, `SpMV`, `SparseOperators` per the header with toolkit cuSPARSE (CUDA 13.2). Descriptors
  and workspace in `init` (call `cusparseSpMV_preprocess` only if the reference does); each call does
  `cusparseSetStream(h, s)` and rebinds vector pointers; no allocation, no sync.
- Must capture into a plain CUDA graph. Also test inside a conditional WHILE body via
  `capture_while` (`core/graph.hpp`) and report the result (it matters for the experimental whole-solve graph).
- Empty matrices (rows, cols or nnz zero) handled without calling cuSPARSE, matching reference results.
- Optimization assessment (DESIGN 10.6): benchmark cuSPARSE vs custom CSR kernels (warp-per-row, CUB,
  shared-memory tiles) on the MPC shapes n = 169, p = 64, m = 2521, nnz(P) = 8977, nnz(A) = 920,
  nnz(G) = 249801 (G about 59% dense, 82% explicit zeros; G^T x is the interesting one). CUDA events,
  warmed, median. Keep cuSPARSE as default; a faster custom kernel within 1e-14 relative goes behind an option.

## Tests
- C++ (`native/tests/cpp/linalg/spmv/`): random CSR (explicit zeros, empty rows, rectangular, empty) vs CPU;
  alpha/beta; transposes; capture/replay bitwise equal to eager; in-place value updates seen; zero
  allocations over many calls.
- pytest parity (`native/tests/python/linalg/spmv/`): reference `SparseMatVecProduct` vs native `_testing`
  (expect bitwise equality; report otherwise).

Hand-off note: `native/docs/modules/spmv.md` (include benchmark numbers and the conditional-body result).
