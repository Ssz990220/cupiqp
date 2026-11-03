# Brief: `cudss` module (cuDSS direct solver wrapper)

Worktree `~/Projects/cupiqp-native-wt/cudss`, branch `task/cudss`. Read `../AGENT_BRIEF.md` first, then
DESIGN.md (sections 4, 7, 10.1), FEASIBILITY.md sections 3.1-3.4, DEPENDENCY_GRAPH.md (row "M5 cudss"),
shared headers, and your contract `native/src/cudss/cudss_solver.hpp`.

## Start here
- Proof of concept already working: `native/poc/cudss_graph_poc.cu` (configuration, counting-free
  memory handler, plain/conditional capture, pageable-memcpy patch) and its log
  `native/poc/results/cudss_graph_poc.log`. Start from it.
- Reference: `cupiqp/sparse/sparse_direct_solver.py` (`_CudssSolver`, `CudssStackedSolver` with B = 1).
- cuDSS 0.8.0.10 headers and libs: `~/Projects/.deps/cudss-0.8.0.10` (CMake already links them; the
  threading-layer path is the compile definition `CUPIQP_CUDSS_MT_LIB`, overridable by env
  `CUDSS_THREADING_LIB` as in the reference).

## Scope (FP64, B = 1)
- `CudssSolver` per the header with the reference configuration (see header comment). Device memory
  handler on the default pool (`cudaMallocFromPoolAsync`/`cudaFreeAsync`) that counts allocs and frees
  into `Counters`. Each call does `cudssSetStream(h, s)`. Count analysis, factor and solve calls.
- Allocation rule (DESIGN 10.1): measure whether cuDSS requests memory in steady state (repeated
  factor+solve after warm-up, eager and in graphs). If it does, add a setup-preallocated device arena
  behind the handler, sized from the warm-up peak, with a counted pool fallback. Report the numbers.
- `patch_pageable_memcpy(graph, keep)`: recursive over child and conditional body graphs.
- Do not read factorization info (`cudssDataGet` blocks and breaks capture). Same as the reference.

## Tests
- C++ (`native/tests/cpp/cudss/`): random quasi-definite KKT-shaped systems (lower view, full P block
  stored) vs a dense CPU solve written in the test; repeated factor with changed values reuses one
  analysis; explicit zero entries that become nonzero; capture/replay bitwise equal to eager; SOLVE in a
  conditional body works after patching and fails without it; allocation counts after warm-up.
- A micro-benchmark executable (not a test) timing factor and solve, eager vs graph, on the MPC KKT shape:
  dim 2754, structure P (169 x 169, dense-ish 8977 nnz), A (64 x 169, 920 nnz), G (2521 x 169, 249801 nnz).
- pytest parity (`native/tests/python/cudss/`): the same matrix and rhs through the reference
  `CudssStackedSolver` and native via `_testing`; solutions should agree to about 1e-14 relative
  (bitwise if cuDSS is deterministic for identical inputs; report).

Hand-off note: `native/docs/modules/cudss.md` (allocation findings, timings).
