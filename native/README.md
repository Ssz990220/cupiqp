# cuPIQP native

Results, benchmark commands and status: [`docs/REPORT.md`](docs/REPORT.md).

Native CUDA C++ port of the cuPIQP sparse solver (milestone 1: sparse, FP64,
batch size 1, fixed sparsity, repeated numeric updates). The pinned Python/Warp
implementation in `../cupiqp/` (commit `d538dda`) is the numerical reference.

The KKT system is solved through a dense Schur complement on the x block
(custom assembly, Cholesky and triangular solves), so **problems are limited
to n <= 2048 variables**; `setup()` rejects larger ones with
`ErrorCode::Unsupported`. The sparse products use custom CSR/CSC kernels.
The library links only the CUDA runtime (no cuDSS or cuSPARSE since
2026-11-04). By default (`GraphMode::Auto`) a solve is one CUDA graph launch.

| Path | Contents |
|---|---|
| `include/cupiqp/` | public C++ API (`cupiqp.hpp`) |
| `src/common/` | buffers, error checks, CUDA graph helpers, reductions, shared views |
| `src/problem/data/` | CSR validation, problem value storage, bound masks |
| `src/problem/ruiz/` | Ruiz equilibration (scaling) |
| `src/linalg/spmv/` | sparse matrix-vector products (custom CSR / CSC kernels) |
| `src/linalg/kkt_assembly/` | KKT pattern, slot maps, value updates |
| `src/linalg/reduced_kkt/` | KKT linear solver: dense Schur complement, Cholesky, triangular solves |
| `src/linalg/kkt_system/` | condensed KKT system solve |
| `src/ipm/kernels/` | interior-point step, residual and termination kernels |
| `src/ipm/solver/` | solver orchestration (implements the public API) |
| `python/bindings/` | nanobind module `cupiqp_native._ext` |
| `python/cupiqp_native/` | Python package mirroring `cupiqp.SparseSolver` |
| `tests/cpp/`, `tests/python/` | module tests, mirroring `src/` |
| `bench/` | native benchmarks |
| `poc/` | feasibility proofs of concept |
| `docs/design/` | feasibility, design, dependency graph, deviations |
| `docs/REPORT.md` | milestone 1 results and how to reproduce them |
| `docs/briefs/` | briefs for module agents |
| `docs/modules/` | per-module hand-off notes |

## Build and test

```bash
cd ~/Projects/cupiqp-native/native
cmake -S . -B ../build/native -DPython_EXECUTABLE=$HOME/Projects/cupiqp-native/.venv/bin/python
cmake --build ../build/native -j
(cd ../build/native && ctest --output-on-failure)
WARP_CACHE_PATH=$HOME/.cache/warp-cupiqp-native \
  ../.venv/bin/python -m pytest -q tests/python
```

Dependencies: CUDA 13.2 toolkit (runtime and CUB), nanobind (Python build
only), GoogleTest (fetched at configure time).

Python: `cupiqp_native.SparseSolver` mirrors `cupiqp.SparseSolver`. The
reference setting `use_deterministic_mode_for_cudss` is accepted and ignored
(there is no cuDSS; the solve is deterministic); see
`docs/design/DEVIATIONS.md` section 4.
