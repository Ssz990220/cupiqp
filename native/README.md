# cuPIQP native

Results, benchmark commands and status: [`docs/REPORT.md`](docs/REPORT.md).

Native CUDA C++ port of the cuPIQP sparse solver (milestone 1: sparse, FP64,
batch size 1, fixed sparsity, repeated numeric updates). The pinned Python/Warp
implementation in `../cupiqp/` (commit `d538dda`) is the numerical reference.

| Path | Contents |
|---|---|
| `include/cupiqp/` | public C++ API (`cupiqp.hpp`) |
| `src/common/` | buffers, error checks, CUDA graph helpers, reductions, shared views |
| `src/problem/data/` | CSR validation, problem value storage, bound masks |
| `src/problem/ruiz/` | Ruiz equilibration (scaling) |
| `src/linalg/spmv/` | cuSPARSE sparse matrix-vector products |
| `src/linalg/cudss/` | cuDSS sparse LDL^T wrapper |
| `src/linalg/kkt_assembly/` | KKT pattern, slot maps, value updates |
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

Dependencies: CUDA 13.2 toolkit, cuDSS 0.8.0.10 (redist under
`~/Projects/.deps/cudss-0.8.0.10`, override with `-DCUPIQP_CUDSS_ROOT=`), nanobind
(Python build only), GoogleTest (fetched at configure time).
