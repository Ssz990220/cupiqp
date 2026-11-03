# Module M2 `spmv`: hand-off note

Graph-safe sparse matrix-vector products `P x`, `A x`, `A^T y`, `G x`, `G^T z`
(reference `cupiqp/sparse/sparse_matvec.py` `SparseMatVecProduct` and
`SparseKKTSolver.eval_*`). FP64, B = 1.

## What is implemented

- `CusparseHandle`: RAII cuSPARSE handle.
- `SpMV`: `y = alpha op(M) x + beta y` for one CSR matrix. The default
  backend mirrors the reference call: one CSR descriptor over the caller's
  arrays (32-bit indices, base 0, `CUDA_R_64F` value and compute type),
  dense-vector descriptors whose pointers are rebound on each call,
  `CUSPARSE_SPMV_ALG_DEFAULT`, host-pointer alpha/beta, and a workspace sized
  once by `cusparseSpMV_bufferSize`. `cusparseSpMV_preprocess` is not called
  (the reference does not call it either). Each call runs `cusparseSetStream(h, s)`,
  `cusparseDnVecSetValues` x2, and `cusparseSpMV`. It does not allocate or
  synchronize, and it can be captured into a graph. In-place updates of
  `M.values` are seen by the next call.
- Empty operators (rows, cols or nnz zero): cuSPARSE is not created or called.
  `y = beta y` is computed directly: `cudaMemsetAsync` to 0 when beta == 0 (y is
  not read, so NaN in y gives 0, as in cuSPARSE), a scale kernel when
  beta != 0 and beta != 1, and nothing when beta == 1. When y has length 0 the
  call does nothing.
- `SparseOperators`: the five products with beta = 0. When a block is absent
  (p == 0 or m == 0), it becomes an empty `0 x n` operator. `A x` and `G x`
  then write nothing, and `A^T y` and `G^T z` write n zeros. KKTSystem skips
  these products, and its zero-initialized work buffers hold the same zeros.
- Opt-in custom backend `SpmvBackend::Custom` (off by default). See the
  optimization assessment below.

## Files

- `src/linalg/spmv/spmv.hpp`: contract plus additions (listed below).
- `src/linalg/spmv/spmv.cu`: implementation.
- `src/linalg/spmv/spmv_kernels.cuh`: `scale_kernel` (empty fallback) and
  `csr_lanes_kernel<LANES>` (custom backend).
- `src/linalg/spmv/sources.cmake`
- `python/bindings/bind_spmv.cpp`: `_testing.SpMVOp(rows, cols, nnz, indptr,
  indices, values, transpose=False, custom=False)`, which is called as
  `(x, y, alpha, beta, stream=0, graph=False)` and has the properties
  `uses_cusparse`, `uses_custom` and `lanes`; and
  `_testing.SparseOps(n, p, m, nnz_P, P_indptr, P_indices, P_values, nnz_A, ...,
  G_values, custom=False)`, called as `.apply(which, alpha, x, y, stream=0, graph=False)`.
  With `graph=True` the binding captures the call, instantiates the graph,
  launches it once, and synchronizes.
- `tests/cpp/linalg/spmv/{CMakeLists.txt, test_spmv.cu, spmv_test_util.cuh, bench_spmv.cu}`
- `tests/python/linalg/spmv/test_spmv_parity.py`

## Public API as implemented (additions to the contract, no signature changed)

```cpp
enum class SpmvBackend { Cusparse, Custom };
class SpMV {
  void init(cusparseHandle_t h, const CsrView<double>& M, bool transpose);          // = Cusparse
  void init(cusparseHandle_t h, const CsrView<double>& M, bool transpose, SpmvBackend backend);
  void operator()(cudaStream_t s, double alpha, const double* x, double beta, double* y);
  int32_t x_len() const; int32_t y_len() const;
  bool uses_cusparse() const; bool uses_custom() const; int lanes() const;
  // private: release(), init_custom(), custom-backend members
};
class SparseOperators {
  void init(cusparseHandle_t h, const SparseDataView<double>& data);                 // = Cusparse
  void init(cusparseHandle_t h, const SparseDataView<double>& data, SpmvBackend backend);
  // P, A, AT, G, GT unchanged
};
```
`init` may be called again: it releases the previous descriptors and buffers.
Under the Custom backend, `init` for op = T copies the pattern to the host,
builds the CSC pattern and value permutation, uploads them, and calls
`cudaDeviceSynchronize` (setup only).

## Tests (run on this worktree, RTX 4070 SUPER)

```bash
cd ~/Projects/cupiqp-native-wt/spmv/build/native
cmake --build . -j 8
ctest --output-on-failure -R Spmv          # 39/39 passed (all 46 ctests in the build pass)
CUPIQP_NATIVE_BUILD=$HOME/Projects/cupiqp-native-wt/spmv/build/native \
  WARP_CACHE_PATH=$HOME/.cache/warp-cupiqp-native \
  $HOME/Projects/cupiqp-native/.venv/bin/python -m pytest -q \
  $HOME/Projects/cupiqp-native-wt/spmv/native/tests/python/linalg/spmv   # 317 passed
./tests/cpp/linalg/spmv/bench_spmv 21 200                               # benchmark
```

C++ (`test_spmv`, 39 tests, each run with both backends):
- Random CSR matrices (explicit zeros, empty rows; shapes 1x1, 1x8, 8x1,
  200x32, 32x200, 37x53 at densities 0.05 and 0.9, and 1024x256), op = N and
  op = T, with (alpha, beta) in {(1,0), (2,0.5), (-1.5,2), (0.75,1)}. Results
  are compared with a long-double CPU product and must agree within
  1e-14 x (|alpha| |op(M)| |x| + |beta| |y|) per entry.
- Empty matrices (0x5, 5x0, 4x6 with nnz = 0, 0x0) for both ops: beta = 0
  turns a NaN-filled y into 0, beta = -2 scales y, beta = 1 leaves y unchanged.
- Capture and replay: replay must be bitwise equal to the eager call wherever
  the backend is deterministic (see below). In every case it must be within
  1e-14 of the CPU result and of the eager result.
- In-place value updates are seen by both eager calls and graph replays.
- Inside a conditional WHILE body (`capture_while`): 6 iterations of
  `t = 0.5 op(M) x; y += t`, captured and replayed 3 times, compared with the
  host-loop path.
- No allocation or synchronization per call. 4000 eager calls leave the default
  memory pool's `UsedMemCurrent` unchanged (device free memory was also
  unchanged, 0 bytes, but that is report-only because the GPU is shared).
  100 calls capture without error under `cudaStreamCaptureModeGlobal`, which
  rejects cudaMalloc and synchronization. The captured graph has kernel nodes
  only: 300 for cuSPARSE, 100 for Custom.
- `SparseOperators` with (n, p, m) = (20,5,30), (20,0,30), (20,5,0), (20,0,0)
  against the CPU. Absent blocks give zeros over NaN-filled outputs.

pytest (317 tests): the reference `SparseMatVecProduct` and the native
`SpMVOp` run on the same Warp CSR arrays and inputs. 9 shapes (the list above
plus 630x169 at density 0.59), N/T, the 4 (alpha, beta) pairs, eager/graph,
cusparse/custom. Also empty matrices against the reference (rows, cols > 0
with nnz = 0), and `SparseOps` against the reference products for the 4
(n, p, m) layouts.

## Conditional graph node result

SpMV works inside a conditional WHILE body. With both backends, the
`capture_while` body (SpMV + axpy + condition kernel) captures, instantiates
and replays correctly, and matches the host loop: bitwise for op = N, and
within 1e-14 for cuSPARSE op = T, which is not reproducible run to run (see
below). Every call binds the handle to the body stream with `cusparseSetStream`.

## Determinism finding (important for parity expectations)

Measured for cuSPARSE 12.7 (toolkit 13.2) CSR, `ALG_DEFAULT`, on 2521x169 at
59% density, 20 eager repeats:

| op | beta | repeats bitwise equal |
|---|---|---|
| N | 0 | yes (0 of 19 differ) |
| N | 0.5 | no (19 of 19 differ from the first) |
| T | 0 | no (19/19) |
| T | 0.5 | no (19/19) |

The reference has the same behavior: two reference calls of `G^T` differ
(max abs 2.7e-14). The reference therefore cannot be matched bitwise on
`A^T y` and `G^T z`, the two transposed products the solver uses. The custom
backend is bitwise reproducible for every op and beta.

## Parity vs reference (pytest inputs, aggregated, eager)

Max abs and max relative difference to the magnitude sum
|alpha| |op(M)| |x| + |beta| |y|:

| backend | op | beta | max abs | max rel | bitwise equal |
|---|---|---|---|---|---|
| cuSPARSE | N | 0 | 0 | 0 | 9/9 |
| cuSPARSE | N | != 0 | 3.6e-15 | 8.4e-17 | 21/27 |
| cuSPARSE | T | 0 | 3.2e-14 | 2.5e-16 | 4/9 |
| cuSPARSE | T | != 0 | 7.1e-14 | 2.8e-16 | 12/27 |
| Custom | N | 0 | 7.1e-15 | 2.3e-16 | 2/9 |
| Custom | N | != 0 | 1.4e-14 | 3.1e-16 | 7/27 |
| Custom | T | 0 | 2.8e-14 | 4.1e-16 | 3/9 |
| Custom | T | != 0 | 7.1e-14 | 4.5e-16 | 8/27 |

On the MPC-like G (2521x169, 82% explicit zeros, `test_parity_summary`):
- G x: native cuSPARSE is bitwise equal to the reference. Custom differs by at
  most 3.6e-15 abs (rel 2.6e-16).
- G^T z: native cuSPARSE vs the reference gives 2.8e-14 abs (rel 1.6e-16), and
  the reference vs itself gives 2.7e-14. Custom vs the reference gives
  9.2e-14 abs (rel 5.2e-16).

The cuSPARSE library differs from the reference's (toolkit 12.7 vs wheel
12.8.6), but where the algorithm is deterministic (op = N, beta = 0) the
results are bitwise equal.

## Optimization assessment (DESIGN 10.6)

The benchmark `bench_spmv` uses synthetic random patterns with the MPC sizes:
P 169x169 (nnz 8977), A 64x169 (nnz 920), G 2521x169 (nnz 249801, 82% stored
zeros). Timings use CUDA events, 3 warm-up batches, and 21 repeats of 200
calls each. The table gives the median over 3 runs of the per-run medians in
us per call. "graph" replays one captured graph of 200 calls, which measures
GPU time. "stream" enqueues the calls eagerly and includes the host overhead.
The GPU was shared with other agents.

| product | cuSPARSE ALG_DEFAULT stream / graph | Custom stream / graph | Custom lanes | speedup (graph) |
|---|---|---|---|---|
| P x | 15.7 / 10.6 | 5.4 / 1.60 | 32 | 6.6x |
| P^T x | 15.6 / 10.6 | 5.4 / 1.67 | 32 | 6.3x |
| A x | 14.5 / 10.2 | 4.9 / 1.35 | 16 | 7.5x |
| A^T y | 14.7 / 10.3 | 4.8 / 1.33 | 8 | 7.7x |
| G x | 26.2 / 9.0 | 7.6 / 3.72 | 8 | 2.4x |
| **G^T z** | **50.2 / 48.8** | **8.0 / 6.71** | 32 | **7.3x** |

Other variants measured (graph us): G x with cuSPARSE CSR_ALG2 11.8, thread
per row 7.5, 4/8/16 lanes per row 3.9/3.6/3.9, warp per row 5.3, block per row
(cub) 12.7. For G^T z: atomic scatter (thread per nnz) 46.5, shared-memory
accumulator scatter 14.7, CSC gather with 8 lanes 12.1, warp per column 6.7,
cub block of 128/256 per column 7.9/7.9. Full table: run `bench_spmv`.

- cuSPARSE's cost on the small products (about 10 us) is mostly fixed per
  call. It launches about 3 kernels per call (300 graph nodes per 100 calls),
  against 1 for Custom.
- cuSPARSE `G^T z` is atomic-bound: it is about the same speed as a naive
  atomic scatter. A gather over the CSC view of the pattern, with one warp per
  column and a value permutation, is 7x faster and deterministic.
- Custom kernel: `LANES` threads per output row stride over the row and are
  reduced with a fixed shuffle tree. LANES is the largest power of two that is
  at most 32, at most the next power of two of the mean row length, and keeps
  rows * LANES <= 32768. This rule picks the best measured variant for each
  product to within noise (G x 8, the rest 16-32). The threshold was tuned on
  this GPU only.
- Tiling and shared memory: a shared-memory accumulator for the transposed
  product (n = 169 fits in shared memory) was slower than the gather and is
  not deterministic. Block-per-row cub reductions lost to warp shuffles at
  these row lengths.
- Tensor cores: none for FP64 on sm_89. On sm_80/sm_90, DMMA could serve G as
  a dense 2521x169 block (59% stored), but the values are updated in place and
  82% of the stored entries are explicit zeros, so a dense copy would need a
  scatter at every update. Not pursued.
- Register reuse: x is read through the L1/L2 caches (169 or 2521 doubles).
  The whole working set (about 4 MB for G) fits in the 48 MB L2, so the
  kernels are not DRAM-bound at these sizes.

Decision: cuSPARSE stays the default, because it matches the reference
bitwise for op = N with beta == 0. `SpmvBackend::Custom` is available through
`SpMV::init(..., backend)` and `SparseOperators::init(h, data, backend)`. Its
error against the exact product is at most 5e-16 relative to the magnitude
sum, which meets the 1e-14 requirement. It is bitwise reproducible, and on
these shapes it is 2.4-7.7x faster on the GPU. G^T z is called in every
residual evaluation and in each condensed KKT apply, so the gain is about
42 us per call.

## Deviations from the reference

- One cuSPARSE handle is shared by the five operators (`SparseOperators`
  takes `h`). The reference creates one handle per `SparseMatVecProduct`.
  Each call sets the stream, so the results are the same.
- Empty operators never call cuSPARSE. The reference never builds A/G
  operators with zero rows. It does call cuSPARSE when nnz = 0 and rows,
  cols > 0, and our direct `beta * y` matches that result bitwise (tested).
- No staging buffers for row-strided x/y views: B = 1 vectors are always
  contiguous.
- With beta == 0 the empty fallback writes 0 without reading y. This is the
  same convention as cuSPARSE.

## Requested changes to shared code

- None required. Suggestion for integration (M5/M8): expose the backend
  choice as a solver setting (for example `deterministic_spmv` or
  `spmv_backend`) passed to `SparseOperators::init`. The custom backend gives
  run-to-run reproducible residuals, which cuSPARSE `A^T`/`G^T` do not.
- Build note: `bench_spmv` is built from `tests/cpp/linalg/spmv/CMakeLists.txt`
  (it is not a ctest), because `native/bench/` does not exist and the
  top-level CMake is read-only for this module.

## Known gaps

- B > 1 is not supported. The reference's block-diagonal batched pattern is
  not ported (milestone 1 is B = 1), and `SparseOperators::init` requires
  `dims.B == 1`.
- The benchmark uses synthetic random patterns with the MPC sizes, not the
  real MPC matrices (no snapshot data in the repository). The row-length
  distribution of the real G may move the best lane count.
- The lane threshold (32768 threads) was tuned on an RTX 4070 SUPER only.
- The Custom backend for op = T builds the CSC pattern on the host in `init`.
  If the pattern changes, `init` must be called again (patterns are fixed
  after setup in this design).
