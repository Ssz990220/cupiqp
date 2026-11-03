# cuPIQP native: design

Status: milestone 1 (sparse, FP64, batch size 1, fixed sparsity, repeated numeric
updates). Numerical reference: pinned Python/Warp cuPIQP at
`d538dda8c93a76a16fcddc2f98c24c98133a7937`. See FEASIBILITY.md for the audit and
proofs of concept; DEPENDENCY_GRAPH.md for module ownership and order.

## 1. Goals and non-goals

Goals
- `libcupiqp_native`: a C++17/CUDA library whose solve path depends only on the
  CUDA runtime, cuSPARSE, and cuDSS (no Python, Warp, or JAX).
- Same algorithm, convergence tests, regularization, scaling, and status
  semantics as the reference sparse backend.
- Two front ends over one core:
  1. the native C++ API (`include/cupiqp/cupiqp.hpp`);
  2. a nanobind module `cupiqp_native` that mirrors the reference
     `cupiqp.SparseSolver` Python surface, so the upstream test suite helpers
     and the benchmark harness can drive the port with minimal edits.
- Zero device allocations and zero host synchronizations inside `update()` and
  inside an IPM iteration, apart from the reference's one continue-flag read per
  iteration in the default graph mode.

Non-goals for milestone 1 (rejected at setup with a clear error): dense and
multistage backends, batch size > 1, FP32, gradients (`enable_grad`, smoothing,
`backward`), verbose per-iteration printing in graph modes, and capture of
`solve()` into a user graph. The code keeps a leading batch axis in all
per-problem buffers and passes `B` to kernels so that batching can be added later.
Only `B = 1` is built and tested.

## 2. Public C++ API

```cpp
namespace cupiqp {

// Host-side CSR pattern. Rows are sorted by column, no duplicates, 0-based.
// Explicit zeros are stored entries and keep their slot forever.
struct CsrPattern {
  int32_t rows = 0, cols = 0;
  std::vector<int32_t> indptr;   // rows + 1
  std::vector<int32_t> indices;  // nnz
  int32_t nnz() const;
  static CsrPattern from_int64(...);  // validated narrowing of int64 patterns
};

// Device pointers to value arrays in the order of the setup() patterns.
// nullptr = block absent (setup) or unchanged (update).
struct ValuesView {
  const double *P = nullptr, *c = nullptr, *A = nullptr, *b = nullptr,
               *G = nullptr, *h_l = nullptr, *h_u = nullptr,
               *x_l = nullptr, *x_u = nullptr;
};

enum class Status : int32_t { Unsolved = -1, Solved = 0, MaxIterReached = 1,
  PrimalInfeasible = 2, DualInfeasible = 3, NumericalIssues = 4 };

enum class GraphMode : int32_t {
  Auto,                    // = IterationGraphHostLoop when enable_cuda_graph, else None
  None,                    // plain launches, host-driven loops (reference NO_GRAPH)
  IterationGraphHostLoop,  // reference default for the sparse backend
  SolveGraphDeviceLoop     // EXPERIMENTAL: whole solve in one graph (FEASIBILITY 3.3)
};

struct Settings { /* every field of cupiqp.Settings for FP64 with the same names and
                     defaults (Settings.for_dtype(float64)), plus graph_mode */ };

struct Info {  // host copy of per-problem diagnostics (reference Result.info)
  Status status; int32_t iter, factor_retries, no_primal_update, no_dual_update;
  double rho, delta, mu, sigma, primal_step, dual_step, primal_res, primal_res_rel,
         dual_res, dual_res_rel, primal_res_reg, primal_res_reg_rel, dual_res_reg,
         dual_res_reg_rel, primal_prox_inf, dual_prox_inf, prev_primal_res,
         prev_dual_res, primal_obj, dual_obj, duality_gap, duality_gap_rel, reg_limit;
};

struct ResultView {  // device pointers, stable from setup() to destruction
  const double *x, *y, *z_l, *z_u, *z_bl, *z_bu, *s_l, *s_u, *s_bl, *s_bu;
  int32_t n, p, m;   // z_l/z_u/s_l/s_u have length m when the side exists, else 0
};

class SparseSolver {
 public:
  explicit SparseSolver(cudaStream_t stream = nullptr);  // nullptr: create and own a stream
  Settings& settings();
  void setup(const CsrPattern& P, const ValuesView& v,
             const CsrPattern* A = nullptr, const CsrPattern* G = nullptr);
  void update(const ValuesView& v);          // async; see section 2.2
  void update_from_host(const ValuesView& host_v);  // values in host memory (pinned preferred)
  void solve();                               // async
  ResultView result() const;
  void download_x(double* host, cudaEvent_t done = nullptr);  // async D2H into caller memory
  Info info();                                // synchronizes the stream
  cudaStream_t stream() const;
  Counters counters() const;                  // syncs, graph launches, cuDSS allocs, captures
  std::vector<DebugBuffer> debug_buffers();   // named device spans for parity tests
};
}
```

### 2.1 Index types and value order
- Patterns are host `int32` CSR (`int64` accepted and narrowed after range
  checks, matching the reference's `nnz < 2^31` rule). Validation is identical to
  `sparse_solver._check_csr`: `indptr[0] == 0`, non-decreasing, `indptr[rows] == nnz`,
  column indices in range and strictly increasing within each row.
- `P` must store both triangles (full symmetric pattern), as in the reference.
- Value arrays are in the stored-entry order of the pattern given to `setup()`.
  Explicit zeros are retained, can later become nonzero, and keep the same KKT
  slot.
- Bounds use `PIQP_INF = 1e20`, tested one-sided as in the reference
  `finite_bound_masks_kernel`: a lower bound (`h_l`, `x_l`) is finite iff
  `v > -1e20`, and an upper bound (`h_u`, `x_u`) is finite iff `v < 1e20`.
  Bound sides given at setup (`h_l`, `h_u`, `x_l`, `x_u`) are fixed for the life of
  the solver; their values may switch between finite and infinite.

### 2.2 Streams, buffer lifetime, and when results can be read
- Every operation is enqueued on `stream()`. A stream passed to the constructor
  is borrowed and never destroyed. An owned stream is created with
  `cudaStreamDefault` (blocking with respect to the legacy default stream), as
  Warp does.
- `setup()` copies the patterns to the device and the values into solver-owned
  storage. It synchronizes once at the end, because analysis needs the host.
- `update()` enqueues device-to-device copies of the given values into
  solver-owned storage and then the scaling and KKT value updates. The caller
  must make the inputs valid in stream order before the call: produce them on
  `stream()`, or make `stream()` wait on an event. The caller may overwrite an
  input buffer once `stream()` has passed the update; any later
  stream-ordered work satisfies this.
- `update_from_host()` copies into fixed pinned staging and then into device
  storage. With pinned caller memory it is one H2D copy per block and does not
  synchronize.
- `solve()` enqueues the full solve. In the default graph mode it also blocks on
  one 8-byte D2H copy of the loop state per IPM iteration, as the reference
  does. On return, the solution is complete in stream order. Device results
  (`result()`) can be consumed by work enqueued later on `stream()`. Host reads
  must go through `download_x` plus `cudaStreamSynchronize`/event, or `info()`.
- All device buffers are allocated in `setup()` and keep their addresses until
  the solver is destroyed. `update()` and `solve()` never allocate. cuDSS
  workspace comes from a counting allocator; see section 4.

### 2.3 Error handling
- API misuse (invalid patterns, wrong call order, unsupported settings, frozen
  setting changed after setup) throws `cupiqp::Error` (derived from
  `std::runtime_error`) with a code and message. No state changes on throw.
- CUDA, cuSPARSE, and cuDSS errors in setup throw `cupiqp::Error` with the
  library status code and call site.
- In the hot path, launches are checked with `cudaGetLastError` (no sync). A
  library error throws, after which the solver is unusable (`Status` reads
  `NumericalIssues`).
- Numerical failures never throw. They become statuses exactly as in the
  reference: factor retries, then `NumericalIssues`; a non-finite KKT solution
  gives `NumericalIssues`; iteration cap gives `MaxIterReached`. The reference
  never reads cuDSS's factorization info, because that query blocks and breaks
  capture. The port does the same: `factor_status` stays 0, and failures surface
  through the solution finite check (`mark_failed`).

## 3. Python binding (nanobind)

`native/python/` builds `cupiqp_native._ext` (nanobind 3.1) and a pure-Python
package `cupiqp_native` that mirrors the reference surface used by tests and the
harness:

- `cupiqp_native.SparseSolver(dtype=float64, stream=None)`, with
  `.settings`, `.setup(P=(indptr, indices, values), c=..., A=..., b=..., G=..., h_l=..., h_u=..., x_l=..., x_u=...)`,
  `.update(...)`, `.solve()`, `.result` and `.stream`.
- Inputs: any object with `__cuda_array_interface__` (Warp, CuPy, Torch) for
  values, and host or device integer arrays for patterns. Value arrays are
  passed to C++ as raw device pointers. A host NumPy value array is rejected,
  as in the reference.
- `.result.x` etc. are returned as zero-copy Warp arrays when `warp` is
  importable (test and harness convenience). Otherwise they are returned as
  DLPack capsules or objects exposing `__cuda_array_interface__`. `.result.info`
  mirrors the reference `Info` (`status`, `iter`, `to_host()`).
- `cupiqp_native.Status`, `Settings` and `PIQP_INF` mirror the reference.
- `cupiqp_native._testing`: per-module entry points (Ruiz, KKT assembly, cuDSS
  wrapper, IPM kernels, ...) operating on caller device arrays, used by
  module parity tests to compare with the reference's own Python/Warp
  components fed the same inputs. Each module owns one `bind_<module>.cpp`.

The binding is a convenience layer for testing and the harness. The native
library and the C++ benchmark do not link Python.

## 4. Lifecycle

| Stage | Work | Frequency |
|---|---|---|
| construct | stream (owned or borrowed), cuSPARSE handle, cuDSS handle/config/data, counting memory handler, threading layer | once |
| setup | validate patterns; upload patterns; allocate every device buffer (data, scaling, KKT, variables, residuals, info, work, loop state, pinned host mirrors); build KKT pattern and index maps on the host; create cuSPARSE descriptors and SpMV buffers; create cuDSS matrix/dense descriptors; copy initial values; finite masks; Ruiz scaling (if `preconditioner_iter > 0`); constraints RHS norm; scatter P/A/G into KKT; `cudssExecute(ANALYSIS)`; synchronize | once |
| update | `unscale_data` (only blocks that are not replaced; replaced blocks are overwritten, so skipping them is exact); copy new values; refresh finite masks if bounds changed; `reset` + fresh Ruiz (device early exit) when a matrix changed and `preconditioner_reuse_on_update == false`, else reuse scaling; refresh `x_b_scaling` if box bounds changed; RHS norm; scatter changed blocks into the KKT values, including masked G when bounds changed | per update |
| solve | upload device settings (pinned staging, only when changed); reset solve state; initial guess (factorize with retries, RHS, KKT solve, centering, mu); residuals; termination; IPM loop; terminate active; unscale solution | per solve |
| IPM iteration | prepare factor + predictor; update regularization; KKT diagonal update; FACTORIZATION; retry loop (conditional node); predictor solve; corrector prep; corrector solve; mark failed; step; update vars; mu; residuals (5 SpMV + fused kernel); rho/delta update; prox update; regularized residuals; advance; termination; continue flag | per iteration |
| destroy | graphs, cuDSS objects, cuSPARSE objects, buffers, owned stream (in that order) | once |

Symbolic analysis runs once, at setup. Numeric factorization runs every
iteration (`FACTORIZATION` phase as in the reference; `REFACTORIZATION` is
recorded as a possible future optimization and is not used). The KKT values
buffer, its pattern, the right-hand side and solution buffers keep stable
addresses. cuDSS descriptors point at them, and graphs capture them by address.

CUDA Graphs
- `IterationGraphHostLoop` (default): `_run_one_iteration` is captured once, on
  the first solve after setup, into `iteration_graph_`. That first capture is
  reported separately in benchmarks. Each solve then runs the prelude eagerly,
  followed by `while (loop_state_host[1]) { cudaGraphLaunch; memcpyAsync(loop_state -> pinned); cudaStreamSynchronize; }`.
  The factorization retry loop inside the iteration is a conditional WHILE node,
  matching `wp.capture_while` under capture.
- `None`: same sequence with plain launches. Both conditional loops (retry and
  IPM) are driven from the host, as Warp's `capture_while` does outside capture.
- `SolveGraphDeviceLoop` (experimental): the whole device solve, including the
  IPM loop as a WHILE node, is captured once. Pageable memcpy nodes produced by
  cuDSS SOLVE are re-pointed to pinned snapshots (FEASIBILITY 3.3), and a
  capture-time self-test compares two snapshots. One host wait per solve.
- `update()` is never captured in milestone 1. It is a fixed, short sequence of
  launches with no host sync; this can be revisited after measuring.
- Graphs are invalidated and recaptured only if settings that change the
  captured sequence change (none of the runtime settings do; they live in
  device memory, as in the reference `DeviceSettings`).

## 5. Module map (reference -> native)

| Module | Native files (owner) | Reference source | Intermediate quantities for parity |
|---|---|---|---|
| M0 core | `include/cupiqp/*.hpp`, `src/common/*` (integration owner) | `settings.py`, `typedef.py`, `results.py`, `utils.py` | settings upload, status codes |
| M1 data | `src/problem/data/*` | `data.py`, `data_kernels.py`, `sparse/sparse_data.py`, `sparse/batched_csr.py`, `sparse/csr_helpers.py`, `sparse_solver._check_csr` | stored values, finite masks, `active_G_row`, `active_x_bound`, `num_finite_bounds` |
| M2 spmv | `src/linalg/spmv/*` | `sparse/sparse_matvec.py` | `P x`, `A x`, `A^T y`, `G x`, `G^T z` |
| M3 ruiz | `src/problem/ruiz/*` | `preconditioner.py`, `preconditioner_kernels.py`, `sparse/sparse_preconditioner*.py` | `delta`, `delta_b`, `cost_scaling`, `x_b_scaling` (init aliases `active_x_bound`), scaled data, unscale factors, RHS norm, unscaled solution |
| M4 kkt_assembly | `src/linalg/kkt_assembly/*` | `sparse/sparse_kkt_solver.py`, `sparse_kkt_solver_kernels.py` | KKT pattern, slot maps, KKT values after scatter and after diagonal update |
| M5 cudss | `src/linalg/cudss/*` (incl. graph patch) | `sparse/sparse_direct_solver.py` | factor/solve on given values; allocation counts |
| M6 kkt_system | `src/linalg/kkt_system/*` | `kkt_systems.py`, `kkt_systems_kernels.py`, `kkt_solver.py` | `x_reg`, `z_reg`, `w_delta_inv`, condensed RHS, recovered step, solve status |
| M7 ipm_kernels | `src/ipm/kernels/*` | `solver_kernels.py` (forward kernels) | each kernel's outputs from identical inputs |
| M8 solver | `src/ipm/kernels/solver/*` | `solver.py`, `sparse/sparse_solver.py` | full iterate per iteration (`max_iter = k`) |
| M9 binding | `python/*` | public Python surface | (consumes M8) |
| M10 bench | `bench/*`, harness adapter | `benchmark/replay_mpc_updates.py` | timings, counters |

Each module also owns `tests/cpp/<module>/`, `tests/python/<module>/` and
`python/bindings/bind_<module>.cpp`.

Shared buffer conventions (defined in `src/common/views.hpp`; set by M0 and frozen
before the parallel work starts):
- Per-problem arrays are row-major `(B, k)`, with row stride `k` and `B = 1`.
- The `Variables` layout matches the reference: primal buffer `[x(n) | s_l(num_hl) | s_u(num_hu) | s_bl(num_xl) | s_bu(num_xu)]` and dual buffer `[y(p) | z_l | z_u | z_bl | z_bu]`, where `num_hl = m` if `h_l` is present else 0, and likewise for the other sides. `s_all`/`z_all` are the tails.
- `Info` is structure-of-arrays (`double* rho` etc. of length `B`; the integer
  counters likewise). The reference uses `(B, 23)` row-major columns. Parity
  tests compare values, not layout.
- Device settings: `floats[13]` and `ints[6]`, in `SettingsFloatIdx` and
  `SettingsIntIdx` order.
- Status codes are the reference `typedef.py` values.

## 6. Numerics and deviations policy
- Kernels reproduce the reference formulas term by term, including the order
  of operations inside each element-wise expression. Reductions use a fixed
  block-tree order; sums may differ from Warp tile reductions in the last bits.
- No change to stopping criteria, tolerances, regularization, retry logic, Ruiz
  limits, or bound handling.
- Deviations that do not change results (documented and tested):
  1. Ruiz early exit decided on the device (FEASIBILITY 3.5).
  2. `unscale_data` skipped for blocks that `update()` replaces.
  3. Fused copies (RHS assembly and disassembly) where the reference issues
     three `wp.copy` calls.
- Any numerical discrepancy found during parity work is recorded in
  `docs/DEVIATIONS.md`, with cause and magnitude.

## 7. Instrumentation
`Counters`: host syncs issued by the solver, graph launches, graph captures,
cuDSS device allocations and frees (through the counting memory handler),
eager kernel launches, and factorization retries. Optional CUDA-event phase
timers (`settings.compute_timings`) cover update, scaling, KKT assembly,
factorization, and KKT solve. The benchmark also wraps runs in `nsys` for
independent CUDA API counts.

## 8. Testing and acceptance
- Module tests (each module's deliverable):
  - C++ GoogleTest against analytic results or a CPU reference implemented in
    the test;
  - pytest parity through `cupiqp_native._testing`, comparing with the pinned
    reference component on the same inputs (reusing `tests/helpers.py`
    problem builders).
- Required coverage: CSR index and value ordering, explicit zeros that become
  nonzero, Hessian symmetry (full P storage), constraint signs and bound masks,
  infinite/finite bound switches, scaling and unscaling round trips, KKT slot
  mapping, status handling (solved, max-iter, numerical issues from a NaN input,
  infeasible), repeated updates with address stability and a single ANALYSIS
  call, graph-enabled vs graph-disabled solves on the same inputs.
- Integration: upstream `tests/solver/test_sparse_solver.py`, `test_easy_qps.py`
  and the Maros-Meszaros subset run through `cupiqp_native` via a conftest
  switch; MPC replay parity (status, iterations, objective, residuals, KKT check
  at 1e-6) against the reference on all 64 snapshots and 48 sampled
  transitions.
- Tolerances for parity with the reference: per-module outputs within 1e-12
  relative (1e-10 for quantities downstream of cuDSS); end to end, the same
  status, iterations within plus or minus 1 (exact match reported as a rate),
  objectives within 1e-8 relative, and the independent KKT check passing at 1e-6.

## 9. Future work hooks
- Batched (`B > 1`) via cuDSS uniform batching or the stacked block-diagonal
  system as in the reference; FP32 via kernel templates (all kernels are
  templated on `T`, with only `double` instantiated).
- JAX: an XLA FFI custom call (`xla::ffi`) over `SparseSolver` with `setup`
  outside JIT and `update+solve` as the custom call on XLA's stream. Not needed
  for the benchmark, which uses Warp arrays.
- `REFACTORIZATION` phase, user-graph capture of `update()` + `solve()`.

## 10. Implementation rules (all modules)

1. **Allocation.** Every device and pinned buffer is allocated in `setup()`.
   `update()`, `solve()` and IPM iterations perform no `cudaMalloc*`/`cudaFree*`
   and no host heap allocation on the hot path (no `std::vector` growth,
   no `std::function` construction inside loops; build closures once). cuDSS
   is the only component that may request device memory internally. Its
   requests go through the counting memory handler (`Counters`). If steady
   state shows any request, M5 serves them from a device arena preallocated at
   setup, sized from the warm-up peak, with a pool fallback that is counted
   and reported.
2. **CPU-GPU interaction.** Data stays on the GPU. Inside an iteration, the only
   host interaction is the reference's continue-flag read in
   `IterationGraphHostLoop` mode. Decisions (retries, termination, Ruiz early
   exit, step lengths) are taken on the device.
3. **Setup may be expensive.** It may build index maps on the host, allocate
   every buffer, create every descriptor and workspace, run the symbolic
   analysis, and capture graphs on first use.
4. **Memory safety.** Device memory is owned by `CudaBuffer<T>`, pinned memory by
   `CudaPinnedBuffer<T>`, and streams and events by `CudaStream`/`CudaEvent`.
   Library handles and descriptors get move-only RAII owners. No raw
   `cudaMalloc`/`cudaFree` outside `core/buffer.hpp`. Views hold non-owning
   pointers and never outlive their owner. Sizes are checked at API
   boundaries; kernels bounds-check their thread index.
5. **CUB.** Block and warp reductions use `cub::BlockReduce` (`core/reduce.cuh`).
   Device-wide scans, reductions and selects use CUB device algorithms, with
   temporary storage preallocated at setup.
6. **Kernels.** Version 1 mirrors the reference Warp kernel: same thread
   mapping, the same formulas in the same order, and the same masks. Parity is
   established on version 1. Each module then records, in its hand-off note,
   an assessment of (1) tiling (e.g. per-row tiles of the KKT/CSR loops),
   (2) tensor cores, and (3) shared-memory and register reuse (fusing
   consecutive element-wise kernels, keeping per-problem scalars in
   registers). An optimized version is merged only when it keeps parity and
   shows a measured gain. Tensor cores: this GPU (Ada, sm_89) has no FP64
   tensor-core path, so FP64 kernels cannot use them here. Record the
   opportunity for FP32 or for data-center GPUs (DMMA on sm_80/sm_90).
