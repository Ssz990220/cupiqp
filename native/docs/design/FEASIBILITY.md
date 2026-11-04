# Native CUDA C++ port of cuPIQP: feasibility assessment

Historical document (design phase, 2026-11-03). Since 2026-11-04 the native
library no longer uses cuDSS or cuSPARSE (REPORT.md section 14): the KKT
system is solved by the dense Schur-complement solver (n <= 2048) and the
SpMV by custom kernels, so the cuDSS findings below (graph patch, warm-up,
deterministic mode) no longer apply to the build. The PoC source
`native/poc/cudss_graph_poc.cu` was deleted with it and stays in git history
(commit `dcadf6a`); its log is kept in `native/poc/results/`.

Pinned reference: `PREDICT-EPFL/cupiqp` @ `d538dda8c93a76a16fcddc2f98c24c98133a7937`
(branch `native-cuda-port` starts there). Milestone 1 scope: sparse backend, FP64,
batch size 1, fixed sparsity, repeated numeric updates.

## 1. Environment (verified 2026-11-03, WSL2 Ubuntu 24.04)

| Item | Version | How verified |
|---|---|---|
| GPU | RTX 4070 SUPER 12 GB, driver 610.88 | `nvidia-smi` |
| CUDA toolkit | 13.2 (nvcc V13.2.78), runtime 13020 | `nvcc --version`, PoC prints `cudaRuntimeGetVersion` |
| cuDSS (native build) | 0.8.0.10, NVIDIA redist `libcudss-linux-x86_64-0.8.0.10_cuda13-archive.tar.xz`, sha256 `ba18f5fd...4937` (matches `redistrib_0.8.0.json`) | header `CUDSS_VERSION_*`, `cudssGetProperty` |
| cuDSS (reference) | `nvidia-cudss-cu13==0.8.0.10` wheel | benchmark venv `pip` metadata |
| Warp | 1.19.0 | `warp.__version__` |
| JAX | not installed in the benchmark venv | `pip list` |
| cuSPARSE | toolkit 13.2 (`/usr/local/cuda`) for native; wheel 12.8.6.72 for the reference | |
| nanobind | 3.1.0 (WSL venv `~/Projects/cupiqp-native/.venv`) | |

The native build uses the redist cuDSS under `~/Projects/.deps/cudss-0.8.0.10`
(outside Git). It has the same version and soname (`libcudss.so.0`) as the
reference wheel, so a Python process that loads both the reference and the
native binding ends up with a single copy of cuDSS 0.8.0.10.

## 2. What the reference does, by layer (source audit)

The sparse solve path is Python orchestration over three kinds of GPU work:

| Work | Implemented with | Port strategy |
|---|---|---|
| All element-wise and per-problem reduction kernels (IPM steps, residuals, termination, Ruiz norms/scaling, KKT diagonal/scatter, slack/dual elimination) | Warp kernels (`*_kernels.py`), JIT-compiled to CUDA | Rewrite as CUDA C++ kernels, one-to-one with the Warp kernels |
| Sparse LDL^T analysis / factorization / solve | cuDSS C API via `nvmath.bindings.cudss` (`sparse/sparse_direct_solver.py`) | Call the same C API with the same configuration |
| Sparse mat-vec (P x, A x, A^T y, G x, G^T z) | cuSPARSE `SpMV` via `nvmath.bindings.cusparse`, `CUSPARSE_SPMV_ALG_DEFAULT`, preallocated buffer (`sparse/sparse_matvec.py`) | Call the same cuSPARSE API |
| KKT pattern construction, index maps, CSR validation | NumPy/SciPy on the host, once at setup | Host C++ at setup |
| Stream, CUDA Graph capture, loop control | Warp (`wp.ScopedCapture`, `wp.capture_while`, `wp.Stream`) | CUDA runtime graph API directly |

JAX's role: `cupiqp/interfaces/jax/*` is an optional differentiable front end
(`jax.ffi`/callback style wrapper over the Warp solver). It is not on the solve
path and JAX is not installed in the benchmark environment. A future JAX binding
for the native library would be an XLA FFI custom call over the C++ API (see
DESIGN.md section 9); nothing in milestone 1 depends on it.

Nothing in the sparse solve path requires Python at run time beyond orchestration,
so a Python-free native solve path is feasible.

## 3. Feasibility issues and proofs of concept

PoC source: `native/poc/cudss_graph_poc.cu`; log: `native/poc/results/cudss_graph_poc.log`.

### 3.1 cuDSS native linkage and configuration — feasible

The PoC links `libcudss.so.0` (0.8.0.10) directly, applies the reference
configuration (`HYBRID_MEMORY_MODE=0`, `USE_CUDA_REGISTER_MEMORY=1`,
`REORDERING_ALG=DEFAULT`, `USE_SUPERPANELS=0`, `IR_N_STEPS=0`, threading layer
`libcudss_mtlayer_gomp.so.0`, device memory handler on the default memory pool
with `cudaMallocFromPoolAsync`/`cudaFreeAsync`), and factors and solves a
quasi-definite KKT (lower view, full P block stored) with residual 3e-12.
In cuDSS 0.8, CSR offsets and indices have separate `cudssDataType_t`
arguments (`CUDSS_R_32I` for both, as in the reference).

### 3.2 CUDA Graph capture of cuDSS phases — verified with the installed version

| Phase | Plain graph capture | Conditional WHILE-node body |
|---|---|---|
| FACTORIZATION | OK | OK |
| REFACTORIZATION | OK | OK |
| SOLVE (and every sub-phase: FWD_PERM, FWD, DIAG, BWD, BWD_PERM, REFINEMENT) | OK | **FAIL** at `cudaGraphInstantiate` (`cudaErrorInvalidValue`) |

Node census of a plain capture: FACTORIZATION = 4 kernels + 3 memsets;
SOLVE = 5 kernels + 4 memsets + **2 memcpy nodes whose source is pageable host
memory** + 2 other memcpys. Conditional bodies reject pageable-source memcpy
nodes, which confirms the reference's explanation (`sparse_direct_solver.py`,
`SparseKKTSolver.supports_conditional_capture == False`).

Consequence: like the reference, the default native mode is **one IPM iteration
per CUDA graph, relaunched from a host loop that reads the device continue flag
once per iteration** (`ITERATION_GRAPH_WITH_HOST_LOOP`). The factorization retry
loop is a conditional WHILE node inside that iteration graph, which is legal
because FACTORIZATION has no pageable copies.

### 3.3 Workaround candidate for a whole-solve graph — feasible but relies on cuDSS internals

A native port can edit a captured graph before instantiation. The PoC captures
SOLVE into a WHILE body, replaces each pageable-source memcpy node by a copy from
a pinned snapshot of the same bytes (`cudaGraphMemcpyNodeSetParams`), and
instantiates successfully. Three body executions per launch give a solution
bitwise equal to an eager solve, before and after an in-place value update plus
refactorization. Both copied payloads are 8 bytes with identical content
(`int64 = 2143188680704`, which looks like an address inside cuDSS's persistent
`cudssData` state) and did not change across the value update.

Risk: the bytes are undocumented cuDSS internals. They could depend on state
that changes when ANALYSIS reruns or with another cuDSS version. Decision:

- default graph mode = reference behavior (iteration graph + host loop);
- the whole-solve device-loop graph (`GraphMode::SolveGraphDeviceLoop`) is an
  **opt-in experimental mode**, guarded by:
  - capture only after ANALYSIS;
  - a recorded cuDSS version check (exactly 0.8.0);
  - a self-test at capture time (snapshot bytes taken from two independent
    captures must agree);
  - parity tests against the default mode on the same inputs.

### 3.4 Steady-state allocations — measurable natively

The reference installs a stream-ordered allocator into cuDSS. The native port
installs its own `cudssDeviceMemHandler_t` whose callbacks count calls, so
allocations made by cuDSS in the hot path are counted directly. Everything else
is allocated at setup. The benchmark will report both these counters and
`nsys` CUDA API counts (`cudaMalloc*`, `cudaFree*`, `cuMemAlloc*`).

### 3.5 Host synchronizations in `update()` — removable without numerical change

Reference `RuizEquilibration.scale_data` reads its convergence measure on the
host after every Ruiz pass (`wp.synchronize_stream` + `.numpy()`), 10 syncs per
update with `preconditioner_iter=10`. That accounts for 10 of the 36
`cuStreamSynchronize` calls in `results/fixed-pattern-runtime-20261102/summary.json`;
the other 26 are the 24 per-iteration continue-flag reads plus 2. The native
port keeps the same early-exit rule but decides it on the device: a device
`ruiz_done` flag, written by the convergence kernel, makes every later pass a
no-op. Passes run and skipped are exactly those of the reference, so values are
identical up to kernel floating-point order. No host sync is needed.

### 3.6 Floating-point operation order — expected small differences

Warp tile reductions (`wp.tile_reduce`, `launch_tiled` with `reduction_block_dim`)
are replaced by CUDA block reductions with a fixed tree shape. Sums and dot
products (objective terms, `mu`, residual norms built from sums) can therefore
differ in the last bits; max-norms are order-independent. cuDSS and cuSPARSE
calls are identical. End-to-end parity is defined on status, iteration count
(exact match expected in most cases, plus or minus 1 tolerated and reported),
objective, and residuals (see DESIGN.md section 8).

### 3.7 Findings during module implementation (cuDSS 0.8.0.10)

- **Data object bound to its ANALYSIS stream.** After ANALYSIS on stream X, every
  later FACTORIZATION/SOLVE on that `cudssData` executes on X regardless of
  `cudssSetStream`. Work issued while capturing another stream produces no
  cuDSS nodes there. The PoC in 3.2/3.3 captured on the analysis stream itself,
  so its results stand. The `cudss` module owns an execution stream for ANALYSIS
  and forks/joins every phase onto the caller's stream with events, so cuDSS
  nodes land in the caller's graph or conditional body (test
  `WorkFollowsTheCallerStream`).
- **`DETERMINISTIC_MODE` crashes intermittently.** SOLVE fails with
  `cudaErrorIllegalAddress` (out-of-bounds shared-memory read in
  `cudss::fwd_dtmn_ker`, per compute-sanitizer). This happened in 4 of 6 gtest
  runs; the reference `use_deterministic_mode=True` also failed in 1 of 3. The
  native solver rejects `use_deterministic_mode_for_cudss = true`.
- **Run-to-run reproducibility.** Default-mode factor+solve is not bitwise
  reproducible (spread up to 6e-13 on test systems), nor are cuSPARSE
  transposed SpMV products (atomics, ~3e-14). Both hold for the reference too,
  so end-to-end parity uses tolerances, and iteration counts may differ by
  one between any two runs (reference vs reference included).
- **Steady-state allocations: none.** After ANALYSIS (8 allocs) and the first
  FACTORIZATION (1 alloc, 1.53 MB), cuDSS requests no memory, eager or in
  graphs. It requests 0 bytes once per eager SOLVE, which the handler answers
  with null without touching the pool. No arena is needed.

### 3.8 cuDSS REFACTORIZATION (evaluated 2026-11-03)

`CUDSS_PHASE_REFACTORIZATION` (numeric re-factorization reusing the last
FACTORIZATION) was measured on the real KKT patterns with
`cupiqp_native_module_bench` (graph replay, 200 reps):

| KKT | FACTORIZATION | REFACTORIZATION |
|---|---|---|
| MPC (dim 2754) | 0.810 ms | 0.809 ms |
| Trajopt (group 0) | 0.716 ms | 0.715 ms |

Accuracy is not the issue. The probe factors at iterate A, writes the KKT
values of iterate B and refactorizes, then compares with a full factorization
at B. The relative difference of the SOLVE result is at most 9e-14, even with
slacks and duals spread over +-8 decades.

There is no speed difference, so it is not wired into the solver: with a
symmetric matrix, DEFAULT reordering and no hybrid memory, cuDSS 0.8 does the
same numeric work in both phases (reordering and symbolic factorization
already happen once, in ANALYSIS). Re-measure with the module benchmark after
a cuDSS upgrade or a change of reordering, pivoting or hybrid settings.

## 4. Conclusion

A standalone C++/CUDA sparse FP64 solver with no Python, Warp, or JAX run-time
dependency is feasible. cuDSS and cuSPARSE are reused unchanged, and 54 forward-path Warp kernel factories
(counted in the `*_kernels.py` files on the sparse path, excluding gradient,
backward and smoothing kernels; some are shared helpers or unused for B = 1
without iterative refinement) are ported to CUDA C++. The CUDA Graph limitation on the cuDSS solve phase is
reproduced with the installed version. The port keeps the reference's
iteration-graph strategy by default and offers an opt-in whole-solve graph
whose workaround is demonstrated above.
