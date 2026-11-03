# M5 cudss: hand-off note

cuDSS sparse LDL^T wrapper (`CudssSolver`), counting device-memory handler
(`CudssMemory`), and the pageable-memcpy graph patch (`patch_pageable_memcpy`).
Reference: `cupiqp/sparse/sparse_direct_solver.py` (`_CudssSolver`,
`CudssStackedSolver` with B = 1). cuDSS 0.8.0.10, CUDA 13.2, RTX 4070 SUPER
(sm_89, WSL2), GPU shared with other agents during all measurements.

## Files

| File | Contents |
|---|---|
| `src/linalg/cudss/cudss_solver.hpp` | contract header; private members and three accessors added (below) |
| `src/linalg/cudss/cudss_solver.cpp` | `CudssSolver` |
| `src/linalg/cudss/cudss_memory.hpp/.cpp` | `CudssMemory`: device memory handler on the default pool, counting |
| `src/linalg/cudss/pageable_memcpy_patch.cpp` | `patch_pageable_memcpy` |
| `src/linalg/cudss/sources.cmake` | the three sources |
| `tests/cpp/linalg/cudss/test_cudss.cu` | GoogleTest suite (9 tests, 1 disabled) |
| `tests/cpp/linalg/cudss/kkt_fixture.hpp` | random quasi-definite KKT in the reference layout + dense CPU LU solve |
| `tests/cpp/linalg/cudss/bench_cudss.cu` | micro-benchmark executable (not a test) |
| `tests/cpp/linalg/cudss/CMakeLists.txt` | test + benchmark targets |
| `tests/python/linalg/cudss/test_cudss_parity.py` | pytest parity vs the reference `CudssStackedSolver` |
| `python/bindings/bind_cudss.cpp` | `_testing.CudssSolver` |

## Public API as implemented

Contract signatures unchanged. Added public accessors: `dim()`, `nnz()`,
`memory()` (const `CudssMemory&`: `zero_byte_requests()`, `zero_byte_frees()`,
`live_bytes()`, `peak_live_bytes()`, `largest_request()`). Added private members:
`local_counters_` (used when `counters == nullptr`), `memory_`, `dim_`, `nnz_`,
`exec_stream_`, `fork_`, `join_`, and `execute()`.

- Configuration identical to the reference: SYMMETRIC, LOWER view, base 0,
  int32 offsets/indices, FP64; HYBRID_MEMORY_MODE 0, USE_CUDA_REGISTER_MEMORY 1,
  REORDERING_ALG DEFAULT, USE_SUPERPANELS 0, IR_N_STEPS 0, DETERMINISTIC_MODE 1
  when requested. Threading layer: `$CUDSS_THREADING_LIB` if it is an existing
  file, else the build-time `CUPIQP_CUDSS_MT_LIB`, else none (the reference skips
  a missing file the same way). The reference prefers its auto-discovered path
  over the env var; here the env var wins, as the brief asks.
- The memory handler is installed before `cudssDataCreate` (the reference
  installs it after; nothing is allocated in between, so no difference).
- `factor()` does not read the factorization status (`cudssDataGet` blocks
  and breaks capture), same as the reference.
- Counters: `cudss_analysis_calls`, `cudss_factor_calls`, `cudss_solve_calls`
  count host-side calls (a capture counts once, replays do not);
  `cudss_device_allocs/frees` count nonzero-size requests (see below).

### Stream handling: cuDSS binds a data object to its ANALYSIS stream (important)

Measured with cuDSS 0.8.0.10 using the raw C API (no wrapper code involved):
after `ANALYSIS` runs with stream X, every later `FACTORIZATION` and `SOLVE` on
that `cudssData` runs on X, whatever `cudssSetStream` says. Capturing stream Y
with `cudssSetStream(h, Y)` gives a graph with 0 nodes (the work runs eagerly on
X, or lands in X's capture if X is capturing). Setting the handle stream before
`cudssDataCreate` does not change this. Consequences without a fix:

- a conditional body captured on a helper stream (exactly what
  `common/capture_while` does) gets no cuDSS nodes; the factorization runs once
  at capture time or lands in the outer graph at the wrong position. Silent.
- eager calls on a stream other than the analysis stream are not ordered with
  that stream's work.

Fix in this module: `CudssSolver` owns a non-blocking `exec_stream_`, binds the
handle to it from `init()` on (so ANALYSIS binds the data to it), and every
phase does `cudaEventRecord(fork, s)`, `cudaStreamWaitEvent(exec, fork)`,
`cudssExecute`, `cudaEventRecord(join, exec)`, `cudaStreamWaitEvent(s, join)`.
Under capture this pulls `exec_stream_` into the caller's capture (plain graph
or conditional body), adding edges only; eagerly it orders the phase between
the caller stream's earlier and later work. `cudssSetStream(h, exec)` is still
called on every call, as the brief asks. Test `WorkFollowsTheCallerStream`
covers it (eager ordering on another stream; capture of another stream has the
7 + 13 cuDSS nodes). The reference passes the Warp stream every call and is
affected only if it ever uses a stream other than its plan stream.

The cost is two event records and two waits per eager call (see timings: eager
SOLVE about 8 us slower than the reference's eager SOLVE; nothing measurable
inside graphs).

## Allocation findings (DESIGN 10.1)

`AllocationCountsAfterWarmup` (dim 350, nnz 3370), requests seen by the handler:

| Stage | nonzero allocs | frees | zero-byte requests |
|---|---|---|---|
| init | 0 | 0 | 0 |
| ANALYSIS | 8 | 3 | 1 |
| first FACTORIZATION | 1 (1.53 MB) | 0 | 0 |
| first SOLVE | 0 | 0 | 1 |
| 10 eager FACTORIZATION | 0 | 0 | 0 |
| 10 eager SOLVE | 0 | 0 | 10 |
| capture FACTORIZATION + SOLVE | 0 | 0 | 1 |
| 10 graph replays | 0 | 0 | 0 |

Captured graph: 20 nodes, no memory alloc/free nodes. The benchmark (MPC shape)
reports 0 nonzero allocs/frees during all timed loops (eager and graph); peak
cuDSS device memory 17.5 MiB, largest single request 12.9 MiB.

cuDSS asks for a 0-byte buffer once per eager SOLVE call and once per capture of
SOLVE, and never frees it. These are answered with `nullptr` without calling the
pool (which would also return null) and counted in
`memory().zero_byte_requests()`, not in `Counters::cudss_device_allocs`, since
no memory is allocated. Conclusion: there are no real steady-state allocations,
so no arena is installed. (A first-fit arena was written and then removed as
unneeded; if another cuDSS version allocates in steady state, the counters will
show it.)

## Conditional bodies and `patch_pageable_memcpy`

`ConditionalBodiesAndPageablePatch` checks, for both `common/capture_while`
(non-blocking helper stream) and an explicit WHILE node whose body is captured
on a blocking stream (PoC style):

- FACTORIZATION in the body: instantiates without a patch; the body holds the
  cuDSS kernels; replays factor correctly.
- SOLVE in the body without the patch: `cudaGraphInstantiate` fails
  (`cudaErrorInvalidValue`), as in FEASIBILITY 3.2.
- SOLVE in the body with the patch applied to the top-level graph (recursion
  into conditional bodies via `cudaGraphNodeGetParams`, and into child graphs):
  exactly 2 nodes patched, instantiation succeeds, replays match eager solves,
  including replays after in-place value updates + eager refactorization
  between replays; the snapshot bytes of two independent captures are equal.

Note: before the stream fix above, the `capture_while` path appeared to
instantiate unpatched SOLVE bodies; that was because the cuDSS work never went
into the body.

The patch requires 1-D copies (asserts otherwise), copies the bytes into a
`CudaPinnedBuffer<unsigned char>` appended to `keep`, and leaves the copy kind
unchanged. The snapshot is only correct while the pageable bytes are constant
after ANALYSIS (true for cuDSS 0.8.0.10 in these tests; the version check and
capture-time self-test of FEASIBILITY 3.3 belong to the solver's experimental
graph mode).

compute-sanitizer (memcheck) on this machine cannot run graphs with conditional
nodes (`cudaErrorUnknown`; the common module's own `CaptureWhile` capture tests
fail the same way under it), so memcheck was run on all other tests:
0 errors.

## Determinism

- Default mode: cuDSS 0.8.0.10 is not bitwise reproducible run to run, neither
  FACTORIZATION + SOLVE (0 of 4 repeats bitwise equal, max rel spread 6.2e-13)
  nor SOLVE alone on one factorization (0 of 4, spread 4.7e-13), on a
  dim-350 system. The reference shows the same spread (pytest "ref vs ref").
  So "capture/replay bitwise equal to eager" cannot hold in the default mode;
  tests compare graph vs eager at 1e-11 relative (measured 4e-14 to 1.4e-13).
- DETERMINISTIC_MODE: when it runs, eager repeats and graph replays are bitwise
  equal to the first eager solve. But SOLVE intermittently faults
  (`cudaErrorIllegalAddress` in 4 of 6 runs of the disabled gtest, 5 of 8
  runs of a standalone program): compute-sanitizer reports an
  out-of-bounds `__shared__` read in `cudss::fwd_dtmn_ker` (inside cuDSS). It
  happens with and without this module's handler, with and without the
  threading layer and each of the non-default config values removed, and the
  reference Python `CudssStackedSolver(use_deterministic_mode=True)` hits
  `illegal memory access` too (1 of 3 runs). The test is therefore
  `DISABLED_DeterministicModeRepeatsAndCapturesBitwise`.

## Tests

Build and C++ tests:

```bash
cd ~/Projects/cupiqp-native-wt/cudss/native
cmake -S . -B ../build/native -DPython_EXECUTABLE=$HOME/Projects/cupiqp-native/.venv/bin/python
cmake --build ../build/native -j 8
(cd ../build/native && ctest --output-on-failure -R Cudss)
```

Result: 9 tests, 9 passed, 1 not run (disabled deterministic-mode test).
`compute-sanitizer ./tests/cpp/linalg/cudss/test_cudss --gtest_filter='-*Conditional*'`:
8 passed, 0 errors.

Python parity:

```bash
CUPIQP_NATIVE_BUILD=$HOME/Projects/cupiqp-native-wt/cudss/build/native \
  WARP_CACHE_PATH=$HOME/.cache/warp-cupiqp-native \
  $HOME/Projects/cupiqp-native/.venv/bin/python -m pytest -q -s \
  $HOME/Projects/cupiqp-native-wt/cudss/native/tests/python/linalg/cudss
```

Result: 6 passed.

C++ accuracy vs the dense CPU LU solve (random quasi-definite KKT, reference
layout): relative error 3e-16 to 3.4e-13, residual 3e-16 to 9.6e-12; one
ANALYSIS reused across value updates and explicit zeros that become nonzero;
garbage in the stored upper part of P does not change the solution beyond the
run-to-run spread.

## Parity vs the reference (pytest, max abs diff / max abs ref)

| Case | native vs ref | ref vs ref (2 runs) | native vs host solve |
|---|---|---|---|
| P only (n 30, nnz 256) | 9.1e-17 | 0 | 2.4e-16 |
| P, A (40, 10; nnz 654) | 8.8e-13 | 4.4e-13 | 8.2e-13 |
| P, G (40, 60; nnz 1104) | 2.6e-16 | 2.3e-16 | 1.4e-15 |
| P, A, G (80, 20, 150; nnz 6316) | 3.4e-12 | 3.4e-12 | 9.6e-13 |
| 5 in-place updates, eager and graph (60, 15, 100) | worst 1.7e-11 (delta 2e-5) | worst 2.0e-11 | worst 1.0e-11 |
| MPC shape (dim 2754, nnz 261569), eager / graph | 5.0e-14 / 7.5e-14 | 4.4e-14 | 5.8e-14 |

Never bitwise equal (default mode). Native vs reference differences match the
reference's own run-to-run spread; tolerance used: 1e-10 (DESIGN 8, downstream
of cuDSS). The brief expected about 1e-14; that holds for the MPC shape and the
well-conditioned cases, not for the delta = 1e-4 .. 2e-5 cases.

## Timings (MPC KKT shape)

Same matrix for both (random pattern with the MPC block sizes: P 8263 stored
entries full storage, A 64 x 169 with 920, G 2521 x 169 with 249801; dim 2754,
KKT nnz 261569), written to a file by the reference script and loaded by
`bench_cudss 50 7 <file>`. 7 repetitions x 50 back-to-back calls, CUDA-event
stream time per call; median of the per-run medians over 3 alternating
native/reference rounds (us):

| | native | reference (Warp/nvmath) |
|---|---|---|
| eager FACTORIZATION | 828 (828, 828, 897) | 839 (839, 839, 839) |
| graph FACTORIZATION | 820 | n/a |
| eager SOLVE | 135 | 127 |
| graph SOLVE | 124 | n/a |
| eager factor + solve | 963 | 969 |
| graph factor + solve | 945 | 951 |

Host enqueue time per call (median of rounds): native eager factor 20 us,
graph factor 7.7 us, eager solve 37.6 us, graph solve 10.7 us, eager
factor + solve 59 us, graph factor + solve 17.5 us (host noise from the shared
machine is large here). ANALYSIS: 42 to 55 ms host wall. Native and reference
are the same library doing the same work, so the GPU times agree within a few
percent; the eager SOLVE gap (~8 us) is the fork/join described above.

Without the file argument `bench_cudss` uses its own random MPC-shaped KKT.

## Deviations from the reference

1. Fork/join onto a solver-owned stream (correctness fix for the stream
   binding; no numerical effect).
2. Zero-byte cuDSS requests answered with null without calling the pool (the
   pool returns null for size 0 as well).
3. Threading-layer path precedence: env var first (brief).

## Optimization assessment (DESIGN 10.6)

No kernels in this module; all GPU work is inside cuDSS. Tiling, tensor cores
and shared-memory reuse do not apply. Possible later options: REFACTORIZATION
instead of FACTORIZATION (DESIGN 4 lists it as future work); skipping the
eager fork/join when the caller stream equals the analysis stream (only if the
solver then guarantees that stream outlives the cuDSS data).

## Requested changes to shared code

- None required. Suggestion for the integration owner: `Counters` could get a
  `cudss_zero_byte_requests` field if the solver wants to report it; for now it
  is available from `CudssSolver::memory()`.
- Whoever captures cuDSS calls (M6/M8) must call `factor/solve` with the stream
  the body is captured on (they do, through `capture_while`); with the fork/join
  in this module that is enough. They must not call cuDSS directly.

## Known gaps

- DETERMINISTIC_MODE is unusable with cuDSS 0.8.0.10 (intermittent fault in
  cuDSS SOLVE); its test is disabled.
- Bitwise capture-vs-eager equality is not testable in default mode (cuDSS is
  not reproducible); tests use 1e-11 relative.
- Conditional-node tests cannot run under compute-sanitizer on this machine.
- The benchmark matrix is random with MPC block sizes, not a real MPC KKT.
