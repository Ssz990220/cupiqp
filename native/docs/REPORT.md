# cuPIQP native port: milestone 1 report (2026-11-03)

Native CUDA C++ port of the cuPIQP sparse solver (FP64, batch size 1, fixed
sparsity, repeated numeric updates). Numerical reference: Python/Warp cuPIQP at
`d538dda`. The native solve path has no Python, Warp or JAX dependency.

## 1. Deliverables

| Item | Location |
|---|---|
| Fork | https://github.com/Ssz990220/cupiqp (parent `PREDICT-EPFL/cupiqp`) |
| Branch | `native-cuda-port`, branched from `d538dda` |
| WSL checkout | `~/Projects/cupiqp-native` (`origin` = fork, `upstream` = PREDICT-EPFL) |
| Feasibility, design, dependency graph | `native/docs/design/{FEASIBILITY,DESIGN,DEPENDENCY_GRAPH}.md` (committed before implementation) |
| Deviations, unsupported features, open issues | `native/docs/design/DEVIATIONS.md` |
| Module hand-off notes | `native/docs/modules/*.md` |
| C++ API | `native/include/cupiqp/cupiqp.hpp` |
| Python binding | `native/python/` (nanobind module `cupiqp_native._ext` + `cupiqp_native.SparseSolver`) |
| Benchmarks and raw results | `native/bench/`, `native/bench/results/` |

The seven modules were developed in parallel on `task/<module>` branches in
worktrees under `~/Projects/cupiqp-native-wt/`. They were merged in dependency
order, and every test was rerun after each merge.

## 2. Tests

| Suite | Result |
|---|---|
| C++ (GoogleTest, all modules + solver integration) | 114 / 114 passed (2 disabled by design: cuDSS deterministic mode, a timing test); 141 / 141 with the later additions (section 11) |
| pytest parity vs the reference components | 1,075 / 1,075 passed |
| Upstream `tests/solver/test_sparse_solver.py` on the native solver (plugin `-p upstream_native`) | 7 passed, 3 skipped (batched, float32: outside milestone 1) |
| Upstream Maros-Meszaros (138 problems, iterative refinement on) | Same failure set as the reference up to run-to-run variation; see DEVIATIONS.md 3 |

Module parity with the reference:
- `data`, `kkt_assembly`, `ruiz`: bitwise identical.
- `ipm_kernels`: integers, element-wise results and min/max are bitwise
  identical; block sums are within 6 ulp.
- `kkt_system`: kernels bitwise identical.
- `spmv`: bitwise for op = N with beta = 0; transposed products within
  1e-14 (cuSPARSE atomics).
- `cudss`: within cuDSS run-to-run spread.

## 3. MPC replay (64 snapshots, one stored CSR pattern, 48 sampled transitions)

Protocol: one persistent solver, explicit zeros kept, every value updated in
stored order, FP64, `eps_abs = eps_duality_gap_abs = 1e-8`, zero relative
tolerances, 10 Ruiz passes recomputed after matrix updates, numerical iterate
reset each solve. 20 repeats, two modes, two excluded warm-up passes. GPU
exclusive, all runs in the same WSL session (2026-11-03). Host wall time on
`CLOCK_MONOTONIC_RAW` (see section 6).

Correctness. Every path: 1,920 / 1,920 solves SOLVED and passing the
independent KKT check at 1e-6. Iterations p50 24.5, max 28 (same as the
reference). 640 / 640 seeds valid.

Latency, median / P95 in ms, `gpu_resident` mode (`host_values` mode is
within 0.2 ms of these):

| Path | update | solve | x download | input ready -> CPU x |
|---|---|---|---|---|
| Reference, harness protocol | 3.18 / 3.40 | 36.99 / 40.50 | 0.05 / 0.06 | **40.29 / 43.77** |
| Native, harness protocol (Python binding) | 0.92 / 0.97 | 37.21 / 40.79 | 0.04 / 0.06 | **38.18 / 41.78** |
| Native C++ (no Python) | 0.91 / 0.95 | 37.44 / 40.75 | 0.02 / 0.03 | 38.37 / 41.70 |
| Native C++, whole-solve graph (opt-in) | 0.90 | 32.90 | | 33.88 / 37.14 |
| Native C++, custom SpMV (opt-in) | 0.90 | 34.90 | | 35.84 / 39.18 |
| Native C++, both options | 0.92 | 31.69 | | **32.66 / 35.73** |

One-time costs, reported separately (ms):

| Cost | Reference | Native |
|---|---|---|
| Setup | 353 (1,031 in the session's first run) | 58 |
| First solve (JIT, first graph capture) | 84 (608 in the session's first run) | 52 |

The reference numbers reproduce the earlier baseline report (40.5 ms,
update 3.2 ms, solve 37.1 ms).

## 4. Trajopt replay (30 snapshots, six fixed patterns, one solver per pattern)

Correctness. Every path: 960 / 960 SOLVED and KKT@1e-6. Iterations p50 30,
max 36. 240 / 240 seeds valid.

| Path (gpu_resident, median / P95 ms) | update | solve | input ready -> CPU x |
|---|---|---|---|
| Reference, harness protocol | 4.05 / 4.37 | 46.34 / 53.92 | 50.46 / 58.26 |
| Native, harness protocol | 0.63 / 0.70 | 46.27 / 54.06 | 47.01 / 54.76 |
| Native C++ | 0.62 / 0.69 | 45.88 / 53.71 | 46.56 / 54.38 |
| Native C++, both options | 0.63 / 0.79 | 39.65 / 46.36 | 40.32 / 47.08 |

## 5. Where the time goes (module micro-benchmarks, MPC, graph replay)

| Piece | Time | Per solve |
|---|---|---|
| cuDSS FACTORIZATION | 0.81 ms | ~26 calls, ~21 ms |
| Condensed KKT solve (elimination + cuDSS SOLVE + recovery) | 0.19 ms | ~51 calls, ~10 ms |
| Host round trip per iteration (sync + continue flag + graph launch) | ~0.18 ms | ~25, ~4.5 ms |
| Ruiz 10 passes (update) | 0.74 ms graph / 0.92 ms eager | 1 |
| G^T z, cuSPARSE vs custom | 50 us vs 8 us | 2 per iteration |

Interpretation:
- About 80% of a solve is cuDSS, which both implementations call
  identically, so the default native solve matches the reference's speed.
- The default-mode gain comes from `update()`: 3.5x on MPC and 6.4x on
  Trajopt. The Ruiz convergence check runs on the device, which removes the
  10 host syncs per update.
- The opt-in modes remove the remaining overheads: the per-iteration host
  round trip (whole-solve graph) and the atomic cuSPARSE transposed product
  (custom SpMV). Together they give about 19% (MPC) and 20% (Trajopt)
  end-to-end.

Nsight Systems, one warmed update + solve (`native/bench/results/mpc-20261103/profile`):

| Configuration | Graph launches | Stream syncs | Alloc / free calls |
|---|---|---|---|
| Reference | 25 | 37 | 0 |
| Native default | 25 | 27 | 0 |
| Native whole-solve graph | 1 | 0 | 0 |

Python binding overhead: the harness and the C++ end-to-end medians differ by
at most 0.2 ms, within noise.

## 6. Measurement note: WSL clock

On 2026-11-03 the NTP-disciplined `CLOCK_MONOTONIC` on this WSL host
(`time.perf_counter`, `std::chrono::steady_clock`) ran about 6% slow against
the Windows QPC clock. `CLOCK_MONOTONIC_RAW` and CUDA events agreed with
Windows. `native/bench/tools/check_clocks.py` shows the comparison. Both
benchmark paths now time on `CLOCK_MONOTONIC_RAW`.

The first run's host times are about 6% low, but its CUDA event times are
valid. It is kept in `results/mpc-20261103-run1-ntp-slewed-wallclock`. In the
main run, wall time over GPU event time is 1.0005, as in the earlier baseline
(1.0006).

## 7. Build and run

```bash
# build (WSL, CUDA 13.2, cuDSS 0.8.0.10 redist in ~/Projects/.deps/cudss-0.8.0.10)
cd ~/Projects/cupiqp-native/native
cmake -S . -B ../build/native -DPython_EXECUTABLE=$HOME/Projects/cupiqp-native/.venv/bin/python
cmake --build ../build/native -j
cd ~/Projects/cupiqp-native
export WARP_CACHE_PATH=$HOME/.cache/warp-cupiqp-native

# tests
(cd build/native && ctest --output-on-failure)
CUPIQP_NATIVE_BUILD=$PWD/build/native .venv/bin/python -m pytest -q native/tests/python
PYTHONPATH=native/tests/python CUPIQP_NATIVE_BUILD=$PWD/build/native \
  .venv/bin/python -m pytest -p upstream_native -q tests/solver/test_sparse_solver.py tests/solver/test_maros_meszaros.py

# snapshot export (once; data stays out of Git)
B=/mnt/c/Users/91354/Projects/gpu-qp-benchmark
.venv/bin/python -I native/bench/tools/export_snapshots.py --bundle $B/qp_benchmark_v2_20261102 --cohort main --out native/bench/data/mpc
.venv/bin/python -I native/bench/tools/export_snapshots.py --bundle $B/datasets/trajopt-20261102 --cohort main --out native/bench/data/trajopt

# path 1: native test suite (C++ only)
build/native/bench/cupiqp_native_replay --data native/bench/data/mpc --out <dir> --repeats 20 [--graph-mode device] [--custom-spmv]
.venv/bin/python -I native/bench/tools/check_solutions.py --results <dir> --bundle $B/qp_benchmark_v2_20261102
build/native/bench/cupiqp_native_module_bench --data native/bench/data/mpc/group_0.bin --reps 200 --out <file>.json

# path 2: existing harness protocol, reference vs native backend
.venv/bin/python native/bench/harness/replay_updates.py --backend reference --dataset mpc --repeats 20 --output <dir>
.venv/bin/python native/bench/harness/replay_updates.py --backend native    --dataset mpc --repeats 20 --output <dir>
#   (--dataset trajopt for Trajopt; native: --graph-mode device|none, --custom-spmv)

# profiles
bash native/bench/tools/profile_all.sh <dir> mpc
```

## 8. Deviations, unsupported features, open issues

See `native/docs/design/DEVIATIONS.md`. In short:
- All intentional changes are numerically neutral and verified by tests.
- The remaining differences come from summation order (at most 6 ulp) and
  from cuDSS/cuSPARSE run-to-run nondeterminism, which the reference shares.
- Not supported in milestone 1: batched, FP32, dense, multistage, gradients,
  and cuDSS deterministic mode (crashes in cuDSS 0.8.0.10).
- Open: the whole-solve graph mode relies on undocumented cuDSS internals
  (opt-in, self-tested at capture). Conditional-node graphs cannot run under
  compute-sanitizer on WSL.

Possible next steps:
- cuDSS `REFACTORIZATION` was evaluated: no speedup with this configuration
  (FEASIBILITY.md 3.8).
- Batching via cuDSS uniform batches.
- FP32 after FP64 parity.
- A JAX FFI binding (design section 9).

## 9. Staged (preallocated) inputs (added 2026-11-03)

`input_buffers()` / `host_input_buffers()` expose solver-owned device and
pinned host input buffers, allocated at setup at fixed addresses.
`update_staged(blocks, from_host)` replays the whole update as one CUDA graph,
including the host-to-device copy for pinned host inputs. The default
iteration-graph mode now also runs the solve prelude and finalization as
graphs, which removes one host sync per solve.

Same-session comparison (`native/bench/results/staged-20261103`, 20 repeats,
every configuration 1,920/1,920 (MPC) and 960/960 (Trajopt) KKT@1e-6,
iterations unchanged). Update, median ms:

| | MPC gpu_resident | MPC host_values (incl. H2D) | Trajopt gpu_resident | Trajopt host_values (incl. H2D) |
|---|---|---|---|---|
| Reference | 3.18 | 3.20 | 4.25 | 4.26 |
| Native `update()` | 0.90 | 0.97 | 0.63 | 0.67 |
| Native `update_staged()` | 0.75 | 0.87 | 0.44 | 0.53 |

End to end, median / P95 ms, gpu_resident:

| | MPC | Trajopt |
|---|---|---|
| Reference | 40.21 / 43.98 | 55.39 / 65.89 |
| Native default | 37.92 / 41.24 | 48.93 / 58.03 |
| Native staged | 38.00 / 43.10 | 49.80 / 59.03 |
| Native staged + whole-solve graph + custom SpMV | 32.56 / 37.12 | 42.45 / 49.85 |

The staged update saves 0.15-0.19 ms per update. That is smaller than the
run-to-run spread of the solve phase, so it does not show in the end-to-end
medians.

Trajopt ran slower overall in this session than in section 4 (the reference
went from 50.5 to 55.4 ms). Compare configurations within one session only;
the harness records GPU telemetry for each run.

## 10. Multi-block reductions for a single QP (added 2026-11-03)

With B = 1 the "one block per problem" reduction kernels (ported from the
reference's `wp.launch_tiled(dim=[B])`) ran on one SM. They now split one
problem over several blocks with a deterministic last-block combine
(`native/src/common/multiblock.cuh`, design and data in
`native/docs/modules/multiblock_reductions.md`). With large B they keep one
block per problem.

| Native C++ replay, default mode, 20 repeats, gpu_resident | MPC before | MPC after | Trajopt before | Trajopt after |
|---|---|---|---|---|
| IPM kernels per iteration (us) | 125.2 | 42.1 | 248.1 | 43.7 |
| End-to-end wall p50 (ms) | 37.91 | 35.79 | 46.53 | 39.96 |
| End-to-end wall p95 (ms) | 41.25 | 38.96 | 54.16 | 46.25 |
| KKT@1e-6 | 960/960 | 960/960 | 480/480 | 480/480 |

Iteration statistics are unchanged. Min/max outputs, integer outputs, sigma
and the corrector rhs remain bitwise equal to the reference; sums differ by
order only (primal_obj up to 11 ulp).

## 11. Reduced (Schur-complement) KKT backend (added 2026-11-03)

Opt-in `Settings::kkt_backend = KktBackend::Reduced` (Python:
`settings.kkt_backend = "reduced"`): `KktSystem` factors the dense Schur
complement H = P + x_reg + [A; G]^T W [A; G] with custom deterministic
kernels (sparse assembly from the KKT values, blocked Cholesky, one-block
triangular solve) instead of cuDSS. Cholesky failures drive the factor-retry
logic (reference dense backend semantics). n <= 2048. Design, tests and data:
`native/docs/modules/reduced_kkt_backend.md`; raw results:
`native/bench/results/reduced-kkt-backend-20261103/`. The default (cuDSS)
path is unchanged.

| Same-session ladder, gpu_resident, 20 repeats, end-to-end p50 (ms) | MPC | Trajopt |
|---|---:|---:|
| Native default (cuDSS) | 35.39 | 40.31 |
| Native default + reduced | 14.40 | 19.25 |
| staged + whole-solve graph + custom SpMV (cuDSS) | 29.93 | 33.98 |
| staged + whole-solve graph + custom SpMV + reduced | 10.81 | 15.88 |
| Reference (Warp) | 41.18 | 50.51 |

All 1,920 MPC and 960 Trajopt transitions of every row pass the KKT check at
1e-6, with the same iteration statistics as cuDSS (MPC p50 / max 24.5 / 28,
Trajopt 30.0 or 29.5 / 36). Maros-Meszaros (3 runs per backend): the
reduced backend solves all 89 problems with n <= 2048 in every run (49 are
above the limit and skipped); cuDSS fails 16 of those 89 in every run and 6
more in some runs. Tests: ctest 141 / 141, pytest 1,246, upstream
`test_sparse_solver.py` 7 passed / 3 skipped with either backend.


## 12. Nsight Compute pass on the reduced KKT kernels (added 2026-11-03)

ncu analysis of every kernel of the reduced backend, then four kernel
changes (one commit each), all inside `native/src/linalg/reduced_kkt/` and
only active with `kkt_backend = Reduced`; cuDSS path and defaults
unchanged. Details, per-kernel ncu tables, the generality sweep (28
synthetic MPC-like problems, edge cases) and open issues:
`native/docs/modules/ncu_kernels.md`; raw results:
`native/bench/results/ncu-20261103/`.

| Change | effect (graph replay, real states) |
|---|---|
| multi-block dataflow triangular solve | solve MPC 58.4 -> 43.9 us, Trajopt 85.8 -> 56.3 us; n = 2016: 1516 -> 280 us |
| diagonal tile: CUDA rsqrt, pipelined pivot chain | factor MPC 190.5 -> 165.9 us, Trajopt 247.8 -> 216.8 us |
| rhs reduction / recovery on compacted data | solve MPC 44.0 -> 38.8 us, Trajopt 57.0 -> 53.1 us |
| pipelined row walk in the H assembly (bitwise same H) | assembly MPC 59.4 -> 53.1 us, Trajopt 61.4 -> 57.0 us |
| total, one iteration (1 factor + 2 solves) | MPC 303.1 -> 234.5 us, Trajopt 415.6 -> 313.9 us |

| Same-session ladder, gpu_resident, 20 repeats, end-to-end p50 (ms), before -> after | MPC | Trajopt |
|---|---:|---:|
| Native default (cuDSS, control) | 36.03 -> 35.89 | 39.98 -> 39.82 |
| Native default + reduced | 14.24 -> 12.32 | 19.39 -> 16.22 |
| same + whole-solve graph (evaluated as the Auto choice, not switched) | 12.74 -> 10.88 | 17.61 -> 14.41 |
| staged + whole-solve graph + custom SpMV + reduced | 10.84 -> 8.98 | 15.96 -> 12.86 |

Every transition passes the KKT check at 1e-6 with unchanged iteration
statistics. Per iteration (Nsight Systems, iteration graph): MPC 616.5 ->
541.8 us, Trajopt 713.5 -> 604.8 us. Synthetic MPC-like sweep (28 cases,
n = 1 to 2030): iteration time faster in all 28 (-7.5% to -35.7%).
Maros-Meszaros, reduced backend, 3 runs: 89 passed, 0 failed, 49 skipped
(n > 2048) in every run before and after. Real-state accuracy within the
run-to-run noise of the states. Tests: ctest 141 / 141, pytest 1,246,
upstream 7 passed / 3 skipped with either backend.
