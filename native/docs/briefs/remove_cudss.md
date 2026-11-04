# Brief: remove the cuDSS path and the cuSPARSE dependency

User request (2026-11-04): remove the cuDSS branch and use the native solver
directly. Drop cuDSS, and cuSPARSE too if nothing needs it any more, as
dependencies.

## Target state

- The only KKT solve is the reduced backend (`native/src/linalg/reduced_kkt`):
  dense Schur complement, custom assembly, Cholesky and triangular solves.
- The only SpMV is the custom deterministic CSR/CSC kernels in
  `native/src/linalg/spmv` (today opt-in through `Settings::custom_spmv`).
- `native/CMakeLists.txt` does not `find_package(cudss)` and does not link
  `CUDA::cusparse`. `ldd` on `libcupiqp_native` and the Python extension shows
  neither `libcudss` nor `libcusparse`. CUB (header only) stays.
- Graph modes: `Auto` = whole solve in one graph (`SolveGraphDeviceLoop`);
  `IterationGraphHostLoop` and `None` stay as explicit choices. The cuDSS
  pageable-memcpy patch, the two-capture self-test and the cuDSS warm-up are
  removed.

## Where to start

1. `native/include/cupiqp/cupiqp.hpp`: `ErrorCode::Cusparse/Cudss`,
   `GraphMode` comments, `KktBackend`, `Settings::use_deterministic_mode_for_cudss`,
   `Settings::custom_spmv`, `Settings::kkt_backend`, the `cudss_*` counters,
   the version string ("(cuDSS 0.8.0)").
2. `native/src/ipm/solver/sparse_solver.cu`: backend selection,
   `effective_mode()`, `ensure_solve_graph`, setup (analysis, warm-up).
3. `native/src/linalg/{cudss,kkt_assembly,kkt_system,spmv}`: delete
   `linalg/cudss`. Check whether `kkt_assembly` (quasi-definite KKT matrix for
   cuDSS) and parts of `kkt_system` are still used by the reduced path before
   deleting them; keep what the reduced path needs, delete the rest.
4. `native/src/common/{check.hpp,handles.hpp,graph.*,api_common.cpp}`: cuSPARSE
   and cuDSS handles, error macros, the memcpy patch hooks.
5. Python: `native/python/bindings/{bind_cudss,bind_solver,bind_spmv,module}.cpp`,
   `native/python/cupiqp_native/_solver.py`, `native/tests/python/upstream_native.py`.
6. Tests: delete `tests/cpp/linalg/cudss`, `tests/python/linalg/cudss`; update
   `test_solver.cu`, `test_kkt_system_class.cu`, `test_spmv.cu`, `bench_spmv.cu`,
   `test_spmv_parity.py`, `test_reduced_backend.py`.
7. Bench: `bench/mpc_replay.cpp` (drop `--kkt-backend`, `--custom-spmv`),
   `bench/module_bench.cu`, `reduced_kkt_probe.cu`, `reduced_kkt_bench.cu`
   (they compare against cuDSS; drop the comparison or the file if it has no
   other use), `bench/harness/*.py`, `bench/tools/*.sh|py`, `bench/CMakeLists.txt`.
8. `native/poc/cudss_graph_poc.cu`: delete (it stays in git history).

## Decisions already made

- The sparsity structure is fixed after setup unless the user asks for a
  change; the backend must stay general for MPC QPs (no tuning to the MPC or
  Trajopt instances).
- The reduced backend's n > 2048 limit becomes a limit of the whole solver:
  setup returns `Unsupported`. Keep it; state it in the README, DESIGN and
  DEVIATIONS. Tests and the upstream Maros-Meszaros runs skip such problems
  (today 49 of 138) and say so in the skip reason.
- Upstream test code is reused through the Python layer. Reference settings
  fields that the upstream tests set (for example
  `use_deterministic_mode_for_cudss`) must still be accepted by the Python
  shim. Accept and ignore when that is harmless, and record it in DEVIATIONS.
  Native-only fields (`kkt_backend`, `custom_spmv`) are removed from the C++
  API and the bindings.
- Results files under `native/bench/results/` are historical records: do not
  edit or delete them. Docs keep their past numbers, but text that describes
  current behavior must no longer mention a cuDSS option. Delete
  `docs/briefs/cudss.md` and `docs/modules/cudss.md`. Add a REPORT section 14
  "cuDSS and cuSPARSE removed" and update DESIGN, DEVIATIONS,
  DEPENDENCY_GRAPH, the module notes, `docs/briefs/README.md` and
  `native/README.md`.
- Style rules (unchanged): RAII `CudaBuffer`, all allocation at setup, no host
  syncs inside the solve, CUB where it fits, folders grouped by layer.

## Do not

- Touch or git-track `/mnt/c/Users/91354/Projects/gpu-qp-benchmark` (read only).
- Delete the three untracked files in `native/bench/results/mpc-20261103/profile/`
  (the user deletes them).
- Use cloud or remote agents, or anything that uploads content.

## Verification (all required, results go in REPORT section 14)

1. Clean configure and build in a fresh build directory. `ldd` shows no
   cuDSS and no cuSPARSE. `grep -riE 'cudss|cusparse'` over `native/src`,
   `native/include`, `native/python` and `native/CMakeLists.txt` is empty.
2. ctest all pass; pytest all pass; upstream sparse tests through
   `upstream_native.py` pass. Report counts and skip reasons.
3. Upstream Maros-Meszaros suite (n <= 2048): solved count vs the previous
   reduced-backend result (89/89).
4. `bash native/bench/tools/run_auto_graph_check.sh` (update it: one
   configuration per dataset) with 20 repeats on MPC and Trajopt. Every solve
   SOLVED and KKT at 1e-6, iteration counts unchanged. Compare p50/P95
   against `native/bench/results/reduced-auto-wholegraph-20261103`
   (10.87 / 11.71 ms MPC, 14.41 / 16.39 ms Trajopt; those used cuSPARSE).
5. Harness path: `bench/harness/replay_updates.py --backend native` on both
   datasets.

## Git

Branch `feature/remove-cudss` from `native-cuda-port`. Commit in logical
steps on that branch only. Do not merge into `native-cuda-port` and do not
push: the repository CLAUDE.md leaves merging and pushing to the user.
