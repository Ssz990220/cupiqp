# Brief for module agents (wave 1)

Read first: `native/docs/DESIGN.md` (especially sections 5, 6, 8 and 10),
`native/docs/DEPENDENCY_GRAPH.md` (your module's row), `native/docs/FEASIBILITY.md`,
the upstream `AGENTS.md`, and the reference sources listed for your module.
The pinned reference Python sources in `cupiqp/` are the numerical authority.

## Workspace
- Your worktree: `~/Projects/cupiqp-native-wt/<module>` on branch `task/<module>`,
  created from `native-cuda-port`. Work and commit only there.
- Python venv (reference cupiqp, Warp 1.19.0, nanobind, pytest, piqp, scipy):
  `~/Projects/cupiqp-native/.venv/bin/python`. Do not install packages.
- Build directory: `<worktree>/build/native` (gitignored). Configure once:
  ```bash
  cd <worktree>/native
  cmake -S . -B ../build/native -DPython_EXECUTABLE=$HOME/Projects/cupiqp-native/.venv/bin/python
  cmake --build ../build/native -j 8
  (cd ../build/native && ctest --output-on-failure -R <YourSuite>)
  CUPIQP_NATIVE_BUILD=<worktree>/build/native \
    WARP_CACHE_PATH=$HOME/.cache/warp-cupiqp-native \
    $HOME/Projects/cupiqp-native/.venv/bin/python -m pytest -q <worktree>/native/tests/python/<module>
  ```
  Use `-j 8` (other agents share the machine). The GPU is shared: keep tests
  small (well under a minute each).

## Ownership
You may create or edit ONLY:
- `native/src/<module_dir>/**` (implementation). The interface header there is a
  contract set by the integration owner. You may add private members, helpers,
  and new public functions. Do not change existing public signatures; if one
  is wrong, implement the closest correct version and explain the requested
  change in your hand-off note.
- `native/tests/cpp/<module_dir>/**` (GoogleTest; `CMakeLists.txt` with
  `cupiqp_add_test(<name> <sources>)`).
- `native/tests/python/<module_dir>/**` (pytest parity tests).
- `native/python/src/bind_<module>.cpp` (your `_testing` entry points; the
  function signature `void bind_<module>(nb::module_& m, nb::module_& testing)`
  is fixed).
- `native/src/<module_dir>/sources.cmake` (append your sources to `CUPIQP_SOURCES`).
- `native/docs/modules/<module>.md` (hand-off note).

Everything else is read-only, including `include/`, `src/core/`, other
modules, top-level CMake, `python/src/module.cpp`, `python/cupiqp_native/*`,
`tests/python/conftest.py` and the reference `cupiqp/` and `tests/`. If you
need a core change, describe it in the hand-off note and work around it
locally.

## Conventions
- C++17/CUDA 13.2, namespace `cupiqp`. Kernels templated on `T`; instantiate
  `double`. Every function that enqueues GPU work takes `cudaStream_t s` and uses
  only it (library handles: call `cusparseSetStream`/`cudssSetStream` with `s` on
  every call). No allocation or synchronization outside `init()`/setup paths.
- RAII: `CudaBuffer<T>`, `CudaPinnedBuffer<T>`, `CudaStream`, `CudaEvent`
  (`src/core/*`). Check every API call with `CUPIQP_*_CHECK`; check launches
  with `CUPIQP_CUDA_CHECK_LAUNCH()`.
- Reductions: `core/reduce.cuh` (`block_sum/max/min<BLOCK>`, `reduction_block_dim`,
  `dispatch_block`), i.e. cub::BlockReduce. One block per problem where the
  reference uses `wp.launch_tiled(dim=[B])`.
- Version 1 of each kernel mirrors the Warp kernel (thread mapping, formula
  order, masks). See DESIGN section 10.6 for the required optimization
  assessment.
- Comments: ASCII only (upstream AGENTS.md rule).
- `_testing` bindings take device addresses as Python `int` (`uintptr_t`) plus
  explicit sizes. On the Python side use `cupiqp_native._interop.dev_ptr(arr,
  dtype, size)` to get addresses from Warp arrays. Allocate test arrays with
  Warp (`wp.array(np_array, dtype=wp.float64, device="cuda:0")`).
- Parity tests reuse the reference: import `cupiqp` and its classes or kernel
  factories, build inputs with `tests.helpers` (upstream) or NumPy/SciPy, run the
  reference component, run the native component on copies of the same
  inputs, compare. Cover the cases listed in DESIGN section 8 that apply.
- Commit on your branch with clear messages, ending each message with:
  ```
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01Xghh14CwbdJRX2uQDMuE1x
  ```
  Use `git -c user.name="sizhe.sui" -c user.email="sizhe.sui@utexas.edu" commit ...`.
  Do not push. Do not merge into `native-cuda-port`.

## Hand-off note (`native/docs/modules/<module>.md`)
Cover: what is implemented; the files; the public API as implemented; test
results (exact commands and pass counts); parity numbers (max abs/rel
differences vs the reference); deviations from the reference and why; the
optimization assessment (tiling, tensor cores, shared memory/registers);
requested changes to shared code; known gaps.
