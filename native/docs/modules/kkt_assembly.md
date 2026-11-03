# Module M4 kkt_assembly: hand-off note

## What is implemented
- Host build (C++, no SciPy) of the pattern of
  `bmat([[P + I_n, None, None], [A, I_p, None], [G, None, I_m]])` after `sum_duplicates()`:
  sorted columns, union pattern, every stored P/A/G entry (explicit zeros included) keeps a slot.
- bmat template values reproduced bitwise: `1.0 + 1.0` on a P diagonal stored by P, `1.0` everywhere
  else. Slots that are never scattered (identity diagonals of y/z, x diagonals not stored by P) keep
  them until `update_diagonal`.
- Slot maps: `P_slots`, `A_slots`, `G_slots` (KKT position of every stored entry), `diag_x/y/z`
  (reference `csr_diag_indices` split), `P_diag_src/dst` (reference `_P_diag_src_cols/_dst_cols`),
  and `G_rows` (host COO rows of G, kept for tests; the device kernel uses `data.G.row_idx`).
- Device kernels (version 1, one thread per `(b, k)`, flattened `t = b * count + k` like Warp):
  `scatter_values_kernel`, `gather_scatter_values_kernel` (P_diag refresh),
  `scatter_masked_G_kernel` (`G * active_G_row[row]`), `update_kkt_diag_kernel`
  (x: `P_diag + x_reg`; y: `-delta[b]`; z: `zr > 0 ? -zr : -1`).
- `init` order matches the reference: pattern/maps upload, template values, `P_diag = 0`, then
  `scatter(P, A if p > 0, G if m > 0)` which also refreshes `P_diag`.
- `scatter` mirrors `update_data` / `_scatter_P_A_G`, including the launch guards
  (`nnz > 0`, `p > 0`, `m > 0`) and the P_diag refresh only when P is flagged.

## Files
- `src/linalg/kkt_assembly/kkt_assembly.hpp` (contract; private members changed, one accessor added)
- `src/linalg/kkt_assembly/kkt_pattern.hpp`, `kkt_pattern.cpp` (host pattern and maps)
- `src/linalg/kkt_assembly/kkt_assembly_kernels.cuh` (kernels, templated on `T`)
- `src/linalg/kkt_assembly/kkt_assembly.cu` (class implementation)
- `src/linalg/kkt_assembly/sources.cmake`
- `tests/cpp/linalg/kkt_assembly/{CMakeLists.txt,test_kkt_assembly.cu}`
- `tests/python/linalg/kkt_assembly/test_kkt_assembly_parity.py`
- `python/bindings/bind_kkt_assembly.cpp` (`_testing.KktAssembly`)

## Public API as implemented
Contract signatures unchanged: `init`, `dim`, `nnz`, `indptr`, `indices`, `values`, `P_diag`,
`scatter`, `update_diagonal`, `host_pattern`. Additions:
- `const KktHostMaps& host_maps() const` (host pattern, template values and every index map).
- Free function `KktHostMaps build_kkt_host_maps(P, A, G)` (in `kkt_pattern.hpp`).
- The private `CsrPattern host_` member was replaced by `KktHostMaps maps_` (`host_pattern()`
  returns `maps_.kkt`).

Requirements on the inputs: `data.dims` must match the patterns (n, p, m, nnz), and when G is
scattered `data.G.row_idx` and `data.active_G_row` must be non-null (InvalidArgument otherwise).
An absent A/G is a pattern with `rows == 0` (an empty `indptr` is accepted). `init` allocates;
`scatter` and `update_diagonal` only launch kernels on `s` (no allocation, no sync) and are
graph-capturable in global capture mode (tested).

## Tests (run on this worktree)
```bash
cd ~/Projects/cupiqp-native-wt/kkt_assembly/native
cmake -S . -B ../build/native -DPython_EXECUTABLE=$HOME/Projects/cupiqp-native/.venv/bin/python
cmake --build ../build/native -j 8
(cd ../build/native && ctest --output-on-failure -R "KktAssembly|KktShapes")
#   -> 100% tests passed, 0 tests failed out of 17 (plus 1 disabled timing test)
CUPIQP_NATIVE_BUILD=~/Projects/cupiqp-native-wt/kkt_assembly/build/native \
  WARP_CACHE_PATH=$HOME/.cache/warp-cupiqp-native \
  $HOME/Projects/cupiqp-native/.venv/bin/python -m pytest -q \
  ~/Projects/cupiqp-native-wt/kkt_assembly/native/tests/python/linalg/kkt_assembly
#   -> 10 passed
```
C++ (GoogleTest, 17 enabled): for 7 shapes (including p = 0, m = 0, p = m = 0, a P with no stored
diagonal, a P with every diagonal stored, n = p = m = 1, and B = 2) the pattern and template values
match an independent dense bmat construction in the test, diagonal slots and P-diagonal maps
match; device pattern, init values, P_diag and the post-`update_diagonal` values match a host
model bitwise. Plus: explicit zeros that become nonzero land in their slots, inactive G rows are
zero, partial flags (A only) leave every other slot unchanged; scatter + diagonal update captured
in `cudaStreamCaptureModeGlobal` (rejects allocation) and replayed equal the eager result bitwise,
and the `values/P_diag/indptr/indices` addresses never change; inconsistent inputs throw.

pytest parity (10) against the reference `SparseKKTSolver` built from a reference `SparseData`
with explicit zeros, P rows missing their diagonal, and rows with both bounds infinite:
- shapes (n, p, m) = (12,4,9), (15,0,7), (10,3,0), (8,0,0), (30,10,40), (1,1,1): `indptr`,
  `indices` identical to `_kkt_mats`; template values bitwise equal to
  `_initialize_kkt_csr(...).data`; every slot map identical to the reference arrays
  (`_P_indices`, `_A_indices`, `_G_indices`, `csr_diag_indices`, `_P_diag_src/dst_cols`);
  KKT values and `P_diag` bitwise equal after init.
- shapes (12,4,9), (15,0,7), (10,3,0), (30,10,40): after `update_data` with all explicit zeros
  turned nonzero plus h_l/h_u switches (active <-> inactive rows), after P-only and G-only updates,
  and after two `update_and_factor_kkt` diagonal updates with random x_reg/delta/z_reg (about 30%
  of z_reg exactly 0): values and `P_diag` bitwise equal, `values` address unchanged.

Parity numbers: max abs difference 0 and max rel difference 0 (bit-for-bit equality, compared as
uint64) in every comparison above.

Mutation check (run once, then reverted): removing the `active_G_row` factor and changing the
inactive z diagonal from -1 to -2 made 6 of 10 pytest cases and 5 C++ tests fail.

## Deviations from the reference
- None in values. The pattern is built with a direct row merge instead of SciPy `bmat` +
  `searchsorted`; it is checked to be identical.
- `-0.0` appears for negative G entries of inactive rows (`G * 0.0`), exactly as in the
  reference; bitwise comparisons therefore distinguish it and pass.
- B > 1 is supported by the kernels and by `init` (template replicated per problem by D2D copies)
  but parity was run with B = 1 only (C++ host-model test covers B = 2).

## Optimization assessment (MPC shape)
Measured with the disabled test `KktAssembly.DISABLED_TimingMpcShape` (synthetic random pattern
with n = 169, p = 64, m = 2521, density 249801 / (2521 * 169), giving nnz(G) = 250104,
nnz(A) = 6493, nnz(P) = 16635; RTX 4070 SUPER under WSL2, shared GPU, two runs):

| call | eager launch (us/call) | inside one CUDA graph (us/call) |
|---|---|---|
| scatter P + A + G (+ P_diag refresh), 4 kernels | 21.0 - 29.9 | 6.7 |
| scatter G only | 7.2 - 7.3 | 3.5 |
| scatter P only (+ P_diag refresh), 2 kernels | 9.2 - 14.0 | 2.2 |
| update_diagonal | 5.0 - 6.8 | 1.3 |

Run with `./build/native/tests/cpp/linalg/kkt_assembly/test_kkt_assembly
--gtest_also_run_disabled_tests '--gtest_filter=*Timing*'`.

- Eager timings are dominated by launch overhead (about 5 us per launch here); inside a graph the
  G scatter (about 6 MB of traffic, L2 resident across repetitions) is 3.5 us.
- Coalescing: the slot maps are already monotonically increasing (P, A, G entries and the KKT are
  both row-major with sorted columns, and the KKT row only inserts the diagonal), so writes of
  consecutive threads hit consecutive or nearly consecutive addresses. Sorting the maps gains
  nothing.
- Fusion: scatter and `update_diagonal` run at different times in the solver (`update()` versus
  each IPM iteration), so fusing those two only helps the first iteration after an update. The
  useful fusion is the 4 scatter launches (P, A, masked G, P_diag refresh) into one kernel over
  the concatenated entry range; that saves 3 launches per `update()` (about 15 us eager on this
  machine, about 1-2 us in a graph). Not implemented: it changes no values, the solver will
  likely capture `update()` in a graph, and the gain is below the cuDSS factorization cost.
  Recommended only if `update()` stays eager.
- Tiling / shared memory: these are pure gathers/scatters with no reuse, so neither tiling nor
  shared memory applies. Per-problem scalar `delta[b]` is a single load per thread already.
- Tensor cores: not applicable (no arithmetic beyond one add/multiply per entry).

## Requested changes to shared code
None required. Note for M6/M8: `scatter` reads `data.G.row_idx` and `data.active_G_row` from the
`SparseDataView`, so the data module must fill `G.row_idx` (contract already says so).

## Known gaps
- Parity tests run B = 1 only (B = 2 covered by the C++ host-model test).
- The MPC timing uses a synthetic random pattern of the MPC dimensions, not the real snapshot
  pattern.
- Tests build `SparseDataView` from a local fixture; they have not yet been run against the real
  `SparseData` (M1) view.
- Changes are left uncommitted on `task/kkt_assembly` (see the report).
