# Module M1 `data` (problem storage): hand-off note

## What is implemented
- `validate_csr`: host check of a CSR pattern, the same as the reference
  `sparse_solver._check_csr`. It checks that `indptr` has `rows + 1` entries,
  that `cols` matches (`expected_cols`, or square when it is negative), that
  nnz < 2^31, that `indptr` starts at 0, never decreases and ends at nnz, that
  column indices are in `[0, cols)`, and that indices are strictly increasing
  within each row. The messages are the reference strings. For the 7 bad
  patterns in the pytest suite they are byte-identical to the reference text.
- `SparseData::init`: validates P (square), A and G (n columns). Then it
  applies the reference block rules: A <-> b; G needs h_l or h_u; h_l/h_u
  need G; P values and c are required. It also rejects A/G values given
  without a pattern. Validation finishes before any state changes. Absent
  A/G become `rows = 0, cols = n` patterns with a one-entry device `indptr = [0]`,
  so the views are valid and empty. The bound sides present at init fix
  `num_hl/num_hu/num_xl/num_xu` (each is `m`/`n` or 0). `init` allocates every
  buffer and uploads `indptr`, `indices` and the COO `row_idx` of P, A and G.
  The initial values are copied device to device in stream order, then the
  masks are computed. `init` does not synchronize.
- `set_values` / `set_values_from_host`: validate first, then copy in stream
  order and return the `UpdateMask`. A bound side absent at setup is rejected
  with the reference message, and nothing is copied. If any bound changed,
  the masks are refreshed once. No allocation, no sync. The host variant
  issues `cudaMemcpyAsync(H2D)` straight from the caller pointer, which works
  for both pinned and pageable memory.
- `update_finite_masks`: a port of `finite_bound_masks_kernel`. A lower bound
  is finite iff `v > -1e20` and an upper bound iff `v < 1e20`. These are the
  reference's one-sided comparisons: NaN gives 0, a lower bound of +inf
  counts as finite and an upper bound of -inf counts as finite. Layout is
  `[hl|hu|xl|xu]`; `active_G_row = max(fl, fu)`, `active_x_bound`,
  `num_finite_bounds`.

## Files
- `src/problem/data/sparse_data.hpp`: the contract, plus the additions listed below.
- `src/problem/data/csr_validation.cpp`: `validate_csr`.
- `src/problem/data/sparse_data.cu`: `SparseData`.
- `src/problem/data/finite_masks_kernels.cuh`: the mask kernels, v1 (serial, mirrors Warp) and v2 (block).
- `src/problem/data/sources.cmake`
- `tests/cpp/problem/data/{CMakeLists.txt,test_sparse_data.cu}`
- `tests/python/problem/data/test_data_parity.py`
- `python/bindings/bind_data.cpp`: `_testing.validate_csr` and `_testing.SparseData`
  (`init`, `set_values`, `set_values_from_host`, `update_finite_masks`,
  `update_finite_masks_reference`, `dims`, `has_bounds`, `view`).

## Public API as implemented
The contract is unchanged. New public members (additive only):
- `void update_finite_masks_reference(cudaStream_t)`: runs kernel version 1, for parity tests and timing.
- `bool has_h_l() / has_h_u() / has_x_l() / has_x_u() const`: the sides given at init. These stay
  true even when the side has length 0 (m == 0).
- `bool initialized() const`
New private members: `initialized_`, `has_*_`, `apply_values`, `check_update`.

## Tests
```
cd native && cmake --build ../build/native -j 8
(cd ../build/native && ctest --output-on-failure -R "ValidateCsr|SparseData|FiniteMasks|SetValues")
  -> 12/12 passed (the full ctest, including the common tests: 19/19)
CUPIQP_NATIVE_BUILD=<wt>/build/native WARP_CACHE_PATH=$HOME/.cache/warp-cupiqp-native \
  ~/Projects/cupiqp-native/.venv/bin/python -m pytest -q <wt>/native/tests/python/problem/data
  -> 24 passed
```
C++ tests cover validation accept/reject, the init rules (including no state
change on throw and a second init), empty views for absent blocks, patterns,
`row_idx` and values with explicit zeros (and zeros becoming nonzero), and
the masks. Mask cases are all-finite, one-sided, mixed/edge values (+-inf,
+-1e20, just inside, NaN, lower +inf, upper -inf) and finite<->infinite
switches, all checked bitwise against a CPU copy of the reference kernel. The
block kernel is checked against v1 at n = 5000, m = 3000. Further tests
cover rejection of absent bound sides, host updates (pinned and pageable),
200 repeated updates with every device address unchanged and no
`cudaMemGetInfo` decrease (2 MiB slack on the shared GPU), and G with 0 rows.

## Parity vs the reference
The reference is a `cupiqp.SparseSolver` with `preconditioner_iter = 0`; its
`solver.data` is compared with the native `SparseData`. The comparison is
bitwise (uint64 view) on P/A/G values, `indptr`, `indices`, `row_indices`,
c, b, the four bounds, `finite_mask_all`, `active_G_row`, `active_x_bound` and
`num_finite_bounds`. It holds for:
- 9 setup cases (all finite, 40% inf, no A, no G, box only, h_u+x_l only,
  h_l+x_u only, no box, n=300/p=40/m=200);
- edge values;
- 6 rounds of bound switches (including a partial update and all-infinite);
- explicit zeros becoming nonzero;
- host updates;
- kernel v1 vs v2.

Max abs/rel difference: 0 (bitwise equal everywhere). The init-rule and
absent-side messages match the reference regexes. As a sanity check, I
mutated `>` to `>=` in the lower-bound test, and the edge-value parity test
failed as expected; the change was reverted.

## Deviations from the reference
1. The masks are refreshed once per `set_values` call; the reference
   refreshes after each bound setter. The final masks are identical.
2. The default mask kernel is v2: one block per problem, threads stride over
   rows and variables, and the count is `block_sum`. v1 (one thread per
   problem, as in Warp) remains available. Outputs are bitwise identical,
   because the count is a sum of 0/1 values and so exact in any order.
3. A null pointer cannot be told apart from an empty array. So `b` may be
   null when A has 0 rows, and "Either h_l or h_u" is enforced only when
   m > 0. The reference raises for G with m == 0 and no bounds.
4. Extra rejections the reference cannot express: A/G values without a
   pattern, a missing P values pointer when nnz_P > 0, a missing c when
   n > 0, and a pattern `cols` that does not match.
5. Batch (B > 1) and value-array shape checks are not done here, because
   `ValuesView` carries no lengths. The binding layer (M9) must check lengths.

## Optimization assessment
Timings: host wall time per call, 200 calls, RTX 4070 SUPER, GPU shared with
other agents.

| n / m | block (v2) | serial (v1) |
|---|---|---|
| 100 / 60 | 5.8 us | 15.4 us |
| 1000 / 600 | 6.2 us | 136 us |
| 10000 / 6000 | 20.3 us | 1325 us |
| 100000 / 60000 | 215 us | 12983 us |

v2 is merged as the default (it keeps parity and shows a measured gain).
- Tiling: with B = 1, v2 uses a single block. For n beyond about 10^4, a
  grid-wide variant would help: many blocks, an integer `atomicAdd` count
  (exact and deterministic), then a final int->double convert. This needs a
  zeroed int scratch value plus one extra tiny kernel. Not done, because
  masks are only recomputed on bound updates.
- Tensor cores: not applicable (comparisons only; no FP64 tensor cores on sm_89).
- Shared memory/registers: none needed. The kernel is a single streaming pass.
  It could be fused into M3's x_b_scaling refresh if a profile shows it matters.
- Not provided: `x_b_scaling_init`, listed under M1 in DESIGN section 5. It
  lives in `ScalingView` (owned by M3) and equals `active_x_bound` when box
  bounds exist, else 1.

## Requested changes to shared code
- `DESIGN.md` 2.1 says "finite iff -1e20 < v < 1e20". The reference test is
  one-sided per side (lower: `v > -1e20`; upper: `v < 1e20`), so a lower
  bound of +inf counts as finite. Suggest rewording the docs; the code follows
  the reference.
- `views.hpp`: nothing required.

## Known gaps
- No B > 1.
- No length validation of value arrays (see deviation 5).
- `set_values_from_host` with pageable memory is effectively synchronous on
  the host. That is a CUDA property; the solver-level `update_from_host` with
  pinned staging is M8's job.
- Changes are left uncommitted in the worktree (see the report).
