# M3 ruiz: hand-off note

Sparse Ruiz equilibration (FP64, B = 1; kernels take B). Port of reference
`cupiqp/preconditioner.py` `RuizEquilibration` + `cupiqp/sparse/sparse_preconditioner.py`
`SparseRuizEquilibration` and their Warp kernels.

## Files
- `src/problem/ruiz/ruiz.hpp`: contract (public signatures unchanged). Added: `dims()`,
  the constants `kMinScaling`/`kMaxScaling`/`kConvergenceTol`, private backend hooks
  (`compute_kkt_norms`, `scale_matrices`, `apply_cost_scaling`, `scale_bounds`), private
  buffers `P_norms_`, `gamma_buf_`, `ones_`. `done_flag_` has 2 ints (flag, block ticket).
- `src/problem/ruiz/ruiz_kernels.cuh`: kernels, one per reference Warp kernel.
- `src/problem/ruiz/ruiz.cu`: host side. `src/problem/ruiz/sources.cmake`.
- `python/bindings/bind_ruiz.cpp`: `_testing.RuizTester(dims: dict, ptrs: dict)` with
  `scale_data`, `unscale_data(replaced: dict)`, `reuse_scaling`, `reset`,
  `refresh_x_b_scaling`, `unscale_solution`/`scale_solution(primal, dual)`,
  `rhs_inf_norm(out)`, `passes_run()`, `state()` (name -> (address, count)).
- `tests/cpp/problem/ruiz/{CMakeLists.txt,test_ruiz.cu}`, `tests/python/problem/ruiz/test_ruiz_parity.py`.

## Public API as implemented
All methods enqueue on `s` only, allocate nothing (except `init`) and never synchronize.
- `init`: allocates everything, fills factors with 1, `x_b_scaling = x_b_scaling_init`.
- `scale_data(data, scale_cost, max_iter, s)`: begin kernel (done = 0, passes = 0), then
  `max_iter` passes of: row norms, column norms (A, G; atomic fmax), clamp + rsqrt,
  scale P/A/G/c, accumulate deltas, [cost scaling: zero P_norms, P norms, gamma, apply],
  convergence kernel. Then the finalize kernel (inverses, bound scaling, unscale factors).
- `unscale_data(data, replaced, s)`, `reuse_scaling`, `reset`, `refresh_x_b_scaling`,
  `unscale_solution`, `scale_solution`, `compute_constraints_rhs_inf_norm_unscaled`: as the reference.
- `passes_run()`: device int32, passes executed by the last `scale_data`.

## Required differences
1. Device early exit. The convergence kernel (one block per problem; batch max via
   atomic fmax into `conv_buf_`; the last block by ticket decides) sets `done = 1` when
   `conv < 1e-3`, exactly the reference host test. Every per-pass kernel returns at once when
   `done != 0`; the `P_norms` memset still runs (scratch only). It counts passes in
   `passes_run`. The finalize kernel runs after the loop regardless. No host sync; the
   whole `scale_data` is graph-capturable (tested).
2. `unscale_data` skips replaced blocks. Every element of P, c, A, G, b, h_l, h_u, x_l, x_u is
   updated from only its own old value and the stored factors (P and c share the cost
   factor but not each other's values), so each of the nine blocks is skipped independently.
   The `x_b_scaling *= delta_b_inv * d_x_inv` update always runs. Retained blocks are
   bitwise equal to the reference after unscale (tested).

## Correction to the brief/contract: x_b_scaling_init
The brief says "aliases active_x_bound when any box side exists, else a ones buffer". The
reference `SparseSolver._init_preconditioner` always passes `Data.active_x_bound`, which is
all zeros when no box side exists (checked: reference `x_b_scaling_init` = 0 for a problem
without x_l/x_u). The port follows the reference: alias `data.active_x_bound` whenever it is
non-null; a ones buffer only when it is null (reference default `active_x_bound=None`).
`test_x_b_scaling_init_alias` covers it. The header comment was updated accordingly.
The integration owner should make sure `SparseDataView.active_x_bound` is always non-null.

## Numerics
- Warp `wp.max`/`wp.min`/`wp.clamp` are fmax/fmin (NaN treated as missing); Warp
  `atomic_max(float64)` is a CAS loop over fmax. Both are reproduced (`FmaxOp`,
  `atomic_fmax`, `warp_clamp`), so max-norms match including the NaN behavior.
- Column norms: the reference uses a per-nonzero atomic max; max is order-independent, so
  the atomic fmax gives deterministic, bitwise-equal results.
- The only order-dependent operation is the cost-scaling mean (`tile_sum` in Warp, cub
  `block_sum` here: Warp reduces lanes with shfl_down 16..1, cub with 1..16). In every tested
  case the result was still bitwise equal, but this is not guaranteed in general.

## Tests (run 2026-11-03)
```
cmake --build ../build/native -j 8
(cd ../build/native && ctest --output-on-failure -R Ruiz)          # 5/5 passed
CUPIQP_NATIVE_BUILD=<wt>/build/native WARP_CACHE_PATH=$HOME/.cache/warp-cupiqp-native \
  ~/Projects/cupiqp-native/.venv/bin/python -m pytest -q -s native/tests/python/problem/ruiz   # 339 passed (87 s)
```
C++ (`test_ruiz`): iter 0 is the identity (bitwise, 5 layouts); scale -> unscale round trip
within 1e-14 relative plus reuse_scaling vs a CPU reference; device early exit vs a CPU
reference with a host-driven break (pass counts equal for 40 configurations, early exit
taken in some; values bitwise without cost scaling, 1e-13 with); no allocation over 20
scale/unscale/reuse/reset cycles (`cudaMemGetInfo`); `reset + scale_data` captured in a
graph and replayed 3 times, bitwise equal to the eager run.

pytest parity vs reference `SparseRuizEquilibration` (488 compared cases, each up to 19
arrays): problems feasible, sparse, sparse with explicit zeros, infinite bounds, badly
scaled (1e-6..1e6), large (n=300, p=40, m=150); 10 block layouts (no A, no G, no A/G,
one-sided h, no box, one-sided box, only P); scale_cost on/off; max_iter 0/1/10 (25 for
large). Compared: scaled P/A/G/c/b/h_l/h_u/x_l/x_u, delta, delta_inv, delta_b,
delta_b_inv, cost_scaling(_inv), x_b_scaling, x_b_scaling_init, both unscale factors, RHS
norm, pass count, unscale_solution/scale_solution, and the solver update sequence
(unscale_data -> set blocks (incl. finite<->infinite bound flips) -> reuse_scaling or
reset + scale_data -> refresh_x_b_scaling -> RHS norm, twice) for 6 update patterns.
Result: max abs diff 0, max rel diff 0, all arrays bitwise equal in all 488 cases; pass
counts equal everywhere (observed 0, 1, 2, 6, 10, 13 passes).

## Optimization assessment (not implemented, not measured)
- Pass fusion: one pass is 6 kernels (10 with cost scaling), so `preconditioner_iter=10`
  is 60-100 launches per fresh Ruiz. For MPC sizes (n up to a few thousand) a single
  persistent kernel, one block per problem, could run the whole pass loop with
  `__syncthreads()` between the phases and the early exit as a loop break, keeping
  `d_iter`, `d_b_iter` and the norms in shared memory (W * 8 bytes; fits up to ~6k rows
  at 48 KiB). In graph mode the launch cost is smaller but each node still costs ~1-2 us.
- Smaller fusions: clamp+rsqrt cannot move into the scale kernel without a grid-wide
  barrier (scaling row r reads d of arbitrary columns). The accumulate kernel (B, W) could
  be folded into the conv-check kernel for B = 1 (both only read d_iter/d_b_iter after
  scaling), saving one launch per pass. gamma + apply could be one single-block kernel when
  nnz_P is small.
- Tiled CSR norms: column norms could use a CSC permutation built at setup (gather, no
  atomics) or warp-per-row tiles for long rows; row loops are one thread per row, which
  under-uses the GPU for short MPC rows but is memory-bound anyway.
- Tensor cores: none (FP64 on sm_89; element-wise and max-reduction work only).

## Requested changes to shared code
- None required. Suggest: the brief/DESIGN text on `x_b_scaling_init` (see above), and the
  repository `CLAUDE.md` forbids agent commits while the brief asks for them; this module's
  changes were left uncommitted for the owner.

## Known gaps
- Only B = 1 is exercised (kernels take B; the multi-block convergence ticket is untested).
- Bitwise parity of the cost-scaling sum is observed, not guaranteed.
- `scale_data` with `n == 0` divides by zero in gamma, as the reference does.
