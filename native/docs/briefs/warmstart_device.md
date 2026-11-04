# Brief: device-side warm start (phase 2 of the warm-start study)

User decision (2026-11-04): implement the proposed design and run the
experiments. Hard constraints from the user: minimum CPU copies, and the whole
solve stays one CUDA graph.

Read first: `native/docs/modules/warmstart_study.md` (phase 1 results, the
recommended strategy x = 0, y/z from the previous solution, s from the new
bounds, s >= 0.1, z >= 1e-4) and `native/docs/briefs/warmstart_study.md`.
Starting code: `SparseSolver::warm_start_buffers` / `solve_warm` in
`native/src/ipm/solver/sparse_solver.cu` (commit 164ddf6, branch
`feature/warmstart-study`).

## Facts about the data that drive the design

- Collision block: rows 1569..2320 = 47 samples x 16 slots, same in every QP.
  Every slot is always filled (no slot has both bounds infinite); the label
  holds an integer pair id (`pair=34(...)`). Adding or removing a collision
  pair means a slot changes content; the row count and pattern do not change.
- Alignment quality only changes speed, never correctness (any strictly
  positive s, z start is valid; clamps bound the damage). Phase 1: raw labels
  10 / 12.6 med/mean iterations, row identity 10 / 10.4, (sample, pair) 10 / 10.0,
  cold 29 / 28.8.

## What to build (all on the device, inside the one solve graph)

1. Warm start from the previous solve, no host round trip:
   x = 0; y = y_prev; z[i] = max(z_prev[match[i]], z_min), or a neutral
   value when there is no match; s[i] from the new bounds at x = 0 clamped to
   s_min. z_prev / y_prev are the solver's own previous result (keep a device
   copy in scaled or unscaled space, as is correct with Ruiz scaling, which
   may change between QPs). One elementwise kernel.
2. Generic row matching (the solver must not know about collisions):
   - optional per-row int64 key plus a group descriptor (for example
     contiguous groups of fixed size, or a group id per row). The solver keeps
     the previous keys and matches each row against the previous keys of the
     same group on the device (linear search within a small group; for large
     groups, use a CUB-sorted approach). Unmatched rows get -1.
   - no keys given: identity matching (row i -> row i).
   - keys are uploaded once per QP through a preallocated pinned buffer and
     one async copy captured in the graph, or given as a device pointer when
     the caller already has them on the GPU. This is the only per-QP host
     traffic the feature adds; measure it.
3. Mid-solve snapshot (Gondzio-Grothey, causal version): ring buffer of K
   iterates (y, z, s; x is not needed), one D2D copy per iteration captured
   in the loop body; at the end of the solve, the warm start for the next QP
   can use the iterate K back (or the last one if fewer than K iterations
   ran). K and the "how far back" choice are settings. All buffers are
   allocated at setup (or on first enable, outside the solve).
4. Fallbacks: the first QP of a run is cold; if the previous status is not
   Solved, the next solve is cold. Decide this on the device (conditional node
   or a flag read by the start kernel), not by a host branch that needs a
   sync.
5. API: settings or options struct (s_min, z_min, neutral z, K, back offset,
   enable flags) plus `set_row_keys(...)` (host or device). Keep the default
   cold path unchanged and bitwise identical; test that.

## Experiments

1. Baseline reproduction: cold, and the phase-1 recommended strategy (now
   device-built), on both runs; iterations and time must match phase 1 or the
   difference must be explained.
2. Matching: identity vs keys (sample index, pair id) vs raw-label-like keys.
3. Robustness curve: corrupt 0, 10, 30, 60, 100 % of the collision rows
   (unmatched, and mismatched to a wrong row), several seeds; iterations
   median / mean / max and failures.
4. Ring-buffer snapshot: K in {4, 8, 12}, back offset in {last, K}; compare
   with the end-of-solve state.
5. Generality: run the MPC replay dataset (`native/bench/data/mpc`, pattern
   fixed per group) with the same s_min / z_min; report whether they carry
   over.
6. Cost: per-solve overhead of the start kernel, the matching, the per-
   iteration ring copy; graph launches per solve (must be 1) and host syncs
   (must be 0); host-to-device bytes per QP.

All runs: SOLVED and KKT at 1e-6 required; CLOCK_MONOTONIC_RAW timing.

## Rules and deliverables

- Style rules: RAII `CudaBuffer`, allocation only at setup or first enable,
  no host sync in the solve, CUB where it fits, folders grouped by layer,
  ASCII-only comments, user-facing docstrings.
- Do not touch or git-track `/mnt/c/Users/91354/Projects/gpu-qp-benchmark`. No
  cloud agents or uploads.
- Branch `feature/warmstart-study`; commit there only, do not merge or push.
- Results in `native/bench/results/warmstart-device-20261104/`; extend
  `native/docs/modules/warmstart_study.md` with a phase-2 section; ctest and
  pytest pass.
