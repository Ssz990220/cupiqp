# Warm-start study on consecutive TrajOpt SQP QPs (2026-11-04)

Brief: `native/docs/briefs/warmstart_study.md`. Branch `feature/warmstart-study`
from `native-cuda-port` (a9f74da). GPU: RTX 4070 SUPER (sm_89), CUDA 13.
Data: `native/bench/data/warmstart/cupiqp_warmstart_sequences_20261104/` (2 runs,
37 + 64 QPs, 99 transitions k -> k+1; n = 241, p = 64, m = 6385).

Phase 2 (the warm start built on the device, `solve_warm_start`) is at the
end of this note.

## Summary

- Warm starting works, and it works the way the user framed it: keep x = 0,
  take y and z from the solution of QP k, set the slacks from the data of
  QP k+1 at x = 0 (s_u = h_u, s_l = -h_l), and clamp s >= 0.1 and z >= 1e-4.
  Over the 99 transitions this cuts the IPM iterations from a median of 29 to 10
  (mean 28.8 to 10.0, worst 33 to 25) and the solve time from 12.0 ms to
  4.2 ms (2.9x at the median). Every solve returned Solved and passed the KKT
  check at 1e-6.
- The options that exist without code changes do nothing useful. Changing
  `rho_init` / `delta_init` leaves the iteration counts unchanged. Loosening
  `eps_abs` and `eps_duality_gap_abs` to 1e-5 saves 0.6 iterations and then
  fails the 1e-6 KKT check on 63 of 99 QPs.
- The safeguard levels matter more than any other choice. Clamping s and z at
  1e-4 (the naive choice) needs a median of 12 iterations and a worst case of
  37. Clamping s at 0.1 and z at 1e-4 needs 10 and 25. Starting x at x_k
  instead of 0 neither helps nor hurts once the slacks are recomputed.
- The assumption holds. QP k+1 is linearized at the trajectory after applying
  the full step x*_k: on the joint-position rows, the bounds of QP k+1 equal
  the bounds of QP k minus C x*_k to 1e-15. It breaks in one place that
  matters: the collision labels contain the sample time, which moves whenever
  delta_t changes. In the pick run, matching collision rows by the raw label
  loses 40% of them and costs 5 extra iterations at the median. Keying a
  collision row by (sample index, link pair) fixes this.
- Recommendation: adopt `x0_duals` (x = 0, y and z from the previous
  solution, s from the new bounds, s >= 0.1, z >= 1e-4, collision rows mapped
  by (sample, pair)), as a device-side warm-start mode of the solver (API
  sketch in the Recommendation section).

## 1. The experimental hook

Public C++ API (`native/include/cupiqp/cupiqp.hpp`), Python
`SparseSolver.warm_start_buffers(host=True)` / `solve_warm(fields="xysz", from_host=True)`:

```cpp
struct WarmStartBuffers { double *x, *y, *z_l, *z_u, *z_bl, *z_bu, *s_l, *s_u, *s_bl, *s_bu; int32_t n, p, m, num_hl, num_hu, num_xl, num_xu; };
enum WarmStartField : uint32_t { kWarmX = 1, kWarmY = 2, kWarmS = 4, kWarmZ = 8, kWarmAll = 15 };
WarmStartBuffers SparseSolver::warm_start_buffers(bool host = false);  // device or pinned host staging
void SparseSolver::solve_warm(uint32_t fields = kWarmAll, bool from_host = false);
```

- The staging buffers live in user (unscaled) coordinates and use the same
  layout as `ResultView`. They are allocated on the first
  `warm_start_buffers()` call (so a solver that never warm starts allocates
  nothing extra) and keep their addresses afterwards.
- `solve_warm` runs a different prelude (`warm_initial_guess` in
  `sparse_solver.cu`). It optionally copies the pinned host start to the
  device, copies it into a scratch buffer and Ruiz-scales it there
  (`RuizEquilibration::scale_solution`). If not all of x, y, s and z are
  selected, it runs the default initial guess first. It then copies in the
  selected parts, sets s and z to zero on infinite bounds, computes mu = s'z /
  #finite and sets the proximal centers to the start. Residuals and the
  termination check follow, as in the default prelude.
- The whole solve stays one CUDA graph: one graph is cached per (fields,
  from_host) pair and captured on first use. Steady-state warm solves are a
  single graph launch with no host sync and no allocation (C++ test
  `WarmStart.RepeatedWarmSolvesAreOneGraphLaunchWithoutSyncOrAllocation`).
- The default path is unchanged. `solve()` still launches the same
  `solve_graph_` with the same captured content; the only addition is one host
  branch. Cold solves of all 101 QPs gave bitwise-identical x with the a9f74da
  build and the hook build (identical SHA-256 over all x; the x also equals
  the recorded `cupiqp_x` bitwise, and the iteration counts equal the recorded
  ones). In three alternating rounds the per-run median cold solve time was
  12.02-12.10 ms on both builds (`results/.../default_path/`).
- Cost of a warm solve outside the IPM iterations: a least-squares fit of
  solve time against iterations gives a fixed cost of 0.14 ms for warm
  solves (including the ~200 KB pinned host-to-device copy of the start)
  against 0.51 ms for cold solves, because a full start skips the initial KKT
  factorization. Each iteration costs 0.40 ms either way.
- Tests: `native/tests/cpp/ipm/solver/test_warm_start.cu` (5 tests: the
  solver's own solution gives at most 1 iteration in every graph mode and from
  both host and device; default solve bitwise unchanged after warm solves;
  perturbed QP; bad calls rejected; no sync and no allocation) and
  `native/tests/python/ipm/solver/test_warm_start.py` (3 tests). Full suite on
  `build/native-ws`: ctest 117/117, native pytest 1087 passed, upstream
  `test_sparse_solver.py` 7 passed, 3 skipped.

## 2. Method

- `native/bench/warmstart/ws_common.py` loads a QP the way the producer
  handed it to cuPIQP: P is the upper triangle of H mirrored, G = C, and the
  bounds are clamped to +/-1e20. It uses eps_abs = eps_duality_gap_abs = 1e-6,
  relative tolerances 0 and max_iter 250. There is one solver per run, and
  values are replaced through the pinned host input buffers and
  `update_staged(from_host=True)`. The KKT check is computed on the host in
  user coordinates: |Ax - b|, the inequality violation, |Px + c + A'y +
  G'(z_u - z_l)|, |primal obj - dual obj| and z >= 0, all <= 1e-6.
- `run_study.py` times every solve as wall time on CLOCK_MONOTONIC_RAW, from
  the launch to the stream sync (median of 3 repeats after one untimed solve
  of the same start). The time covers the start's host-to-device copy but not
  the host-side NumPy construction of the start; a production version would
  build the start in one device kernel. The distance from the cold solution
  is max |x - x_cold|.
- Two modes. `chain` (the deployment case): the state of QP k is the
  strategy's own solution of QP k. `coldprev`: the state of QP k is the cold
  solution of QP k. Both modes give the same numbers to within a few percent
  (`results/warmstart-20261104/{chain,coldprev}/summary_*.md`). The tables
  below show `chain`.
- Tuning: the safeguard levels were tuned on the place run only
  (`sweep_tune*.sh`, results in `results/.../tune/`), so the pick run is held
  out. The sweeps ran before the (sample, pair) collision map existed and used
  raw labels; on the place run the two maps give identical iteration counts.

## 3. Strategies

All strategies with x = 0 compute the slacks from the QP k+1 data at x = 0.
"Clamp" means s := max(s, s_min) and z := max(z, z_min) on finite sides.
Duals of rows of QP k+1 that have no match in QP k start at z_min.

| name | start |
|---|---|
| `cold` | default PIQP initial guess (eq. 12 of Schwan et al. 2023) |
| `cold_settings@...` | default initial guess, only settings changed (options that exist today) |
| `full_prev` | control: x_k, y_k, s_k, z_k of QP k as they are; clamp 1e-4 |
| `full_prev_xdata` | control: x_k, y_k, z_k; s recomputed on QP k+1 at x_k; clamp s 0.1, z 1e-4 |
| `x0_prev_slack` | x = 0, y_k, s_k and z_k of QP k (the assumption says s_k is about the slack of QP k+1 at 0); clamp 1e-4 |
| `x0_duals:s_min,z_min` | **x = 0, y_k, z_k, s from QP k+1 bounds; clamp** |
| `...,map=label / position` | same, with collision rows mapped by raw label / by row index |
| `...@rho=1e-8,delta=1e-6` | same, with lower initial proximal parameters |
| `x0_duals_center` | x0_duals with clamp 1e-4, then z moved so that s_i z_i lies in [0.1 mu0, 10 mu0], mu0 = 1e-4 (Gondzio-style complementarity band; Gondzio 1998, "Warm start of the primal-dual method applied in the cutting-plane scheme") |
| `active_set` | rows with z_k > 1e-6: s = 1e-3, z = max(z_k, 1e-2); other rows: s = max(s, 1e-2), z = 1e-3 |
| `default_xy_prev_z` | default initial guess for x, y, s; only z from QP k (fields = "z") |
| `gg_data:back=8` | Gondzio and Grothey (2002, "Reoptimization with the primal-dual interior point method", SIAM J. Optim. 13(3)): start from an earlier, better-centered iterate of QP k, here the state after N_k - 8 iterations (obtained by an untimed truncated re-solve); x = 0, s from data; clamp s 1e-2, z 1e-8 |

## 4. Results (chained, all 99 transitions)

Time is the median wall time of one solve on the RTX 4070 SUPER (cold: 12.0 ms).
Fail = status other than Solved. All warm strategies pass the 1e-6 KKT check
on every QP, and their max |x - x_cold| is at most 6.6e-6, which is the
solver tolerance.

| strategy | iter median | iter mean | iter max | ms median | ms mean | ms max | speedup | fail | KKT pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cold | 29 | 28.8 | 33 | 12.03 | 11.98 | 13.64 | 1.00 | 0 | 99/99 |
| cold_settings@rho=1e-8,delta=1e-6 | 29 | 28.9 | 33 | 12.03 | 12.02 | 13.65 | 1.00 | 0 | 99/99 |
| cold_settings@rho=1e-4,delta=1e-2 | 29 | 28.6 | 31 | 12.03 | 11.92 | 12.85 | 1.00 | 0 | 99/99 |
| cold_settings@eps_abs=1e-5,eps_duality_gap_abs=1e-5 | 28 | 28.2 | 32 | 11.69 | 11.75 | 13.25 | 1.03 | 0 | **36/99** |
| full_prev | 12 | 13.4 | 37 | 5.01 | 5.55 | 15.08 | 2.40 | 0 | 99/99 |
| full_prev_xdata (s 0.1, z 1e-4) | 9 | 10.1 | 26 | 3.79 | 4.23 | 10.64 | 3.17 | 0 | 99/99 |
| x0_prev_slack | 12 | 13.4 | 40 | 5.00 | 5.57 | 16.30 | 2.41 | 0 | 99/99 |
| x0_duals (s 1e-4, z 1e-4) | 12 | 13.4 | 37 | 5.02 | 5.54 | 15.10 | 2.40 | 0 | 99/99 |
| **x0_duals (s 0.1, z 1e-4)** | **10** | **10.0** | **25** | **4.17** | **4.19** | **10.25** | **2.89** | 0 | 99/99 |
| x0_duals (s 0.1, z 1e-4), map=label | 10 | 12.6 | 25 | 4.18 | 5.25 | 10.26 | 2.88 | 0 | 99/99 |
| x0_duals (s 0.1, z 1e-4), map=position | 10 | 10.4 | 25 | 4.18 | 4.34 | 10.25 | 2.88 | 0 | 99/99 |
| x0_duals (s 0.1, z 1e-4) @rho=1e-8,delta=1e-6 | 10 | 10.0 | 25 | 4.17 | 4.18 | 10.24 | 2.89 | 0 | 99/99 |
| x0_duals_center (mu0 1e-4) | 11 | 13.1 | 41 | 4.57 | 5.44 | 16.71 | 2.63 | 0 | 99/99 |
| active_set | 10 | 11.0 | 24 | 4.56 | 4.60 | 9.85 | 2.64 | 0 | 99/99 |
| default_xy_prev_z | 12 | 13.6 | 25 | 5.29 | 5.94 | 10.74 | 2.27 | 0 | 99/99 |
| gg_data back=8 (needs an iterate snapshot) | 9 | 9.8 | 23 | 3.77 | 4.10 | 9.44 | 3.19 | 0 | 99/99 |

Per run (chained), for the recommended `x0_duals (s 0.1, z 1e-4)`. The place
run was used for tuning; the pick run was held out.

| run | cold iter med / mean | warm iter med / mean / max | warm k = 1 | warm k >= 10 med / max | ms cold -> warm (median) |
|---|---|---|---:|---|---|
| 29-79b3-place (36 transitions) | 29 / 29.2 | 8 / 8.5 / 17 | 17 | 7 / 11 | 12.03 -> 3.36 (3.6x) |
| 55-5d59-pick (63 transitions) | 29 / 28.5 | 10 / 10.8 / 25 | 19 | 10 / 15 | 12.03 -> 4.19 (2.9x) |

Full per-run tables, including the time-per-iteration fit:
`native/bench/results/warmstart-20261104/chain/summary_per_run.md`.

Observations:
- The worst cases are the first transitions (k = 1 and 2), where the first
  SQP step changes delta_t by about 12% and the active set changes most.
  From there the iteration counts fall as the SQP converges.
- On the place run, starting x at 0 or at x_k makes no difference: the
  iteration sequences of `full_prev` and `x0_prev_slack` are identical. With
  recomputed slacks and the same clamps, `full_prev_xdata` and `x0_duals`
  differ by 0.1 iterations on average. What x = 0 buys is simplicity, not
  speed: the slack at x = 0 is the bound itself, so building the start needs
  no product G x.
- The safeguard levels decide the result (sweep in `tune/`). Clamps that are
  too small (1e-8 or 1e-6 on both sides) give a median of 10 but tails of 78
  to 116 iterations at k = 1. Clamps that are too large (1 on both sides) give
  15 to 16. The plateau is s_min in [3e-2, 3e-1] with z_min in [1e-4, 1e-3].
- Lowering rho_init / delta_init does not change the iterations, cold or warm.
  The proximal centers already start at the warm point.
- Gondzio-Grothey (start from an earlier iterate of QP k) is the best on
  average (mean 9.8, max 23), but it needs the solver to snapshot an iterate
  during the solve of QP k. Its advantage over `x0_duals` is small, so it is
  not worth that machinery now.

## 5. The assumption "QP k+1 is derived from the solution of QP k"

Script: `native/bench/warmstart/assumption.py`. Output:
`results/warmstart-20261104/assumption.{md,jsonl}`.

- **Re-solve equals the record.** The cold re-solve of every QP reproduces
  the recorded `cupiqp_x` and `cupiqp_ineq_duals` bitwise, with the same
  iteration counts. The captured QPs are therefore exactly the ones whose
  solution TrajOpt applied.
- **Full step, linearized at trajectory + x*_k.** For each row group, fit the
  bound change of QP k+1 against C_k x*_k, so that bound_{k+1} = bound_k -
  alpha C_k x*_k:
  - Joint-position rows (linear in the trajectory): alpha = 1.0000 on every
    transition, with residual <= 2e-15. QP k+1 is the QP at the trajectory
    after applying the full step x*_k; there is no line-search damping.
  - EEF and collision rows (nonlinear): median alpha 0.995-1.001, median
    error 1e-5 to 1e-7. The max error (about 1 for EEF, 0.1 for collision) is
    linearization error at the first transitions.
  - Velocity and acceleration rows: median alpha 0.92-1.0, but as low as
    0.02. These rows scale with delta_t, so they are strongly nonlinear in the
    step whenever delta_t moves.
  - The delta_t row bounds the step itself (lb -0.2, ub 0.1 every QP), so it
    does not shift (alpha = 0). That is consistent with the assumption.
- **Slack prediction.** Because of the above, the slack of QP k at its
  solution predicts the slack of QP k+1 at x = 0 exactly for the linear rows
  and to the linearization error for the others. That is why `x0_prev_slack`
  (s_k) and `x0_duals` (bounds of QP k+1) behave the same at equal clamps.
- **Active set.** On average 94% (place) and 90% (pick) of the rows active in
  QP k+1 (|z| > 1e-4) were active in QP k. The minimum is 54% / 26% at the
  first transitions. The relative change of the inequality duals is 5% (place)
  and 3% (pick) at the median, with a maximum of 0.96-1.16 at the first
  transitions. The relative change of y is under 1% at the median.
- **Where it breaks: collision labels.** The raw `ineq_labels` of collision
  rows embed the sample time (`t=0.549s`), and the sample times scale with
  delta_t. In the pick run, only 447 of 752 collision labels of QP k+1 occur
  in QP k (place: 679). Matched by raw label, the dual change jumps to 0.59 at
  the median. Keying a collision row by (sample index in the block, link pair)
  finds 750 of 752 rows in both runs and brings the dual change back to 3%.
  Slot reassignment, where the same sample has a different pair in a slot,
  is real but rare: mapping by row index instead of (sample, pair) matches 161
  instead of 164 active collision rows. It costs 0.4 iterations on average in
  the pick run (10.4 vs 10.0 overall). With raw labels the pick run needs 16
  iterations at the median instead of 10.

The assumption holds, but the dual transfer must not use the raw collision
labels.

## 6. Recommendation

Adopt `x0_duals` with s_min = 0.1 and z_min = 1e-4 as a solver warm-start
mode. Do not adopt the parameter-only options; they bring nothing.

API sketch, a production version of the hook:

```cpp
struct WarmStartOptions {
  double s_min = 0.1, z_min = 1e-4;   // clamps on finite sides
  bool keep_y = true;                  // y from the previous solve
};
// Uses the previous solution still held in the solver's result buffers
// (user coordinates after finalize): x = 0, y = y_prev,
// z_l/u = max(z_prev[row_map], z_min), s_u = max(h_u, s_min),
// s_l = max(-h_l, s_min), all in one elementwise kernel inside the solve graph.
// row_map: optional device int32 array (m), row of the previous QP for each
// row of the new one, -1 for new rows (start at z_min). nullptr = identity.
void SparseSolver::set_row_map(const int32_t* row_map);  // stable address, staged like the inputs
void SparseSolver::solve_warm_from_previous(const WarmStartOptions& opt = {});
```

- No host traffic is needed. The previous y and z are already in the
  solver's result buffers. The new slacks at x = 0 are the new bounds, which
  are already on the device after `update_staged()`. The row map is constant
  within an SQP run except for the collision block, and the caller (TrajOpt)
  already knows its pair-to-slot assignment.
- The caller must provide the row map for the collision block, keyed by
  (sample, pair), not by the raw label text. Identity (by position) is an
  acceptable fallback: it cost 0.4 iterations on average here.
- Keep the first QP of each SQP run cold. Optionally fall back to cold when
  the previous solve did not report Solved.
- Expected gain on this GPU: 12.0 ms to about 4.2 ms per QP (2.9x at the
  median, 3.6x on the place run), with the worst case dropping from 33 to 25
  iterations.

## 7. Not done / limits

- A device-side strategy kernel (the API above) is not implemented. In the
  study the start is built in NumPy, outside the timed interval. The timed
  interval includes the host-to-device copy and the scaling of the start.
- Gondzio-Grothey needs an in-solve iterate snapshot. It was emulated by a
  truncated re-solve and not implemented in the solver.
- Two runs from one robot and scene. The safeguard levels were tuned on one
  run and held out on the other, but other problem families (MPC) may need
  different clamps.
- Times are from the RTX 4070 SUPER, not the producer's A1000 (cold 31 ms
  there). The iteration counts transfer; the absolute milliseconds do not.

## Files

- Hook: `native/include/cupiqp/cupiqp.hpp`, `native/src/ipm/solver/sparse_solver.cu`,
  `native/python/bindings/bind_solver.cpp`, `native/python/cupiqp_native/_solver.py`.
- Tests: `native/tests/cpp/ipm/solver/test_warm_start.cu`, `native/tests/python/ipm/solver/test_warm_start.py`.
- Study: `native/bench/warmstart/` (`ws_common.py`, `ws_strategies.py`, `run_study.py`,
  `summarize.py`, `assumption.py`, `default_path_cost.py`, `run_final.sh`,
  `run_default_path_cost.sh`, `sweep_tune{,2,3}.sh`).
- Results: `native/bench/results/warmstart-20261104/` (`chain/`, `coldprev/`, `tune/`,
  `default_path/`, `assumption.{md,jsonl}`; solution `.npz` files are not tracked).

Reproduce (from the repository root, after building `build/native-ws`):

```bash
native/bench/warmstart/run_final.sh native/bench/results/warmstart-20261104/chain --chain
native/bench/warmstart/run_final.sh native/bench/results/warmstart-20261104/coldprev
.venv/bin/python native/bench/warmstart/assumption.py --cold-dir native/bench/results/warmstart-20261104/chain --out native/bench/results/warmstart-20261104
.venv/bin/python native/bench/warmstart/summarize.py --per-run native/bench/results/warmstart-20261104/chain/records.jsonl
```

---

# Phase 2: the warm start built on the device (2026-11-04)

Brief: `native/docs/briefs/warmstart_device.md`. Same branch, GPU, data and
solver settings as phase 1 (eps_abs = eps_duality_gap_abs = 1e-6, relative
tolerances 0, max_iter 250). Results: `native/bench/results/warmstart-device-20261104/`.

## Summary

- The phase-1 strategy now runs entirely on the GPU, inside the one CUDA
  graph of the solve: `set_warm_start(options)` once, then
  `solve_warm_start()` per QP. Every steady-state solve is one graph launch
  with zero host syncs and zero captures (counted on every recorded solve of
  the experiments and asserted in a C++ test, which also checks that no
  memory is allocated).
- Host-to-device traffic of the feature per QP: 0 bytes with identity
  matching or device-side keys; 8 bytes per keyed row with host keys (6,016 B
  for the 752-row collision block with fixed blocks of 16 rows; 9,024 B with
  int32 group ids; 51,080 B if all 6,385 rows are keyed). The previous y and z
  never leave the GPU.
- It reproduces phase 1 exactly. Iterations are identical on 99 of 99
  transitions for cold, identity matching, (sample, pair) keys and
  raw-label keys: 10 / 10.0 / 25 (median / mean / max) with keys,
  10 / 10.4 / 25 with identity, against 29 / 28.8 / 33 cold. Median solve time
  goes from 12.02 ms to 4.13 ms (2.9x). Every solve is Solved and passes the
  KKT check at 1e-6.
- The default `solve()` is unchanged: the SHA-256 over x of all 101 cold
  solves is identical to the HEAD build (b0bdc64) and to phase 1, and the
  medians are the same within 0.01 ms in three alternating rounds.
- The cost is small. The warm prelude is about 0.34 ms cheaper than the
  cold one (no initial factorization). Matching with fixed blocks adds
  under 5 us per solve, including the read of the host keys (within noise).
  Group ids sorted with CUB add about 30 us, keying all rows about 70 us, and
  the ring buffer 1-2 us per IPM iteration.
- Robustness: corrupted matching never causes a failure, and it never costs
  more than cold. With wrong keys on half of the collision rows (as
  realized), the mean is 14.4-14.9 iterations; with every row unmatched it is
  18.2 (max 26). Unmatched rows are worse than wrong rows, and identity
  matching (10.4) beats any heavily corrupted key set.
- Ring buffer (Gondzio-Grothey, causal): no gain. The mean is 9.93-10.06
  against 10.00 for the end-of-solve iterate for every K in {4, 8, 12} and
  every offset up to K/2. Going back the full K lowers the worst case
  (25 -> 17 with K = 12) but raises the pick-run median from 10 to 13.
- MPC: s_min = 0.1 and z_min = 1e-4 carry over. They are the best (or tied
  best) of the 11 settings tried: 24 -> 10 iterations (mean 23.6 -> 10.1)
  at eps 1e-6, and 24 -> 11 (24.6 -> 11.1) at eps 1e-8. Time drops from
  7.75 to 3.26 ms (2.4x). 48 of 48 transitions are Solved with KKT at 1e-6.
- Recommendation: ship `solve_warm_start` with the defaults (s_min 0.1,
  z_min 1e-4, previous y, cold fallback on). Have TrajOpt key the collision
  block by link-pair id with `key_group_size = 16` (one block per sample),
  passed from pinned host memory. Use identity matching where no keys exist.
  Leave the ring buffer off.

## 1. Design and API

Public C++ API (`native/include/cupiqp/cupiqp.hpp`); Python
`SparseSolver.set_warm_start(**fields)`, `row_key_buffers(host)`,
`solve_warm_start(keys_from_host)`, `reset_warm_start()`,
`cupiqp_native.WarmStartOptions`, `cupiqp_native.RowMatch`.

```cpp
enum class RowMatch : int32_t { Identity = 0, Keys = 1 };
struct WarmStartOptions {
  double s_min = 0.1, z_min = 1e-4, z_unmatched = 1e-4;
  bool previous_y = true, cold_if_not_solved = true;
  int32_t snapshot_slots = 0, snapshot_back = 0;          // ring buffer (off)
  RowMatch row_match = RowMatch::Identity;
  bool row_groups = false;  int32_t key_group_size = 0;   // groups: ids or fixed blocks
  int32_t key_row_begin = 0, key_row_count = -1;           // keyed inequality rows
};
struct RowKeyBuffers { int64_t* keys; int32_t* groups; int32_t count; };
void SparseSolver::set_warm_start(const WarmStartOptions& = {});  // allocates; outside the solve
RowKeyBuffers SparseSolver::row_key_buffers(bool host = true);    // pinned host or device staging
void SparseSolver::solve_warm_start(bool keys_from_host = true);
void SparseSolver::reset_warm_start();                           // next one is cold
```

Code: `native/src/ipm/solver/warm_start_device.{hpp,cu}` (class
`WarmStartDevice`: buffers and kernels), wired into the solve in
`native/src/ipm/solver/sparse_solver.cu` (`previous_solve_start`,
`previous_start`, the `ring_store` call at the end of `run_one_iteration`,
the `record_previous` call in `solve_finalize`).

What runs inside the one solve graph:

1. **Keys** (only with `RowMatch::Keys`). The first kernel reads the keys of the new QP
   (and the group ids) from the pinned staging. It reads them over the bus
   through the mapped device address of the pinned buffer
   (`cudaHostGetDevicePointer`), so there is no memcpy node. On this machine a
   6 KB `cudaMemcpyAsync` node in the graph cost about 50 us; the kernel read
   costs a few us (measured below, switch `CUPIQP_WARM_KEYS_MEMCPY=1`). With
   `keys_from_host = false` the device staging is used as is (0 bytes).
2. **Row match** `match[i]`, the previous keyed row of keyed row i, or -1.
   The row itself is preferred when its previous key is equal; otherwise:
   - `key_group_size = g`: a linear search of its block of g rows (one
     kernel; the brief's "linear search within a small group");
   - `row_groups` (any group-id layout) or no groups: CUB
     `DeviceRadixSort::SortPairs` of the previous (group, key) pairs (a
     decomposer, group most significant, stable, so the lowest row wins among
     duplicates), then a binary search per row;
   - no keys: identity, no kernel.
   The current keys then become the previous keys (one pack kernel).
3. **Start.** Two graph variants, chosen on the host without a sync. The first
   `solve_warm_start()` of a sequence (after `set_warm_start`,
   `reset_warm_start`, `solve()` or `solve_warm()`) captures the default
   initial guess. Later ones capture a 1-thread decision kernel that writes
   `cold = (prev status != Solved)` and `warm = !cold` to the device, then two
   conditional IF nodes (`capture_if`): the cold body is the default initial
   guess, the warm body is one elementwise kernel (`build_start_kernel`) plus
   mu and the proximal centers. The start kernel works directly in the
   scaled coordinates of the new QP: x = 0; y = y_prev D^-1 c;
   s = max(scaled bound, s_min D); z = max(z_prev[match], z_min) D^-1 c
   (`z_unmatched` for match -1), with s = z = 0 on infinite bounds. Box bounds
   use identity and delta_b. The slack at x = 0 is the scaled bound itself, so no
   product G x is needed.
4. **Ring buffer** (optional). With `snapshot_slots = K > 0`, one small kernel
   at the end of each IPM iteration copies (y, z, s) of the current
   iterate into slot `iter % (K + 1)` (the slot comes from the device iteration
   counter, so one captured node serves all iterations).
5. **Record.** In the finalization, before the solution is unscaled (still
   in the scaling of this QP), one kernel picks the end-of-solve iterate or
   iterate N - back (if N - back >= 1), unscales its y and z into the
   previous-solve buffer (user coordinates; Ruiz scaling may change on the
   next update) and keeps the final status for the next decision.

All buffers are allocated in `set_warm_start()` (which syncs, invalidates
the feature's graphs and makes the next solve cold). `solve()`, its graphs
and `solve_prelude` for prev_mode 0 are untouched. The feature also works in
`GraphMode::None` and `IterationGraphHostLoop`, where a conditional IF costs
one host sync, as the loops there already do.

Tests: `native/tests/cpp/ipm/solver/test_warm_start_device.cu` (7 tests, all
three graph modes. The first solve is cold and bitwise equal to `solve()`,
and `solve()` stays bitwise unchanged afterwards. On the increment form of a
QP (x0 shifted out) the warm start needs at most 3 iterations, and on a
perturbed one fewer than cold. A previous MaxIterReached gives a cold start
bitwise equal to `solve()`. Key matching recovers a row permutation for
global keys, group ids and fixed blocks, from host and device. There is one
launch, no sync, no capture and no allocation in steady state. The ring buffer
and the graph modes agree bitwise. Bad options are rejected),
`native/tests/python/ipm/solver/test_warm_start_device.py` (3 tests). Full
suite on `build/native-ws`: ctest 124/124 (3 disabled timing tests not run),
native pytest 1090 passed, upstream `test_sparse_solver.py` 7 passed, 3 skipped.
compute-sanitizer memcheck and initcheck report 0 errors for a keyed
chain with the ring buffer in `GraphMode::None`. In graph modes the sanitizer
cannot be used on this machine: it fails on every test that captures a
conditional node, including the pre-existing `SpmvTest.InsideConditionalWhileBody`.

## 2. Experiment 1: reproduction of phase 1 (`main/`)

Chains over all QPs of each run, the first QP cold. Four passes per configuration,
the first a warm-up; time is the per-QP median of the other three (CLOCK_MONOTONIC_RAW,
launch to stream sync, keys already written to pinned memory). Aggregates
over the 99 warm-started transitions.

| config | iter med | iter mean | iter max | ms med | ms mean | ms max | speedup | fail | KKT pass | launches / syncs per solve |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cold (`solve()`) | 29 | 28.80 | 33 | 12.02 | 11.98 | 13.62 | 1.00 | 0 | 99/99 | 1 / 0 |
| identity matching | 10 | 10.39 | 25 | 4.13 | 4.30 | 10.18 | 2.91 | 0 | 99/99 | 1 / 0 |
| **keys (sample, pair), fixed blocks of 16** | **10** | **10.00** | **25** | **4.13** | **4.15** | **10.18** | **2.91** | 0 | 99/99 | 1 / 0 |
| keys (sample, pair), group ids + sort | 10 | 10.00 | 25 | 4.16 | 4.17 | 10.21 | 2.89 | 0 | 99/99 | 1 / 0 |
| all rows keyed by (sample, pair) / label hash | 10 | 10.00 | 25 | 4.20 | 4.21 | 10.25 | 2.86 | 0 | 99/99 | 1 / 0 |
| all rows keyed by raw-label hash | 10 | 12.64 | 25 | 4.20 | 5.29 | 10.25 | 2.86 | 0 | 99/99 | 1 / 0 |
| identity, s_min = 1e-4 (naive clamp) | 13 | 13.54 | 37 | 5.33 | 5.56 | 15.02 | 2.25 | 0 | 99/99 | 1 / 0 |
| keys (group ids), y = 0 | 10 | 10.42 | 25 | 4.16 | 4.33 | 10.21 | 2.89 | 0 | 99/99 | 1 / 0 |

Per run (keys, fixed blocks): place 29 / 29.25 -> 8 / 8.53 / 17 iterations,
12.02 -> 3.35 ms (3.6x); pick 29 / 28.54 -> 10 / 10.84 / 25, 12.07 -> 4.14 ms
(2.9x). Max |x - x_cold| <= 3.7e-6 (solver tolerance) everywhere.

Comparison with phase 1, per QP: the iteration counts of cold, identity
(phase-1 `map=position`), (sample, pair) keys (`x0_duals`) and raw-label keys
(`map=label`) are identical on 99 of 99 transitions. The naive clamp s_min =
z_min = 1e-4 differs on 6 transitions of the pick run (k = 35-47, by 1-4
iterations; mean 13.54 vs 13.4). The device kernel clamps in scaled
coordinates, max(scaled bound, s_min D), whereas phase 1 clamped
max(bound, s_min) in user coordinates and then scaled. The two agree up to
rounding of the cumulative Ruiz factors, and with the tiny clamp the chain
is sensitive enough for that rounding to change a few counts. Solve times are
within 0.05 ms of phase 1 at the median. Phase 1 timed a 200 KB start upload
and the default scaling, while phase 2 runs the start kernel and the
matching; neither matters at this resolution.

The fit of time against iterations per QP (`main/summary.md`) gives
0.39-0.41 ms per IPM iteration in all modes and a fixed cost of 0.65 ms
for cold against 0.05-0.06 ms for identity or block keys, 0.09 ms with group
ids and 0.14-0.21 ms with all rows keyed.

## 3. Experiment 2: matching methods

The table above answers it. What matters is whether the keys find the
right row, not how they are passed. Every scheme that finds the
(sample, pair) row gives 10.00. Identity costs 0.39 iterations on average
(10.39, pick run only: 11.46 vs 10.84). Raw labels, whose collision text contains the
sample time, mismatch 29% of the collision rows (pick 40%) and cost 2.6
iterations on average (pick run: 15.0 vs 10.8). Rows outside the collision
block do not need keys; keying them (all 6,385 rows) changes nothing.
Among the equivalent schemes, fixed blocks are cheapest: no group ids
(6,016 B instead of 9,024 B per QP) and no sort.

## 4. Experiment 3: robustness to bad matching (`robust/`)

Keys (sample, pair) with group ids. On a fraction f of the collision rows of
every QP after the first, the key is replaced by either:
- a fresh key that no other QP uses (**unmatched**: the row starts at
  `z_unmatched`), or
- the committed key of another row of the same sample in the previous QP
  (**mismatch**: the row takes the dual of a wrong row).

There were 3 seeds per point, on both runs, for 297 warm solves per point.
Because the keys of QP k are also the previous keys of QP k+1, a corrupted
row also breaks the match of its successor. The realized fraction of
collision rows whose device match differs from the (sample, pair) truth was
measured by reading back the device match and is about 2f - f^2. At f = 100%
mismatch, the copied keys often coincide with the true ones, so the
realized rate falls to 43%.

| corruption | nominal f | realized wrong | iter med | iter mean | iter max | fail | KKT pass |
|---|---:|---:|---:|---:|---:|---:|---:|
| none (cold reference: 29 / 28.80 / 33) | 0% | 0.0% | 10 | 10.00 | 25 | 0 | 99/99 |
| unmatched | 10% | 19.1% | 12 | 12.46 | 25 | 0 | 297/297 |
| unmatched | 30% | 50.3% | 15 | 14.86 | 26 | 0 | 297/297 |
| unmatched | 60% | 83.4% | 19 | 17.25 | 26 | 0 | 297/297 |
| unmatched | 100% | 99.8% | 20 | 18.23 | 26 | 0 | 297/297 |
| mismatch | 10% | 19.2% | 11 | 12.06 | 25 | 0 | 297/297 |
| mismatch | 30% | 49.9% | 15 | 14.38 | 25 | 0 | 297/297 |
| mismatch | 60% | 79.9% | 17 | 16.07 | 26 | 0 | 297/297 |
| mismatch | 100% | 42.7% | 13 | 12.96 | 25 | 0 | 297/297 |

Per run: place goes from 8.5 (clean) to 11.7 (all unmatched), and pick from 10.8
to 22.0 (median 22, max 26). Curve:
- Bad matching only costs speed: there are no failures, the KKT check passes
  everywhere, and the worst case (26) stays under cold (33).
- The cost rises steeply at first (+2.5 iterations at 19% realized) and then
  saturates (18.2 at 100% unmatched). Almost all of it is on the pick run,
  where the collision duals are active.
- A wrong dual is slightly better than none: at the same realized rate,
  mismatch is 0.4-1.2 iterations cheaper than unmatched. So
  `z_unmatched = z_min = 1e-4` is a poor guess for the collision rows of this
  problem, and identity, which takes the dual of whatever pair now sits in the slot,
  beats losing the match: 10.39 against 18.23 with every row unmatched. If keys
  are unreliable, use identity, not keys that miss.

## 5. Experiment 4: ring-buffer snapshot (`ring/`)

Keys (sample, pair) with group ids, s_min 0.1, z_min 1e-4, except where noted.
"back b" means the next QP starts from iterate N - b of this solve (b = 0:
the end-of-solve iterate, i.e. no change). Four passes.

| K | back | iter med | iter mean | iter max | ms med | ms mean |
|---:|---:|---:|---:|---:|---:|---:|
| - | 0 (no ring) | 10 | 10.00 | 25 | 4.18 | 4.22 |
| 4 | 0 | 10 | 10.00 | 25 | 4.20 | 4.23 |
| 4 | 1 | 9 | 9.99 | 25 | 3.81 | 4.20 |
| 4 | 2 | 9 | 10.06 | 26 | 3.80 | 4.23 |
| 4 | 4 | 9 | 9.93 | 24 | 3.86 | 4.20 |
| 8 | 1 | 9 | 9.99 | 25 | 3.81 | 4.20 |
| 8 | 4 | 9 | 9.93 | 24 | 3.80 | 4.18 |
| 8 | 8 | 10 | 10.41 | 24 | 4.21 | 4.39 |
| 12 | 1 | 9 | 9.99 | 25 | 3.81 | 4.20 |
| 12 | 6 | 9 | 10.04 | 25 | 3.81 | 4.22 |
| 12 | 12 | 10 | 10.31 | 17 | 4.21 | 4.35 |
| 8 | 8, s_min 1e-2, z_min 1e-8 (phase-1 gg clamps) | 10 | 10.31 | 26 | 4.20 | 4.35 |
| 8 | 4, s_min 1e-2, z_min 1e-8 | 9 | 10.65 | 33 | 3.80 | 4.46 |

All are Solved with KKT at 1e-6, 1 launch, 0 syncs. Verdict: no gain worth the
feature. The mean is flat (9.93-10.06) for every K and offset up to K/2.
The median drops from 10 to 9 only because the pooled distribution sits at
the 9/10 boundary; per run, the medians are unchanged (8 and 10). Going back
the full K helps the worst transition (max 25 -> 17 with K = 12) but
costs on the converging tail (pick median 10 -> 13). The phase-1
`gg_data back=8` result (mean 9.8, max 23) was not causal: it used an
iterate of a cold re-solve of QP k. The causal snapshot of the warm solve is
shorter (about 10 iterations), so N - 8 is nearly the start itself. The ring
costs 1-2 us per IPM iteration (section 7), and `snapshot_back = 0`
gives results bitwise equal to no ring (C++ test). Keep it off, and keep the
code only as an experimental option.

## 6. Experiment 5: MPC replay dataset (`mpc_eps1e-6/`, `mpc_eps1e-8/`)

`native/bench/data/mpc` (n = 169, p = 64, m = 2521, one pattern):
16 trajectories x 4 control steps, the steps 13-24 control periods apart.
Each trajectory is a chain (its first QP cold, three warm), for 48
transitions. Identity matching. P, A, C and bounds come from the bundle as
the replay harness uses them (bounds clamped to +/-1e20). Four passes.

| config (eps 1e-6) | iter med | iter mean | iter max | ms med | speedup | fail | KKT pass |
|---|---:|---:|---:|---:|---:|---:|---:|
| cold | 24 | 23.58 | 27 | 7.75 | 1.00 | 0 | 48/48 |
| **s_min 0.1, z_min 1e-4 (default)** | **10** | **10.06** | **13** | **3.26** | **2.38** | 0 | 48/48 |
| s_min 1e-3 | 11 | 11.21 | 14 | 3.58 | 2.17 | 0 | 48/48 |
| s_min 1e-2 | 10 | 10.58 | 13 | 3.33 | 2.33 | 0 | 48/48 |
| s_min 1 | 12 | 12.06 | 18 | 3.89 | 1.99 | 0 | 48/48 |
| z_min 1e-6 | 11 | 11.46 | 15 | 3.58 | 2.17 | 0 | 48/48 |
| z_min 1e-2 | 11 | 11.00 | 15 | 3.57 | 2.17 | 0 | 48/48 |
| s_min 1e-2, z_min 1e-2 | 12 | 11.85 | 16 | 3.90 | 1.99 | 0 | 48/48 |
| s_min 1, z_min 1e-2 | 10 | 10.46 | 14 | 3.27 | 2.37 | 0 | 48/48 |
| s_min 1e-4, z_min 1e-4 | 11 | 11.69 | 16 | 3.59 | 2.16 | 0 | 48/48 |
| default, y = 0 | 10 | 10.06 | 13 | 3.26 | 2.38 | 0 | 48/48 |
| default, ring K 8 back 8 | 10 | 9.98 | 13 | 3.27 | 2.37 | 0 | 48/48 |

At eps 1e-8 (the replay harness tolerance) the ranking is the same: cold
24 / 24.58 / 28, default 11 / 11.06 / 14, 8.07 -> 3.58 ms (2.26x); every
other setting is equal or worse (`mpc_eps1e-8/summary.md`). The phase-1
clamps carry over: they are the best setting on MPC as well, even though
these QPs are far apart (13-24 control steps) and not SQP increments, and
the plateau is the same shape (too small s_min or z_min costs 1-2
iterations, s_min = 1 costs 2). y = 0 gives identical iterations here,
because the 64 equality rows are almost all explicit zeros (880 of 920
entries of A).

## 7. Experiment 6: cost (`cost/`)

`cost_device.py`, pick run QP 20 (QP 19 as the previous solve),
compute_timings events for GPU time, medians of 300 solves.

Fixed cost: max_iter = 1 (0 is rejected by the settings check), so a solve is
prelude + one IPM iteration + finalization. cold_if_not_solved is off so
that the warm branch is always taken.

| solve (max_iter = 1) | GPU ms | wall ms | H2D bytes / solve (feature) | launches | syncs | captures |
|---|---:|---:|---:|---:|---:|---:|
| cold `solve()` | 0.845 | 0.866 | 0 | 1 | 0 | 0 |
| warm, identity | 0.507 | 0.527 | 0 | 1 | 0 | 0 |
| warm, keys in fixed blocks, host keys | 0.499 | 0.519 | 6,016 | 1 | 0 | 0 |
| warm, keys in fixed blocks, device keys | 0.507 | 0.528 | 0 | 1 | 0 | 0 |
| warm, keys + group ids (CUB sort), host | 0.537 | 0.557 | 9,024 | 1 | 0 | 0 |
| warm, all 6,385 rows keyed (CUB sort), host | 0.579 | 0.599 | 51,080 | 1 | 0 | 0 |

- The warm start is about 0.34 ms cheaper than the cold initial guess
  (no initial factorization or KKT solve).
- Fixed blocks: matching plus reading the host keys costs nothing
  measurable (+-10 us run-to-run noise). The CUB sort of 752 (group, key)
  pairs costs about 30 us, and of 6,385 keys about 70 us.
- Host keys through a `cudaMemcpyAsync` node instead of the in-kernel read
  (`CUPIQP_WARM_KEYS_MEMCPY=1`, `cost/cost_memcpy_node.txt`): +54 us for
  6 KB, +47 us for 9 KB in two copies, +59 us for 51 KB. That latency, not
  the bytes, is why the kernel reads the pinned buffer directly.
- Ring buffer: full solves of the same QP from the same previous state with
  K = 0 and K = 8 (both 11 iterations): +8 us GPU per solve, 0.7 us per IPM
  iteration in the final run (1.0-2.3 us in earlier runs; at the noise level,
  under 0.5% of a 0.4 ms iteration).
- The record of the previous y and z and the start kernel are included in
  the numbers above.
- Host traffic not counted above: none. The previous y and z (and the status
  and the decision flags) stay on the device. The only per-QP host traffic is
  the optional keys.

Kernel-level Nsight Systems traces could not be taken. With
`--cuda-graph-trace=node`, the whole-solve graph aborts with an illegal
address on this machine (nsys 2025.6.3, driver 13.3), also for the plain
`solve()` of the unchanged default path. The same chains run clean without the
profiler and under compute-sanitizer in `GraphMode::None`.

## 8. Recommendation

1. Use `solve_warm_start()` with the defaults for SQP sequences:
   x = 0, previous y, slacks from the new bounds with s >= 0.1, previous z
   with z >= 1e-4, cold start on the first QP and after a failed solve. On this
   GPU: 12.0 -> 4.1 ms median per TrajOpt QP (2.9x), 7.75 -> 3.26 ms per MPC QP
   (2.4x), with no new host syncs and no extra launches.
2. TrajOpt should pass the collision block as keys:
   `row_match = Keys, key_row_begin = 1569, key_row_count = 752,
   key_group_size = 16`, key = link-pair id, written into
   `row_key_buffers(true)` before each solve (6 KB per QP over the bus). If
   the pair ids are already on the GPU, write the device staging instead (0 B).
   Without keys, identity costs 0.4 iterations on average, about 0.15 ms.
3. Do not key rows whose meaning never changes, and do not key by raw labels
   that embed sample times. If keys can be wrong, identity is the safer
   fallback: missed rows cost more than mismatched ones.
4. Leave the ring buffer off; it adds no measurable gain on either dataset.

## 9. Not done / limits

- Kernel-level traces (see section 7); costs are differential wall and
  CUDA-event times instead.
- Batch size 1 only, like the solver.
- `set_warm_start()` re-captures graphs and syncs; it is a setup call.
- The host must not rewrite the pinned key staging until the solve that
  reads it has completed (same contract as the pinned input buffers).
- Times are from the RTX 4070 SUPER under WSL2; the memcpy-node latency in
  particular may differ on native Linux and the producer's A1000.

## Files (phase 2)

- Code: `native/src/ipm/solver/warm_start_device.{hpp,cu}`,
  `native/src/ipm/solver/sparse_solver.cu`, `native/include/cupiqp/cupiqp.hpp`,
  `native/python/bindings/bind_solver.cpp`, `native/python/cupiqp_native/{__init__,_solver}.py`.
- Tests: `native/tests/cpp/ipm/solver/test_warm_start_device.cu`,
  `native/tests/python/ipm/solver/test_warm_start_device.py`.
- Scripts: `native/bench/warmstart/run_device.py` (TrajOpt chains, matching
  modes, corruption, ring), `run_device_all.sh` (stages main, robust, ring),
  `run_mpc.py` + `mpc_common.py` + `run_mpc_all.sh` (MPC), `cost_device.py`,
  `summarize_device.py`.
- Results: `native/bench/results/warmstart-device-20261104/{main,robust,ring,mpc_eps1e-6,mpc_eps1e-8,cost,default_path}/`
  (`records.jsonl` and `summary.md` per experiment; `default_path/cold_{base,hook}_N.json`
  compare the HEAD build b0bdc64 ("base") with this branch ("hook"); `cold_final.json`
  is the final build).

Reproduce (repository root, after building `build/native-ws`):

```bash
native/bench/warmstart/run_device_all.sh all
native/bench/warmstart/run_mpc_all.sh
.venv/bin/python native/bench/warmstart/cost_device.py --out native/bench/results/warmstart-device-20261104/cost/cost.json
.venv/bin/python native/bench/warmstart/summarize_device.py native/bench/results/warmstart-device-20261104/main/records.jsonl --per-run --fit
.venv/bin/python native/bench/warmstart/summarize_device.py native/bench/results/warmstart-device-20261104/robust/records.jsonl --robust
```
