# Warm-start study on consecutive TrajOpt SQP QPs (2026-11-04)

Brief: `native/docs/briefs/warmstart_study.md`. Branch `feature/warmstart-study`
from `native-cuda-port` (a9f74da). GPU: RTX 4070 SUPER (sm_89), CUDA 13.
Data: `native/bench/data/warmstart/cupiqp_warmstart_sequences_20261104/` (2 runs,
37 + 64 QPs, 99 transitions k -> k+1; n = 241, p = 64, m = 6385).

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
