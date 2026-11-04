# Brief: warm-start study on consecutive TrajOpt SQP QPs

User request (2026-11-04): use the captured SQP sequences to find out whether,
and how, cuPIQP can be warm started. The optimization variables are first-order
increments (SQP steps), not zeroth-order values, so x = 0 is already a good
primal start even when warm starting. Working assumption, stated by the user
and to be tested rather than trusted: QP k+1 is the QP derived from the
solution of QP k, so the solution of QP k can seed QP k+1.

## Data

`native/bench/data/warmstart/cupiqp_warmstart_sequences_20261104/` (gitignored,
do not commit). Read its `README.md` first. Key facts:
- 2 runs: `29-79b3-place-6f87a360` (37 QPs) and `55-5d59-pick-d59c9e60` (64 QPs),
  every QP of each run in order.
- n = 241, p = 64 (16 reserved all-zero equality rows), m = 6385 inequality rows
  `lb <= C x <= ub`, nnz(C) = 164465. Pattern identical within a run.
- `iterations/iter_NNN/problem.npz`: H (CSC, full), A (CSC), C (CSR), g, b, lb, ub,
  `ineq_labels`, and the recorded cold cuPIQP result (`cupiqp_x`,
  `cupiqp_ineq_duals` = z_u - z_l, status, iterations). Equality duals and
  slacks were not recorded: re-solve QP k to get the full state.
- Absent bounds are +/-inf (the producer clamps them to +/-1e20 before cuPIQP).
- The collision block (47 samples x 16 slots) holds the 16 closest pairs of
  each iteration, so a collision row can mean a different link pair in QP k and
  QP k+1. Use `ineq_labels` to map duals across iterations.
- Capture settings: eps_abs = eps_duality_gap_abs = 1e-6, relative tolerances 0,
  max_iter 250, reduced KKT, whole-solve graph, cold start. Cold baseline on the
  producer's A1000: 25-33 IPM iterations, about 31 ms per QP, flat even when
  consecutive QPs barely change. Re-measure everything on this GPU.

## Where to start in the code

- Initial point today (no user-supplied start anywhere):
  `native/src/ipm/solver/sparse_solver.cu` `initial_guess()` and
  `solve_prelude()`; reference `cupiqp/solver.py` `_initial_guess()` (eq. 12
  of Schwan et al. 2023: x = y = 0, s = z = 1, one KKT solve, then
  `init_guess_center` puts s and z on the central path) and
  `cupiqp/solver_kernels.py` `create_init_guess_rhs_kernel`.
- Proximal centers: `prox_` is copied from the initial point; `rho_init`,
  `delta_init` in `Settings` (`native/include/cupiqp/cupiqp.hpp`).
- Ruiz scaling (`native/src/problem/ruiz`): a user-supplied start is in
  unscaled space and must be scaled the same way the solution is unscaled.
- Python layer: `native/python/cupiqp_native`, harness examples in
  `native/bench/harness/` and `native/bench/tools/export_snapshots.py`.

## What to find out

1. Which warm-start options exist today without code changes (for example
   `rho_init` / `delta_init`, tolerances). Measure them.
2. Add an experimental initial-point hook to the native solver (C++ and Python)
   so external starts can be tested inside the real solver. Keep the default
   path unchanged and bitwise identical; no allocation or host sync added to the
   solve; the start must work with the whole-solve graph (stage it into
   preallocated device buffers before launch).
3. Test strategies. The list below is a starting point; add others from the
   IPM warm-start literature if they look promising, and say where they come
   from:
   - x = 0, duals (y, z) from the solution of QP k (mapped by `ineq_labels`
     for the collision block), slacks recomputed from QP k+1 data at x = 0,
     with a safeguard that keeps s and z strictly positive;
   - the same plus a centering step or shift so the complementarity products
     s_i z_i sit in a band around a chosen mu0 (Gondzio-style or similar);
   - active-set guess from QP k: rows active in QP k start with small s and
     large z, the others the opposite;
   - warm starting the proximal centers and starting rho / delta / mu lower;
   - controls: cold start; the full previous (x, y, s, z) including x_k,
     to show what x = 0 buys.
4. For each strategy and each QP transition, report: status, IPM iterations,
   solve time on this GPU, KKT check at 1e-6 against QP k+1, and the distance of
   the solution from the cold solution. Aggregate per run and overall (median,
   mean, worst, any failure).
5. Test the user's assumption: compare our re-solve of QP k with the recorded
   `cupiqp_x`, and check how well the seed derived from QP k predicts QP k+1's
   active set and duals (for example the fraction of active rows that stay
   active, relative dual change). State whether the assumption holds and where
   it breaks (for example collision-slot reassignment).

## Rules

- Correctness first. Do not change stopping criteria, tolerances,
  regularization or scaling of the default path.
- Style rules: RAII `CudaBuffer`, all allocation at setup, no host syncs in
  the solve, CUB where it fits, folders grouped by layer; ASCII-only comments.
- Do not touch or git-track `/mnt/c/Users/91354/Projects/gpu-qp-benchmark`.
- No cloud or remote agents, no uploads.
- Build in a separate directory (for example `build/native-ws`). `build/native`
  is stale (still the old cuDSS build).

## Deliverables

- Code on branch `feature/warmstart-study` from `native-cuda-port` (a9f74da).
  Commit on that branch only; do not merge or push.
- Scripts under `native/bench/warmstart/` (or `bench/tools/` if small), results
  in `native/bench/results/warmstart-20261104/`, note
  `native/docs/modules/warmstart_study.md` with the tables, the answer to the
  assumption question and a recommendation (which strategy, if any, to adopt,
  and what an API for it should look like).
- ctest and pytest still pass.
