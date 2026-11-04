# Brief: feasibility of a reduced (Schur-complement) KKT solve instead of cuDSS

Worktree `~/Projects/cupiqp-native-wt/reduced_kkt`, branch `exp/reduced-kkt`
(from `native-cuda-port`). Shared rules: `README.md` in this folder. This is a
measurement experiment. **Do not change the solver's behavior.** Add a probe
program and, at most, read-only accessors. Commit on your branch only.

## Background

cuDSS (closed source) takes about 70% of an IPM iteration: FACTORIZATION
about 807 us plus 2 x SOLVE about 104 us on MPC
(`native/bench/results/kernel-profile-20261103`). It factors the condensed
quasi-definite KKT matrix built by `linalg/kkt_system` and
`linalg/kkt_assembly`:

```
K = [ P + diag(x_reg)   A^T      G^T            ]
    [ A                 -delta I  0              ]
    [ G                 0         -diag(z_reg)   ]
```

Facts about K:
- It is stored as CSR (`build_kkt_host_maps` in `kkt_assembly/kkt_pattern.hpp`).
  The P block is stored in full (both triangles); A and G sit in the
  lower-left; the identity diagonals are explicit; cuDSS reads the lower
  triangle.
- Inactive inequality rows (both bounds infinite) have a z diagonal of -1
  and a zeroed G row.
- `z_reg_inv = w_u + w_l <= 2 / delta` and `x_reg >= rho > 0`
  (reference `cupiqp/kkt_systems_kernels.py`).

Eliminating dz and dy gives the n x n SPD system the reference's own dense
backend uses (`cupiqp/dense/dense_kkt_solver.py`):

```
H = P + diag(x_reg) + (1/delta) A^T A + G^T diag(z_reg_inv) G
H dx = r_x + A^T r_y / delta + G^T diag(z_reg_inv) r_z
dy = (A dx - r_y) / delta,   dz = diag(z_reg_inv) (G dx - r_z)
```

All of these quantities can be read off K: P + x_reg from the leading block,
delta from the y diagonal, z_reg from minus the z diagonal, and A and G from
the off-diagonal blocks. Dimensions:
- MPC: n = 169, p = 64, m = 2521, K dim 2754, G has 249,801 stored entries
  (99.1 per row; 98,921 numerically nonzero in some snapshot).
- Trajopt pattern 0: n = 241, p = 64, m = 6385, G has 164,465 stored entries.

This GPU's FP64 throughput is about 1/64 of its FP32 throughput (Ada,
RTX 4070 SUPER).

## What to build: `native/bench/reduced_kkt_probe.cu` (+ CMake target)

1. **Real KKT states.** For a few MPC snapshots and every Trajopt pattern
   (data: `~/Projects/cupiqp-native/native/bench/data/{mpc,trajopt}/group_*.bin`,
   reader `native/bench/snapshot_io.hpp`):
   - run the public `SparseSolver` (default settings of the replay:
     eps_abs = eps_duality_gap_abs = 1e-8, relative 0) with `max_iter = k`
     for k = 1, 2, ... up to convergence (include the last iterations, where
     rho and delta are smallest);
   - copy `kkt.values` from `debug_buffers()`. After a solve stopped at k, it
     holds the matrix factored in iteration k. Also record rho and delta
     from `info()`.
2. **Right-hand sides.** Use a random one and a realistic one (for example
   the residual vector of that state; read the `res.*` debug buffers).
3. **Method A (baseline).** A fresh `CudssSolver` with the solver's
   configuration on that K: factor + solve.
4. **Method B (reduced).**
   - Assemble H on the GPU from K's blocks.
   - Cholesky with cuSOLVER `potrf` (FP64; preallocated workspace), solve
     with `potrs`.
   - Recover dy and dz.
   Assembly variants to time:
   - (a) dense: build `diag(sqrt(z_reg_inv)) G` densely, then cuBLAS DSYRK,
     plus the A term;
   - (b) a deterministic custom kernel that works on the stored sparsity
     (for example per output entry or per column pair over CSC columns);
   - (c) the same as (b) but skipping numerically zero entries at run time.
   Results must be deterministic (no float atomics for H).
   Everything allocated once; time with CUDA events, eager and as a CUDA
   graph, median of many runs.
5. **Accuracy.** For each state and rhs, compute in long double on the CPU
   the relative residual of the full condensed system
   `||K x - b||_inf / (||K||_inf ||x||_inf + ||b||_inf)` for A and for B,
   plus the relative difference between the two solutions. Report against
   iteration index, rho and delta. Also report whether `potrf` succeeded.
6. **Optional, only if time remains.** Run one step of iterative refinement
   on B (one extra reduced solve with the residual), to see whether it
   closes any accuracy gap.

## Report (`native/docs/modules/reduced_kkt.md` + raw data under `native/bench/results/reduced-kkt-<date>/`)

- **Timing table per dataset.**
  - cuDSS: factor alone, solve alone, and the per-iteration equivalent
    (1 factor + 2 solves).
  - Reduced method: each assembly variant, Cholesky, the solve with
    recovery, and the per-iteration equivalent (1 assembly + 1 factor +
    2 solves).
- **Accuracy table/plot data** across iterations for both methods, including
  the last iterations.
- **A clear verdict.** Is the reduced method faster per iteration on MPC and
  on Trajopt, and by how much? Is it accurate enough? Base the accuracy
  judgment on the condensed residual staying near cuDSS's level (say within
  10x, and below about 1e-10 relative), and state the threshold you used.
  Say where it breaks down, if it does.
- What an integration would need: new backend selection, graph capture
  (cuSOLVER potrf/potrs capturability), workspace, and the expected
  deviations.

Reply with a concise summary: files, commands, the timing and accuracy
numbers, the verdict, and open issues. Report only measured numbers.
