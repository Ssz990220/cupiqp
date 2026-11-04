// Kernels of the reduced (Schur-complement) KKT backend. See
// reduced_kkt_solver.hpp for the method. Every kernel is deterministic: no
// atomics, every sum is taken in a fixed order.
//
// Storage: H is n x n, column-major, lower triangle (H[j * n + i], i >= j);
// after the Cholesky it holds L (H = L L^T) in the same place, and dinv[j] =
// 1 / L_jj. K is the KKT values buffer of KktAssembly (CSR, rows x | y | z).
#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace cupiqp {
namespace reduced_kkt {

constexpr int kTile = 32;  // Cholesky / triangular-solve tile (one warp)
constexpr int kSub = 8;    // columns per block of the trailing update

__device__ __forceinline__ double warp_sum(double v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

// w[k] = -1 / K_diag of y / z row k: 1 / delta on y rows, z_reg_inv on active
// z rows, 1 on inactive z rows (diagonal -1, G row zeroed in K).
__global__ void weights_kernel(const double* __restrict__ K, const int32_t* __restrict__ Dslot, int mm,
                               double* __restrict__ w) {
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k < mm) w[k] = -1.0 / K[Dslot[k]];
}

// Per factorization, compact the nonzero entries of every row k of M = [A; G]
// (rows n .. n+p+m-1 of K; explicit zeros and the zeroed rows of inactive
// constraints are dropped): row k's run starting at K slot Kbase[k] keeps its
// place, Cv / Cc hold the nonzero values / columns in ascending column
// order, and pre[Kbase[k] + q] = number of nonzeros among the first q + 1
// stored entries. One warp per row, ballot prefix counts (deterministic).
__global__ void compact_rows_kernel(const double* __restrict__ K, const int32_t* __restrict__ Kidx,
                                    const int32_t* __restrict__ Kbase, const int32_t* __restrict__ Mlen, int mm,
                                    double* __restrict__ Cv, int32_t* __restrict__ Cc, int32_t* __restrict__ pre) {
  const int k = int((int64_t(blockIdx.x) * blockDim.x + threadIdx.x) / 32), lane = threadIdx.x % 32;
  if (k >= mm) return;
  const int kb = Kbase[k], len = Mlen[k];
  int count = 0;
  for (int q0 = 0; q0 < len; q0 += 32) {
    const int q = q0 + lane;
    const double v = q < len ? K[kb + q] : 0.0;
    const bool nz = q < len && v != 0.0;
    const unsigned bal = __ballot_sync(0xffffffffu, nz);
    const int before = count + __popc(bal & ((1u << lane) - 1u));
    if (nz) {
      Cv[kb + before] = v;
      Cc[kb + before] = Kidx[kb + q];
    }
    if (q < len) pre[kb + q] = before + (nz ? 1 : 0);
    count += __popc(bal);
  }
}

// One block per task. A task is (row i, a contiguous range [e0, e1) of
// column i's CSC list of M); a heavy row is split over several tasks so that
// the blocks get comparable work. The task computes, for j <= i,
//   T_ij = sum_{e in range, k = Crow[e]} w_k M_ki M_kj   (+ (P + x_reg)_ij if first),
// and writes it to row i of H (single-task rows) or to its workspace slot
// (row i of a split row, summed by reduce_split_rows_kernel). Warp v walks a
// contiguous chunk of the range into its own shared accumulator; the lanes
// prefetch 32 entries' metadata and a = w_k M_ki at once, entries with a == 0
// are dropped by a ballot, then the warp walks row k's compacted nonzeros
// with columns <= i (pre gives their count), lanes splitting them (distinct
// columns), __syncwarp between rows. Fixed final order: P, warp 0..W-1.
// Side output for the solves: Ca[e] = w_k M_ki for every CSC entry e (the
// tasks partition the CSC entries, so every entry is written once).
template <int W>
__global__ void __launch_bounds__(W * 32) assemble_kernel(
    const double* __restrict__ K, const double* __restrict__ w, const int32_t* __restrict__ Pptr,
    const int32_t* __restrict__ Pcol, const int32_t* __restrict__ Pslot, const int32_t* __restrict__ Kbase,
    const int32_t* __restrict__ Crow, const int32_t* __restrict__ Coff, const double* __restrict__ Cv,
    const int32_t* __restrict__ Cc, const int32_t* __restrict__ pre, const int32_t* __restrict__ task_row,
    const int32_t* __restrict__ task_e0, const int32_t* __restrict__ task_e1,
    const int64_t* __restrict__ task_out, int n, double* __restrict__ H, double* __restrict__ ws,
    double* __restrict__ Ca) {
  extern __shared__ double sh[];  // (W + 1) * n
  const int task = blockIdx.x, i = task_row[task];
  // task_out < 0: single-task row, written to H directly (adds P); else
  // (workspace offset << 1) | (0 for the row's first task, which adds P; 1 otherwise).
  const int64_t out = task_out[task];
  const int tid = threadIdx.x;
  for (int t = tid; t < (W + 1) * (i + 1); t += blockDim.x) sh[t % (i + 1) + (t / (i + 1)) * n] = 0.0;
  __syncthreads();
  const int e0 = task_e0[task], e1 = task_e1[task];
  const bool add_P = out < 0 || (out & 1) == 0;
  if (add_P)
    for (int e = Pptr[i] + tid; e < Pptr[i + 1]; e += blockDim.x) sh[W * n + Pcol[e]] = K[Pslot[e]];
  const int warp = tid / 32, lane = tid % 32;
  const int len = e1 - e0;
  const int b0 = e0 + int(int64_t(len) * warp / W), b1 = e0 + int(int64_t(len) * (warp + 1) / W);
  double* acc = sh + warp * n;
  for (int base = b0; base < b1; base += 32) {
    const int e = base + lane;
    const bool ok = e < b1;
    int kb = 0, cnt = 0;
    double a = 0.0;
    if (ok) {
      const int k = Crow[e], off = Coff[e];
      kb = Kbase[k];
      a = w[k] * K[kb + off];
      cnt = pre[kb + off];
      Ca[e] = a;
    }
    unsigned mask = __ballot_sync(0xffffffffu, ok && a != 0.0);
    while (mask) {
      const int src = __ffs(mask) - 1;
      mask &= mask - 1;
      const double as = __shfl_sync(0xffffffffu, a, src);
      const int kbs = __shfl_sync(0xffffffffu, kb, src);
      const int cs = __shfl_sync(0xffffffffu, cnt, src);
      for (int t = lane; t < cs; t += 32) acc[Cc[kbs + t]] += as * Cv[kbs + t];
      __syncwarp();
    }
  }
  __syncthreads();
  double* dst = out < 0 ? nullptr : ws + (out >> 1);
  for (int j = tid; j <= i; j += blockDim.x) {
    double s = sh[W * n + j];
#pragma unroll
    for (int v = 0; v < W; ++v) s += sh[v * n + j];
    if (dst) dst[j] = s;
    else H[size_t(j) * n + i] = s;
  }
}

// Rows split over several tasks: H_ij = sum of the task partials in task
// order (j <= i). One block per split row.
__global__ void reduce_split_rows_kernel(const int32_t* __restrict__ srow, const int32_t* __restrict__ stask0,
                                         const int32_t* __restrict__ stask1, const int64_t* __restrict__ task_out,
                                         const double* __restrict__ ws, int n, double* __restrict__ H) {
  const int r = blockIdx.x, i = srow[r], t0 = stask0[r], t1 = stask1[r];
  for (int j = threadIdx.x; j <= i; j += blockDim.x) {
    double s = 0.0;
    for (int t = t0; t < t1; ++t) s += ws[(task_out[t] >> 1) + j];
    H[size_t(j) * n + i] = s;
  }
}

// Right-looking Cholesky of one 32 x 32 diagonal tile held by a warp, lane i
// owning row i in registers (a[t] = H_it for t <= i; rows past n are identity
// rows). On return a[t] = L_it (t <= i), inv = 1 / L_ii, and fail is the first
// global column + 1 with a non-positive or non-finite pivot (0 = none).
// 1 / sqrt is CUDA's double rsqrt (a hardware seed and FP64 refinement: on
// this class of GPU about half the latency of the float-seed Newton
// iterations it replaces, and at most 2 ulp). The pivot chain is software
// pipelined: step j first updates column j + 1, takes the next pivot and
// starts its rsqrt, and only then applies the remaining columns of the
// rank-1 update, so the rsqrt latency overlaps independent FMAs. Same
// operations and order per entry as the plain loop; only the issue order
// changes.
__device__ __forceinline__ void factor_diag_tile(double (&a)[kTile], int lane, int g0, int n, double& inv,
                                                 int& fail) {
  fail = 0;
  inv = 1.0;
  double d = __shfl_sync(0xffffffffu, a[0], 0);
  double r = rsqrt(d);
#pragma unroll
  for (int j = 0; j < kTile; ++j) {
    if (fail == 0 && g0 + j < n && !(d > 0.0 && d <= 1.7976931348623157e308)) fail = g0 + j + 1;
    if (lane == j) {
      a[j] = d * r;
      inv = r;
    } else if (lane > j) {
      a[j] *= r;
    }
    if (j + 1 < kTile) {
      const double l1 = __shfl_sync(0xffffffffu, a[j], j + 1);
      if (lane >= j + 1) a[j + 1] = fma(-a[j], l1, a[j + 1]);
      d = __shfl_sync(0xffffffffu, a[j + 1], j + 1);
      const double rn = rsqrt(d);
#pragma unroll
      for (int t = j + 2; t < kTile; ++t) {
        const double ltj = __shfl_sync(0xffffffffu, a[j], t);
        if (lane >= t) a[t] = fma(-a[j], ltj, a[t]);
      }
      r = rn;
    }
  }
}

// Tile column kk of the blocked right-looking Cholesky. Every block factors
// the diagonal tile (kk, kk) itself (one warp, registers; redundant work runs
// in parallel on other SMs instead of costing a launch). Block 0 writes L_kk,
// dinv and the status; block b >= 1 solves the panel tile (kk + b, kk):
// X = A L_kk^{-T}, lane = row, column by column. status: assigned at kk = 0
// (so every factorization starts from 0), then only the first failure is kept.
__global__ void __launch_bounds__(32) chol_panel_kernel(double* __restrict__ H, int n, int kk,
                                                        double* __restrict__ dinv, int32_t* __restrict__ status) {
  __shared__ double Ls[kTile][kTile + 1];
  __shared__ double dinv_s[kTile];
  const int lane = threadIdx.x, g0 = kk * kTile, gi = g0 + lane;
  double a[kTile];
#pragma unroll
  for (int t = 0; t < kTile; ++t)
    a[t] = gi < n ? (t <= lane ? H[size_t(g0 + t) * n + gi] : 0.0) : (t == lane ? 1.0 : 0.0);
  double inv;
  int fail;
  factor_diag_tile(a, lane, g0, n, inv, fail);
  if (blockIdx.x == 0) {
    if (gi < n) {
#pragma unroll
      for (int t = 0; t < kTile; ++t)
        if (t <= lane) H[size_t(g0 + t) * n + gi] = a[t];
      dinv[gi] = inv;
    }
    if (lane == 0) {
      if (kk == 0) *status = fail;
      else if (fail != 0 && *status == 0) *status = fail;
    }
    return;
  }
#pragma unroll
  for (int t = 0; t < kTile; ++t) Ls[lane][t] = a[t];
  dinv_s[lane] = inv;
  __syncwarp();
  const int r = (kk + blockIdx.x) * kTile + lane;
  double x[kTile];
#pragma unroll
  for (int t = 0; t < kTile; ++t) x[t] = (r < n && g0 + t < n) ? H[size_t(g0 + t) * n + r] : 0.0;
#pragma unroll
  for (int c = 0; c < kTile; ++c) {
    x[c] *= dinv_s[c];
#pragma unroll
    for (int t = c + 1; t < kTile; ++t) x[t] = fma(-x[c], Ls[t][c], x[t]);
  }
  if (r < n) {
#pragma unroll
    for (int t = 0; t < kTile; ++t)
      if (g0 + t < n) H[size_t(g0 + t) * n + r] = x[t];
  }
}

// Trailing update after tile column kk: sub-tile (rows of tile ii, columns
// jsub of tile jj) -= L(ii, kk) L(jj, kk)^T for kk < jj <= ii, lower part
// only. Grid (mt * kTile / kSub, mt), block (kTile, kSub).
__global__ void __launch_bounds__(kTile * kSub) chol_update_kernel(double* __restrict__ H, int n, int kk) {
  const int jt = blockIdx.x / (kTile / kSub), js = blockIdx.x % (kTile / kSub), it = blockIdx.y;
  if (jt > it) return;
  __shared__ double Li[kTile][kTile + 1], Lj[kSub][kTile + 1];
  const int g0 = kk * kTile, ri = (kk + 1 + it) * kTile, rj = (kk + 1 + jt) * kTile + js * kSub;
  const int tx = threadIdx.x, ty = threadIdx.y;  // tx: row within tile, ty: column within sub-tile
  for (int l = ty; l < kTile; l += kSub) {
    const int gc = g0 + l;
    Li[tx][l] = (ri + tx < n && gc < n) ? H[size_t(gc) * n + ri + tx] : 0.0;
  }
  {
    const int lin = ty * kTile + tx, c = lin / kTile, l = lin % kTile;
    const int gc = g0 + l;
    Lj[c][l] = (rj + c < n && gc < n) ? H[size_t(gc) * n + rj + c] : 0.0;
  }
  __syncthreads();
  const int gr = ri + tx, gcol = rj + ty;
  if (gr < n && gcol < n && gr >= gcol) {
    double v = 0.0;
#pragma unroll 8
    for (int l = 0; l < kTile; ++l) v = fma(Li[tx][l], Lj[ty][l], v);
    H[size_t(gcol) * n + gr] -= v;
  }
}

// x[i] = b[i] + sum_k (w_k M_ki) b[n + k]: one block of RHS_BLOCK threads per
// column i of M, over the weighted CSC values Ca written by the assembly
// (entries with Ca == 0, i.e. explicit zeros and inactive rows, skip the
// gather of b), thread-strided partial sums, then warp shuffles and a
// fixed-order sum over the warps.
constexpr int kRhsBlock = 128;
__global__ void __launch_bounds__(kRhsBlock) reduce_rhs_kernel(const double* __restrict__ Ca,
                                                               const int32_t* __restrict__ Cptr,
                                                               const int32_t* __restrict__ Crow,
                                                               const double* __restrict__ b, int n,
                                                               double* __restrict__ x) {
  __shared__ double part[kRhsBlock / 32];
  const int i = blockIdx.x, tid = threadIdx.x;
  double s = 0.0;
  for (int e = Cptr[i] + tid; e < Cptr[i + 1]; e += kRhsBlock) {
    const double a = Ca[e];
    if (a != 0.0) s = fma(a, b[n + Crow[e]], s);
  }
  s = warp_sum(s);
  if (tid % 32 == 0) part[tid / 32] = s;
  __syncthreads();
  if (tid == 0) {
    double t = b[i];
    double sum = 0.0;
#pragma unroll
    for (int v = 0; v < kRhsBlock / 32; ++v) sum += part[v];
    x[i] = t + sum;
  }
}

// Inter-block flags of the multi-block triangular solve: release / acquire
// at GPU scope (the data a flag publishes is read with __ldcg, bypassing L1).
__device__ __forceinline__ uint32_t ld_acquire_gpu(const uint32_t* p) {
  uint32_t v;
  asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void st_release_gpu(uint32_t* p, uint32_t v) {
  asm volatile("st.release.gpu.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
// Warp-wide wait until *f == epoch (lane 0 polls).
__device__ __forceinline__ void warp_wait_flag(const uint32_t* f, uint32_t epoch, int lane) {
  if (lane == 0)
    while (ld_acquire_gpu(f) != epoch) {
    }
  __syncwarp();
}

// In place x <- L^-T L^-1 x (x of length n) as a dataflow over 32-row tiles,
// one warp per block and 2N blocks (N = ceil(n / 32)): tasks 0 .. N-1 are the
// forward tiles 0 .. N-1, tasks N .. 2N-1 the backward tiles N-1 .. 0. A
// block takes its task from a ticket counter (atomicAdd), so every task it
// waits for belongs to a block that is already running (no deadlock whatever
// the residency or block order); the ticket also gives the epoch, so the
// flags never need resetting (flag value = epoch when the tile is published).
// Forward tile b: z = b_b - sum_{t<b} L_bt y_t (tiles taken in ascending t as
// their flags appear, the tile of L prefetched before the wait; lane = row,
// 4 interleaved accumulators), then the diagonal substitution in the scaled,
// division-free form z_c = b_c - sum_{t<c} (L_ct / L_tt) z_t (one shuffle +
// one FMA per column), y = z / L_cc. Backward tile b: same with L^T (lane =
// column of L, t descending). Fixed summation order: deterministic. Every
// value written to x is published by its tile's flag; x is updated in place
// (a forward tile is only overwritten by its backward task, which starts
// after every forward task has consumed it).
__global__ void __launch_bounds__(32) trsv_flow_kernel(const double* __restrict__ L, const double* __restrict__ dinv,
                                                       int n, double* __restrict__ x,
                                                       unsigned long long* __restrict__ ticket,
                                                       uint32_t* __restrict__ flags) {
  const int lane = threadIdx.x, N = (n + kTile - 1) / kTile;
  unsigned long long tk = 0;
  if (lane == 0) tk = atomicAdd(ticket, 1ull);
  tk = __shfl_sync(0xffffffffu, tk, 0);
  const int task = int(tk % (unsigned long long)(2 * N));
  const uint32_t epoch = uint32_t(tk / (unsigned long long)(2 * N)) + 1u;
  const bool fwd = task < N;
  const int b = fwd ? task : 2 * N - 1 - task, b0 = b * kTile, gi = b0 + lane;
  const bool in = gi < n;
  // Scaled diagonal tile: forward l[c] = L(gi, b0 + c) / L(b0 + c, b0 + c) for
  // c < lane; backward l[c] = L(b0 + c, gi) / L(b0 + c, b0 + c) for c > lane.
  double l[kTile];
#pragma unroll
  for (int c = 0; c < kTile; ++c) {
    const int gc = b0 + c;
    l[c] = 0.0;
    if (fwd) {
      if (in && c < lane) l[c] = L[size_t(gc) * n + gi] * dinv[gc];
    } else {
      if (in && c > lane && gc < n) l[c] = L[size_t(gi) * n + gc] * dinv[gc];
    }
  }
  const double di = in ? dinv[gi] : 0.0;
  double a0 = 0.0, a1 = 0.0, a2 = 0.0, a3 = 0.0, r = 0.0;
  if (fwd) {
    r = in ? __ldcg(x + gi) : 0.0;
    for (int t = 0; t < b; ++t) {
      const int t0 = t * kTile;
      double Lr[kTile];
#pragma unroll
      for (int c = 0; c < kTile; ++c) Lr[c] = in ? L[size_t(t0 + c) * n + gi] : 0.0;
      warp_wait_flag(flags + t, epoch, lane);
      const double xv = __ldcg(x + t0 + lane);
#pragma unroll
      for (int c = 0; c < kTile; c += 4) {
        a0 = fma(Lr[c], __shfl_sync(0xffffffffu, xv, c), a0);
        a1 = fma(Lr[c + 1], __shfl_sync(0xffffffffu, xv, c + 1), a1);
        a2 = fma(Lr[c + 2], __shfl_sync(0xffffffffu, xv, c + 2), a2);
        a3 = fma(Lr[c + 3], __shfl_sync(0xffffffffu, xv, c + 3), a3);
      }
    }
  } else {
    warp_wait_flag(flags + b, epoch, lane);  // forward tile b
    r = in ? __ldcg(x + gi) : 0.0;
    for (int t = N - 1; t > b; --t) {
      const int t0 = t * kTile;
      double Lr[kTile];
#pragma unroll
      for (int q = 0; q < kTile; ++q) Lr[q] = (in && t0 + q < n) ? L[size_t(gi) * n + t0 + q] : 0.0;
      warp_wait_flag(flags + N + t, epoch, lane);  // backward tile t
      const double xv = t0 + lane < n ? __ldcg(x + t0 + lane) : 0.0;
#pragma unroll
      for (int q = 0; q < kTile; q += 4) {
        a0 = fma(Lr[q], __shfl_sync(0xffffffffu, xv, q), a0);
        a1 = fma(Lr[q + 1], __shfl_sync(0xffffffffu, xv, q + 1), a1);
        a2 = fma(Lr[q + 2], __shfl_sync(0xffffffffu, xv, q + 2), a2);
        a3 = fma(Lr[q + 3], __shfl_sync(0xffffffffu, xv, q + 3), a3);
      }
    }
  }
  double z = in ? r - ((a0 + a1) + (a2 + a3)) : 0.0;
  if (fwd) {
#pragma unroll
    for (int c = 0; c < kTile - 1; ++c) {
      const double zc = __shfl_sync(0xffffffffu, z, c);
      if (lane > c) z = fma(-l[c], zc, z);
    }
  } else {
#pragma unroll
    for (int c = kTile - 1; c > 0; --c) {
      const double zc = __shfl_sync(0xffffffffu, z, c);
      if (lane < c) z = fma(-l[c], zc, z);
    }
  }
  if (in) x[gi] = z * di;
  __threadfence();
  __syncwarp();
  if (lane == 0) st_release_gpu(flags + (fwd ? b : N + b), epoch);
}

// x[n + k] = w_k (M_k dx - b[n + k])   (one warp per row k of M), over the
// row's compacted nonzeros (compact_rows_kernel: Cv / Cc, count pre of the
// row's last stored entry).
__global__ void recover_kernel(const double* __restrict__ Cv, const int32_t* __restrict__ Cc,
                               const int32_t* __restrict__ pre, const double* __restrict__ w,
                               const int32_t* __restrict__ Kbase, const int32_t* __restrict__ Mlen,
                               const double* __restrict__ b, int n, int mm, double* __restrict__ x) {
  const int k = int((int64_t(blockIdx.x) * blockDim.x + threadIdx.x) / 32), lane = threadIdx.x % 32;
  if (k >= mm) return;
  const int kb = Kbase[k], len = Mlen[k];
  const int cnt = len > 0 ? pre[kb + len - 1] : 0;
  double s = 0.0;
  for (int q = lane; q < cnt; q += 32) s = fma(Cv[kb + q], x[Cc[kb + q]], s);
  s = warp_sum(s);
  if (lane == 0) x[n + k] = w[k] * (s - b[n + k]);
}

}  // namespace reduced_kkt
}  // namespace cupiqp
