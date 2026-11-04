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
template <int W>
__global__ void __launch_bounds__(W * 32) assemble_kernel(
    const double* __restrict__ K, const double* __restrict__ w, const int32_t* __restrict__ Pptr,
    const int32_t* __restrict__ Pcol, const int32_t* __restrict__ Pslot, const int32_t* __restrict__ Kbase,
    const int32_t* __restrict__ Crow, const int32_t* __restrict__ Coff, const double* __restrict__ Cv,
    const int32_t* __restrict__ Cc, const int32_t* __restrict__ pre, const int32_t* __restrict__ task_row,
    const int32_t* __restrict__ task_e0, const int32_t* __restrict__ task_e1,
    const int64_t* __restrict__ task_out, int n, double* __restrict__ H, double* __restrict__ ws) {
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

// 1 / sqrt(d): float seed refined by two Newton steps in FP64 (a few FP64
// instructions instead of the long FP64 sqrt + division sequences, which
// dominate the diagonal tile on GPUs with reduced FP64 rate). Outside the
// float range the exact expression is used. Deterministic.
__device__ __forceinline__ double rsqrt_refined(double d) {
  if (d > 1e-36 && d < 1e36) {
    double y = double(rsqrtf(float(d)));
    y = fma(y, fma(-0.5 * d * y, y, 0.5), y);
    y = fma(y, fma(-0.5 * d * y, y, 0.5), y);
    return y;
  }
  return 1.0 / sqrt(d);
}

// Right-looking Cholesky of one 32 x 32 diagonal tile held by a warp, lane i
// owning row i in registers (a[t] = H_it for t <= i; rows past n are identity
// rows). On return a[t] = L_it (t <= i), inv = 1 / L_ii, and fail is the first
// global column + 1 with a non-positive or non-finite pivot (0 = none).
__device__ __forceinline__ void factor_diag_tile(double (&a)[kTile], int lane, int g0, int n, double& inv,
                                                 int& fail) {
  fail = 0;
  inv = 1.0;
#pragma unroll
  for (int j = 0; j < kTile; ++j) {
    const double d = __shfl_sync(0xffffffffu, a[j], j);
    if (fail == 0 && g0 + j < n && !(d > 0.0 && d <= 1.7976931348623157e308)) fail = g0 + j + 1;
    const double r = rsqrt_refined(d);
    if (lane == j) {
      a[j] = d * r;
      inv = r;
    } else if (lane > j) {
      a[j] *= r;
    }
#pragma unroll
    for (int t = j + 1; t < kTile; ++t) {
      const double ltj = __shfl_sync(0xffffffffu, a[j], t);
      if (lane >= t) a[t] = fma(-a[j], ltj, a[t]);
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

// x[i] = b[i] + sum_k M_ki w_k b[n + k]: one block of RHS_BLOCK threads per
// column i of M (Cslot: K slot of each CSC entry), thread-strided partial
// sums, then warp shuffles and a fixed-order sum over the warps.
constexpr int kRhsBlock = 128;
__global__ void __launch_bounds__(kRhsBlock) reduce_rhs_kernel(
    const double* __restrict__ K, const double* __restrict__ w, const int32_t* __restrict__ Cptr,
    const int32_t* __restrict__ Crow, const int32_t* __restrict__ Cslot, const double* __restrict__ b, int n,
    double* __restrict__ x) {
  __shared__ double part[kRhsBlock / 32];
  const int i = blockIdx.x, tid = threadIdx.x;
  double s = 0.0;
  for (int e = Cptr[i] + tid; e < Cptr[i + 1]; e += kRhsBlock) {
    const int k = Crow[e];
    s = fma(K[Cslot[e]], w[k] * b[n + k], s);
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

// In place x <- L^-T L^-1 x (x of length n), one block, x staged in shared
// memory. Per 32-row tile: one warp solves the diagonal block (lane = row,
// shuffles; the n-step substitution chain), then the whole block applies the
// panel to the remaining entries with 4 threads per entry, each summing 8
// panel columns, combined by a fixed shuffle butterfly (deterministic).
// Forward reads columns of L (coalesced over rows); backward reads rows of
// the tile (contiguous 8-entry runs of L's columns).
template <int BLOCK>
__global__ void __launch_bounds__(BLOCK) trsv_kernel(const double* __restrict__ L, const double* __restrict__ dinv,
                                                     int n, double* __restrict__ x) {
  extern __shared__ double xs[];  // n
  constexpr int kPer = 4, kSeg = kTile / kPer, kItems = BLOCK / kPer;
  const int tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
  const int q = tid % kPer, item = tid / kPer;
  for (int i = tid; i < n; i += BLOCK) xs[i] = x[i];
  __syncthreads();
  const int N = (n + kTile - 1) / kTile;
  // Forward: L y = b.
  for (int kk = 0; kk < N; ++kk) {
    const int g0 = kk * kTile, gi = g0 + lane;
    if (warp == 0) {
      // z_c = L_cc y_c satisfies z_c = b_c - sum_{t<c} (L_ct / L_tt) z_t: the
      // chain is one shuffle and one FMA per column; y = z / L_cc afterwards.
      double l[kTile];
#pragma unroll
      for (int c = 0; c < kTile; ++c)
        l[c] = (gi < n && c < lane) ? L[size_t(g0 + c) * n + gi] * dinv[g0 + c] : 0.0;
      double z = gi < n ? xs[gi] : 0.0;
#pragma unroll
      for (int c = 0; c < kTile - 1; ++c) {
        const double zc = __shfl_sync(0xffffffffu, z, c);
        if (lane > c) z = fma(-l[c], zc, z);
      }
      if (gi < n) xs[gi] = z * dinv[gi];
    }
    __syncthreads();
    const int c0 = g0 + q * kSeg;
    double xv[kSeg];
#pragma unroll
    for (int c = 0; c < kSeg; ++c) xv[c] = c0 + c < n ? xs[c0 + c] : 0.0;
    for (int base = g0 + kTile; base < n; base += kItems) {
      const int i = base + item;
      double s = 0.0;
      if (i < n) {
#pragma unroll
        for (int c = 0; c < kSeg; ++c)
          if (c0 + c < n) s = fma(L[size_t(c0 + c) * n + i], xv[c], s);
      }
      s += __shfl_xor_sync(0xffffffffu, s, 1);
      s += __shfl_xor_sync(0xffffffffu, s, 2);
      if (q == 0 && i < n) xs[i] -= s;
    }
    __syncthreads();
  }
  // Backward: L^T x = y.
  for (int kk = N - 1; kk >= 0; --kk) {
    const int g0 = kk * kTile, gi = g0 + lane;
    if (warp == 0) {
      // u_c = L_cc x_c satisfies u_c = y_c - sum_{t>c} (L_tc / L_tt) u_t.
      double l[kTile];  // l[c] = L(g0 + c, gi) / L(g0 + c, g0 + c) for c > lane
#pragma unroll
      for (int c = 0; c < kTile; ++c)
        l[c] = (gi < n && c > lane && g0 + c < n) ? L[size_t(gi) * n + g0 + c] * dinv[g0 + c] : 0.0;
      double u = gi < n ? xs[gi] : 0.0;
#pragma unroll
      for (int c = kTile - 1; c > 0; --c) {
        const double uc = __shfl_sync(0xffffffffu, u, c);
        if (lane < c) u = fma(-l[c], uc, u);
      }
      if (gi < n) xs[gi] = u * dinv[gi];
    }
    __syncthreads();
    const int c0 = g0 + q * kSeg;
    double xv[kSeg];
#pragma unroll
    for (int c = 0; c < kSeg; ++c) xv[c] = c0 + c < n ? xs[c0 + c] : 0.0;
    for (int base = 0; base < g0; base += kItems) {
      const int j = base + item;
      double s = 0.0;
      if (j < g0) {
#pragma unroll
        for (int c = 0; c < kSeg; ++c)
          if (c0 + c < n) s = fma(L[size_t(j) * n + c0 + c], xv[c], s);
      }
      s += __shfl_xor_sync(0xffffffffu, s, 1);
      s += __shfl_xor_sync(0xffffffffu, s, 2);
      if (q == 0 && j < g0) xs[j] -= s;
    }
    __syncthreads();
  }
  for (int i = tid; i < n; i += BLOCK) x[i] = xs[i];
}

// x[n + k] = w_k (M_k dx - b[n + k])   (one warp per row k of M).
__global__ void recover_kernel(const double* __restrict__ K, const int32_t* __restrict__ Kidx,
                               const double* __restrict__ w, const int32_t* __restrict__ Kbase,
                               const int32_t* __restrict__ Mlen, const double* __restrict__ b, int n, int mm,
                               double* __restrict__ x) {
  const int k = int((int64_t(blockIdx.x) * blockDim.x + threadIdx.x) / 32), lane = threadIdx.x % 32;
  if (k >= mm) return;
  const int kb = Kbase[k], len = Mlen[k];
  double s = 0.0;
  for (int q = lane; q < len; q += 32) s = fma(K[kb + q], x[Kidx[kb + q]], s);
  s = warp_sum(s);
  if (lane == 0) x[n + k] = w[k] * (s - b[n + k]);
}

}  // namespace reduced_kkt
}  // namespace cupiqp
