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

// Block i computes row i of the lower triangle of
//   H = (P + diag x_reg) + M^T diag(w) M,   M = [A; G] (rows n .. n+p+m-1 of K).
// H_ij = (P + x_reg)_ij + sum_k w_k M_ki M_kj (j <= i). Warp v walks a
// contiguous chunk of column i's row list (k ascending, CSC of M) into its own
// shared accumulator. The lanes prefetch the metadata and w_k M_ki of 32 rows
// at once; rows with w_k M_ki == 0 (explicit zeros, masked inactive rows) are
// dropped by a warp ballot. Row k's off-diagonal entries are contiguous in K
// (from Kbase[k]) with ascending columns, so the warp reads entries
// Kbase[k] .. Kbase[k] + off (columns <= i) straight from K's CSR arrays, the
// lanes splitting them (distinct columns), __syncwarp between rows. Final sum
// in a fixed order: P, warp 0, ..., warp W-1.
template <int W>
__global__ void __launch_bounds__(W * 32) assemble_kernel(
    const double* __restrict__ K, const int32_t* __restrict__ Kidx, const double* __restrict__ w,
    const int32_t* __restrict__ Pptr, const int32_t* __restrict__ Pcol, const int32_t* __restrict__ Pslot,
    const int32_t* __restrict__ Kbase, const int32_t* __restrict__ Cptr, const int32_t* __restrict__ Crow,
    const int32_t* __restrict__ Coff, int n, double* __restrict__ H) {
  extern __shared__ double sh[];  // (W + 1) * n
  const int i = blockIdx.x;
  const int tid = threadIdx.x;
  for (int t = tid; t < (W + 1) * n; t += blockDim.x) sh[t] = 0.0;
  __syncthreads();
  for (int e = Pptr[i] + tid; e < Pptr[i + 1]; e += blockDim.x) sh[W * n + Pcol[e]] = K[Pslot[e]];
  const int warp = tid / 32, lane = tid % 32;
  const int c0 = Cptr[i], len = Cptr[i + 1] - c0;
  const int b0 = c0 + int(int64_t(len) * warp / W), b1 = c0 + int(int64_t(len) * (warp + 1) / W);
  double* acc = sh + warp * n;
  for (int base = b0; base < b1; base += 32) {
    const int e = base + lane;
    const bool ok = e < b1;
    int kb = 0, off = -1;
    double a = 0.0;
    if (ok) {
      const int k = Crow[e];
      off = Coff[e];
      kb = Kbase[k];
      a = w[k] * K[kb + off];
    }
    unsigned mask = __ballot_sync(0xffffffffu, ok && a != 0.0);
    while (mask) {
      const int src = __ffs(mask) - 1;
      mask &= mask - 1;
      const double as = __shfl_sync(0xffffffffu, a, src);
      const int kbs = __shfl_sync(0xffffffffu, kb, src);
      const int offs = __shfl_sync(0xffffffffu, off, src);
      for (int t = lane; t <= offs; t += 32) acc[Kidx[kbs + t]] += as * K[kbs + t];
      __syncwarp();
    }
  }
  __syncthreads();
  for (int j = tid; j <= i; j += blockDim.x) {
    double s = sh[W * n + j];
#pragma unroll
    for (int v = 0; v < W; ++v) s += sh[v * n + j];
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

// x[i] = b[i] + sum_k M_ki w_k b[n + k]   (one warp per column i of M).
__global__ void reduce_rhs_kernel(const double* __restrict__ K, const double* __restrict__ w,
                                  const int32_t* __restrict__ Kbase, const int32_t* __restrict__ Cptr,
                                  const int32_t* __restrict__ Crow, const int32_t* __restrict__ Coff,
                                  const double* __restrict__ b, int n, double* __restrict__ x) {
  const int i = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x % 32;
  if (i >= n) return;
  double s = 0.0;
  for (int e = Cptr[i] + lane; e < Cptr[i + 1]; e += 32) {
    const int k = Crow[e];
    s = fma(K[Kbase[k] + Coff[e]], w[k] * b[n + k], s);
  }
  s = warp_sum(s);
  if (lane == 0) x[i] = b[i] + s;
}

// In place x <- L^-T L^-1 x (x of length n), one block, x staged in shared
// memory. Per tile: one warp solves the 32 x 32 diagonal block (lane = row,
// shuffles), then the whole block applies the panel to the remaining rows
// (forward: thread per row, coalesced column reads; backward: warp per
// column, coalesced reads of L's columns, fixed-order shuffle sums).
template <int BLOCK>
__global__ void __launch_bounds__(BLOCK) trsv_kernel(const double* __restrict__ L, const double* __restrict__ dinv,
                                                     int n, double* __restrict__ x) {
  extern __shared__ double xs[];  // n
  const int tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
  constexpr int kWarps = BLOCK / 32;
  for (int i = tid; i < n; i += BLOCK) xs[i] = x[i];
  __syncthreads();
  const int N = (n + kTile - 1) / kTile;
  // Forward: L y = b.
  for (int kk = 0; kk < N; ++kk) {
    const int g0 = kk * kTile, gi = g0 + lane;
    if (warp == 0) {
      double l[kTile];
#pragma unroll
      for (int c = 0; c < kTile; ++c) l[c] = (gi < n && c < lane) ? L[size_t(g0 + c) * n + gi] : 0.0;
      double y = gi < n ? xs[gi] : 0.0;
      const double di = gi < n ? dinv[gi] : 0.0;
#pragma unroll
      for (int c = 0; c < kTile; ++c) {
        if (lane == c) y *= di;
        const double yc = __shfl_sync(0xffffffffu, y, c);
        if (lane > c) y = fma(-l[c], yc, y);
      }
      if (gi < n) xs[gi] = y;
    }
    __syncthreads();
    const int cmax = min(kTile, n - g0);
    for (int i = g0 + kTile + tid; i < n; i += BLOCK) {
      double s = 0.0;
      for (int c = 0; c < cmax; ++c) s = fma(L[size_t(g0 + c) * n + i], xs[g0 + c], s);
      xs[i] -= s;
    }
    __syncthreads();
  }
  // Backward: L^T x = y.
  for (int kk = N - 1; kk >= 0; --kk) {
    const int g0 = kk * kTile, gi = g0 + lane;
    if (warp == 0) {
      double l[kTile];  // l[c] = L(g0 + c, gi) for c > lane
#pragma unroll
      for (int c = 0; c < kTile; ++c) l[c] = (gi < n && c > lane && g0 + c < n) ? L[size_t(gi) * n + g0 + c] : 0.0;
      double v = gi < n ? xs[gi] : 0.0;
      const double di = gi < n ? dinv[gi] : 0.0;
#pragma unroll
      for (int c = kTile - 1; c >= 0; --c) {
        if (lane == c) v *= di;
        const double vc = __shfl_sync(0xffffffffu, v, c);
        if (lane < c) v = fma(-l[c], vc, v);
      }
      if (gi < n) xs[gi] = v;
    }
    __syncthreads();
    for (int j = warp; j < g0; j += kWarps) {
      const double t = gi < n ? L[size_t(j) * n + gi] * xs[gi] : 0.0;
      const double s = warp_sum(t);
      if (lane == 0) xs[j] -= s;
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
