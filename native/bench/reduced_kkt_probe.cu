// Feasibility probe: reduced (Schur-complement) n x n Cholesky solve of the
// condensed KKT system versus cuDSS, on real KKT matrices of MPC / Trajopt
// solves (see native/docs/briefs/reduced_kkt.md). Measurement only: the solver
// is used through its public API and is not modified.
//
// For every selected snapshot case the public SparseSolver is run with
// max_iter = k for k = 1 .. (iterations to convergence); after each run the
// KKT values factored in iteration k are copied from debug_buffers()
// ("kkt.values"), together with rho / delta (info()) and the res
// buffers used for realistic right-hand sides. On each such state:
//
//   Method A: CudssSolver (the solver's configuration) factor + solve.
//   Method B: H = P + diag(x_reg) + A^T A / delta + G^T diag(z_reg_inv) G
//             read off K, assembled on the GPU by one of three variants,
//             cuSOLVER potrf / potrs (FP64), dy / dz recovered.
//     (a) dense:  M_s = diag(sqrt(w)) [A; G] dense, cuBLAS DSYRK
//     (b) sparse: deterministic per-output-row kernel over the stored pattern
//     (c) as (b), skipping rows k with w_k * M_ki == 0 at run time
//   with w = -1 / diag(K) on the y / z rows (1/delta, z_reg_inv; inactive
//   rows: 1 on a zeroed row).
//
// Accuracy: long double CPU residual of the full condensed system
//   ||K x - b||_inf / (||K||_inf ||x||_inf + ||b||_inf)
// for A and B, the relative difference of the solutions, potrf info, and one
// optional step of iterative refinement on B (FP64 residual).
// Timing: CUDA events, eager and as a replayed CUDA graph, median / P95.
//
// Usage: cupiqp_native_reduced_kkt_probe --data <group_k.bin> --label <name>
//          --cases 0,21 [--reps 100] [--time-every 1] [--out-dir <dir>]
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <functional>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "common/buffer.hpp"
#include "common/graph.hpp"
#include "common/handles.hpp"
#include "cupiqp/cupiqp.hpp"
#include "linalg/cudss/cudss_solver.hpp"
#include "linalg/kkt_assembly/kkt_pattern.hpp"
#include "snapshot_io.hpp"

using namespace cupiqp;

#define PROBE_CUBLAS_CHECK(expr)                                                                \
  do {                                                                                          \
    cublasStatus_t st_ = (expr);                                                                \
    if (st_ != CUBLAS_STATUS_SUCCESS)                                                           \
      throw std::runtime_error(std::string(#expr) + " failed: " + std::to_string(int(st_)));   \
  } while (0)
#define PROBE_CUSOLVER_CHECK(expr)                                                              \
  do {                                                                                          \
    cusolverStatus_t st_ = (expr);                                                              \
    if (st_ != CUSOLVER_STATUS_SUCCESS)                                                         \
      throw std::runtime_error(std::string(#expr) + " failed: " + std::to_string(int(st_)));   \
  } while (0)

namespace {

constexpr int kWarps = 16;  // warps per output row in the sparse assembly (8: 236 us, 16: 190 us, 32: 187 us on MPC)

// ------------------------------------------------------------------ kernels

// w[k] = -1 / K_diag of y / z row k (1/delta, z_reg_inv; inactive rows 1).
__global__ void weights_kernel(const double* __restrict__ K, const int32_t* __restrict__ Dslot, int mm,
                               double* __restrict__ w) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k < mm) w[k] = -1.0 / K[Dslot[k]];
}

// (a) dense: Md (mm x n, column-major) = sqrt(w_k) * M_kj on stored entries.
__global__ void scatter_dense_kernel(const double* __restrict__ K, const double* __restrict__ w,
                                     const int32_t* __restrict__ Mrow, const int32_t* __restrict__ Mcol,
                                     const int32_t* __restrict__ Mslot, int nnzM, int mm, double* __restrict__ Md) {
  int q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q < nnzM) {
    int k = Mrow[q];
    Md[size_t(Mcol[q]) * mm + k] = sqrt(w[k]) * K[Mslot[q]];
  }
}

// (a) dense: lower-triangular P + diag(x_reg) entries into the zeroed H.
__global__ void scatter_P_kernel(const double* __restrict__ K, const int32_t* __restrict__ Prow,
                                 const int32_t* __restrict__ Pcol, const int32_t* __restrict__ Pslot, int nnzP,
                                 int n, double* __restrict__ H) {
  int e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e < nnzP) H[size_t(Pcol[e]) * n + Prow[e]] = K[Pslot[e]];
}

// (b) / (c) sparse: block i computes row i of the lower triangle of H,
// H_ij = (P + diag x_reg)_ij + sum_k w_k M_ki M_kj (j <= i). Warp v walks a
// contiguous chunk of column i's row list (k ascending) into its own shared
// accumulator. The lanes prefetch the metadata and the scalar w_k M_ki of 32
// consecutive rows k at once (hiding the load latency); the warp then
// processes those rows in order, the lanes splitting row k's entries
// j <= i (distinct columns), with __syncwarp between rows. The off-diagonal
// entries of KKT row n + k are contiguous in K (Kbase[k] .. ), so values and
// column indices are read directly from K's CSR arrays. Final sum in fixed
// order: P, warp 0, ..., warp W-1. Deterministic, no atomics.
// SKIP: rows with w_k M_ki == 0 (explicit zeros, masked rows) are dropped
// with a warp ballot before the row loop.
template <int W, bool SKIP>
__global__ void __launch_bounds__(W * 32) syrk_sparse_kernel(
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
    unsigned mask = __ballot_sync(0xffffffffu, ok && (!SKIP || a != 0.0));
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

// ---------------------------------------------------------------- tiled Cholesky (exploratory)
// Right-looking blocked Cholesky, lower, column-major n x n (ld n), tile 32,
// one launch triple per tile column kk: diagonal tile (one 32 x 32 block,
// right-looking inside the tile), panel solve (one 32 x 32 block per tile
// below), trailing update (one 32 x 8 block per 32 x 8 sub-tile). The
// diagonal kernel stores 1 / L_jj in dinv for the panel solve. Out-of-range
// entries act as the identity. Deterministic. info: first failing global
// column + 1 (0 = success), like potrf.
constexpr int kTile = 32;
constexpr int kSub = 8;  // columns per trailing-update block

__global__ void __launch_bounds__(kTile * kTile) chol_diag_kernel(double* __restrict__ H, int n, int kk,
                                                                  double* __restrict__ dinv, int* __restrict__ info) {
  __shared__ double t[kTile][kTile + 1];
  __shared__ double inv_s;
  __shared__ int fail;
  const int tx = threadIdx.x, ty = threadIdx.y, g0 = kk * kTile;  // tx: row, ty: column
  const int gi = g0 + tx, gj = g0 + ty;
  t[tx][ty] = (gi < n && gj < n) ? (tx >= ty ? H[size_t(gj) * n + gi] : 0.0) : (tx == ty ? 1.0 : 0.0);
  if (tx == 0 && ty == 0) fail = 0;
  for (int j = 0; j < kTile; ++j) {
    __syncthreads();
    if (tx == j && ty == j) {
      const double d = t[j][j];
      if (!(d > 0.0) && fail == 0) fail = g0 + j + 1;
      const double r = sqrt(d);
      t[j][j] = r;
      inv_s = 1.0 / r;
      if (g0 + j < n) dinv[g0 + j] = inv_s;
    }
    __syncthreads();
    if (ty == j && tx > j) t[tx][j] *= inv_s;
    __syncthreads();
    if (ty > j && tx >= ty) t[tx][ty] -= t[tx][j] * t[ty][j];
  }
  __syncthreads();
  if (gi < n && gj < n && tx >= ty) H[size_t(gj) * n + gi] = t[tx][ty];
  if (tx == 0 && ty == 0 && fail != 0 && *info == 0) *info = fail;
}

// Tile (ii, kk), ii = kk + 1 + blockIdx.x: X = A * L_kk^{-T}, column by column.
__global__ void __launch_bounds__(kTile * kTile) chol_trsm_kernel(double* __restrict__ H, int n, int kk,
                                                                  const double* __restrict__ dinv) {
  __shared__ double L[kTile][kTile + 1];
  __shared__ double X[kTile][kTile + 1];
  const int tx = threadIdx.x, ty = threadIdx.y, g0 = kk * kTile, r0 = (kk + 1 + blockIdx.x) * kTile;
  const int gi = g0 + tx, gj = g0 + ty, gr = r0 + tx;
  L[tx][ty] = (gi < n && gj < n && tx >= ty) ? H[size_t(gj) * n + gi] : 0.0;
  X[tx][ty] = (gr < n && gj < n) ? H[size_t(gj) * n + gr] : 0.0;
  __syncthreads();
  for (int c = 0; c < kTile; ++c) {
    if (ty == c) X[tx][c] *= (g0 + c < n) ? dinv[g0 + c] : 1.0;
    __syncthreads();
    if (ty > c) X[tx][ty] -= X[tx][c] * L[ty][c];
    __syncthreads();
  }
  if (gr < n && gj < n) H[size_t(gj) * n + gr] = X[tx][ty];
}

// Sub-tile (rows of tile ii, columns jsub of tile jj) -= L(ii, kk) L(jj, kk)^T
// for kk < jj <= ii; grid (mt * kTile / kSub, mt).
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
    // Lj[c][l] for c < kSub, l < kTile: 256 threads load 8 x 32 entries.
    const int lin = ty * kTile + tx, c = lin / kTile, l = lin % kTile;
    const int gc = g0 + l;
    Lj[c][l] = (rj + c < n && gc < n) ? H[size_t(gc) * n + rj + c] : 0.0;
  }
  __syncthreads();
  const int gr = ri + tx, gcol = rj + ty;
  if (gr < n && gcol < n && gr >= gcol) {
    double v = 0.0;
#pragma unroll 8
    for (int l = 0; l < kTile; ++l) v += Li[tx][l] * Lj[ty][l];
    H[size_t(gcol) * n + gr] -= v;
  }
}

__device__ __forceinline__ double warp_sum(double v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

// x[i] = b[i] + sum_k M_ki w_k b[n + k]   (one warp per column i)
__global__ void reduced_rhs_kernel(const double* __restrict__ K, const double* __restrict__ w,
                                   const int32_t* __restrict__ Cptr, const int32_t* __restrict__ Crow,
                                   const int32_t* __restrict__ Cpos, const int32_t* __restrict__ Mslot,
                                   const double* __restrict__ b, int n, double* __restrict__ x) {
  const int i = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x % 32;
  if (i >= n) return;
  double s = 0.0;
  for (int e = Cptr[i] + lane; e < Cptr[i + 1]; e += 32) {
    const int k = Crow[e];
    s += K[Mslot[Cpos[e]]] * (w[k] * b[n + k]);
  }
  s = warp_sum(s);
  if (lane == 0) x[i] = b[i] + s;
}

// x[n + k] = w_k (M_k dx - b[n + k])   (one warp per row k)
__global__ void recover_kernel(const double* __restrict__ K, const double* __restrict__ w,
                               const int32_t* __restrict__ Mptr, const int32_t* __restrict__ Mcol,
                               const int32_t* __restrict__ Mslot, const double* __restrict__ b, int n, int mm,
                               double* __restrict__ x) {
  const int k = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x % 32;
  if (k >= mm) return;
  double s = 0.0;
  for (int q = Mptr[k] + lane; q < Mptr[k + 1]; q += 32) s += K[Mslot[q]] * x[Mcol[q]];
  s = warp_sum(s);
  if (lane == 0) x[n + k] = w[k] * (s - b[n + k]);
}

// ------------------------------------------------------------------ helpers
struct Stat {
  double p50 = 0, p95 = 0;
};
Stat stat(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  auto q = [&](double f) { return v[std::min(v.size() - 1, size_t(f * (v.size() - 1) + 0.5))]; };
  return {q(0.5), q(0.95)};
}

template <class T>
CudaBuffer<T> upload(const std::vector<T>& h, cudaStream_t s) {
  CudaBuffer<T> d(std::max<size_t>(h.size(), 1));
  if (!h.empty()) d.copy_from_host(h.data(), h.size(), s);
  return d;
}

std::vector<double> download(const double* d, size_t n, cudaStream_t s) {
  std::vector<double> h(n);
  CUPIQP_CUDA_CHECK(cudaMemcpyAsync(h.data(), d, sizeof(double) * n, cudaMemcpyDeviceToHost, s));
  CUPIQP_CUDA_CHECK(cudaStreamSynchronize(s));
  return h;
}

// ------------------------------------------------------------------ reduced solver
enum class Variant { Dense = 0, Sparse = 1, SparseSkip = 2 };
enum class Chol { Cusolver = 0, Tiled = 1 };
const char* variant_name(Variant v) {
  return v == Variant::Dense ? "a_dense_syrk" : v == Variant::Sparse ? "b_sparse" : "c_sparse_skip0";
}

struct Reduced {
  int n = 0, p = 0, m = 0, mm = 0, dim = 0, nnzP = 0, nnzM = 0;
  std::vector<int32_t> hPptr, hProw, hPcol, hPslot, hMptr, hMrow, hMcol, hMslot, hDslot, hCptr, hCrow, hCpos;
  std::vector<int32_t> hKbase, hCoff;
  CudaBuffer<int32_t> Pptr, Prow, Pcol, Pslot, Mptr, Mrow, Mcol, Mslot, Dslot, Cptr, Crow, Cpos, Kbase, Coff, Kidx;
  CudaBuffer<double> w, H, H0, Md, work, dinv;
  CudaBuffer<int> info, info_solve;
  cublasHandle_t cublas = nullptr;
  cusolverDnHandle_t cusolver = nullptr;
  CudaBuffer<unsigned char> cublas_ws;
  int lwork = 0;

  void init(const CsrPattern& kp, int n_, int p_, int m_, cudaStream_t s) {
    n = n_; p = p_; m = m_; mm = p + m; dim = n + mm;
    hPptr.assign(n + 1, 0);
    hMptr.assign(mm + 1, 0);
    hDslot.assign(mm, -1);
    for (int r = 0; r < dim; ++r) {
      for (int e = kp.indptr[r]; e < kp.indptr[r + 1]; ++e) {
        const int c = kp.indices[e];
        if (r < n) {
          if (c <= r) { hProw.push_back(r); hPcol.push_back(c); hPslot.push_back(e); }
          if (c >= n) throw std::runtime_error("unexpected upper off-diagonal entry in KKT x block");
        } else {
          const int k = r - n;
          if (c == r) hDslot[k] = e;
          else if (c < n) { hMrow.push_back(k); hMcol.push_back(c); hMslot.push_back(e); }
          else throw std::runtime_error("unexpected entry in KKT y/z diagonal block");
        }
      }
      if (r < n) hPptr[r + 1] = int(hProw.size());
      else hMptr[r - n + 1] = int(hMrow.size());
    }
    for (int k = 0; k < mm; ++k)
      if (hDslot[k] < 0) throw std::runtime_error("missing y/z diagonal");
    nnzP = int(hProw.size());
    nnzM = int(hMrow.size());
    // CSC of M with the position of each entry in M's row storage.
    hCptr.assign(n + 1, 0);
    for (int q = 0; q < nnzM; ++q) hCptr[hMcol[q] + 1]++;
    for (int i = 0; i < n; ++i) hCptr[i + 1] += hCptr[i];
    hCrow.resize(nnzM);
    hCpos.resize(nnzM);
    std::vector<int> fillp(hCptr.begin(), hCptr.end() - 1);
    for (int q = 0; q < nnzM; ++q) {
      int at = fillp[hMcol[q]]++;
      hCrow[at] = hMrow[q];
      hCpos[at] = q;
    }
    // Off-diagonal entries of each KKT row n + k are contiguous in K.
    hKbase.assign(mm, 0);
    for (int k = 0; k < mm; ++k) {
      hKbase[k] = hMptr[k] < hMptr[k + 1] ? hMslot[hMptr[k]] : 0;
      for (int q = hMptr[k]; q < hMptr[k + 1]; ++q)
        if (hMslot[q] != hKbase[k] + (q - hMptr[k])) throw std::runtime_error("KKT row entries not contiguous");
    }
    hCoff.resize(nnzM);
    for (int e = 0; e < nnzM; ++e) hCoff[e] = hCpos[e] - hMptr[hCrow[e]];
    Kbase = upload(hKbase, s); Coff = upload(hCoff, s); Kidx = upload(kp.indices, s);
    Pptr = upload(hPptr, s); Prow = upload(hProw, s); Pcol = upload(hPcol, s); Pslot = upload(hPslot, s);
    Mptr = upload(hMptr, s); Mrow = upload(hMrow, s); Mcol = upload(hMcol, s); Mslot = upload(hMslot, s);
    Dslot = upload(hDslot, s); Cptr = upload(hCptr, s); Crow = upload(hCrow, s); Cpos = upload(hCpos, s);
    w.allocate(mm);
    H.allocate(size_t(n) * n);
    H0.allocate(size_t(n) * n);
    Md.allocate(size_t(mm) * n);
    Md.zero_async(s);
    H.zero_async(s);
    info.allocate(1);
    info_solve.allocate(1);
    dinv.allocate(n);
    PROBE_CUBLAS_CHECK(cublasCreate(&cublas));
    cublas_ws.allocate(size_t(32) << 20);
    PROBE_CUBLAS_CHECK(cublasSetWorkspace(cublas, cublas_ws.data(), cublas_ws.size()));
    PROBE_CUSOLVER_CHECK(cusolverDnCreate(&cusolver));
    PROBE_CUSOLVER_CHECK(cusolverDnDpotrf_bufferSize(cusolver, CUBLAS_FILL_MODE_LOWER, n, H.data(), n, &lwork));
    work.allocate(std::max(lwork, 1));
    const size_t shm = size_t(kWarps + 1) * n * sizeof(double);
    if (shm > 48 * 1024) {
      CUPIQP_CUDA_CHECK(cudaFuncSetAttribute(syrk_sparse_kernel<kWarps, false>,
                                             cudaFuncAttributeMaxDynamicSharedMemorySize, int(shm)));
      CUPIQP_CUDA_CHECK(cudaFuncSetAttribute(syrk_sparse_kernel<kWarps, true>,
                                             cudaFuncAttributeMaxDynamicSharedMemorySize, int(shm)));
    }
  }
  ~Reduced() {
    if (cublas) cublasDestroy(cublas);
    if (cusolver) cusolverDnDestroy(cusolver);
  }

  void assemble(Variant v, const double* K, cudaStream_t s) {
    weights_kernel<<<(mm + 255) / 256, 256, 0, s>>>(K, Dslot.data(), mm, w.data());
    CUPIQP_CUDA_CHECK_LAUNCH();
    if (v == Variant::Dense) {
      scatter_dense_kernel<<<(nnzM + 255) / 256, 256, 0, s>>>(K, w.data(), Mrow.data(), Mcol.data(), Mslot.data(),
                                                              nnzM, mm, Md.data());
      CUPIQP_CUDA_CHECK_LAUNCH();
      CUPIQP_CUDA_CHECK(cudaMemsetAsync(H.data(), 0, sizeof(double) * size_t(n) * n, s));
      scatter_P_kernel<<<(nnzP + 255) / 256, 256, 0, s>>>(K, Prow.data(), Pcol.data(), Pslot.data(), nnzP, n,
                                                          H.data());
      CUPIQP_CUDA_CHECK_LAUNCH();
      const double one = 1.0;
      PROBE_CUBLAS_CHECK(cublasSetStream(cublas, s));
      PROBE_CUBLAS_CHECK(cublasDsyrk(cublas, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, n, mm, &one, Md.data(), mm, &one,
                                     H.data(), n));
    } else {
      const size_t shm = size_t(kWarps + 1) * n * sizeof(double);
      if (v == Variant::Sparse)
        syrk_sparse_kernel<kWarps, false><<<n, kWarps * 32, shm, s>>>(
            K, Kidx.data(), w.data(), Pptr.data(), Pcol.data(), Pslot.data(), Kbase.data(), Cptr.data(),
            Crow.data(), Coff.data(), n, H.data());
      else
        syrk_sparse_kernel<kWarps, true><<<n, kWarps * 32, shm, s>>>(
            K, Kidx.data(), w.data(), Pptr.data(), Pcol.data(), Pslot.data(), Kbase.data(), Cptr.data(),
            Crow.data(), Coff.data(), n, H.data());
      CUPIQP_CUDA_CHECK_LAUNCH();
    }
  }
  void factor(cudaStream_t s, Chol c = Chol::Cusolver) {
    if (c == Chol::Tiled) {
      const int N = (n + kTile - 1) / kTile;
      CUPIQP_CUDA_CHECK(cudaMemsetAsync(info.data(), 0, sizeof(int), s));
      for (int kk = 0; kk < N; ++kk) {
        chol_diag_kernel<<<1, dim3(kTile, kTile), 0, s>>>(H.data(), n, kk, dinv.data(), info.data());
        CUPIQP_CUDA_CHECK_LAUNCH();
        const int mt = N - kk - 1;
        if (mt == 0) break;
        chol_trsm_kernel<<<mt, dim3(kTile, kTile), 0, s>>>(H.data(), n, kk, dinv.data());
        CUPIQP_CUDA_CHECK_LAUNCH();
        chol_update_kernel<<<dim3(mt * (kTile / kSub), mt), dim3(kTile, kSub), 0, s>>>(H.data(), n, kk);
        CUPIQP_CUDA_CHECK_LAUNCH();
      }
      return;
    }
    PROBE_CUSOLVER_CHECK(cusolverDnSetStream(cusolver, s));
    PROBE_CUSOLVER_CHECK(
        cusolverDnDpotrf(cusolver, CUBLAS_FILL_MODE_LOWER, n, H.data(), n, work.data(), lwork, info.data()));
  }
  // x (dim) = solution of K x = b using the factored H.
  void solve(const double* K, const double* b, double* x, cudaStream_t s) {
    reduced_rhs_kernel<<<(n * 32 + 255) / 256, 256, 0, s>>>(K, w.data(), Cptr.data(), Crow.data(), Cpos.data(),
                                                            Mslot.data(), b, n, x);
    CUPIQP_CUDA_CHECK_LAUNCH();
    PROBE_CUSOLVER_CHECK(cusolverDnSetStream(cusolver, s));
    PROBE_CUSOLVER_CHECK(cusolverDnDpotrs(cusolver, CUBLAS_FILL_MODE_LOWER, n, 1, H.data(), n, x, n, info_solve.data()));
    recover_kernel<<<int((int64_t(mm) * 32 + 255) / 256), 256, 0, s>>>(K, w.data(), Mptr.data(), Mcol.data(),
                                                                       Mslot.data(), b, n, mm, x);
    CUPIQP_CUDA_CHECK_LAUNCH();
  }
};

// ------------------------------------------------------------------ CPU accuracy (long double)
struct HostK {
  int n = 0, dim = 0;
  const CsrPattern* kp = nullptr;
  // y = K x with K symmetric: P block stored in full, A / G only in the lower part.
  template <class R>
  std::vector<R> apply(const std::vector<double>& v, const std::vector<R>& x) const {
    std::vector<R> y(dim, R(0));
    for (int r = 0; r < dim; ++r)
      for (int e = kp->indptr[r]; e < kp->indptr[r + 1]; ++e) {
        const int c = kp->indices[e];
        y[r] += R(v[e]) * x[c];
        if (r >= n && c != r) y[c] += R(v[e]) * x[r];
      }
    return y;
  }
  long double norm_inf(const std::vector<double>& v) const {
    std::vector<long double> rs(dim, 0.0L);
    for (int r = 0; r < dim; ++r)
      for (int e = kp->indptr[r]; e < kp->indptr[r + 1]; ++e) {
        const int c = kp->indices[e];
        rs[r] += std::fabs((long double)v[e]);
        if (r >= n && c != r) rs[c] += std::fabs((long double)v[e]);
      }
    return *std::max_element(rs.begin(), rs.end());
  }
  long double rel_residual(const std::vector<double>& v, long double Knorm, const std::vector<double>& x,
                           const std::vector<double>& b) const {
    std::vector<long double> xl(x.begin(), x.end());
    auto y = apply<long double>(v, xl);
    long double r = 0, xn = 0, bn = 0;
    for (int i = 0; i < dim; ++i) {
      if (!std::isfinite(x[i])) return INFINITY;
      r = std::max(r, std::fabs(y[i] - (long double)b[i]));
      xn = std::max(xn, std::fabs((long double)x[i]));
      bn = std::max(bn, std::fabs((long double)b[i]));
    }
    return r / (Knorm * xn + bn);
  }
};

// Componentwise (Oettli-Prager) backward error: max_i |K x - b|_i / (|K| |x| + |b|)_i.
long double cw_backward(const HostK& hk, const std::vector<double>& v, const std::vector<double>& x,
                        const std::vector<double>& b) {
  const CsrPattern& kp = *hk.kp;
  std::vector<long double> y(hk.dim, 0.0L), a(hk.dim, 0.0L);
  for (int r = 0; r < hk.dim; ++r)
    for (int e = kp.indptr[r]; e < kp.indptr[r + 1]; ++e) {
      const int c = kp.indices[e];
      const long double ve = v[e];
      y[r] += ve * x[c];
      a[r] += std::fabs(ve * x[c]);
      if (r >= hk.n && c != r) {
        y[c] += ve * x[r];
        a[c] += std::fabs(ve * x[r]);
      }
    }
  long double w = 0;
  for (int i = 0; i < hk.dim; ++i) {
    if (!std::isfinite(x[i])) return INFINITY;
    const long double den = a[i] + std::fabs((long double)b[i]);
    const long double num = std::fabs(y[i] - (long double)b[i]);
    if (den > 0) w = std::max(w, num / den);
    else if (num > 0) return INFINITY;
  }
  return w;
}

double rel_diff(const std::vector<double>& a, const std::vector<double>& b, int lo, int hi) {
  double num = 0, den = 0;
  for (int i = lo; i < hi; ++i) {
    num = std::max(num, std::fabs(a[i] - b[i]));
    den = std::max(den, std::fabs(a[i]));
  }
  return den > 0 ? num / den : num;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    std::string data_path, label = "data", out_dir = ".";
    std::vector<int> cases = {0};
    int reps = 100, time_every = 1;
    bool do_ir = true;
    for (int i = 1; i < argc; ++i) {
      std::string a = argv[i];
      if (a == "--data") data_path = argv[++i];
      else if (a == "--label") label = argv[++i];
      else if (a == "--out-dir") out_dir = argv[++i];
      else if (a == "--reps") reps = std::stoi(argv[++i]);
      else if (a == "--time-every") time_every = std::stoi(argv[++i]);
      else if (a == "--no-ir") do_ir = false;
      else if (a == "--cases") {
        cases.clear();
        std::stringstream ss(argv[++i]);
        std::string t;
        while (std::getline(ss, t, ',')) cases.push_back(std::stoi(t));
      } else throw std::runtime_error("unknown argument " + a);
    }
    if (data_path.empty()) throw std::runtime_error("--data is required");
    bench::Snapshots g = bench::load_snapshots(data_path);
    const int n = g.n, p = g.p, m = g.m, dim = n + p + m;
    KktHostMaps maps = build_kkt_host_maps(g.P, g.A, g.C);
    const CsrPattern& kp = maps.kkt;
    const int nnzK = kp.nnz();
    std::fprintf(stderr, "[%s] n=%d p=%d m=%d dim=%d nnzK=%d\n", label.c_str(), n, p, m, dim, nnzK);

    CudaStream stream = CudaStream::create();
    cudaStream_t s = stream.get();
    Counters counters;

    // Device copy of the snapshot values (for SparseSolver::setup).
    CudaBuffer<double> packets(g.total);

    // KKT on the device for both methods.
    CudaBuffer<int32_t> d_indptr = upload(kp.indptr, s), d_indices = upload(kp.indices, s);
    CudaBuffer<double> Kd(nnzK);
    CudssSolver cudss(&counters);
    cudss.init(dim, nnzK, d_indptr.data(), d_indices.data(), Kd.data(), false, s);
    bool analyzed = false;
    Reduced red;
    red.init(kp, n, p, m, s);
    std::fprintf(stderr, "[%s] reduced: nnzP(lower)=%d nnzM=%d potrf lwork=%d\n", label.c_str(), red.nnzP,
                 red.nnzM, red.lwork);
    HostK hk{n, dim, &kp};

    CudaBuffer<double> b1(dim), b2(dim), xB(dim), xB2(dim), corr(dim), Hsave(size_t(n) * n);
    std::mt19937_64 rng(12345);
    std::uniform_real_distribution<double> U(-1.0, 1.0);
    std::vector<double> rhs_rand(dim);
    for (auto& v : rhs_rand) v = U(rng);

    std::ofstream acc_csv(out_dir + "/accuracy_" + label + ".csv");
    acc_csv << "label,case,k,iter,status,factor_retries,rho_factored,delta_factored,rho_after,delta_after,rhs,variant,potrf_info,res_A,res_B,res_B_ir,cw_A,cw_B,cw_B_ir,"
               "reldiff_AB,reldiff_AB_x,reldiff_AB_yz,cholL_diag_ratio,nnzM_nonzero,H_dense_vs_sparse_reldiff\n";
    std::ofstream tim_csv(out_dir + "/timing_" + label + ".csv");
    tim_csv << "label,case,k,op,eager_p50_ms,eager_p95_ms,graph_p50_ms,graph_p95_ms\n";
    acc_csv.precision(6);
    tim_csv.precision(6);

    CudaEvent e0 = CudaEvent::create(true), e1 = CudaEvent::create(true);
    auto measure = [&](const std::string& case_tag, const std::string& name,
                       const std::function<void(cudaStream_t)>& op) {
      for (int w = 0; w < 5; ++w) op(s);
      CUPIQP_CUDA_CHECK(cudaStreamSynchronize(s));
      std::vector<double> gpu;
      for (int i = 0; i < reps; ++i) {
        e0.record(s);
        op(s);
        e1.record(s);
        CUPIQP_CUDA_CHECK(cudaEventSynchronize(e1.get()));
        float ms = 0;
        CUPIQP_CUDA_CHECK(cudaEventElapsedTime(&ms, e0.get(), e1.get()));
        gpu.push_back(ms);
      }
      Stat eg = stat(gpu), gg;
      std::string graph_err;
      try {
        GraphExec ge;
        ge.instantiate(capture(s, op, &counters));
        for (int w = 0; w < 5; ++w) ge.launch(s, &counters);
        CUPIQP_CUDA_CHECK(cudaStreamSynchronize(s));
        std::vector<double> gv;
        for (int i = 0; i < reps; ++i) {
          e0.record(s);
          ge.launch(s, &counters);
          e1.record(s);
          CUPIQP_CUDA_CHECK(cudaEventSynchronize(e1.get()));
          float ms = 0;
          CUPIQP_CUDA_CHECK(cudaEventElapsedTime(&ms, e0.get(), e1.get()));
          gv.push_back(ms);
        }
        gg = stat(gv);
      } catch (const std::exception& ex) {
        graph_err = ex.what();
        cudaGetLastError();
        gg = {NAN, NAN};
      }
      tim_csv << label << "," << case_tag << "," << name << "," << eg.p50 << "," << eg.p95 << "," << gg.p50 << ","
              << gg.p95 << "\n";
      if (!graph_err.empty()) std::fprintf(stderr, "  graph capture failed for %s: %s\n", name.c_str(), graph_err.c_str());
    };

    for (int c : cases) {
      if (c >= g.cases) throw std::runtime_error("case out of range");
      packets.copy_from_host(g.case_values(c), g.total, s);
      CUPIQP_CUDA_CHECK(cudaStreamSynchronize(s));
      SparseSolver solver;
      auto& st = solver.settings();
      st.eps_abs = 1e-8;
      st.eps_duality_gap_abs = 1e-8;
      st.eps_rel = 0.0;
      st.eps_duality_gap_rel = 0.0;
      st.preconditioner_iter = 10;
      solver.setup(g.P, g.view(packets.data()), &g.A, &g.C);
      solver.solve();
      Info full = solver.info();
      std::fprintf(stderr, "[%s] case %d: %s in %d iterations\n", label.c_str(), c, status_name(full.status),
                   full.iter);
      auto dbg = solver.debug_buffers();
      auto find = [&](const std::string& nm) -> const DebugBuffer& {
        for (auto& d : dbg)
          if (d.name == nm) return d;
        throw std::runtime_error("no debug buffer " + nm);
      };
      const DebugBuffer &dK = find("kkt.values"), &dRp = find("res.primal"), &dRd = find("res.dual");
      if (dK.cols != nnzK) throw std::runtime_error("kkt.values size differs from the rebuilt KKT pattern");
      if (dRd.cols != p + 2 * m) throw std::runtime_error("expected both h_l and h_u present");

      const int kmax = full.iter;
      // rho / delta factored in iteration k are the values info() reports
      // after iteration k - 1 (rho_init / delta_init for k = 1); delta is
      // also read off K's y diagonal (kdelta) and must agree.
      double rho_factored = st.rho_init, delta_prev = st.delta_init;
      for (int k = 1; k <= kmax; ++k) {
        st.max_iter = k;
        solver.solve();
        Info inf = solver.info();
        CUPIQP_CUDA_CHECK(cudaMemcpyAsync(Kd.data(), dK.ptr, sizeof(double) * nnzK, cudaMemcpyDeviceToDevice,
                                          solver.stream()));
        CUPIQP_CUDA_CHECK(cudaStreamSynchronize(solver.stream()));
        std::vector<double> Kh = download(Kd.data(), nnzK, s);
        auto rp = download(static_cast<const double*>(dRp.ptr), dRp.cols, solver.stream());
        auto rd = download(static_cast<const double*>(dRd.ptr), dRd.cols, solver.stream());
        // Sanity: every y diagonal holds the same value (-delta of the factored iteration).
        const double kdelta = p > 0 ? -Kh[maps.diag_y[0]] : 0.0;
        for (int k2 = 0; k2 < p; ++k2)
          if (Kh[maps.diag_y[k2]] != -kdelta) throw std::runtime_error("y diagonal not constant: pattern mismatch?");
        if (kdelta != delta_prev) std::fprintf(stderr, "  note: K delta %.3e != previous info.delta %.3e\n", kdelta, delta_prev);
        int nnzM_nonzero = 0;
        for (int q = 0; q < red.nnzM; ++q) nnzM_nonzero += Kh[red.hMslot[q]] != 0.0;

        // Right-hand sides.
        // Realistic rhs: the IPM residuals of this state, [r_x | r_y | r_zu - r_zl].
        std::vector<double> rhs_res(dim);
        for (int i = 0; i < n; ++i) rhs_res[i] = rp[i];
        for (int i = 0; i < p; ++i) rhs_res[n + i] = rd[i];
        for (int i = 0; i < m; ++i) rhs_res[n + p + i] = rd[p + m + i] - rd[p + i];
        const long double Knorm = hk.norm_inf(Kh);

        if (!analyzed) {
          cudss.analyze(s);
          analyzed = true;
        }
        cudss.factor(s);
        // Reduced: dense H for the comparison, then the sparse variants.
        red.assemble(Variant::Dense, Kd.data(), s);
        auto Hdense = download(red.H.data(), size_t(n) * n, s);
        red.assemble(Variant::Sparse, Kd.data(), s);
        auto Hsparse = download(red.H.data(), size_t(n) * n, s);
        red.assemble(Variant::SparseSkip, Kd.data(), s);
        auto Hskip = download(red.H.data(), size_t(n) * n, s);
        double hdiff = 0, hmax = 0;
        bool skip_identical = true;
        for (int j = 0; j < n; ++j)
          for (int i = j; i < n; ++i) {
            size_t at = size_t(j) * n + i;
            hdiff = std::max(hdiff, std::fabs(Hdense[at] - Hsparse[at]));
            hmax = std::max(hmax, std::fabs(Hsparse[at]));
            skip_identical = skip_identical && (Hsparse[at] == Hskip[at]);
          }
        const double h_rel = hdiff / hmax;
        if (!skip_identical) std::fprintf(stderr, "  note: (b) and (c) H differ bitwise at k=%d\n", k);
        // Determinism check: rerun (b) and compare bitwise.
        red.assemble(Variant::Sparse, Kd.data(), s);
        auto Hsparse2 = download(red.H.data(), size_t(n) * n, s);
        for (size_t t = 0; t < Hsparse.size(); ++t)
          if (Hsparse[t] != Hsparse2[t] && !(std::isnan(Hsparse[t]) && std::isnan(Hsparse2[t])))
            throw std::runtime_error("sparse assembly not deterministic");

        struct RhsCase {
          const char* name;
          const std::vector<double>* b;
        } rhss[] = {{"random", &rhs_rand}, {"residual", &rhs_res}};
        struct Config {
          Variant v;
          Chol c;
          const char* name;
        } configs[] = {{Variant::Sparse, Chol::Cusolver, "b_sparse+potrf"},
                       {Variant::Dense, Chol::Cusolver, "a_dense_syrk+potrf"},
                       {Variant::Sparse, Chol::Tiled, "b_sparse+tiled_chol"}};
        for (const Config& cfg : configs) {
          red.assemble(cfg.v, Kd.data(), s);
          red.factor(s, cfg.c);
          int pinfo = 0;
          CUPIQP_CUDA_CHECK(cudaMemcpyAsync(&pinfo, red.info.data(), sizeof(int), cudaMemcpyDeviceToHost, s));
          auto L = download(red.H.data(), size_t(n) * n, s);
          double dmin = INFINITY, dmax = 0;
          for (int i = 0; i < n; ++i) {
            double d = std::fabs(L[size_t(i) * n + i]);
            dmin = std::min(dmin, d);
            dmax = std::max(dmax, d);
          }
          const double lratio = (dmax / dmin) * (dmax / dmin);
          for (auto& rc : rhss) {
            b1.copy_from_host(rc.b->data(), dim, s);
            CUPIQP_CUDA_CHECK(cudaMemcpyAsync(cudss.rhs(), b1.data(), sizeof(double) * dim,
                                              cudaMemcpyDeviceToDevice, s));
            cudss.solve(s);
            auto xA = download(cudss.sol(), dim, s);
            red.solve(Kd.data(), b1.data(), xB.data(), s);
            auto xb = download(xB.data(), dim, s);
            const long double resA = hk.rel_residual(Kh, Knorm, xA, *rc.b);
            const long double resB = hk.rel_residual(Kh, Knorm, xb, *rc.b);
            long double resBir = NAN, cwBir = NAN;
            const long double cwA = cw_backward(hk, Kh, xA, *rc.b), cwB = cw_backward(hk, Kh, xb, *rc.b);
            if (do_ir) {
              // One refinement step: r = b - K xB in FP64 (CPU), correction by the reduced solve.
              auto Kx = hk.apply<double>(Kh, xb);
              std::vector<double> r(dim);
              for (int i = 0; i < dim; ++i) r[i] = (*rc.b)[i] - Kx[i];
              corr.copy_from_host(r.data(), dim, s);
              red.solve(Kd.data(), corr.data(), xB2.data(), s);
              auto dx = download(xB2.data(), dim, s);
              std::vector<double> x2(dim);
              for (int i = 0; i < dim; ++i) x2[i] = xb[i] + dx[i];
              resBir = hk.rel_residual(Kh, Knorm, x2, *rc.b);
              cwBir = cw_backward(hk, Kh, x2, *rc.b);
            }
            acc_csv << label << "," << c << "," << k << "," << inf.iter << "," << status_name(inf.status) << ","
                    << inf.factor_retries << "," << rho_factored << "," << kdelta << "," << inf.rho << "," << inf.delta
                    << "," << rc.name << ","
                    << cfg.name << "," << pinfo << "," << double(resA) << "," << double(resB) << ","
                    << double(resBir) << "," << double(cwA) << "," << double(cwB) << "," << double(cwBir) << ","
                    << rel_diff(xA, xb, 0, dim) << "," << rel_diff(xA, xb, 0, n) << ","
                    << rel_diff(xA, xb, n, dim) << "," << lratio << "," << nnzM_nonzero << "," << h_rel << "\n";
            if (rc.name == std::string("residual"))
              std::fprintf(stderr,
                           "  %-20s k=%2d rho=%.1e delta=%.1e potrf=%d resA=%.2e resB=%.2e resB_ir=%.2e cwA=%.2e cwB=%.2e "
                           "cwBir=%.2e diff=%.2e Lratio=%.1e Hab=%.1e\n",
                           cfg.name, k, rho_factored, kdelta, pinfo, double(resA), double(resB), double(resBir), double(cwA),
                           double(cwB), double(cwBir),
                           rel_diff(xA, xb, 0, dim), lratio, h_rel);
          }
        }
        acc_csv.flush();
        rho_factored = inf.rho;
        delta_prev = inf.delta;

        // ---------------------------------------------------------- timing
        if (time_every > 0 && ((k - 1) % time_every == 0 || k == kmax)) {
          const std::string tag = std::to_string(c) + "," + std::to_string(k);
          b1.copy_from_host(rhs_rand.data(), dim, s);
          b2.copy_from_host(rhs_res.data(), dim, s);
          CUPIQP_CUDA_CHECK(cudaMemcpyAsync(cudss.rhs(), b1.data(), sizeof(double) * dim, cudaMemcpyDeviceToDevice, s));
          measure(tag, "cudss_factor", [&](cudaStream_t st2) { cudss.factor(st2); });
          measure(tag, "cudss_solve", [&](cudaStream_t st2) { cudss.solve(st2); });
          measure(tag, "cudss_iter_1f2s", [&](cudaStream_t st2) {
            cudss.factor(st2);
            cudss.solve(st2);
            cudss.solve(st2);
          });
          for (Variant v : {Variant::Dense, Variant::Sparse, Variant::SparseSkip})
            measure(tag, std::string("assemble_") + variant_name(v),
                    [&](cudaStream_t st2) { red.assemble(v, Kd.data(), st2); });
          red.assemble(Variant::Sparse, Kd.data(), s);
          CUPIQP_CUDA_CHECK(cudaMemcpyAsync(Hsave.data(), red.H.data(), sizeof(double) * n * n,
                                            cudaMemcpyDeviceToDevice, s));
          measure(tag, "copyH", [&](cudaStream_t st2) {
            CUPIQP_CUDA_CHECK(cudaMemcpyAsync(red.H.data(), Hsave.data(), sizeof(double) * n * n,
                                              cudaMemcpyDeviceToDevice, st2));
          });
          measure(tag, "copyH_potrf", [&](cudaStream_t st2) {
            CUPIQP_CUDA_CHECK(cudaMemcpyAsync(red.H.data(), Hsave.data(), sizeof(double) * n * n,
                                              cudaMemcpyDeviceToDevice, st2));
            red.factor(st2);
          });
          measure(tag, "copyH_tiled_chol", [&](cudaStream_t st2) {
            CUPIQP_CUDA_CHECK(cudaMemcpyAsync(red.H.data(), Hsave.data(), sizeof(double) * n * n,
                                              cudaMemcpyDeviceToDevice, st2));
            red.factor(st2, Chol::Tiled);
          });
          measure(tag, "reduced_solve_recover", [&](cudaStream_t st2) { red.solve(Kd.data(), b1.data(), xB.data(), st2); });
          for (Variant v : {Variant::Dense, Variant::Sparse, Variant::SparseSkip})
            measure(tag, std::string("reduced_iter_1a1f2s_") + variant_name(v), [&](cudaStream_t st2) {
              red.assemble(v, Kd.data(), st2);
              red.factor(st2);
              red.solve(Kd.data(), b1.data(), xB.data(), st2);
              red.solve(Kd.data(), b2.data(), xB2.data(), st2);
            });
          for (Variant v : {Variant::Sparse, Variant::SparseSkip})
            measure(tag, std::string("reduced_iter_1a1f2s_") + variant_name(v) + "_tiled_chol", [&](cudaStream_t st2) {
              red.assemble(v, Kd.data(), st2);
              red.factor(st2, Chol::Tiled);
              red.solve(Kd.data(), b1.data(), xB.data(), st2);
              red.solve(Kd.data(), b2.data(), xB2.data(), st2);
            });
          tim_csv.flush();
        }
      }
    }
    std::fprintf(stderr, "[%s] done\n", label.c_str());
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
