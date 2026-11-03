// M3 ruiz kernels. Version 1: each kernel mirrors one Warp kernel of
//   cupiqp/preconditioner_kernels.py and cupiqp/sparse/sparse_preconditioner_kernels.py
// (same thread mapping, formulas in the same order, same masks). Warp's
// wp.max / wp.min / wp.clamp on floats are fmax / fmin (NaN treated as
// missing), and wp.atomic_max on float64 is a CAS loop over that fmax; the
// helpers below reproduce both.
//
// Arrays are row-major (B, k). A 2-D Warp launch dim=(B, W) becomes a 1-D
// grid of B * W threads with (b, k) = (t / W, t % W).
//
// Device early exit (FEASIBILITY 3.5): every per-pass kernel takes
// `done` (nullable). When *done != 0 the kernel returns immediately, so the
// passes after convergence are no-ops. Only ruiz_conv_check_kernel sets it.
#pragma once

#include <cuda_runtime.h>

#include <cstdint>

#include "common/reduce.cuh"

namespace cupiqp::ruiz_detail {

// Warp wp.max / wp.min on floating point: C fmax / fmin.
struct FmaxOp {
  template <class T>
  __device__ __forceinline__ T operator()(const T& a, const T& b) const { return fmax(a, b); }
};

// Warp clamp(x, a, b) = fmin(fmax(a, x), b).
template <class T>
__device__ __forceinline__ T warp_clamp(T x, T lo, T hi) { return fmin(fmax(lo, x), hi); }

// Warp builtin.h atomic_max(double*, double): CAS loop over fmax.
__device__ __forceinline__ double atomic_fmax(double* address, double val) {
  unsigned long long* address_as_ull = reinterpret_cast<unsigned long long*>(address);
  unsigned long long old = *address_as_ull;
  unsigned long long assumed;
  do {
    assumed = old;
    unsigned long long max_as_ull = __double_as_longlong(fmax(__longlong_as_double(assumed), val));
    if (max_as_ull == assumed) return __longlong_as_double(old);
    old = atomicCAS(address_as_ull, assumed, max_as_ull);
  } while (assumed != old);
  return __longlong_as_double(old);
}

// Reference ruiz_limit_scaling: below min -> 1, above max -> max.
template <class T>
__device__ __forceinline__ T ruiz_limit_scaling(T d, T min_scaling, T max_scaling) {
  if (d < min_scaling) return T(1.0);
  if (d > max_scaling) return max_scaling;
  return d;
}

// Reference finite_value: v where mask marks a finite bound, else 0.
template <class T>
__device__ __forceinline__ T finite_value(T v, T mask) { return mask > T(0.5) ? v : T(0.0); }

__device__ __forceinline__ bool skipped(const int32_t* done) { return done != nullptr && *done != 0; }

struct Csr {
  const int32_t* indptr;
  const int32_t* indices;
  const int32_t* row_idx;
  int32_t nnz;
};

// Which blocks to touch (unscale_data skips replaced ones).
struct BlockFlags {
  bool P, c, A, G;
  bool b, h_l, h_u, x_l, x_u;
};

template <class T>
__global__ void ruiz_fill_kernel(T* out, int64_t count, T value) {
  int64_t t = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (t < count) out[t] = value;
}

// ------------------------------------------------------------ scale_data
// Reset the early-exit state at the start of scale_data.
__global__ void ruiz_begin_kernel(int32_t* done, int32_t* passes_run, double* conv_buf) {
  done[0] = 0;
  done[1] = 0;
  passes_run[0] = 0;
  conv_buf[0] = 0.0;
}

// Reference sparse_compute_row_inf_norm_kernel, dim (B, max(n, p, m)).
template <class T>
__global__ void ruiz_row_norm_kernel(int32_t B, int32_t n, int32_t p, int32_t m, int32_t R,
                                     const T* __restrict__ P_data, Csr P, const T* __restrict__ A_data, Csr A,
                                     const T* __restrict__ G_data, Csr G, const T* __restrict__ x_b_scaling,
                                     T* __restrict__ d_iter, T* __restrict__ d_b_iter, const int32_t* done) {
  int64_t t = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (t >= int64_t(B) * R) return;
  if (skipped(done)) return;
  const int32_t b = int32_t(t / R), j = int32_t(t % R);
  const int32_t W = n + p + m;
  if (j < n) {
    T v = T(0.0);
    const T* Pb = P_data + int64_t(b) * P.nnz;
    for (int32_t k = P.indptr[j]; k < P.indptr[j + 1]; ++k) v = fmax(v, fabs(Pb[k]));
    const T xbs = x_b_scaling[int64_t(b) * n + j];
    v = fmax(v, xbs);
    d_iter[int64_t(b) * W + j] = v;
    d_b_iter[int64_t(b) * n + j] = xbs;
  }
  if (p > 0 && j < p) {
    T v = T(0.0);
    const T* Ab = A_data + int64_t(b) * A.nnz;
    for (int32_t k = A.indptr[j]; k < A.indptr[j + 1]; ++k) v = fmax(v, fabs(Ab[k]));
    d_iter[int64_t(b) * W + n + j] = v;
  }
  if (m > 0 && j < m) {
    T v = T(0.0);
    const T* Gb = G_data + int64_t(b) * G.nnz;
    for (int32_t k = G.indptr[j]; k < G.indptr[j + 1]; ++k) v = fmax(v, fabs(Gb[k]));
    d_iter[int64_t(b) * W + n + p + j] = v;
  }
}

// Reference sparse_compute_col_inf_norm_kernel, dim (B, max(nnz_A, nnz_G)).
// Atomic fmax is order-independent, so the result is deterministic.
template <class T>
__global__ void ruiz_col_norm_kernel(int32_t B, int32_t n, int32_t p, int32_t m, int32_t K,
                                     const T* __restrict__ A_data, Csr A, const T* __restrict__ G_data, Csr G,
                                     T* d_iter, const int32_t* done) {
  int64_t t = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (t >= int64_t(B) * K) return;
  if (skipped(done)) return;
  const int32_t b = int32_t(t / K), k = int32_t(t % K);
  T* row = d_iter + int64_t(b) * (n + p + m);
  if (p > 0 && k < A.nnz) atomic_fmax(row + A.indices[k], fabs(A_data[int64_t(b) * A.nnz + k]));
  if (m > 0 && k < G.nnz) atomic_fmax(row + G.indices[k], fabs(G_data[int64_t(b) * G.nnz + k]));
}

// Reference clamp_rsqrt_kernel, dim (B, n + p + m).
template <class T>
__global__ void ruiz_clamp_rsqrt_kernel(int32_t B, int32_t n, int32_t W, T* __restrict__ d_iter,
                                        T* __restrict__ d_b_iter, T lo, T hi, const int32_t* done) {
  int64_t t = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (t >= int64_t(B) * W) return;
  if (skipped(done)) return;
  const int32_t b = int32_t(t / W), k = int32_t(t % W);
  T d = d_iter[t];
  d = ruiz_limit_scaling(d, lo, hi);
  d_iter[t] = T(1.0) / sqrt(d);
  if (k < n) {
    const int64_t i = int64_t(b) * n + k;
    T db = d_b_iter[i];
    db = ruiz_limit_scaling(db, lo, hi);
    d_b_iter[i] = T(1.0) / sqrt(db);
  }
}

// Reference sparse_scale_matrices_kernel, dim (B, max(n, p, m)). `d` is a
// (B, n + p + m) array [d_x | d_y | d_z]; cost_factor is (B,).
template <class T>
__global__ void ruiz_scale_matrices_kernel(int32_t B, int32_t n, int32_t p, int32_t m, int32_t R,
                                           T* __restrict__ P_data, Csr P, T* __restrict__ A_data, Csr A,
                                           T* __restrict__ G_data, Csr G, T* __restrict__ c,
                                           const T* __restrict__ d, const T* __restrict__ cost_factor,
                                           BlockFlags f, const int32_t* done) {
  int64_t t = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (t >= int64_t(B) * R) return;
  if (skipped(done)) return;
  const int32_t b = int32_t(t / R), r = int32_t(t % R);
  const T* d_x = d + int64_t(b) * (n + p + m);
  if (r < n) {
    const T d_xr = d_x[r];
    if (f.P) {
      T* Pb = P_data + int64_t(b) * P.nnz;
      for (int32_t k = P.indptr[r]; k < P.indptr[r + 1]; ++k) {
        const int32_t cc = P.indices[k];
        Pb[k] = Pb[k] * d_xr * d_x[cc] * cost_factor[b];
      }
    }
    if (f.c) {
      T* cb = c + int64_t(b) * n;
      cb[r] = cb[r] * d_xr * cost_factor[b];
    }
  }
  if (f.A && p > 0 && r < p) {
    const T d_yr = d_x[n + r];
    T* Ab = A_data + int64_t(b) * A.nnz;
    for (int32_t k = A.indptr[r]; k < A.indptr[r + 1]; ++k) {
      const int32_t cc = A.indices[k];
      Ab[k] = Ab[k] * d_yr * d_x[cc];
    }
  }
  if (f.G && m > 0 && r < m) {
    const T d_zr = d_x[n + p + r];
    T* Gb = G_data + int64_t(b) * G.nnz;
    for (int32_t k = G.indptr[r]; k < G.indptr[r + 1]; ++k) {
      const int32_t cc = G.indices[k];
      Gb[k] = Gb[k] * d_zr * d_x[cc];
    }
  }
}

// Reference accumulate_deltas_kernel, dim (B, n + p + m).
template <class T>
__global__ void ruiz_accumulate_deltas_kernel(int32_t B, int32_t n, int32_t W, T* __restrict__ delta,
                                              T* __restrict__ delta_b, T* __restrict__ x_b_scaling,
                                              const T* __restrict__ d_iter, const T* __restrict__ d_b_iter,
                                              const int32_t* done) {
  int64_t t = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (t >= int64_t(B) * W) return;
  if (skipped(done)) return;
  const int32_t b = int32_t(t / W), k = int32_t(t % W);
  const T di = d_iter[t];
  delta[t] = delta[t] * di;
  if (k < n) {
    const int64_t i = int64_t(b) * n + k;
    const T dbi = d_b_iter[i];
    delta_b[i] = delta_b[i] * dbi;
    x_b_scaling[i] = x_b_scaling[i] * dbi * di;
  }
}

// Reference sparse_P_norms_kernel, dim (B, nnz_P); norms zeroed before.
template <class T>
__global__ void ruiz_P_norms_kernel(int32_t B, int32_t n, const T* __restrict__ P_data, Csr P, T* norms,
                                    const int32_t* done) {
  int64_t t = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (t >= int64_t(B) * P.nnz) return;
  if (skipped(done)) return;
  const int32_t b = int32_t(t / P.nnz), k = int32_t(t % P.nnz);
  const T v = fabs(P_data[t]);
  atomic_fmax(norms + int64_t(b) * n + P.row_idx[k], v);
  atomic_fmax(norms + int64_t(b) * n + P.indices[k], v);
}

// Reference sparse_compute_gamma_kernel: one block per problem.
template <int BLOCK, class T>
__global__ void ruiz_compute_gamma_kernel(int32_t n, const T* __restrict__ norms, const T* __restrict__ c,
                                          T* __restrict__ gamma, T lo, T hi, const int32_t* done) {
  if (skipped(done)) return;  // uniform over the block
  const int32_t b = blockIdx.x, i = threadIdx.x;
  T acc_sum = T(0.0), acc_max = T(0.0);
  for (int32_t k = i; k < n; k += BLOCK) {
    acc_sum += norms[int64_t(b) * n + k];
    acc_max = fmax(acc_max, fabs(c[int64_t(b) * n + k]));
  }
  const T total = block_sum<BLOCK>(acc_sum);
  const T c_max = block_reduce<BLOCK>(acc_max, FmaxOp{});
  if (i == 0) {
    T g = total / T(n);
    g = warp_clamp(g, lo, hi);
    g = fmax(g, c_max);
    g = warp_clamp(g, lo, hi);
    gamma[b] = T(1.0) / g;
  }
}

// Reference sparse_apply_gamma_kernel, dim (B, max(nnz_P, n)).
template <class T>
__global__ void ruiz_apply_gamma_kernel(int32_t B, int32_t n, int32_t nnz_P, int32_t K, T* __restrict__ P_data,
                                        T* __restrict__ c, T* __restrict__ cost_scaling,
                                        const T* __restrict__ gamma, const int32_t* done) {
  int64_t t = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (t >= int64_t(B) * K) return;
  if (skipped(done)) return;
  const int32_t b = int32_t(t / K), i = int32_t(t % K);
  const T g = gamma[b];
  if (i < nnz_P) P_data[int64_t(b) * nnz_P + i] = P_data[int64_t(b) * nnz_P + i] * g;
  if (i < n) c[int64_t(b) * n + i] = c[int64_t(b) * n + i] * g;
  if (i == 0) cost_scaling[b] = cost_scaling[b] * g;
}

// Reference conv_check_kernel plus the host decision
// `if conv < convergence_tol: break`, moved to the device. One block per
// problem; the batch-wide max is accumulated in conv_buf (zeroed by
// ruiz_begin_kernel and re-zeroed here), and the last block to finish
// decides. done[0]: early-exit flag; done[1]: block ticket.
template <int BLOCK, class T>
__global__ void ruiz_conv_check_kernel(int32_t B, int32_t n, int32_t W, const T* __restrict__ d_iter,
                                       const T* __restrict__ d_b_iter, T* conv_buf, int32_t* done,
                                       int32_t* passes_run, T tol) {
  if (done[0] != 0) return;  // uniform over the grid (only the last block writes it, after all have read it)
  const int32_t b = blockIdx.x, i = threadIdx.x;
  if (b == 0 && i == 0) passes_run[0] += 1;
  T v = T(0.0);
  for (int32_t k = i; k < W; k += BLOCK) v = fmax(v, fabs(T(1.0) - d_iter[int64_t(b) * W + k]));
  for (int32_t k = i; k < n; k += BLOCK) v = fmax(v, fabs(T(1.0) - d_b_iter[int64_t(b) * n + k]));
  const T red = block_reduce<BLOCK>(v, FmaxOp{});
  if (i == 0) {
    atomic_fmax(conv_buf, red);
    __threadfence();
    const int32_t ticket = atomicAdd(done + 1, 1);
    if (ticket == B - 1) {
      __threadfence();
      const T conv = *reinterpret_cast<volatile T*>(conv_buf);
      if (conv < tol) done[0] = 1;
      conv_buf[0] = T(0.0);
      done[1] = 0;
    }
  }
}

// Reference finalize_and_scale_bounds_kernel
// (create_calc_scaling_inv_and_scale_bounds_kernel), dim (B, n+p+m+1+num_duals).
template <class T>
__global__ void ruiz_finalize_kernel(int32_t B, int32_t n, int32_t p, int32_t m, int32_t num_hl, int32_t num_hu,
                                     int32_t num_xl, int32_t num_xu, const T* __restrict__ delta,
                                     T* __restrict__ delta_inv, const T* __restrict__ delta_b,
                                     T* __restrict__ delta_b_inv, const T* __restrict__ cost_scaling,
                                     T* __restrict__ cost_scaling_inv, T* data_b, T* data_h_l, T* data_h_u,
                                     T* data_x_l, T* data_x_u, T* __restrict__ dual_res_unscale_factor,
                                     T* __restrict__ primal_res_unscale_factor) {
  const int32_t W = n + p + m;
  const int32_t num_duals = p + num_hl + num_hu + num_xl + num_xu;
  const int32_t K = W + 1 + num_duals;
  int64_t t = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (t >= int64_t(B) * K) return;
  const int32_t b = int32_t(t / K), k = int32_t(t % K);
  const T* db = delta + int64_t(b) * W;
  const T* dbb = delta_b + int64_t(b) * n;
  if (k < n) {
    const int64_t i = int64_t(b) * n + k;
    delta_inv[int64_t(b) * W + k] = T(1.0) / db[k];
    delta_b_inv[i] = T(1.0) / dbb[k];
    if (num_xl) data_x_l[i] = data_x_l[i] * dbb[k];
    if (num_xu) data_x_u[i] = data_x_u[i] * dbb[k];
    dual_res_unscale_factor[i] = T(1.0) / (cost_scaling[b] * db[k]);
  } else if (k < n + p) {
    delta_inv[int64_t(b) * W + k] = T(1.0) / db[k];
    const int64_t i = int64_t(b) * p + (k - n);
    data_b[i] = data_b[i] * db[k];
  } else if (k < W) {
    delta_inv[int64_t(b) * W + k] = T(1.0) / db[k];
    const int64_t i = int64_t(b) * m + (k - n - p);
    if (num_hl) data_h_l[i] = data_h_l[i] * db[k];
    if (num_hu) data_h_u[i] = data_h_u[i] * db[k];
  } else if (k < W + 1) {
    cost_scaling_inv[b] = T(1.0) / cost_scaling[b];
  } else {
    const int32_t j = k - (W + 1);
    T* out = primal_res_unscale_factor + int64_t(b) * num_duals;
    if (j < p) {
      out[j] = T(1.0) / db[n + j];
    } else if (j < p + num_hl) {
      out[j] = T(1.0) / db[n + p + (j - p)];
    } else if (j < p + num_hl + num_hu) {
      out[j] = T(1.0) / db[n + p + (j - p - num_hl)];
    } else if (j < p + num_hl + num_hu + num_xl) {
      out[j] = T(1.0) / dbb[j - p - num_hl - num_hu];
    } else {
      out[j] = T(1.0) / dbb[j - p - num_hl - num_hu - num_xl];
    }
  }
}

// ------------------------------------------------------------ bounds, x_b_scaling, reset
// Reference scale_bounds_kernel / unscale_bounds_kernel (identical form;
// called with (delta, delta_b) or (delta_inv, delta_b_inv)), dim (B, n+p+m).
template <class T>
__global__ void ruiz_bounds_kernel(int32_t B, int32_t n, int32_t p, int32_t m, const T* __restrict__ d,
                                   const T* __restrict__ d_b, T* data_b, T* data_h_l, T* data_h_u, T* data_x_l,
                                   T* data_x_u, BlockFlags f) {
  const int32_t W = n + p + m;
  int64_t t = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (t >= int64_t(B) * W) return;
  const int32_t b = int32_t(t / W), k = int32_t(t % W);
  if (k < n) {
    const int64_t i = int64_t(b) * n + k;
    const T dx = d_b[i];
    if (f.x_l) data_x_l[i] = data_x_l[i] * dx;
    if (f.x_u) data_x_u[i] = data_x_u[i] * dx;
  } else if (k < n + p) {
    const int64_t i = int64_t(b) * p + (k - n);
    if (f.b) data_b[i] = data_b[i] * d[t];
  } else {
    const int64_t i = int64_t(b) * m + (k - n - p);
    const T dz = d[t];
    if (f.h_l) data_h_l[i] = data_h_l[i] * dz;
    if (f.h_u) data_h_u[i] = data_h_u[i] * dz;
  }
}

// Reference wp.map(_product3, x, y, z, out=out): out = x * y * z over (B, n),
// with z the x-block of a (B, n+p+m) array.
template <class T>
__global__ void ruiz_product3_kernel(int32_t B, int32_t n, int32_t W, T* out, const T* x, const T* y,
                                     const T* z) {
  int64_t t = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (t >= int64_t(B) * n) return;
  const int32_t b = int32_t(t / n), k = int32_t(t % n);
  out[t] = x[t] * y[t] * z[int64_t(b) * W + k];
}

// Reference reset(): fill_(1.0) of every factor, x_b_scaling = init.
template <class T>
__global__ void ruiz_reset_kernel(int32_t B, int32_t n, int32_t W, int32_t num_duals, int32_t K, T* delta,
                                  T* delta_inv, T* delta_b, T* delta_b_inv, T* cost_scaling, T* cost_scaling_inv,
                                  T* x_b_scaling, const T* x_b_scaling_init, T* dual_res_unscale_factor,
                                  T* primal_res_unscale_factor) {
  int64_t t = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (t >= int64_t(B) * K) return;
  const int32_t b = int32_t(t / K), k = int32_t(t % K);
  if (k < W) {
    delta[int64_t(b) * W + k] = T(1.0);
    delta_inv[int64_t(b) * W + k] = T(1.0);
  }
  if (k < n) {
    const int64_t i = int64_t(b) * n + k;
    delta_b[i] = T(1.0);
    delta_b_inv[i] = T(1.0);
    x_b_scaling[i] = x_b_scaling_init[i];
    dual_res_unscale_factor[i] = T(1.0);
  }
  if (k < num_duals) primal_res_unscale_factor[int64_t(b) * num_duals + k] = T(1.0);
  if (k == 0) {
    cost_scaling[b] = T(1.0);
    cost_scaling_inv[b] = T(1.0);
  }
}

// ------------------------------------------------------------ solution, RHS norm
// Reference unscale_solution_kernel, dim (B, num_primal + num_dual), over the
// packed buffers [x | s_l | s_u | s_bl | s_bu] and [y | z_l | z_u | z_bl | z_bu].
template <class T>
__global__ void ruiz_solution_kernel(int32_t B, int32_t n, int32_t p, int32_t m, int32_t num_hl, int32_t num_hu,
                                     int32_t num_xl, int32_t num_xu, T* primal, T* dual,
                                     const T* __restrict__ delta, const T* __restrict__ delta_inv,
                                     const T* __restrict__ delta_b, const T* __restrict__ delta_b_inv,
                                     const T* __restrict__ cost_scaling_inv) {
  const int32_t num_primal = n + num_hl + num_hu + num_xl + num_xu;
  const int32_t num_dual = p + num_hl + num_hu + num_xl + num_xu;
  const int32_t K = num_primal + num_dual;
  int64_t tt = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tt >= int64_t(B) * K) return;
  const int32_t b = int32_t(tt / K), t = int32_t(tt % K);
  const int32_t W = n + p + m;
  const T* d = delta + int64_t(b) * W;
  const T* di = delta_inv + int64_t(b) * W;
  const T* d_b = delta_b + int64_t(b) * n;
  const T* d_bi = delta_b_inv + int64_t(b) * n;
  const T c_inv = cost_scaling_inv[b];
  T* pr = primal + int64_t(b) * num_primal;
  T* du = dual + int64_t(b) * num_dual;

  const int32_t end_x = n;
  const int32_t end_sl = end_x + num_hl;
  const int32_t end_su = end_sl + num_hu;
  const int32_t end_sbl = end_su + num_xl;
  const int32_t end_sbu = end_sbl + num_xu;
  const int32_t end_y = end_sbu + p;
  const int32_t end_zl = end_y + num_hl;
  const int32_t end_zu = end_zl + num_hu;
  const int32_t end_zbl = end_zu + num_xl;
  const int32_t end_zbu = end_zbl + num_xu;

  if (t < end_x) {
    pr[t] = pr[t] * d[t];
  } else if (t < end_sl) {
    const int32_t k = t - end_x;
    pr[t] = pr[t] * di[n + p + k];
  } else if (t < end_su) {
    const int32_t k = t - end_sl;
    pr[t] = pr[t] * di[n + p + k];
  } else if (t < end_sbl) {
    const int32_t k = t - end_su;
    pr[t] = pr[t] * d_bi[k];
  } else if (t < end_sbu) {
    const int32_t k = t - end_sbl;
    pr[t] = pr[t] * d_bi[k];
  } else if (t < end_y) {
    const int32_t k = t - end_sbu;
    du[k] = du[k] * d[n + k] * c_inv;
  } else if (t < end_zl) {
    const int32_t k = t - end_y;
    du[p + k] = du[p + k] * d[n + p + k] * c_inv;
  } else if (t < end_zu) {
    const int32_t k = t - end_zl;
    du[p + num_hl + k] = du[p + num_hl + k] * d[n + p + k] * c_inv;
  } else if (t < end_zbl) {
    const int32_t k = t - end_zu;
    du[p + num_hl + num_hu + k] = du[p + num_hl + num_hu + k] * d_b[k] * c_inv;
  } else if (t < end_zbu) {
    const int32_t k = t - end_zbl;
    du[p + num_hl + num_hu + num_xl + k] = du[p + num_hl + num_hu + num_xl + k] * d_b[k] * c_inv;
  }
}

// Reference compute_constraints_rhs_inf_norm_unscaled kernel: one block per problem.
template <int BLOCK, class T>
__global__ void ruiz_rhs_inf_norm_kernel(int32_t n, int32_t p, int32_t m, int32_t num_hl, int32_t num_hu,
                                         int32_t num_xl, int32_t num_xu, const T* __restrict__ delta_inv,
                                         const T* __restrict__ delta_b_inv, const T* data_b, const T* data_h_l,
                                         const T* data_h_u, const T* data_x_l, const T* data_x_u,
                                         const T* finite_mask_all, int32_t num_ineq, T* out) {
  const int32_t b = blockIdx.x, i = threadIdx.x;
  const int32_t W = n + p + m;
  const T* di = delta_inv + int64_t(b) * W;
  const T* dbi = delta_b_inv + int64_t(b) * n;
  const T* mask = finite_mask_all + int64_t(b) * num_ineq;
  const T* mhl = mask;
  const T* mhu = mask + num_hl;
  const T* mxl = mask + num_hl + num_hu;
  const T* mxu = mask + num_hl + num_hu + num_xl;
  T v = T(0.0);
  for (int32_t k = i; k < p; k += BLOCK) v = fmax(v, fabs(data_b[int64_t(b) * p + k] * di[n + k]));
  for (int32_t k = i; k < num_hu; k += BLOCK)
    v = fmax(v, fabs(finite_value(data_h_u[int64_t(b) * m + k], mhu[k]) * di[n + p + k]));
  for (int32_t k = i; k < num_hl; k += BLOCK)
    v = fmax(v, fabs(finite_value(data_h_l[int64_t(b) * m + k], mhl[k]) * di[n + p + k]));
  for (int32_t k = i; k < num_xu; k += BLOCK)
    v = fmax(v, fabs(finite_value(data_x_u[int64_t(b) * n + k], mxu[k]) * dbi[k]));
  for (int32_t k = i; k < num_xl; k += BLOCK)
    v = fmax(v, fabs(finite_value(data_x_l[int64_t(b) * n + k], mxl[k]) * dbi[k]));
  const T red = block_reduce<BLOCK>(v, FmaxOp{});
  if (i == 0) out[b] = red;
}

}  // namespace cupiqp::ruiz_detail
