// M6 kkt_system: forward condensed-KKT kernels (version 1, 1:1 with the Warp
// kernels named in kkt_system_kernels.hpp) and their launchers.
#include "linalg/kkt_system/kkt_system_kernels.hpp"

#include <cuda_runtime.h>

#include <cmath>

#include "common/check.hpp"
#include "common/multiblock.cuh"
#include "common/reduce.cuh"

namespace cupiqp {
namespace kkt_kernels {
namespace {

constexpr int kBlock = 256;

// Reference Warp atomic_max for float64 (warp/native/builtin.h): CAS loop on
// the bit pattern with fmax, so a NaN value never replaces a number.
__device__ __forceinline__ void atomic_max_warp(double* address, double val) {
  unsigned long long* address_as_ull = reinterpret_cast<unsigned long long*>(address);
  unsigned long long old = *address_as_ull;
  unsigned long long assumed;
  do {
    assumed = old;
    unsigned long long max_as_ull = __double_as_longlong(fmax(__longlong_as_double(assumed), val));
    if (max_as_ull == assumed) return;
    old = atomicCAS(address_as_ull, assumed, max_as_ull);
  } while (assumed != old);
}

// Reference update_w_delta_inv (wp.func).
template <class T>
__device__ __forceinline__ T update_w_delta_inv(const RowMat<const T>& s_all, const RowMat<const T>& z_all,
                                                const RowMat<const T>& finite_mask_all, const RowMat<T>& m_s_all,
                                                const RowMat<T>& m_z_inv_all, const RowMat<T>& w_delta_inv_all,
                                                T delta, int64_t b, int64_t i) {
  if (finite_mask_all(b, i) > T(0.5)) {
    T s = s_all(b, i);
    T z_inv = T(1.0) / z_all(b, i);
    T w = T(1.0) / (s * z_inv + delta);
    m_s_all(b, i) = s;
    m_z_inv_all(b, i) = z_inv;
    w_delta_inv_all(b, i) = w;
    return w;
  } else {
    m_s_all(b, i) = T(0.0);
    m_z_inv_all(b, i) = T(0.0);
    w_delta_inv_all(b, i) = T(0.0);
    return T(0.0);
  }
}

template <class T>
__global__ void update_regularizations_kernel(int32_t B, int32_t nx, int32_t nz, bool has_h_l, bool has_h_u,
                                              bool has_x_l, bool has_x_u, int32_t off_hu, int32_t off_xl,
                                              int32_t off_xu, RowMat<const T> s_all, RowMat<const T> z_all,
                                              RowMat<const T> finite_mask_all, RowMat<const T> x_b_scaling,
                                              const T* rho_in, const T* delta_in, RowMat<T> m_s_all,
                                              RowMat<T> m_z_inv_all, RowMat<T> w_delta_inv_all, T* rho_out,
                                              T* delta_out, RowMat<T> x_reg, RowMat<T> z_reg, RowMat<T> z_reg_inv) {
  const int64_t W = int64_t(nx) + nz;
  const int64_t tid = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tid >= int64_t(B) * W) return;
  const int64_t b = tid / W;
  const int64_t t = tid % W;
  T delta = delta_in[b];
  if (t == 0) {
    rho_out[b] = rho_in[b];
    delta_out[b] = delta;
  }
  if (t < nx) {
    T box_weight = T(0.0);
    if (has_x_u)
      box_weight += update_w_delta_inv(s_all, z_all, finite_mask_all, m_s_all, m_z_inv_all, w_delta_inv_all, delta,
                                       b, off_xu + t);
    if (has_x_l)
      box_weight += update_w_delta_inv(s_all, z_all, finite_mask_all, m_s_all, m_z_inv_all, w_delta_inv_all, delta,
                                       b, off_xl + t);
    T xb_scaling = x_b_scaling(b, t);
    x_reg(b, t) = rho_in[b] + xb_scaling * xb_scaling * box_weight;
  } else if (t < W) {
    const int64_t tz = t - nx;
    T tmp = T(0.0);
    if (has_h_u)
      tmp += update_w_delta_inv(s_all, z_all, finite_mask_all, m_s_all, m_z_inv_all, w_delta_inv_all, delta, b,
                                off_hu + tz);
    if (has_h_l)
      tmp += update_w_delta_inv(s_all, z_all, finite_mask_all, m_s_all, m_z_inv_all, w_delta_inv_all, delta, b, tz);
    if (tmp > T(0.0)) {  // this row of G is active
      z_reg(b, tz) = T(1.0) / tmp;
      z_reg_inv(b, tz) = tmp;
    } else {
      z_reg(b, tz) = T(0.0);
      z_reg_inv(b, tz) = T(0.0);
    }
  }
}

template <class T>
__global__ void eliminate_slacks_kernel(int32_t B, int32_t W, RowMat<const T> rhs_z_all, RowMat<const T> rhs_s_all,
                                        RowMat<const T> m_z_inv_all, RowMat<T> updated_rhs_z_all) {
  const int64_t tid = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tid >= int64_t(B) * W) return;
  const int64_t b = tid / W, i = tid % W;
  updated_rhs_z_all(b, i) = -m_z_inv_all(b, i) * rhs_s_all(b, i) + rhs_z_all(b, i);
}

template <class T>
__global__ void eliminate_duals_kernel(int32_t B, int32_t nx, int32_t nz, bool has_h_l, bool has_h_u, bool has_x_l,
                                       bool has_x_u, RowMat<const T> rhs_x, RowMat<const T> w_bu_delta_inv,
                                       RowMat<const T> w_bl_delta_inv, RowMat<const T> x_b_scaling,
                                       RowMat<const T> rhs_z_bu, RowMat<const T> rhs_z_bl, RowMat<T> rhs_x_updated,
                                       RowMat<const T> w_u_delta_inv, RowMat<const T> w_l_delta_inv,
                                       RowMat<const T> rhs_z_u, RowMat<const T> rhs_z_l, RowMat<const T> z_reg,
                                       RowMat<T> rhs_z_updated) {
  const int64_t W = int64_t(nx) + nz;
  const int64_t tid = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tid >= int64_t(B) * W) return;
  const int64_t b = tid / W, t = tid % W;
  if (t < nx) {
    T xb_scaling = x_b_scaling(b, t);
    T box_term = T(0.0);
    if (has_x_u) box_term += w_bu_delta_inv(b, t) * rhs_z_bu(b, t);
    if (has_x_l) box_term -= w_bl_delta_inv(b, t) * rhs_z_bl(b, t);
    rhs_x_updated(b, t) = rhs_x(b, t) + xb_scaling * box_term;
  } else if (t < W) {
    const int64_t tz = t - nx;
    T val = T(0.0);
    if (has_h_u) val += w_u_delta_inv(b, tz) * rhs_z_u(b, tz);
    if (has_h_l) val -= w_l_delta_inv(b, tz) * rhs_z_l(b, tz);
    T zr = z_reg(b, tz);  // explicit augmented diagonal magnitude = 1 / weight
    if (zr > T(0.0)) {
      rhs_z_updated(b, tz) = val * zr;
    } else {
      rhs_z_updated(b, tz) = T(0.0);
    }
  }
}

template <class T>
__global__ void recover_duals_kernel(int32_t B, int32_t num_hu, int32_t num_hl, int32_t num_xu, int32_t num_xl,
                                     RowMat<const T> G_dx, RowMat<const T> lhs_x, RowMat<const T> w_u_delta_inv,
                                     RowMat<const T> rhs_z_u, RowMat<T> lhs_z_u, RowMat<const T> w_l_delta_inv,
                                     RowMat<const T> rhs_z_l, RowMat<T> lhs_z_l, RowMat<const T> w_bu_delta_inv,
                                     RowMat<const T> m_z_bu_inv, RowMat<const T> rhs_z_bu, RowMat<const T> rhs_s_bu,
                                     RowMat<T> lhs_z_bu, RowMat<const T> w_bl_delta_inv, RowMat<const T> m_z_bl_inv,
                                     RowMat<const T> rhs_z_bl, RowMat<const T> rhs_s_bl, RowMat<T> lhs_z_bl,
                                     RowMat<const T> x_b_scaling) {
  const int64_t W = int64_t(num_hu) + num_hl + num_xu + num_xl;
  const int64_t tid = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tid >= int64_t(B) * W) return;
  const int64_t b = tid / W, t = tid % W;
  if (t < num_hu) {
    lhs_z_u(b, t) = (G_dx(b, t) - rhs_z_u(b, t)) * w_u_delta_inv(b, t);
  } else if (t < int64_t(num_hu) + num_hl) {
    const int64_t j = t - num_hu;
    lhs_z_l(b, j) = (-G_dx(b, j) - rhs_z_l(b, j)) * w_l_delta_inv(b, j);
  } else if (t < int64_t(num_hu) + num_hl + num_xu) {
    const int64_t j = t - num_hu - num_hl;
    lhs_z_bu(b, j) =
        (x_b_scaling(b, j) * lhs_x(b, j) - rhs_z_bu(b, j) + m_z_bu_inv(b, j) * rhs_s_bu(b, j)) * w_bu_delta_inv(b, j);
  } else {
    const int64_t j = t - num_hu - num_hl - num_xu;
    lhs_z_bl(b, j) =
        -(x_b_scaling(b, j) * lhs_x(b, j) + rhs_z_bl(b, j) - m_z_bl_inv(b, j) * rhs_s_bl(b, j)) * w_bl_delta_inv(b, j);
  }
}

template <class T>
__global__ void recover_slacks_kernel(int32_t B, int32_t W, RowMat<const T> rhs_s_all, RowMat<const T> lhs_z_all,
                                      RowMat<const T> m_s_all, RowMat<const T> m_z_inv_all, RowMat<T> lhs_s_all) {
  const int64_t tid = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tid >= int64_t(B) * W) return;
  const int64_t b = tid / W, i = tid % W;
  lhs_s_all(b, i) = m_z_inv_all(b, i) * ((-m_s_all(b, i)) * lhs_z_all(b, i) + rhs_s_all(b, i));
}

template <class T>
__global__ void condensed_kkt_combine_kernel(int32_t B, int32_t n, int32_t p, int32_t m, RowMat<const T> P_x,
                                             RowMat<const T> x_reg, RowMat<const T> lhs_x, RowMat<const T> AT_y,
                                             RowMat<const T> GT_z, RowMat<const T> A_x, RowMat<const T> lhs_y,
                                             const T* delta, RowMat<const T> G_x, RowMat<const T> active_G_row,
                                             RowMat<const T> z_reg, RowMat<const T> lhs_z, RowMat<const T> rhs_x,
                                             RowMat<const T> rhs_y, RowMat<const T> rhs_z, int32_t residual,
                                             RowMat<T> out_x, RowMat<T> out_y, RowMat<T> out_z, T* norm_out) {
  const int64_t W = int64_t(n) + p + m;
  const int64_t tid = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tid >= int64_t(B) * W) return;
  const int64_t b = tid / W, t = tid % W;
  if (t < n) {
    T v = P_x(b, t) + x_reg(b, t) * lhs_x(b, t);
    if (p > 0) v = v + AT_y(b, t);
    if (m > 0) v = v + GT_z(b, t);
    if (residual == 1) {
      v = rhs_x(b, t) - v;
      atomic_max_warp(norm_out, fabs(v));
    }
    out_x(b, t) = v;
  } else if (t < int64_t(n) + p) {
    const int64_t k = t - n;
    T v = A_x(b, k) - delta[b] * lhs_y(b, k);
    if (residual == 1) {
      v = rhs_y(b, k) - v;
      atomic_max_warp(norm_out, fabs(v));
    }
    out_y(b, k) = v;
  } else {
    const int64_t k = t - n - p;
    T zr = z_reg(b, k);
    T diag_term = lhs_z(b, k);
    if (zr > T(0.0)) diag_term = zr * lhs_z(b, k);
    T v = active_G_row(b, k) * G_x(b, k) - diag_term;
    if (residual == 1) {
      v = rhs_z(b, k) - v;
      atomic_max_warp(norm_out, fabs(v));
    }
    out_z(b, k) = v;
  }
}

template <class T>
__global__ void mul_kernel(int32_t B, int32_t W, RowMat<const T> a, RowMat<const T> c, RowMat<T> out) {
  const int64_t tid = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tid >= int64_t(B) * W) return;
  const int64_t b = tid / W, i = tid % W;
  out(b, i) = a(b, i) * c(b, i);
}

template <class T>
__global__ void refinement_begin_kernel(const int32_t* settings_i, T* rhs_norm, T* new_error, T* error,
                                        int32_t* step, int32_t* loop) {
  rhs_norm[0] = T(0.0);
  new_error[0] = T(0.0);
  error[0] = T(INFINITY);
  step[0] = 0;
  loop[0] = settings_i[SI_IR_MAX_ITER] > 0 ? 1 : 0;
}

template <class T>
__global__ void refinement_decide_kernel(const T* settings_f, const int32_t* settings_i, const T* rhs_norm,
                                         T* new_error, T* error, int32_t* step, int32_t* accept, int32_t* correct,
                                         int32_t* loop) {
  int32_t i = step[0];
  T previous = error[0];
  T current = new_error[0];
  new_error[0] = T(0.0);
  error[0] = current;
  T tol = settings_f[SF_IR_EPS_ABS] + settings_f[SF_IR_EPS_REL] * rhs_norm[0];
  int32_t stop = 0;
  int32_t take = 0;
  if (!isfinite(current)) {
    stop = 1;
  } else if (current <= tol) {
    stop = 1;
    if (i > 0) take = 1;
  } else if (i > 0) {
    T rate = previous / current;
    if (rate < settings_f[SF_IR_MIN_IMPROVEMENT_RATE]) {
      stop = 1;
      if (rate > T(1.0)) take = 1;
    } else {
      take = 1;
    }
  }
  accept[0] = take;
  correct[0] = 1 - stop;
  step[0] = i + 1;
  loop[0] = (stop == 0 && i + 1 < settings_i[SI_IR_MAX_ITER]) ? 1 : 0;
}

template <class T>
__global__ void refinement_accept_kernel(int32_t B, int32_t n, int32_t p, int32_t m, const int32_t* accept,
                                         RowMat<const T> cand_x, RowMat<const T> cand_y, RowMat<const T> cand_z,
                                         RowMat<T> lhs_x, RowMat<T> lhs_y, RowMat<T> lhs_z) {
  const int64_t W = int64_t(n) + p + m;
  const int64_t tid = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tid >= int64_t(B) * W) return;
  if (accept[0] == 0) return;
  const int64_t b = tid / W, t = tid % W;
  if (t < n) {
    lhs_x(b, t) = cand_x(b, t);
  } else if (t < int64_t(n) + p) {
    lhs_y(b, t - n) = cand_y(b, t - n);
  } else {
    lhs_z(b, t - n - p) = cand_z(b, t - n - p);
  }
}

template <class T>
__global__ void refinement_rhs_norm_kernel(int32_t B, int32_t n, int32_t p, int32_t m, RowMat<const T> rhs_x,
                                           RowMat<const T> rhs_y, RowMat<const T> rhs_z, T* rhs_norm) {
  const int64_t W = int64_t(n) + p + m;
  const int64_t tid = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tid >= int64_t(B) * W) return;
  const int64_t b = tid / W, t = tid % W;
  T v;
  if (t < n) {
    v = rhs_x(b, t);
  } else if (t < int64_t(n) + p) {
    v = rhs_y(b, t - n);
  } else {
    v = rhs_z(b, t - n - p);
  }
  atomic_max_warp(rhs_norm, fabs(v));
}

template <class T>
__global__ void refinement_add_kernel(int32_t B, int32_t n, int32_t p, int32_t m, RowMat<const T> lhs_x,
                                      RowMat<const T> lhs_y, RowMat<const T> lhs_z, RowMat<T> cand_x,
                                      RowMat<T> cand_y, RowMat<T> cand_z) {
  const int64_t W = int64_t(n) + p + m;
  const int64_t tid = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tid >= int64_t(B) * W) return;
  const int64_t b = tid / W, t = tid % W;
  if (t < n) {
    cand_x(b, t) = cand_x(b, t) + lhs_x(b, t);
  } else if (t < int64_t(n) + p) {
    cand_y(b, t - n) = cand_y(b, t - n) + lhs_y(b, t - n);
  } else {
    cand_z(b, t - n - p) = cand_z(b, t - n - p) + lhs_z(b, t - n - p);
  }
}

// Multi-block reduction (common/multiblock.cuh): block (b, j) visits
// t = j * BLOCK + i, t += nb * BLOCK of the concatenated row (one block per
// problem, the reference mapping, when nb == 1).
template <int BLOCK, class T>
__global__ void __launch_bounds__(BLOCK)
solution_finite_kernel(MultiblockGrid g, int32_t n, int32_t p, int32_t m, RowMat<const T> x, RowMat<const T> y,
                       RowMat<const T> z, int32_t* status) {
  __shared__ MultiblockStorage<BLOCK, 1, int32_t> sm;
  const int64_t b = g.problem();
  const int64_t total = int64_t(n) + p + m;
  int32_t bad[1] = {0};
  for (int64_t t = g.first<BLOCK>(); t < total; t += g.stride<BLOCK>()) {
    T v;
    if (t < n) {
      v = x(b, t);
    } else if (t < int64_t(n) + p) {
      v = y(b, t - n);
    } else {
      v = z(b, t - n - p);
    }
    if (!isfinite(v)) bad[0] = 1;  // false for NaN, +inf and -inf
  }
  if (!multiblock_reduce<BLOCK, 1, reduce_kinds(ReduceKind::Max)>(g, bad, sm)) return;
  if (threadIdx.x == 0) status[b] = sm.result[0];
}

template <class T>
__global__ void assemble_rhs_kernel(int32_t B, int32_t n, int32_t p, int32_t m, RowMat<const T> x,
                                    RowMat<const T> y, RowMat<const T> z, RowMat<T> out) {
  const int64_t W = int64_t(n) + p + m;
  const int64_t tid = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tid >= int64_t(B) * W) return;
  const int64_t b = tid / W, t = tid % W;
  T v;
  if (t < n) {
    v = x(b, t);
  } else if (t < int64_t(n) + p) {
    v = y(b, t - n);
  } else {
    v = z(b, t - n - p);
  }
  out(b, t) = v;
}

template <class T>
__global__ void split_sol_kernel(int32_t B, int32_t n, int32_t p, int32_t m, RowMat<const T> in, RowMat<T> x,
                                 RowMat<T> y, RowMat<T> z) {
  const int64_t W = int64_t(n) + p + m;
  const int64_t tid = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tid >= int64_t(B) * W) return;
  const int64_t b = tid / W, t = tid % W;
  T v = in(b, t);
  if (t < n) {
    x(b, t) = v;
  } else if (t < int64_t(n) + p) {
    y(b, t - n) = v;
  } else {
    z(b, t - n - p) = v;
  }
}

}  // namespace

void update_regularizations(const Dims& d, CDMat s_all, CDMat z_all, CDMat finite_mask_all, CDMat x_b_scaling,
                            const double* rho_in, const double* delta_in, DMat m_s_all, DMat m_z_inv_all,
                            DMat w_delta_inv_all, double* rho_out, double* delta_out, DMat x_reg, DMat z_reg,
                            DMat z_reg_inv, cudaStream_t s) {
  const int64_t total = int64_t(d.B) * (int64_t(d.n) + d.m);
  if (total == 0) return;
  update_regularizations_kernel<double><<<grid_for(total, kBlock), kBlock, 0, s>>>(
      d.B, d.n, d.m, d.has_h_l(), d.has_h_u(), d.has_x_l(), d.has_x_u(), d.off_hu(), d.off_xl(), d.off_xu(), s_all,
      z_all, finite_mask_all, x_b_scaling, rho_in, delta_in, m_s_all, m_z_inv_all, w_delta_inv_all, rho_out,
      delta_out, x_reg, z_reg, z_reg_inv);
  CUPIQP_CUDA_CHECK_LAUNCH();
}

void eliminate_slacks(int32_t B, int32_t num_ineq, CDMat rhs_z_all, CDMat rhs_s_all, CDMat m_z_inv_all,
                      DMat updated_rhs_z_all, cudaStream_t s) {
  const int64_t total = int64_t(B) * num_ineq;
  if (total == 0) return;
  eliminate_slacks_kernel<double><<<grid_for(total, kBlock), kBlock, 0, s>>>(B, num_ineq, rhs_z_all, rhs_s_all,
                                                                             m_z_inv_all, updated_rhs_z_all);
  CUPIQP_CUDA_CHECK_LAUNCH();
}

void eliminate_duals(const Dims& d, CDMat rhs_x, CDMat w_bu_delta_inv, CDMat w_bl_delta_inv, CDMat x_b_scaling,
                     CDMat rhs_z_bu, CDMat rhs_z_bl, DMat rhs_x_updated, CDMat w_u_delta_inv,
                     CDMat w_l_delta_inv, CDMat rhs_z_u, CDMat rhs_z_l, CDMat z_reg, DMat rhs_z_updated,
                     cudaStream_t s) {
  const int64_t total = int64_t(d.B) * (int64_t(d.n) + d.m);
  if (total == 0) return;
  eliminate_duals_kernel<double><<<grid_for(total, kBlock), kBlock, 0, s>>>(
      d.B, d.n, d.m, d.has_h_l(), d.has_h_u(), d.has_x_l(), d.has_x_u(), rhs_x, w_bu_delta_inv, w_bl_delta_inv,
      x_b_scaling, rhs_z_bu, rhs_z_bl, rhs_x_updated, w_u_delta_inv, w_l_delta_inv, rhs_z_u, rhs_z_l, z_reg,
      rhs_z_updated);
  CUPIQP_CUDA_CHECK_LAUNCH();
}

void recover_duals(const Dims& d, CDMat G_dx, CDMat lhs_x, CDMat w_u_delta_inv, CDMat rhs_z_u, DMat lhs_z_u,
                   CDMat w_l_delta_inv, CDMat rhs_z_l, DMat lhs_z_l, CDMat w_bu_delta_inv, CDMat m_z_bu_inv,
                   CDMat rhs_z_bu, CDMat rhs_s_bu, DMat lhs_z_bu, CDMat w_bl_delta_inv, CDMat m_z_bl_inv,
                   CDMat rhs_z_bl, CDMat rhs_s_bl, DMat lhs_z_bl, CDMat x_b_scaling, cudaStream_t s) {
  const int64_t total = int64_t(d.B) * d.num_ineq();
  if (total == 0) return;
  recover_duals_kernel<double><<<grid_for(total, kBlock), kBlock, 0, s>>>(
      d.B, d.num_hu, d.num_hl, d.num_xu, d.num_xl, G_dx, lhs_x, w_u_delta_inv, rhs_z_u, lhs_z_u, w_l_delta_inv,
      rhs_z_l, lhs_z_l, w_bu_delta_inv, m_z_bu_inv, rhs_z_bu, rhs_s_bu, lhs_z_bu, w_bl_delta_inv, m_z_bl_inv,
      rhs_z_bl, rhs_s_bl, lhs_z_bl, x_b_scaling);
  CUPIQP_CUDA_CHECK_LAUNCH();
}

void recover_slacks(int32_t B, int32_t num_ineq, CDMat rhs_s_all, CDMat lhs_z_all, CDMat m_s_all,
                    CDMat m_z_inv_all, DMat lhs_s_all, cudaStream_t s) {
  const int64_t total = int64_t(B) * num_ineq;
  if (total == 0) return;
  recover_slacks_kernel<double><<<grid_for(total, kBlock), kBlock, 0, s>>>(B, num_ineq, rhs_s_all, lhs_z_all,
                                                                           m_s_all, m_z_inv_all, lhs_s_all);
  CUPIQP_CUDA_CHECK_LAUNCH();
}

void condensed_kkt_combine(int32_t B, int32_t n, int32_t p, int32_t m, CDMat P_x, CDMat x_reg, CDMat lhs_x,
                           CDMat AT_y, CDMat GT_z, CDMat A_x, CDMat lhs_y, const double* delta, CDMat G_x,
                           CDMat active_G_row, CDMat z_reg, CDMat lhs_z, CDMat rhs_x, CDMat rhs_y, CDMat rhs_z,
                           bool residual, DMat out_x, DMat out_y, DMat out_z, double* norm_out, cudaStream_t s) {
  const int64_t total = int64_t(B) * (int64_t(n) + p + m);
  if (total == 0) return;
  condensed_kkt_combine_kernel<double><<<grid_for(total, kBlock), kBlock, 0, s>>>(
      B, n, p, m, P_x, x_reg, lhs_x, AT_y, GT_z, A_x, lhs_y, delta, G_x, active_G_row, z_reg, lhs_z, rhs_x, rhs_y,
      rhs_z, residual ? 1 : 0, out_x, out_y, out_z, norm_out);
  CUPIQP_CUDA_CHECK_LAUNCH();
}

void mul(int32_t B, int32_t k, CDMat a, CDMat b, DMat out, cudaStream_t s) {
  const int64_t total = int64_t(B) * k;
  if (total == 0) return;
  mul_kernel<double><<<grid_for(total, kBlock), kBlock, 0, s>>>(B, k, a, b, out);
  CUPIQP_CUDA_CHECK_LAUNCH();
}

void refinement_begin(const int32_t* settings_i, double* rhs_norm, double* new_error, double* error, int32_t* step,
                      int32_t* loop, cudaStream_t s) {
  refinement_begin_kernel<double><<<1, 1, 0, s>>>(settings_i, rhs_norm, new_error, error, step, loop);
  CUPIQP_CUDA_CHECK_LAUNCH();
}

void refinement_decide(const double* settings_f, const int32_t* settings_i, const double* rhs_norm,
                       double* new_error, double* error, int32_t* step, int32_t* accept, int32_t* correct,
                       int32_t* loop, cudaStream_t s) {
  refinement_decide_kernel<double><<<1, 1, 0, s>>>(settings_f, settings_i, rhs_norm, new_error, error, step, accept,
                                                   correct, loop);
  CUPIQP_CUDA_CHECK_LAUNCH();
}

void refinement_accept(int32_t B, int32_t n, int32_t p, int32_t m, const int32_t* accept, CDMat cand_x,
                       CDMat cand_y, CDMat cand_z, DMat lhs_x, DMat lhs_y, DMat lhs_z, cudaStream_t s) {
  const int64_t total = int64_t(B) * (int64_t(n) + p + m);
  if (total == 0) return;
  refinement_accept_kernel<double><<<grid_for(total, kBlock), kBlock, 0, s>>>(B, n, p, m, accept, cand_x, cand_y,
                                                                              cand_z, lhs_x, lhs_y, lhs_z);
  CUPIQP_CUDA_CHECK_LAUNCH();
}

void refinement_rhs_norm(int32_t B, int32_t n, int32_t p, int32_t m, CDMat rhs_x, CDMat rhs_y, CDMat rhs_z,
                         double* rhs_norm, cudaStream_t s) {
  const int64_t total = int64_t(B) * (int64_t(n) + p + m);
  if (total == 0) return;
  refinement_rhs_norm_kernel<double><<<grid_for(total, kBlock), kBlock, 0, s>>>(B, n, p, m, rhs_x, rhs_y, rhs_z,
                                                                                rhs_norm);
  CUPIQP_CUDA_CHECK_LAUNCH();
}

void refinement_add(int32_t B, int32_t n, int32_t p, int32_t m, CDMat lhs_x, CDMat lhs_y, CDMat lhs_z,
                    DMat cand_x, DMat cand_y, DMat cand_z, cudaStream_t s) {
  const int64_t total = int64_t(B) * (int64_t(n) + p + m);
  if (total == 0) return;
  refinement_add_kernel<double><<<grid_for(total, kBlock), kBlock, 0, s>>>(B, n, p, m, lhs_x, lhs_y, lhs_z, cand_x,
                                                                           cand_y, cand_z);
  CUPIQP_CUDA_CHECK_LAUNCH();
}

void solution_finite(int32_t B, int32_t n, int32_t p, int32_t m, CDMat x, CDMat y, CDMat z, int32_t* status,
                     cudaStream_t s, const ReductionWorkspace& ws) {
  if (B == 0) return;
  const int64_t width = int64_t(n) + p + m;
  const int nb = blocks_per_problem(ws, B, width, 1);
  const MultiblockGrid g = multiblock_grid(ws, nb);
  dispatch_block(multiblock_block_dim(width, nb), [&](auto blk) {
    constexpr int BLOCK = decltype(blk)::value;
    solution_finite_kernel<BLOCK, double><<<multiblock_dim(B, nb), BLOCK, 0, s>>>(g, n, p, m, x, y, z, status);
  });
  CUPIQP_CUDA_CHECK_LAUNCH();
}

void assemble_rhs(int32_t B, int32_t n, int32_t p, int32_t m, CDMat x, CDMat y, CDMat z, DMat out,
                  cudaStream_t s) {
  const int64_t total = int64_t(B) * (int64_t(n) + p + m);
  if (total == 0) return;
  assemble_rhs_kernel<double><<<grid_for(total, kBlock), kBlock, 0, s>>>(B, n, p, m, x, y, z, out);
  CUPIQP_CUDA_CHECK_LAUNCH();
}

void split_sol(int32_t B, int32_t n, int32_t p, int32_t m, CDMat in, DMat x, DMat y, DMat z, cudaStream_t s) {
  const int64_t total = int64_t(B) * (int64_t(n) + p + m);
  if (total == 0) return;
  split_sol_kernel<double><<<grid_for(total, kBlock), kBlock, 0, s>>>(B, n, p, m, in, x, y, z);
  CUPIQP_CUDA_CHECK_LAUNCH();
}

}  // namespace kkt_kernels
}  // namespace cupiqp
