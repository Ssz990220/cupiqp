// Device helpers of the IPM kernels: ports of the module-level @wp.func
// helpers of cupiqp/solver_kernels.py (finite_value, step_candidate,
// calculate_step, calculate_mu, calculate_sigma, apply_finetune,
// boundary_shift) and the Warp float min/max semantics.
//
// Warp's wp.min / wp.max / wp.clamp on floats lower to fmin / fmax (a NaN
// operand is ignored unless both are NaN), and so do wp.tile_min / tile_max.
// The helpers below keep those semantics; common/reduce.cuh MinOp / MaxOp use
// ordered compares instead, so the block min/max here use FminOp / FmaxOp.
//
// Block helpers must be reached by every thread of the block (they contain
// __syncthreads); the thread mapping is the reference one: thread i of the
// block visits k = i, i + BLOCK, i + 2 BLOCK, ...
#pragma once

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

#include "common/multiblock.hpp"
#include "common/reduce.cuh"
#include "common/views.hpp"
#include "ipm/kernels/ipm_kernels.hpp"

namespace cupiqp::ipm_detail {

struct FminOp {
  template <class T>
  __device__ __forceinline__ T operator()(const T& a, const T& b) const { return ::fmin(a, b); }
};
struct FmaxOp {
  template <class T>
  __device__ __forceinline__ T operator()(const T& a, const T& b) const { return ::fmax(a, b); }
};

template <int BLOCK, class T>
__device__ __forceinline__ T block_fmin(T v) { return block_reduce<BLOCK>(v, FminOp{}); }
template <int BLOCK, class T>
__device__ __forceinline__ T block_fmax(T v) { return block_reduce<BLOCK>(v, FmaxOp{}); }

// Warp wp.min / wp.max / wp.clamp on floats.
template <class T>
__device__ __forceinline__ T wmin(T a, T b) { return ::fmin(a, b); }
template <class T>
__device__ __forceinline__ T wmax(T a, T b) { return ::fmax(a, b); }
template <class T>
__device__ __forceinline__ T wclamp(T x, T lo, T hi) { return ::fmin(::fmax(lo, x), hi); }

// Machine epsilon of the dtype (np.finfo(dtype).eps).
template <class T> struct MachineEps;
template <> struct MachineEps<double> { static constexpr double value = 2.220446049250313e-16; };
template <> struct MachineEps<float> { static constexpr float value = 1.1920928955078125e-07f; };

// v where mask marks a finite bound, else 0.
template <class T>
__device__ __forceinline__ T finite_value(T v, T mask) { return mask > T(0.5) ? v : T(0); }

// alpha = -b / a when a < 0, else 1.
template <class T>
__device__ __forceinline__ T step_candidate(T a, T b) { return a < T(0) ? -b / a : T(1); }

template <class T>
__device__ __forceinline__ T step_candidate_masked(T a, T b, T mask) {
  // wp.where evaluates both operands; the select result is the same.
  T cand = step_candidate(a, b);
  return mask > T(0.5) ? cand : T(1);
}

// Step lengths (alpha_s, alpha_z) of problem b over finite entries, times tau.
template <int BLOCK, class T>
__device__ __forceinline__ void calculate_step(RowsView<T> s_all, RowsView<T> z_all,
                                               RowsView<T> finite_mask_all, RowsView<T> step_s_all,
                                               RowsView<T> step_z_all, T tau, int32_t num_ineq,
                                               int64_t b, int i, T& alpha_s, T& alpha_z) {
  T min_s = T(1);
  T min_z = T(1);
  for (int k = i; k < num_ineq; k += BLOCK) {
    T mask = finite_mask_all(b, k);
    min_s = wmin(min_s, step_candidate_masked(step_s_all(b, k), s_all(b, k), mask));
    min_z = wmin(min_z, step_candidate_masked(step_z_all(b, k), z_all(b, k), mask));
  }
  T red_s = block_fmin<BLOCK>(min_s);
  T red_z = block_fmin<BLOCK>(min_z);
  alpha_s = red_s * tau;
  alpha_z = red_z * tau;
}

// mu = sum_k(s_k * z_k * mask_k) / num_finite_bounds[b], 0 without finite bounds.
template <int BLOCK, class T>
__device__ __forceinline__ T calculate_mu(RowsView<T> s_all, RowsView<T> z_all,
                                          RowsView<T> finite_mask_all, const T* num_finite_bounds,
                                          int32_t num_ineq, int64_t b, int i) {
  T acc = T(0);
  for (int k = i; k < num_ineq; k += BLOCK) acc += s_all(b, k) * z_all(b, k) * finite_mask_all(b, k);
  T total = block_sum<BLOCK>(acc);
  T mu = T(0);
  if (num_finite_bounds[b] > T(0)) mu = total / num_finite_bounds[b];
  return mu;
}

// sigma = clip(mu_aff / mu, 0, 1)^3, 0 when mu * num_finite_bounds <= 0.
template <int BLOCK, class T>
__device__ __forceinline__ T calculate_sigma(RowsView<T> s_all, RowsView<T> z_all,
                                             RowsView<T> step_s_all, RowsView<T> step_z_all,
                                             RowsView<T> finite_mask_all,
                                             const T* num_finite_bounds, T alpha_s, T alpha_z, T mu,
                                             int32_t num_ineq, int64_t b, int i) {
  T acc = T(0);
  for (int k = i; k < num_ineq; k += BLOCK) {
    acc += (s_all(b, k) + alpha_s * step_s_all(b, k)) * (z_all(b, k) + alpha_z * step_z_all(b, k)) *
           finite_mask_all(b, k);
  }
  T total = block_sum<BLOCK>(acc);
  T sigma = T(0);
  T denominator = mu * num_finite_bounds[b];
  if (denominator > T(0)) {
    T val = wclamp(total / denominator, T(0), T(1));
    sigma = val * val * val;
  }
  return sigma;
}

// Regularization finetuning of problem b if finetune_mask[b].
template <class T>
__device__ __forceinline__ void apply_finetune(const bool* finetune_mask, T* reg_limit,
                                               int32_t* no_primal_update, int32_t* no_dual_update,
                                               const T* settings, int64_t b) {
  if (finetune_mask[b]) {
    reg_limit[b] = settings[SF_REG_FINETUNE_LOWER_LIMIT];
    no_primal_update[b] = 0;
    no_dual_update[b] = 0;
  }
}

// z += eps on finite entries with z < eps; z = 0 on infinite entries.
template <int BLOCK, class T>
__device__ __forceinline__ void boundary_shift(RowsView<T> z_all, RowsView<T> finite_mask_all,
                                               T eps, int32_t num_ineq, int64_t b, int i) {
  for (int k = i; k < num_ineq; k += BLOCK) {
    if (finite_mask_all(b, k) > T(0.5)) {
      if (z_all(b, k) < eps) z_all(b, k) = z_all(b, k) + eps;
    } else {
      z_all(b, k) = T(0);
    }
  }
}

// Blocks per problem and block size of a multi-block reduction launch over
// rows of `width` elements carrying K values: the block size is the reference
// reduction_block_dim of one block's slice (of the whole row when nb == 1).
struct MbLaunch {
  int nb;
  int block;
};
inline MbLaunch mb_launch(const ReductionWorkspace& ws, int32_t B, int64_t width, int K) {
  const int nb = blocks_per_problem(ws, B, width, K);
  return {nb, multiblock_block_dim(width, nb)};
}

// Element-wise launch geometry (Warp launches use 256-thread blocks).
inline constexpr int kElementwiseBlock = 256;

}  // namespace cupiqp::ipm_detail
