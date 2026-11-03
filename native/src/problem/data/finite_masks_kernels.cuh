// Finite-bound mask kernels of module data (reference
// cupiqp/data_kernels.py create_finite_bound_masks_kernel).
//
// A lower bound is finite iff v > -PIQP_INF and an upper bound is finite iff
// v < PIQP_INF (exactly the reference comparisons: NaN gives 0 on both sides,
// a lower bound of +inf counts as finite, an upper bound of -inf counts as
// finite). Outputs, per problem b (row stride = width):
//   finite_mask_all (B, num_ineq) in the packed layout [hl | hu | xl | xu],
//   active_G_row    (B, m) = max(fl, fu) of each inequality row,
//   active_x_bound  (B, n) = max(fl, fu) of each variable,
//   num_finite_bounds (B,) = number of finite bounds of the problem.
// Absent sides (num_* == 0) are never read or written.
#pragma once

#include <cuda_runtime.h>

#include <cstdint>

#include "common/reduce.cuh"
#include "cupiqp/cupiqp.hpp"

namespace cupiqp {

// Shapes of the mask kernels (taken from Dims; B rows each).
struct FiniteMaskShape {
  int32_t B = 1, m = 0, n = 0;
  int32_t num_hl = 0, num_hu = 0, num_xl = 0, num_xu = 0;
  __host__ __device__ int32_t num_ineq() const { return num_hl + num_hu + num_xl + num_xu; }
};

template <class T>
__device__ __forceinline__ T finite_lower(T v) {
  return v > T(-PIQP_INF) ? T(1) : T(0);
}
template <class T>
__device__ __forceinline__ T finite_upper(T v) {
  return v < T(PIQP_INF) ? T(1) : T(0);
}
template <class T>
__device__ __forceinline__ T mask_max(T a, T b) {
  return a > b ? a : b;  // operands are 0 or 1
}

// Version 1: mirrors the Warp kernel. One thread per problem (launch with
// B threads), sequential loops over the m rows and the n variables.
template <class T>
__global__ void finite_bound_masks_serial_kernel(FiniteMaskShape sh, const T* __restrict__ h_l,
                                                 const T* __restrict__ h_u, const T* __restrict__ x_l,
                                                 const T* __restrict__ x_u, T* __restrict__ finite_mask_all,
                                                 T* __restrict__ active_G_row, T* __restrict__ active_x_bound,
                                                 T* __restrict__ num_finite_bounds) {
  const int32_t b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= sh.B) return;
  const int32_t m = sh.m, n = sh.n, ni = sh.num_ineq();
  const bool has_hl = sh.num_hl > 0, has_hu = sh.num_hu > 0;
  const bool has_xl = sh.num_xl > 0, has_xu = sh.num_xu > 0;
  T* fm = finite_mask_all + static_cast<int64_t>(b) * ni;
  const int32_t off_hl = 0;
  const int32_t off_hu = sh.num_hl;
  const int32_t off_xl = sh.num_hl + sh.num_hu;
  const int32_t off_xu = off_xl + sh.num_xl;

  T count = T(0);
  for (int32_t i = 0; i < m; ++i) {
    T fl = T(0), fu = T(0);
    if (has_hl) {
      fl = finite_lower(h_l[static_cast<int64_t>(b) * sh.num_hl + i]);
      fm[off_hl + i] = fl;
      count += fl;
    }
    if (has_hu) {
      fu = finite_upper(h_u[static_cast<int64_t>(b) * sh.num_hu + i]);
      fm[off_hu + i] = fu;
      count += fu;
    }
    active_G_row[static_cast<int64_t>(b) * m + i] = mask_max(fl, fu);
  }
  for (int32_t j = 0; j < n; ++j) {
    T fl = T(0), fu = T(0);
    if (has_xl) {
      fl = finite_lower(x_l[static_cast<int64_t>(b) * sh.num_xl + j]);
      fm[off_xl + j] = fl;
      count += fl;
    }
    if (has_xu) {
      fu = finite_upper(x_u[static_cast<int64_t>(b) * sh.num_xu + j]);
      fm[off_xu + j] = fu;
      count += fu;
    }
    active_x_bound[static_cast<int64_t>(b) * n + j] = mask_max(fl, fu);
  }
  num_finite_bounds[b] = count;
}

// Version 2: one block per problem, threads stride over the m rows and the n
// variables, the count is a block sum. The per-element formulas are those of
// version 1; the count is a sum of 0/1 values (exact in any order below 2^53),
// so the outputs are bitwise identical to version 1.
template <int BLOCK, class T>
__global__ void __launch_bounds__(BLOCK)
    finite_bound_masks_block_kernel(FiniteMaskShape sh, const T* __restrict__ h_l,
                                    const T* __restrict__ h_u, const T* __restrict__ x_l,
                                    const T* __restrict__ x_u, T* __restrict__ finite_mask_all,
                                    T* __restrict__ active_G_row, T* __restrict__ active_x_bound,
                                    T* __restrict__ num_finite_bounds) {
  const int32_t b = blockIdx.x;  // grid == B: no early return before block_sum
  const int32_t m = sh.m, n = sh.n, ni = sh.num_ineq();
  const bool has_hl = sh.num_hl > 0, has_hu = sh.num_hu > 0;
  const bool has_xl = sh.num_xl > 0, has_xu = sh.num_xu > 0;
  T* fm = finite_mask_all + static_cast<int64_t>(b) * ni;
  const int32_t off_hl = 0;
  const int32_t off_hu = sh.num_hl;
  const int32_t off_xl = sh.num_hl + sh.num_hu;
  const int32_t off_xu = off_xl + sh.num_xl;

  T count = T(0);
  for (int32_t i = threadIdx.x; i < m; i += BLOCK) {
    T fl = T(0), fu = T(0);
    if (has_hl) {
      fl = finite_lower(h_l[static_cast<int64_t>(b) * sh.num_hl + i]);
      fm[off_hl + i] = fl;
      count += fl;
    }
    if (has_hu) {
      fu = finite_upper(h_u[static_cast<int64_t>(b) * sh.num_hu + i]);
      fm[off_hu + i] = fu;
      count += fu;
    }
    active_G_row[static_cast<int64_t>(b) * m + i] = mask_max(fl, fu);
  }
  for (int32_t j = threadIdx.x; j < n; j += BLOCK) {
    T fl = T(0), fu = T(0);
    if (has_xl) {
      fl = finite_lower(x_l[static_cast<int64_t>(b) * sh.num_xl + j]);
      fm[off_xl + j] = fl;
      count += fl;
    }
    if (has_xu) {
      fu = finite_upper(x_u[static_cast<int64_t>(b) * sh.num_xu + j]);
      fm[off_xu + j] = fu;
      count += fu;
    }
    active_x_bound[static_cast<int64_t>(b) * n + j] = mask_max(fl, fu);
  }
  const T total = block_sum<BLOCK>(count);
  if (threadIdx.x == 0) num_finite_bounds[b] = total;
}

}  // namespace cupiqp
