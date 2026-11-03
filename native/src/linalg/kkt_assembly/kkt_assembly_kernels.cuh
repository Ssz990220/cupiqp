// KKT value kernels, version 1: one thread per (b, k) like the reference Warp
// kernels in cupiqp/sparse/sparse_kkt_solver_kernels.py (launch dim (B, count),
// flattened as t = b * count + k).
#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace cupiqp::kkt_kernels {

inline constexpr int kBlock = 256;

inline unsigned grid_for(int64_t threads) {
  return static_cast<unsigned>((threads + kBlock - 1) / kBlock);
}

// Reference scatter_values_kernel: dst[b, dst_cols[k]] = src[b, k].
template <class T>
__global__ void scatter_values_kernel(const T* __restrict__ src, const int32_t* __restrict__ dst_cols,
                                      T* __restrict__ dst, int32_t B, int32_t nnz, int32_t dst_stride) {
  const int64_t t = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (t >= static_cast<int64_t>(B) * nnz) return;
  const int64_t b = t / nnz;
  const int64_t k = t - b * nnz;
  dst[b * dst_stride + dst_cols[k]] = src[b * nnz + k];
}

// Reference gather_scatter_values_kernel: dst[b, dst_cols[k]] = src[b, src_cols[k]].
template <class T>
__global__ void gather_scatter_values_kernel(const T* __restrict__ src, const int32_t* __restrict__ src_cols,
                                             const int32_t* __restrict__ dst_cols, T* __restrict__ dst,
                                             int32_t B, int32_t num, int32_t src_stride,
                                             int32_t dst_stride) {
  const int64_t t = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (t >= static_cast<int64_t>(B) * num) return;
  const int64_t b = t / num;
  const int64_t k = t - b * num;
  dst[b * dst_stride + dst_cols[k]] = src[b * src_stride + src_cols[k]];
}

// Reference scatter_masked_G_kernel:
// kkt[b, G_slots[k]] = G[b, k] * active_G_row[b, G_row_idx[k]].
template <class T>
__global__ void scatter_masked_G_kernel(const T* __restrict__ G_data, const T* __restrict__ active_G_row,
                                        const int32_t* __restrict__ G_row_idx,
                                        const int32_t* __restrict__ G_slots, T* __restrict__ kkt,
                                        int32_t B, int32_t nnz_G, int32_t m, int32_t kkt_nnz) {
  const int64_t t = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (t >= static_cast<int64_t>(B) * nnz_G) return;
  const int64_t b = t / nnz_G;
  const int64_t k = t - b * nnz_G;
  const int32_t row = G_row_idx[k];
  kkt[b * kkt_nnz + G_slots[k]] = G_data[b * nnz_G + k] * active_G_row[b * m + row];
}

// Reference update_kkt_diag_kernel, launch (B, n + p + m):
//   x: P_diag + x_reg;  y: -delta[b];  z: -z_reg if z_reg > 0 else -1.
template <class T>
__global__ void update_kkt_diag_kernel(const T* __restrict__ P_diag, const T* __restrict__ x_reg,
                                       const T* __restrict__ delta, const T* __restrict__ z_reg,
                                       const int32_t* __restrict__ diag_x, const int32_t* __restrict__ diag_y,
                                       const int32_t* __restrict__ diag_z, T* __restrict__ kkt, int32_t B,
                                       int32_t n, int32_t p, int32_t m, int32_t kkt_nnz) {
  const int64_t dim = static_cast<int64_t>(n) + p + m;
  const int64_t g = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (g >= static_cast<int64_t>(B) * dim) return;
  const int64_t b = g / dim;
  const int32_t t = static_cast<int32_t>(g - b * dim);
  T* row = kkt + b * kkt_nnz;
  if (t < n) {
    row[diag_x[t]] = P_diag[b * n + t] + x_reg[b * n + t];
  } else if (t < n + p) {
    const int32_t i = t - n;
    row[diag_y[i]] = -delta[b];
  } else {
    const int32_t k = t - n - p;
    const T zr = z_reg[b * m + k];
    row[diag_z[k]] = zr > T(0) ? -zr : -T(1);
  }
}

}  // namespace cupiqp::kkt_kernels
