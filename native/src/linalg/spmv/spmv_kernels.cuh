// Kernels of module spmv.
//  * scale_kernel: the y = beta * y fallback used when op(M) has no stored
//    entries (rows, cols or nnz zero) and no product is computed.
//  * csr_lanes_kernel: custom backend (SpmvBackend::Custom). LANES threads per
//    output row stride over the row's entries, then a fixed shuffle tree
//    reduces them: bitwise reproducible. With perm != nullptr the values are
//    gathered through perm (op = T as the CSC view of a CSR matrix).
#pragma once

#include <cstdint>

namespace cupiqp::spmv_detail {

template <class T>
__global__ void scale_kernel(int32_t k, T beta, T* __restrict__ y) {
  const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < k) y[i] = beta * y[i];
}

template <int LANES, class T>
__global__ void csr_lanes_kernel(int32_t rows, const int32_t* __restrict__ indptr,
                                 const int32_t* __restrict__ indices, const T* __restrict__ values,
                                 const int32_t* __restrict__ perm, const T* __restrict__ x, T alpha, T beta,
                                 T* __restrict__ y) {
  const int64_t gid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int32_t row = static_cast<int32_t>(gid / LANES);
  const int lane = static_cast<int>(gid % LANES);
  T s = T(0);
  if (row < rows) {
    const int32_t end = indptr[row + 1];
    if (perm != nullptr) {
      for (int32_t k = indptr[row] + lane; k < end; k += LANES) s += values[perm[k]] * x[indices[k]];
    } else {
      for (int32_t k = indptr[row] + lane; k < end; k += LANES) s += values[k] * x[indices[k]];
    }
  }
  // Every lane of the warp reaches the shuffles (no early return above).
#pragma unroll
  for (int o = LANES / 2; o > 0; o /= 2) s += __shfl_down_sync(0xffffffffu, s, o, LANES);
  if (row < rows && lane == 0) y[row] = beta == T(0) ? alpha * s : alpha * s + beta * y[row];
}

}  // namespace cupiqp::spmv_detail
