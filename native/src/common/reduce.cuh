// Block reductions for the "one block per problem" kernels that replace the
// reference Warp tile reductions (wp.launch_tiled with reduction_block_dim).
// Implemented with cub::BlockReduce (BLOCK_REDUCE_WARP_REDUCTIONS: fixed tree,
// bitwise reproducible run to run). Results may differ in the last bits from
// Warp's tile_reduce order (DESIGN.md section 6).
//
// Usage inside a kernel launched with blockDim.x == BLOCK:
//   T local = 0; for (int i = threadIdx.x; i < k; i += BLOCK) local += v[i];
//   T total = block_sum<BLOCK>(local);   // every thread receives the result
// Every thread of the block must reach each call (no divergent early return).
#pragma once

#include <cuda_runtime.h>

#include <cub/block/block_reduce.cuh>
#include <cstdint>
#include <type_traits>

namespace cupiqp {

// Reference cupiqp.utils.reduction_block_dim: 32 threads for rows of up to 256
// entries, doubling up to 256 threads (about 8 entries per thread).
inline int reduction_block_dim(int64_t width) {
  int block = 32;
  while (block < 256 && 8 * static_cast<int64_t>(block) < width) block *= 2;
  return block;
}

struct SumOp {
  template <class T>
  __device__ __forceinline__ T operator()(const T& a, const T& b) const { return a + b; }
};
// Max/min follow Warp's wp.max / wp.min / tile_max semantics: for floating
// point they are fmax / fmin (a NaN operand is ignored unless both are NaN);
// integers use ordinary compares.
struct MaxOp {
  template <class T>
  __device__ __forceinline__ T operator()(const T& a, const T& b) const {
    if constexpr (std::is_floating_point_v<T>) return fmax(a, b);
    else return a > b ? a : b;
  }
};
struct MinOp {
  template <class T>
  __device__ __forceinline__ T operator()(const T& a, const T& b) const {
    if constexpr (std::is_floating_point_v<T>) return fmin(a, b);
    else return a < b ? a : b;
  }
};

// Block-wide reduction broadcast to all threads (cub::BlockReduce + shared
// broadcast). Safe to call several times in a row in one kernel.
template <int BLOCK, class T, class Op>
__device__ __forceinline__ T block_reduce(T v, Op op) {
  using Reduce = cub::BlockReduce<T, BLOCK, cub::BLOCK_REDUCE_WARP_REDUCTIONS>;
  __shared__ typename Reduce::TempStorage temp;
  __shared__ T result;
  T r = Reduce(temp).Reduce(v, op);  // valid in thread 0
  if (threadIdx.x == 0) result = r;
  __syncthreads();
  r = result;
  __syncthreads();  // temp/result may be reused by the next call
  return r;
}

template <int BLOCK, class T>
__device__ __forceinline__ T block_sum(T v) { return block_reduce<BLOCK>(v, SumOp{}); }
template <int BLOCK, class T>
__device__ __forceinline__ T block_max(T v) { return block_reduce<BLOCK>(v, MaxOp{}); }
template <int BLOCK, class T>
__device__ __forceinline__ T block_min(T v) { return block_reduce<BLOCK>(v, MinOp{}); }

// Dispatch a templated reduction kernel launch on a runtime block size chosen
// by reduction_block_dim. F is called with std::integral_constant<int, BLOCK>.
template <class F>
inline void dispatch_block(int block, F&& f) {
  switch (block) {
    case 32: f(std::integral_constant<int, 32>{}); break;
    case 64: f(std::integral_constant<int, 64>{}); break;
    case 128: f(std::integral_constant<int, 128>{}); break;
    default: f(std::integral_constant<int, 256>{}); break;
  }
}

// Grid size for an element-wise kernel over `total` threads of `block` each.
inline unsigned grid_for(int64_t total, int block = 256) {
  if (total <= 0) return 0;
  return static_cast<unsigned>((total + block - 1) / block);
}

}  // namespace cupiqp
