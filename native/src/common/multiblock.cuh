// Multi-block per-problem reductions: device primitive (host side and the
// workspace are in common/multiblock.hpp).
//
// Launch geometry: grid = (B, nb) with nb = blocks_per_problem(...), block =
// BLOCK threads. Block (b, j) handles the elements k = j * BLOCK + i,
// k += nb * BLOCK of problem b (with nb == 1 this is the "one block per
// problem" mapping of the reference tiled kernels).
//
// Each thread accumulates K values; multiblock_reduce() combines them over
// the block (warp shuffles in a fixed tree, then the warps in index order),
// and, with nb > 1, over the nb blocks of the problem: every block writes its
// K partials, fences, and takes a ticket; the last block reads the partials
// back in block order (lane l of a warp folds blocks l, l + 32, ...; then a
// fixed shuffle tree), resets the ticket to 0 and returns true. With nb == 1
// no partials or tickets are touched. The reduction tree is fixed for a fixed
// (BLOCK, nb), so results are bitwise reproducible run to run.
//
// Combine kinds follow common/reduce.cuh: Sum; Min / Max are fmin / fmax for
// floating point (a NaN operand is ignored unless both are NaN, like Warp
// wp.min / wp.max / tile_min / tile_max) and ordinary compares for integers.
// Floating Min / Max run on order-preserving integer keys (see Lane below):
// the result is the exact fmin / fmax of the operands (for a +0 / -0 tie,
// max returns +0 and min -0). The identity used for lanes without a partial
// is 0 (Sum), the NaN key (floating Min / Max) or the integer limits.
#pragma once

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "common/multiblock.hpp"
#include "common/reduce.cuh"

namespace cupiqp {

enum class ReduceKind : uint32_t { Sum = 0, Min = 1, Max = 2 };

// Compile-time list of K combine kinds (2 bits each, value k at bits 2k).
template <class... Ks>
__host__ __device__ constexpr uint64_t reduce_kinds(Ks... ks) {
  uint64_t code = 0;
  int i = 0;
  ((code |= uint64_t(static_cast<uint32_t>(ks)) << (2 * i++)), ...);
  return code;
}
__host__ __device__ constexpr ReduceKind reduce_kind_at(uint64_t code, int k) {
  return static_cast<ReduceKind>((code >> (2 * k)) & 3u);
}

namespace multiblock_detail {

// Raw 64/32-bit pattern of a floating value.
__device__ __forceinline__ long long bits_of(double x) { return __double_as_longlong(x); }
__device__ __forceinline__ double from_bits(long long b) { return __longlong_as_double(b); }
__device__ __forceinline__ int bits_of(float x) { return __float_as_int(x); }
__device__ __forceinline__ float from_bits(int b) { return __int_as_float(b); }

// Per-kind lane arithmetic. Lane values are "encoded": for floating Min / Max
// they carry an order-preserving signed integer key in the value's bits
// (non-NaN x: bits ^ (sign ? 0x7ff..f : 0), so integer order == numeric
// order with -0 < +0; NaN: the key that loses every comparison, i.e. NaN is
// ignored unless every operand is NaN, exactly fmin / fmax). Compares of
// keys are integer instructions; FP64 runs at a small fraction of the FP32
// rate on consumer GPUs, so this keeps the min / max trees off the FP64 pipe.
template <class T>
struct LaneSum {
  __device__ __forceinline__ static T enc(T x) { return x; }
  __device__ __forceinline__ static T dec(T x) { return x; }
  __device__ __forceinline__ static T comb(T a, T b) { return a + b; }
  __device__ __forceinline__ static T identity() { return T(0); }
};

template <ReduceKind KIND, class T, bool FLOAT = std::is_floating_point_v<T>>
struct LaneMinMax;

template <ReduceKind KIND, class T>
struct LaneMinMax<KIND, T, false> {  // integers: ordinary compares
  static constexpr bool kMax = KIND == ReduceKind::Max;
  __device__ __forceinline__ static T enc(T x) { return x; }
  __device__ __forceinline__ static T dec(T x) { return x; }
  __device__ __forceinline__ static T comb(T a, T b) { return kMax ? (a > b ? a : b) : (a < b ? a : b); }
  __device__ __forceinline__ static T identity() {
    return kMax ? std::numeric_limits<T>::lowest() : std::numeric_limits<T>::max();
  }
};

template <ReduceKind KIND, class T>
struct LaneMinMax<KIND, T, true> {  // floating min / max on integer keys
  static constexpr bool kMax = KIND == ReduceKind::Max;
  using K = decltype(bits_of(T(0)));
  static constexpr K kMagnitude = std::numeric_limits<K>::max();  // 0x7ff..f
  // NaN key: loses every comparison. Never the key of a non-NaN value.
  static constexpr K kNanKey = kMax ? std::numeric_limits<K>::min() : std::numeric_limits<K>::max();
  static constexpr K kInfBits = sizeof(K) == 8 ? K(0x7ff0000000000000LL) : K(0x7f800000);
  __device__ __forceinline__ static T enc(T x) {
    const K b = bits_of(x);
    if ((b & kMagnitude) > kInfBits) return from_bits(kNanKey);  // NaN (integer test)
    return from_bits(b ^ ((b >> (8 * sizeof(K) - 1)) & kMagnitude));
  }
  __device__ __forceinline__ static T dec(T e) {
    const K k = bits_of(e);
    if (k == kNanKey) return T(NAN);
    return from_bits(k ^ ((k >> (8 * sizeof(K) - 1)) & kMagnitude));
  }
  __device__ __forceinline__ static T comb(T a, T b) {
    const K ka = bits_of(a), kb = bits_of(b);
    return kMax ? (kb > ka ? b : a) : (kb < ka ? b : a);
  }
  __device__ __forceinline__ static T identity() { return from_bits(kNanKey); }
};

template <ReduceKind KIND, class T>
using Lane = std::conditional_t<KIND == ReduceKind::Sum, LaneSum<T>, LaneMinMax<KIND, T>>;

// Call f(Lane<kind, T>{}) for a kind known after unrolling.
template <class T, class F>
__device__ __forceinline__ void with_lane(ReduceKind kind, F&& f) {
  switch (kind) {
    case ReduceKind::Sum: f(Lane<ReduceKind::Sum, T>{}); break;
    case ReduceKind::Min: f(Lane<ReduceKind::Min, T>{}); break;
    default: f(Lane<ReduceKind::Max, T>{}); break;
  }
}

// Fixed shuffle tree; the result is valid in lane 0.
template <class L, class T>
__device__ __forceinline__ T warp_reduce(T v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) v = L::comb(v, __shfl_down_sync(0xffffffffu, v, off));
  return v;
}

}  // namespace multiblock_detail

// Device-side launch context of a multi-block reduction kernel.
struct MultiblockGrid {
  void* partials = nullptr;
  unsigned int* tickets = nullptr;
  int nb = 1;  // blocks per problem (== gridDim.y)

#if defined(__CUDACC__)
  __device__ __forceinline__ int64_t problem() const { return blockIdx.x; }
  // First element of this thread and the stride over the problem row.
  template <int BLOCK>
  __device__ __forceinline__ int64_t first() const { return int64_t(blockIdx.y) * BLOCK + threadIdx.x; }
  template <int BLOCK>
  __device__ __forceinline__ int64_t stride() const { return int64_t(nb) * BLOCK; }
#endif
};

inline MultiblockGrid multiblock_grid(const ReductionWorkspace& ws, int nb) {
  MultiblockGrid g;
  g.nb = nb;
  if (nb > 1) {
    g.partials = ws.partials;
    g.tickets = ws.tickets;
  }
  return g;
}

inline dim3 multiblock_dim(int32_t B, int nb) {
  return dim3(static_cast<unsigned>(B), static_cast<unsigned>(nb), 1u);
}

// Shared memory of one multiblock_reduce call site.
template <int BLOCK, int K, class T>
struct MultiblockStorage {
  // Last block: lanes per value (power of two <= 32, kGroup * K <= BLOCK)
  // and partial blocks staged at a time.
  static constexpr int kGroup = (BLOCK / K >= 32) ? 32 : (BLOCK / K >= 16) ? 16 : (BLOCK / K >= 8) ? 8
                                : (BLOCK / K >= 4) ? 4 : (BLOCK / K >= 2) ? 2 : 1;
  static constexpr int kChunk = (BLOCK / K > 32) ? BLOCK / K : 32;
  T warp_vals[BLOCK / 32][K];
  T stage[kChunk][K];  // last block: staged partials
  T result[K];
  int is_last;
};

// Reduce the K per-thread values v of every thread of every block of the
// problem. Must be reached by every thread of the block. Returns true (in
// every thread) in the one block that holds the problem's results in
// sm.result; that block may then run the scalar finalization. Other blocks
// return false and must not touch sm.result.
template <int BLOCK, int K, uint64_t KINDS, class T>
__device__ __forceinline__ bool multiblock_reduce(const MultiblockGrid& g, T (&v)[K],
                                                  MultiblockStorage<BLOCK, K, T>& sm) {
  static_assert(BLOCK % 32 == 0 && BLOCK >= 32 && BLOCK <= 1024, "BLOCK must be a multiple of 32");
  static_assert(K >= 1 && K <= kMultiblockMaxValues && K <= BLOCK, "too many values");
  static_assert(sizeof(T) <= 8, "partials are 8-byte slots");
  using namespace multiblock_detail;
  constexpr int NW = BLOCK / 32;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;

  // Encode, then reduce all K values over the warp with the fixed shuffle
  // tree, level by level so the K chains overlap (results in lane 0).
#pragma unroll
  for (int k = 0; k < K; ++k)
    with_lane<T>(reduce_kind_at(KINDS, k), [&](auto L) { v[k] = decltype(L)::enc(v[k]); });
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    T other[K];
#pragma unroll
    for (int k = 0; k < K; ++k) other[k] = __shfl_down_sync(0xffffffffu, v[k], off);
#pragma unroll
    for (int k = 0; k < K; ++k)
      with_lane<T>(reduce_kind_at(KINDS, k), [&](auto L) { v[k] = decltype(L)::comb(v[k], other[k]); });
  }
  if (lane == 0) {
#pragma unroll
    for (int k = 0; k < K; ++k) sm.warp_vals[warp][k] = v[k];
  }
  __syncthreads();

  // Thread k (< K) folds the warp results of value k in warp order (encoded).
  T block_val{};
  if (tid < K) {
    with_lane<T>(reduce_kind_at(KINDS, tid), [&](auto L) {
      using LL = decltype(L);
      T acc = sm.warp_vals[0][tid];
#pragma unroll
      for (int w = 1; w < NW; ++w) acc = LL::comb(acc, sm.warp_vals[w][tid]);
      block_val = acc;
    });
  }

  if (g.nb <= 1) {
    if (tid < K)
      with_lane<T>(reduce_kind_at(KINDS, tid), [&](auto L) { sm.result[tid] = decltype(L)::dec(block_val); });
    __syncthreads();
    return true;
  }

  const int64_t b = blockIdx.x;
  T* part = static_cast<T*>(g.partials);
  // Encoded partials of problem b: [block j][value k] at (b * nb + j) * K + k.
  if (tid < K) __stcg(part + (b * g.nb + blockIdx.y) * K + tid, block_val);
  __threadfence();
  __syncthreads();
  if (tid == 0) {
    const unsigned int ticket = atomicAdd(g.tickets + b, 1u);
    sm.is_last = ticket == static_cast<unsigned int>(g.nb - 1);
  }
  __syncthreads();
  if (!sm.is_last) return false;
  __threadfence();

  // Last block. The partials are staged through shared memory CH blocks at a
  // time (independent coalesced loads). Lane group k (R consecutive lanes,
  // thread (k, r) = (tid / R, tid % R)) handles value k: lane r folds blocks
  // r, r + R, r + 2R, ... in order, then a fixed shuffle tree over the R
  // lanes. Fixed order for a fixed (BLOCK, nb).
  using Storage = MultiblockStorage<BLOCK, K, T>;
  constexpr int R = Storage::kGroup;
  constexpr int CH = Storage::kChunk;
  const int k = tid / R;
  const int r = tid % R;
  const bool active = k < K;
  const ReduceKind kind = reduce_kind_at(KINDS, active ? k : 0);
  T acc{};
  with_lane<T>(kind, [&](auto L) { acc = decltype(L)::identity(); });
  const T* base = part + b * g.nb * K;
  T* stage = &sm.stage[0][0];
  for (int c0 = 0; c0 < g.nb; c0 += CH) {
    const int nblk = min(CH, g.nb - c0);
    for (int t = tid; t < nblk * K; t += BLOCK) stage[t] = __ldcg(base + int64_t(c0) * K + t);
    __syncthreads();
    if (active) {
      with_lane<T>(kind, [&](auto L) {
        for (int j = r; j < nblk; j += R) acc = decltype(L)::comb(acc, sm.stage[j][k]);
      });
    }
    __syncthreads();
  }
#pragma unroll
  for (int off = R / 2; off > 0; off >>= 1) {
    const T other = __shfl_down_sync(0xffffffffu, acc, off, R);
    with_lane<T>(kind, [&](auto L) { acc = decltype(L)::comb(acc, other); });
  }
  if (active && r == 0) with_lane<T>(kind, [&](auto L) { sm.result[k] = decltype(L)::dec(acc); });
  if (tid == 0) g.tickets[b] = 0u;  // ready for the next launch / graph replay
  __syncthreads();
  return true;
}

}  // namespace cupiqp
