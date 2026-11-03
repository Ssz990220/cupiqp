// Multi-block per-problem reductions: host side (workspace and launch
// geometry). The device primitive is in common/multiblock.cuh.
//
// The "one block per problem" kernels (reference wp.launch_tiled(dim=[B]))
// leave the GPU almost idle when B is small (B == 1: one SM). A multi-block
// reduction splits each problem row over `blocks_per_problem` blocks
// (grid = (B, blocks_per_problem)); each block reduces its slice to K values,
// writes them to a partial buffer and takes a ticket; the last block of the
// problem combines the partials in a fixed order, resets the ticket and runs
// the scalar finalization. Results are bitwise reproducible run to run for a
// fixed grid.
//
// The workspace (partials, tickets, per-problem scratch scalars) is allocated
// once at setup (DESIGN section 10.1). Tickets are zero between launches: the
// last block of each problem resets its ticket, so graph replays work.
// One workspace must not be used by two kernels running concurrently (on two
// streams); kernels serialized on one stream may share it.
#pragma once

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdint>

#include "common/buffer.hpp"
#include "common/check.hpp"

namespace cupiqp {

// Most values one multi-block reduction may carry per problem (K).
inline constexpr int kMultiblockMaxValues = 24;
// Per-problem scratch scalars of the workspace (intermediate reduced values
// handed from one kernel to the next, e.g. init_guess_center min(s), min(z)).
inline constexpr int kMultiblockScalarsPerProblem = 4;
// Heuristic (measured on an RTX 4070 SUPER, 56 SMs, B = 1, MPC / Trajopt
// widths 5k-13k; native/docs/modules/multiblock_reductions.md): at most
// about this many blocks per SM over the whole grid ...
inline constexpr int kMultiblockTargetBlocksPerSm = 3;
// ... and at least kMultiblockSliceBase + kMultiblockSlicePerValue * K
// elements per block (K values per problem: the per-block reduction and the
// final combine grow with K, so wide-K kernels get fewer, fuller blocks) ...
inline constexpr int64_t kMultiblockSliceBase = 256;
inline constexpr int64_t kMultiblockSlicePerValue = 8;
// ... with this many threads per block when a problem is split (1-3
// elements per thread at the minimum slice).
inline constexpr int kMultiblockBlock = 128;
// gridDim.y limit.
inline constexpr int kMultiblockMaxBlocksPerProblem = 65535;

// Non-owning view of a reduction workspace, passed by value to launchers.
// A default-constructed view (no buffers) means one block per problem.
struct ReductionWorkspace {
  void* partials = nullptr;            // 8-byte slots (double or int32 values)
  int64_t partial_capacity = 0;        // slots
  unsigned int* tickets = nullptr;     // (ticket_capacity,), zero between launches
  int64_t ticket_capacity = 0;         // problems
  double* scalars = nullptr;           // (scalar_capacity,) scratch
  int64_t scalar_capacity = 0;
  int sm_count = 0;                    // multiprocessors of the device (0: unknown)
  int forced_blocks_per_problem = 0;   // > 0: use this instead of the heuristic (tests)
};

// Minimum elements per block of a reduction carrying K values per problem.
inline int64_t multiblock_min_slice(int K) {
  return kMultiblockSliceBase + kMultiblockSlicePerValue * std::max(K, 1);
}

// Heuristic blocks per problem for B problems of `width` elements carrying K
// values each, on a device with `sm_count` SMs. 1 when B alone fills the GPU
// (B >= kMultiblockTargetBlocksPerSm * sm_count) or the row is short.
inline int heuristic_blocks_per_problem(int32_t B, int64_t width, int sm_count, int K = 1) {
  if (B <= 0 || sm_count <= 0 || width <= 0) return 1;
  const int64_t min_slice = multiblock_min_slice(K);
  const int64_t target = int64_t(kMultiblockTargetBlocksPerSm) * sm_count;
  if (B >= target) return 1;
  const int64_t by_target = (target + B - 1) / B;
  const int64_t by_width = std::max<int64_t>(1, width / min_slice);
  const int64_t nb = std::min<int64_t>({by_target, by_width, kMultiblockMaxBlocksPerProblem});
  return static_cast<int>(std::max<int64_t>(1, nb));
}

// Blocks per problem for one launch carrying K values per problem: the
// forced value or the heuristic, clamped to what the workspace can hold.
// Returns 1 without a workspace.
inline int blocks_per_problem(const ReductionWorkspace& ws, int32_t B, int64_t width, int K) {
  if (B <= 0 || ws.partials == nullptr || ws.tickets == nullptr || ws.ticket_capacity < B) return 1;
  int64_t nb = ws.forced_blocks_per_problem > 0
                   ? ws.forced_blocks_per_problem
                   : heuristic_blocks_per_problem(B, width, ws.sm_count, K);
  const int64_t fit = ws.partial_capacity / (int64_t(B) * std::max(K, 1));
  nb = std::min<int64_t>({nb, fit, kMultiblockMaxBlocksPerProblem});
  return static_cast<int>(std::max<int64_t>(1, nb));
}

// Block size of a reduction launch over rows of `width` elements with nb
// blocks per problem: the reference reduction_block_dim(width) for one
// block per problem, kMultiblockBlock when a problem is split.
inline int multiblock_block_dim(int64_t width, int nb) {
  if (nb > 1) return kMultiblockBlock;
  int block = 32;
  while (block < 256 && 8 * static_cast<int64_t>(block) < width) block *= 2;
  return block;
}

// Number of SMs of the current device.
inline int current_device_sm_count() {
  int dev = 0, sms = 0;
  CUPIQP_CUDA_CHECK(cudaGetDevice(&dev));
  CUPIQP_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
  return sms;
}

// Owner of a reduction workspace (allocated once, at setup).
class ReductionWorkspaceBuffers {
 public:
  // Size for B problems whose reduced rows have at most max_width elements.
  // forced_blocks_per_problem > 0 sizes (and forces) that many blocks per
  // problem instead of the heuristic (tests). Queries the SM count once.
  void allocate(int32_t B, int64_t max_width, cudaStream_t s, int forced_blocks_per_problem = 0) {
    CUPIQP_REQUIRE(B >= 0, "ReductionWorkspaceBuffers: negative batch size");
    sm_count_ = current_device_sm_count();
    forced_ = forced_blocks_per_problem;
    // Capacity for the largest split any K may get (K = 1 splits the most).
    const int64_t nb = forced_ > 0 ? forced_ : heuristic_blocks_per_problem(B, max_width, sm_count_, 1);
    const size_t Bs = static_cast<size_t>(std::max(B, 0));
    // One block per problem needs no partials or tickets.
    if (nb > 1 && B > 0) {
      partials_.allocate(Bs * static_cast<size_t>(nb) * kMultiblockMaxValues);
      tickets_.allocate(Bs);
      tickets_.zero_async(s);
    }
    scalars_.allocate(Bs * kMultiblockScalarsPerProblem);
    scalars_.zero_async(s);
  }

  ReductionWorkspace view() const {
    ReductionWorkspace w;
    w.partials = const_cast<double*>(partials_.data());
    w.partial_capacity = static_cast<int64_t>(partials_.size());
    w.tickets = const_cast<unsigned int*>(tickets_.data());
    w.ticket_capacity = static_cast<int64_t>(tickets_.size());
    w.scalars = const_cast<double*>(scalars_.data());
    w.scalar_capacity = static_cast<int64_t>(scalars_.size());
    w.sm_count = sm_count_;
    w.forced_blocks_per_problem = forced_;
    return w;
  }

  int sm_count() const { return sm_count_; }

 private:
  CudaBuffer<double> partials_;
  CudaBuffer<unsigned int> tickets_;
  CudaBuffer<double> scalars_;
  int sm_count_ = 0;
  int forced_ = 0;
};

}  // namespace cupiqp
