// Multi-block per-problem reductions (common/multiblock.cuh / .hpp): odd
// widths, width < blocks, B > 1 with several blocks per problem, NaN
// semantics of fmin / fmax, integer max, bitwise reproducibility, graph
// capture with several replays, the blocks-per-problem heuristic.
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "common/buffer.hpp"
#include "common/graph.hpp"
#include "common/handles.hpp"
#include "common/multiblock.cuh"
#include "common/multiblock.hpp"

using namespace cupiqp;

namespace {

constexpr uint64_t kSumMinMax = reduce_kinds(ReduceKind::Sum, ReduceKind::Min, ReduceKind::Max);

// out[b] = (sum, fmin, fmax) of row b (width w, row stride ld); the
// finalizing block also counts its finalizations in done[b].
// NAN_INIT: per-thread min/max accumulators start at NaN instead of +-inf.
template <int BLOCK, bool NAN_INIT = false>
__global__ void sum_min_max_kernel(MultiblockGrid g, const double* v, int64_t w, int64_t ld, double* out,
                                   int* done) {
  __shared__ MultiblockStorage<BLOCK, 3, double> sm;
  const int64_t b = g.problem();
  double acc[3] = {0.0, NAN_INIT ? NAN : INFINITY, NAN_INIT ? NAN : -INFINITY};
  for (int64_t k = g.first<BLOCK>(); k < w; k += g.stride<BLOCK>()) {
    const double x = v[b * ld + k];
    acc[0] += x;
    acc[1] = fmin(acc[1], x);
    acc[2] = fmax(acc[2], x);
  }
  if (!multiblock_reduce<BLOCK, 3, kSumMinMax>(g, acc, sm)) return;
  if (threadIdx.x == 0) {
    out[3 * b + 0] = sm.result[0];
    out[3 * b + 1] = sm.result[1];
    out[3 * b + 2] = sm.result[2];
    done[b] += 1;
  }
}

// out[b] = max over row b of flags (int32).
template <int BLOCK>
__global__ void int_max_kernel(MultiblockGrid g, const int32_t* f, int64_t w, int32_t* out) {
  __shared__ MultiblockStorage<BLOCK, 1, int32_t> sm;
  const int64_t b = g.problem();
  int32_t acc[1] = {0};
  for (int64_t k = g.first<BLOCK>(); k < w; k += g.stride<BLOCK>()) acc[0] = max(acc[0], f[b * w + k]);
  if (!multiblock_reduce<BLOCK, 1, reduce_kinds(ReduceKind::Max)>(g, acc, sm)) return;
  if (threadIdx.x == 0) out[b] = sm.result[0];
}

struct Problem {
  int B;
  int64_t w, ld;
  std::vector<double> h;
};

Problem make(int B, int64_t w, int64_t pad, uint64_t seed) {
  Problem p{B, w, w + pad, {}};
  std::mt19937_64 gen(seed);
  std::uniform_real_distribution<double> d(-3, 3);
  p.h.resize(static_cast<size_t>(B) * p.ld);
  for (auto& x : p.h) x = d(gen);
  return p;
}

// Launch the test kernel with nb blocks per problem; returns (B, 3) results.
std::vector<double> run(const Problem& p, int nb, int block, ReductionWorkspaceBuffers& ws, cudaStream_t s,
                        const CudaBuffer<double>& v, CudaBuffer<double>& out, CudaBuffer<int>& done) {
  const ReductionWorkspace view = ws.view();
  const int nbe = blocks_per_problem(view, p.B, p.w, 3);
  EXPECT_EQ(nbe, nb);
  const MultiblockGrid g = multiblock_grid(view, nbe);
  dispatch_block(block, [&](auto bc) {
    constexpr int BLK = decltype(bc)::value;
    sum_min_max_kernel<BLK><<<multiblock_dim(p.B, nbe), BLK, 0, s>>>(g, v.data(), p.w, p.ld,
                                                                       out.data(), done.data());
  });
  CUPIQP_CUDA_CHECK_LAUNCH();
  std::vector<double> r(static_cast<size_t>(p.B) * 3);
  CUPIQP_CUDA_CHECK(cudaMemcpyAsync(r.data(), out.data(), r.size() * sizeof(double), cudaMemcpyDeviceToHost, s));
  CUPIQP_CUDA_CHECK(cudaStreamSynchronize(s));
  return r;
}

void check_against_host(const Problem& p, const std::vector<double>& r) {
  for (int b = 0; b < p.B; ++b) {
    long double sum = 0;
    double mn = INFINITY, mx = -INFINITY;
    for (int64_t k = 0; k < p.w; ++k) {
      const double x = p.h[b * p.ld + k];
      sum += x;
      mn = std::fmin(mn, x);
      mx = std::fmax(mx, x);
    }
    EXPECT_NEAR(r[3 * b + 0], static_cast<double>(sum), 1e-12 * (p.w + 1)) << "b=" << b << " w=" << p.w;
    EXPECT_EQ(r[3 * b + 1], mn) << "b=" << b << " w=" << p.w;
    EXPECT_EQ(r[3 * b + 2], mx) << "b=" << b << " w=" << p.w;
  }
}

std::vector<unsigned> tickets_of(const ReductionWorkspaceBuffers& ws, int B, cudaStream_t s) {
  std::vector<unsigned> t(B, 0);
  const ReductionWorkspace v = ws.view();
  if (v.tickets)
    CUPIQP_CUDA_CHECK(cudaMemcpyAsync(t.data(), v.tickets, B * sizeof(unsigned), cudaMemcpyDeviceToHost, s));
  CUPIQP_CUDA_CHECK(cudaStreamSynchronize(s));
  return t;
}

}  // namespace

TEST(Multiblock, HeuristicBlocksPerProblem) {
  // B alone fills the GPU: one block per problem.
  EXPECT_EQ(heuristic_blocks_per_problem(4096, 100000, 56), 1);
  EXPECT_EQ(heuristic_blocks_per_problem(kMultiblockTargetBlocksPerSm * 56, 100000, 56), 1);
  // Short rows are not split; at least multiblock_min_slice(K) per block.
  EXPECT_EQ(heuristic_blocks_per_problem(1, multiblock_min_slice(1) - 1, 56, 1), 1);
  EXPECT_EQ(heuristic_blocks_per_problem(1, 2 * multiblock_min_slice(1), 56, 1), 2);
  EXPECT_EQ(heuristic_blocks_per_problem(1, 5042, 56, 2), 5042 / multiblock_min_slice(2));
  EXPECT_LT(heuristic_blocks_per_problem(1, 13075, 56, 23), heuristic_blocks_per_problem(1, 13075, 56, 1));
  // Long rows: about kMultiblockTargetBlocksPerSm * SMs blocks in total.
  EXPECT_EQ(heuristic_blocks_per_problem(1, 1 << 24, 56), kMultiblockTargetBlocksPerSm * 56);
  EXPECT_EQ(heuristic_blocks_per_problem(4, 1 << 24, 56), (kMultiblockTargetBlocksPerSm * 56 + 3) / 4);
  // No workspace: one block. Block size: reference rule unsplit, fixed split.
  EXPECT_EQ(blocks_per_problem(ReductionWorkspace{}, 1, 1 << 24, 3), 1);
  EXPECT_EQ(multiblock_block_dim(256, 1), 32);
  EXPECT_EQ(multiblock_block_dim(2521, 1), 256);
  EXPECT_EQ(multiblock_block_dim(5042, 9), kMultiblockBlock);
}

TEST(Multiblock, WidthsBlocksAndBatches) {
  CudaStream st = CudaStream::create();
  cudaStream_t s = st.get();
  struct Case {
    int B;
    int64_t w, pad;
    int nb, block;
  };
  const Case cases[] = {
      {1, 1, 0, 1, 32},     {1, 7, 0, 5, 32},       {1, 31, 3, 64, 32},    {1, 1000, 0, 7, 64},
      {1, 5042, 0, 13, 128}, {3, 2521, 5, 9, 256},  {5, 777, 1, 40, 32},   {2, 12770, 0, 100, 256},
      {4, 3, 0, 8, 64},     {7, 100003, 11, 3, 256}, {2, 100, 0, 1, 64},
  };
  for (const Case& c : cases) {
    Problem p = make(c.B, c.w, c.pad, 1000 + c.w);
    ReductionWorkspaceBuffers ws;
    ws.allocate(c.B, c.w, s, c.nb);
    CudaBuffer<double> v(p.h.size()), out(3 * c.B);
    CudaBuffer<int> done(c.B);
    done.zero_async(s);
    v.copy_from_host(p.h.data(), p.h.size(), s);
    auto r = run(p, c.nb, c.block, ws, s, v, out, done);
    check_against_host(p, r);
    std::vector<int> d(c.B);
    CUPIQP_CUDA_CHECK(cudaMemcpy(d.data(), done.data(), c.B * sizeof(int), cudaMemcpyDeviceToHost));
    for (int b = 0; b < c.B; ++b) EXPECT_EQ(d[b], 1) << "exactly one finalizing block per problem";
    for (unsigned t : tickets_of(ws, c.B, s)) EXPECT_EQ(t, 0u) << "tickets reset by the last block";
  }
}

TEST(Multiblock, NanSemanticsOfMinMax) {
  CudaStream st = CudaStream::create();
  cudaStream_t s = st.get();
  for (bool nan_init : {false, true}) {
    for (int nb : {1, 4, 50}) {
      // Row 0: a few NaNs among finite values (ignored by fmin / fmax);
      // row 1: all NaN; row 2: all positive; row 3: all negative (the
      // cross-block identity must not leak into min / max).
      Problem p = make(4, 300, 0, 3);
      p.h[5] = NAN;
      p.h[150] = NAN;
      p.h[299] = NAN;
      for (int64_t k = 0; k < p.w; ++k) {
        p.h[p.ld + k] = NAN;
        p.h[2 * p.ld + k] = 1.0 + std::fabs(p.h[2 * p.ld + k]);
        p.h[3 * p.ld + k] = -1.0 - std::fabs(p.h[3 * p.ld + k]);
      }
      ReductionWorkspaceBuffers ws;
      ws.allocate(p.B, p.w, s, nb);
      CudaBuffer<double> v(p.h.size()), out(3 * p.B);
      CudaBuffer<int> done(p.B);
      done.zero_async(s);
      v.copy_from_host(p.h.data(), p.h.size(), s);
      const ReductionWorkspace view = ws.view();
      const MultiblockGrid g = multiblock_grid(view, blocks_per_problem(view, p.B, p.w, 3));
      if (nan_init)
        sum_min_max_kernel<64, true><<<multiblock_dim(p.B, g.nb), 64, 0, s>>>(g, v.data(), p.w, p.ld, out.data(),
                                                                              done.data());
      else
        sum_min_max_kernel<64, false><<<multiblock_dim(p.B, g.nb), 64, 0, s>>>(g, v.data(), p.w, p.ld,
                                                                               out.data(), done.data());
      CUPIQP_CUDA_CHECK_LAUNCH();
      std::vector<double> r(3 * p.B);
      CUPIQP_CUDA_CHECK(cudaMemcpyAsync(r.data(), out.data(), r.size() * sizeof(double), cudaMemcpyDeviceToHost, s));
      CUPIQP_CUDA_CHECK(cudaStreamSynchronize(s));
      const std::string tag = " (nb=" + std::to_string(nb) + ", nan_init=" + std::to_string(nan_init) + ")";
      for (int b = 0; b < p.B; ++b) {
        double mn = nan_init ? NAN : INFINITY, mx = nan_init ? NAN : -INFINITY;
        for (int64_t k = 0; k < p.w; ++k) {
          mn = std::fmin(mn, p.h[b * p.ld + k]);
          mx = std::fmax(mx, p.h[b * p.ld + k]);
        }
        if (std::isnan(mn)) EXPECT_TRUE(std::isnan(r[3 * b + 1])) << "b=" << b << tag;
        else EXPECT_EQ(r[3 * b + 1], mn) << "b=" << b << tag;
        if (std::isnan(mx)) EXPECT_TRUE(std::isnan(r[3 * b + 2])) << "b=" << b << tag;
        else EXPECT_EQ(r[3 * b + 2], mx) << "b=" << b << tag;
      }
      EXPECT_TRUE(std::isnan(r[0])) << "a NaN propagates through the sum" << tag;
      // All-NaN row: +-inf with +-inf accumulators, NaN with NaN accumulators.
      EXPECT_EQ(std::isnan(r[4]), nan_init) << tag;
      EXPECT_EQ(std::isnan(r[5]), nan_init) << tag;
    }
  }
}

TEST(Multiblock, IntegerMax) {
  CudaStream st = CudaStream::create();
  cudaStream_t s = st.get();
  const int B = 3;
  const int64_t w = 4001;
  std::vector<int32_t> h(B * w, 0);
  h[1 * w + 4000] = 1;  // last entry of row 1
  h[2 * w + 17] = 1;
  for (int nb : {1, 2, 33, 70}) {
    ReductionWorkspaceBuffers ws;
    ws.allocate(B, w, s, nb);
    CudaBuffer<int32_t> f(h.size()), out(B);
    f.copy_from_host(h.data(), h.size(), s);
    const ReductionWorkspace view = ws.view();
    const int n = blocks_per_problem(view, B, w, 1);
    EXPECT_EQ(n, nb);
    int_max_kernel<128><<<multiblock_dim(B, n), 128, 0, s>>>(multiblock_grid(view, n), f.data(), w, out.data());
    CUPIQP_CUDA_CHECK_LAUNCH();
    std::vector<int32_t> r(B);
    CUPIQP_CUDA_CHECK(cudaMemcpyAsync(r.data(), out.data(), B * sizeof(int32_t), cudaMemcpyDeviceToHost, s));
    CUPIQP_CUDA_CHECK(cudaStreamSynchronize(s));
    EXPECT_EQ(r, (std::vector<int32_t>{0, 1, 1})) << "nb=" << nb;
  }
}

TEST(Multiblock, BitwiseReproducible) {
  CudaStream st = CudaStream::create();
  cudaStream_t s = st.get();
  Problem p = make(2, 12770, 0, 99);
  for (auto& x : p.h) x *= 1e3;  // large-magnitude sums: order matters in the last bits
  ReductionWorkspaceBuffers ws;
  ws.allocate(2, p.w, s, 60);
  CudaBuffer<double> v(p.h.size()), out(6);
  CudaBuffer<int> done(2);
  done.zero_async(s);
  v.copy_from_host(p.h.data(), p.h.size(), s);
  const auto first = run(p, 60, 128, ws, s, v, out, done);
  for (int rep = 0; rep < 50; ++rep) {
    const auto r = run(p, 60, 128, ws, s, v, out, done);
    ASSERT_EQ(std::memcmp(r.data(), first.data(), r.size() * sizeof(double)), 0) << "rep " << rep;
  }
}

TEST(Multiblock, GraphCaptureAndReplays) {
  CudaStream st = CudaStream::create();
  cudaStream_t s = st.get();
  Problem p = make(3, 5042, 2, 5);
  const int nb = 20;
  ReductionWorkspaceBuffers ws;
  ws.allocate(p.B, p.w, s, nb);
  CudaBuffer<double> v(p.h.size()), out(3 * p.B);
  CudaBuffer<int> done(p.B);
  done.zero_async(s);
  CUPIQP_CUDA_CHECK(cudaStreamSynchronize(s));
  const ReductionWorkspace view = ws.view();
  const MultiblockGrid g = multiblock_grid(view, nb);
  // Two launches per replay share the workspace (serialized on one stream).
  cudaGraph_t graph = capture(
      s,
      [&](cudaStream_t cs) {
        for (int rep = 0; rep < 2; ++rep)
          sum_min_max_kernel<128><<<multiblock_dim(p.B, nb), 128, 0, cs>>>(g, v.data(), p.w, p.ld, out.data(),
                                                                          done.data());
        CUPIQP_CUDA_CHECK_LAUNCH();
      },
      nullptr);
  cudaGraphExec_t exec = nullptr;
  CUPIQP_CUDA_CHECK(cudaGraphInstantiate(&exec, graph, 0));
  for (int replay = 0; replay < 5; ++replay) {
    Problem q = make(p.B, p.w, 2, 100 + replay);
    v.copy_from_host(q.h.data(), q.h.size(), s);
    CUPIQP_CUDA_CHECK(cudaGraphLaunch(exec, s));
    std::vector<double> r(3 * p.B);
    CUPIQP_CUDA_CHECK(cudaMemcpyAsync(r.data(), out.data(), r.size() * sizeof(double), cudaMemcpyDeviceToHost, s));
    CUPIQP_CUDA_CHECK(cudaStreamSynchronize(s));
    check_against_host(q, r);
    std::vector<int> d(p.B);
    CUPIQP_CUDA_CHECK(cudaMemcpy(d.data(), done.data(), p.B * sizeof(int), cudaMemcpyDeviceToHost));
    for (int b = 0; b < p.B; ++b) EXPECT_EQ(d[b], 2 * (replay + 1));
    for (unsigned t : tickets_of(ws, p.B, s)) EXPECT_EQ(t, 0u);
  }
  CUPIQP_CUDA_CHECK(cudaGraphExecDestroy(exec));
  CUPIQP_CUDA_CHECK(cudaGraphDestroy(graph));
}
