// M0 core tests: buffers, CUB block reductions, capture_while semantics.
#include <gtest/gtest.h>

#include <cmath>
#include <numeric>
#include <random>
#include <vector>

#include "common/buffer.hpp"
#include "common/graph.hpp"
#include "common/handles.hpp"
#include "common/reduce.cuh"

using namespace cupiqp;

namespace {

template <int BLOCK>
__global__ void reduce_kernel(const double* v, int k, double* out) {
  double s = 0, mx = -INFINITY, mn = INFINITY;
  for (int i = threadIdx.x; i < k; i += BLOCK) {
    s += v[i];
    mx = fmax(mx, v[i]);
    mn = fmin(mn, v[i]);
  }
  double ts = block_sum<BLOCK>(s);
  double tmx = block_max<BLOCK>(mx);
  double tmn = block_min<BLOCK>(mn);
  if (threadIdx.x == BLOCK - 1) {  // every thread holds the result
    out[0] = ts;
    out[1] = tmx;
    out[2] = tmn;
  }
}

__global__ void dec(int* counter, int* cond, int* trace) {
  int v = --(*counter);
  trace[v] += 1;
  *cond = v > 0;
}
__global__ void set_int(int* p, int v) { *p = v; }
__global__ void inner_body(int* inner_counter, int* inner_cond, int* total) {
  *total += 1;
  int v = --(*inner_counter);
  *inner_cond = v > 0;
}
__global__ void outer_tail(int* outer_counter, int* outer_cond, int* inner_counter, int* inner_cond) {
  int v = --(*outer_counter);
  *outer_cond = v > 0;
  *inner_counter = 2;
  *inner_cond = 1;
}

}  // namespace

TEST(Core, BufferRoundTripAndEmpty) {
  CudaStream s = CudaStream::create();
  CudaBuffer<double> b(5);
  std::vector<double> h{1, 2, 3, 4, 5}, back(5);
  b.copy_from_host(h.data(), 5, s.get());
  CUPIQP_CUDA_CHECK(cudaMemcpyAsync(back.data(), b.data(), 5 * sizeof(double), cudaMemcpyDeviceToHost, s.get()));
  CUPIQP_CUDA_CHECK(cudaStreamSynchronize(s.get()));
  EXPECT_EQ(h, back);
  CudaBuffer<double> e(0);
  EXPECT_EQ(e.data(), nullptr);
  EXPECT_TRUE(e.empty());
  CudaBuffer<double> moved = std::move(b);
  EXPECT_EQ(b.data(), nullptr);
  EXPECT_EQ(moved.size(), 5u);
  EXPECT_THROW(moved.allocate(3), Error);
}

TEST(Core, ReductionBlockDimMatchesReference) {
  EXPECT_EQ(reduction_block_dim(1), 32);
  EXPECT_EQ(reduction_block_dim(256), 32);
  EXPECT_EQ(reduction_block_dim(257), 64);
  EXPECT_EQ(reduction_block_dim(1024), 128);
  EXPECT_EQ(reduction_block_dim(2521), 256);
  EXPECT_EQ(reduction_block_dim(1 << 20), 256);
}

TEST(Core, BlockReductionsAllWidths) {
  std::mt19937_64 gen(7);
  std::uniform_real_distribution<double> d(-3, 3);
  CudaStream s = CudaStream::create();
  for (int k : {1, 7, 31, 32, 33, 255, 256, 257, 1000, 2521, 10000}) {
    std::vector<double> h(k);
    for (auto& x : h) x = d(gen);
    CudaBuffer<double> v(k), out(3);
    v.copy_from_host(h.data(), k, s.get());
    int block = reduction_block_dim(k);
    dispatch_block(block, [&](auto bc) {
      constexpr int B = decltype(bc)::value;
      reduce_kernel<B><<<1, B, 0, s.get()>>>(v.data(), k, out.data());
    });
    CUPIQP_CUDA_CHECK_LAUNCH();
    double r[3];
    CUPIQP_CUDA_CHECK(cudaMemcpyAsync(r, out.data(), sizeof r, cudaMemcpyDeviceToHost, s.get()));
    CUPIQP_CUDA_CHECK(cudaStreamSynchronize(s.get()));
    long double ref = 0;
    for (double x : h) ref += x;
    EXPECT_NEAR(r[0], (double)ref, 1e-12 * k) << "k=" << k;
    EXPECT_EQ(r[1], *std::max_element(h.begin(), h.end()));
    EXPECT_EQ(r[2], *std::min_element(h.begin(), h.end()));
  }
}

class CaptureWhile : public ::testing::Test {
 protected:
  void SetUp() override {
    stream = CudaStream::create();
    ints.allocate(16);
    ints.zero_async(stream.get());
  }
  int read(int i) {
    int v;
    CUPIQP_CUDA_CHECK(cudaMemcpyAsync(&v, ints.data() + i, sizeof v, cudaMemcpyDeviceToHost, stream.get()));
    CUPIQP_CUDA_CHECK(cudaStreamSynchronize(stream.get()));
    return v;
  }
  CudaStream stream;
  CudaBuffer<int> ints;  // [counter, cond, trace0..trace7, ...]
  Counters counters;
};

TEST_F(CaptureWhile, EagerHostLoopRunsBodyWhileConditionHolds) {
  LoopContext ctx(&counters);
  int *counter = ints.data(), *cond = ints.data() + 1, *trace = ints.data() + 2;
  set_int<<<1, 1, 0, stream.get()>>>(counter, 3);
  set_int<<<1, 1, 0, stream.get()>>>(cond, 1);
  capture_while(ctx, stream.get(), cond, [&](cudaStream_t s) { dec<<<1, 1, 0, s>>>(counter, cond, trace); });
  EXPECT_EQ(read(0), 0);
  EXPECT_EQ(read(2) + read(3) + read(4), 3);
  EXPECT_EQ(counters.host_syncs, 4);  // 3 true checks + 1 false
}

TEST_F(CaptureWhile, ZeroInitialConditionSkipsBody) {
  LoopContext ctx(&counters);
  int *counter = ints.data(), *cond = ints.data() + 1, *trace = ints.data() + 2;
  set_int<<<1, 1, 0, stream.get()>>>(counter, 3);
  capture_while(ctx, stream.get(), cond, [&](cudaStream_t s) { dec<<<1, 1, 0, s>>>(counter, cond, trace); });
  EXPECT_EQ(read(0), 3);
}

TEST_F(CaptureWhile, CapturedConditionalNodeReplays) {
  LoopContext ctx(&counters);
  int *counter = ints.data(), *cond = ints.data() + 1, *trace = ints.data() + 2;
  cudaGraph_t g = capture(stream.get(), [&](cudaStream_t s) {
    set_int<<<1, 1, 0, s>>>(counter, 4);
    set_int<<<1, 1, 0, s>>>(cond, 1);
    capture_while(ctx, s, cond, [&](cudaStream_t b) { dec<<<1, 1, 0, b>>>(counter, cond, trace); });
  }, &counters);
  GraphExec exec;
  exec.instantiate(g);
  int syncs_before = counters.host_syncs;
  exec.launch(stream.get(), &counters);
  exec.launch(stream.get(), &counters);
  EXPECT_EQ(read(0), 0);
  // Two launches, each 4 iterations touching trace[3], [2], [1], [0].
  for (int i = 0; i < 4; ++i) EXPECT_EQ(read(2 + i), 2);
  EXPECT_EQ(counters.host_syncs, syncs_before);
  EXPECT_EQ(counters.graph_launches, 2);
}

TEST_F(CaptureWhile, NestedConditionalNodes) {
  LoopContext ctx(&counters);
  int *oc = ints.data(), *ocond = ints.data() + 1, *ic = ints.data() + 2, *icond = ints.data() + 3,
      *total = ints.data() + 4;
  cudaGraph_t g = capture(stream.get(), [&](cudaStream_t s) {
    set_int<<<1, 1, 0, s>>>(oc, 3);
    set_int<<<1, 1, 0, s>>>(ocond, 1);
    set_int<<<1, 1, 0, s>>>(ic, 2);
    set_int<<<1, 1, 0, s>>>(icond, 1);
    capture_while(ctx, s, ocond, [&](cudaStream_t b) {
      capture_while(ctx, b, icond, [&](cudaStream_t bb) { inner_body<<<1, 1, 0, bb>>>(ic, icond, total); });
      outer_tail<<<1, 1, 0, b>>>(oc, ocond, ic, icond);
    });
  }, &counters);
  GraphExec exec;
  exec.instantiate(g);
  exec.launch(stream.get(), &counters);
  EXPECT_EQ(read(4), 6);  // 3 outer x 2 inner
  // Same nesting eagerly gives the same count.
  set_int<<<1, 1, 0, stream.get()>>>(total, 0);
  set_int<<<1, 1, 0, stream.get()>>>(oc, 3);
  set_int<<<1, 1, 0, stream.get()>>>(ocond, 1);
  set_int<<<1, 1, 0, stream.get()>>>(ic, 2);
  set_int<<<1, 1, 0, stream.get()>>>(icond, 1);
  capture_while(ctx, stream.get(), ocond, [&](cudaStream_t b) {
    capture_while(ctx, b, icond, [&](cudaStream_t bb) { inner_body<<<1, 1, 0, bb>>>(ic, icond, total); });
    outer_tail<<<1, 1, 0, b>>>(oc, ocond, ic, icond);
  });
  EXPECT_EQ(read(4), 6);
}
