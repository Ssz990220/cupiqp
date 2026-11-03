// M6 kkt_system tests that need no dependency module: capture_if semantics
// (host path and conditional IF node), and the finite check / rhs assembly
// launchers on a B = 1 problem.
#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "common/buffer.hpp"
#include "common/graph.hpp"
#include "common/handles.hpp"
#include "linalg/kkt_system/capture_if.hpp"
#include "linalg/kkt_system/kkt_system_kernels.hpp"

using namespace cupiqp;

namespace {

__global__ void add_one(int32_t* v) { *v += 1; }
__global__ void set_val(int32_t* p, int32_t v) { *p = v; }
__global__ void countdown_step(int32_t* counter, int32_t* loop, int32_t* odd) {
  int32_t v = --(*counter);
  *odd = v & 1;
  *loop = v > 0;
}

int32_t read(const int32_t* d) {
  int32_t h = 0;
  CUPIQP_CUDA_CHECK(cudaMemcpy(&h, d, sizeof(int32_t), cudaMemcpyDeviceToHost));
  return h;
}

}  // namespace

TEST(KktSystemCaptureIf, HostPath) {
  Counters counters;
  LoopContext ctx(&counters);
  CudaStream s = CudaStream::create();
  CudaBuffer<int32_t> cond(1), hits(1);
  CudaPinnedBuffer<int32_t> flag(1);
  hits.zero_async(s.get());
  for (int c : {0, 1, 5, 0}) {
    set_val<<<1, 1, 0, s.get()>>>(cond.data(), c);
    capture_if(ctx, s.get(), cond.data(), flag.data(), [&](cudaStream_t st) { add_one<<<1, 1, 0, st>>>(hits.data()); });
  }
  CUPIQP_CUDA_CHECK(cudaStreamSynchronize(s.get()));
  EXPECT_EQ(read(hits.data()), 2);
  EXPECT_EQ(counters.host_syncs, 4);
}

TEST(KktSystemCaptureIf, ConditionalNodeNestedInWhile) {
  // while (counter > 0) { counter--; if (counter is odd) hits++; }
  Counters counters;
  LoopContext ctx(&counters);
  CudaStream s = CudaStream::create();
  CudaBuffer<int32_t> counter(1), loop(1), odd(1), hits(1);
  CudaPinnedBuffer<int32_t> flag(1);
  GraphExec g;
  g.instantiate(capture(
      s.get(),
      [&](cudaStream_t cs) {
        set_val<<<1, 1, 0, cs>>>(counter.data(), 7);
        set_val<<<1, 1, 0, cs>>>(loop.data(), 1);
        set_val<<<1, 1, 0, cs>>>(hits.data(), 0);
        capture_while(ctx, cs, loop.data(), [&](cudaStream_t bs) {
          countdown_step<<<1, 1, 0, bs>>>(counter.data(), loop.data(), odd.data());
          capture_if(ctx, bs, odd.data(), flag.data(),
                     [&](cudaStream_t is) { add_one<<<1, 1, 0, is>>>(hits.data()); });
        });
      },
      &counters));
  for (int rep = 0; rep < 2; ++rep) {
    g.launch(s.get(), &counters);
    CUPIQP_CUDA_CHECK(cudaStreamSynchronize(s.get()));
    EXPECT_EQ(read(hits.data()), 3);  // v = 6..0: odd at 5, 3, 1
    EXPECT_EQ(read(counter.data()), 0);
  }
  EXPECT_EQ(ctx.depth, 0);
}

TEST(KktSystemKernels, FiniteCheckAndAssembly) {
  CudaStream s = CudaStream::create();
  const int n = 300, p = 7, m = 40, W = n + p + m;
  std::vector<double> h(W);
  for (int i = 0; i < W; ++i) h[i] = 0.5 * i - 3.0;
  CudaBuffer<double> xyz(W), out(W);
  CudaBuffer<int32_t> status(1);
  xyz.copy_from_host(h.data(), W, s.get());
  const double* x = xyz.data();
  kkt_kernels::assemble_rhs(1, n, p, m, CDMat{x, n}, CDMat{x + n, p}, CDMat{x + n + p, m}, DMat{out.data(), W},
                            s.get());
  kkt_kernels::solution_finite(1, n, p, m, CDMat{x, n}, CDMat{x + n, p}, CDMat{x + n + p, m}, status.data(),
                               s.get());
  std::vector<double> back(W);
  CUPIQP_CUDA_CHECK(cudaMemcpyAsync(back.data(), out.data(), W * sizeof(double), cudaMemcpyDeviceToHost, s.get()));
  CUPIQP_CUDA_CHECK(cudaStreamSynchronize(s.get()));
  EXPECT_EQ(back, h);
  EXPECT_EQ(read(status.data()), 0);
  for (int pos : {0, n + 3, W - 1}) {
    double bad = (pos == 0) ? NAN : (pos == W - 1 ? -INFINITY : INFINITY);
    CUPIQP_CUDA_CHECK(cudaMemcpy(xyz.data() + pos, &bad, sizeof(double), cudaMemcpyHostToDevice));
    kkt_kernels::solution_finite(1, n, p, m, CDMat{x, n}, CDMat{x + n, p}, CDMat{x + n + p, m}, status.data(),
                                 s.get());
    CUPIQP_CUDA_CHECK(cudaStreamSynchronize(s.get()));
    EXPECT_EQ(read(status.data()), 1) << pos;
    CUPIQP_CUDA_CHECK(cudaMemcpy(xyz.data() + pos, &h[pos], sizeof(double), cudaMemcpyHostToDevice));
  }
}
