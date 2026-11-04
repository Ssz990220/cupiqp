// M2 spmv tests: SpMV / SparseOperators against a CPU reference, empty
// matrices, alpha / beta, transposes, graph capture (plain and inside a
// conditional WHILE body), in-place value updates and no allocation per call.
#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <limits>
#include <tuple>
#include <vector>

#include "common/graph.hpp"
#include "common/handles.hpp"
#include "linalg/spmv/spmv.hpp"
#include "spmv_test_util.cuh"

using namespace cupiqp;
using namespace cupiqp::spmv_test;

namespace {

bool bitwise_equal(const std::vector<double>& a, const std::vector<double>& b, bool quiet = false) {
  if (!quiet && a.size() == b.size()) {
    double md = 0;
    int cnt = 0;
    for (size_t i = 0; i < a.size(); ++i) {
      if (std::memcmp(&a[i], &b[i], sizeof(double)) != 0) ++cnt;
      md = std::fmax(md, std::fabs(a[i] - b[i]));
    }
    if (cnt) std::printf("  [diff] %d of %zu entries differ, max abs %.3e\n", cnt, a.size(), md);
  }
  return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(double)) == 0);
}

void expect_close(const std::vector<double>& got, const std::vector<double>& want, const std::vector<double>& mag,
                  double rtol = 1e-14) {
  ASSERT_EQ(got.size(), want.size());
  for (size_t i = 0; i < got.size(); ++i) {
    const double tol = rtol * (mag[i] + 1e-300) + 1e-300;
    EXPECT_LE(std::fabs(got[i] - want[i]), tol) << "i=" << i << " got " << got[i] << " want " << want[i];
  }
}

__global__ void decrement(int32_t* counter, int32_t* cond) {
  const int32_t v = --(*counter);
  *cond = v > 0;
}
__global__ void set_int(int32_t* p, int32_t v) { *p = v; }
__global__ void axpy(int32_t k, const double* t, double* y) {
  const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < k) y[i] += t[i];
}

class SpmvTest : public ::testing::Test {
 protected:
  CudaStream stream = CudaStream::create(cudaStreamNonBlocking);
};

}  // namespace

// ------------------------------------------------------------ random vs CPU
using Case = std::tuple<int, int, double>;  // rows, cols, density
class SpmvRandom : public SpmvTest, public ::testing::WithParamInterface<std::tuple<Case, bool>> {};

TEST_P(SpmvRandom, MatchesCpuWithAlphaBeta) {
  const auto [shape, transpose] = GetParam();
  const auto [rows, cols, density] = shape;
  HostCsr M = random_csr(rows, cols, density, 1000 * rows + cols);
  DeviceCsr dM(M);
  SpMV op;
  op.init(dM.view(), transpose);
  EXPECT_TRUE(op.computes_product());
  const int32_t nx = transpose ? rows : cols, ny = transpose ? cols : rows;
  ASSERT_EQ(op.x_len(), nx);
  ASSERT_EQ(op.y_len(), ny);
  const std::vector<double> x = random_vec(nx, 7);
  const std::vector<double> y0 = random_vec(ny, 8);
  for (auto [alpha, beta] : {std::pair{1.0, 0.0}, std::pair{2.0, 0.5}, std::pair{-1.5, 2.0}, std::pair{0.75, 1.0}}) {
    CudaBuffer<double> dx = to_device(x), dy = to_device(y0);
    op(stream.get(), alpha, dx.data(), beta, dy.data());
    std::vector<double> want = y0, mag;
    cpu_spmv(M, transpose, alpha, x, beta, want, &mag);
    expect_close(to_host(dy, stream.get()), want, mag);
  }
}

INSTANTIATE_TEST_SUITE_P(Shapes, SpmvRandom,
                         ::testing::Combine(::testing::Values(Case{1, 1, 1.0}, Case{1, 8, 0.5}, Case{8, 1, 0.5},
                                                              Case{200, 32, 0.3}, Case{32, 200, 0.3},
                                                              Case{37, 53, 0.05}, Case{37, 53, 0.9},
                                                              Case{1024, 256, 0.3}),
                                            ::testing::Bool()));

// ------------------------------------------------------------ empty matrices
TEST_F(SpmvTest, EmptyMatricesSkipTheProduct) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  // (rows, cols, store nothing): rows == 0, cols == 0, nnz == 0.
  for (auto [rows, cols] : {std::pair{0, 5}, std::pair{5, 0}, std::pair{4, 6}, std::pair{0, 0}}) {
    HostCsr M;
    M.rows = rows;
    M.cols = cols;
    M.indptr.assign(rows + 1, 0);
    DeviceCsr dM(M);
    for (bool transpose : {false, true}) {
      SpMV op;
      op.init(dM.view(), transpose);
      EXPECT_FALSE(op.computes_product());
      const int32_t ny = op.y_len(), nx = op.x_len();
      CudaBuffer<double> dx = to_device(std::vector<double>(nx, 1.0));
      // beta == 0: y = 0 even if y held NaN.
      CudaBuffer<double> dy = to_device(std::vector<double>(ny, nan));
      op(stream.get(), 3.0, dx.data(), 0.0, dy.data());
      for (double v : to_host(dy, stream.get())) EXPECT_EQ(v, 0.0);
      // beta != 0: y = beta y; beta == 1: unchanged.
      std::vector<double> y0 = random_vec(ny, 3);
      CudaBuffer<double> dy2 = to_device(y0);
      op(stream.get(), 3.0, dx.data(), -2.0, dy2.data());
      op(stream.get(), 3.0, dx.data(), 1.0, dy2.data());
      auto got = to_host(dy2, stream.get());
      for (int32_t i = 0; i < ny; ++i) EXPECT_EQ(got[i], -2.0 * y0[i]);
    }
  }
}

// ------------------------------------------------------------ graph capture
// The kernels use fixed summation orders, so capture/replay must be bitwise
// equal to eager for every op and beta.
TEST_F(SpmvTest, CaptureReplayMatchesEager) {
  HostCsr M = random_csr(300, 120, 0.4, 11);
  DeviceCsr dM(M);
  for (bool transpose : {false, true}) {
    for (double beta : {0.0, 0.5}) {
      SpMV op;
      op.init(dM.view(), transpose);
      const std::vector<double> x = random_vec(op.x_len(), 1), y0 = random_vec(op.y_len(), 2);
      CudaBuffer<double> dx = to_device(x), dy = to_device(y0);
      op(stream.get(), 1.25, dx.data(), beta, dy.data());
      const auto eager = to_host(dy, stream.get());
      std::vector<double> want = y0, mag;
      cpu_spmv(M, transpose, 1.25, x, beta, want, &mag);
      expect_close(eager, want, mag);

      Counters c;
      GraphExec exec;
      exec.instantiate(capture(stream.get(), [&](cudaStream_t s) { op(s, 1.25, dx.data(), beta, dy.data()); }, &c));
      for (int rep = 0; rep < 5; ++rep) {
        dy.copy_from_host(y0.data(), y0.size(), stream.get());
        exec.launch(stream.get(), &c);
        const auto replay = to_host(dy, stream.get());
        EXPECT_TRUE(bitwise_equal(replay, eager)) << "transpose=" << transpose << " beta=" << beta;
        expect_close(replay, want, mag);
        expect_close(replay, eager, mag);
      }
    }
  }
}

// Eager calls are bitwise reproducible. The op = T init uploads a ~1 MB CSC
// pattern right before the first call on a non-blocking stream (regression
// check for the pageable-copy race fixed in init).
TEST_F(SpmvTest, EagerRepeatability) {
  HostCsr M = random_csr(2521, 169, 0.59, 11, 0.8, 0.0);
  DeviceCsr dM(M);
  for (bool transpose : {false, true}) {
    for (double beta : {0.0, 0.5}) {
      SpMV op;
      op.init(dM.view(), transpose);
      const auto y0 = random_vec(op.y_len(), 2);
      CudaBuffer<double> dx = to_device(random_vec(op.x_len(), 1)), dy(op.y_len());
      std::vector<double> first;
      int differ = 0;
      for (int r = 0; r < 20; ++r) {
        dy.copy_from_host(y0.data(), y0.size(), stream.get());
        op(stream.get(), 1.0, dx.data(), beta, dy.data());
        auto got = to_host(dy, stream.get());
        if (r == 0) {
          first = got;
          std::vector<double> want = y0, mag;
          cpu_spmv(M, transpose, 1.0, random_vec(op.x_len(), 1), beta, want, &mag);
          expect_close(got, want, mag);
        }
        else differ += !bitwise_equal(got, first, /*quiet=*/true);
      }
      EXPECT_EQ(differ, 0) << "2521x169 op=" << (transpose ? "T" : "N") << " beta=" << beta;
    }
  }
}

TEST_F(SpmvTest, InPlaceValueUpdatesSeenEagerAndReplay) {
  HostCsr M = random_csr(64, 48, 0.3, 21);
  DeviceCsr dM(M);
  for (bool transpose : {false, true}) {
    SpMV op;
    op.init(dM.view(), transpose);
    const std::vector<double> x = random_vec(op.x_len(), 5);
    CudaBuffer<double> dx = to_device(x), dy(op.y_len());
    Counters c;
    GraphExec exec;
    exec.instantiate(capture(stream.get(), [&](cudaStream_t s) { op(s, 1.0, dx.data(), 0.0, dy.data()); }, &c));
    for (int version = 0; version < 3; ++version) {
      HostCsr Mv = M;
      for (size_t k = 0; k < Mv.values.size(); ++k) Mv.values[k] = (version + 1.0) * (k % 7 == 0 ? 1.0 : Mv.values[k]);
      dM.values.copy_from_host(Mv.values.data(), Mv.values.size(), stream.get());
      std::vector<double> want(op.y_len(), 0.0), mag;
      cpu_spmv(Mv, transpose, 1.0, x, 0.0, want, &mag);
      op(stream.get(), 1.0, dx.data(), 0.0, dy.data());
      expect_close(to_host(dy, stream.get()), want, mag);
      dy.zero_async(stream.get());
      exec.launch(stream.get(), &c);
      expect_close(to_host(dy, stream.get()), want, mag);
    }
    dM.values.copy_from_host(M.values.data(), M.values.size(), stream.get());
  }
}

// Inside a conditional WHILE body (the whole-solve graph pattern): K
// iterations of t = alpha op(M) x; y += t, host loop vs captured node
// (bitwise equal).
TEST_F(SpmvTest, InsideConditionalWhileBody) {
  HostCsr M = random_csr(2521 / 8, 169, 0.6, 31, 0.8, 0.05);
  DeviceCsr dM(M);
  for (bool transpose : {false, true}) {
    SpMV op;
    op.init(dM.view(), transpose);
    const std::vector<double> x = random_vec(op.x_len(), 9), y0 = random_vec(op.y_len(), 10);
    CudaBuffer<double> dx = to_device(x), dy = to_device(y0), dt(op.y_len());
    CudaBuffer<int32_t> ints(2);
    int32_t *counter = ints.data(), *cond = ints.data() + 1;
    const int32_t ny = op.y_len();
    Counters c;
    LoopContext ctx(&c);
    constexpr int K = 6;
    auto body = [&](cudaStream_t s) {
      op(s, 0.5, dx.data(), 0.0, dt.data());
      axpy<<<(ny + 255) / 256, 256, 0, s>>>(ny, dt.data(), dy.data());
      decrement<<<1, 1, 0, s>>>(counter, cond);
    };
    auto program = [&](cudaStream_t s) {
      set_int<<<1, 1, 0, s>>>(counter, K);
      set_int<<<1, 1, 0, s>>>(cond, 1);
      capture_while(ctx, s, cond, body);
    };
    // Host loop (not capturing).
    program(stream.get());
    const auto eager = to_host(dy, stream.get());
    std::vector<double> t(ny, 0.0), tmag;
    cpu_spmv(M, transpose, 0.5, x, 0.0, t, &tmag);
    std::vector<double> want = y0, mag(ny);
    for (int k = 0; k < K; ++k)
      for (int i = 0; i < ny; ++i) want[i] += t[i];
    for (int i = 0; i < ny; ++i) mag[i] = K * tmag[i] + std::fabs(y0[i]) + std::fabs(want[i]);
    expect_close(eager, want, mag);
    // Captured conditional node.
    GraphExec exec;
    exec.instantiate(capture(stream.get(), program, &c));
    for (int rep = 0; rep < 3; ++rep) {
      dy.copy_from_host(y0.data(), y0.size(), stream.get());
      exec.launch(stream.get(), &c);
      const auto replay = to_host(dy, stream.get());
      EXPECT_TRUE(bitwise_equal(replay, eager)) << "rep=" << rep;
      expect_close(replay, want, mag);
    }
  }
}

// ------------------------------------------------------------ no allocation
// Calls under a global-mode capture fail on any cudaMalloc / sync; the default
// memory pool must not grow over many eager calls.
TEST_F(SpmvTest, NoAllocationOrSyncPerCall) {
  HostCsr M = random_csr(500, 169, 0.5, 41);
  DeviceCsr dM(M);
  SpMV fwd, tr;
  fwd.init(dM.view(), false);
  tr.init(dM.view(), true);
  CudaBuffer<double> x1 = to_device(random_vec(169, 1)), y1(500), x2 = to_device(random_vec(500, 2)), y2(169);

  cudaMemPool_t pool;
  CUPIQP_CUDA_CHECK(cudaDeviceGetDefaultMemPool(&pool, 0));
  uint64_t used_before = 0, used_after = 0;
  CUPIQP_CUDA_CHECK(cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemCurrent, &used_before));
  size_t free_before = 0, free_after = 0, total = 0;
  CUPIQP_CUDA_CHECK(cudaMemGetInfo(&free_before, &total));
  for (int i = 0; i < 2000; ++i) {
    fwd(stream.get(), 1.0, x1.data(), 0.0, y1.data());
    tr(stream.get(), 1.0, x2.data(), 0.0, y2.data());
  }
  CUPIQP_CUDA_CHECK(cudaStreamSynchronize(stream.get()));
  CUPIQP_CUDA_CHECK(cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemCurrent, &used_after));
  CUPIQP_CUDA_CHECK(cudaMemGetInfo(&free_after, &total));
  EXPECT_EQ(used_before, used_after);
  // Device-wide free memory is shared with other processes: report only.
  std::printf("[ info ] free device memory change over 4000 calls: %lld bytes\n",
              static_cast<long long>(free_before) - static_cast<long long>(free_after));

  cudaGraph_t g = nullptr;
  CUPIQP_CUDA_CHECK(cudaStreamBeginCapture(stream.get(), cudaStreamCaptureModeGlobal));
  for (int i = 0; i < 50; ++i) {
    fwd(stream.get(), 1.0, x1.data(), 0.0, y1.data());
    tr(stream.get(), 1.0, x2.data(), 0.0, y2.data());
  }
  CUPIQP_CUDA_CHECK(cudaStreamEndCapture(stream.get(), &g));
  size_t nodes = 0;
  CUPIQP_CUDA_CHECK(cudaGraphGetNodes(g, nullptr, &nodes));
  std::printf("[ info ] graph nodes for 100 SpMV calls: %zu\n", nodes);
  // Only kernel nodes (no memset / memcpy / alloc nodes).
  std::vector<cudaGraphNode_t> all(nodes);
  CUPIQP_CUDA_CHECK(cudaGraphGetNodes(g, all.data(), &nodes));
  for (auto nd : all) {
    cudaGraphNodeType t;
    CUPIQP_CUDA_CHECK(cudaGraphNodeGetType(nd, &t));
    EXPECT_TRUE(t == cudaGraphNodeTypeKernel || t == cudaGraphNodeTypeMemset) << "node type " << (int)t;
  }
  CUPIQP_CUDA_CHECK(cudaGraphDestroy(g));
}

// ------------------------------------------------------------ SparseOperators
namespace {
struct OpsFixture {
  HostCsr P, A, G;
  std::unique_ptr<DeviceCsr> dP, dA, dG;
  SparseDataView<double> view(int n, int p, int m) {
    dP = std::make_unique<DeviceCsr>(P);
    dA = std::make_unique<DeviceCsr>(A);
    dG = std::make_unique<DeviceCsr>(G);
    SparseDataView<double> d;
    d.dims.n = n;
    d.dims.p = p;
    d.dims.m = m;
    d.P = dP->view();
    d.A = dA->view();
    d.G = dG->view();
    return d;
  }
};
}  // namespace

TEST_F(SpmvTest, SparseOperatorsMatchCpu) {
  for (auto [n, p, m] : {std::tuple{20, 5, 30}, std::tuple{20, 0, 30}, std::tuple{20, 5, 0}, std::tuple{20, 0, 0}}) {
    OpsFixture f;
    f.P = random_csr(n, n, 0.3, 1, 0.2, 0.0);
    f.A = p ? random_csr(p, n, 0.4, 2) : HostCsr{0, n, {0}, {}, {}};
    f.G = m ? random_csr(m, n, 0.4, 3) : HostCsr{0, n, {0}, {}, {}};
    SparseOperators ops;
    ops.init(f.view(n, p, m));
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const auto xn = random_vec(n, 4), xp = random_vec(p, 5), xm = random_vec(m, 6);
    CudaBuffer<double> dxn = to_device(xn), dxp = to_device(xp), dxm = to_device(xm);
    auto check = [&](auto fn, const HostCsr& M, bool tr, const std::vector<double>& x, CudaBuffer<double>& dx) {
      const int32_t ny = tr ? M.cols : M.rows;
      CudaBuffer<double> dy = to_device(std::vector<double>(ny, nan));
      fn(stream.get(), -0.5, dx.data(), dy.data());
      std::vector<double> want(ny, 0.0), mag;
      cpu_spmv(M, tr, -0.5, x, 0.0, want, &mag);
      expect_close(to_host(dy, stream.get()), want, mag);
    };
    check([&](cudaStream_t s, double a, const double* x, double* y) { ops.P(s, a, x, y); }, f.P, false, xn, dxn);
    check([&](cudaStream_t s, double a, const double* x, double* y) { ops.A(s, a, x, y); }, f.A, false, xn, dxn);
    check([&](cudaStream_t s, double a, const double* x, double* y) { ops.AT(s, a, x, y); }, f.A, true, xp, dxp);
    check([&](cudaStream_t s, double a, const double* x, double* y) { ops.G(s, a, x, y); }, f.G, false, xn, dxn);
    check([&](cudaStream_t s, double a, const double* x, double* y) { ops.GT(s, a, x, y); }, f.G, true, xm, dxm);
  }
}

