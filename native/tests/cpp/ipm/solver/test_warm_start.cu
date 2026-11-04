// Experimental initial-point hook (SparseSolver::warm_start_buffers /
// solve_warm) through the public C++ API.
#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <vector>

#include "common/buffer.hpp"
#include "cupiqp/cupiqp.hpp"

using namespace cupiqp;

namespace {

CsrPattern dense_pattern(int rows, int cols) {
  CsrPattern M;
  M.rows = rows;
  M.cols = cols;
  M.indptr.push_back(0);
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) M.indices.push_back(c);
    M.indptr.push_back(static_cast<int32_t>(M.indices.size()));
  }
  return M;
}

// Random feasible QP with equality, two-sided inequality and box rows; some
// bounds are infinite (the same construction as test_solver.cu).
struct Qp {
  int n, p, m;
  CsrPattern P, A, G;
  std::vector<double> Pv, Av, Gv, c, b, h_l, h_u, x_l, x_u;
  CudaBuffer<double> dP, dA, dG, dc, db, dhl, dhu, dxl, dxu;

  Qp(int n_, int p_, int m_, unsigned seed) : n(n_), p(p_), m(m_) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> U(-1, 1), U01(0, 1);
    P = dense_pattern(n, n);
    std::vector<double> L(n * n);
    for (auto& v : L) v = U(gen);
    Pv.assign(n * n, 0.0);
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < n; ++j) {
        double s = i == j ? 1.0 : 0.0;
        for (int k = 0; k < n; ++k) s += L[i * n + k] * L[j * n + k];
        Pv[i * n + j] = s;
      }
    std::vector<double> x0(n);
    for (auto& v : x0) v = U(gen);
    c.resize(n);
    for (auto& v : c) v = 3 * U(gen);
    A = dense_pattern(p, n);
    Av.resize(p * n);
    for (auto& v : Av) v = U(gen);
    b.assign(p, 0.0);
    for (int r = 0; r < p; ++r)
      for (int j = 0; j < n; ++j) b[r] += Av[r * n + j] * x0[j];
    G = dense_pattern(m, n);
    Gv.resize(m * n);
    for (auto& v : Gv) v = U(gen);
    h_l.resize(m);
    h_u.resize(m);
    for (int r = 0; r < m; ++r) {
      double g = 0;
      for (int j = 0; j < n; ++j) g += Gv[r * n + j] * x0[j];
      h_l[r] = U01(gen) < 0.3 ? -INFINITY : g - 0.2 * U01(gen);
      h_u[r] = U01(gen) < 0.3 ? INFINITY : g + 0.2 * U01(gen);
    }
    x_l.resize(n);
    x_u.resize(n);
    for (int j = 0; j < n; ++j) {
      x_l[j] = U01(gen) < 0.5 ? -INFINITY : x0[j] - 0.5;
      x_u[j] = U01(gen) < 0.5 ? INFINITY : x0[j] + 0.5;
    }
    upload();
  }
  void upload() {
    auto put = [](CudaBuffer<double>& d, const std::vector<double>& h) {
      if (d.size() == 0) d.allocate(h.size());
      CUPIQP_CUDA_CHECK(cudaMemcpy(d.data(), h.data(), sizeof(double) * h.size(), cudaMemcpyHostToDevice));
    };
    put(dP, Pv); put(dA, Av); put(dG, Gv); put(dc, c); put(db, b);
    put(dhl, h_l); put(dhu, h_u); put(dxl, x_l); put(dxu, x_u);
  }
  ValuesView view() const {
    ValuesView v;
    v.P = dP.data(); v.A = dA.data(); v.G = dG.data(); v.c = dc.data(); v.b = db.data();
    v.h_l = dhl.data(); v.h_u = dhu.data(); v.x_l = dxl.data(); v.x_u = dxu.data();
    return v;
  }
};

std::vector<double> to_host(const double* d, int n, cudaStream_t s) {
  std::vector<double> h(n);
  if (n) CUPIQP_CUDA_CHECK(cudaMemcpyAsync(h.data(), d, sizeof(double) * n, cudaMemcpyDeviceToHost, s));
  CUPIQP_CUDA_CHECK(cudaStreamSynchronize(s));
  return h;
}

double max_abs_diff(const std::vector<double>& a, const std::vector<double>& b) {
  double m = 0;
  for (size_t i = 0; i < a.size(); ++i) m = std::fmax(m, std::fabs(a[i] - b[i]));
  return m;
}

// Copy the current solution into the warm-start staging buffers (device or
// pinned host).
void stage_solution(SparseSolver& s, bool host) {
  const ResultView r = s.result();
  const WarmStartBuffers w = s.warm_start_buffers(host);
  const cudaMemcpyKind kind = host ? cudaMemcpyDeviceToHost : cudaMemcpyDeviceToDevice;
  auto cp = [&](double* dst, const double* src, int len) {
    if (len) CUPIQP_CUDA_CHECK(cudaMemcpyAsync(dst, src, sizeof(double) * len, kind, s.stream()));
  };
  cp(w.x, r.x, r.n); cp(w.y, r.y, r.p);
  cp(w.s_l, r.s_l, r.num_hl); cp(w.s_u, r.s_u, r.num_hu); cp(w.s_bl, r.s_bl, r.num_xl); cp(w.s_bu, r.s_bu, r.num_xu);
  cp(w.z_l, r.z_l, r.num_hl); cp(w.z_u, r.z_u, r.num_hu); cp(w.z_bl, r.z_bl, r.num_xl); cp(w.z_bu, r.z_bu, r.num_xu);
  CUPIQP_CUDA_CHECK(cudaStreamSynchronize(s.stream()));
}

const GraphMode kModes[] = {GraphMode::None, GraphMode::IterationGraphHostLoop, GraphMode::SolveGraphDeviceLoop};

}  // namespace

TEST(WarmStart, OwnSolutionTerminatesAtOnce) {
  Qp q(30, 5, 40, 7);
  for (GraphMode mode : kModes) {
    for (bool host : {false, true}) {
      SparseSolver s;
      s.settings().graph_mode = mode;
      s.setup(q.P, q.view(), &q.A, &q.G);
      s.solve();
      Info cold = s.info();
      ASSERT_EQ(cold.status, Status::Solved);
      auto x_cold = to_host(s.result().x, q.n, s.stream());
      stage_solution(s, host);
      s.solve_warm(kWarmAll, host);
      Info warm = s.info();
      EXPECT_EQ(warm.status, Status::Solved) << int(mode) << " host " << host;
      EXPECT_LE(warm.iter, 1) << int(mode) << " host " << host;
      EXPECT_LT(max_abs_diff(to_host(s.result().x, q.n, s.stream()), x_cold), 1e-9);
    }
  }
}

TEST(WarmStart, DefaultSolveIsBitwiseUnchanged) {
  Qp q(30, 5, 40, 9);
  for (GraphMode mode : kModes) {
    SparseSolver ref;
    ref.settings().graph_mode = mode;
    ref.setup(q.P, q.view(), &q.A, &q.G);
    ref.solve();
    const Info ri = ref.info();
    auto x_ref = to_host(ref.result().x, q.n, ref.stream());

    SparseSolver s;
    s.settings().graph_mode = mode;
    s.setup(q.P, q.view(), &q.A, &q.G);
    s.solve();
    stage_solution(s, false);
    s.solve_warm(kWarmAll, false);
    s.solve_warm(kWarmZ | kWarmY, false);
    s.solve();  // default path after warm solves
    const Info si = s.info();
    EXPECT_EQ(si.iter, ri.iter);
    EXPECT_EQ(max_abs_diff(to_host(s.result().x, q.n, s.stream()), x_ref), 0.0) << int(mode);
  }
}

TEST(WarmStart, PerturbedQpConvergesFromPreviousSolution) {
  Qp q(40, 6, 60, 13);
  SparseSolver s;
  s.setup(q.P, q.view(), &q.A, &q.G);
  s.solve();
  ASSERT_EQ(s.info().status, Status::Solved);
  stage_solution(s, true);
  // Perturb the cost and the inequality bounds slightly.
  std::mt19937 gen(5);
  std::uniform_real_distribution<double> U(-1e-2, 1e-2);
  for (auto& v : q.c) v += U(gen);
  for (auto& v : q.h_u) if (std::isfinite(v)) v += std::fabs(U(gen));
  q.upload();
  s.update(q.view());
  s.solve();
  const Info cold = s.info();
  ASSERT_EQ(cold.status, Status::Solved);
  auto x_cold = to_host(s.result().x, q.n, s.stream());
  for (uint32_t fields : {uint32_t(kWarmAll), uint32_t(kWarmZ), uint32_t(kWarmY | kWarmS | kWarmZ)}) {
    s.solve_warm(fields, true);
    const Info warm = s.info();
    EXPECT_EQ(warm.status, Status::Solved) << fields;
    EXPECT_LT(max_abs_diff(to_host(s.result().x, q.n, s.stream()), x_cold), 1e-4) << fields;
  }
}

TEST(WarmStart, RejectsBadCalls) {
  Qp q(10, 2, 10, 3);
  SparseSolver s;
  s.setup(q.P, q.view(), &q.A, &q.G);
  EXPECT_THROW(s.solve_warm(kWarmAll, false), Error);  // no buffers yet
  (void)s.warm_start_buffers(false);
  EXPECT_THROW(s.solve_warm(0u, false), Error);
  EXPECT_THROW(s.solve_warm(32u, false), Error);
}

TEST(WarmStart, RepeatedWarmSolvesAreOneGraphLaunchWithoutSyncOrAllocation) {
  Qp q(30, 5, 40, 21);
  SparseSolver s;
  s.settings().graph_mode = GraphMode::SolveGraphDeviceLoop;
  s.setup(q.P, q.view(), &q.A, &q.G);
  s.solve();
  stage_solution(s, true);
  s.solve_warm(kWarmAll, true);  // captures the warm graph
  CUPIQP_CUDA_CHECK(cudaStreamSynchronize(s.stream()));
  size_t free0 = 0, total = 0;
  CUPIQP_CUDA_CHECK(cudaMemGetInfo(&free0, &total));
  const Counters c0 = s.counters();
  for (int k = 0; k < 5; ++k) s.solve_warm(kWarmAll, true);
  const Counters c1 = s.counters();
  CUPIQP_CUDA_CHECK(cudaStreamSynchronize(s.stream()));
  size_t free1 = 0;
  CUPIQP_CUDA_CHECK(cudaMemGetInfo(&free1, &total));
  EXPECT_EQ(c1.host_syncs, c0.host_syncs);
  EXPECT_EQ(c1.graph_captures, c0.graph_captures);
  EXPECT_EQ(c1.graph_launches, c0.graph_launches + 5);
  EXPECT_EQ(free0, free1);
  EXPECT_EQ(s.info().status, Status::Solved);
}
