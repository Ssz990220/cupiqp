// ipm/solver integration tests through the public C++ API (no Python).
#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <vector>

#include "common/buffer.hpp"
#include "cupiqp/cupiqp.hpp"

using namespace cupiqp;

namespace {

struct HostQp {
  CsrPattern P, A, G;
  std::vector<double> Pv, Av, Gv, c, b, h_l, h_u, x_l, x_u;
};

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

// Random feasible QP: P = L L' + I (dense pattern, full storage), A x = b with
// b = A x0, h_l <= G x <= h_u around G x0, box bounds around x0. A fraction of
// stored entries are explicit zeros; some bounds are infinite.
HostQp random_qp(int n, int p, int m, unsigned seed, double zero_frac = 0.3) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<double> U(-1, 1);
  std::uniform_real_distribution<double> U01(0, 1);
  HostQp q;
  q.P = dense_pattern(n, n);
  std::vector<double> L(n * n);
  for (auto& v : L) v = U(gen) * (U01(gen) < zero_frac ? 0.0 : 1.0);
  q.Pv.assign(n * n, 0.0);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) {
      double s = i == j ? 1.0 : 0.0;
      for (int k = 0; k < n; ++k) s += L[i * n + k] * L[j * n + k];
      q.Pv[i * n + j] = s;
    }
  std::vector<double> x0(n);
  for (auto& v : x0) v = U(gen);
  q.c.resize(n);
  for (auto& v : q.c) v = U(gen);
  q.A = dense_pattern(p, n);
  q.Av.resize(p * n);
  for (auto& v : q.Av) v = U01(gen) < zero_frac ? 0.0 : U(gen);
  q.b.assign(p, 0.0);
  for (int r = 0; r < p; ++r)
    for (int j = 0; j < n; ++j) q.b[r] += q.Av[r * n + j] * x0[j];
  q.G = dense_pattern(m, n);
  q.Gv.resize(m * n);
  for (auto& v : q.Gv) v = U01(gen) < zero_frac ? 0.0 : U(gen);
  q.h_l.resize(m);
  q.h_u.resize(m);
  for (int r = 0; r < m; ++r) {
    double g = 0;
    for (int j = 0; j < n; ++j) g += q.Gv[r * n + j] * x0[j];
    q.h_l[r] = U01(gen) < 0.3 ? -INFINITY : g - U01(gen);
    q.h_u[r] = U01(gen) < 0.3 ? INFINITY : g + U01(gen);
  }
  q.x_l.resize(n);
  q.x_u.resize(n);
  for (int j = 0; j < n; ++j) {
    q.x_l[j] = U01(gen) < 0.5 ? -INFINITY : x0[j] - 1;
    q.x_u[j] = U01(gen) < 0.5 ? INFINITY : x0[j] + 1;
  }
  return q;
}

// Device copies of the value arrays of a HostQp.
struct DeviceQp {
  CudaBuffer<double> P, A, G, c, b, h_l, h_u, x_l, x_u;
  explicit DeviceQp(const HostQp& q) { upload(q); }
  void upload(const HostQp& q) {
    auto put = [](CudaBuffer<double>& d, const std::vector<double>& h) {
      if (d.size() == 0 && !h.empty()) d.allocate(h.size());
      if (!h.empty()) CUPIQP_CUDA_CHECK(cudaMemcpy(d.data(), h.data(), sizeof(double) * h.size(), cudaMemcpyHostToDevice));
    };
    put(P, q.Pv); put(A, q.Av); put(G, q.Gv); put(c, q.c); put(b, q.b);
    put(h_l, q.h_l); put(h_u, q.h_u); put(x_l, q.x_l); put(x_u, q.x_u);
  }
  ValuesView view() const {
    ValuesView v;
    v.P = P.data(); v.A = A.data(); v.G = G.data(); v.c = c.data(); v.b = b.data();
    v.h_l = h_l.data(); v.h_u = h_u.data(); v.x_l = x_l.data(); v.x_u = x_u.data();
    return v;
  }
};

std::vector<double> host_x(SparseSolver& s) {
  std::vector<double> x(s.n());
  CUPIQP_CUDA_CHECK(cudaMemcpyAsync(x.data(), s.result().x, sizeof(double) * s.n(), cudaMemcpyDeviceToHost, s.stream()));
  CUPIQP_CUDA_CHECK(cudaStreamSynchronize(s.stream()));
  return x;
}

double max_abs_diff(const std::vector<double>& a, const std::vector<double>& b) {
  double m = 0;
  for (size_t i = 0; i < a.size(); ++i) m = std::fmax(m, std::fabs(a[i] - b[i]));
  return m;
}

// KKT check in the user space for min 1/2 x'Px + c'x, A x = b, h_l <= Gx <= h_u, x_l <= x <= x_u.
double primal_violation(const HostQp& q, const std::vector<double>& x) {
  const int n = q.P.rows, p = q.A.rows, m = q.G.rows;
  double v = 0;
  for (int r = 0; r < p; ++r) {
    double s = -q.b[r];
    for (int j = 0; j < n; ++j) s += q.Av[r * n + j] * x[j];
    v = std::fmax(v, std::fabs(s));
  }
  for (int r = 0; r < m; ++r) {
    double s = 0;
    for (int j = 0; j < n; ++j) s += q.Gv[r * n + j] * x[j];
    v = std::fmax(v, std::fmax(q.h_l[r] - s, s - q.h_u[r]));
  }
  for (int j = 0; j < n; ++j) v = std::fmax(v, std::fmax(q.x_l[j] - x[j], x[j] - q.x_u[j]));
  return v;
}

SparseSolver make_solver(const HostQp& q, const DeviceQp& d, GraphMode mode) {
  SparseSolver s;
  s.settings().graph_mode = mode;
  s.settings().enable_cuda_graph = mode != GraphMode::None;
  s.setup(q.P, d.view(), &q.A, &q.G);
  return s;
}

}  // namespace

TEST(Solver, AnalyticEqualityQp) {
  // min 1/2 (x0^2 + x1^2)  s.t. x0 + x1 = 2  ->  x = (1, 1)
  CsrPattern P{2, 2, {0, 1, 2}, {0, 1}}, A{1, 2, {0, 2}, {0, 1}};
  std::vector<double> Pv{1, 1}, Av{1, 1}, c{0, 0}, b{2};
  CudaBuffer<double> dP(2), dA(2), dc(2), db(1);
  dP.copy_from_host(Pv.data(), 2, nullptr); dA.copy_from_host(Av.data(), 2, nullptr);
  dc.copy_from_host(c.data(), 2, nullptr); db.copy_from_host(b.data(), 1, nullptr);
  CUPIQP_CUDA_CHECK(cudaDeviceSynchronize());
  for (GraphMode mode : {GraphMode::None, GraphMode::IterationGraphHostLoop, GraphMode::SolveGraphDeviceLoop}) {
    SparseSolver s;
    s.settings().graph_mode = mode;
    ValuesView v; v.P = dP.data(); v.A = dA.data(); v.c = dc.data(); v.b = db.data();
    s.setup(P, v, &A, nullptr);
    s.solve();
    Info info = s.info();
    EXPECT_EQ(info.status, Status::Solved) << static_cast<int>(mode);
    auto x = host_x(s);
    EXPECT_NEAR(x[0], 1.0, 1e-7);
    EXPECT_NEAR(x[1], 1.0, 1e-7);
    EXPECT_NEAR(info.primal_obj, 1.0, 1e-7);
  }
}

TEST(Solver, GraphModesAgreeOnRepeatedSolves) {
  HostQp q = random_qp(30, 6, 25, 11);
  DeviceQp d(q);
  std::vector<double> x_ref;
  int iter_ref = -1;
  for (GraphMode mode : {GraphMode::None, GraphMode::IterationGraphHostLoop, GraphMode::SolveGraphDeviceLoop}) {
    SparseSolver s = make_solver(q, d, mode);
    for (int rep = 0; rep < 3; ++rep) {
      s.solve();
      Info info = s.info();
      ASSERT_EQ(info.status, Status::Solved) << "mode " << static_cast<int>(mode) << " rep " << rep;
      auto x = host_x(s);
      EXPECT_LT(primal_violation(q, x), 1e-6);
      if (x_ref.empty()) {
        x_ref = x;
        iter_ref = info.iter;
      } else {
        EXPECT_LT(max_abs_diff(x, x_ref), 1e-9) << "mode " << static_cast<int>(mode);
        EXPECT_NEAR(info.iter, iter_ref, 1) << "mode " << static_cast<int>(mode);
      }
    }
    Counters c = s.counters();
    EXPECT_EQ(c.cudss_analysis_calls, 1);
  }
}

TEST(Solver, RepeatedUpdatesKeepAddressesAndAllocateNothing) {
  HostQp q = random_qp(25, 5, 20, 3);
  DeviceQp d(q);
  SparseSolver s = make_solver(q, d, GraphMode::IterationGraphHostLoop);
  s.solve();
  ASSERT_EQ(s.info().status, Status::Solved);
  const ResultView r0 = s.result();
  const auto dbg0 = s.debug_buffers();
  const Counters c0 = s.counters();
  size_t free0 = 0, total = 0;
  CUPIQP_CUDA_CHECK(cudaMemGetInfo(&free0, &total));
  std::mt19937 gen(5);
  std::uniform_real_distribution<double> U(-0.5, 0.5);
  for (int k = 0; k < 5; ++k) {
    HostQp q2 = q;
    // Turn explicit zeros into nonzeros and perturb values, same patterns.
    for (auto& v : q2.Gv) v = v == 0.0 ? 0.05 * U(gen) : v * (1 + 0.1 * U(gen));
    for (auto& v : q2.c) v += U(gen);
    // Re-derive feasible bounds from a fresh point.
    std::vector<double> x0(q.P.rows);
    for (auto& v : x0) v = U(gen);
    for (int r = 0; r < q.A.rows; ++r) {
      q2.b[r] = 0;
      for (int j = 0; j < q.P.rows; ++j) q2.b[r] += q2.Av[r * q.P.rows + j] * x0[j];
    }
    for (int r = 0; r < q.G.rows; ++r) {
      double g = 0;
      for (int j = 0; j < q.P.rows; ++j) g += q2.Gv[r * q.P.rows + j] * x0[j];
      q2.h_l[r] = (k + r) % 4 == 0 ? -INFINITY : g - 0.5;  // finite/infinite switches
      q2.h_u[r] = (k + r) % 5 == 0 ? INFINITY : g + 0.5;
    }
    for (int j = 0; j < q.P.rows; ++j) { q2.x_l[j] = x0[j] - 1; q2.x_u[j] = x0[j] + 1; }
    d.upload(q2);
    s.update(d.view());
    s.solve();
    Info info = s.info();
    EXPECT_EQ(info.status, Status::Solved) << "update " << k;
    EXPECT_LT(primal_violation(q2, host_x(s)), 1e-6) << "update " << k;
  }
  const ResultView r1 = s.result();
  EXPECT_EQ(r0.x, r1.x);
  EXPECT_EQ(r0.y, r1.y);
  EXPECT_EQ(r0.z_l, r1.z_l);
  const auto dbg1 = s.debug_buffers();
  ASSERT_EQ(dbg0.size(), dbg1.size());
  for (size_t i = 0; i < dbg0.size(); ++i) EXPECT_EQ(dbg0[i].ptr, dbg1[i].ptr) << dbg0[i].name;
  size_t free1 = 0;
  CUPIQP_CUDA_CHECK(cudaMemGetInfo(&free1, &total));
  EXPECT_EQ(free0, free1);
  const Counters c1 = s.counters();
  EXPECT_EQ(c1.cudss_analysis_calls, 1);
  EXPECT_EQ(c1.cudss_device_allocs, c0.cudss_device_allocs);
  EXPECT_EQ(c1.graph_captures, c0.graph_captures);  // the iteration graph is reused
}

TEST(Solver, NanInputGivesNumericalIssues) {
  HostQp q = random_qp(10, 2, 8, 9);
  DeviceQp d(q);
  for (GraphMode mode : {GraphMode::None, GraphMode::IterationGraphHostLoop}) {
    SparseSolver s = make_solver(q, d, mode);
    s.solve();
    ASSERT_EQ(s.info().status, Status::Solved);
    HostQp bad = q;
    bad.c[3] = NAN;
    DeviceQp dbad(bad);
    s.update(dbad.view());
    s.solve();
    EXPECT_EQ(s.info().status, Status::NumericalIssues) << static_cast<int>(mode);
    s.update(d.view());  // recovers with good data
    s.solve();
    EXPECT_EQ(s.info().status, Status::Solved) << static_cast<int>(mode);
  }
}

TEST(Solver, IterationCapGivesMaxIterReached) {
  HostQp q = random_qp(20, 4, 15, 21);
  DeviceQp d(q);
  SparseSolver s = make_solver(q, d, GraphMode::IterationGraphHostLoop);
  s.settings().max_iter = 2;
  s.solve();
  Info info = s.info();
  EXPECT_EQ(info.status, Status::MaxIterReached);
  EXPECT_EQ(info.iter_total, 2);
  s.settings().max_iter = 250;  // runtime setting: next solve converges
  s.solve();
  EXPECT_EQ(s.info().status, Status::Solved);
}

TEST(Solver, StructuralSettingChangeAfterSetupThrows) {
  HostQp q = random_qp(8, 2, 6, 2);
  DeviceQp d(q);
  SparseSolver s = make_solver(q, d, GraphMode::None);
  s.settings().preconditioner_iter = 3;
  EXPECT_THROW(s.solve(), Error);
  s.settings().preconditioner_iter = 10;
  EXPECT_NO_THROW(s.solve());
}

TEST(Solver, DeterministicCudssModeRejected) {
  HostQp q = random_qp(8, 2, 6, 2);
  DeviceQp d(q);
  SparseSolver s;
  s.settings().use_deterministic_mode_for_cudss = true;
  EXPECT_THROW(s.setup(q.P, d.view(), &q.A, &q.G), Error);
}
