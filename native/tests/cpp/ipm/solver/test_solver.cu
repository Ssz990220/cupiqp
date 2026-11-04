// ipm/solver integration tests through the public C++ API (no Python).
#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
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

SparseSolver make_solver(const HostQp& q, const DeviceQp& d, GraphMode mode,
                         KktBackend backend = KktBackend::Cudss, bool custom_spmv = false) {
  SparseSolver s;
  s.settings().kkt_backend = backend;
  s.settings().custom_spmv = custom_spmv;
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

TEST(Solver, StagedUpdateMatchesPlainUpdate) {
  HostQp q = random_qp(25, 5, 20, 41);
  DeviceQp d(q);
  // A second problem with the same patterns.
  HostQp q2 = q;
  std::mt19937 gen(8);
  std::uniform_real_distribution<double> U(-0.3, 0.3);
  for (auto& v : q2.c) v += U(gen);
  for (auto& v : q2.Gv) v = v == 0.0 ? 0.0 : v * (1 + U(gen));
  for (auto& v : q2.Pv) v *= 1.05;
  DeviceQp d2(q2);

  SparseSolver plain = make_solver(q, d, GraphMode::IterationGraphHostLoop);
  plain.update(d2.view());
  plain.solve();
  Info ip = plain.info();
  ASSERT_EQ(ip.status, Status::Solved);
  auto xp = host_x(plain);

  for (bool from_host : {false, true}) {
    SparseSolver st = make_solver(q, d, GraphMode::IterationGraphHostLoop);
    InputBuffers ib = from_host ? st.host_input_buffers() : st.input_buffers();
    ASSERT_EQ(ib.len_P, q.P.nnz());
    ASSERT_EQ(ib.len_G, q.G.nnz());
    ASSERT_EQ(ib.len_x_u, q.P.rows);
    auto put = [&](double* dst, const std::vector<double>& h) {
      CUPIQP_CUDA_CHECK(cudaMemcpy(dst, h.data(), sizeof(double) * h.size(), cudaMemcpyDefault));
    };
    put(ib.P, q2.Pv); put(ib.c, q2.c); put(ib.A, q2.Av); put(ib.b, q2.b); put(ib.G, q2.Gv);
    put(ib.h_l, q2.h_l); put(ib.h_u, q2.h_u); put(ib.x_l, q2.x_l); put(ib.x_u, q2.x_u);
    for (int rep = 0; rep < 3; ++rep) {
      const Counters c0 = st.counters();
      st.update_staged(UpdateBlocks::all(), from_host);
      st.solve();
      Info is = st.info();
      ASSERT_EQ(is.status, Status::Solved) << "from_host " << from_host;
      EXPECT_NEAR(is.iter, ip.iter, 1);
      EXPECT_LT(max_abs_diff(host_x(st), xp), 1e-8) << "from_host " << from_host;
      const Counters c1 = st.counters();
      if (rep > 0) {  // update graph, prelude/finalize and iteration graphs are reused
        EXPECT_EQ(c1.graph_captures, c0.graph_captures);
        EXPECT_EQ(c1.cudss_device_allocs, c0.cudss_device_allocs);
        EXPECT_LE(c1.host_syncs - c0.host_syncs, is.iter + 2);  // loop checks (+ info())
      }
    }
  }
}

TEST(Solver, StagedPartialUpdateOnlyReplacesNamedBlocks) {
  HostQp q = random_qp(20, 4, 15, 52);
  DeviceQp d(q);
  HostQp q2 = q;
  for (auto& v : q2.c) v = -v;
  DeviceQp d2(q2);
  SparseSolver plain = make_solver(q, d, GraphMode::None);
  ValuesView only_c;
  only_c.c = d2.c.data();
  plain.update(only_c);
  plain.solve();
  ASSERT_EQ(plain.info().status, Status::Solved);
  auto xp = host_x(plain);

  SparseSolver st = make_solver(q, d, GraphMode::IterationGraphHostLoop);
  InputBuffers ib = st.input_buffers();
  CUPIQP_CUDA_CHECK(cudaMemcpy(ib.c, q2.c.data(), sizeof(double) * q2.c.size(), cudaMemcpyHostToDevice));
  UpdateBlocks blocks;
  blocks.c = true;
  st.update_staged(blocks);
  st.solve();
  ASSERT_EQ(st.info().status, Status::Solved);
  EXPECT_LT(max_abs_diff(host_x(st), xp), 1e-8);
}

// ------------------------------------------------------------------ reduced KKT backend

namespace {
constexpr GraphMode kAllModes[] = {GraphMode::None, GraphMode::IterationGraphHostLoop,
                                   GraphMode::SolveGraphDeviceLoop};

bool all_finite(const std::vector<double>& v) {
  for (double x : v)
    if (!std::isfinite(x)) return false;
  return true;
}
}  // namespace

TEST(SolverReduced, AnalyticEqualityQp) {
  CsrPattern P{2, 2, {0, 1, 2}, {0, 1}}, A{1, 2, {0, 2}, {0, 1}};
  std::vector<double> Pv{1, 1}, Av{1, 1}, c{0, 0}, b{2};
  CudaBuffer<double> dP(2), dA(2), dc(2), db(1);
  dP.copy_from_host(Pv.data(), 2, nullptr); dA.copy_from_host(Av.data(), 2, nullptr);
  dc.copy_from_host(c.data(), 2, nullptr); db.copy_from_host(b.data(), 1, nullptr);
  CUPIQP_CUDA_CHECK(cudaDeviceSynchronize());
  for (GraphMode mode : kAllModes) {
    SparseSolver s;
    s.settings().kkt_backend = KktBackend::Reduced;
    s.settings().graph_mode = mode;
    ValuesView v; v.P = dP.data(); v.A = dA.data(); v.c = dc.data(); v.b = db.data();
    s.setup(P, v, &A, nullptr);
    s.solve();
    Info info = s.info();
    EXPECT_EQ(info.status, Status::Solved) << static_cast<int>(mode);
    auto x = host_x(s);
    EXPECT_NEAR(x[0], 1.0, 1e-7);
    EXPECT_NEAR(x[1], 1.0, 1e-7);
    EXPECT_EQ(s.counters().cudss_analysis_calls, 0);
    EXPECT_EQ(s.counters().cudss_factor_calls, 0);
  }
}

// Same solution as cuDSS; with the custom SpMV every step is deterministic,
// so all three graph modes and repeated solves agree bit for bit.
TEST(SolverReduced, MatchesCudssAndGraphModesAgreeBitwise) {
  for (unsigned seed : {11u, 12u}) {
    HostQp q = random_qp(40, 6, 70, seed);
    DeviceQp d(q);
    SparseSolver ref = make_solver(q, d, GraphMode::IterationGraphHostLoop);
    ref.solve();
    ASSERT_EQ(ref.info().status, Status::Solved);
    const auto x_cudss = host_x(ref);
    const int iter_cudss = ref.info().iter;
    std::vector<double> x_ref;
    int iter_ref = -1;
    for (GraphMode mode : kAllModes) {
      SparseSolver s = make_solver(q, d, mode, KktBackend::Reduced, /*custom_spmv=*/true);
      for (int rep = 0; rep < 3; ++rep) {
        s.solve();
        Info info = s.info();
        ASSERT_EQ(info.status, Status::Solved) << "mode " << static_cast<int>(mode) << " rep " << rep;
        auto x = host_x(s);
        EXPECT_LT(primal_violation(q, x), 1e-6);
        EXPECT_LT(max_abs_diff(x, x_cudss), 1e-6);
        EXPECT_NEAR(info.iter, iter_cudss, 2);
        if (x_ref.empty()) {
          x_ref = x;
          iter_ref = info.iter;
        } else {
          EXPECT_EQ(x, x_ref) << "mode " << static_cast<int>(mode) << " rep " << rep;
          EXPECT_EQ(info.iter, iter_ref);
        }
      }
      EXPECT_EQ(s.counters().cudss_analysis_calls, 0);
      EXPECT_EQ(s.counters().cudss_device_allocs, 0);
    }
  }
}

TEST(SolverReduced, EqualityOnlyAndBoxOnlyQps) {
  // Box-only: no A, no G (H = P + x_reg).
  HostQp q = random_qp(35, 0, 0, 13);
  DeviceQp d(q);
  for (GraphMode mode : kAllModes) {
    SparseSolver r;
    r.settings().kkt_backend = KktBackend::Reduced;
    r.settings().graph_mode = mode;
    r.setup(q.P, d.view(), nullptr, nullptr);
    r.solve();
    ASSERT_EQ(r.info().status, Status::Solved) << static_cast<int>(mode);
    SparseSolver c;
    c.setup(q.P, d.view(), nullptr, nullptr);
    c.solve();
    ASSERT_EQ(c.info().status, Status::Solved);
    EXPECT_LT(max_abs_diff(host_x(r), host_x(c)), 1e-6);
    EXPECT_LT(primal_violation(q, host_x(r)), 1e-6);
  }
  // Equality-only (no G, no bounds).
  HostQp e = random_qp(20, 5, 0, 14);
  e.x_l.assign(20, -INFINITY);
  e.x_u.assign(20, INFINITY);
  DeviceQp de(e);
  ValuesView v = de.view();
  v.x_l = v.x_u = nullptr;
  for (GraphMode mode : kAllModes) {
    SparseSolver r;
    r.settings().kkt_backend = KktBackend::Reduced;
    r.settings().graph_mode = mode;
    r.setup(e.P, v, &e.A, nullptr);
    r.solve();
    ASSERT_EQ(r.info().status, Status::Solved) << static_cast<int>(mode);
    EXPECT_LT(primal_violation(e, host_x(r)), 1e-6);
  }
}

TEST(SolverReduced, RepeatedUpdatesAllocateNothing) {
  HostQp q = random_qp(25, 5, 20, 3);
  DeviceQp d(q);
  for (GraphMode mode : kAllModes) {
    SparseSolver s = make_solver(q, d, mode, KktBackend::Reduced);
    s.solve();
    ASSERT_EQ(s.info().status, Status::Solved);
    const Counters c0 = s.counters();
    size_t free0 = 0, total = 0, free1 = 0;
    CUPIQP_CUDA_CHECK(cudaMemGetInfo(&free0, &total));
    std::mt19937 gen(5);
    std::uniform_real_distribution<double> U(-0.5, 0.5);
    for (int k = 0; k < 4; ++k) {
      HostQp q2 = q;
      for (auto& v : q2.c) v += U(gen);
      d.upload(q2);
      s.update(d.view());
      s.solve();
      EXPECT_EQ(s.info().status, Status::Solved) << "update " << k;
    }
    d.upload(q);
    CUPIQP_CUDA_CHECK(cudaMemGetInfo(&free1, &total));
    EXPECT_EQ(free0, free1) << static_cast<int>(mode);
    EXPECT_EQ(s.counters().graph_captures, c0.graph_captures) << static_cast<int>(mode);
  }
}

TEST(SolverReduced, RejectsLargeN) {
  const int n = 2049;
  HostQp q;
  q.P.rows = q.P.cols = n;
  q.P.indptr.resize(n + 1);
  for (int i = 0; i <= n; ++i) q.P.indptr[i] = i;
  for (int i = 0; i < n; ++i) q.P.indices.push_back(i);
  std::vector<double> ones(n, 1.0);
  CudaBuffer<double> dP(n), dc(n);
  dP.copy_from_host(ones.data(), n, nullptr);
  dc.copy_from_host(ones.data(), n, nullptr);
  CUPIQP_CUDA_CHECK(cudaDeviceSynchronize());
  ValuesView v;
  v.P = dP.data();
  v.c = dc.data();
  SparseSolver s;
  s.settings().kkt_backend = KktBackend::Reduced;
  try {
    s.setup(q.P, v, nullptr, nullptr);
    FAIL() << "expected Unsupported";
  } catch (const Error& e) {
    EXPECT_EQ(e.code(), ErrorCode::Unsupported);
  }
  SparseSolver ok;  // cuDSS takes it
  ok.setup(q.P, v, nullptr, nullptr);
  ok.solve();
  EXPECT_EQ(ok.info().status, Status::Solved);
}

TEST(SolverReduced, BackendIsFrozenAfterSetup) {
  HostQp q = random_qp(8, 2, 6, 2);
  DeviceQp d(q);
  SparseSolver s = make_solver(q, d, GraphMode::None, KktBackend::Reduced);
  s.settings().kkt_backend = KktBackend::Cudss;
  EXPECT_THROW(s.solve(), Error);
  s.settings().kkt_backend = KktBackend::Reduced;
  EXPECT_NO_THROW(s.solve());
}

// Forced Cholesky failures: P = -10 I (not convex) with box bounds and an
// equality block of rank < n, so H = P + x_reg + A^T A / delta is indefinite
// until rho has grown past the curvature. The factor-retry kernel raises
// rho / delta (x100 per retry). With the default max_factor_retires the
// factorization recovers and the IPM runs on to a final status (measured:
// PRIMAL_INFEASIBLE after 13 iterations and 5 retries; the problem is not
// convex); with max_factor_retires = 2 the solve must end NUMERICAL_ISSUES.
// Neither may produce a non-finite result.
TEST(SolverReduced, FactorFailureRetriesThenRecoversOrReportsNumericalIssues) {
  const int n = 12, p = 3;
  HostQp base = random_qp(n, p, 0, 71);
  for (GraphMode mode : kAllModes) {
    for (int which : {0, 1}) {
      HostQp q = base;
      const double alpha = 10.0;
      q.P = CsrPattern{n, n, {}, {}};
      q.P.indptr.push_back(0);
      for (int i = 0; i < n; ++i) {
        q.P.indices.push_back(i);
        q.P.indptr.push_back(i + 1);
      }
      q.Pv.assign(n, -alpha);
      for (int j = 0; j < n; ++j) {  // keep the problem bounded
        q.x_l[j] = -1.0;
        q.x_u[j] = 1.0;
      }
      DeviceQp d(q);
      SparseSolver s;
      s.settings().kkt_backend = KktBackend::Reduced;
      s.settings().graph_mode = mode;
      s.settings().max_iter = 50;
      if (which == 1) s.settings().max_factor_retires = 2;
      s.setup(q.P, d.view(), &q.A, nullptr);
      s.solve();
      Info info = s.info();
      SCOPED_TRACE(::testing::Message() << "mode " << static_cast<int>(mode) << " case " << which << " status "
                                        << static_cast<int>(info.status) << " retries " << info.factor_retries
                                        << " iter " << info.iter << " rho " << info.rho);
      std::printf("[ info ] mode %d case %d: status %s, factor_retries %d, iter %d, rho %.3g\n",
                  static_cast<int>(mode), which, status_name(info.status), info.factor_retries, info.iter, info.rho);
      EXPECT_GT(info.factor_retries, 0);
      EXPECT_NE(info.status, Status::Unsolved);
      if (which == 1) {
        EXPECT_EQ(info.status, Status::NumericalIssues);
        EXPECT_EQ(info.factor_retries, 2);
      } else {
        EXPECT_NE(info.status, Status::NumericalIssues);
        EXPECT_GT(info.iter, 0);
      }
      EXPECT_TRUE(all_finite(host_x(s)));
      EXPECT_TRUE(std::isfinite(info.rho));
      EXPECT_TRUE(std::isfinite(info.delta));
    }
  }
}
