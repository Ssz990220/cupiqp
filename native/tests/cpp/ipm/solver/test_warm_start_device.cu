// Warm start from the previous solve, built on the device
// (SparseSolver::set_warm_start / solve_warm_start) through the public C++ API.
#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <string>
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


const GraphMode kModes[] = {GraphMode::None, GraphMode::IterationGraphHostLoop, GraphMode::SolveGraphDeviceLoop};

// Perturb the cost and loosen some finite upper bounds (the next QP of a sequence).
void perturb(Qp& q, unsigned seed, double eps = 1e-2) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<double> U(-eps, eps);
  for (auto& v : q.c) v += U(gen);
  for (auto& v : q.h_u) if (std::isfinite(v)) v += std::fabs(U(gen));
  q.upload();
}

std::vector<int32_t> to_host_i(const int32_t* d, int n, cudaStream_t s) {
  std::vector<int32_t> h(n);
  if (n) CUPIQP_CUDA_CHECK(cudaMemcpyAsync(h.data(), d, sizeof(int32_t) * n, cudaMemcpyDeviceToHost, s));
  CUPIQP_CUDA_CHECK(cudaStreamSynchronize(s));
  return h;
}

const void* debug_ptr(const SparseSolver& s, const std::string& name) {
  for (const auto& b : s.debug_buffers())
    if (b.name == name) return b.ptr;
  return nullptr;
}

}  // namespace

// The first solve_warm_start() is the default cold solve; the default solve()
// stays bitwise identical after any number of warm solves.
TEST(WarmStartDevice, FirstSolveIsColdAndDefaultPathUnchanged) {
  for (GraphMode mode : kModes) {
    Qp q(30, 5, 40, 9);
    SparseSolver ref;
    ref.settings().graph_mode = mode;
    ref.setup(q.P, q.view(), &q.A, &q.G);
    ref.solve();
    const Info ri = ref.info();
    const auto x_ref = to_host(ref.result().x, q.n, ref.stream());

    SparseSolver s;
    s.settings().graph_mode = mode;
    s.setup(q.P, q.view(), &q.A, &q.G);
    WarmStartOptions o;
    o.snapshot_slots = 4;
    s.set_warm_start(o);
    s.solve_warm_start();
    EXPECT_EQ(s.info().iter, ri.iter) << int(mode);
    EXPECT_EQ(max_abs_diff(to_host(s.result().x, q.n, s.stream()), x_ref), 0.0) << int(mode);
    s.solve_warm_start();
    s.solve_warm_start();
    s.solve();
    EXPECT_EQ(s.info().iter, ri.iter) << int(mode);
    EXPECT_EQ(max_abs_diff(to_host(s.result().x, q.n, s.stream()), x_ref), 0.0) << int(mode);
  }
}

// The QP in the increment d = x - x0 (the SQP form the warm start is built
// for): c += P x0, b -= A x0, bounds -= G x0 / x0. Its solution is x* - x0
// with the same duals.
void shift(Qp& q, const std::vector<double>& x0) {
  const int n = q.n;
  std::vector<double> c = q.c;
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) c[i] += q.Pv[i * n + j] * x0[j];
  q.c = c;
  for (int r = 0; r < q.p; ++r)
    for (int j = 0; j < n; ++j) q.b[r] -= q.Av[r * n + j] * x0[j];
  for (int r = 0; r < q.m; ++r) {
    double g = 0;
    for (int j = 0; j < n; ++j) g += q.Gv[r * n + j] * x0[j];
    if (std::isfinite(q.h_l[r])) q.h_l[r] -= g;
    if (std::isfinite(q.h_u[r])) q.h_u[r] -= g;
  }
  for (int j = 0; j < n; ++j) {
    if (std::isfinite(q.x_l[j])) q.x_l[j] -= x0[j];
    if (std::isfinite(q.x_u[j])) q.x_u[j] -= x0[j];
  }
  q.upload();
}

// Warm start of the increment QP at the previous solution: x = 0 is optimal,
// the slacks from the new bounds are the optimal slacks, so with tiny clamps
// it needs at most a few iterations. A perturbed increment QP converges in
// fewer iterations than cold, to the same solution.
TEST(WarmStartDevice, WarmStartFromPreviousSolve) {
  for (GraphMode mode : kModes) {
    Qp q(40, 6, 60, 13);
    SparseSolver s;
    s.settings().graph_mode = mode;
    s.setup(q.P, q.view(), &q.A, &q.G);
    WarmStartOptions o;
    o.s_min = 1e-9;
    o.z_min = 1e-9;
    s.set_warm_start(o);
    s.solve_warm_start();
    ASSERT_EQ(s.info().status, Status::Solved);
    shift(q, to_host(s.result().x, q.n, s.stream()));
    s.update(q.view());
    s.solve_warm_start();
    EXPECT_EQ(s.info().status, Status::Solved);
    EXPECT_LE(s.info().iter, 3) << int(mode);
    double xmax = 0;
    for (double v : to_host(s.result().x, q.n, s.stream())) xmax = std::fmax(xmax, std::fabs(v));
    EXPECT_LT(xmax, 1e-6) << int(mode);

    // Default clamps, perturbed increment QP: warm beats cold.
    s.set_warm_start(WarmStartOptions{});
    s.solve_warm_start();  // cold, records the state
    shift(q, to_host(s.result().x, q.n, s.stream()));
    perturb(q, 5);
    s.update(q.view());
    s.solve();
    const Info cold = s.info();
    const auto x_cold = to_host(s.result().x, q.n, s.stream());
    // solve() made the next solve_warm_start() cold: record a state again on
    // the unshifted QP (same duals as the shifted ones), then warm start.
    Qp prev(40, 6, 60, 13);
    s.update(prev.view());
    s.solve_warm_start();
    s.update(q.view());
    s.solve_warm_start();
    const Info warm = s.info();
    EXPECT_EQ(warm.status, Status::Solved);
    EXPECT_LT(warm.iter, cold.iter) << int(mode);
    EXPECT_LT(max_abs_diff(to_host(s.result().x, q.n, s.stream()), x_cold), 1e-5);
  }
}

// Previous status not Solved -> the next start is cold (decided on the device):
// bitwise identical to the default solve.
TEST(WarmStartDevice, ColdFallbackWhenPreviousNotSolved) {
  for (GraphMode mode : kModes) {
    Qp q(30, 5, 40, 17);
    SparseSolver ref;
    ref.settings().graph_mode = mode;
    ref.setup(q.P, q.view(), &q.A, &q.G);
    ref.solve();
    const auto x_ref = to_host(ref.result().x, q.n, ref.stream());

    SparseSolver s;
    s.settings().graph_mode = mode;
    s.setup(q.P, q.view(), &q.A, &q.G);
    s.set_warm_start(WarmStartOptions{});
    s.solve_warm_start();
    s.settings().max_iter = 2;
    s.solve_warm_start();
    ASSERT_EQ(s.info().status, Status::MaxIterReached);
    s.settings().max_iter = 250;
    s.solve_warm_start();  // previous not Solved: cold
    EXPECT_EQ(s.info().iter, ref.info().iter) << int(mode);
    EXPECT_EQ(max_abs_diff(to_host(s.result().x, q.n, s.stream()), x_ref), 0.0) << int(mode);
    s.solve_warm_start();  // previous Solved: warm
    EXPECT_LT(s.info().iter, ref.info().iter) << int(mode);
  }
}

// Keys recover a row permutation of G: the match equals the permutation and
// the warm solve equals the warm solve of the unpermuted QP (up to rounding).
TEST(WarmStartDevice, KeyMatchingFollowsRowPermutation) {
  const int n = 30, p = 5, m = 48;
  for (bool groups : {false, true}) {
    for (bool host : {true, false}) {
      Qp q(n, p, m, 23);
      SparseSolver s;
      s.setup(q.P, q.view(), &q.A, &q.G);
      WarmStartOptions o;
      o.row_match = RowMatch::Keys;
      o.row_groups = groups;
      o.key_row_begin = 8;
      o.key_row_count = 32;  // rows 8..39 keyed, groups of 8
      s.set_warm_start(o);
      RowKeyBuffers kb = s.row_key_buffers(host);
      ASSERT_EQ(kb.count, 32);
      ASSERT_EQ(kb.groups != nullptr, groups);
      auto write_keys = [&](const std::vector<int64_t>& keys, const std::vector<int32_t>& grp) {
        if (host) {
          for (int j = 0; j < 32; ++j) kb.keys[j] = keys[j];
          if (groups) for (int j = 0; j < 32; ++j) kb.groups[j] = grp[j];
        } else {
          CUPIQP_CUDA_CHECK(cudaMemcpyAsync(kb.keys, keys.data(), 8 * 32, cudaMemcpyHostToDevice, s.stream()));
          if (groups)
            CUPIQP_CUDA_CHECK(cudaMemcpyAsync(kb.groups, grp.data(), 4 * 32, cudaMemcpyHostToDevice, s.stream()));
          CUPIQP_CUDA_CHECK(cudaStreamSynchronize(s.stream()));
        }
      };
      // QP 0: key of keyed row j = 100 + j % 8 within group j / 8 (groups),
      // or 100 + j (no groups: keys must be unique).
      std::vector<int64_t> k0(32), k1(32);
      std::vector<int32_t> g(32);
      for (int j = 0; j < 32; ++j) {
        g[j] = j / 8;
        k0[j] = groups ? 100 + j % 8 : 100 + j;
      }
      write_keys(k0, g);
      s.solve_warm_start(host);
      ASSERT_EQ(s.info().status, Status::Solved);
      // QP 1: reverse the rows inside each group of 8; one row gets a new key.
      Qp q1(n, p, m, 23);
      std::vector<int> perm(32);
      for (int j = 0; j < 32; ++j) perm[j] = (j / 8) * 8 + 7 - j % 8;  // new keyed row j = old row perm[j]
      for (int j = 0; j < 32; ++j) {
        const int rn = 8 + j, ro = 8 + perm[j];
        for (int c = 0; c < n; ++c) q1.Gv[rn * n + c] = q.Gv[ro * n + c];
        q1.h_l[rn] = q.h_l[ro];
        q1.h_u[rn] = q.h_u[ro];
        k1[j] = k0[perm[j]];
      }
      k1[5] = 999;
      q1.upload();
      s.update(q1.view());
      write_keys(k1, g);
      s.solve_warm_start(host);
      EXPECT_EQ(s.info().status, Status::Solved);
      const auto match = to_host_i(static_cast<const int32_t*>(debug_ptr(s, "warm.match")), 32, s.stream());
      for (int j = 0; j < 32; ++j) EXPECT_EQ(match[j], j == 5 ? -1 : perm[j]) << j;
    }
  }
}

// Steady state: one graph launch per solve, no host sync, no capture, no
// allocation (host keys and the snapshot ring buffer enabled).
TEST(WarmStartDevice, OneGraphLaunchNoSyncNoAllocation) {
  Qp q(30, 5, 40, 21);
  SparseSolver s;
  s.settings().graph_mode = GraphMode::SolveGraphDeviceLoop;
  s.setup(q.P, q.view(), &q.A, &q.G);
  WarmStartOptions o;
  o.row_match = RowMatch::Keys;
  o.row_groups = true;
  o.snapshot_slots = 4;
  o.snapshot_back = 2;
  s.set_warm_start(o);
  RowKeyBuffers kb = s.row_key_buffers(true);
  for (int j = 0; j < kb.count; ++j) { kb.keys[j] = j; kb.groups[j] = j / 4; }
  s.solve_warm_start(true);  // captures the cold-record graph
  s.solve_warm_start(true);  // captures the device-decided graph
  CUPIQP_CUDA_CHECK(cudaStreamSynchronize(s.stream()));
  size_t free0 = 0, total = 0;
  CUPIQP_CUDA_CHECK(cudaMemGetInfo(&free0, &total));
  const Counters c0 = s.counters();
  for (int k = 0; k < 5; ++k) s.solve_warm_start(true);
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

// The ring buffer does not change the solve; snapshot_back = 0 records the
// same start as no ring buffer. All graph modes agree bitwise.
TEST(WarmStartDevice, SnapshotRingAndGraphModesAgree) {
  std::vector<std::vector<double>> xs;
  std::vector<int> its;
  for (GraphMode mode : kModes) {
    for (int K : {0, 4}) {
      for (int back : {0, 3}) {
        if (back > K) continue;
        Qp q(30, 5, 40, 31);
        SparseSolver s;
        s.settings().graph_mode = mode;
        s.setup(q.P, q.view(), &q.A, &q.G);
        WarmStartOptions o;
        o.snapshot_slots = K;
        o.snapshot_back = back;
        s.set_warm_start(o);
        s.solve_warm_start();
        perturb(q, 3);
        s.update(q.view());
        s.solve_warm_start();
        ASSERT_EQ(s.info().status, Status::Solved);
        xs.push_back(to_host(s.result().x, q.n, s.stream()));
        its.push_back(s.info().iter);
      }
    }
  }
  // Order per mode: (K0,b0), (K4,b0), (K4,b3).
  for (size_t i = 0; i < xs.size(); ++i) {
    const size_t ref = (i % 3 == 2) ? 2 : 0;  // back 3 compared with back 3 of the first mode
    EXPECT_EQ(its[i], its[ref]) << i;
    EXPECT_EQ(max_abs_diff(xs[i], xs[ref]), 0.0) << i;
  }
}

TEST(WarmStartDevice, RejectsBadCalls) {
  Qp q(10, 2, 10, 3);
  SparseSolver s;
  s.setup(q.P, q.view(), &q.A, &q.G);
  EXPECT_THROW(s.solve_warm_start(), Error);
  EXPECT_THROW(s.row_key_buffers(true), Error);
  WarmStartOptions o;
  o.s_min = 0.0;
  EXPECT_THROW(s.set_warm_start(o), Error);
  o = WarmStartOptions{};
  o.snapshot_back = 2;
  o.snapshot_slots = 1;
  EXPECT_THROW(s.set_warm_start(o), Error);
  o = WarmStartOptions{};
  o.row_match = RowMatch::Keys;
  o.key_row_begin = 5;
  o.key_row_count = 6;
  EXPECT_THROW(s.set_warm_start(o), Error);
}
