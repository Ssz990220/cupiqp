// ReducedKktSolver against a dense CPU reference (long double).
//
// A random KKT matrix in KktAssembly's layout is built on the host from
// build_kkt_host_maps: x rows P + diag(x_reg) (full P stored), y rows A with
// diagonal -delta, z rows G with diagonal -z_reg, or, for inactive rows, a
// zeroed G row with diagonal -1 (as KktAssembly writes them). Checked:
//   * the assembled lower triangle of H = P + x_reg + M^T W M;
//   * factor + solve against the CPU solution of K x = b;
//   * the factor_status on success, on a non-positive pivot (first failing
//     column + 1, as the CPU Cholesky finds it) and on a NaN value;
//   * bitwise determinism, and bitwise agreement of a plain graph replay
//     and of a replay inside a conditional (IF / WHILE) body.
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#include "common/buffer.hpp"
#include "common/graph.hpp"
#include "common/handles.hpp"
#include "linalg/kkt_assembly/kkt_pattern.hpp"
#include "linalg/reduced_kkt/reduced_kkt_solver.hpp"

using namespace cupiqp;

namespace {

using LD = long double;

template <class T>
std::vector<T> down(const T* d, size_t k) {
  std::vector<T> h(k);
  if (k) CUPIQP_CUDA_CHECK(cudaMemcpy(h.data(), d, k * sizeof(T), cudaMemcpyDeviceToHost));
  return h;
}

CsrPattern dense_to_pattern(const std::vector<char>& nz, int rows, int cols) {
  CsrPattern c;
  c.rows = rows;
  c.cols = cols;
  if (rows == 0) return c;
  c.indptr.push_back(0);
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j)
      if (nz[size_t(i) * cols + j]) c.indices.push_back(j);
    c.indptr.push_back(int32_t(c.indices.size()));
  }
  return c;
}

struct Case {
  int n, p, m;
  double density = 0.1;
  double inactive_frac = 0.2;
  unsigned seed = 1;
};

struct Problem {
  int n = 0, p = 0, m = 0, dim = 0;
  KktHostMaps maps;
  std::vector<double> K;               // KKT values (nnz)
  std::vector<LD> Pd, Md, wv, xreg;    // dense P (n x n), M (mm x n), w, x_reg
  std::vector<int> inactive;           // z rows

  explicit Problem(const Case& c) : n(c.n), p(c.p), m(c.m), dim(c.n + c.p + c.m) {
    std::mt19937 g(c.seed);
    std::uniform_real_distribution<double> u(-1.0, 1.0), pr(0.0, 1.0), pos(0.1, 2.0);
    const int mm = p + m;
    // P = R^T R (symmetric PSD), pattern of the product plus an explicit zero.
    std::vector<double> R(size_t(n) * n, 0.0);
    for (auto& v : R) v = pr(g) < c.density ? u(g) : 0.0;
    std::vector<double> P(size_t(n) * n, 0.0);
    std::vector<char> Pnz(size_t(n) * n, 0);
    for (int i = 0; i < n; ++i)
      for (int k = 0; k < n; ++k) {
        const double rki = R[size_t(k) * n + i];
        if (rki == 0.0) continue;
        for (int j = 0; j < n; ++j) {
          const double rkj = R[size_t(k) * n + j];
          if (rkj == 0.0) continue;
          P[size_t(i) * n + j] += rki * rkj;
          Pnz[size_t(i) * n + j] = 1;
        }
      }
    if (n > 1) Pnz[1] = Pnz[n] = 1;  // explicit zero allowed
    std::vector<double> M(size_t(mm) * n, 0.0);
    std::vector<char> Mnz(size_t(mm) * n, 0);
    for (int k = 0; k < mm; ++k) {
      for (int j = 0; j < n; ++j)
        if (pr(g) < c.density) {
          M[size_t(k) * n + j] = u(g);
          Mnz[size_t(k) * n + j] = 1;
        }
      const int j = int(pr(g) * n) % n;
      if (!Mnz[size_t(k) * n + j]) {
        M[size_t(k) * n + j] = u(g);
        Mnz[size_t(k) * n + j] = 1;
      }
    }
    std::vector<char> Anz(Mnz.begin(), Mnz.begin() + size_t(p) * n), Gnz(Mnz.begin() + size_t(p) * n, Mnz.end());
    maps = build_kkt_host_maps(dense_to_pattern(Pnz, n, n), dense_to_pattern(Anz, p, n),
                               dense_to_pattern(Gnz, m, n));
    const CsrPattern& kp = maps.kkt;
    K.assign(kp.nnz(), 0.0);
    xreg.resize(n);
    const double delta = 1e-4;
    inactive.assign(m, 0);
    for (int k = 0; k < m; ++k) inactive[k] = pr(g) < c.inactive_frac;
    std::vector<double> zreg(m);
    for (auto& v : zreg) v = pos(g) * 1e-2;
    for (int i = 0; i < n; ++i) xreg[i] = 1e-6 + 0.5 * pr(g);
    for (int r = 0; r < dim; ++r)
      for (int e = kp.indptr[r]; e < kp.indptr[r + 1]; ++e) {
        const int col = kp.indices[e];
        double v = 0;
        if (r < n) v = P[size_t(r) * n + col] + (col == r ? xreg[r] : 0.0);
        else if (r < n + p) v = col == r ? -delta : M[size_t(r - n) * n + col];
        else {
          const int k = r - n - p;
          if (col == r) v = inactive[k] ? -1.0 : -zreg[k];
          else v = inactive[k] ? 0.0 : M[size_t(p + k) * n + col];
        }
        K[e] = v;
      }
    Pd.assign(P.begin(), P.end());
    Md.resize(size_t(mm) * n);
    for (int k = 0; k < mm; ++k)
      for (int j = 0; j < n; ++j) {
        const bool zero_row = k >= p && inactive[k - p];
        Md[size_t(k) * n + j] = zero_row ? 0.0L : LD(M[size_t(k) * n + j]);
      }
    wv.resize(mm);
    for (int k = 0; k < mm; ++k) wv[k] = k < p ? 1.0L / delta : (inactive[k - p] ? 1.0L : 1.0L / LD(zreg[k - p]));
  }

  // Dense H (row-major, full) from the current K values, in long double.
  std::vector<LD> H() const {
    const int mm = p + m;
    std::vector<LD> h(size_t(n) * n, 0.0L);
    const CsrPattern& kp = maps.kkt;
    for (int r = 0; r < n; ++r)
      for (int e = kp.indptr[r]; e < kp.indptr[r + 1]; ++e) h[size_t(r) * n + kp.indices[e]] += K[e];
    std::vector<LD> Mk(size_t(mm) * n, 0.0L), w(mm);
    for (int k = 0; k < mm; ++k) {
      for (int e = kp.indptr[n + k]; e < kp.indptr[n + k + 1]; ++e) {
        if (kp.indices[e] == n + k) w[k] = -1.0L / K[e];
        else Mk[size_t(k) * n + kp.indices[e]] = K[e];
      }
    }
    for (int k = 0; k < mm; ++k)
      for (int i = 0; i < n; ++i) {
        const LD a = w[k] * Mk[size_t(k) * n + i];
        if (a == 0) continue;
        for (int j = 0; j < n; ++j) h[size_t(i) * n + j] += a * Mk[size_t(k) * n + j];
      }
    return h;
  }

  // In-place Cholesky of a full row-major matrix; returns the first failing
  // column + 1 (0 = success).
  static int cholesky(std::vector<LD>& a, int n) {
    for (int j = 0; j < n; ++j) {
      LD d = a[size_t(j) * n + j];
      for (int k = 0; k < j; ++k) d -= a[size_t(j) * n + k] * a[size_t(j) * n + k];
      if (!(d > 0)) return j + 1;
      d = std::sqrt(d);
      a[size_t(j) * n + j] = d;
      for (int i = j + 1; i < n; ++i) {
        LD s = a[size_t(i) * n + j];
        for (int k = 0; k < j; ++k) s -= a[size_t(i) * n + k] * a[size_t(j) * n + k];
        a[size_t(i) * n + j] = s / d;
      }
    }
    return 0;
  }

  // CPU solution of K x = b through the same elimination, in long double.
  std::vector<LD> solve(const std::vector<double>& b) const {
    const int mm = p + m;
    std::vector<LD> L = H();
    EXPECT_EQ(cholesky(L, n), 0);
    const CsrPattern& kp = maps.kkt;
    std::vector<LD> Mk(size_t(mm) * n, 0.0L), w(mm);
    for (int k = 0; k < mm; ++k)
      for (int e = kp.indptr[n + k]; e < kp.indptr[n + k + 1]; ++e) {
        if (kp.indices[e] == n + k) w[k] = -1.0L / K[e];
        else Mk[size_t(k) * n + kp.indices[e]] = K[e];
      }
    std::vector<LD> x(dim, 0.0L), r(n);
    for (int i = 0; i < n; ++i) {
      LD s = b[i];
      for (int k = 0; k < mm; ++k) s += Mk[size_t(k) * n + i] * w[k] * b[n + k];
      r[i] = s;
    }
    for (int i = 0; i < n; ++i) {
      LD s = r[i];
      for (int k = 0; k < i; ++k) s -= L[size_t(i) * n + k] * r[k];
      r[i] = s / L[size_t(i) * n + i];
    }
    for (int i = n - 1; i >= 0; --i) {
      LD s = r[i];
      for (int k = i + 1; k < n; ++k) s -= L[size_t(k) * n + i] * r[k];
      r[i] = s / L[size_t(i) * n + i];
    }
    for (int i = 0; i < n; ++i) x[i] = r[i];
    for (int k = 0; k < mm; ++k) {
      LD s = 0;
      for (int j = 0; j < n; ++j) s += Mk[size_t(k) * n + j] * r[j];
      x[n + k] = w[k] * (s - b[n + k]);
    }
    return x;
  }

  // ||K x - b||_inf / (||K||_inf ||x||_inf + ||b||_inf), K symmetric (lower y/z).
  LD residual(const std::vector<double>& x, const std::vector<double>& b) const {
    const CsrPattern& kp = maps.kkt;
    std::vector<LD> y(dim, 0.0L), rs(dim, 0.0L);
    for (int r = 0; r < dim; ++r)
      for (int e = kp.indptr[r]; e < kp.indptr[r + 1]; ++e) {
        const int c = kp.indices[e];
        y[r] += LD(K[e]) * x[c];
        rs[r] += std::fabs(LD(K[e]));
        if (r >= n && c != r) {
          y[c] += LD(K[e]) * x[r];
          rs[c] += std::fabs(LD(K[e]));
        }
      }
    LD num = 0, kn = 0, xn = 0, bn = 0;
    for (int i = 0; i < dim; ++i) {
      num = std::max(num, std::fabs(y[i] - LD(b[i])));
      kn = std::max(kn, rs[i]);
      xn = std::max(xn, std::fabs(LD(x[i])));
      bn = std::max(bn, std::fabs(LD(b[i])));
    }
    return num / (kn * xn + bn);
  }
};

struct Device {
  CudaStream stream = CudaStream::create();
  CudaBuffer<int32_t> idx;
  CudaBuffer<double> K;
  ReducedKktSolver solver;
  explicit Device(const Problem& pr) {
    cudaStream_t s = stream.get();
    idx.allocate(pr.maps.kkt.indices.size());
    idx.copy_from_host(pr.maps.kkt.indices.data(), pr.maps.kkt.indices.size(), s);
    K.allocate(pr.K.size());
    K.copy_from_host(pr.K.data(), pr.K.size(), s);
    solver.init(pr.maps, idx.data(), K.data(), s);
  }
  void set_K(const std::vector<double>& v) {
    K.copy_from_host(v.data(), v.size(), stream.get());
  }
  void set_rhs(const std::vector<double>& b) {
    CUPIQP_CUDA_CHECK(cudaMemcpyAsync(solver.rhs(), b.data(), b.size() * sizeof(double), cudaMemcpyHostToDevice,
                                      stream.get()));
  }
  void sync() { CUPIQP_CUDA_CHECK(cudaStreamSynchronize(stream.get())); }
};

std::vector<double> random_rhs(int dim, unsigned seed) {
  std::mt19937 g(seed);
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  std::vector<double> b(dim);
  for (auto& v : b) v = u(g);
  return b;
}

void check_case(const Case& c) {
  SCOPED_TRACE(::testing::Message() << "n=" << c.n << " p=" << c.p << " m=" << c.m << " seed=" << c.seed);
  Problem pr(c);
  Device dv(pr);
  cudaStream_t s = dv.stream.get();
  // Assembly.
  dv.solver.assemble(s);
  dv.sync();
  const int n = pr.n;
  auto Hg = down(dv.solver.H(), size_t(n) * n);
  auto Hc = pr.H();
  LD hmax = 0, hdiff = 0;
  for (int i = 0; i < n; ++i)
    for (int j = 0; j <= i; ++j) {
      hmax = std::max(hmax, std::fabs(Hc[size_t(i) * n + j]));
      hdiff = std::max(hdiff, std::fabs(Hc[size_t(i) * n + j] - LD(Hg[size_t(j) * n + i])));
    }
  EXPECT_LE(double(hdiff / hmax), 1e-14);
  // Factor + solve.
  auto b = random_rhs(pr.dim, c.seed + 100);
  dv.set_rhs(b);
  dv.solver.factor(s);
  dv.solver.solve(s);
  dv.sync();
  EXPECT_EQ(down(dv.solver.factor_status(), 1)[0], 0);
  auto x = down(dv.solver.sol(), pr.dim);
  auto xr = pr.solve(b);
  LD num = 0, den = 0;
  for (int i = 0; i < pr.dim; ++i) {
    num = std::max(num, std::fabs(LD(x[i]) - xr[i]));
    den = std::max(den, std::fabs(xr[i]));
  }
  EXPECT_LE(double(num / den), 1e-9);
  EXPECT_LE(double(pr.residual(x, b)), 1e-12);
  // Inactive rows: dz = -b exactly.
  for (int k = 0; k < pr.m; ++k)
    if (pr.inactive[k]) EXPECT_EQ(x[pr.n + pr.p + k], -b[pr.n + pr.p + k]);
  // Deterministic: a second factor + solve is bitwise identical.
  auto L1 = down(dv.solver.H(), size_t(n) * n);
  dv.solver.factor(s);
  dv.solver.solve(s);
  dv.sync();
  EXPECT_EQ(down(dv.solver.sol(), pr.dim), x);
  EXPECT_EQ(down(dv.solver.H(), size_t(n) * n), L1);
}

}  // namespace

TEST(ReducedKkt, MatchesDenseReference) {
  const Case cases[] = {
      {1, 1, 1, 0.5, 0.0, 1},     {9, 3, 6, 0.4, 0.3, 2},      {31, 5, 40, 0.2, 0.2, 3},
      {32, 0, 50, 0.2, 0.2, 4},   {33, 10, 0, 0.2, 0.0, 5},    {100, 20, 300, 0.05, 0.25, 6},
      {169, 50, 800, 0.03, 0.3, 7}, {241, 80, 1200, 0.03, 0.1, 8}, {300, 0, 0, 0.02, 0.0, 9},
      {800, 40, 900, 0.01, 0.2, 10},
  };
  for (const Case& c : cases) check_case(c);
}

TEST(ReducedKkt, EqualityOnlyAndBoxOnly) {
  check_case({64, 30, 0, 0.1, 0.0, 21});  // m = 0
  check_case({70, 0, 90, 0.1, 0.5, 22});  // p = 0
  check_case({65, 0, 0, 0.1, 0.0, 23});   // box-only QP: p = m = 0, H = P + x_reg
}

TEST(ReducedKkt, AllInequalitiesInactive) { check_case({40, 0, 60, 0.1, 1.0, 24}); }

TEST(ReducedKkt, NonPositivePivotSetsFactorStatus) {
  for (int j : {0, 5, 37, 70}) {
    Problem pr({80, 10, 50, 0.05, 0.2, 31});
    // Make the x diagonal of variable j strongly negative.
    pr.K[pr.maps.diag_x[j]] = -1e8;
    auto Hc = pr.H();
    const int expect = Problem::cholesky(Hc, pr.n);
    ASSERT_NE(expect, 0);
    Device dv(pr);
    dv.solver.factor(dv.stream.get());
    dv.sync();
    EXPECT_EQ(down(dv.solver.factor_status(), 1)[0], expect) << "j " << j;
    // A valid matrix afterwards resets the status.
    Problem ok({80, 10, 50, 0.05, 0.2, 31});
    dv.set_K(ok.K);
    dv.solver.factor(dv.stream.get());
    dv.sync();
    EXPECT_EQ(down(dv.solver.factor_status(), 1)[0], 0);
  }
}

TEST(ReducedKkt, NonFiniteValueSetsFactorStatus) {
  for (double bad : {double(NAN), double(INFINITY)}) {
    Problem pr({50, 5, 30, 0.1, 0.2, 41});
    pr.K[pr.maps.A_slots.empty() ? pr.maps.diag_x[3] : pr.maps.A_slots[0]] = bad;
    Device dv(pr);
    dv.solver.factor(dv.stream.get());
    dv.sync();
    EXPECT_NE(down(dv.solver.factor_status(), 1)[0], 0) << bad;
  }
  // A zero y diagonal (delta = 0) gives an infinite weight.
  Problem pr({50, 5, 30, 0.1, 0.2, 42});
  pr.K[pr.maps.diag_y[0]] = 0.0;
  Device dv(pr);
  dv.solver.factor(dv.stream.get());
  dv.sync();
  EXPECT_NE(down(dv.solver.factor_status(), 1)[0], 0);
}

TEST(ReducedKkt, InitRejectsLargeN) {
  Problem pr({8, 1, 1, 0.5, 0.0, 51});
  pr.maps.n = ReducedKktSolver::kMaxN + 1;
  CudaBuffer<int32_t> idx(1);
  CudaBuffer<double> K(1);
  ReducedKktSolver solver;
  try {
    solver.init(pr.maps, idx.data(), K.data(), nullptr);
    FAIL() << "expected an Unsupported error";
  } catch (const Error& e) {
    EXPECT_EQ(e.code(), ErrorCode::Unsupported);
  }
}

// Plain graph replay and replay inside conditional IF / WHILE bodies (the
// whole-solve graph mode captures the factorization and the solves there)
// reproduce the eager result bit for bit, with no allocation.
TEST(ReducedKkt, CapturedInConditionalBodiesIsBitwiseEager) {
  Problem pr({241, 30, 500, 0.03, 0.2, 61});
  Device dv(pr);
  cudaStream_t s = dv.stream.get();
  auto b = random_rhs(pr.dim, 62);
  dv.set_rhs(b);
  dv.solver.factor(s);
  dv.solver.solve(s);
  dv.sync();
  const auto x0 = down(dv.solver.sol(), pr.dim);
  Counters counters;
  LoopContext loops(&counters);
  CudaBuffer<int32_t> one(1), loop(1);
  const int32_t h1 = 1;
  one.copy_from_host(&h1, 1, s);
  CudaPinnedBuffer<int32_t> flag(1);
  auto body = [&](cudaStream_t bs) {
    dv.solver.factor(bs);
    dv.solver.solve(bs);
  };
  auto run_and_compare = [&](cudaGraph_t g, const char* what) {
    GraphExec ge;
    ge.instantiate(g);
    CUPIQP_CUDA_CHECK(cudaMemsetAsync(dv.solver.sol(), 0, sizeof(double) * pr.dim, s));
    ge.launch(s, &counters);
    dv.sync();
    EXPECT_EQ(down(dv.solver.sol(), pr.dim), x0) << what;
    EXPECT_EQ(down(dv.solver.factor_status(), 1)[0], 0) << what;
  };
  run_and_compare(capture(s, body, &counters), "plain");
  run_and_compare(capture(s, [&](cudaStream_t cs) { capture_if(loops, cs, one.data(), flag.data(), body); },
                          &counters),
                  "if");
  // WHILE: one iteration (the body clears the loop flag).
  run_and_compare(capture(s,
                          [&](cudaStream_t cs) {
                            CUPIQP_CUDA_CHECK(cudaMemcpyAsync(loop.data(), one.data(), sizeof(int32_t),
                                                              cudaMemcpyDeviceToDevice, cs));
                            capture_while(loops, cs, loop.data(), [&](cudaStream_t ws) {
                              body(ws);
                              CUPIQP_CUDA_CHECK(cudaMemsetAsync(loop.data(), 0, sizeof(int32_t), ws));
                            });
                          },
                          &counters),
                  "while");
}
