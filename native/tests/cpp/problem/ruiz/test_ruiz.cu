// M3 ruiz C++ tests: identity at iter 0, scale -> unscale round trip,
// device early exit vs a host-driven CPU reference loop, no allocation in
// the steady state, and graph capture/replay of scale_data.
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#include "common/buffer.hpp"
#include "common/handles.hpp"
#include "problem/ruiz/ruiz.hpp"

using namespace cupiqp;

namespace {

constexpr double kInf = 1e20;  // PIQP_INF

struct HostCsr {
  int rows = 0, cols = 0;
  std::vector<int32_t> indptr{0}, indices, row_idx;
  std::vector<double> values;
  int nnz() const { return int(values.size()); }
};

// Random sparse matrix; scale[i] / colscale[j] spread magnitudes over decades.
HostCsr random_csr(std::mt19937& g, int rows, int cols, double density, const std::vector<double>& rs,
                   const std::vector<double>& cs) {
  std::uniform_real_distribution<double> u(0, 1);
  std::normal_distribution<double> nrm;
  HostCsr M;
  M.rows = rows;
  M.cols = cols;
  M.indptr.assign(1, 0);
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      if (u(g) < density || j == i % cols) {
        M.indices.push_back(j);
        M.row_idx.push_back(i);
        M.values.push_back(nrm(g) * rs[i] * cs[j]);
      }
    }
    M.indptr.push_back(int32_t(M.indices.size()));
  }
  return M;
}

HostCsr random_sym(std::mt19937& g, int n, double density, const std::vector<double>& s) {
  std::uniform_real_distribution<double> u(0, 1);
  std::normal_distribution<double> nrm;
  std::vector<double> D(size_t(n) * n, 0.0);
  std::vector<char> nz(size_t(n) * n, 0);
  for (int i = 0; i < n; ++i) {
    D[size_t(i) * n + i] = (n + std::fabs(nrm(g))) * s[i] * s[i];
    nz[size_t(i) * n + i] = 1;
    for (int j = i + 1; j < n; ++j)
      if (u(g) < density) {
        double v = nrm(g) * s[i] * s[j];
        D[size_t(i) * n + j] = D[size_t(j) * n + i] = v;
        nz[size_t(i) * n + j] = nz[size_t(j) * n + i] = 1;
      }
  }
  HostCsr M;
  M.rows = M.cols = n;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j)
      if (nz[size_t(i) * n + j]) {
        M.indices.push_back(j);
        M.row_idx.push_back(i);
        M.values.push_back(D[size_t(i) * n + j]);
      }
    M.indptr.push_back(int32_t(M.indices.size()));
  }
  return M;
}

struct HostProblem {
  Dims dims;
  HostCsr P, A, G;
  std::vector<double> c, b, h_l, h_u, x_l, x_u;
};

HostProblem make_problem(unsigned seed, int n, int p, int m, bool hl, bool hu, bool xl, bool xu,
                         double decades) {
  std::mt19937 g(seed);
  std::uniform_real_distribution<double> e(-decades, decades), u(0, 1);
  std::normal_distribution<double> nrm;
  auto scales = [&](int k) {
    std::vector<double> s(k);
    for (auto& v : s) v = std::pow(10.0, e(g));
    return s;
  };
  HostProblem q;
  auto sx = scales(n), sy = scales(p), sz = scales(m);
  q.P = random_sym(g, n, 0.2, sx);
  q.A = random_csr(g, p, n, 0.2, sy, sx);
  q.G = random_csr(g, m, n, 0.2, sz, sx);
  q.c.resize(n);
  for (int i = 0; i < n; ++i) q.c[i] = nrm(g) * sx[i] * 10;
  q.b.resize(p);
  for (int i = 0; i < p; ++i) q.b[i] = nrm(g) * sy[i];
  auto bound = [&](int k, double sign, const std::vector<double>& s) {
    std::vector<double> v(k);
    for (int i = 0; i < k; ++i) v[i] = u(g) < 0.25 ? sign * INFINITY : sign * (1.0 + u(g)) * s[i];
    return v;
  };
  if (hl) q.h_l = bound(m, -1, sz);
  if (hu) q.h_u = bound(m, +1, sz);
  if (xl) q.x_l = bound(n, -1, std::vector<double>(n, 1.0));
  if (xu) q.x_u = bound(n, +1, std::vector<double>(n, 1.0));
  Dims& d = q.dims;
  d.n = n;
  d.p = p;
  d.m = m;
  d.num_hl = hl ? m : 0;
  d.num_hu = hu ? m : 0;
  d.num_xl = xl ? n : 0;
  d.num_xu = xu ? n : 0;
  d.nnz_P = q.P.nnz();
  d.nnz_A = q.A.nnz();
  d.nnz_G = q.G.nnz();
  return q;
}

// Device copy of a HostProblem, with host-computed masks (reference
// finite_bound_masks_kernel: lower finite iff v > -INF, upper iff v < INF).
struct DeviceProblem {
  CudaBuffer<int32_t> Pp, Pi, Pr, Ap, Ai, Ar, Gp, Gi, Gr;
  CudaBuffer<double> Pv, Av, Gv, c, b, h_l, h_u, x_l, x_u, mask, aG, axb, nfb;
  SparseDataView<double> view;

  DeviceProblem(const HostProblem& q, cudaStream_t s) {
    const Dims& d = q.dims;
    auto up_i = [&](CudaBuffer<int32_t>& B, const std::vector<int32_t>& h) {
      B.allocate(h.size());
      B.copy_from_host(h.data(), h.size(), s);
    };
    auto up_d = [&](CudaBuffer<double>& B, const std::vector<double>& h) {
      B.allocate(h.size());
      B.copy_from_host(h.data(), h.size(), s);
    };
    up_i(Pp, q.P.indptr), up_i(Pi, q.P.indices), up_i(Pr, q.P.row_idx), up_d(Pv, q.P.values);
    up_i(Ap, q.A.indptr), up_i(Ai, q.A.indices), up_i(Ar, q.A.row_idx), up_d(Av, q.A.values);
    up_i(Gp, q.G.indptr), up_i(Gi, q.G.indices), up_i(Gr, q.G.row_idx), up_d(Gv, q.G.values);
    up_d(c, q.c), up_d(b, q.b), up_d(h_l, q.h_l), up_d(h_u, q.h_u), up_d(x_l, q.x_l), up_d(x_u, q.x_u);
    std::vector<double> hm, ag(d.m, 0.0), ax(d.n, 0.0), nf(1, 0.0);
    auto side = [&](const std::vector<double>& v, bool lower, std::vector<double>& act) {
      for (size_t i = 0; i < v.size(); ++i) {
        double f = lower ? (v[i] > -kInf) : (v[i] < kInf);
        hm.push_back(f);
        act[i] = std::max(act[i], f);
        nf[0] += f;
      }
    };
    side(q.h_l, true, ag), side(q.h_u, false, ag), side(q.x_l, true, ax), side(q.x_u, false, ax);
    up_d(mask, hm), up_d(aG, ag), up_d(axb, ax), up_d(nfb, nf);
    auto csr = [&](CsrView<double>& V, const HostCsr& H, CudaBuffer<int32_t>& p, CudaBuffer<int32_t>& i,
                   CudaBuffer<int32_t>& r, CudaBuffer<double>& v) {
      V.rows = H.rows, V.cols = H.cols, V.nnz = H.nnz();
      V.indptr = p.data(), V.indices = i.data(), V.row_idx = r.data(), V.values = v.data();
    };
    view.dims = d;
    csr(view.P, q.P, Pp, Pi, Pr, Pv);
    csr(view.A, q.A, Ap, Ai, Ar, Av);
    csr(view.G, q.G, Gp, Gi, Gr, Gv);
    view.c = c.data(), view.b = b.data(), view.h_l = h_l.data(), view.h_u = h_u.data();
    view.x_l = x_l.data(), view.x_u = x_u.data();
    view.finite_mask_all = mask.data(), view.active_G_row = aG.data(), view.active_x_bound = axb.data();
    view.num_finite_bounds = nfb.data();
  }

  HostProblem download(const HostProblem& like) const {
    HostProblem q = like;
    auto dn = [](const CudaBuffer<double>& B, std::vector<double>& h) {
      if (!h.empty()) CUPIQP_CUDA_CHECK(cudaMemcpy(h.data(), B.data(), h.size() * 8, cudaMemcpyDeviceToHost));
    };
    CUPIQP_CUDA_CHECK(cudaDeviceSynchronize());
    dn(Pv, q.P.values), dn(Av, q.A.values), dn(Gv, q.G.values), dn(c, q.c), dn(b, q.b);
    dn(h_l, q.h_l), dn(h_u, q.h_u), dn(x_l, q.x_l), dn(x_u, q.x_u);
    return q;
  }
};

std::vector<double> dl(const double* p, size_t n) {
  std::vector<double> h(n);
  CUPIQP_CUDA_CHECK(cudaDeviceSynchronize());
  if (n) CUPIQP_CUDA_CHECK(cudaMemcpy(h.data(), p, n * 8, cudaMemcpyDeviceToHost));
  return h;
}
int passes(const RuizEquilibration& r) {
  int32_t h = -1;
  CUPIQP_CUDA_CHECK(cudaDeviceSynchronize());
  CUPIQP_CUDA_CHECK(cudaMemcpy(&h, r.passes_run(), 4, cudaMemcpyDeviceToHost));
  return h;
}

// ------------------------------------------------------------ CPU reference
// Sequential port of reference RuizEquilibration.scale_data (sparse backend)
// with the host-side early exit. Operation order follows the Warp kernels;
// only the cost-scaling mean is summed in a different order than on the GPU.
struct CpuRuiz {
  std::vector<double> delta, delta_b, x_b, cost{1.0};
  int passes = 0;

  void run(HostProblem& q, const std::vector<double>& xb_init, bool scale_cost, int max_iter) {
    const Dims& d = q.dims;
    const int n = d.n, p = d.p, m = d.m, W = n + p + m;
    const double lo = RuizEquilibration::kMinScaling, hi = RuizEquilibration::kMaxScaling;
    delta.assign(W, 1.0), delta_b.assign(n, 1.0), x_b = xb_init, cost.assign(1, 1.0);
    auto lim = [&](double v) { return v < lo ? 1.0 : (v > hi ? hi : v); };
    for (int it = 0; it < max_iter; ++it) {
      ++passes;
      std::vector<double> di(W, 0.0), dbi(n);
      for (int j = 0; j < n; ++j) {
        double v = 0;
        for (int k = q.P.indptr[j]; k < q.P.indptr[j + 1]; ++k) v = std::fmax(v, std::fabs(q.P.values[k]));
        di[j] = std::fmax(v, x_b[j]);
        dbi[j] = x_b[j];
      }
      for (int j = 0; j < p; ++j)
        for (int k = q.A.indptr[j]; k < q.A.indptr[j + 1]; ++k) di[n + j] = std::fmax(di[n + j], std::fabs(q.A.values[k]));
      for (int j = 0; j < m; ++j)
        for (int k = q.G.indptr[j]; k < q.G.indptr[j + 1]; ++k)
          di[n + p + j] = std::fmax(di[n + p + j], std::fabs(q.G.values[k]));
      for (int k = 0; k < q.A.nnz(); ++k) di[q.A.indices[k]] = std::fmax(di[q.A.indices[k]], std::fabs(q.A.values[k]));
      for (int k = 0; k < q.G.nnz(); ++k) di[q.G.indices[k]] = std::fmax(di[q.G.indices[k]], std::fabs(q.G.values[k]));
      for (auto& v : di) v = 1.0 / std::sqrt(lim(v));
      for (auto& v : dbi) v = 1.0 / std::sqrt(lim(v));
      for (int r = 0; r < n; ++r) {
        for (int k = q.P.indptr[r]; k < q.P.indptr[r + 1]; ++k)
          q.P.values[k] = q.P.values[k] * di[r] * di[q.P.indices[k]] * 1.0;
        q.c[r] = q.c[r] * di[r] * 1.0;
      }
      for (int r = 0; r < p; ++r)
        for (int k = q.A.indptr[r]; k < q.A.indptr[r + 1]; ++k)
          q.A.values[k] = q.A.values[k] * di[n + r] * di[q.A.indices[k]];
      for (int r = 0; r < m; ++r)
        for (int k = q.G.indptr[r]; k < q.G.indptr[r + 1]; ++k)
          q.G.values[k] = q.G.values[k] * di[n + p + r] * di[q.G.indices[k]];
      for (int k = 0; k < W; ++k) delta[k] *= di[k];
      for (int k = 0; k < n; ++k) {
        delta_b[k] *= dbi[k];
        x_b[k] = x_b[k] * dbi[k] * di[k];
      }
      if (scale_cost) {
        std::vector<double> pn(n, 0.0);
        for (int k = 0; k < q.P.nnz(); ++k) {
          double v = std::fabs(q.P.values[k]);
          pn[q.P.row_idx[k]] = std::fmax(pn[q.P.row_idx[k]], v);
          pn[q.P.indices[k]] = std::fmax(pn[q.P.indices[k]], v);
        }
        double sum = 0, cmax = 0;
        for (int k = 0; k < n; ++k) sum += pn[k], cmax = std::fmax(cmax, std::fabs(q.c[k]));
        double g = std::fmin(std::fmax(lo, sum / n), hi);
        g = std::fmax(g, cmax);
        g = std::fmin(std::fmax(lo, g), hi);
        const double gamma = 1.0 / g;
        for (auto& v : q.P.values) v *= gamma;
        for (auto& v : q.c) v *= gamma;
        cost[0] *= gamma;
      }
      double conv = 0;
      for (double v : di) conv = std::fmax(conv, std::fabs(1.0 - v));
      for (double v : dbi) conv = std::fmax(conv, std::fabs(1.0 - v));
      if (conv < RuizEquilibration::kConvergenceTol) break;  // host-driven early exit
    }
    for (int k = 0; k < n; ++k) {
      if (!q.x_l.empty()) q.x_l[k] *= delta_b[k];
      if (!q.x_u.empty()) q.x_u[k] *= delta_b[k];
    }
    for (int k = 0; k < p; ++k) q.b[k] *= delta[n + k];
    for (int k = 0; k < m; ++k) {
      if (!q.h_l.empty()) q.h_l[k] *= delta[n + p + k];
      if (!q.h_u.empty()) q.h_u[k] *= delta[n + p + k];
    }
  }
};

void expect_close(const std::vector<double>& a, const std::vector<double>& b, double rtol, const char* what) {
  ASSERT_EQ(a.size(), b.size()) << what;
  for (size_t i = 0; i < a.size(); ++i) {
    if (std::isinf(a[i]) || std::isinf(b[i])) {
      EXPECT_EQ(a[i], b[i]) << what << "[" << i << "]";
      continue;
    }
    EXPECT_LE(std::fabs(a[i] - b[i]), rtol * std::fabs(a[i])) << what << "[" << i << "] " << a[i] << " vs " << b[i];
  }
}
void expect_problem(const HostProblem& a, const HostProblem& b, double rtol) {
  expect_close(a.P.values, b.P.values, rtol, "P");
  expect_close(a.A.values, b.A.values, rtol, "A");
  expect_close(a.G.values, b.G.values, rtol, "G");
  expect_close(a.c, b.c, rtol, "c");
  expect_close(a.b, b.b, rtol, "b");
  expect_close(a.h_l, b.h_l, rtol, "h_l");
  expect_close(a.h_u, b.h_u, rtol, "h_u");
  expect_close(a.x_l, b.x_l, rtol, "x_l");
  expect_close(a.x_u, b.x_u, rtol, "x_u");
}

struct Layout {
  bool hl, hu, xl, xu;
  int p, m;
};
const Layout kLayouts[] = {{true, true, true, true, 6, 10}, {false, true, false, true, 6, 10},
                           {true, false, true, false, 0, 10}, {false, false, false, false, 6, 0},
                           {true, true, false, false, 0, 0 /* G absent -> no h */}};

HostProblem layout_problem(unsigned seed, const Layout& L, double decades) {
  const bool has_G = L.m > 0;
  return make_problem(seed, 20, L.p, L.m, L.hl && has_G, L.hu && has_G, L.xl, L.xu, decades);
}

}  // namespace

TEST(Ruiz, IterZeroIsIdentity) {
  CudaStream s = CudaStream::create();
  for (const auto& L : kLayouts) {
    HostProblem q = layout_problem(1, L, 3.0);
    DeviceProblem dp(q, s.get());
    RuizEquilibration r;
    r.init(dp.view, s.get());
    r.scale_data(dp.view, true, 0, s.get());
    HostProblem out = dp.download(q);
    expect_problem(q, out, 0.0);
    EXPECT_EQ(passes(r), 0);
    auto v = r.view();
    const size_t W = size_t(q.dims.n) + q.dims.p + q.dims.m;
    for (double x : dl(v.delta, W)) EXPECT_EQ(x, 1.0);
    for (double x : dl(v.delta_inv, W)) EXPECT_EQ(x, 1.0);
    for (double x : dl(v.delta_b, q.dims.n)) EXPECT_EQ(x, 1.0);
    for (double x : dl(v.cost_scaling, 1)) EXPECT_EQ(x, 1.0);
    for (double x : dl(v.primal_res_unscale_factor, q.dims.p + q.dims.num_ineq())) EXPECT_EQ(x, 1.0);
    EXPECT_EQ(dl(v.x_b_scaling, q.dims.n), dl(dp.axb.data(), q.dims.n));
  }
}

TEST(Ruiz, ScaleUnscaleRoundTrip) {
  CudaStream s = CudaStream::create();
  for (const auto& L : kLayouts) {
    HostProblem q = layout_problem(2, L, 4.0);
    DeviceProblem dp(q, s.get());
    RuizEquilibration r;
    r.init(dp.view, s.get());
    r.scale_data(dp.view, true, 10, s.get());
    r.unscale_data(dp.view, UpdateMask{}, s.get());
    HostProblem out = dp.download(q);
    expect_problem(q, out, 1e-14);
    // x_b_scaling returns to x_b_scaling_init.
    expect_close(dl(dp.axb.data(), q.dims.n), dl(r.view().x_b_scaling, q.dims.n), 1e-14, "x_b_scaling");
    // reuse_scaling reproduces the scaled data of scale_data.
    HostProblem cpu = q;
    CpuRuiz ref;
    ref.run(cpu, dl(dp.axb.data(), q.dims.n), true, 10);
    r.reuse_scaling(dp.view, s.get());
    expect_problem(cpu, dp.download(q), 1e-13);
  }
}

TEST(Ruiz, DeviceEarlyExitMatchesHostLoop) {
  CudaStream s = CudaStream::create();
  int converged_early = 0;
  for (const auto& L : kLayouts) {
    for (bool cost : {false, true}) {
      for (int max_iter : {1, 3, 10, 60}) {
        HostProblem q = layout_problem(3, L, 3.0);
        DeviceProblem dp(q, s.get());
        RuizEquilibration r;
        r.init(dp.view, s.get());
        r.scale_data(dp.view, cost, max_iter, s.get());
        HostProblem cpu = q;
        CpuRuiz ref;
        ref.run(cpu, dl(dp.axb.data(), q.dims.n), cost, max_iter);
        EXPECT_EQ(passes(r), ref.passes) << "max_iter " << max_iter << " cost " << cost;
        if (ref.passes < max_iter) ++converged_early;
        const double tol = cost ? 1e-13 : 0.0;  // without cost scaling every op is order-independent
        expect_problem(cpu, dp.download(q), tol);
        auto v = r.view();
        expect_close(ref.delta, dl(v.delta, ref.delta.size()), tol, "delta");
        expect_close(ref.delta_b, dl(v.delta_b, ref.delta_b.size()), tol, "delta_b");
        expect_close(ref.x_b, dl(v.x_b_scaling, ref.x_b.size()), tol, "x_b_scaling");
        expect_close(ref.cost, dl(v.cost_scaling, 1), tol, "cost_scaling");
      }
    }
  }
  EXPECT_GT(converged_early, 0);  // the early exit was exercised
}

TEST(Ruiz, NoAllocationInSteadyState) {
  CudaStream s = CudaStream::create();
  HostProblem q = layout_problem(4, kLayouts[0], 3.0);
  DeviceProblem dp(q, s.get());
  RuizEquilibration r;
  r.init(dp.view, s.get());
  CudaBuffer<double> rhs(1);
  auto cycle = [&] {
    r.scale_data(dp.view, true, 10, s.get());
    r.compute_constraints_rhs_inf_norm_unscaled(dp.view, rhs.data(), s.get());
    UpdateMask mk;
    mk.b = true;
    r.unscale_data(dp.view, mk, s.get());
    r.reuse_scaling(dp.view, s.get());
    r.refresh_x_b_scaling(s.get());
    r.unscale_data(dp.view, UpdateMask{}, s.get());
    r.reset(s.get());
  };
  cycle();  // warm up (module load)
  CUPIQP_CUDA_CHECK(cudaStreamSynchronize(s.get()));
  size_t free0 = 0, free1 = 0, total = 0;
  CUPIQP_CUDA_CHECK(cudaMemGetInfo(&free0, &total));
  for (int i = 0; i < 20; ++i) cycle();
  CUPIQP_CUDA_CHECK(cudaStreamSynchronize(s.get()));
  CUPIQP_CUDA_CHECK(cudaMemGetInfo(&free1, &total));
  // Other processes share the GPU; a same-process allocation would show up
  // as a decrease of at least the 2 MiB allocation granularity.
  EXPECT_LT(free0 > free1 ? free0 - free1 : 0, size_t(2) << 20);
}

TEST(Ruiz, ScaleDataCapturesAndReplays) {
  CudaStream s = CudaStream::create(cudaStreamNonBlocking);
  for (bool cost : {false, true}) {
    HostProblem q = layout_problem(5, kLayouts[0], 3.0);
    // Eager reference.
    DeviceProblem eager(q, s.get());
    RuizEquilibration re;
    re.init(eager.view, s.get());
    re.scale_data(eager.view, cost, 10, s.get());
    HostProblem want = eager.download(q);
    const int want_passes = passes(re);
    // Captured.
    DeviceProblem cap(q, s.get());
    RuizEquilibration rc;
    rc.init(cap.view, s.get());
    CUPIQP_CUDA_CHECK(cudaStreamSynchronize(s.get()));
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    CUPIQP_CUDA_CHECK(cudaStreamBeginCapture(s.get(), cudaStreamCaptureModeThreadLocal));
    rc.reset(s.get());
    rc.scale_data(cap.view, cost, 10, s.get());
    CUPIQP_CUDA_CHECK(cudaStreamEndCapture(s.get(), &graph));
    CUPIQP_CUDA_CHECK(cudaGraphInstantiate(&exec, graph, 0));
    for (int rep = 0; rep < 3; ++rep) {
      // Restore the unscaled data, then replay.
      DeviceProblem fresh(q, s.get());
      auto copy = [&](CudaBuffer<double>& dst, const CudaBuffer<double>& src) {
        if (dst.size()) dst.copy_from_device(src.data(), src.size(), s.get());
      };
      copy(cap.Pv, fresh.Pv), copy(cap.Av, fresh.Av), copy(cap.Gv, fresh.Gv), copy(cap.c, fresh.c);
      copy(cap.b, fresh.b), copy(cap.h_l, fresh.h_l), copy(cap.h_u, fresh.h_u), copy(cap.x_l, fresh.x_l);
      copy(cap.x_u, fresh.x_u);
      CUPIQP_CUDA_CHECK(cudaGraphLaunch(exec, s.get()));
      CUPIQP_CUDA_CHECK(cudaStreamSynchronize(s.get()));
      expect_problem(want, cap.download(q), 0.0);
      EXPECT_EQ(passes(rc), want_passes);
      EXPECT_EQ(dl(re.view().delta, q.dims.n + q.dims.p + q.dims.m),
                dl(rc.view().delta, q.dims.n + q.dims.p + q.dims.m));
      EXPECT_EQ(dl(re.view().cost_scaling, 1), dl(rc.view().cost_scaling, 1));
    }
    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(graph);
  }
}
