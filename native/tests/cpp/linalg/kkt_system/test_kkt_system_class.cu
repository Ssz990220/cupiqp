// M6 kkt_system class-level tests (built only when the spmv, kkt_assembly and
// cudss implementations are present, see CMakeLists.txt).
//
// A small random QP block structure (n = 9, p = 3, m = 6, all four bound
// sides present, some infinite entries, one G row with both sides infinite)
// is assembled with KktAssembly / CudssSolver / SparseOperators; KktSystem
// runs update_scalings_and_factor + solve, and the un-condensed system of the
// reference KKTSystem docstring (with the D_b * D_x box scaling) is checked
// on the CPU on the rows and columns of finite bounds:
//   (P + rho I) dx + A^T dy + G^T (dz_u - dz_l) + xbs (dz_bu - dz_bl) = r_x
//   A dx - delta dy = r_y
//    G dx - delta dz_u  + ds_u  = r_z_u      -G dx - delta dz_l  + ds_l  = r_z_l
//   xbs dx - delta dz_bu + ds_bu = r_z_bu    -xbs dx - delta dz_bl + ds_bl = r_z_bl
//   S dz + Z ds = r_s                        (every finite inequality entry)
// Entries of infinite bounds must come out exactly zero.
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#include "common/buffer.hpp"
#include "common/graph.hpp"
#include "common/handles.hpp"
#include "linalg/kkt_system/kkt_system.hpp"

using namespace cupiqp;

namespace {

struct HostCsr {
  CsrPattern pat;
  std::vector<double> val;
  std::vector<int32_t> row_idx;
};

HostCsr random_csr(int rows, int cols, double density, std::mt19937& g, bool symmetric_pd) {
  std::uniform_real_distribution<double> u(-1.0, 1.0), pr(0.0, 1.0);
  std::vector<double> dense(size_t(rows) * cols, 0.0);
  std::vector<char> nz(size_t(rows) * cols, 0);
  if (symmetric_pd) {
    // P = M^T M + diag, full pattern stored, with one explicit zero.
    std::vector<double> M(size_t(rows) * cols, 0.0);
    for (auto& v : M) v = pr(g) < density ? u(g) : 0.0;
    for (int i = 0; i < rows; ++i)
      for (int j = 0; j < cols; ++j) {
        double s = 0;
        for (int k = 0; k < rows; ++k) s += M[size_t(k) * cols + i] * M[size_t(k) * cols + j];
        dense[size_t(i) * cols + j] = s + (i == j ? 1.0 : 0.0);
        nz[size_t(i) * cols + j] = (s != 0.0 || i == j);
      }
    nz[1] = 1;  // explicit zero at (0, 1) / (1, 0) if absent
    nz[size_t(1) * cols + 0] = 1;
  } else {
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j)
        if (pr(g) < density) {
          dense[size_t(i) * cols + j] = u(g);
          nz[size_t(i) * cols + j] = 1;
        }
      int j = int(pr(g) * cols) % cols;  // at least one entry per row
      if (!nz[size_t(i) * cols + j]) {
        dense[size_t(i) * cols + j] = u(g);
        nz[size_t(i) * cols + j] = 1;
      }
    }
  }
  HostCsr c;
  c.pat.rows = rows;
  c.pat.cols = cols;
  c.pat.indptr.push_back(0);
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j)
      if (nz[size_t(i) * cols + j]) {
        c.pat.indices.push_back(j);
        c.val.push_back(dense[size_t(i) * cols + j]);
        c.row_idx.push_back(i);
      }
    c.pat.indptr.push_back(int32_t(c.pat.indices.size()));
  }
  return c;
}

void matvec(const HostCsr& M, const std::vector<double>& x, std::vector<double>& y, bool transpose) {
  y.assign(transpose ? M.pat.cols : M.pat.rows, 0.0);
  for (int i = 0; i < M.pat.rows; ++i)
    for (int k = M.pat.indptr[i]; k < M.pat.indptr[i + 1]; ++k) {
      int j = M.pat.indices[k];
      if (transpose)
        y[j] += M.val[k] * x[i];
      else
        y[i] += M.val[k] * x[j];
    }
}

template <class T>
std::vector<T> down(const T* d, size_t k) {
  std::vector<T> h(k);
  if (k) CUPIQP_CUDA_CHECK(cudaMemcpy(h.data(), d, k * sizeof(T), cudaMemcpyDeviceToHost));
  return h;
}

template <class T>
void up(CudaBuffer<T>& b, const std::vector<T>& h) {
  b.allocate(h.size());
  if (!h.empty()) CUPIQP_CUDA_CHECK(cudaMemcpy(b.data(), h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
}

struct Fixture {
  int n = 9, p = 3, m = 6;
  Dims d;
  HostCsr P, A, G;
  std::vector<double> mask, active_G, xbs, s_all, z_all, rhs_primal, rhs_dual;
  double rho = 1e-6, delta = 1e-4;

  // device storage
  CudaBuffer<int32_t> Pi, Pj, Pr, Ai, Aj, Ar, Gi, Gj, Gr;
  CudaBuffer<double> Pv, Av, Gv, mask_d, active_d, xbs_d, rho_d, delta_d, vars_p, vars_d, rhs_p, rhs_d, lhs_p, lhs_d;
  CudaBuffer<double> sf;
  CudaBuffer<int32_t> si;
  SparseDataView<double> data;
  ScalingView<double> scaling;

  explicit Fixture(unsigned seed) {
    std::mt19937 g(seed);
    std::uniform_real_distribution<double> pos(0.05, 2.0), sn(-1.0, 1.0), pr(0.0, 1.0);
    P = random_csr(n, n, 0.3, g, true);
    A = random_csr(p, n, 0.4, g, false);
    G = random_csr(m, n, 0.4, g, false);
    d.B = 1;
    d.n = n;
    d.p = p;
    d.m = m;
    d.num_hl = m;
    d.num_hu = m;
    d.num_xl = n;
    d.num_xu = n;
    d.nnz_P = P.pat.nnz();
    d.nnz_A = A.pat.nnz();
    d.nnz_G = G.pat.nnz();
    const int ni = d.num_ineq();
    mask.assign(ni, 1.0);
    for (int i = 0; i < ni; ++i) mask[i] = pr(g) < 0.75 ? 1.0 : 0.0;
    mask[d.off_hl() + 2] = 0.0;  // G row 2: both sides infinite
    mask[d.off_hu() + 2] = 0.0;
    mask[d.off_hl() + 0] = 1.0;  // G row 0: two-sided
    mask[d.off_hu() + 0] = 1.0;
    active_G.assign(m, 0.0);
    for (int i = 0; i < m; ++i) active_G[i] = (mask[d.off_hl() + i] > 0.5 || mask[d.off_hu() + i] > 0.5) ? 1.0 : 0.0;
    xbs.resize(n);
    for (auto& v : xbs) v = pos(g);
    s_all.resize(ni);
    z_all.resize(ni);
    for (int i = 0; i < ni; ++i) {
      s_all[i] = pos(g);
      z_all[i] = pos(g);
    }
    rhs_primal.resize(d.num_primal());
    rhs_dual.resize(d.num_dual());
    for (auto& v : rhs_primal) v = sn(g);
    for (auto& v : rhs_dual) v = sn(g);

    auto upcsr = [](const HostCsr& c, CudaBuffer<int32_t>& i, CudaBuffer<int32_t>& j, CudaBuffer<int32_t>& r,
                    CudaBuffer<double>& v, CsrView<double>& view) {
      up(i, c.pat.indptr);
      up(j, c.pat.indices);
      up(r, c.row_idx);
      up(v, c.val);
      view.rows = c.pat.rows;
      view.cols = c.pat.cols;
      view.nnz = c.pat.nnz();
      view.indptr = i.data();
      view.indices = j.data();
      view.row_idx = r.data();
      view.values = v.data();
    };
    data.dims = d;
    upcsr(P, Pi, Pj, Pr, Pv, data.P);
    upcsr(A, Ai, Aj, Ar, Av, data.A);
    upcsr(G, Gi, Gj, Gr, Gv, data.G);
    up(mask_d, mask);
    up(active_d, active_G);
    up(xbs_d, xbs);
    data.finite_mask_all = mask_d.data();
    data.active_G_row = active_d.data();
    scaling.x_b_scaling = xbs_d.data();
    up(rho_d, std::vector<double>{rho});
    up(delta_d, std::vector<double>{delta});
    std::vector<double> vp(d.num_primal(), 0.0), vd(d.num_dual(), 0.0);
    std::copy(s_all.begin(), s_all.end(), vp.begin() + n);
    std::copy(z_all.begin(), z_all.end(), vd.begin() + p);
    up(vars_p, vp);
    up(vars_d, vd);
    up(rhs_p, rhs_primal);
    up(rhs_d, rhs_dual);
    up(lhs_p, std::vector<double>(d.num_primal(), 123.0));
    up(lhs_d, std::vector<double>(d.num_dual(), 123.0));
    std::vector<double> f(SF_COUNT, 0.0);
    std::vector<int32_t> ii(SI_COUNT, 0);
    f[SF_IR_EPS_ABS] = 1e-12;
    f[SF_IR_EPS_REL] = 1e-12;
    f[SF_IR_MIN_IMPROVEMENT_RATE] = 5.0;
    ii[SI_IR_MAX_ITER] = 10;
    up(sf, f);
    up(si, ii);
  }

  VariablesView<double> view(CudaBuffer<double>& pp, CudaBuffer<double>& dd) const {
    VariablesView<double> v;
    v.dims = d;
    v.primal = pp.data();
    v.dual = dd.data();
    return v;
  }

  // Max abs residual of the un-condensed system over finite rows, and the max
  // abs value of lhs entries on infinite bounds.
  void check(const std::vector<double>& lp, const std::vector<double>& ld, double* res, double* inf_entries) const {
    const int ni = d.num_ineq();
    std::vector<double> dx(lp.begin(), lp.begin() + n), dy(ld.begin(), ld.begin() + p);
    std::vector<double> ds(lp.begin() + n, lp.end()), dz(ld.begin() + p, ld.end());
    std::vector<double> zeff(ni, 0.0);
    for (int i = 0; i < ni; ++i) zeff[i] = mask[i] > 0.5 ? dz[i] : 0.0;
    std::vector<double> Px, ATy, Ax, Gx, GTz, zdiff(m);
    matvec(P, dx, Px, false);
    matvec(A, dy, ATy, true);
    matvec(A, dx, Ax, false);
    matvec(G, dx, Gx, false);
    for (int i = 0; i < m; ++i) zdiff[i] = zeff[d.off_hu() + i] - zeff[d.off_hl() + i];
    matvec(G, zdiff, GTz, true);
    double r = 0;
    for (int j = 0; j < n; ++j) {
      double v = Px[j] + rho * dx[j] + ATy[j] + GTz[j] +
                 xbs[j] * (zeff[d.off_xu() + j] - zeff[d.off_xl() + j]);
      r = std::max(r, std::fabs(v - rhs_primal[j]));
    }
    for (int i = 0; i < p; ++i) r = std::max(r, std::fabs(Ax[i] - delta * dy[i] - rhs_dual[i]));
    double inf_max = 0;
    for (int k = 0; k < ni; ++k) {
      if (mask[k] < 0.5) {
        inf_max = std::max({inf_max, std::fabs(dz[k]), std::fabs(ds[k])});
        continue;
      }
      double coupling;
      if (k < d.off_hu()) coupling = -Gx[k - d.off_hl()];
      else if (k < d.off_xl()) coupling = Gx[k - d.off_hu()];
      else if (k < d.off_xu()) coupling = -xbs[k - d.off_xl()] * dx[k - d.off_xl()];
      else coupling = xbs[k - d.off_xu()] * dx[k - d.off_xu()];
      r = std::max(r, std::fabs(coupling - delta * dz[k] + ds[k] - rhs_dual[p + k]));
      r = std::max(r, std::fabs(s_all[k] * dz[k] + z_all[k] * ds[k] - rhs_primal[n + k]));
    }
    *res = r;
    *inf_entries = inf_max;
  }
};

struct Stack {
  // Declared first: destroyed last. cuDSS frees its workspace on the last
  // bound stream when its data object is destroyed.
  CudaStream stream = CudaStream::create();
  Counters counters;
  CusparseHandle cusparse;
  KktAssembly kkt;
  CudssSolver cudss{&counters};
  SparseOperators ops;
  LoopContext loops{&counters};
  KktSystem sys;

  explicit Stack(Fixture& f, bool deterministic = false) {
    cudaStream_t s = stream.get();
    kkt.init(f.P.pat, f.A.pat, f.G.pat, f.data, s);
    cudss.init(kkt.dim(), kkt.nnz(), kkt.indptr(), kkt.indices(), kkt.values(), deterministic, s);
    cudss.analyze(s);
    ops.init(cusparse.get(), f.data);
    Settings settings;
    sys.init(f.d, settings, &kkt, &cudss, &ops, DeviceSettingsView<double>{f.sf.data(), f.si.data()}, &loops, s);
  }
};

void run_and_check(unsigned seed, bool ir) {
  Fixture f(seed);
  Stack st(f);
  cudaStream_t s = st.stream.get();
  st.sys.update_scalings_and_factor(f.data, f.scaling, ir, f.rho_d.data(), f.delta_d.data(),
                                    f.view(f.vars_p, f.vars_d), s);
  st.sys.solve(f.data, f.scaling, f.view(f.rhs_p, f.rhs_d), f.view(f.lhs_p, f.lhs_d), s);
  CUPIQP_CUDA_CHECK(cudaStreamSynchronize(s));
  auto lp = down(f.lhs_p.data(), f.lhs_p.size());
  auto ld = down(f.lhs_d.data(), f.lhs_d.size());
  double res = 0, inf_entries = 0;
  f.check(lp, ld, &res, &inf_entries);
  EXPECT_LE(res, 1e-9) << "seed " << seed << " ir " << ir;
  EXPECT_EQ(inf_entries, 0.0);
  EXPECT_EQ(down(st.sys.solve_status(), 1)[0], 0);
  EXPECT_EQ(down(st.sys.factor_status(), 1)[0], 0);
}

}  // namespace

TEST(KktSystemClass, FullSystemResidual) {
  for (unsigned seed : {1u, 2u, 3u}) run_and_check(seed, false);
}

TEST(KktSystemClass, FullSystemResidualWithIterativeRefinement) {
  for (unsigned seed : {4u, 5u}) run_and_check(seed, true);
}

// Two cuDSS factorizations of the same matrix may differ in the last bits
// (non-deterministic mode; deterministic mode showed intermittent illegal
// addresses in cuDSS SOLVE on this machine, see docs/modules/kkt_system.md),
// so captured and eager results are compared to 1e-10 relative.
TEST(KktSystemClass, CapturedSolveMatchesEager) {
  Fixture f(7);
  Stack st(f);
  cudaStream_t s = st.stream.get();
  st.sys.update_scalings_and_factor(f.data, f.scaling, false, f.rho_d.data(), f.delta_d.data(),
                                    f.view(f.vars_p, f.vars_d), s);
  st.sys.solve(f.data, f.scaling, f.view(f.rhs_p, f.rhs_d), f.view(f.lhs_p, f.lhs_d), s);
  CUPIQP_CUDA_CHECK(cudaStreamSynchronize(s));
  auto lp0 = down(f.lhs_p.data(), f.lhs_p.size());
  auto ld0 = down(f.lhs_d.data(), f.lhs_d.size());
  f.lhs_p.zero_async(s);
  f.lhs_d.zero_async(s);
  GraphExec g;
  g.instantiate(capture(
      s,
      [&](cudaStream_t cs) {
        st.sys.update_scalings_and_factor(f.data, f.scaling, false, f.rho_d.data(), f.delta_d.data(),
                                          f.view(f.vars_p, f.vars_d), cs);
        st.sys.solve(f.data, f.scaling, f.view(f.rhs_p, f.rhs_d), f.view(f.lhs_p, f.lhs_d), cs);
      },
      &st.counters));
  g.launch(s, &st.counters);
  CUPIQP_CUDA_CHECK(cudaStreamSynchronize(s));
  auto near = [](const std::vector<double>& a, const std::vector<double>& b) {
    double scale = 0, diff = 0;
    for (size_t i = 0; i < a.size(); ++i) {
      scale = std::max(scale, std::fabs(b[i]));
      diff = std::max(diff, std::fabs(a[i] - b[i]));
    }
    return diff <= 1e-10 * scale;
  };
  EXPECT_TRUE(near(down(f.lhs_p.data(), f.lhs_p.size()), lp0));
  EXPECT_TRUE(near(down(f.lhs_d.data(), f.lhs_d.size()), ld0));
  EXPECT_EQ(down(st.sys.solve_status(), 1)[0], 0);
}

TEST(KktSystemClass, NanInputSetsSolveStatus) {
  Fixture f(9);
  f.rhs_primal[0] = NAN;
  CUPIQP_CUDA_CHECK(cudaMemcpy(f.rhs_p.data(), f.rhs_primal.data(), sizeof(double), cudaMemcpyHostToDevice));
  Stack st(f);
  cudaStream_t s = st.stream.get();
  st.sys.update_scalings_and_factor(f.data, f.scaling, false, f.rho_d.data(), f.delta_d.data(),
                                    f.view(f.vars_p, f.vars_d), s);
  st.sys.solve(f.data, f.scaling, f.view(f.rhs_p, f.rhs_d), f.view(f.lhs_p, f.lhs_d), s);
  CUPIQP_CUDA_CHECK(cudaStreamSynchronize(s));
  EXPECT_EQ(down(st.sys.solve_status(), 1)[0], 1);
}
