// Generality sweep and accuracy check of the reduced KKT backend
// (ReducedKktSolver), used to measure every kernel change
// (docs/modules/ncu_kernels.md).
//
//   cupiqp_native_reduced_kkt_sweep --synthetic [--reps 100] [--only name,...]
//       Synthetic MPC-like problems (see mpc_cases()): horizon T, nx states
//       and nu inputs per stage, dynamics equalities, stage constraints in G
//       with block (stage-dense), banded, sparse or globally coupled rows,
//       plus edge cases (p = 0, m = 0, box-only, mostly inactive rows,
//       mostly explicit zeros, n near the 2048 limit) and two value scalings
//       ("mild" and "ill": weights spread over 12 decades). Per case: graph
//       replay medians of factor / solve / one iteration (1 factor + 2
//       solves) and the accuracy of one factor + solve on two random rhs.
//   cupiqp_native_reduced_kkt_sweep --data <group_k.bin> --case 0 --label mpc
//       Real KKT states: the public solver runs with
//       max_iter = k for every k up to convergence; each state's kkt.values
//       is factored and solved on a random rhs and on the IPM residual rhs.
//
// Accuracy (long double, CPU, full condensed K):
//   res = ||K x - b||_inf / (||K||_inf ||x||_inf + ||b||_inf)   (normwise)
//   cw  = max_i |K x - b|_i / (|K| |x| + |b|)_i                 (componentwise)
// Prints one JSON object per line.
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "common/buffer.hpp"
#include "common/graph.hpp"
#include "common/handles.hpp"
#include "cupiqp/cupiqp.hpp"
#include "linalg/kkt_assembly/kkt_pattern.hpp"
#include "linalg/reduced_kkt/reduced_kkt_solver.hpp"
#include "snapshot_io.hpp"

using namespace cupiqp;

namespace {

double graph_us(cudaStream_t s, int reps, const std::function<void(cudaStream_t)>& op) {
  Counters c;
  GraphExec ge;
  ge.instantiate(capture(s, op, &c));
  for (int w = 0; w < 5; ++w) ge.launch(s, &c);
  CUPIQP_CUDA_CHECK(cudaStreamSynchronize(s));
  CudaEvent e0 = CudaEvent::create(true), e1 = CudaEvent::create(true);
  std::vector<double> t;
  for (int i = 0; i < reps; ++i) {
    e0.record(s);
    ge.launch(s, &c);
    e1.record(s);
    CUPIQP_CUDA_CHECK(cudaEventSynchronize(e1.get()));
    float ms = 0;
    CUPIQP_CUDA_CHECK(cudaEventElapsedTime(&ms, e0.get(), e1.get()));
    t.push_back(ms * 1e3);
  }
  std::sort(t.begin(), t.end());
  return t[t.size() / 2];
}

// K x and |K| |x| with K symmetric: x rows stored in full, y / z rows lower
// part (columns < n) plus the diagonal.
struct HostK {
  int n = 0, dim = 0;
  const CsrPattern* kp = nullptr;
  void residual(const std::vector<double>& v, const std::vector<double>& x, const std::vector<double>& b,
                double& res, double& cw) const {
    std::vector<long double> y(dim, 0.0L), a(dim, 0.0L), rs(dim, 0.0L);
    for (int r = 0; r < dim; ++r)
      for (int e = kp->indptr[r]; e < kp->indptr[r + 1]; ++e) {
        const int c = kp->indices[e];
        const long double ve = v[e];
        y[r] += ve * x[c];
        a[r] += std::fabs(ve * x[c]);
        rs[r] += std::fabs(ve);
        if (r >= n && c != r) {
          y[c] += ve * x[r];
          a[c] += std::fabs(ve * x[r]);
          rs[c] += std::fabs(ve);
        }
      }
    long double rmax = 0, xn = 0, bn = 0, knorm = 0, w = 0;
    for (int i = 0; i < dim; ++i) {
      if (!std::isfinite(x[i])) {
        res = cw = INFINITY;
        return;
      }
      const long double num = std::fabs(y[i] - (long double)b[i]);
      rmax = std::max(rmax, num);
      xn = std::max(xn, std::fabs((long double)x[i]));
      bn = std::max(bn, std::fabs((long double)b[i]));
      knorm = std::max(knorm, rs[i]);
      const long double den = a[i] + std::fabs((long double)b[i]);
      if (den > 0) w = std::max(w, num / den);
      else if (num > 0) w = INFINITY;
    }
    res = double(rmax / (knorm * xn + bn));
    cw = double(w);
  }
};

struct Acc {
  double res = 0, cw = 0;
  int status = 0;
};

// One factor + solve of K x = b with ReducedKktSolver; accuracy on the CPU.
Acc check(ReducedKktSolver& red, const HostK& hk, const std::vector<double>& Kh, const std::vector<double>& b,
          cudaStream_t s) {
  CUPIQP_CUDA_CHECK(cudaMemcpyAsync(red.rhs(), b.data(), sizeof(double) * b.size(), cudaMemcpyHostToDevice, s));
  red.factor(s);
  red.solve(s);
  std::vector<double> x(b.size());
  Acc a;
  CUPIQP_CUDA_CHECK(cudaMemcpyAsync(x.data(), red.sol(), sizeof(double) * x.size(), cudaMemcpyDeviceToHost, s));
  CUPIQP_CUDA_CHECK(cudaMemcpyAsync(&a.status, red.factor_status(), sizeof(int32_t), cudaMemcpyDeviceToHost, s));
  CUPIQP_CUDA_CHECK(cudaStreamSynchronize(s));
  hk.residual(Kh, x, b, a.res, a.cw);
  return a;
}

// ------------------------------------------------------------ MPC-like cases
enum class GMode { None, Block, Banded, Sparse, Global };

struct Case {
  std::string name;
  int T, nx, nu;
  bool dynamics = true;  // A: x_0 = x0, x_{k+1} = A x_k + B u_k
  GMode g = GMode::Block;
  int rows_per_stage = 0;  // stage constraints per stage (on x_k, u_k)
  double density = 1.0;    // Sparse / Global: fraction of the candidate columns per row
  int band = 3;            // Banded: half bandwidth within the stage variables
  double inactive = 0.0;   // fraction of z rows inactive (diag -1, G row zeroed)
  double zeros = 0.0;      // fraction of G stored entries that are explicit zeros
  bool ill = false;        // weights over 12 decades
};

std::vector<Case> mpc_cases() {
  std::vector<Case> v;
  auto add = [&](Case c) { v.push_back(c); };
  // Horizon (n and m scale with T).
  for (int T : {5, 10, 20, 40, 80}) add({"T" + std::to_string(T) + "_x6u3_block", T, 6, 3, true, GMode::Block, 18});
  // States / inputs per stage.
  for (auto xu : {std::pair<int, int>{2, 1}, {4, 2}, {12, 4}, {20, 8}, {32, 8}})
    add({"T20_x" + std::to_string(xu.first) + "u" + std::to_string(xu.second) + "_block", 20, xu.first, xu.second,
         true, GMode::Block, 2 * (xu.first + xu.second)});
  // Constraint structure and density.
  add({"T20_x8u4_banded", 20, 8, 4, true, GMode::Banded, 24, 1.0, 2});
  add({"T20_x8u4_sparse0.2", 20, 8, 4, true, GMode::Sparse, 24, 0.2});
  add({"T20_x8u4_sparse0.5", 20, 8, 4, true, GMode::Sparse, 24, 0.5});
  add({"T20_x8u4_global0.05", 20, 8, 4, true, GMode::Global, 24, 0.05});
  add({"T20_x8u4_global0.3", 20, 8, 4, true, GMode::Global, 6, 0.3});
  add({"T20_x8u4_rows120", 20, 8, 4, true, GMode::Block, 120});  // many rows per stage (MPC-like m ~ 15 n)
  // Edge cases.
  add({"edge_p0", 20, 8, 4, false, GMode::Block, 24});
  add({"edge_m0", 20, 8, 4, true, GMode::None, 0});
  add({"edge_boxonly", 20, 8, 4, false, GMode::None, 0});
  add({"edge_inactive0.9", 20, 8, 4, true, GMode::Block, 24, 1.0, 3, 0.9});
  add({"edge_zeros0.8", 20, 8, 4, true, GMode::Block, 24, 1.0, 3, 0.0, 0.8});
  add({"edge_n1", 0, 1, 0, false, GMode::Block, 3});
  add({"edge_n33", 2, 9, 3, true, GMode::Block, 4});
  add({"near_limit_T100_x16u4", 100, 16, 4, true, GMode::Block, 8});
  add({"near_limit_T50_x30u10_sparse", 50, 30, 10, true, GMode::Sparse, 20, 0.2});
  // Ill-conditioned weights.
  add({"ill_T20_x6u3_block", 20, 6, 3, true, GMode::Block, 18, 1.0, 3, 0.3, 0.0, true});
  add({"ill_T40_x12u4_sparse", 40, 12, 4, true, GMode::Sparse, 32, 0.3, 3, 0.3, 0.0, true});
  add({"ill_T80_x6u3_global", 80, 6, 3, true, GMode::Global, 4, 0.05, 3, 0.3, 0.0, true});
  return v;
}

CsrPattern to_csr(int rows, int cols, std::vector<std::vector<int>>& r) {
  CsrPattern c;
  c.rows = rows;
  c.cols = cols;
  c.indptr.push_back(0);
  for (auto& v : r) {
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());
    c.indices.insert(c.indices.end(), v.begin(), v.end());
    c.indptr.push_back(int32_t(c.indices.size()));
  }
  return c;
}

struct Built {
  KktHostMaps maps;
  std::vector<double> K;
};

Built build(const Case& cs, uint32_t seed) {
  std::mt19937 g(seed);
  std::uniform_real_distribution<double> u(-1, 1), u01(0, 1);
  const int T = cs.T, nx = cs.nx, nu = cs.nu;
  const int n = (T + 1) * nx + T * nu;
  auto xo = [&](int k) { return k * nx; };
  auto uo = [&](int k) { return (T + 1) * nx + k * nu; };
  // P: block diagonal (dense Q per state block, dense R per input block).
  std::vector<std::vector<int>> Pr(n);
  auto dense_block = [&](int o, int s) {
    for (int i = 0; i < s; ++i)
      for (int j = 0; j < s; ++j) Pr[o + i].push_back(o + j);
  };
  for (int k = 0; k <= T; ++k) dense_block(xo(k), nx);
  for (int k = 0; k < T; ++k) dense_block(uo(k), nu);
  std::vector<std::vector<int>> Ar, Gr;
  if (cs.dynamics) {
    for (int i = 0; i < nx; ++i) Ar.push_back({xo(0) + i});
    for (int k = 0; k < T; ++k)
      for (int i = 0; i < nx; ++i) {
        std::vector<int> r = {xo(k + 1) + i};
        for (int j = 0; j < nx; ++j) r.push_back(xo(k) + j);
        for (int j = 0; j < nu; ++j) r.push_back(uo(k) + j);
        Ar.push_back(r);
      }
  }
  if (cs.g != GMode::None) {
    for (int k = 0; k <= T; ++k) {
      std::vector<int> stage;
      for (int j = 0; j < nx; ++j) stage.push_back(xo(k) + j);
      if (k < T)
        for (int j = 0; j < nu; ++j) stage.push_back(uo(k) + j);
      const int sv = int(stage.size());
      for (int q = 0; q < cs.rows_per_stage; ++q) {
        std::vector<int> r;
        switch (cs.g) {
          case GMode::Block: r = stage; break;
          case GMode::Banded: {
            const int c = q % sv;
            for (int j = std::max(0, c - cs.band); j <= std::min(sv - 1, c + cs.band); ++j) r.push_back(stage[j]);
            break;
          }
          case GMode::Sparse:
            for (int j : stage)
              if (u01(g) < cs.density) r.push_back(j);
            if (r.empty()) r.push_back(stage[q % sv]);
            break;
          case GMode::Global:
            for (int j = 0; j < n; ++j)
              if (u01(g) < cs.density) r.push_back(j);
            r.push_back(stage[q % sv]);
            break;
          default: break;
        }
        Gr.push_back(r);
      }
    }
  }
  const int p = int(Ar.size()), m = int(Gr.size());
  CsrPattern P = to_csr(n, n, Pr), A = to_csr(p, n, Ar), G = to_csr(m, n, Gr);
  Built b;
  b.maps = build_kkt_host_maps(P, A, G);
  const KktHostMaps& mp = b.maps;
  std::vector<double>& K = b.K;
  K.assign(mp.kkt.nnz(), 0.0);
  // Symmetric P values: hash of (min, max).
  auto sym = [&](int i, int j) {
    const uint32_t a = uint32_t(std::min(i, j)) * 2654435761u ^ uint32_t(std::max(i, j)) * 40503u ^ seed;
    return double(a % 2001) / 1000.0 - 1.0;
  };
  for (int r = 0, e = 0; r < n; ++r)
    for (int q = P.indptr[r]; q < P.indptr[r + 1]; ++q, ++e) K[mp.P_slots[e]] = 0.1 * sym(r, P.indices[q]);
  for (size_t e = 0; e < mp.A_slots.size(); ++e) K[mp.A_slots[e]] = u(g);
  for (size_t e = 0; e < mp.G_slots.size(); ++e) K[mp.G_slots[e]] = u01(g) < cs.zeros ? 0.0 : u(g);
  for (int i = 0; i < n; ++i) K[mp.diag_x[i]] = (cs.ill ? 1e-8 : 1.0) + 2.0 * u01(g);
  const double delta = cs.ill ? 1e-9 : 1e-4;
  for (int e : mp.diag_y) K[e] = -delta;
  for (int k = 0; k < m; ++k) {
    const int slot = mp.diag_z[k];
    if (u01(g) < cs.inactive) {
      K[slot] = -1.0;
      for (int e = mp.kkt.indptr[n + p + k]; e < mp.kkt.indptr[n + p + k + 1]; ++e)
        if (e != slot) K[e] = 0.0;
    } else {
      K[slot] = cs.ill ? -std::pow(10.0, -9.0 + 12.0 * u01(g)) : -(0.1 + 2.0 * u01(g));
    }
  }
  return b;
}

void run_synthetic(int reps, const std::set<std::string>& only, cudaStream_t s) {
  for (const Case& cs : mpc_cases()) {
    if (!only.empty() && !only.count(cs.name)) continue;
    Built b = build(cs, 7);
    const KktHostMaps& mp = b.maps;
    const int dim = mp.kkt.rows;
    CudaBuffer<int32_t> ix(std::max<size_t>(mp.kkt.indices.size(), 1));
    ix.copy_from_host(mp.kkt.indices.data(), mp.kkt.indices.size(), s);
    CudaBuffer<double> Kd(std::max<size_t>(b.K.size(), 1));
    Kd.copy_from_host(b.K.data(), b.K.size(), s);
    ReducedKktSolver red;
    red.init(mp, ix.data(), Kd.data(), s);
    HostK hk{mp.n, dim, &mp.kkt};
    std::mt19937 g(11);
    std::uniform_real_distribution<double> u(-1, 1);
    double res = 0, cw = 0;
    int st = 0;
    for (int r = 0; r < 2; ++r) {
      std::vector<double> rhs(dim);
      for (auto& v : rhs) v = u(g);
      Acc a = check(red, hk, b.K, rhs, s);
      res = std::max(res, a.res);
      cw = std::max(cw, a.cw);
      st = std::max(st, a.status);
    }
    const double tf = graph_us(s, reps, [&](cudaStream_t c) { red.factor(c); });
    const double ts = graph_us(s, reps, [&](cudaStream_t c) { red.solve(c); });
    const double ti = graph_us(s, reps, [&](cudaStream_t c) {
      red.factor(c);
      red.solve(c);
      red.solve(c);
    });
    int64_t nnzM = 0;
    for (int r = mp.n; r < dim; ++r) nnzM += mp.kkt.indptr[r + 1] - mp.kkt.indptr[r] - 1;
    std::printf(
        "{\"case\":\"%s\",\"n\":%d,\"p\":%d,\"m\":%d,\"nnzM\":%lld,\"factor_us\":%.2f,\"solve_us\":%.2f,"
        "\"iter_us\":%.2f,\"res\":%.3e,\"cw\":%.3e,\"factor_status\":%d}\n",
        cs.name.c_str(), mp.n, mp.p, mp.m, (long long)nnzM, tf, ts, ti, res, cw, st);
    std::fflush(stdout);
  }
}

void run_real(const std::string& data, int c, const std::string& label, cudaStream_t s) {
  bench::Snapshots g = bench::load_snapshots(data);
  KktHostMaps maps = build_kkt_host_maps(g.P, g.A, g.C);
  const int n = g.n, p = g.p, m = g.m, dim = n + p + m;
  CudaBuffer<double> packets(g.total), Kd(maps.kkt.nnz());
  CudaBuffer<int32_t> ix(maps.kkt.indices.size());
  ix.copy_from_host(maps.kkt.indices.data(), maps.kkt.indices.size(), s);
  packets.copy_from_host(g.case_values(c), g.total, s);
  CUPIQP_CUDA_CHECK(cudaStreamSynchronize(s));
  ReducedKktSolver red;
  red.init(maps, ix.data(), Kd.data(), s);
  HostK hk{n, dim, &maps.kkt};
  SparseSolver solver;
  auto& st = solver.settings();
  st.eps_abs = st.eps_duality_gap_abs = 1e-8;
  st.eps_rel = st.eps_duality_gap_rel = 0.0;
  solver.setup(g.P, g.view(packets.data()), &g.A, &g.C);
  solver.solve();
  const int N = solver.info().iter;
  const DebugBuffer *dK = nullptr, *dRp = nullptr, *dRd = nullptr;
  auto dbg = solver.debug_buffers();
  for (auto& d : dbg) {
    if (d.name == "kkt.values") dK = &d;
    if (d.name == "res.primal") dRp = &d;
    if (d.name == "res.dual") dRd = &d;
  }
  const bool with_res = dRp && dRd && dRd->cols == p + 2 * m;
  std::mt19937 rg(12345);
  std::uniform_real_distribution<double> U(-1, 1);
  std::vector<double> rnd(dim);
  for (auto& v : rnd) v = U(rg);
  for (int k = 1; k <= N; ++k) {
    st.max_iter = k;
    solver.solve();
    std::vector<double> Kh(maps.kkt.nnz()), rp, rd;
    CUPIQP_CUDA_CHECK(cudaMemcpyAsync(Kd.data(), dK->ptr, sizeof(double) * Kh.size(), cudaMemcpyDeviceToDevice,
                                      solver.stream()));
    CUPIQP_CUDA_CHECK(cudaMemcpyAsync(Kh.data(), dK->ptr, sizeof(double) * Kh.size(), cudaMemcpyDeviceToHost,
                                      solver.stream()));
    if (with_res) {
      rp.resize(dRp->cols);
      rd.resize(dRd->cols);
      CUPIQP_CUDA_CHECK(cudaMemcpyAsync(rp.data(), dRp->ptr, sizeof(double) * rp.size(), cudaMemcpyDeviceToHost,
                                        solver.stream()));
      CUPIQP_CUDA_CHECK(cudaMemcpyAsync(rd.data(), dRd->ptr, sizeof(double) * rd.size(), cudaMemcpyDeviceToHost,
                                        solver.stream()));
    }
    CUPIQP_CUDA_CHECK(cudaStreamSynchronize(solver.stream()));
    Acc ar = check(red, hk, Kh, rnd, s), ae{NAN, NAN, 0};
    if (with_res) {
      std::vector<double> b(dim);
      for (int i = 0; i < n; ++i) b[i] = rp[i];
      for (int i = 0; i < p; ++i) b[n + i] = rd[i];
      for (int i = 0; i < m; ++i) b[n + p + i] = rd[p + m + i] - rd[p + i];
      ae = check(red, hk, Kh, b, s);
    }
    std::printf(
        "{\"label\":\"%s\",\"case\":%d,\"k\":%d,\"res_random\":%.3e,\"cw_random\":%.3e,\"res_residual\":%.3e,"
        "\"cw_residual\":%.3e,\"factor_status\":%d}\n",
        label.c_str(), c, k, ar.res, ar.cw, ae.res, ae.cw, ar.status);
    std::fflush(stdout);
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    std::string data, label = "data";
    std::set<std::string> only;
    int c = 0, reps = 100;
    bool synth = false;
    for (int i = 1; i < argc; ++i) {
      std::string a = argv[i];
      if (a == "--data") data = argv[++i];
      else if (a == "--case") c = std::stoi(argv[++i]);
      else if (a == "--label") label = argv[++i];
      else if (a == "--reps") reps = std::stoi(argv[++i]);
      else if (a == "--synthetic") synth = true;
      else if (a == "--only") {
        std::stringstream ss(argv[++i]);
        std::string t;
        while (std::getline(ss, t, ',')) only.insert(t);
      } else throw std::runtime_error("unknown argument " + a);
    }
    CudaStream stream = CudaStream::create();
    cudaStream_t s = stream.get();
    if (synth) run_synthetic(reps, only, s);
    if (!data.empty()) run_real(data, c, label, s);
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
