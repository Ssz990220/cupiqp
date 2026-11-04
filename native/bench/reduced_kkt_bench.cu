// Microbenchmark of the reduced KKT backend (ReducedKktSolver) against cuDSS
// on the same KKT values, as replayed CUDA graphs (median of --reps).
//
//   cupiqp_native_reduced_kkt_bench --data <group_k.bin> --case 0 [--reps 200]
//       Real KKT states: the public solver (reduced backend, replay
//       settings) runs to convergence with max_iter = k for k = N/2 and N;
//       kkt.values of iteration k is copied and timed.
//   cupiqp_native_reduced_kkt_bench --sweep 128,256,512,1024,2048 [--reps 50]
//       Synthetic KKT matrices (P with ~8 entries per row, m = 2n rows of
//       G with ~8 entries each, p = n / 4) for the n limit.
// Prints one JSON object per line: label, n, p, m, op, graph_us.
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <functional>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "common/buffer.hpp"
#include "common/graph.hpp"
#include "common/handles.hpp"
#include "cupiqp/cupiqp.hpp"
#include "linalg/cudss/cudss_solver.hpp"
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

// Times reduced assemble / cholesky / solve / one iteration (1 factor + 2
// solves) and the cuDSS equivalents on the KKT values in Kd.
void time_state(const std::string& label, const KktHostMaps& maps, CudaBuffer<double>& Kd, int reps,
                bool with_cudss, cudaStream_t s) {
  const CsrPattern& kp = maps.kkt;
  const int dim = kp.rows;
  CudaBuffer<int32_t> ip(kp.indptr.size()), ix(kp.indices.size());
  ip.copy_from_host(kp.indptr.data(), kp.indptr.size(), s);
  ix.copy_from_host(kp.indices.data(), kp.indices.size(), s);
  ReducedKktSolver red;
  red.init(maps, ix.data(), Kd.data(), s);
  std::vector<double> b(dim);
  std::mt19937 g(1);
  std::uniform_real_distribution<double> u(-1, 1);
  for (auto& v : b) v = u(g);
  CUPIQP_CUDA_CHECK(cudaMemcpyAsync(red.rhs(), b.data(), sizeof(double) * dim, cudaMemcpyHostToDevice, s));
  red.factor(s);
  int32_t st = -1;
  CUPIQP_CUDA_CHECK(cudaMemcpyAsync(&st, red.factor_status(), sizeof(int32_t), cudaMemcpyDeviceToHost, s));
  CUPIQP_CUDA_CHECK(cudaStreamSynchronize(s));
  auto emit = [&](const char* op, double us) {
    std::printf("{\"label\":\"%s\",\"n\":%d,\"p\":%d,\"m\":%d,\"op\":\"%s\",\"graph_us\":%.2f,\"factor_status\":%d}\n",
                label.c_str(), maps.n, maps.p, maps.m, op, us, st);
    std::fflush(stdout);
  };
  emit("reduced_assemble", graph_us(s, reps, [&](cudaStream_t c) { red.assemble(c); }));
  emit("reduced_factor", graph_us(s, reps, [&](cudaStream_t c) { red.factor(c); }));
  emit("reduced_solve", graph_us(s, reps, [&](cudaStream_t c) { red.solve(c); }));
  emit("reduced_iter_1f2s", graph_us(s, reps, [&](cudaStream_t c) {
         red.factor(c);
         red.solve(c);
         red.solve(c);
       }));
  if (!with_cudss) return;
  Counters cnt;
  CudssSolver cu(&cnt);
  cu.init(dim, kp.nnz(), ip.data(), ix.data(), Kd.data(), false, s);
  cu.analyze(s);
  CUPIQP_CUDA_CHECK(cudaMemcpyAsync(cu.rhs(), b.data(), sizeof(double) * dim, cudaMemcpyHostToDevice, s));
  cu.factor(s);
  cu.solve(s);
  CUPIQP_CUDA_CHECK(cudaStreamSynchronize(s));
  emit("cudss_factor", graph_us(s, reps, [&](cudaStream_t c) { cu.factor(c); }));
  emit("cudss_solve", graph_us(s, reps, [&](cudaStream_t c) { cu.solve(c); }));
  emit("cudss_iter_1f2s", graph_us(s, reps, [&](cudaStream_t c) {
         cu.factor(c);
         cu.solve(c);
         cu.solve(c);
       }));
}

std::vector<int> parse_list(const std::string& v) {
  std::vector<int> out;
  std::stringstream ss(v);
  std::string t;
  while (std::getline(ss, t, ',')) out.push_back(std::stoi(t));
  return out;
}

// Synthetic KKT values in KktAssembly's layout (see test_reduced_kkt.cu).
void synthetic(int n, int reps, bool with_cudss, cudaStream_t s) {
  const int p = n / 4, m = 2 * n;
  std::mt19937 g(n);
  std::uniform_int_distribution<int> col(0, n - 1);
  std::uniform_real_distribution<double> u(-1, 1), pos(0.1, 2);
  auto rows = [&](int r, int per, bool sym) {
    std::vector<std::vector<int>> cols(r);
    for (int i = 0; i < r; ++i) {
      for (int k = 0; k < per; ++k) cols[i].push_back(col(g));
      if (sym) cols[i].push_back(i);
    }
    if (sym)
      for (int i = 0; i < r; ++i)
        for (int j : std::vector<int>(cols[i])) cols[j].push_back(i);
    CsrPattern c;
    c.rows = r;
    c.cols = n;
    c.indptr.push_back(0);
    for (auto& v : cols) {
      std::sort(v.begin(), v.end());
      v.erase(std::unique(v.begin(), v.end()), v.end());
      c.indices.insert(c.indices.end(), v.begin(), v.end());
      c.indptr.push_back(int32_t(c.indices.size()));
    }
    return c;
  };
  CsrPattern P = rows(n, 4, true), A = rows(p, 8, false), G = rows(m, 8, false);
  KktHostMaps maps = build_kkt_host_maps(P, A, G);
  std::vector<double> K(maps.kkt.nnz(), 0.0);
  for (int e : maps.P_slots) K[e] = 0.01 * u(g);
  for (int e : maps.A_slots) K[e] = u(g);
  for (int e : maps.G_slots) K[e] = u(g);
  for (int i = 0; i < n; ++i) K[maps.diag_x[i]] = 10.0 + pos(g);
  for (int e : maps.diag_y) K[e] = -1e-4;
  for (int e : maps.diag_z) K[e] = -pos(g);
  CudaBuffer<double> Kd(K.size());
  Kd.copy_from_host(K.data(), K.size(), s);
  time_state("synthetic", maps, Kd, reps, with_cudss, s);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    std::string data;
    std::vector<int> sweep;
    int c = 0, reps = 200;
    bool with_cudss = true;
    for (int i = 1; i < argc; ++i) {
      std::string a = argv[i];
      if (a == "--data") data = argv[++i];
      else if (a == "--case") c = std::stoi(argv[++i]);
      else if (a == "--reps") reps = std::stoi(argv[++i]);
      else if (a == "--sweep") sweep = parse_list(argv[++i]);
      else if (a == "--no-cudss") with_cudss = false;
      else throw std::runtime_error("unknown argument " + a);
    }
    CudaStream stream = CudaStream::create();
    cudaStream_t s = stream.get();
    for (int n : sweep) synthetic(n, reps, with_cudss, s);
    if (data.empty()) return 0;
    bench::Snapshots g = bench::load_snapshots(data);
    KktHostMaps maps = build_kkt_host_maps(g.P, g.A, g.C);
    CudaBuffer<double> packets(g.total), Kd(maps.kkt.nnz());
    packets.copy_from_host(g.case_values(c), g.total, s);
    CUPIQP_CUDA_CHECK(cudaStreamSynchronize(s));
    SparseSolver solver;
    auto& st = solver.settings();
    st.eps_abs = st.eps_duality_gap_abs = 1e-8;
    st.eps_rel = st.eps_duality_gap_rel = 0.0;
    st.kkt_backend = KktBackend::Reduced;
    solver.setup(g.P, g.view(packets.data()), &g.A, &g.C);
    solver.solve();
    const int N = solver.info().iter;
    const void* kv = nullptr;
    for (auto& d : solver.debug_buffers())
      if (d.name == "kkt.values") kv = d.ptr;
    for (int k : {N / 2, N}) {
      st.max_iter = k;
      solver.solve();
      CUPIQP_CUDA_CHECK(cudaMemcpyAsync(Kd.data(), kv, sizeof(double) * Kd.size(), cudaMemcpyDeviceToDevice,
                                        solver.stream()));
      CUPIQP_CUDA_CHECK(cudaStreamSynchronize(solver.stream()));
      time_state("case" + std::to_string(c) + "_k" + std::to_string(k), maps, Kd, reps, with_cudss, s);
    }
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
