// Native module micro-benchmarks on a real snapshot pattern (MPC or Trajopt).
//
// Isolates the pieces of one update and one IPM iteration with the internal
// module APIs (the same objects the solver uses, built in the solver's setup
// order): numeric value update, Ruiz scaling (fresh 10 passes), KKT scatter,
// KKT diagonal update + factorization, the reduced KKT factorization alone,
// its solve alone, the condensed KKT solve, the five SpMV products, and the
// public update() / solve(). Every operation is timed
// eagerly (CUDA events around each call, plus host enqueue time) and, where it
// is capturable, as a replayed CUDA graph. Median and P95 over --reps calls.
//
// Usage: cupiqp_native_module_bench --data native/bench/data/mpc/group_0.bin
//          [--reps 200] [--out results.json]
#include <cuda_runtime.h>
#include <time.h>

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <functional>
#include <string>
#include <vector>

#include "common/buffer.hpp"
#include "common/graph.hpp"
#include "common/handles.hpp"
#include "common/views.hpp"
#include "cupiqp/cupiqp.hpp"
#include "linalg/kkt_assembly/kkt_assembly.hpp"
#include "linalg/kkt_system/kkt_system.hpp"
#include "linalg/reduced_kkt/reduced_kkt_solver.hpp"
#include "linalg/spmv/spmv.hpp"
#include "problem/data/sparse_data.hpp"
#include "problem/ruiz/ruiz.hpp"
#include "snapshot_io.hpp"

using namespace cupiqp;

namespace {

double raw_ms() {
  timespec ts;
  clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
  return ts.tv_sec * 1e3 + ts.tv_nsec * 1e-6;
}

struct Stat {
  double p50 = 0, p95 = 0;
};
Stat stat(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  auto q = [&](double f) { return v[std::min(v.size() - 1, size_t(f * (v.size() - 1) + 0.5))]; };
  return {q(0.5), q(0.95)};
}

struct Result {
  std::string name;
  Stat eager_gpu, eager_host, graph_gpu;
  bool graph = false;
};

__global__ void fill_kernel(double* p, int64_t n, double v) {
  int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
  if (i < n) p[i] = v;
}
void fill(cudaStream_t s, double* p, int64_t n, double v) {
  if (n > 0) fill_kernel<<<unsigned((n + 255) / 256), 256, 0, s>>>(p, n, v);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    std::string data_path, out_path;
    int reps = 200;
    for (int i = 1; i < argc; ++i) {
      std::string a = argv[i];
      if (a == "--data") data_path = argv[++i];
      else if (a == "--reps") reps = std::stoi(argv[++i]);
      else if (a == "--out") out_path = argv[++i];
      else throw std::runtime_error("unknown argument " + a);
    }
    if (data_path.empty()) throw std::runtime_error("--data is required");
    bench::Snapshots g = bench::load_snapshots(data_path);
    CudaStream stream = CudaStream::create();
    cudaStream_t s = stream.get();
    Counters counters;
    LoopContext loops(&counters);

    // Packed snapshot values on the device (two cases: alternate between them).
    CudaBuffer<double> packets(g.total * 2);
    packets.copy_from_host(g.case_values(0), g.total, s);
    CUPIQP_CUDA_CHECK(cudaMemcpyAsync(packets.data() + g.total, g.case_values(std::min(1, g.cases - 1)),
                                      sizeof(double) * g.total, cudaMemcpyHostToDevice, s));
    auto view_of = [&](int k) { return g.view(packets.data() + g.total * k); };

    // Modules, in the solver's setup order.
    Settings settings;
    SparseData data;
    data.init(g.P, &g.A, &g.C, view_of(0), s);
    const Dims d = data.dims();
    auto dv = data.view();
    RuizEquilibration ruiz;
    ruiz.init(dv, s);
    ruiz.scale_data(dv, false, 10, s);
    SparseOperators ops;
    ops.init(dv);
    KktAssembly kkt;
    kkt.init(data.P_pattern(), data.A_pattern(), data.G_pattern(), dv, s);
    ReducedKktSolver reduced;
    reduced.init(kkt, s);

    // Device settings (reference DeviceSettings values) and solver-like state.
    CudaBuffer<double> sf(SF_COUNT);
    CudaBuffer<int32_t> si(SI_COUNT);
    double hf[SF_COUNT] = {settings.rho_init, settings.delta_init, 1e-8, 0.0, 1e-8, 0.0,
                           settings.infeasibility_threshold, settings.reg_lower_limit,
                           settings.reg_finetune_lower_limit, settings.tau,
                           settings.iterative_refinement_eps_abs, settings.iterative_refinement_eps_rel,
                           settings.iterative_refinement_min_improvement_rate};
    int32_t hi[SI_COUNT] = {1, 7, 7, 250, 10, 10};
    sf.copy_from_host(hf, SF_COUNT, s);
    si.copy_from_host(hi, SI_COUNT, s);
    KktSystem kkt_sys;
    kkt_sys.init(d, settings, &kkt, &reduced, &ops, {sf.data(), si.data()}, &loops, s);

    CudaBuffer<double> vp(size_t(d.num_primal())), vd(size_t(d.num_dual())), rp(vp.size()), rd(vd.size()),
        lp(vp.size()), ld(vd.size()), rho(1), delta(1), xbuf(d.n), ybuf(std::max(d.n, d.m));
    VariablesView<double> vars{d, vp.data(), vd.data()}, rhs{d, rp.data(), rd.data()}, lhs{d, lp.data(), ld.data()};
    fill(s, vp.data(), d.n, 0.0);
    fill(s, vp.data() + d.n, d.num_ineq(), 1.0);
    fill(s, vd.data(), d.p, 0.0);
    fill(s, vd.data() + d.p, d.num_ineq(), 1.0);
    fill(s, rp.data(), vp.size(), 0.5);
    fill(s, rd.data(), vd.size(), -0.25);
    fill(s, rho.data(), 1, 1e-6);
    fill(s, delta.data(), 1, 1e-4);
    fill(s, xbuf.data(), d.n, 1.0);
    fill(s, ybuf.data(), ybuf.size(), 1.0);
    CUPIQP_CUDA_CHECK(cudaStreamSynchronize(s));

    // ------------------------------------------------------------------ timing harness
    CudaEvent e0 = CudaEvent::create(true), e1 = CudaEvent::create(true);
    std::vector<Result> results;
    auto measure = [&](const std::string& name, const std::function<void(cudaStream_t)>& op, bool capturable) {
      Result r;
      r.name = name;
      for (int w = 0; w < 5; ++w) op(s);
      CUPIQP_CUDA_CHECK(cudaStreamSynchronize(s));
      std::vector<double> gpu, host;
      for (int i = 0; i < reps; ++i) {
        e0.record(s);
        double t0 = raw_ms();
        op(s);
        double t1 = raw_ms();
        e1.record(s);
        CUPIQP_CUDA_CHECK(cudaEventSynchronize(e1.get()));
        float ms = 0;
        CUPIQP_CUDA_CHECK(cudaEventElapsedTime(&ms, e0.get(), e1.get()));
        gpu.push_back(ms);
        host.push_back(t1 - t0);
      }
      r.eager_gpu = stat(gpu);
      r.eager_host = stat(host);
      if (capturable) {
        GraphExec ge;
        ge.instantiate(capture(s, op, &counters));
        for (int w = 0; w < 5; ++w) ge.launch(s, &counters);
        CUPIQP_CUDA_CHECK(cudaStreamSynchronize(s));
        std::vector<double> gg;
        for (int i = 0; i < reps; ++i) {
          e0.record(s);
          ge.launch(s, &counters);
          e1.record(s);
          CUPIQP_CUDA_CHECK(cudaEventSynchronize(e1.get()));
          float ms = 0;
          CUPIQP_CUDA_CHECK(cudaEventElapsedTime(&ms, e0.get(), e1.get()));
          gg.push_back(ms);
        }
        r.graph_gpu = stat(gg);
        r.graph = true;
      }
      results.push_back(r);
      std::fprintf(stderr, "%-44s eager %8.3f / %8.3f ms (host %7.3f)%s\n", name.c_str(), r.eager_gpu.p50,
                   r.eager_gpu.p95, r.eager_host.p50,
                   capturable ? (" | graph " + std::to_string(r.graph_gpu.p50).substr(0, 7) + " ms").c_str() : "");
    };

    int flip = 0;
    measure("update: copy values + finite masks", [&](cudaStream_t st) { data.set_values(view_of(flip ^= 1), st); }, true);
    measure("update: Ruiz unscale(skip replaced) + reset + 10 passes",
            [&](cudaStream_t st) {
              data.set_values(view_of(flip ^= 1), st);
              ruiz.unscale_data(dv, UpdateMask::all(), st);
              ruiz.reset(st);
              ruiz.scale_data(dv, false, 10, st);
            },
            true);
    measure("update: KKT scatter P, A, G (masked)", [&](cudaStream_t st) { kkt.scatter(dv, true, true, true, st); }, true);
    measure("iteration: regularization + KKT diag + factor",
            [&](cudaStream_t st) {
              kkt_sys.update_scalings_and_factor(dv, ruiz.view(), false, rho.data(), delta.data(), vars, st);
            },
            true);
    measure("iteration: reduced KKT factor only", [&](cudaStream_t st) { reduced.factor(st); }, true);
    measure("iteration: reduced KKT solve only", [&](cudaStream_t st) { reduced.solve(st); }, true);
    measure("iteration: condensed KKT solve (elim + solve + recover)",
            [&](cudaStream_t st) { kkt_sys.solve(dv, ruiz.view(), rhs, lhs, st); }, true);
    measure("spmv: P x", [&](cudaStream_t st) { ops.P(st, 1.0, xbuf.data(), ybuf.data()); }, true);
    measure("spmv: A x", [&](cudaStream_t st) { ops.A(st, 1.0, xbuf.data(), ybuf.data()); }, true);
    measure("spmv: A^T y", [&](cudaStream_t st) { ops.AT(st, 1.0, ybuf.data(), xbuf.data()); }, true);
    measure("spmv: G x", [&](cudaStream_t st) { ops.G(st, 1.0, xbuf.data(), ybuf.data()); }, true);
    measure("spmv: G^T z", [&](cudaStream_t st) { ops.GT(st, 1.0, ybuf.data(), xbuf.data()); }, true);

    // Public API: one update (all values) and one solve, default graph mode.
    SparseSolver solver(s);
    solver.settings().eps_abs = solver.settings().eps_duality_gap_abs = 1e-8;
    solver.settings().eps_rel = solver.settings().eps_duality_gap_rel = 0.0;
    solver.setup(g.P, view_of(0), &g.A, &g.C);
    solver.solve();
    measure("public: SparseSolver::update (all values)", [&](cudaStream_t) { solver.update(view_of(flip ^= 1)); },
            false);
    measure("public: SparseSolver::solve (default graph mode)", [&](cudaStream_t) { solver.solve(); }, false);

    if (!out_path.empty()) {
      std::ofstream f(out_path);
      f << "{\"data\":\"" << data_path << "\",\"reps\":" << reps << ",\"n\":" << d.n << ",\"p\":" << d.p
        << ",\"m\":" << d.m << ",\"kkt_dim\":" << kkt.dim() << ",\"kkt_nnz\":" << kkt.nnz() << ",\"ops\":[";
      for (size_t i = 0; i < results.size(); ++i) {
        const auto& r = results[i];
        f << (i ? "," : "") << "{\"name\":\"" << r.name << "\",\"eager_gpu_ms\":{\"p50\":" << r.eager_gpu.p50
          << ",\"p95\":" << r.eager_gpu.p95 << "},\"eager_host_enqueue_ms\":{\"p50\":" << r.eager_host.p50
          << ",\"p95\":" << r.eager_host.p95 << "}";
        if (r.graph) f << ",\"graph_gpu_ms\":{\"p50\":" << r.graph_gpu.p50 << ",\"p95\":" << r.graph_gpu.p95 << "}";
        f << "}";
      }
      f << "]}\n";
    }
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
