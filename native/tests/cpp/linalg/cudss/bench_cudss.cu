// M5 cudss micro-benchmark (not a test): FACTORIZATION and SOLVE timings,
// eager vs CUDA graph, on the MPC KKT shape
//   n = 169 (P: 8977 stored entries, full storage), A: 64 x 169 (920),
//   G: 2521 x 169 (249801); dim 2754.
// The pattern is random with these counts (not the real MPC pattern).
//
// For each variant: `reps` repetitions of `iters` back-to-back calls on one
// stream, timed by CUDA events (per-call GPU time = elapsed / iters) and by the
// host clock around the enqueue loop (per-call enqueue time). Prints the
// median and min/max over repetitions.
//
// Usage: bench_cudss [iters=50] [reps=7] [matrix.bin]
// matrix.bin (optional): int32 dim, int32 nnz, int32 indptr[dim + 1],
// int32 indices[nnz], float64 values[nnz] (lower view, full P stored), e.g.
// written by a script to time the native wrapper on the exact matrix the
// reference is timed on.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <vector>

#include "common/buffer.hpp"
#include "common/graph.hpp"
#include "common/handles.hpp"
#include "kkt_fixture.hpp"
#include "linalg/cudss/cudss_solver.hpp"

using namespace cupiqp;

namespace {

struct Stat {
  double med, lo, hi;
};
Stat stats(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return {v[v.size() / 2], v.front(), v.back()};
}

}  // namespace

int main(int argc, char** argv) {
  const int iters = argc > 1 ? std::atoi(argv[1]) : 50;
  const int reps = argc > 2 ? std::atoi(argv[2]) : 7;
  cupiqp_test::Kkt k;
  if (argc > 3) {
    std::FILE* f = std::fopen(argv[3], "rb");
    if (!f) {
      std::fprintf(stderr, "cannot open %s\n", argv[3]);
      return 1;
    }
    int32_t hdr[2];
    bool ok = std::fread(hdr, sizeof(int32_t), 2, f) == 2;
    k.dim = hdr[0];
    k.indptr.resize(k.dim + 1);
    k.indices.resize(hdr[1]);
    k.values.resize(hdr[1]);
    ok = ok && std::fread(k.indptr.data(), sizeof(int32_t), k.indptr.size(), f) == k.indptr.size();
    ok = ok && std::fread(k.indices.data(), sizeof(int32_t), k.indices.size(), f) == k.indices.size();
    ok = ok && std::fread(k.values.data(), sizeof(double), k.values.size(), f) == k.values.size();
    std::fclose(f);
    if (!ok) {
      std::fprintf(stderr, "short read from %s\n", argv[3]);
      return 1;
    }
  } else {
    k = cupiqp_test::make_kkt(169, 64, 2521, 8977, 920, 249801, 2024);
    cupiqp_test::fill_values(k, 2025);
  }
  CudaStream stream = CudaStream::create();
  cudaStream_t s = stream.get();
  CudaBuffer<int32_t> indptr(k.indptr.size()), indices(k.indices.size());
  CudaBuffer<double> values(k.values.size());
  indptr.copy_from_host(k.indptr.data(), k.indptr.size(), s);
  indices.copy_from_host(k.indices.data(), k.indices.size(), s);
  values.copy_from_host(k.values.data(), k.values.size(), s);
  Counters counters;
  CudssSolver solver(&counters);
  solver.init(k.dim, k.nnz(), indptr.data(), indices.data(), values.data(), false, s);
  std::vector<double> b(k.dim, 1.0);
  CUPIQP_CUDA_CHECK(cudaMemcpyAsync(solver.rhs(), b.data(), sizeof(double) * k.dim, cudaMemcpyHostToDevice, s));
  CUPIQP_CUDA_CHECK(cudaStreamSynchronize(s));

  auto t0 = std::chrono::steady_clock::now();
  solver.analyze(s);
  CUPIQP_CUDA_CHECK(cudaStreamSynchronize(s));
  double analysis_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  solver.factor(s);
  solver.solve(s);
  CUPIQP_CUDA_CHECK(cudaStreamSynchronize(s));
  std::vector<double> x(k.dim);
  CUPIQP_CUDA_CHECK(cudaMemcpy(x.data(), solver.sol(), sizeof(double) * k.dim, cudaMemcpyDeviceToHost));
  std::printf("cuDSS bench: dim %d, nnz %d (lower+full P), analysis %.2f ms (host wall), residual %.3e\n", k.dim,
              k.nnz(), analysis_ms, cupiqp_test::residual_inf(k, x, b));

  GraphExec g_factor, g_solve, g_both;
  g_factor.instantiate(capture(s, [&](cudaStream_t c) { solver.factor(c); }, &counters));
  g_solve.instantiate(capture(s, [&](cudaStream_t c) { solver.solve(c); }, &counters));
  g_both.instantiate(capture(s, [&](cudaStream_t c) {
    solver.factor(c);
    solver.solve(c);
  }, &counters));

  CudaEvent e0 = CudaEvent::create(true), e1 = CudaEvent::create(true);
  int64_t allocs0 = counters.cudss_device_allocs, frees0 = counters.cudss_device_frees;
  auto bench = [&](const char* name, const std::function<void()>& call) {
    for (int i = 0; i < 5; ++i) call();  // warm-up
    CUPIQP_CUDA_CHECK(cudaStreamSynchronize(s));
    std::vector<double> gpu, host;
    for (int r = 0; r < reps; ++r) {
      e0.record(s);
      auto h0 = std::chrono::steady_clock::now();
      for (int i = 0; i < iters; ++i) call();
      auto h1 = std::chrono::steady_clock::now();
      e1.record(s);
      CUPIQP_CUDA_CHECK(cudaEventSynchronize(e1.get()));
      float ms = 0;
      CUPIQP_CUDA_CHECK(cudaEventElapsedTime(&ms, e0.get(), e1.get()));
      gpu.push_back(1e3 * ms / iters);
      host.push_back(std::chrono::duration<double, std::micro>(h1 - h0).count() / iters);
    }
    Stat g = stats(gpu), h = stats(host);
    std::printf("  %-26s per call: stream time %8.1f us (min %8.1f, max %8.1f)   host enqueue %7.1f us (min %7.1f, max %7.1f)\n",
                name, g.med, g.lo, g.hi, h.med, h.lo, h.hi);
  };
  std::printf("%d reps x %d calls, medians:\n", reps, iters);
  bench("eager factor", [&] { solver.factor(s); });
  bench("graph factor", [&] { g_factor.launch(s, &counters); });
  bench("eager solve", [&] { solver.solve(s); });
  bench("graph solve", [&] { g_solve.launch(s, &counters); });
  bench("eager factor+solve", [&] {
    solver.factor(s);
    solver.solve(s);
  });
  bench("graph factor+solve", [&] { g_both.launch(s, &counters); });
  std::printf("cuDSS nonzero device requests during the timed loops: allocs %lld, frees %lld; zero-byte requests total %lld\n",
              (long long)(counters.cudss_device_allocs - allocs0), (long long)(counters.cudss_device_frees - frees0),
              (long long)solver.memory().zero_byte_requests());
  std::printf("cuDSS device memory held: peak %.2f MiB, largest request %.2f MiB\n",
              solver.memory().peak_live_bytes() / 1048576.0, solver.memory().largest_request() / 1048576.0);
  return 0;
}
