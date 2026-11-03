// M5 cudss tests: CudssSolver against a dense CPU solve, analysis reuse,
// explicit zeros, graph capture vs eager, conditional bodies with the
// pageable-memcpy patch, and steady-state allocation counts.
#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <memory>
#include <random>
#include <vector>

#include "common/buffer.hpp"
#include "common/graph.hpp"
#include "common/handles.hpp"
#include "kkt_fixture.hpp"
#include "linalg/cudss/cudss_solver.hpp"

using namespace cupiqp;
using cupiqp_test::Kkt;

namespace {

__global__ void count_down(int32_t* counter, int32_t* cond) {
  int32_t v = --(*counter);
  *cond = v > 0 ? 1 : 0;
}
__global__ void set_condition(cudaGraphConditionalHandle h, const int32_t* cond) {
  cudaGraphSetConditional(h, *cond != 0 ? 1u : 0u);
}
__global__ void set_two(int32_t* p, int32_t a, int32_t b) {
  p[0] = a;
  p[1] = b;
}

// Device copy of a KKT plus a solver over it.
struct Rig {
  Kkt k;
  CudaStream stream = CudaStream::create();
  CudaBuffer<int32_t> indptr, indices;
  CudaBuffer<double> values;
  Counters counters;
  std::unique_ptr<CudssSolver> solver;

  Rig(Kkt kkt, bool deterministic) : k(std::move(kkt)) {
    cudaStream_t s = stream.get();
    indptr.allocate(k.indptr.size());
    indices.allocate(k.indices.size());
    values.allocate(k.values.size());
    indptr.copy_from_host(k.indptr.data(), k.indptr.size(), s);
    indices.copy_from_host(k.indices.data(), k.indices.size(), s);
    values.copy_from_host(k.values.data(), k.values.size(), s);
    solver = std::make_unique<CudssSolver>(&counters);
    solver->init(k.dim, k.nnz(), indptr.data(), indices.data(), values.data(), deterministic, s);
    sync();
  }
  cudaStream_t s() const { return stream.get(); }
  void sync() { CUPIQP_CUDA_CHECK(cudaStreamSynchronize(stream.get())); }
  void upload_values() {
    values.copy_from_host(k.values.data(), k.values.size(), s());
    sync();
  }
  std::vector<double> sol() {
    std::vector<double> x(k.dim);
    CUPIQP_CUDA_CHECK(cudaMemcpyAsync(x.data(), solver->sol(), sizeof(double) * k.dim, cudaMemcpyDeviceToHost, s()));
    sync();
    return x;
  }
};

std::vector<double> random_rhs(int dim, uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> uni(-1.0, 1.0);
  std::vector<double> b(dim);
  for (double& v : b) v = uni(rng);
  return b;
}

void upload_rhs(Rig& r, const std::vector<double>& b) {
  CUPIQP_CUDA_CHECK(cudaMemcpyAsync(r.solver->rhs(), b.data(), sizeof(double) * b.size(), cudaMemcpyHostToDevice, r.s()));
  r.sync();
}

// Kernel nodes inside the (first) conditional node's body of `g`.
int body_kernel_nodes(cudaGraph_t g) {
  size_t n = 0;
  CUPIQP_CUDA_CHECK(cudaGraphGetNodes(g, nullptr, &n));
  std::vector<cudaGraphNode_t> nodes(n);
  CUPIQP_CUDA_CHECK(cudaGraphGetNodes(g, nodes.data(), &n));
  for (auto nd : nodes) {
    cudaGraphNodeType t;
    CUPIQP_CUDA_CHECK(cudaGraphNodeGetType(nd, &t));
    if (t != cudaGraphNodeTypeConditional) continue;
    cudaGraphNodeParams p{};
    CUPIQP_CUDA_CHECK(cudaGraphNodeGetParams(nd, &p));
    cudaGraph_t body = p.conditional.phGraph_out[0];
    size_t m = 0;
    CUPIQP_CUDA_CHECK(cudaGraphGetNodes(body, nullptr, &m));
    std::vector<cudaGraphNode_t> bn(m);
    CUPIQP_CUDA_CHECK(cudaGraphGetNodes(body, bn.data(), &m));
    int kernels = 0;
    for (auto b : bn) {
      CUPIQP_CUDA_CHECK(cudaGraphNodeGetType(b, &t));
      kernels += t == cudaGraphNodeTypeKernel;
    }
    return kernels;
  }
  return -1;
}

double max_rel_err(const std::vector<double>& x, const std::vector<double>& ref) {
  double num = 0, den = 0;
  for (size_t i = 0; i < x.size(); ++i) {
    num = std::max(num, std::fabs(x[i] - ref[i]));
    den = std::max(den, std::fabs(ref[i]));
  }
  return num / den;
}

bool bitwise_equal(const std::vector<double>& a, const std::vector<double>& b) {
  return a.size() == b.size() && std::memcmp(a.data(), b.data(), sizeof(double) * a.size()) == 0;
}

Kkt small_kkt(uint64_t seed, int n = 60, int p = 15, int m = 90) {
  Kkt k = cupiqp_test::make_kkt(n, p, m, n * 8, p * 6, m * 10, seed);
  cupiqp_test::fill_values(k, seed + 1);
  return k;
}

}  // namespace

TEST(Cudss, SolvesRandomQuasiDefiniteSystems) {
  struct Case { int n, p, m; uint64_t seed; };
  for (Case c : {Case{30, 0, 0, 1}, Case{40, 10, 0, 2}, Case{40, 0, 70, 3}, Case{60, 15, 90, 4}, Case{120, 30, 200, 5}}) {
    Kkt k = cupiqp_test::make_kkt(c.n, c.p, c.m, c.n * 8, c.p * 6, c.m * 10, c.seed);
    cupiqp_test::fill_values(k, c.seed + 100);
    Rig r(k, false);
    std::vector<double> b = random_rhs(k.dim, c.seed + 200);
    upload_rhs(r, b);
    r.solver->analyze(r.s());
    r.solver->factor(r.s());
    r.solver->solve(r.s());
    std::vector<double> x = r.sol();
    std::vector<double> ref = cupiqp_test::dense_solve(cupiqp_test::dense_lower_sym(k), b);
    double rel = max_rel_err(x, ref), res = cupiqp_test::residual_inf(k, x, b);
    std::printf("[case n=%d p=%d m=%d nnz=%d] rel err vs dense %.3e, residual %.3e\n", c.n, c.p, c.m, k.nnz(), rel,
                res);
    EXPECT_LT(rel, 1e-10);
    EXPECT_LT(res, 1e-10);
  }
}

TEST(Cudss, UpperEntriesAreIgnored) {
  Kkt k = small_kkt(11);
  Rig r(k, false);
  std::vector<double> b = random_rhs(k.dim, 12);
  upload_rhs(r, b);
  r.solver->analyze(r.s());
  r.solver->factor(r.s());
  r.solver->solve(r.s());
  std::vector<double> x0 = r.sol();
  // Garbage in the stored upper part of P must not change the solution.
  for (int row = 0; row < k.n; ++row)
    for (int q = k.indptr[row]; q < k.indptr[row + 1]; ++q)
      if (k.indices[q] > row) r.k.values[q] = 1e3;
  r.upload_values();
  r.solver->factor(r.s());
  r.solver->solve(r.s());
  std::vector<double> x1 = r.sol();
  EXPECT_LT(max_rel_err(x1, x0), 1e-11);  // run-to-run spread is ~1e-13 (non-deterministic mode)
}

TEST(Cudss, RepeatedFactorReusesOneAnalysis) {
  Kkt k = small_kkt(21);
  Rig r(k, false);
  r.solver->analyze(r.s());
  for (int round = 0; round < 4; ++round) {
    cupiqp_test::fill_values(r.k, 1000 + round, 1e-6 * (round + 1), 1e-4 / (round + 1));
    r.upload_values();
    std::vector<double> b = random_rhs(k.dim, 2000 + round);
    upload_rhs(r, b);
    r.solver->factor(r.s());
    r.solver->solve(r.s());
    std::vector<double> x = r.sol();
    std::vector<double> ref = cupiqp_test::dense_solve(cupiqp_test::dense_lower_sym(r.k), b);
    EXPECT_LT(max_rel_err(x, ref), 1e-10) << "round " << round;
  }
  EXPECT_EQ(r.counters.cudss_analysis_calls, 1);
  EXPECT_EQ(r.counters.cudss_factor_calls, 4);
  EXPECT_EQ(r.counters.cudss_solve_calls, 4);
}

TEST(Cudss, ExplicitZerosThatBecomeNonzero) {
  Kkt k = small_kkt(31);
  // Zero every A/G entry and half of the P off-diagonals (stored, value 0).
  Kkt kz = k;
  int zeroed = 0;
  for (size_t i = 0; i < kz.offdiag_lower.size(); ++i) {
    int q = kz.offdiag_lower[i];
    int row = static_cast<int>(std::upper_bound(kz.indptr.begin(), kz.indptr.end(), q) - kz.indptr.begin()) - 1;
    int col = kz.indices[q];
    if (row >= kz.n || i % 2 == 0) {
      kz.values[q] = 0.0;
      if (row < kz.n) kz.values[cupiqp_test::slot_of(kz, col, row)] = 0.0;
      ++zeroed;
    }
  }
  ASSERT_GT(zeroed, 0);
  Rig r(kz, false);
  std::vector<double> b = random_rhs(k.dim, 32);
  upload_rhs(r, b);
  r.solver->analyze(r.s());  // analysis on the pattern with explicit zeros
  r.solver->factor(r.s());
  r.solver->solve(r.s());
  std::vector<double> x0 = r.sol();
  EXPECT_LT(max_rel_err(x0, cupiqp_test::dense_solve(cupiqp_test::dense_lower_sym(kz), b)), 1e-10);
  // Same slots, now nonzero: no new analysis.
  r.k = k;
  r.upload_values();
  r.solver->factor(r.s());
  r.solver->solve(r.s());
  std::vector<double> x1 = r.sol();
  EXPECT_LT(max_rel_err(x1, cupiqp_test::dense_solve(cupiqp_test::dense_lower_sym(k), b)), 1e-10);
  EXPECT_EQ(r.counters.cudss_analysis_calls, 1);
}

// cuDSS binds a data object to the stream of its ANALYSIS. The solver must
// still put FACTORIZATION/SOLVE on whatever stream the caller passes: work on
// another stream is ordered after that stream's prior work (eager), and a
// capture of another stream contains the cuDSS kernels.
TEST(Cudss, WorkFollowsTheCallerStream) {
  Kkt k = small_kkt(71, 120, 30, 200);
  Rig r(k, false);
  std::vector<double> b = random_rhs(k.dim, 72);
  upload_rhs(r, b);
  r.solver->analyze(r.s());
  r.solver->factor(r.s());
  r.solver->solve(r.s());
  r.sync();
  CudaStream other = CudaStream::create(cudaStreamNonBlocking);
  cudaStream_t o = other.get();
  // Eager: values and rhs written on `o` right before factor/solve on `o`.
  cupiqp_test::fill_values(r.k, 73);
  std::vector<double> b2 = random_rhs(k.dim, 74);
  r.values.copy_from_host(r.k.values.data(), r.k.values.size(), o);
  CUPIQP_CUDA_CHECK(cudaMemcpyAsync(r.solver->rhs(), b2.data(), sizeof(double) * k.dim, cudaMemcpyHostToDevice, o));
  r.solver->factor(o);
  r.solver->solve(o);
  std::vector<double> x(k.dim);
  CUPIQP_CUDA_CHECK(cudaMemcpyAsync(x.data(), r.solver->sol(), sizeof(double) * k.dim, cudaMemcpyDeviceToHost, o));
  CUPIQP_CUDA_CHECK(cudaStreamSynchronize(o));
  EXPECT_LT(max_rel_err(x, cupiqp_test::dense_solve(cupiqp_test::dense_lower_sym(r.k), b2)), 1e-10);
  // Capture on `o`: the graph holds the cuDSS kernels (7 + 13 nodes for this
  // size, at least a few kernels each).
  for (int phase = 0; phase < 2; ++phase) {
    cudaGraph_t g = capture(o, [&](cudaStream_t s) {
      if (phase == 0) r.solver->factor(s);
      else r.solver->solve(s);
    }, &r.counters);
    size_t n = 0;
    CUPIQP_CUDA_CHECK(cudaGraphGetNodes(g, nullptr, &n));
    EXPECT_GE(n, 5u) << (phase == 0 ? "factor" : "solve");
    CUPIQP_CUDA_CHECK(cudaGraphDestroy(g));
  }
}

// Eager repeatability (reported, default mode): factor + solve repeated on the
// same values, and solve repeated on one factorization.
TEST(Cudss, EagerRepeatability) {
  Kkt k = small_kkt(41, 120, 30, 200);
  Rig r(k, false);
  upload_rhs(r, random_rhs(k.dim, 42));
  r.solver->analyze(r.s());
  std::vector<std::vector<double>> xs, ys;
  for (int i = 0; i < 5; ++i) {
    r.solver->factor(r.s());
    r.solver->solve(r.s());
    xs.push_back(r.sol());
  }
  for (int i = 0; i < 5; ++i) {
    r.solver->solve(r.s());
    ys.push_back(r.sol());
  }
  int same_fs = 0, same_s = 0;
  double spread = 0;
  for (int i = 1; i < 5; ++i) {
    same_fs += bitwise_equal(xs[i], xs[0]);
    same_s += bitwise_equal(ys[i], ys[0]);
    spread = std::max(spread, max_rel_err(xs[i], xs[0]));
  }
  double spread_s = 0;
  for (int i = 1; i < 5; ++i) spread_s = std::max(spread_s, max_rel_err(ys[i], ys[0]));
  std::printf("[default mode] bitwise repeats: factor+solve %d of 4 (max rel spread %.3e), "
              "solve only %d of 4 (max rel spread %.3e)\n", same_fs, spread, same_s, spread_s);
  EXPECT_LT(spread, 1e-11);
  EXPECT_LT(spread_s, 1e-11);
}

// Plain-graph capture of FACTORIZATION + SOLVE, replayed after in-place value
// updates (stable addresses): matches the dense solve of the new values and an
// eager factor + solve. Bitwise equality is not available in the default mode
// (cuDSS is not run-to-run reproducible there, see EagerRepeatability) and
// DETERMINISTIC_MODE faults in cuDSS 0.8.0.10 (see the disabled test below).
TEST(Cudss, CaptureReplayMatchesEager) {
  Kkt k = small_kkt(51, 120, 30, 200);
  Rig r(k, false);
  std::vector<double> b = random_rhs(k.dim, 52);
  upload_rhs(r, b);
  r.solver->analyze(r.s());
  r.solver->factor(r.s());
  r.solver->solve(r.s());
  r.sync();
  int64_t f0 = r.counters.cudss_factor_calls, s0 = r.counters.cudss_solve_calls;
  GraphExec fs, so;
  fs.instantiate(capture(r.s(), [&](cudaStream_t s) {
    r.solver->factor(s);
    r.solver->solve(s);
  }, &r.counters));
  so.instantiate(capture(r.s(), [&](cudaStream_t s) { r.solver->solve(s); }, &r.counters));
  EXPECT_EQ(r.counters.cudss_factor_calls, f0 + 1);  // a capture counts as a host-side call
  EXPECT_EQ(r.counters.cudss_solve_calls, s0 + 2);
  for (int round = 0; round < 3; ++round) {
    cupiqp_test::fill_values(r.k, 3000 + round);
    r.upload_values();
    CUPIQP_CUDA_CHECK(cudaMemsetAsync(r.solver->sol(), 0, sizeof(double) * k.dim, r.s()));
    fs.launch(r.s(), &r.counters);
    std::vector<double> graph = r.sol();
    std::vector<double> ref = cupiqp_test::dense_solve(cupiqp_test::dense_lower_sym(r.k), b);
    double rel = max_rel_err(graph, ref);
    EXPECT_LT(rel, 1e-10) << "round " << round;
    r.solver->factor(r.s());
    r.solver->solve(r.s());
    std::vector<double> eager = r.sol();
    EXPECT_LT(max_rel_err(graph, eager), 1e-11) << "round " << round;
    CUPIQP_CUDA_CHECK(cudaMemsetAsync(r.solver->sol(), 0, sizeof(double) * k.dim, r.s()));
    so.launch(r.s(), &r.counters);
    EXPECT_LT(max_rel_err(r.sol(), eager), 1e-11) << "round " << round;
    std::printf("[round %d] factor+solve graph: rel err vs dense %.3e, vs eager %.3e\n", round, rel,
                max_rel_err(graph, eager));
  }
}

// cuDSS 0.8.0.10 DETERMINISTIC_MODE: SOLVE intermittently faults with an
// out-of-bounds shared-memory read in cudss::fwd_dtmn_ker (compute-sanitizer),
// with or without this module's memory handler and threading layer, and the
// reference Python path shows the same fault. The fault is sticky (it poisons
// the CUDA context), so the test is disabled; run it explicitly with
// --gtest_also_run_disabled_tests --gtest_filter='*Deterministic*'.
TEST(Cudss, DISABLED_DeterministicModeRepeatsAndCapturesBitwise) {
  Kkt k = small_kkt(55, 120, 30, 200);
  Rig r(k, true);
  std::vector<double> b = random_rhs(k.dim, 56);
  upload_rhs(r, b);
  r.solver->analyze(r.s());
  r.solver->factor(r.s());
  r.solver->solve(r.s());
  std::vector<double> x0 = r.sol();
  GraphExec fs;
  fs.instantiate(capture(r.s(), [&](cudaStream_t s) {
    r.solver->factor(s);
    r.solver->solve(s);
  }, &r.counters));
  for (int i = 0; i < 3; ++i) {
    r.solver->factor(r.s());
    r.solver->solve(r.s());
    EXPECT_TRUE(bitwise_equal(r.sol(), x0));
    CUPIQP_CUDA_CHECK(cudaMemsetAsync(r.solver->sol(), 0, sizeof(double) * k.dim, r.s()));
    fs.launch(r.s(), &r.counters);
    EXPECT_TRUE(bitwise_equal(r.sol(), x0));
  }
}

// Conditional WHILE bodies (FEASIBILITY 3.2/3.3).
//  * FACTORIZATION in a body: legal without any patch.
//  * SOLVE in a body: cudaGraphInstantiate rejects the two pageable-source
//    memcpy nodes; patch_pageable_memcpy (applied to the top-level graph,
//    recursing into the body) makes it legal, and replays match eager solves,
//    also after value updates + eager refactorizations between replays, with
//    the same snapshot bytes in two independent captures.
// Both with common/capture_while (body captured on its non-blocking helper
// stream) and with an explicit node whose body is captured on a blocking
// stream (the PoC). In both cases the cuDSS work must land IN the body, which
// requires the solver's fork/join onto its analysis-bound stream (see
// cudss_solver.hpp); the node counts below check that.
TEST(Cudss, ConditionalBodiesAndPageablePatch) {
  Kkt k = small_kkt(61, 120, 30, 200);
  Rig r(k, false);
  std::vector<double> b = random_rhs(k.dim, 62);
  upload_rhs(r, b);
  r.solver->analyze(r.s());

  LoopContext ctx(&r.counters);
  CudaBuffer<int32_t> loop(2);  // [counter, cond]
  int32_t* counter = loop.data();
  int32_t* cond = loop.data() + 1;
  auto body = [&](bool do_solve, cudaStream_t s) {
    if (do_solve) {
      CUPIQP_CUDA_CHECK(cudaMemsetAsync(r.solver->sol(), 0, sizeof(double) * r.k.dim, s));
      r.solver->solve(s);
    } else {
      r.solver->factor(s);
    }
    count_down<<<1, 1, 0, s>>>(counter, cond);
    CUPIQP_CUDA_CHECK_LAUNCH();
  };
  // style 0: common/capture_while (body on its non-blocking helper stream).
  // style 1: PoC style, explicit WHILE node (default condition 1), body
  //          captured on a blocking stream.
  CudaStream blocking = CudaStream::create(cudaStreamDefault);
  auto build = [&](int style, bool do_solve) -> cudaGraph_t {
    if (style == 0)
      return capture(r.s(), [&](cudaStream_t s) {
        capture_while(ctx, s, cond, [&](cudaStream_t h) { body(do_solve, h); });
      }, &r.counters);
    cudaGraph_t g;
    CUPIQP_CUDA_CHECK(cudaGraphCreate(&g, 0));
    cudaGraphConditionalHandle h;
    CUPIQP_CUDA_CHECK(cudaGraphConditionalHandleCreate(&h, g, 1, cudaGraphCondAssignDefault));
    cudaGraphNodeParams np{};
    np.type = cudaGraphNodeTypeConditional;
    np.conditional.handle = h;
    np.conditional.type = cudaGraphCondTypeWhile;
    np.conditional.size = 1;
    cudaGraphNode_t node;
    CUPIQP_CUDA_CHECK(cudaGraphAddNode(&node, g, nullptr, nullptr, 0, &np));
    cudaStream_t bs = blocking.get();
    CUPIQP_CUDA_CHECK(cudaStreamBeginCaptureToGraph(bs, np.conditional.phGraph_out[0], nullptr, nullptr, 0,
                                                    cudaStreamCaptureModeThreadLocal));
    body(do_solve, bs);
    set_condition<<<1, 1, 0, bs>>>(h, cond);
    CUPIQP_CUDA_CHECK_LAUNCH();
    cudaGraph_t out;
    CUPIQP_CUDA_CHECK(cudaStreamEndCapture(bs, &out));
    return g;
  };
  auto run = [&](GraphExec& g, int iters) {
    set_two<<<1, 1, 0, r.s()>>>(loop.data(), iters, 1);
    CUPIQP_CUDA_CHECK_LAUNCH();
    g.launch(r.s(), &r.counters);
    return r.sol();
  };
  auto eager_factor_solve = [&]() {
    r.solver->factor(r.s());
    r.solver->solve(r.s());
    return r.sol();
  };

  for (int style = 0; style < 2; ++style) {
    SCOPED_TRACE(style == 0 ? "capture_while (non-blocking helper)" : "explicit node, blocking stream");
    // FACTORIZATION body, no patch; then an eager SOLVE on its factors.
    std::vector<double> eager = eager_factor_solve();
    {
      GraphExec g;
      g.instantiate(build(style, false));
      EXPECT_GE(body_kernel_nodes(g.graph()), 3) << "cuDSS factorization kernels must be inside the body";
      run(g, 3);
      r.solver->solve(r.s());
      EXPECT_LT(max_rel_err(r.sol(), eager), 1e-11);
    }
    // SOLVE body without the patch.
    {
      GraphExec g;
      EXPECT_THROW(g.instantiate(build(style, true)), Error);
      cudaGetLastError();
    }
    // SOLVE body with the patch, two independent captures.
    std::vector<std::vector<unsigned char>> snapshots[2];
    for (int round = 0; round < 2; ++round) {
      std::vector<CudaPinnedBuffer<unsigned char>> keep;
      int patched = -1;
      GraphExec g;
      g.instantiate(build(style, true), [&](cudaGraph_t graph) { patched = patch_pageable_memcpy(graph, keep); });
      EXPECT_EQ(patched, 2);
      for (auto& kb : keep) snapshots[round].emplace_back(kb.data(), kb.data() + kb.size());
      EXPECT_LT(max_rel_err(run(g, 3), eager), 1e-11) << "round " << round;
      // New values + eager refactorization, then replay the same graph.
      cupiqp_test::fill_values(r.k, 70 + 2 * style + round);
      r.upload_values();
      eager = eager_factor_solve();
      std::vector<double> ref = cupiqp_test::dense_solve(cupiqp_test::dense_lower_sym(r.k), b);
      std::vector<double> x = run(g, 2);
      EXPECT_LT(max_rel_err(x, eager), 1e-11) << "replay after update, round " << round;
      EXPECT_LT(max_rel_err(x, ref), 1e-10) << "replay after update, round " << round;
    }
    EXPECT_EQ(snapshots[0], snapshots[1]);
  }
}

// Steady state: no nonzero cuDSS device requests after warm-up, eager or in
// graphs. Zero-byte requests (no memory) are reported separately.
TEST(Cudss, AllocationCountsAfterWarmup) {
  Kkt k = small_kkt(81, 120, 30, 200);
  Rig r(k, false);
  upload_rhs(r, random_rhs(k.dim, 82));
  const CudssMemory& mem = r.solver->memory();
  struct Snap { int64_t a, f, z; };
  auto snap = [&]() { return Snap{r.counters.cudss_device_allocs, r.counters.cudss_device_frees, mem.zero_byte_requests()}; };
  std::vector<std::pair<const char*, Snap>> rows;
  Snap prev = snap();
  auto mark = [&](const char* what) {
    r.sync();
    Snap now = snap();
    rows.push_back({what, Snap{now.a - prev.a, now.f - prev.f, now.z - prev.z}});
    prev = now;
  };
  mark("init");
  r.solver->analyze(r.s());
  mark("analysis");
  r.solver->factor(r.s());
  mark("first factor");
  r.solver->solve(r.s());
  mark("first solve");
  for (int i = 0; i < 10; ++i) r.solver->factor(r.s());
  mark("10 eager factor");
  for (int i = 0; i < 10; ++i) r.solver->solve(r.s());
  mark("10 eager solve");
  GraphExec g;
  g.instantiate(capture(r.s(), [&](cudaStream_t s) {
    r.solver->factor(s);
    r.solver->solve(s);
  }, &r.counters));
  mark("capture factor+solve");
  for (int i = 0; i < 10; ++i) g.launch(r.s(), &r.counters);
  mark("10 graph replays");
  // Captured graphs contain no memory alloc/free nodes.
  size_t n = 0;
  CUPIQP_CUDA_CHECK(cudaGraphGetNodes(g.graph(), nullptr, &n));
  std::vector<cudaGraphNode_t> nodes(n);
  CUPIQP_CUDA_CHECK(cudaGraphGetNodes(g.graph(), nodes.data(), &n));
  int mem_nodes = 0;
  for (auto nd : nodes) {
    cudaGraphNodeType t;
    CUPIQP_CUDA_CHECK(cudaGraphNodeGetType(nd, &t));
    mem_nodes += (t == cudaGraphNodeTypeMemAlloc || t == cudaGraphNodeTypeMemFree);
  }
  for (auto& [what, d] : rows)
    std::printf("  %-22s allocs %lld frees %lld zero-byte requests %lld\n", what, (long long)d.a, (long long)d.f,
                (long long)d.z);
  std::printf("  peak live %zu B, largest request %zu B, live now %zu B, graph nodes %zu (mem alloc/free %d)\n",
              mem.peak_live_bytes(), mem.largest_request(), mem.live_bytes(), n, mem_nodes);
  for (size_t i = 4; i < rows.size(); ++i) {
    EXPECT_EQ(rows[i].second.a, 0) << rows[i].first;
    EXPECT_EQ(rows[i].second.f, 0) << rows[i].first;
  }
  EXPECT_EQ(mem_nodes, 0);
}
