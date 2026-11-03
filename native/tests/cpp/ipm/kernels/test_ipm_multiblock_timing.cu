// Timing of the multi-block reduction kernels at the MPC and Trajopt shapes
// (B = 1) for several blocks-per-problem values. Disabled by default; run
// with --gtest_also_run_disabled_tests. Each launch is timed inside a CUDA
// graph of kLaunches launches (as in the solver's iteration graph).
#include <gtest/gtest.h>

#include <cstdio>
#include <random>
#include <vector>

#include "common/buffer.hpp"
#include "common/graph.hpp"
#include "common/handles.hpp"
#include "common/multiblock.cuh"
#include "common/multiblock.hpp"
#include "ipm/kernels/ipm_kernels.hpp"

using namespace cupiqp;

namespace {

constexpr int kLaunches = 200;

template <class F>
double time_us(cudaStream_t s, F&& launch) {
  cudaGraph_t g = capture(
      s,
      [&](cudaStream_t cs) {
        for (int i = 0; i < kLaunches; ++i) launch(cs);
      },
      nullptr);
  cudaGraphExec_t e = nullptr;
  CUPIQP_CUDA_CHECK(cudaGraphInstantiate(&e, g, 0));
  CudaEvent a = CudaEvent::create(true), b = CudaEvent::create(true);
  CUPIQP_CUDA_CHECK(cudaGraphLaunch(e, s));  // warm-up
  float best = 1e30f;
  for (int rep = 0; rep < 5; ++rep) {
    a.record(s);
    CUPIQP_CUDA_CHECK(cudaGraphLaunch(e, s));
    b.record(s);
    CUPIQP_CUDA_CHECK(cudaStreamSynchronize(s));
    float ms = 0;
    CUPIQP_CUDA_CHECK(cudaEventElapsedTime(&ms, a.get(), b.get()));
    best = std::min(best, ms);
  }
  CUPIQP_CUDA_CHECK(cudaGraphExecDestroy(e));
  CUPIQP_CUDA_CHECK(cudaGraphDestroy(g));
  return 1e3 * best / kLaunches;
}

struct Shape {
  const char* name;
  int n, p, m;
};

}  // namespace

TEST(IpmMultiblockTiming, DISABLED_KernelsVsBlocksPerProblem) {
  CudaStream st = CudaStream::create();
  cudaStream_t s = st.get();
  for (Shape sh : {Shape{"mpc", 169, 64, 2521}, Shape{"trajopt", 241, 64, 6385}}) {
    Dims d;
    d.B = 1; d.n = sh.n; d.p = sh.p; d.m = sh.m; d.num_hl = sh.m; d.num_hu = sh.m;
    const int ni = d.num_ineq();
    const int64_t W = 64 * int64_t(sh.n + sh.p + 2 * ni + sh.m);
    std::vector<double> h(W);
    std::mt19937_64 gen(1);
    std::uniform_real_distribution<double> u(0.5, 2.0);
    for (auto& x : h) x = u(gen);
    CudaBuffer<double> pool(W);
    pool.copy_from_host(h.data(), h.size(), s);
    int64_t off = 0;
    auto take = [&](int64_t len) { double* q = pool.data() + off; off += len; return q; };
    std::vector<double> settings(SF_COUNT, 0.0);
    settings[SF_TAU] = 0.99;
    CudaBuffer<double> sett(SF_COUNT);
    sett.copy_from_host(settings.data(), SF_COUNT, s);
    CudaBuffer<double> scal(64);
    scal.zero_async(s);
    double* sc = scal.data();

    UpdateResidualNrArgs<double> a;
    RowsView<double>* rv[] = {&a.minus_Px, &a.AT_y, &a.GT_zh_assembled, &a.zb_assembled, &a.data_c,
                              &a.data_x_l, &a.data_x_u, &a.result_x, &a.x_b_scaling, &a.delta_b_inv,
                              &a.res_nr_x};
    for (auto* r : rv) *r = rows(take(sh.n), sh.n);
    a.A_x = rows(take(sh.p), sh.p); a.data_b = rows(take(sh.p), sh.p); a.result_y = rows(take(sh.p), sh.p);
    a.res_nr_y = rows(take(sh.p), sh.p);
    RowsView<double>* rm[] = {&a.G_x, &a.data_h_l, &a.data_h_u, &a.finite_mask_hl, &a.finite_mask_hu,
                              &a.result_z_hl, &a.result_z_hu, &a.result_s_hl, &a.result_s_hu,
                              &a.res_nr_z_hl, &a.res_nr_z_hu};
    for (auto* r : rm) *r = rows(take(sh.m), sh.m);
    a.delta_inv = rows(take(sh.n + sh.p + sh.m), sh.n + sh.p + sh.m);
    a.cost_scaling_inv = sc; a.constraints_rhs_inf_norm = sc + 1;
    double** info[] = {&a.info_primal_obj, &a.info_dual_obj, &a.info_duality_gap, &a.info_duality_gap_rel,
                       &a.info_primal_res, &a.info_primal_res_rel, &a.info_dual_res, &a.info_dual_res_rel,
                       &a.info_prev_primal_res, &a.info_prev_dual_res};
    int q = 2;
    for (auto** f : info) *f = sc + q++;

    RowsView<double> s_all = rows(take(ni), ni), z_all = rows(take(ni), ni), mask = rows(take(ni), ni);
    RowsView<double> ds = rows(take(ni), ni), dz = rows(take(ni), ni);

    for (int nb : {1, 2, 4, 8, 16, 32, 64, 128}) {
      ReductionWorkspaceBuffers ws;
      ws.allocate(1, std::max(ni, sh.n + sh.p + sh.m), s, nb);
      const ReductionWorkspace v = ws.view();
      const double t_nr = time_us(s, [&](cudaStream_t cs) { launch_update_residual_nr<double>(cs, d, a, v); });
      const double t_step = time_us(s, [&](cudaStream_t cs) {
        launch_calculate_step<double>(cs, 1, ni, s_all, z_all, mask, ds, dz, sett.data(), sc + 20, sc + 21, v);
      });
      const double t_mu = time_us(s, [&](cudaStream_t cs) {
        launch_calculate_mu<double>(cs, 1, ni, s_all, z_all, mask, sc + 22, sc + 23, v);
      });
      std::printf("%-8s nb=%4d  residual_nr %7.2f us  calculate_step %6.2f us  calculate_mu %6.2f us\n", sh.name,
                  nb, t_nr, t_step, t_mu);
    }
  }
}

// The primitive alone: K values per thread (no memory traffic) reduced over
// nb blocks. MODE 0: no reduction (baseline), 1: residual_nr kinds (K = 23)
// or one sum (K = 1), 2: all sums, 3: all max.
namespace {
constexpr ReduceKind kSum = ReduceKind::Sum, kMax = ReduceKind::Max;
constexpr uint64_t kNrLike = reduce_kinds(kMax, kSum, kSum, kMax, kMax, kMax, kSum, kMax, kMax, kSum, kMax, kMax,
                                          kMax, kSum, kMax, kMax, kMax, kSum, kMax, kMax, kSum, kMax, kMax);
constexpr uint64_t kAllMax = reduce_kinds(kMax, kMax, kMax, kMax, kMax, kMax, kMax, kMax, kMax, kMax, kMax, kMax,
                                          kMax, kMax, kMax, kMax, kMax, kMax, kMax, kMax, kMax, kMax, kMax);
template <int K, int MODE>
constexpr uint64_t synth_kinds() {
  if constexpr (MODE == 2) return 0;  // all Sum
  else if constexpr (MODE == 3) return kAllMax;
  else if constexpr (K == 23) return kNrLike;
  else return reduce_kinds(kSum);
}

template <int BLOCK, int K, int MODE>
__global__ void synth_kernel(MultiblockGrid g, double* out) {
  __shared__ MultiblockStorage<BLOCK, K, double> sm;
  double v[K];
#pragma unroll
  for (int k = 0; k < K; ++k) v[k] = threadIdx.x * 0.5 + k;
  if (MODE == 0) {
    double a = 0;
#pragma unroll
    for (int k = 0; k < K; ++k) a += v[k];
    if (a == 1.2345) out[0] = a;
    return;
  }
  if (!multiblock_reduce<BLOCK, K, synth_kinds<K, MODE>()>(g, v, sm)) return;
  if (threadIdx.x == 0) out[blockIdx.x] = sm.result[0];
}
}  // namespace

TEST(IpmMultiblockTiming, DISABLED_PrimitiveAlone) {
  CudaStream st = CudaStream::create();
  cudaStream_t s = st.get();
  CudaBuffer<double> out(16);
  for (int nb : {1, 8, 32, 128}) {
    ReductionWorkspaceBuffers ws;
    ws.allocate(1, 1 << 20, s, nb);
    const MultiblockGrid g = multiblock_grid(ws.view(), nb);
    const dim3 grid(1, nb);
    auto t = [&](auto kern) { return time_us(s, [&](cudaStream_t cs) { kern<<<grid, 128, 0, cs>>>(g, out.data()); }); };
    const double t0 = t(synth_kernel<128, 23, 0>), t23 = t(synth_kernel<128, 23, 1>), t1 = t(synth_kernel<128, 1, 1>);
    const double tsum = t(synth_kernel<128, 23, 2>), tmax = t(synth_kernel<128, 23, 3>);
    std::printf("nb=%4d  none %6.2f us  K=23 mixed %6.2f us  K=1 %6.2f us  K=23 sum %6.2f us  K=23 max %6.2f us\n", nb,
                t0, t23, t1, tsum, tmax);
  }
}
