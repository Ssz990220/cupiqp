// M7 ipm_kernels: hand-computed cases for step length, mu, sigma, the
// corrector rhs, termination, retry and loop control.
#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "common/buffer.hpp"
#include "common/handles.hpp"
#include "ipm/kernels/ipm_kernels.hpp"

using namespace cupiqp;

namespace {

class IpmKernels : public ::testing::Test {
 protected:
  CudaStream stream = CudaStream::create();
  cudaStream_t s() { return stream.get(); }

  template <class T>
  CudaBuffer<T> up(const std::vector<T>& h) {
    CudaBuffer<T> b(h.size());
    b.copy_from_host(h.data(), h.size(), s());
    return b;
  }
  CudaBuffer<bool> upb(const std::vector<int>& h) {
    std::vector<unsigned char> c(h.begin(), h.end());
    CudaBuffer<bool> b(h.size());
    if (!h.empty())
      CUPIQP_CUDA_CHECK(cudaMemcpyAsync(b.data(), c.data(), c.size(), cudaMemcpyHostToDevice, s()));
    CUPIQP_CUDA_CHECK(cudaStreamSynchronize(s()));
    return b;
  }
  template <class T>
  std::vector<T> down(const CudaBuffer<T>& b) {
    std::vector<T> h(b.size());
    CUPIQP_CUDA_CHECK(cudaMemcpyAsync(h.data(), b.data(), b.size() * sizeof(T), cudaMemcpyDeviceToHost, s()));
    CUPIQP_CUDA_CHECK(cudaStreamSynchronize(s()));
    return h;
  }
  std::vector<int> downb(const CudaBuffer<bool>& b) {
    std::vector<unsigned char> h(b.size());
    CUPIQP_CUDA_CHECK(cudaMemcpyAsync(h.data(), b.data(), b.size(), cudaMemcpyDeviceToHost, s()));
    CUPIQP_CUDA_CHECK(cudaStreamSynchronize(s()));
    return std::vector<int>(h.begin(), h.end());
  }

  static std::vector<double> settings_f(double tau = 0.99) {
    std::vector<double> f(SF_COUNT, 0.0);
    f[SF_RHO_INIT] = 1e-6;
    f[SF_DELTA_INIT] = 1e-4;
    f[SF_EPS_ABS] = 1e-8;
    f[SF_EPS_REL] = 1e-9;
    f[SF_EPS_DUALITY_GAP_ABS] = 1e-8;
    f[SF_EPS_DUALITY_GAP_REL] = 1e-9;
    f[SF_INFEASIBILITY_THRESHOLD] = 0.9;
    f[SF_REG_LOWER_LIMIT] = 1e-10;
    f[SF_REG_FINETUNE_LOWER_LIMIT] = 1e-13;
    f[SF_TAU] = tau;
    return f;
  }
  static std::vector<int32_t> settings_i(int check_gap = 1) {
    std::vector<int32_t> v(SI_COUNT, 0);
    v[SI_CHECK_DUALITY_GAP] = check_gap;
    v[SI_REG_FINETUNE_PRIMAL_UPDATE_THRESHOLD] = 3;
    v[SI_REG_FINETUNE_DUAL_UPDATE_THRESHOLD] = 3;
    v[SI_MAX_ITER] = 10;
    v[SI_MAX_FACTOR_RETIRES] = 2;
    v[SI_IR_MAX_ITER] = 10;
    return v;
  }
};

TEST_F(IpmKernels, CalculateStepHandComputed) {
  // Entry 3 has an infinite bound: its huge negative directions never limit the step.
  auto sv = up<double>({1, 2, 3, 4});
  auto zv = up<double>({1, 1, 1, 1});
  auto ds = up<double>({-2, 1, -1, -8000});
  auto dz = up<double>({-0.5, -4, 1, -100});
  auto mask = up<double>({1, 1, 1, 0});
  auto st = up(settings_f(0.99));
  CudaBuffer<double> as(1), az(1);
  launch_calculate_step<double>(s(), 1, 4, rows(sv.data(), 4), rows(zv.data(), 4), rows(mask.data(), 4),
                                rows(ds.data(), 4), rows(dz.data(), 4), st.data(), as.data(), az.data());
  EXPECT_DOUBLE_EQ(down(as)[0], 0.5 * 0.99);   // -1 / -2
  EXPECT_DOUBLE_EQ(down(az)[0], 0.25 * 0.99);  // -1 / -4
}

TEST_F(IpmKernels, CalculateStepFullStepCappedAtOne) {
  auto sv = up<double>({1, 1});
  auto d = up<double>({0.5, 2});
  auto mask = up<double>({1, 1});
  auto st = up(settings_f(0.995));
  CudaBuffer<double> as(1), az(1);
  launch_calculate_step<double>(s(), 1, 2, rows(sv.data(), 2), rows(sv.data(), 2), rows(mask.data(), 2),
                                rows(d.data(), 2), rows(d.data(), 2), st.data(), as.data(), az.data());
  EXPECT_DOUBLE_EQ(down(as)[0], 0.995);
  EXPECT_DOUBLE_EQ(down(az)[0], 0.995);
}

TEST_F(IpmKernels, CalculateMuBatchAndNoFiniteBounds) {
  // Row 0: finite entries 0, 1, 3 -> (2 + 4 + 8) / 3. Row 1: no finite bound -> 0.
  auto sv = up<double>({1, 2, 3, 4, 5, 5, 5, 5});
  auto zv = up<double>({2, 2, 2, 2, 5, 5, 5, 5});
  auto mask = up<double>({1, 1, 0, 1, 0, 0, 0, 0});
  auto nfb = up<double>({3, 0});
  CudaBuffer<double> mu(2);
  launch_calculate_mu<double>(s(), 2, 4, rows(sv.data(), 4), rows(zv.data(), 4), rows(mask.data(), 4),
                              nfb.data(), mu.data());
  auto h = down(mu);
  EXPECT_DOUBLE_EQ(h[0], 14.0 / 3.0);
  EXPECT_EQ(h[1], 0.0);
}

TEST_F(IpmKernels, CalculateMuWideRowUses256Threads) {
  const int k = 3000;  // reduction_block_dim -> 256
  std::vector<double> ones(k, 1.0), halves(k, 0.5);
  auto sv = up(ones), zv = up(halves), mask = up(ones);
  auto nfb = up<double>({double(k)});
  CudaBuffer<double> mu(1);
  launch_calculate_mu<double>(s(), 1, k, rows(sv.data(), k), rows(zv.data(), k), rows(mask.data(), k),
                              nfb.data(), mu.data());
  EXPECT_DOUBLE_EQ(down(mu)[0], 0.5);
}

TEST_F(IpmKernels, CorrectorStepSigmaAndRhs) {
  // s = z = 1, ds = dz = -0.5, tau = 1: alpha = min(2, 1) = 1,
  // mu_aff = 0.5 * 0.5 = 0.25, sigma = (0.25 / 1)^3 = 1/64,
  // res = -1 - 0.25 + 1/64 on finite entries.
  auto one = up<double>({1, 1, 1});
  auto d = up<double>({-0.5, -0.5, -0.5});
  auto mask = up<double>({1, 1, 0});
  auto nfb = up<double>({2});
  auto mu = up<double>({1});
  auto st = up(settings_f(1.0));
  auto res = up<double>({-1, -1, -1});
  CudaBuffer<double> ps(1), dsb(1), sig(1);
  PrepareCorrectorStepArgs<double> a;
  a.s_all = rows(one.data(), 3); a.z_all = rows(one.data(), 3);
  a.step_s_all = rows(d.data(), 3); a.step_z_all = rows(d.data(), 3);
  a.finite_mask_all = rows(mask.data(), 3); a.num_finite_bounds = nfb.data(); a.mu = mu.data();
  a.settings = st.data(); a.primal_step = ps.data(); a.dual_step = dsb.data(); a.sigma = sig.data();
  a.res_s_all = rows(res.data(), 3);
  launch_prepare_corrector_step<double>(s(), 1, 3, a);
  EXPECT_DOUBLE_EQ(down(ps)[0], 1.0);
  EXPECT_DOUBLE_EQ(down(dsb)[0], 1.0);
  EXPECT_DOUBLE_EQ(down(sig)[0], 1.0 / 64.0);
  auto r = down(res);
  EXPECT_DOUBLE_EQ(r[0], -1.25 + 1.0 / 64.0);
  EXPECT_DOUBLE_EQ(r[1], -1.25 + 1.0 / 64.0);
  EXPECT_EQ(r[2], 0.0);
}

TEST_F(IpmKernels, SigmaZeroWhenMuIsZero) {
  auto one = up<double>({1, 1});
  auto d = up<double>({-0.5, -0.5});
  auto mask = up<double>({1, 1});
  auto nfb = up<double>({2});
  auto mu = up<double>({0});
  auto st = up(settings_f(1.0));
  auto res = up<double>({-1, -1});
  CudaBuffer<double> ps(1), dsb(1), sig(1);
  PrepareCorrectorStepArgs<double> a;
  a.s_all = rows(one.data(), 2); a.z_all = rows(one.data(), 2);
  a.step_s_all = rows(d.data(), 2); a.step_z_all = rows(d.data(), 2);
  a.finite_mask_all = rows(mask.data(), 2); a.num_finite_bounds = nfb.data(); a.mu = mu.data();
  a.settings = st.data(); a.primal_step = ps.data(); a.dual_step = dsb.data(); a.sigma = sig.data();
  a.res_s_all = rows(res.data(), 2);
  launch_prepare_corrector_step<double>(s(), 1, 2, a);
  EXPECT_EQ(down(sig)[0], 0.0);
}

TEST_F(IpmKernels, TerminationDecisions) {
  // b0 converged; b1 primal infeasible; b2 dual infeasible; b3 unsolved and
  // finetuned (rho == reg_limit, no_primal_update > 3); b4 inactive;
  // b5 converged in residuals but the duality gap is open.
  const int B = 6;
  auto pr = up<double>({1e-9, 1, 1, 1, 0, 1e-9});
  auto prr = up<double>({1, 1, 1, 1, 0, 1});
  auto dr = up<double>({1e-9, 1, 1, 1, 0, 1e-9});
  auto drr = up<double>({1, 1, 1, 1, 0, 1});
  auto gap = up<double>({1e-9, 1, 1, 1, 0, 1});
  auto gapr = up<double>({1, 1, 1, 1, 0, 1});
  auto prreg = up<double>({1, 1e-9, 1, 1, 0, 1});
  auto prregr = up<double>({1, 1, 1, 1, 0, 1});
  auto drreg = up<double>({1, 1, 1e-9, 1, 0, 1});
  auto drregr = up<double>({1, 1, 1, 1, 0, 1});
  auto ppi = up<double>({0.5, 2, 0.5, 0.5, 0, 0.5});
  auto dpi = up<double>({0.5, 0.5, 2, 0.5, 0, 0.5});
  auto rho = up<double>({1e-6, 1e-6, 1e-6, 1e-10, 1e-6, 1e-6});
  auto delta = up<double>({1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6});
  auto reg = up<double>({1e-10, 1e-10, 1e-10, 1e-10, 1e-10, 1e-10});
  auto npu = up<int32_t>({0, 0, 6, 4, 0, 0});
  auto ndu = up<int32_t>({0, 6, 0, 0, 0, 0});
  auto it = up<int32_t>({7});
  auto sf = up(settings_f());
  auto si = up(settings_i(1));
  auto active = upb({1, 1, 1, 1, 0, 1});
  auto status = up<int32_t>(std::vector<int32_t>(B, STATUS_UNSOLVED));
  auto iteration = up<int32_t>(std::vector<int32_t>(B, 3));
  auto finetune = upb({1, 1, 1, 0, 1, 1});
  UpdateTerminationArgs<double> a;
  a.primal_res = pr.data(); a.primal_res_rel = prr.data(); a.dual_res = dr.data(); a.dual_res_rel = drr.data();
  a.duality_gap = gap.data(); a.duality_gap_rel = gapr.data();
  a.primal_res_reg = prreg.data(); a.primal_res_reg_rel = prregr.data();
  a.dual_res_reg = drreg.data(); a.dual_res_reg_rel = drregr.data();
  a.primal_prox_inf = ppi.data(); a.dual_prox_inf = dpi.data();
  a.rho = rho.data(); a.delta = delta.data(); a.reg_limit = reg.data();
  a.no_primal_update = npu.data(); a.no_dual_update = ndu.data(); a.current_iter = it.data();
  a.settings_f = sf.data(); a.settings_i = si.data();
  a.active = active.data(); a.status = status.data(); a.iteration = iteration.data();
  a.finetune = finetune.data();
  launch_update_termination<double>(s(), B, a);
  EXPECT_EQ(down(status), (std::vector<int32_t>{STATUS_SOLVED, STATUS_PRIMAL_INFEASIBLE, STATUS_DUAL_INFEASIBLE,
                                                STATUS_UNSOLVED, STATUS_UNSOLVED, STATUS_UNSOLVED}));
  EXPECT_EQ(downb(active), (std::vector<int>{0, 0, 0, 1, 0, 1}));
  EXPECT_EQ(down(iteration), (std::vector<int32_t>{7, 7, 7, 7, 3, 7}));
  EXPECT_EQ(downb(finetune), (std::vector<int>{0, 0, 0, 1, 0, 0}));

  // Without the duality-gap check b5 converges.
  auto si0 = up(settings_i(0));
  auto active2 = upb({0, 0, 0, 0, 0, 1});
  a.settings_i = si0.data();
  a.active = active2.data();
  launch_update_termination<double>(s(), B, a);
  EXPECT_EQ(down(status)[5], STATUS_SOLVED);
}

TEST_F(IpmKernels, FactorRetryAndAnyFlag) {
  // max_factor_retires = 2. b0 fails (retries 0 -> 1, retry), b1 fails (1 -> 2,
  // numerical issues), b2 ok, b3 inactive and failing (ignored).
  const int B = 4;
  auto fs = up<int32_t>({1, 1, 0, 1});
  auto it = up<int32_t>({5});
  auto sf = up(settings_f());
  auto si = up(settings_i());
  auto active = upb({1, 1, 1, 0});
  auto status = up<int32_t>(std::vector<int32_t>(B, STATUS_UNSOLVED));
  auto iteration = up<int32_t>(std::vector<int32_t>(B, 0));
  auto retries = up<int32_t>({0, 1, 0, 0});
  auto rho = up<double>({1e-6, 1e-6, 1e-6, 1e-6});
  auto delta = up<double>({1e-4, 1e-4, 1e-4, 1e-4});
  auto reg = up<double>({1e-10, 1e-10, 1e-10, 1e-10});
  auto needs = up<int32_t>({9, 9, 9, 9});
  FactorRetryArgs<double> a;
  a.factor_status = fs.data(); a.current_iter = it.data(); a.settings_f = sf.data(); a.settings_i = si.data();
  a.active = active.data(); a.status = status.data(); a.iteration = iteration.data();
  a.retries = retries.data(); a.rho = rho.data(); a.delta = delta.data(); a.reg_limit = reg.data();
  a.needs_retry = needs.data();
  launch_factor_retry<double>(s(), B, a);
  EXPECT_EQ(down(needs), (std::vector<int32_t>{1, 0, 0, 0}));
  EXPECT_EQ(down(status), (std::vector<int32_t>{STATUS_UNSOLVED, STATUS_NUMERICAL_ISSUES, STATUS_UNSOLVED,
                                                STATUS_UNSOLVED}));
  EXPECT_EQ(downb(active), (std::vector<int>{1, 0, 1, 0}));
  EXPECT_EQ(down(iteration), (std::vector<int32_t>{0, 5, 0, 0}));
  EXPECT_DOUBLE_EQ(down(rho)[0], 1e-4);
  EXPECT_DOUBLE_EQ(down(delta)[0], 1e-2);
  EXPECT_DOUBLE_EQ(down(reg)[0], 1e-9);

  CudaBuffer<int32_t> any(1);
  launch_any_flag(s(), B, needs.data(), any.data());
  EXPECT_EQ(down(any)[0], 1);
  auto zeros = up<int32_t>({0, 0, 0, 0});
  launch_any_flag(s(), B, zeros.data(), any.data());
  EXPECT_EQ(down(any)[0], 0);
}

TEST_F(IpmKernels, LoopControl) {
  auto active = upb({0, 1, 0});
  auto si = up(settings_i());  // max_iter = 10
  auto it = up<int32_t>({9});
  CudaBuffer<int32_t> flag(1);
  launch_update_continue_flag(s(), 3, active.data(), it.data(), si.data(), flag.data());
  EXPECT_EQ(down(flag)[0], 1);
  launch_advance_iteration(s(), it.data());
  EXPECT_EQ(down(it)[0], 10);
  launch_update_continue_flag(s(), 3, active.data(), it.data(), si.data(), flag.data());
  EXPECT_EQ(down(flag)[0], 0);  // max_iter reached

  auto status = up<int32_t>({STATUS_SOLVED, STATUS_UNSOLVED, STATUS_DUAL_INFEASIBLE});
  auto iteration = up<int32_t>({4, 0, 6});
  launch_terminate_active(s(), 3, active.data(), it.data(), STATUS_MAX_ITER_REACHED, status.data(),
                          iteration.data());
  EXPECT_EQ(down(status), (std::vector<int32_t>{STATUS_SOLVED, STATUS_MAX_ITER_REACHED, STATUS_DUAL_INFEASIBLE}));
  EXPECT_EQ(down(iteration), (std::vector<int32_t>{4, 10, 6}));
  EXPECT_EQ(downb(active), (std::vector<int>{0, 0, 0}));
  launch_update_continue_flag(s(), 3, active.data(), it.data(), si.data(), flag.data());
  EXPECT_EQ(down(flag)[0], 0);  // nothing active
}

TEST_F(IpmKernels, LaunchersAreCapturable) {
  // The termination + continue flag sequence records into a CUDA graph.
  auto active = upb({1});
  auto si = up(settings_i());
  auto it = up<int32_t>({0});
  CudaBuffer<int32_t> flag(1);
  cudaGraph_t g = nullptr;
  CUPIQP_CUDA_CHECK(cudaStreamBeginCapture(s(), cudaStreamCaptureModeThreadLocal));
  launch_advance_iteration(s(), it.data());
  launch_update_continue_flag(s(), 1, active.data(), it.data(), si.data(), flag.data());
  CUPIQP_CUDA_CHECK(cudaStreamEndCapture(s(), &g));
  cudaGraphExec_t ge = nullptr;
  CUPIQP_CUDA_CHECK(cudaGraphInstantiate(&ge, g, 0));
  for (int k = 0; k < 3; ++k) CUPIQP_CUDA_CHECK(cudaGraphLaunch(ge, s()));
  EXPECT_EQ(down(it)[0], 3);
  EXPECT_EQ(down(flag)[0], 1);
  CUPIQP_CUDA_CHECK(cudaGraphExecDestroy(ge));
  CUPIQP_CUDA_CHECK(cudaGraphDestroy(g));
}

}  // namespace
