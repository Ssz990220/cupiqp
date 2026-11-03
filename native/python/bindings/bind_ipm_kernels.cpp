// nanobind entry points of module ipm_kernels (owned by that module).
//
// _testing.ipm_<launcher>(args: dict, ...) runs one launcher on the current
// device's legacy default stream and synchronizes. `args` maps the reference
// kernel parameter names (cupiqp/solver_kernels.py) to device addresses:
//   (B,) arrays     -> int address
//   (B, k) arrays   -> (int address, int row stride in elements)
//   scalar params   -> int / float
// Missing keys mean a null pointer (absent block). Sizes are explicit
// arguments. Used by the pytest parity tests only.
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdint>
#include <string>

#include "common/check.hpp"
#include "common/multiblock.hpp"
#include "ipm/kernels/ipm_kernels.hpp"

namespace nb = nanobind;
using namespace cupiqp;

namespace {

template <class P>
P ptr(const nb::dict& d, const char* key) {
  if (!d.contains(key)) return nullptr;
  nb::handle h = d[key];
  if (h.is_none()) return nullptr;
  return reinterpret_cast<P>(static_cast<uintptr_t>(nb::cast<uint64_t>(h)));
}
double* F(const nb::dict& d, const char* k) { return ptr<double*>(d, k); }
int32_t* I(const nb::dict& d, const char* k) { return ptr<int32_t*>(d, k); }
bool* Bp(const nb::dict& d, const char* k) { return ptr<bool*>(d, k); }

RowsView<double> R(const nb::dict& d, const char* key) {
  if (!d.contains(key)) return {};
  nb::handle h = d[key];
  if (h.is_none()) return {};
  auto t = nb::cast<std::tuple<uint64_t, int64_t>>(h);
  return {reinterpret_cast<double*>(static_cast<uintptr_t>(std::get<0>(t))), std::get<1>(t)};
}

Dims dims_of(int B, int n, int p, int m, int num_hl, int num_hu, int num_xl, int num_xu) {
  Dims d;
  d.B = B; d.n = n; d.p = p; d.m = m;
  d.num_hl = num_hl; d.num_hu = num_hu; d.num_xl = num_xl; d.num_xu = num_xu;
  return d;
}

void ipm_sync() { CUPIQP_CUDA_CHECK(cudaStreamSynchronize(nullptr)); }

// Blocks per problem of the multi-block reductions: 0 = the solver heuristic,
// > 0 forces that many (set with _testing.ipm_set_blocks_per_problem).
int g_blocks_per_problem = 0;

// Workspace of one reduction launch (allocated per call: tests only).
ReductionWorkspaceBuffers make_ws(int B, int64_t width) {
  ReductionWorkspaceBuffers w;
  w.allocate(B, width, nullptr, g_blocks_per_problem);
  return w;
}

}  // namespace

void bind_ipm_kernels(nb::module_& m, nb::module_& testing) {
  (void)m;
  cudaStream_t s0 = nullptr;

  testing.def("ipm_set_blocks_per_problem", [](int nbp) { g_blocks_per_problem = nbp; });

  testing.def("ipm_calculate_step", [s0](const nb::dict& a, int B, int num_ineq) {
    auto ws = make_ws(B, num_ineq);
    launch_calculate_step<double>(s0, B, num_ineq, R(a, "s_all"), R(a, "z_all"), R(a, "finite_mask_all"),
                                  R(a, "step_s_all"), R(a, "step_z_all"), F(a, "settings"),
                                  F(a, "alpha_s"), F(a, "alpha_z"), ws.view());
    ipm_sync();
  });

  testing.def("ipm_calculate_mu", [s0](const nb::dict& a, int B, int num_ineq) {
    auto ws = make_ws(B, num_ineq);
    launch_calculate_mu<double>(s0, B, num_ineq, R(a, "s_all"), R(a, "z_all"), R(a, "finite_mask_all"),
                                F(a, "num_finite_bounds"), F(a, "mu"), ws.view());
    ipm_sync();
  });

  testing.def("ipm_init_guess_center", [s0](const nb::dict& a, int B, int num_ineq) {
    auto ws = make_ws(B, num_ineq);
    launch_init_guess_center<double>(s0, B, num_ineq, R(a, "finite_mask_all"),
                                     F(a, "num_finite_bounds"), R(a, "s_all"), R(a, "z_all"),
                                     F(a, "mu"), ws.view());
    ipm_sync();
  });

  testing.def("ipm_prepare_factor_and_predictor_step", [s0](const nb::dict& a, int B, int num_ineq) {
    auto ws = make_ws(B, num_ineq);
    PrepareFactorAndPredictorStepArgs<double> x;
    x.active_mask = Bp(a, "active_mask");
    x.finetune_mask = Bp(a, "finetune_mask");
    x.settings = F(a, "settings");
    x.finite_mask_all = R(a, "finite_mask_all");
    x.num_finite_bounds = F(a, "num_finite_bounds");
    x.s_all = R(a, "s_all");
    x.z_all = R(a, "z_all");
    x.reg_limit = F(a, "reg_limit");
    x.no_primal_update = I(a, "no_primal_update");
    x.no_dual_update = I(a, "no_dual_update");
    x.mu = F(a, "mu");
    x.res_s_all = R(a, "res_s_all");
    launch_prepare_factor_and_predictor_step<double>(s0, B, num_ineq, x, ws.view());
    ipm_sync();
  });

  testing.def("ipm_prepare_corrector_step", [s0](const nb::dict& a, int B, int num_ineq) {
    auto ws = make_ws(B, num_ineq);
    PrepareCorrectorStepArgs<double> x;
    x.s_all = R(a, "s_all");
    x.z_all = R(a, "z_all");
    x.step_s_all = R(a, "step_s_all");
    x.step_z_all = R(a, "step_z_all");
    x.finite_mask_all = R(a, "finite_mask_all");
    x.num_finite_bounds = F(a, "num_finite_bounds");
    x.mu = F(a, "mu");
    x.settings = F(a, "settings");
    x.primal_step = F(a, "primal_step");
    x.dual_step = F(a, "dual_step");
    x.sigma = F(a, "sigma");
    x.res_s_all = R(a, "res_s_all");
    launch_prepare_corrector_step<double>(s0, B, num_ineq, x, ws.view());
    ipm_sync();
  });

  testing.def("ipm_update_vars_after_corrector_step",
              [s0](const nb::dict& a, int B, int n, int p, int num_ineq) {
                launch_update_vars_after_corrector_step<double>(
                    s0, B, n, p, num_ineq, Bp(a, "active_mask"), R(a, "finite_mask_all"),
                    F(a, "primal_step"), F(a, "dual_step"), R(a, "step_primals_all"),
                    R(a, "step_duals_all"), R(a, "primals_all"), R(a, "duals_all"));
                ipm_sync();
              });

  testing.def("ipm_run_full_newton_step", [s0](const nb::dict& a, int B, int n, int p) {
    launch_run_full_newton_step<double>(s0, B, n, p, Bp(a, "active_mask"), R(a, "step_x"),
                                        R(a, "step_y"), R(a, "result_x"), R(a, "result_y"),
                                        F(a, "primal_step"), F(a, "dual_step"));
    ipm_sync();
  });

  testing.def("ipm_init_guess_rhs", [s0](const nb::dict& a, int B, int n, int p, int m, int num_hl,
                                         int num_hu, int num_xl, int num_xu) {
    InitGuessRhsArgs<double> x;
    x.c = R(a, "c"); x.b_eq = R(a, "b_eq");
    x.h_l = R(a, "h_l"); x.h_u = R(a, "h_u"); x.x_l = R(a, "x_l"); x.x_u = R(a, "x_u");
    x.finite_mask_hl = R(a, "finite_mask_hl"); x.finite_mask_hu = R(a, "finite_mask_hu");
    x.finite_mask_xl = R(a, "finite_mask_xl"); x.finite_mask_xu = R(a, "finite_mask_xu");
    x.res_x = R(a, "res_x"); x.res_y = R(a, "res_y");
    x.res_zl = R(a, "res_zl"); x.res_zu = R(a, "res_zu");
    x.res_zbl = R(a, "res_zbl"); x.res_zbu = R(a, "res_zbu");
    launch_init_guess_rhs<double>(s0, dims_of(B, n, p, m, num_hl, num_hu, num_xl, num_xu), x);
    ipm_sync();
  });

  testing.def("ipm_prepare_zu_minus_zl_and_zbu_minus_zbl",
              [s0](const nb::dict& a, int B, int n, int p, int m, int num_hl, int num_hu, int num_xl,
                   int num_xu) {
                launch_prepare_zu_minus_zl_and_zbu_minus_zbl<double>(
                    s0, dims_of(B, n, p, m, num_hl, num_hu, num_xl, num_xu), R(a, "z_u"), R(a, "z_l"),
                    R(a, "z_bl"), R(a, "z_bu"), R(a, "x_b_scaling"), R(a, "zu_minus_zl"),
                    R(a, "zbu_minus_zbl"));
                ipm_sync();
              });

  testing.def("ipm_update_residual_nr", [s0](const nb::dict& a, int B, int n, int p, int m, int num_hl,
                                             int num_hu, int num_xl, int num_xu) {
    auto ws = make_ws(B, int64_t(n) + p + num_hl + num_hu + num_xl + num_xu);
    UpdateResidualNrArgs<double> x;
    x.minus_Px = R(a, "minus_Px"); x.A_x = R(a, "A_x"); x.AT_y = R(a, "AT_y"); x.G_x = R(a, "G_x");
    x.GT_zh_assembled = R(a, "GT_zh_assembled"); x.zb_assembled = R(a, "zb_assembled");
    x.data_c = R(a, "data_c"); x.data_b = R(a, "data_b");
    x.data_h_l = R(a, "data_h_l"); x.data_h_u = R(a, "data_h_u");
    x.data_x_l = R(a, "data_x_l"); x.data_x_u = R(a, "data_x_u");
    x.finite_mask_hl = R(a, "finite_mask_hl"); x.finite_mask_hu = R(a, "finite_mask_hu");
    x.finite_mask_xl = R(a, "finite_mask_xl"); x.finite_mask_xu = R(a, "finite_mask_xu");
    x.result_x = R(a, "result_x"); x.result_y = R(a, "result_y");
    x.result_z_hl = R(a, "result_z_hl"); x.result_z_hu = R(a, "result_z_hu");
    x.result_z_xl = R(a, "result_z_xl"); x.result_z_xu = R(a, "result_z_xu");
    x.result_s_hl = R(a, "result_s_hl"); x.result_s_hu = R(a, "result_s_hu");
    x.result_s_xl = R(a, "result_s_xl"); x.result_s_xu = R(a, "result_s_xu");
    x.x_b_scaling = R(a, "x_b_scaling");
    x.cost_scaling_inv = F(a, "cost_scaling_inv");
    x.delta_inv = R(a, "delta_inv");
    x.delta_b_inv = R(a, "delta_b_inv");
    x.constraints_rhs_inf_norm = F(a, "constraints_rhs_inf_norm");
    x.res_nr_x = R(a, "res_nr_x"); x.res_nr_y = R(a, "res_nr_y");
    x.res_nr_z_hl = R(a, "res_nr_z_hl"); x.res_nr_z_hu = R(a, "res_nr_z_hu");
    x.res_nr_z_xl = R(a, "res_nr_z_xl"); x.res_nr_z_xu = R(a, "res_nr_z_xu");
    x.info_primal_obj = F(a, "info_primal_obj");
    x.info_dual_obj = F(a, "info_dual_obj");
    x.info_duality_gap = F(a, "info_duality_gap");
    x.info_duality_gap_rel = F(a, "info_duality_gap_rel");
    x.info_primal_res = F(a, "info_primal_res");
    x.info_primal_res_rel = F(a, "info_primal_res_rel");
    x.info_dual_res = F(a, "info_dual_res");
    x.info_dual_res_rel = F(a, "info_dual_res_rel");
    x.info_prev_primal_res = F(a, "info_prev_primal_res");
    x.info_prev_dual_res = F(a, "info_prev_dual_res");
    launch_update_residual_nr<double>(s0, dims_of(B, n, p, m, num_hl, num_hu, num_xl, num_xu), x,
                                      ws.view());
    ipm_sync();
  });

  testing.def("ipm_update_residuals_r", [s0](const nb::dict& a, int B, int n, int num_duals) {
    auto ws = make_ws(B, int64_t(n) + num_duals);
    UpdateResidualsRArgs<double> x;
    x.rho = F(a, "rho"); x.delta = F(a, "delta");
    x.res_nr_x = R(a, "res_nr_x"); x.res_nr_duals = R(a, "res_nr_duals");
    x.result_x = R(a, "result_x"); x.result_duals = R(a, "result_duals");
    x.prox_x = R(a, "prox_x"); x.prox_duals = R(a, "prox_duals");
    x.res_x = R(a, "res_x"); x.res_duals = R(a, "res_duals");
    x.dual_res_unscale_factor = R(a, "dual_res_unscale_factor");
    x.primal_res_unscale_factor = R(a, "primal_res_unscale_factor");
    x.primal_res = F(a, "primal_res"); x.primal_res_rel = F(a, "primal_res_rel");
    x.dual_res = F(a, "dual_res"); x.dual_res_rel = F(a, "dual_res_rel");
    x.primal_res_reg = F(a, "primal_res_reg"); x.primal_res_reg_rel = F(a, "primal_res_reg_rel");
    x.dual_res_reg = F(a, "dual_res_reg"); x.dual_res_reg_rel = F(a, "dual_res_reg_rel");
    x.primal_prox_inf = F(a, "primal_prox_inf"); x.dual_prox_inf = F(a, "dual_prox_inf");
    launch_update_residuals_r<double>(s0, B, n, num_duals, x, ws.view());
    ipm_sync();
  });

  auto rho_delta_args = [](const nb::dict& a) {
    UpdateRhoDeltaArgs<double> x;
    x.active_mask = Bp(a, "active_mask");
    x.info_dual_res = F(a, "info_dual_res");
    x.info_prev_dual_res = F(a, "info_prev_dual_res");
    x.info_dual_res_rel = F(a, "info_dual_res_rel");
    x.info_dual_prox_inf = F(a, "info_dual_prox_inf");
    x.info_primal_res = F(a, "info_primal_res");
    x.info_prev_primal_res = F(a, "info_prev_primal_res");
    x.info_primal_res_rel = F(a, "info_primal_res_rel");
    x.info_primal_prox_inf = F(a, "info_primal_prox_inf");
    x.info_reg_limit = F(a, "info_reg_limit");
    x.info_rho = F(a, "info_rho");
    x.info_delta = F(a, "info_delta");
    x.info_no_primal_update = I(a, "info_no_primal_update");
    x.info_no_dual_update = I(a, "info_no_dual_update");
    return x;
  };

  testing.def("ipm_update_rho_delta_with_ineq", [s0, rho_delta_args](const nb::dict& a, int B) {
    launch_update_rho_delta_with_ineq<double>(s0, B, rho_delta_args(a), Bp(a, "dual_improved_out"),
                                              Bp(a, "primal_improved_out"), F(a, "settings"),
                                              I(a, "current_iter"));
    ipm_sync();
  });

  testing.def("ipm_update_prox_vars", [s0](const nb::dict& a, int B, int n, int num_duals) {
    launch_update_prox_vars<double>(s0, B, n, num_duals, Bp(a, "active_mask"), Bp(a, "dual_improved"),
                                    Bp(a, "primal_improved"), R(a, "result_x"), R(a, "prox_x"),
                                    R(a, "result_duals"), R(a, "prox_duals"));
    ipm_sync();
  });

  testing.def("ipm_update_rho_delta_without_ineq",
              [s0, rho_delta_args](const nb::dict& a, int B, int n, int p) {
                launch_update_rho_delta_without_ineq<double>(
                    s0, B, n, p, rho_delta_args(a), R(a, "result_x"), R(a, "prox_x"), R(a, "result_y"),
                    R(a, "prox_y"), F(a, "settings"), I(a, "current_iter"));
                ipm_sync();
              });

  testing.def("ipm_apply_finetune", [s0](const nb::dict& a, int B) {
    launch_apply_finetune<double>(s0, B, Bp(a, "mask"), F(a, "reg_limit"), I(a, "no_primal_update"),
                                  I(a, "no_dual_update"), F(a, "settings"));
    ipm_sync();
  });

  testing.def("ipm_reset_solve_state", [s0](const nb::dict& a, int B) {
    ResetSolveStateArgs<double> x;
    x.settings = F(a, "settings");
    x.rho = F(a, "rho"); x.delta = F(a, "delta"); x.reg_limit = F(a, "reg_limit"); x.mu = F(a, "mu");
    x.primal_step = F(a, "primal_step"); x.dual_step = F(a, "dual_step");
    x.no_primal_update = I(a, "no_primal_update"); x.no_dual_update = I(a, "no_dual_update");
    x.status = I(a, "status"); x.iteration = I(a, "iteration"); x.retries = I(a, "retries");
    x.active = Bp(a, "active");
    launch_reset_solve_state<double>(s0, B, x);
    ipm_sync();
  });

  testing.def("ipm_update_termination", [s0](const nb::dict& a, int B) {
    UpdateTerminationArgs<double> x;
    x.primal_res = F(a, "primal_res"); x.primal_res_rel = F(a, "primal_res_rel");
    x.dual_res = F(a, "dual_res"); x.dual_res_rel = F(a, "dual_res_rel");
    x.duality_gap = F(a, "duality_gap"); x.duality_gap_rel = F(a, "duality_gap_rel");
    x.primal_res_reg = F(a, "primal_res_reg"); x.primal_res_reg_rel = F(a, "primal_res_reg_rel");
    x.dual_res_reg = F(a, "dual_res_reg"); x.dual_res_reg_rel = F(a, "dual_res_reg_rel");
    x.primal_prox_inf = F(a, "primal_prox_inf"); x.dual_prox_inf = F(a, "dual_prox_inf");
    x.rho = F(a, "rho"); x.delta = F(a, "delta"); x.reg_limit = F(a, "reg_limit");
    x.no_primal_update = I(a, "no_primal_update"); x.no_dual_update = I(a, "no_dual_update");
    x.current_iter = I(a, "current_iter");
    x.settings_f = F(a, "settings_f"); x.settings_i = I(a, "settings_i");
    x.active = Bp(a, "active"); x.status = I(a, "status"); x.iteration = I(a, "iteration");
    x.finetune = Bp(a, "finetune");
    launch_update_termination<double>(s0, B, x);
    ipm_sync();
  });

  testing.def("ipm_update_continue_flag", [s0](const nb::dict& a, int B) {
    launch_update_continue_flag(s0, B, Bp(a, "active"), I(a, "current_iter"), I(a, "settings_i"),
                                I(a, "continue_flag"));
    ipm_sync();
  });

  testing.def("ipm_factor_retry", [s0](const nb::dict& a, int B) {
    FactorRetryArgs<double> x;
    x.factor_status = I(a, "factor_status"); x.current_iter = I(a, "current_iter");
    x.settings_f = F(a, "settings_f"); x.settings_i = I(a, "settings_i");
    x.active = Bp(a, "active"); x.status = I(a, "status"); x.iteration = I(a, "iteration");
    x.retries = I(a, "retries"); x.rho = F(a, "rho"); x.delta = F(a, "delta");
    x.reg_limit = F(a, "reg_limit"); x.needs_retry = I(a, "needs_retry");
    launch_factor_retry<double>(s0, B, x);
    ipm_sync();
  });

  testing.def("ipm_mark_failed", [s0](const nb::dict& a, int B) {
    launch_mark_failed(s0, B, I(a, "failed"), I(a, "current_iter"), Bp(a, "active"), I(a, "status"),
                       I(a, "iteration"));
    ipm_sync();
  });

  testing.def("ipm_any_flag", [s0](const nb::dict& a, int B) {
    launch_any_flag(s0, B, I(a, "flags"), I(a, "out"));
    ipm_sync();
  });

  testing.def("ipm_terminate_active", [s0](const nb::dict& a, int B) {
    launch_terminate_active(s0, B, Bp(a, "active"), I(a, "current_iter"),
                            nb::cast<int32_t>(a["new_status"]), I(a, "status"), I(a, "iteration"));
    ipm_sync();
  });

  testing.def("ipm_advance_iteration", [s0](const nb::dict& a) {
    launch_advance_iteration(s0, I(a, "current_iter"));
    ipm_sync();
  });
}
