// M7 ipm_kernels: forward interior-point kernels of the reference
// cupiqp/solver_kernels.py (state reset, initial guess, predictor/corrector,
// step length, mu, residuals, rho/delta and prox updates, termination,
// factorization retry, loop control).
//
// One launcher per reference kernel, named launch_<reference name without
// create_ / _kernel>. Arguments follow the order of the reference kernel
// parameters (the inputs of its wp.launch call). Launchers with many array
// arguments take a <Name>Args struct whose fields carry the reference parameter
// names in the reference order.
//
// Array conventions:
// - (B,) arrays are plain device pointers.
// - (B, k) arrays are RowsView<T>: base pointer plus row stride, so column
//   slices of the Variables buffers (strided when B > 1) are passed as they are
//   in the reference. A contiguous (B, k) array has stride k.
// - Warp bool arrays are bool* (1 byte per entry).
// - Absent blocks (width 0) may have a null pointer; they are never read.
//
// Widths that Warp reads from the array shapes are passed explicitly: either
// as ints or through Dims (B, n, p, m, num_hl, num_hu, num_xl, num_xu).
//
// Every launcher enqueues on `s` only, allocates nothing and does not
// synchronize, so every launcher is CUDA graph capturable. B == 0 or an empty
// launch domain enqueues nothing.
//
// Tiled reference kernels (wp.launch_tiled, dim=[B]) run one block per problem
// with reduction_block_dim(width) threads of the same width the reference
// uses; their sums use cub::BlockReduce (common/reduce.cuh), whose order
// differs from Warp tile_sum, so sums may differ in the last bits. Min/max use
// fmin/fmax as Warp does (a NaN operand is ignored unless both are NaN).
#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>

#include "common/views.hpp"

namespace cupiqp {

// (B, k) device array with a row stride (elements). Element (b, j) is
// data[b * stride + j].
template <class T>
struct RowsView {
  T* data = nullptr;
  int64_t stride = 0;
#if defined(__CUDACC__)
  __host__ __device__
#endif
  T& operator()(int64_t b, int64_t j) const { return data[b * stride + j]; }
};

template <class T>
inline RowsView<T> rows(T* data, int64_t stride) { return RowsView<T>{data, stride}; }

// Sub-array views of a Variables buffer (reference Variables properties).
template <class T> inline RowsView<T> rows_x(const VariablesView<T>& v) { return {v.x(), v.primal_stride()}; }
template <class T> inline RowsView<T> rows_s_all(const VariablesView<T>& v) { return {v.s_all(), v.primal_stride()}; }
template <class T> inline RowsView<T> rows_s_l(const VariablesView<T>& v) { return {v.s_l(), v.primal_stride()}; }
template <class T> inline RowsView<T> rows_s_u(const VariablesView<T>& v) { return {v.s_u(), v.primal_stride()}; }
template <class T> inline RowsView<T> rows_s_bl(const VariablesView<T>& v) { return {v.s_bl(), v.primal_stride()}; }
template <class T> inline RowsView<T> rows_s_bu(const VariablesView<T>& v) { return {v.s_bu(), v.primal_stride()}; }
template <class T> inline RowsView<T> rows_primals_all(const VariablesView<T>& v) { return {v.primal, v.primal_stride()}; }
template <class T> inline RowsView<T> rows_y(const VariablesView<T>& v) { return {v.y(), v.dual_stride()}; }
template <class T> inline RowsView<T> rows_z_all(const VariablesView<T>& v) { return {v.z_all(), v.dual_stride()}; }
template <class T> inline RowsView<T> rows_z_l(const VariablesView<T>& v) { return {v.z_l(), v.dual_stride()}; }
template <class T> inline RowsView<T> rows_z_u(const VariablesView<T>& v) { return {v.z_u(), v.dual_stride()}; }
template <class T> inline RowsView<T> rows_z_bl(const VariablesView<T>& v) { return {v.z_bl(), v.dual_stride()}; }
template <class T> inline RowsView<T> rows_z_bu(const VariablesView<T>& v) { return {v.z_bu(), v.dual_stride()}; }
template <class T> inline RowsView<T> rows_duals_all(const VariablesView<T>& v) { return {v.dual, v.dual_stride()}; }

// ============================================================ step, mu, centering

// create_calculate_step_kernel. Tiled over num_ineq. Per problem, over finite
// entries: alpha_s = tau * min(1, min_i(ds_i < 0 ? -s_i / ds_i : 1)), and
// alpha_z likewise from (z, dz). tau = settings.floats[SF_TAU].
template <class T>
void launch_calculate_step(cudaStream_t s, int32_t B, int32_t num_ineq,
                           RowsView<T> s_all, RowsView<T> z_all, RowsView<T> finite_mask_all,
                           RowsView<T> step_s_all, RowsView<T> step_z_all, const T* settings,
                           T* alpha_s, T* alpha_z);

// create_calculate_mu_kernel. Tiled over num_ineq.
// mu = sum_i(s_i * z_i * mask_i) / num_finite_bounds (0 without finite bounds).
template <class T>
void launch_calculate_mu(cudaStream_t s, int32_t B, int32_t num_ineq,
                         RowsView<T> s_all, RowsView<T> z_all, RowsView<T> finite_mask_all,
                         const T* num_finite_bounds, T* mu);

// create_init_guess_center_kernel. Tiled over num_ineq. Shifts s and z into the
// positive orthant and onto the central path; mu receives the clipped measure.
template <class T>
void launch_init_guess_center(cudaStream_t s, int32_t B, int32_t num_ineq,
                              RowsView<T> finite_mask_all, const T* num_finite_bounds,
                              RowsView<T> s_all, RowsView<T> z_all, T* mu);

// create_prepare_factor_and_predictor_step_kernel. Tiled over num_ineq:
// apply_finetune (thread 0), boundary_shift of z (active problems only),
// mu = calculate_mu, res_s_all = -s * z on finite entries (0 elsewhere).
template <class T>
struct PrepareFactorAndPredictorStepArgs {
  const bool* active_mask = nullptr;      // (B,)
  const bool* finetune_mask = nullptr;    // (B,)
  const T* settings = nullptr;            // DeviceSettings.floats
  RowsView<T> finite_mask_all;            // (B, num_ineq)
  const T* num_finite_bounds = nullptr;   // (B,)
  RowsView<T> s_all;                      // (B, num_ineq)
  RowsView<T> z_all;                      // (B, num_ineq) in-out
  T* reg_limit = nullptr;                 // (B,) in-out
  int32_t* no_primal_update = nullptr;    // (B,) in-out
  int32_t* no_dual_update = nullptr;      // (B,) in-out
  T* mu = nullptr;                        // (B,) out
  RowsView<T> res_s_all;                  // (B, num_ineq) out
};
template <class T>
void launch_prepare_factor_and_predictor_step(cudaStream_t s, int32_t B, int32_t num_ineq,
                                              const PrepareFactorAndPredictorStepArgs<T>& a);

// create_prepare_corrector_step_kernel. Tiled over num_ineq: affine step
// lengths, sigma, and the corrector rhs res_s_all = (res_s_all - ds * dz +
// sigma * mu) * mask. Writes primal_step, dual_step, sigma.
template <class T>
struct PrepareCorrectorStepArgs {
  RowsView<T> s_all;                     // (B, num_ineq)
  RowsView<T> z_all;                     // (B, num_ineq)
  RowsView<T> step_s_all;                // (B, num_ineq) affine ds
  RowsView<T> step_z_all;                // (B, num_ineq) affine dz
  RowsView<T> finite_mask_all;           // (B, num_ineq)
  const T* num_finite_bounds = nullptr;  // (B,)
  const T* mu = nullptr;                 // (B,)
  const T* settings = nullptr;           // DeviceSettings.floats
  T* primal_step = nullptr;              // (B,) out
  T* dual_step = nullptr;                // (B,) out
  T* sigma = nullptr;                    // (B,) out
  RowsView<T> res_s_all;                 // (B, num_ineq) in-out
};
template <class T>
void launch_prepare_corrector_step(cudaStream_t s, int32_t B, int32_t num_ineq,
                                   const PrepareCorrectorStepArgs<T>& a);

// create_update_vars_after_corrector_step_kernel. Element-wise over
// (B, n + num_ineq + p + num_ineq); active problems only. Primal/dual tail
// entries with an infinite bound are set to 0.
template <class T>
void launch_update_vars_after_corrector_step(cudaStream_t s, int32_t B, int32_t n, int32_t p,
                                             int32_t num_ineq, const bool* active_mask,
                                             RowsView<T> finite_mask_all, const T* primal_step,
                                             const T* dual_step, RowsView<T> step_primals_all,
                                             RowsView<T> step_duals_all, RowsView<T> primals_all,
                                             RowsView<T> duals_all);

// create_run_full_newton_step_kernel. Element-wise over (B, n + p); active
// problems only: x += dx, y += dy, primal_step = dual_step = 1.
template <class T>
void launch_run_full_newton_step(cudaStream_t s, int32_t B, int32_t n, int32_t p,
                                 const bool* active_mask, RowsView<T> step_x, RowsView<T> step_y,
                                 RowsView<T> result_x, RowsView<T> result_y, T* primal_step,
                                 T* dual_step);

// ============================================================ initial guess rhs, residuals

// create_init_guess_rhs_kernel. Element-wise over (B, n + p + num_ineq).
// Uses d.n, d.p, d.num_hl, d.num_hu, d.num_xl, d.num_xu.
template <class T>
struct InitGuessRhsArgs {
  RowsView<T> c, b_eq, h_l, h_u, x_l, x_u;
  RowsView<T> finite_mask_hl, finite_mask_hu, finite_mask_xl, finite_mask_xu;
  RowsView<T> res_x, res_y, res_zl, res_zu, res_zbl, res_zbu;  // out
};
template <class T>
void launch_init_guess_rhs(cudaStream_t s, const Dims& d, const InitGuessRhsArgs<T>& a);

// create_prepare_zu_minus_zl_and_zbu_minus_zbl_kernel. Element-wise over
// (B, m + n): zu_minus_zl = z_u - z_l, zbu_minus_zbl = x_b_scaling * (z_bu - z_bl),
// absent sides (d.num_hl / num_hu / num_xl / num_xu == 0) contribute 0.
template <class T>
void launch_prepare_zu_minus_zl_and_zbu_minus_zbl(cudaStream_t s, const Dims& d, RowsView<T> z_u,
                                                  RowsView<T> z_l, RowsView<T> z_bl,
                                                  RowsView<T> z_bu, RowsView<T> x_b_scaling,
                                                  RowsView<T> zu_minus_zl,
                                                  RowsView<T> zbu_minus_zbl);

// create_update_residual_nr_kernel. Tiled with
// reduction_block_dim(max(n, p, m)). Non-regularized residuals, objectives,
// duality gap and the residual norms (see the reference docstring).
template <class T>
struct UpdateResidualNrArgs {
  RowsView<T> minus_Px;          // (B, n) (may alias res_nr_x)
  RowsView<T> A_x;               // (B, p)
  RowsView<T> AT_y;              // (B, n)
  RowsView<T> G_x;               // (B, m)
  RowsView<T> GT_zh_assembled;   // (B, n)
  RowsView<T> zb_assembled;      // (B, n)
  RowsView<T> data_c, data_b, data_h_l, data_h_u, data_x_l, data_x_u;
  RowsView<T> finite_mask_hl, finite_mask_hu, finite_mask_xl, finite_mask_xu;
  RowsView<T> result_x, result_y;
  RowsView<T> result_z_hl, result_z_hu, result_z_xl, result_z_xu;
  RowsView<T> result_s_hl, result_s_hu, result_s_xl, result_s_xu;
  RowsView<T> x_b_scaling;                     // (B, n)
  const T* cost_scaling_inv = nullptr;         // (B,)
  RowsView<T> delta_inv;                       // (B, n + p + m)
  RowsView<T> delta_b_inv;                     // (B, n)
  const T* constraints_rhs_inf_norm = nullptr; // (B,)
  RowsView<T> res_nr_x, res_nr_y, res_nr_z_hl, res_nr_z_hu, res_nr_z_xl, res_nr_z_xu;  // out
  T* info_primal_obj = nullptr;
  T* info_dual_obj = nullptr;
  T* info_duality_gap = nullptr;
  T* info_duality_gap_rel = nullptr;
  T* info_primal_res = nullptr;       // in (copied to prev) and out
  T* info_primal_res_rel = nullptr;
  T* info_dual_res = nullptr;         // in (copied to prev) and out
  T* info_dual_res_rel = nullptr;
  T* info_prev_primal_res = nullptr;  // out
  T* info_prev_dual_res = nullptr;    // out
};
template <class T>
void launch_update_residual_nr(cudaStream_t s, const Dims& d, const UpdateResidualNrArgs<T>& a);

// create_update_residuals_r_kernel. Tiled with
// reduction_block_dim(max(n, num_duals)). Regularized residuals, their norms,
// the prox infeasibility measures and the relative regularized norms.
template <class T>
struct UpdateResidualsRArgs {
  const T* rho = nullptr;
  const T* delta = nullptr;
  RowsView<T> res_nr_x, res_nr_duals, result_x, result_duals, prox_x, prox_duals;
  RowsView<T> res_x, res_duals;  // out
  RowsView<T> dual_res_unscale_factor;    // (B, n)
  RowsView<T> primal_res_unscale_factor;  // (B, num_duals)
  const T* primal_res = nullptr;
  const T* primal_res_rel = nullptr;
  const T* dual_res = nullptr;
  const T* dual_res_rel = nullptr;
  T* primal_res_reg = nullptr;
  T* primal_res_reg_rel = nullptr;
  T* dual_res_reg = nullptr;
  T* dual_res_reg_rel = nullptr;
  T* primal_prox_inf = nullptr;
  T* dual_prox_inf = nullptr;
};
template <class T>
void launch_update_residuals_r(cudaStream_t s, int32_t B, int32_t n, int32_t num_duals,
                               const UpdateResidualsRArgs<T>& a);

// ============================================================ regularization updates

// create_update_rho_delta_with_ineq_kernel. Element-wise over (B,); active
// problems only. Decides the improvement flags from the pre-update rho/delta,
// updates rho, delta and the stagnation counters.
template <class T>
struct UpdateRhoDeltaArgs {
  const bool* active_mask = nullptr;
  const T* info_dual_res = nullptr;
  const T* info_prev_dual_res = nullptr;
  const T* info_dual_res_rel = nullptr;
  const T* info_dual_prox_inf = nullptr;
  const T* info_primal_res = nullptr;
  const T* info_prev_primal_res = nullptr;
  const T* info_primal_res_rel = nullptr;
  const T* info_primal_prox_inf = nullptr;
  const T* info_reg_limit = nullptr;
  T* info_rho = nullptr;                   // in-out
  T* info_delta = nullptr;                 // in-out
  int32_t* info_no_primal_update = nullptr;  // in-out
  int32_t* info_no_dual_update = nullptr;    // in-out
};
template <class T>
void launch_update_rho_delta_with_ineq(cudaStream_t s, int32_t B, const UpdateRhoDeltaArgs<T>& a,
                                       bool* dual_improved_out, bool* primal_improved_out,
                                       const T* settings, const int32_t* current_iter);

// create_update_prox_vars_kernel. Element-wise over (B, n + num_duals); active
// problems only: prox_x = result_x where dual_improved, prox_duals =
// result_duals where primal_improved.
template <class T>
void launch_update_prox_vars(cudaStream_t s, int32_t B, int32_t n, int32_t num_duals,
                             const bool* active_mask, const bool* dual_improved,
                             const bool* primal_improved, RowsView<T> result_x, RowsView<T> prox_x,
                             RowsView<T> result_duals, RowsView<T> prox_duals);

// create_update_rho_delta_without_ineq_kernel. Element-wise over (B, n + p);
// active problems only. Fused rho/delta/counter update (thread 0 of each row)
// and prox update of x and y.
template <class T>
void launch_update_rho_delta_without_ineq(cudaStream_t s, int32_t B, int32_t n, int32_t p,
                                          const UpdateRhoDeltaArgs<T>& a, RowsView<T> result_x,
                                          RowsView<T> prox_x, RowsView<T> result_y,
                                          RowsView<T> prox_y, const T* settings,
                                          const int32_t* current_iter);

// create_apply_finetune_kernel. Element-wise over (B,): where mask,
// reg_limit = reg_finetune_lower_limit and both counters = 0.
template <class T>
void launch_apply_finetune(cudaStream_t s, int32_t B, const bool* mask, T* reg_limit,
                           int32_t* no_primal_update, int32_t* no_dual_update, const T* settings);

// ============================================================ solve state and loop control

// create_reset_solve_state_kernel. Element-wise over (B,).
template <class T>
struct ResetSolveStateArgs {
  const T* settings = nullptr;  // DeviceSettings.floats
  T* rho = nullptr;
  T* delta = nullptr;
  T* reg_limit = nullptr;
  T* mu = nullptr;
  T* primal_step = nullptr;
  T* dual_step = nullptr;
  int32_t* no_primal_update = nullptr;
  int32_t* no_dual_update = nullptr;
  int32_t* status = nullptr;
  int32_t* iteration = nullptr;
  int32_t* retries = nullptr;
  bool* active = nullptr;
};
template <class T>
void launch_reset_solve_state(cudaStream_t s, int32_t B, const ResetSolveStateArgs<T>& a);

// Convenience: the reset as the reference _reset_solve_state calls it.
template <class T>
inline ResetSolveStateArgs<T> reset_solve_state_args(const T* settings_floats,
                                                     const InfoView<T>& info, bool* active) {
  ResetSolveStateArgs<T> a;
  a.settings = settings_floats;
  a.rho = info.rho; a.delta = info.delta; a.reg_limit = info.reg_limit; a.mu = info.mu;
  a.primal_step = info.primal_step; a.dual_step = info.dual_step;
  a.no_primal_update = info.no_primal_update; a.no_dual_update = info.no_dual_update;
  a.status = info.status; a.iteration = info.iter; a.retries = info.factor_retries;
  a.active = active;
  return a;
}

// create_update_termination_kernel. Element-wise over (B,): convergence and
// infeasibility decisions, status / iteration / active updates, finetune mask.
template <class T>
struct UpdateTerminationArgs {
  const T* primal_res = nullptr;
  const T* primal_res_rel = nullptr;
  const T* dual_res = nullptr;
  const T* dual_res_rel = nullptr;
  const T* duality_gap = nullptr;
  const T* duality_gap_rel = nullptr;
  const T* primal_res_reg = nullptr;
  const T* primal_res_reg_rel = nullptr;
  const T* dual_res_reg = nullptr;
  const T* dual_res_reg_rel = nullptr;
  const T* primal_prox_inf = nullptr;
  const T* dual_prox_inf = nullptr;
  const T* rho = nullptr;
  const T* delta = nullptr;
  const T* reg_limit = nullptr;
  const int32_t* no_primal_update = nullptr;
  const int32_t* no_dual_update = nullptr;
  const int32_t* current_iter = nullptr;  // (1,)
  const T* settings_f = nullptr;
  const int32_t* settings_i = nullptr;
  bool* active = nullptr;       // in-out
  int32_t* status = nullptr;    // in-out
  int32_t* iteration = nullptr; // in-out
  bool* finetune = nullptr;     // out
};
template <class T>
void launch_update_termination(cudaStream_t s, int32_t B, const UpdateTerminationArgs<T>& a);

// Convenience: the termination as the reference _update_termination calls it.
template <class T>
inline UpdateTerminationArgs<T> update_termination_args(const InfoView<T>& info,
                                                        const int32_t* current_iter,
                                                        const DeviceSettingsView<T>& settings,
                                                        bool* active, bool* finetune) {
  UpdateTerminationArgs<T> a;
  a.primal_res = info.primal_res; a.primal_res_rel = info.primal_res_rel;
  a.dual_res = info.dual_res; a.dual_res_rel = info.dual_res_rel;
  a.duality_gap = info.duality_gap; a.duality_gap_rel = info.duality_gap_rel;
  a.primal_res_reg = info.primal_res_reg; a.primal_res_reg_rel = info.primal_res_reg_rel;
  a.dual_res_reg = info.dual_res_reg; a.dual_res_reg_rel = info.dual_res_reg_rel;
  a.primal_prox_inf = info.primal_prox_inf; a.dual_prox_inf = info.dual_prox_inf;
  a.rho = info.rho; a.delta = info.delta; a.reg_limit = info.reg_limit;
  a.no_primal_update = info.no_primal_update; a.no_dual_update = info.no_dual_update;
  a.current_iter = current_iter;
  a.settings_f = settings.floats; a.settings_i = settings.ints;
  a.active = active; a.status = info.status; a.iteration = info.iter;
  a.finetune = finetune;
  return a;
}

// Convenience: the rho/delta update inputs as the reference passes them.
template <class T>
inline UpdateRhoDeltaArgs<T> update_rho_delta_args(const bool* active, const InfoView<T>& info) {
  UpdateRhoDeltaArgs<T> a;
  a.active_mask = active;
  a.info_dual_res = info.dual_res; a.info_prev_dual_res = info.prev_dual_res;
  a.info_dual_res_rel = info.dual_res_rel; a.info_dual_prox_inf = info.dual_prox_inf;
  a.info_primal_res = info.primal_res; a.info_prev_primal_res = info.prev_primal_res;
  a.info_primal_res_rel = info.primal_res_rel; a.info_primal_prox_inf = info.primal_prox_inf;
  a.info_reg_limit = info.reg_limit; a.info_rho = info.rho; a.info_delta = info.delta;
  a.info_no_primal_update = info.no_primal_update; a.info_no_dual_update = info.no_dual_update;
  return a;
}

// update_continue_flag_kernel. One block of reduction_block_dim(B) threads:
// continue_flag[0] = (any active and current_iter[0] < max_iter) ? 1 : 0.
void launch_update_continue_flag(cudaStream_t s, int32_t B, const bool* active,
                                 const int32_t* current_iter, const int32_t* settings_i,
                                 int32_t* continue_flag);

// create_factor_retry_kernel. Element-wise over (B,): reaction of the active
// problems to a nonzero factor status (raise rho/delta and ask for a retry, or
// NUMERICAL_ISSUES once max_factor_retires is reached).
template <class T>
struct FactorRetryArgs {
  const int32_t* factor_status = nullptr;  // (B,)
  const int32_t* current_iter = nullptr;   // (1,)
  const T* settings_f = nullptr;
  const int32_t* settings_i = nullptr;
  bool* active = nullptr;       // in-out
  int32_t* status = nullptr;    // in-out
  int32_t* iteration = nullptr; // in-out
  int32_t* retries = nullptr;   // in-out
  T* rho = nullptr;             // in-out
  T* delta = nullptr;           // in-out
  T* reg_limit = nullptr;       // in-out
  int32_t* needs_retry = nullptr;  // out
};
template <class T>
void launch_factor_retry(cudaStream_t s, int32_t B, const FactorRetryArgs<T>& a);

// mark_failed_kernel. Element-wise over (B,): active problems with a nonzero
// `failed` flag become NUMERICAL_ISSUES at current_iter and leave the active set.
void launch_mark_failed(cudaStream_t s, int32_t B, const int32_t* failed,
                        const int32_t* current_iter, bool* active, int32_t* status,
                        int32_t* iteration);

// any_flag_kernel. One block of reduction_block_dim(B) threads:
// out[0] = 1 if any flags[b] != 0, else 0.
void launch_any_flag(cudaStream_t s, int32_t B, const int32_t* flags, int32_t* out);

// terminate_active_kernel. Element-wise over (B,): every active problem gets
// new_status and current_iter and is deactivated.
void launch_terminate_active(cudaStream_t s, int32_t B, bool* active, const int32_t* current_iter,
                             int32_t new_status, int32_t* status, int32_t* iteration);

// advance_iteration_kernel. current_iter[0] += 1 (one thread).
void launch_advance_iteration(cudaStream_t s, int32_t* current_iter);

}  // namespace cupiqp
