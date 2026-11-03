// Solve-state, regularization and loop-control kernels (reference
// cupiqp/solver_kernels.py: reset_solve_state, update_termination,
// update_continue_flag, factor_retry, mark_failed, any_flag, terminate_active,
// advance_iteration, apply_finetune, update_rho_delta_with_ineq,
// update_prox_vars, update_rho_delta_without_ineq). Version 1: mirrors the Warp
// kernels.
#include "common/check.hpp"
#include "common/reduce.cuh"
#include "ipm/kernels/ipm_device_helpers.cuh"
#include "ipm/kernels/ipm_kernels.hpp"

namespace cupiqp {
namespace {

using namespace ipm_detail;

__device__ __forceinline__ int64_t flat_tid() { return int64_t(blockIdx.x) * blockDim.x + threadIdx.x; }

// ------------------------------------------------------------------ apply_finetune
template <class T>
__global__ void apply_finetune_kernel(int32_t B, const bool* mask, T* reg_limit,
                                      int32_t* no_primal_update, int32_t* no_dual_update,
                                      const T* settings) {
  const int64_t b = flat_tid();
  if (b >= B) return;
  apply_finetune(mask, reg_limit, no_primal_update, no_dual_update, settings, b);
}

// ------------------------------------------------------------------ reset_solve_state
template <class T>
__global__ void reset_solve_state_kernel(int32_t B, ResetSolveStateArgs<T> a) {
  const int64_t b = flat_tid();
  if (b >= B) return;
  a.rho[b] = a.settings[SF_RHO_INIT];
  a.delta[b] = a.settings[SF_DELTA_INIT];
  a.reg_limit[b] = a.settings[SF_REG_LOWER_LIMIT];
  a.mu[b] = T(0);
  a.primal_step[b] = T(0);
  a.dual_step[b] = T(0);
  a.no_primal_update[b] = 0;
  a.no_dual_update[b] = 0;
  a.status[b] = STATUS_UNSOLVED;
  a.iteration[b] = 0;
  a.retries[b] = 0;
  a.active[b] = true;
}

// ------------------------------------------------------------------ update_termination
template <class T>
__global__ void update_termination_kernel(int32_t B, UpdateTerminationArgs<T> a) {
  const int64_t b = flat_tid();
  if (b >= B) return;
  if (!a.active[b]) {
    a.finetune[b] = false;
    return;
  }
  a.iteration[b] = a.current_iter[0];
  const T eps_abs = a.settings_f[SF_EPS_ABS];
  const T eps_rel = a.settings_f[SF_EPS_REL];
  const int32_t check_duality_gap = a.settings_i[SI_CHECK_DUALITY_GAP];
  const T eps_gap_abs = a.settings_f[SF_EPS_DUALITY_GAP_ABS];
  const T eps_gap_rel = a.settings_f[SF_EPS_DUALITY_GAP_REL];
  const T infeas_thresh = a.settings_f[SF_INFEASIBILITY_THRESHOLD];
  const int32_t primal_update_threshold = a.settings_i[SI_REG_FINETUNE_PRIMAL_UPDATE_THRESHOLD];
  const int32_t dual_update_threshold = a.settings_i[SI_REG_FINETUNE_DUAL_UPDATE_THRESHOLD];
  const T reg_finetune_lower = a.settings_f[SF_REG_FINETUNE_LOWER_LIMIT];

  const bool primal_ok = (a.primal_res[b] < eps_abs) || (a.primal_res_rel[b] < eps_rel);
  const bool dual_ok = (a.dual_res[b] < eps_abs) || (a.dual_res_rel[b] < eps_rel);
  bool converged = primal_ok && dual_ok;
  if (check_duality_gap != 0) {
    const bool gap_ok = (a.duality_gap[b] < eps_gap_abs) || (a.duality_gap_rel[b] < eps_gap_rel);
    converged = converged && gap_ok;
  }

  const int32_t thr_d = dual_update_threshold < 5 ? dual_update_threshold : 5;
  const int32_t thr_p = primal_update_threshold < 5 ? primal_update_threshold : 5;
  const bool primal_infeasible =
      (a.no_dual_update[b] > thr_d) && (a.primal_prox_inf[b] > infeas_thresh) &&
      ((a.primal_res_reg[b] < eps_abs) || (a.primal_res_reg_rel[b] < eps_rel));
  const bool dual_infeasible =
      (a.no_primal_update[b] > thr_p) && (a.dual_prox_inf[b] > infeas_thresh) &&
      ((a.dual_res_reg[b] < eps_abs) || (a.dual_res_reg_rel[b] < eps_rel));

  if (converged) {
    a.status[b] = STATUS_SOLVED;
    a.active[b] = false;
  } else if (primal_infeasible) {
    a.status[b] = STATUS_PRIMAL_INFEASIBLE;
    a.active[b] = false;
  } else if (dual_infeasible) {
    a.status[b] = STATUS_DUAL_INFEASIBLE;
    a.active[b] = false;
  }

  if (!a.active[b]) {
    a.finetune[b] = false;
    return;
  }
  const bool regs_tunable = (a.reg_limit[b] != reg_finetune_lower) &&
                            (a.dual_prox_inf[b] < infeas_thresh) &&
                            (a.primal_prox_inf[b] < infeas_thresh);
  a.finetune[b] =
      regs_tunable && (((a.no_primal_update[b] > primal_update_threshold) && (a.rho[b] == a.reg_limit[b])) ||
                       ((a.no_dual_update[b] > dual_update_threshold) && (a.delta[b] == a.reg_limit[b])));
}

// ------------------------------------------------------------------ update_continue_flag
template <int BLOCK>
__global__ void __launch_bounds__(BLOCK)
update_continue_flag_kernel(int32_t B, const bool* active, const int32_t* current_iter,
                            const int32_t* settings_i, int32_t* continue_flag) {
  const int i = threadIdx.x;
  int32_t any_active = 0;
  for (int k = i; k < B; k += BLOCK) {
    if (active[k]) any_active = 1;
  }
  const int32_t total = block_max<BLOCK>(any_active);
  if (i == 0) {
    continue_flag[0] = (total > 0 && current_iter[0] < settings_i[SI_MAX_ITER]) ? 1 : 0;
  }
}

// ------------------------------------------------------------------ factor_retry
template <class T>
__global__ void factor_retry_kernel(int32_t B, FactorRetryArgs<T> a) {
  const int64_t b = flat_tid();
  if (b >= B) return;
  a.needs_retry[b] = 0;
  if (a.active[b] && a.factor_status[b] != 0) {
    a.retries[b] = a.retries[b] + 1;
    if (a.retries[b] >= a.settings_i[SI_MAX_FACTOR_RETIRES]) {
      a.status[b] = STATUS_NUMERICAL_ISSUES;
      a.iteration[b] = a.current_iter[0];
      a.active[b] = false;
    } else {
      a.rho[b] = a.rho[b] * T(100.0);
      a.delta[b] = a.delta[b] * T(100.0);
      a.reg_limit[b] = wmin(T(10.0) * a.reg_limit[b], a.settings_f[SF_EPS_ABS]);
      a.needs_retry[b] = 1;
    }
  }
}

// ------------------------------------------------------------------ mark_failed
__global__ void mark_failed_kernel(int32_t B, const int32_t* failed, const int32_t* current_iter,
                                   bool* active, int32_t* status, int32_t* iteration) {
  const int64_t b = flat_tid();
  if (b >= B) return;
  if (active[b] && failed[b] != 0) {
    status[b] = STATUS_NUMERICAL_ISSUES;
    iteration[b] = current_iter[0];
    active[b] = false;
  }
}

// ------------------------------------------------------------------ any_flag
template <int BLOCK>
__global__ void __launch_bounds__(BLOCK)
any_flag_kernel(int32_t B, const int32_t* flags, int32_t* out) {
  const int i = threadIdx.x;
  int32_t acc = 0;
  for (int k = i; k < B; k += BLOCK) {
    if (flags[k] != 0) acc = 1;
  }
  const int32_t total = block_max<BLOCK>(acc);
  if (i == 0) out[0] = total;
}

// ------------------------------------------------------------------ terminate_active
__global__ void terminate_active_kernel(int32_t B, bool* active, const int32_t* current_iter,
                                        int32_t new_status, int32_t* status, int32_t* iteration) {
  const int64_t b = flat_tid();
  if (b >= B) return;
  if (active[b]) {
    status[b] = new_status;
    iteration[b] = current_iter[0];
    active[b] = false;
  }
}

// ------------------------------------------------------------------ advance_iteration
__global__ void advance_iteration_kernel(int32_t* current_iter) { current_iter[0] = current_iter[0] + 1; }

// ------------------------------------------------------------------ rho / delta updates
// Improvement test of the reference kernels:
//   res < 0.95 * prev_res  or  res < eps_abs  or  res_rel < eps_rel
template <class T>
__device__ __forceinline__ bool improved(T res, T prev_res, T res_rel, T eps_abs, T eps_rel) {
  return (res < T(0.95) * prev_res) || (res < eps_abs) || (res_rel < eps_rel);
}

// rho/delta and counter update shared by both kernels (reference formulas).
template <class T>
__device__ __forceinline__ void update_reg(const UpdateRhoDeltaArgs<T>& a, int64_t b, bool dual_improved,
                                           bool primal_improved, bool iter_under_5, T infeas_thresh,
                                           T old_rho, T old_delta) {
  const T rho_fast = wmax(a.info_reg_limit[b], T(0.1) * old_rho);
  const T rho_slow = wmax(a.info_reg_limit[b], T(0.5) * old_rho);
  const bool rho_slow_ok = (!dual_improved) && (iter_under_5 || (a.info_dual_prox_inf[b] < infeas_thresh));
  if (dual_improved) {
    a.info_rho[b] = rho_fast;
  } else if (rho_slow_ok) {
    a.info_rho[b] = rho_slow;
  }

  const T delta_fast = wmax(a.info_reg_limit[b], T(0.1) * old_delta);
  const T delta_slow = wmax(a.info_reg_limit[b], T(0.5) * old_delta);
  const bool delta_slow_ok =
      (!primal_improved) && (iter_under_5 || (a.info_primal_prox_inf[b] < infeas_thresh));
  if (primal_improved) {
    a.info_delta[b] = delta_fast;
  } else if (delta_slow_ok) {
    a.info_delta[b] = delta_slow;
  }

  if (dual_improved) {
    a.info_no_primal_update[b] = 0;
  } else {
    a.info_no_primal_update[b] = a.info_no_primal_update[b] + 1;
  }
  if (primal_improved) {
    a.info_no_dual_update[b] = 0;
  } else {
    a.info_no_dual_update[b] = a.info_no_dual_update[b] + 1;
  }
}

template <class T>
__global__ void update_rho_delta_with_ineq_kernel(int32_t B, UpdateRhoDeltaArgs<T> a,
                                                  bool* dual_improved_out, bool* primal_improved_out,
                                                  const T* settings, const int32_t* current_iter) {
  const int64_t b = flat_tid();
  if (b >= B) return;
  const T eps_abs = settings[SF_EPS_ABS];
  const T eps_rel = settings[SF_EPS_REL];
  const T reg_finetune_lower = settings[SF_REG_FINETUNE_LOWER_LIMIT];
  const T infeas_thresh = settings[SF_INFEASIBILITY_THRESHOLD];
  if (!a.active_mask[b]) return;
  const bool iter_under_5 = current_iter[0] < 5;
  const T old_rho = a.info_rho[b];
  const T old_delta = a.info_delta[b];
  const bool dual_improved =
      improved(a.info_dual_res[b], a.info_prev_dual_res[b], a.info_dual_res_rel[b], eps_abs, eps_rel) ||
      ((old_rho == reg_finetune_lower) && (a.info_dual_prox_inf[b] < infeas_thresh));
  const bool primal_improved =
      improved(a.info_primal_res[b], a.info_prev_primal_res[b], a.info_primal_res_rel[b], eps_abs,
               eps_rel) ||
      ((old_delta == reg_finetune_lower) && (a.info_primal_prox_inf[b] < infeas_thresh));
  dual_improved_out[b] = dual_improved;
  primal_improved_out[b] = primal_improved;
  update_reg(a, b, dual_improved, primal_improved, iter_under_5, infeas_thresh, old_rho, old_delta);
}

template <class T>
__global__ void update_prox_vars_kernel(int32_t B, int32_t n, int32_t num_duals, const bool* active_mask,
                                        const bool* dual_improved, const bool* primal_improved,
                                        RowsView<T> result_x, RowsView<T> prox_x,
                                        RowsView<T> result_duals, RowsView<T> prox_duals) {
  const int64_t width = int64_t(n) + num_duals;
  const int64_t tid = flat_tid();
  if (tid >= int64_t(B) * width) return;
  const int64_t b = tid / width;
  const int64_t i = tid - b * width;
  if (!active_mask[b]) return;
  if (i < n) {
    prox_x(b, i) = dual_improved[b] ? result_x(b, i) : prox_x(b, i);
  } else if (i < width) {
    const int64_t t = i - n;
    prox_duals(b, t) = primal_improved[b] ? result_duals(b, t) : prox_duals(b, t);
  }
}

template <class T>
__global__ void update_rho_delta_without_ineq_kernel(int32_t B, int32_t n, int32_t p,
                                                     UpdateRhoDeltaArgs<T> a, RowsView<T> result_x,
                                                     RowsView<T> prox_x, RowsView<T> result_y,
                                                     RowsView<T> prox_y, const T* settings,
                                                     const int32_t* current_iter) {
  const int64_t width = int64_t(n) + p;
  const int64_t tid = flat_tid();
  if (tid >= int64_t(B) * width) return;
  const int64_t b = tid / width;
  const int64_t i = tid - b * width;
  const T eps_abs = settings[SF_EPS_ABS];
  const T eps_rel = settings[SF_EPS_REL];
  const T infeas_thresh = settings[SF_INFEASIBILITY_THRESHOLD];
  if (!a.active_mask[b]) return;
  const bool iter_under_5 = current_iter[0] < 5;
  // These flags never read rho / delta, which thread (b, 0) overwrites below,
  // so every thread of row b sees the same flags.
  const bool dual_improved =
      improved(a.info_dual_res[b], a.info_prev_dual_res[b], a.info_dual_res_rel[b], eps_abs, eps_rel);
  const bool primal_improved = improved(a.info_primal_res[b], a.info_prev_primal_res[b],
                                        a.info_primal_res_rel[b], eps_abs, eps_rel);
  if (i == 0) {
    update_reg(a, b, dual_improved, primal_improved, iter_under_5, infeas_thresh, a.info_rho[b],
               a.info_delta[b]);
  }
  if (i < n) {
    prox_x(b, i) = dual_improved ? result_x(b, i) : prox_x(b, i);
  } else if (i < width) {
    const int64_t t = i - n;
    prox_y(b, t) = primal_improved ? result_y(b, t) : prox_y(b, t);
  }
}

unsigned grid1(int64_t total) { return grid_for(total, kElementwiseBlock); }

}  // namespace

// ================================================================== launchers

template <class T>
void launch_apply_finetune(cudaStream_t s, int32_t B, const bool* mask, T* reg_limit,
                           int32_t* no_primal_update, int32_t* no_dual_update, const T* settings) {
  if (B <= 0) return;
  apply_finetune_kernel<T><<<grid1(B), kElementwiseBlock, 0, s>>>(B, mask, reg_limit, no_primal_update,
                                                                  no_dual_update, settings);
  CUPIQP_CUDA_CHECK_LAUNCH();
}

template <class T>
void launch_reset_solve_state(cudaStream_t s, int32_t B, const ResetSolveStateArgs<T>& a) {
  if (B <= 0) return;
  reset_solve_state_kernel<T><<<grid1(B), kElementwiseBlock, 0, s>>>(B, a);
  CUPIQP_CUDA_CHECK_LAUNCH();
}

template <class T>
void launch_update_termination(cudaStream_t s, int32_t B, const UpdateTerminationArgs<T>& a) {
  if (B <= 0) return;
  update_termination_kernel<T><<<grid1(B), kElementwiseBlock, 0, s>>>(B, a);
  CUPIQP_CUDA_CHECK_LAUNCH();
}

void launch_update_continue_flag(cudaStream_t s, int32_t B, const bool* active,
                                 const int32_t* current_iter, const int32_t* settings_i,
                                 int32_t* continue_flag) {
  dispatch_block(reduction_block_dim(B), [&](auto bd) {
    constexpr int BLOCK = decltype(bd)::value;
    update_continue_flag_kernel<BLOCK><<<1, BLOCK, 0, s>>>(B, active, current_iter, settings_i,
                                                          continue_flag);
  });
  CUPIQP_CUDA_CHECK_LAUNCH();
}

template <class T>
void launch_factor_retry(cudaStream_t s, int32_t B, const FactorRetryArgs<T>& a) {
  if (B <= 0) return;
  factor_retry_kernel<T><<<grid1(B), kElementwiseBlock, 0, s>>>(B, a);
  CUPIQP_CUDA_CHECK_LAUNCH();
}

void launch_mark_failed(cudaStream_t s, int32_t B, const int32_t* failed,
                        const int32_t* current_iter, bool* active, int32_t* status,
                        int32_t* iteration) {
  if (B <= 0) return;
  mark_failed_kernel<<<grid1(B), kElementwiseBlock, 0, s>>>(B, failed, current_iter, active, status,
                                                           iteration);
  CUPIQP_CUDA_CHECK_LAUNCH();
}

void launch_any_flag(cudaStream_t s, int32_t B, const int32_t* flags, int32_t* out) {
  dispatch_block(reduction_block_dim(B), [&](auto bd) {
    constexpr int BLOCK = decltype(bd)::value;
    any_flag_kernel<BLOCK><<<1, BLOCK, 0, s>>>(B, flags, out);
  });
  CUPIQP_CUDA_CHECK_LAUNCH();
}

void launch_terminate_active(cudaStream_t s, int32_t B, bool* active, const int32_t* current_iter,
                             int32_t new_status, int32_t* status, int32_t* iteration) {
  if (B <= 0) return;
  terminate_active_kernel<<<grid1(B), kElementwiseBlock, 0, s>>>(B, active, current_iter, new_status,
                                                                status, iteration);
  CUPIQP_CUDA_CHECK_LAUNCH();
}

void launch_advance_iteration(cudaStream_t s, int32_t* current_iter) {
  advance_iteration_kernel<<<1, 1, 0, s>>>(current_iter);
  CUPIQP_CUDA_CHECK_LAUNCH();
}

template <class T>
void launch_update_rho_delta_with_ineq(cudaStream_t s, int32_t B, const UpdateRhoDeltaArgs<T>& a,
                                       bool* dual_improved_out, bool* primal_improved_out,
                                       const T* settings, const int32_t* current_iter) {
  if (B <= 0) return;
  update_rho_delta_with_ineq_kernel<T><<<grid1(B), kElementwiseBlock, 0, s>>>(
      B, a, dual_improved_out, primal_improved_out, settings, current_iter);
  CUPIQP_CUDA_CHECK_LAUNCH();
}

template <class T>
void launch_update_prox_vars(cudaStream_t s, int32_t B, int32_t n, int32_t num_duals,
                             const bool* active_mask, const bool* dual_improved,
                             const bool* primal_improved, RowsView<T> result_x, RowsView<T> prox_x,
                             RowsView<T> result_duals, RowsView<T> prox_duals) {
  const int64_t total = int64_t(B) * (int64_t(n) + num_duals);
  if (total <= 0) return;
  update_prox_vars_kernel<T><<<grid1(total), kElementwiseBlock, 0, s>>>(
      B, n, num_duals, active_mask, dual_improved, primal_improved, result_x, prox_x, result_duals,
      prox_duals);
  CUPIQP_CUDA_CHECK_LAUNCH();
}

template <class T>
void launch_update_rho_delta_without_ineq(cudaStream_t s, int32_t B, int32_t n, int32_t p,
                                          const UpdateRhoDeltaArgs<T>& a, RowsView<T> result_x,
                                          RowsView<T> prox_x, RowsView<T> result_y,
                                          RowsView<T> prox_y, const T* settings,
                                          const int32_t* current_iter) {
  const int64_t total = int64_t(B) * (int64_t(n) + p);
  if (total <= 0) return;
  update_rho_delta_without_ineq_kernel<T><<<grid1(total), kElementwiseBlock, 0, s>>>(
      B, n, p, a, result_x, prox_x, result_y, prox_y, settings, current_iter);
  CUPIQP_CUDA_CHECK_LAUNCH();
}

#define CUPIQP_IPM_CONTROL_INSTANTIATE(T)                                                          \
  template void launch_apply_finetune<T>(cudaStream_t, int32_t, const bool*, T*, int32_t*,         \
                                         int32_t*, const T*);                                      \
  template void launch_reset_solve_state<T>(cudaStream_t, int32_t, const ResetSolveStateArgs<T>&); \
  template void launch_update_termination<T>(cudaStream_t, int32_t,                                \
                                             const UpdateTerminationArgs<T>&);                     \
  template void launch_factor_retry<T>(cudaStream_t, int32_t, const FactorRetryArgs<T>&);          \
  template void launch_update_rho_delta_with_ineq<T>(cudaStream_t, int32_t,                        \
                                                     const UpdateRhoDeltaArgs<T>&, bool*, bool*,   \
                                                     const T*, const int32_t*);                    \
  template void launch_update_prox_vars<T>(cudaStream_t, int32_t, int32_t, int32_t, const bool*,   \
                                           const bool*, const bool*, RowsView<T>, RowsView<T>,     \
                                           RowsView<T>, RowsView<T>);                              \
  template void launch_update_rho_delta_without_ineq<T>(cudaStream_t, int32_t, int32_t, int32_t,   \
                                                        const UpdateRhoDeltaArgs<T>&, RowsView<T>, \
                                                        RowsView<T>, RowsView<T>, RowsView<T>,     \
                                                        const T*, const int32_t*);

CUPIQP_IPM_CONTROL_INSTANTIATE(double)
#undef CUPIQP_IPM_CONTROL_INSTANTIATE

}  // namespace cupiqp
