// Step, mu, centering, predictor/corrector and variable-update kernels
// (reference cupiqp/solver_kernels.py). Version 1: mirrors the Warp kernels
// (thread mapping, formula order, masks).
#include "common/check.hpp"
#include "common/reduce.cuh"
#include "ipm/kernels/ipm_device_helpers.cuh"
#include "ipm/kernels/ipm_kernels.hpp"

namespace cupiqp {
namespace {

using namespace ipm_detail;

// ------------------------------------------------------------------ calculate_step
template <int BLOCK, class T>
__global__ void __launch_bounds__(BLOCK)
calculate_step_kernel(int32_t num_ineq, RowsView<T> s_all, RowsView<T> z_all,
                      RowsView<T> finite_mask_all, RowsView<T> step_s_all, RowsView<T> step_z_all,
                      const T* settings, T* alpha_s, T* alpha_z) {
  const int64_t b = blockIdx.x;
  const int i = threadIdx.x;
  T a_s, a_z;
  calculate_step<BLOCK>(s_all, z_all, finite_mask_all, step_s_all, step_z_all, settings[SF_TAU],
                        num_ineq, b, i, a_s, a_z);
  if (i == 0) {
    alpha_s[b] = a_s;
    alpha_z[b] = a_z;
  }
}

// ------------------------------------------------------------------ calculate_mu
template <int BLOCK, class T>
__global__ void __launch_bounds__(BLOCK)
calculate_mu_kernel(int32_t num_ineq, RowsView<T> s_all, RowsView<T> z_all,
                    RowsView<T> finite_mask_all, const T* num_finite_bounds, T* mu) {
  const int64_t b = blockIdx.x;
  const int i = threadIdx.x;
  T mu_b = calculate_mu<BLOCK>(s_all, z_all, finite_mask_all, num_finite_bounds, num_ineq, b, i);
  if (i == 0) mu[b] = mu_b;
}

// ------------------------------------------------------------------ init_guess_center
template <int BLOCK, class T>
__global__ void __launch_bounds__(BLOCK)
init_guess_center_kernel(int32_t num_ineq, RowsView<T> finite_mask_all, const T* num_finite_bounds,
                         RowsView<T> s_all, RowsView<T> z_all, T* mu) {
  const int64_t b = blockIdx.x;
  const int i = threadIdx.x;

  T min_s = T(INFINITY);
  T min_z = T(INFINITY);
  for (int k = i; k < num_ineq; k += BLOCK) {
    min_s = wmin(min_s, s_all(b, k));
    min_z = wmin(min_z, z_all(b, k));
  }
  T red_s = block_fmin<BLOCK>(min_s);
  T red_z = block_fmin<BLOCK>(min_z);
  T delta_s = -red_s;
  T delta_z = -red_z;

  T acc = T(0);
  for (int k = i; k < num_ineq; k += BLOCK) {
    T sv = s_all(b, k) + delta_s;
    T zv = z_all(b, k) + delta_z;
    s_all(b, k) = sv;
    z_all(b, k) = zv;
    acc += sv * zv * finite_mask_all(b, k);
  }
  T total = block_sum<BLOCK>(acc);

  T denom = num_finite_bounds[b];
  T mu_b = T(0);
  if (denom > T(0)) mu_b = total / denom;
  // mu must be positive here, otherwise sqrt(mu) in the projection gives z = 0.
  mu_b = wmax(mu_b, T(1e-10));

  for (int k = i; k < num_ineq; k += BLOCK) {
    if (finite_mask_all(b, k) > T(0.5)) {
      T c_bk = z_all(b, k) - delta_z;
      T z_new = (c_bk + sqrt(c_bk * c_bk + T(4.0) * mu_b)) * T(0.5);
      z_all(b, k) = z_new;
      s_all(b, k) = z_new - c_bk;
    } else {
      z_all(b, k) = T(0);
      s_all(b, k) = T(0);
    }
  }
  if (i == 0) mu[b] = mu_b;
}

// ------------------------------------------------------------------ prepare_factor_and_predictor_step
template <int BLOCK, class T>
__global__ void __launch_bounds__(BLOCK)
prepare_factor_and_predictor_step_kernel(int32_t num_ineq, PrepareFactorAndPredictorStepArgs<T> a) {
  const int64_t b = blockIdx.x;
  const int i = threadIdx.x;
  if (i == 0) {
    apply_finetune(a.finetune_mask, a.reg_limit, a.no_primal_update, a.no_dual_update, a.settings, b);
  }
  if (a.active_mask[b]) boundary_shift<BLOCK>(a.z_all, a.finite_mask_all, MachineEps<T>::value, num_ineq, b, i);
  // Each thread reads back only the z entries it shifted itself.
  T mu_b = calculate_mu<BLOCK>(a.s_all, a.z_all, a.finite_mask_all, a.num_finite_bounds, num_ineq, b, i);
  if (i == 0) a.mu[b] = mu_b;
  for (int k = i; k < num_ineq; k += BLOCK) {
    T v = -a.s_all(b, k) * a.z_all(b, k);
    a.res_s_all(b, k) = a.finite_mask_all(b, k) > T(0.5) ? v : T(0);
  }
}

// ------------------------------------------------------------------ prepare_corrector_step
template <int BLOCK, class T>
__global__ void __launch_bounds__(BLOCK)
prepare_corrector_step_kernel(int32_t num_ineq, PrepareCorrectorStepArgs<T> a) {
  const int64_t b = blockIdx.x;
  const int i = threadIdx.x;
  T mu_b = a.mu[b];
  T alpha_s, alpha_z;
  calculate_step<BLOCK>(a.s_all, a.z_all, a.finite_mask_all, a.step_s_all, a.step_z_all,
                        a.settings[SF_TAU], num_ineq, b, i, alpha_s, alpha_z);
  T sigma_b = calculate_sigma<BLOCK>(a.s_all, a.z_all, a.step_s_all, a.step_z_all, a.finite_mask_all,
                                     a.num_finite_bounds, alpha_s, alpha_z, mu_b, num_ineq, b, i);
  if (i == 0) {
    a.primal_step[b] = alpha_s;
    a.dual_step[b] = alpha_z;
    a.sigma[b] = sigma_b;
  }
  for (int k = i; k < num_ineq; k += BLOCK) {
    T r = a.res_s_all(b, k) - a.step_s_all(b, k) * a.step_z_all(b, k) + sigma_b * mu_b;
    a.res_s_all(b, k) = r * a.finite_mask_all(b, k);
  }
}

// ------------------------------------------------------------------ update_vars_after_corrector_step
template <class T>
__global__ void update_vars_after_corrector_step_kernel(
    int32_t B, int32_t n, int32_t p, int32_t num_ineq, const bool* active_mask,
    RowsView<T> finite_mask_all, const T* primal_step, const T* dual_step,
    RowsView<T> step_primals_all, RowsView<T> step_duals_all, RowsView<T> primals_all,
    RowsView<T> duals_all) {
  const int64_t width = int64_t(n) + num_ineq + p + num_ineq;
  const int64_t tid = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tid >= int64_t(B) * width) return;
  const int64_t b = tid / width;
  const int64_t t = tid - b * width;
  if (!active_mask[b]) return;
  const int64_t n_primal = int64_t(n) + num_ineq;
  const int64_t n_dual = int64_t(p) + num_ineq;
  if (t < n) {
    primals_all(b, t) = primals_all(b, t) + primal_step[b] * step_primals_all(b, t);
  } else if (t < n_primal) {
    const int64_t k = t - n;
    if (finite_mask_all(b, k) > T(0.5)) {
      primals_all(b, t) = primals_all(b, t) + primal_step[b] * step_primals_all(b, t);
    } else {
      primals_all(b, t) = T(0);
    }
  } else {
    const int64_t j = t - n_primal;
    if (j < p) {
      duals_all(b, j) = duals_all(b, j) + dual_step[b] * step_duals_all(b, j);
    } else if (j < n_dual) {
      const int64_t k = j - p;
      if (finite_mask_all(b, k) > T(0.5)) {
        duals_all(b, j) = duals_all(b, j) + dual_step[b] * step_duals_all(b, j);
      } else {
        duals_all(b, j) = T(0);
      }
    }
  }
}

// ------------------------------------------------------------------ run_full_newton_step
template <class T>
__global__ void run_full_newton_step_kernel(int32_t B, int32_t n, int32_t p, const bool* active_mask,
                                            RowsView<T> step_x, RowsView<T> step_y,
                                            RowsView<T> result_x, RowsView<T> result_y,
                                            T* primal_step, T* dual_step) {
  const int64_t width = int64_t(n) + p;
  const int64_t tid = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tid >= int64_t(B) * width) return;
  const int64_t b = tid / width;
  const int64_t t = tid - b * width;
  if (!active_mask[b]) return;
  if (t < n) {
    result_x(b, t) = result_x(b, t) + step_x(b, t);
  } else if (t < width) {
    const int64_t idx = t - n;
    result_y(b, idx) = result_y(b, idx) + step_y(b, idx);
  }
  if (t == 0) {
    primal_step[b] = T(1);
    dual_step[b] = T(1);
  }
}

}  // namespace

// ================================================================== launchers

template <class T>
void launch_calculate_step(cudaStream_t s, int32_t B, int32_t num_ineq, RowsView<T> s_all,
                           RowsView<T> z_all, RowsView<T> finite_mask_all, RowsView<T> step_s_all,
                           RowsView<T> step_z_all, const T* settings, T* alpha_s, T* alpha_z) {
  if (B <= 0) return;
  dispatch_block(reduction_block_dim(num_ineq), [&](auto bd) {
    constexpr int BLOCK = decltype(bd)::value;
    calculate_step_kernel<BLOCK, T><<<B, BLOCK, 0, s>>>(num_ineq, s_all, z_all, finite_mask_all,
                                                       step_s_all, step_z_all, settings, alpha_s,
                                                       alpha_z);
  });
  CUPIQP_CUDA_CHECK_LAUNCH();
}

template <class T>
void launch_calculate_mu(cudaStream_t s, int32_t B, int32_t num_ineq, RowsView<T> s_all,
                         RowsView<T> z_all, RowsView<T> finite_mask_all, const T* num_finite_bounds,
                         T* mu) {
  if (B <= 0) return;
  dispatch_block(reduction_block_dim(num_ineq), [&](auto bd) {
    constexpr int BLOCK = decltype(bd)::value;
    calculate_mu_kernel<BLOCK, T><<<B, BLOCK, 0, s>>>(num_ineq, s_all, z_all, finite_mask_all,
                                                     num_finite_bounds, mu);
  });
  CUPIQP_CUDA_CHECK_LAUNCH();
}

template <class T>
void launch_init_guess_center(cudaStream_t s, int32_t B, int32_t num_ineq,
                              RowsView<T> finite_mask_all, const T* num_finite_bounds,
                              RowsView<T> s_all, RowsView<T> z_all, T* mu) {
  if (B <= 0) return;
  dispatch_block(reduction_block_dim(num_ineq), [&](auto bd) {
    constexpr int BLOCK = decltype(bd)::value;
    init_guess_center_kernel<BLOCK, T><<<B, BLOCK, 0, s>>>(num_ineq, finite_mask_all,
                                                          num_finite_bounds, s_all, z_all, mu);
  });
  CUPIQP_CUDA_CHECK_LAUNCH();
}

template <class T>
void launch_prepare_factor_and_predictor_step(cudaStream_t s, int32_t B, int32_t num_ineq,
                                              const PrepareFactorAndPredictorStepArgs<T>& a) {
  if (B <= 0) return;
  dispatch_block(reduction_block_dim(num_ineq), [&](auto bd) {
    constexpr int BLOCK = decltype(bd)::value;
    prepare_factor_and_predictor_step_kernel<BLOCK, T><<<B, BLOCK, 0, s>>>(num_ineq, a);
  });
  CUPIQP_CUDA_CHECK_LAUNCH();
}

template <class T>
void launch_prepare_corrector_step(cudaStream_t s, int32_t B, int32_t num_ineq,
                                   const PrepareCorrectorStepArgs<T>& a) {
  if (B <= 0) return;
  dispatch_block(reduction_block_dim(num_ineq), [&](auto bd) {
    constexpr int BLOCK = decltype(bd)::value;
    prepare_corrector_step_kernel<BLOCK, T><<<B, BLOCK, 0, s>>>(num_ineq, a);
  });
  CUPIQP_CUDA_CHECK_LAUNCH();
}

template <class T>
void launch_update_vars_after_corrector_step(cudaStream_t s, int32_t B, int32_t n, int32_t p,
                                             int32_t num_ineq, const bool* active_mask,
                                             RowsView<T> finite_mask_all, const T* primal_step,
                                             const T* dual_step, RowsView<T> step_primals_all,
                                             RowsView<T> step_duals_all, RowsView<T> primals_all,
                                             RowsView<T> duals_all) {
  const int64_t total = int64_t(B) * (int64_t(n) + num_ineq + p + num_ineq);
  if (total <= 0) return;
  update_vars_after_corrector_step_kernel<T>
      <<<grid_for(total, kElementwiseBlock), kElementwiseBlock, 0, s>>>(
          B, n, p, num_ineq, active_mask, finite_mask_all, primal_step, dual_step, step_primals_all,
          step_duals_all, primals_all, duals_all);
  CUPIQP_CUDA_CHECK_LAUNCH();
}

template <class T>
void launch_run_full_newton_step(cudaStream_t s, int32_t B, int32_t n, int32_t p,
                                 const bool* active_mask, RowsView<T> step_x, RowsView<T> step_y,
                                 RowsView<T> result_x, RowsView<T> result_y, T* primal_step,
                                 T* dual_step) {
  const int64_t total = int64_t(B) * (int64_t(n) + p);
  if (total <= 0) return;
  run_full_newton_step_kernel<T><<<grid_for(total, kElementwiseBlock), kElementwiseBlock, 0, s>>>(
      B, n, p, active_mask, step_x, step_y, result_x, result_y, primal_step, dual_step);
  CUPIQP_CUDA_CHECK_LAUNCH();
}

// ------------------------------------------------------------------ instantiations
#define CUPIQP_IPM_STEP_INSTANTIATE(T)                                                            \
  template void launch_calculate_step<T>(cudaStream_t, int32_t, int32_t, RowsView<T>, RowsView<T>, \
                                         RowsView<T>, RowsView<T>, RowsView<T>, const T*, T*, T*); \
  template void launch_calculate_mu<T>(cudaStream_t, int32_t, int32_t, RowsView<T>, RowsView<T>,   \
                                       RowsView<T>, const T*, T*);                                 \
  template void launch_init_guess_center<T>(cudaStream_t, int32_t, int32_t, RowsView<T>, const T*, \
                                            RowsView<T>, RowsView<T>, T*);                         \
  template void launch_prepare_factor_and_predictor_step<T>(                                       \
      cudaStream_t, int32_t, int32_t, const PrepareFactorAndPredictorStepArgs<T>&);                \
  template void launch_prepare_corrector_step<T>(cudaStream_t, int32_t, int32_t,                   \
                                                 const PrepareCorrectorStepArgs<T>&);              \
  template void launch_update_vars_after_corrector_step<T>(                                        \
      cudaStream_t, int32_t, int32_t, int32_t, int32_t, const bool*, RowsView<T>, const T*,        \
      const T*, RowsView<T>, RowsView<T>, RowsView<T>, RowsView<T>);                               \
  template void launch_run_full_newton_step<T>(cudaStream_t, int32_t, int32_t, int32_t,            \
                                               const bool*, RowsView<T>, RowsView<T>, RowsView<T>, \
                                               RowsView<T>, T*, T*);

CUPIQP_IPM_STEP_INSTANTIATE(double)
#undef CUPIQP_IPM_STEP_INSTANTIATE

}  // namespace cupiqp
