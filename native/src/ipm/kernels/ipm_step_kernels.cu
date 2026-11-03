// Step, mu, centering, predictor/corrector and variable-update kernels
// (reference cupiqp/solver_kernels.py). Version 1: mirrors the Warp kernels
// (thread mapping, formula order, masks).
#include "common/check.hpp"
#include "common/multiblock.cuh"
#include "common/reduce.cuh"
#include "ipm/kernels/ipm_device_helpers.cuh"
#include "ipm/kernels/ipm_kernels.hpp"

namespace cupiqp {
namespace {

using namespace ipm_detail;

// Multi-block reductions (common/multiblock.cuh): grid (B, nb), block (b, j)
// visits k = j * BLOCK + i, k += nb * BLOCK of problem b; the last block of
// each problem finalizes. With nb == 1 this is the reference thread mapping.
constexpr uint64_t kMinMin = reduce_kinds(ReduceKind::Min, ReduceKind::Min);
constexpr uint64_t kSum1 = reduce_kinds(ReduceKind::Sum);

// ------------------------------------------------------------------ calculate_step
// Also the first phase of prepare_corrector_step (alpha -> primal/dual_step).
template <int BLOCK, class T>
__global__ void __launch_bounds__(BLOCK)
calculate_step_kernel(MultiblockGrid g, int32_t num_ineq, RowsView<T> s_all, RowsView<T> z_all,
                      RowsView<T> finite_mask_all, RowsView<T> step_s_all, RowsView<T> step_z_all,
                      const T* settings, T* alpha_s, T* alpha_z) {
  __shared__ MultiblockStorage<BLOCK, 2, T> sm;
  const int64_t b = g.problem();
  T v[2] = {T(1), T(1)};
  for (int64_t k = g.first<BLOCK>(); k < num_ineq; k += g.stride<BLOCK>()) {
    T mask = finite_mask_all(b, k);
    v[0] = wmin(v[0], step_candidate_masked(step_s_all(b, k), s_all(b, k), mask));
    v[1] = wmin(v[1], step_candidate_masked(step_z_all(b, k), z_all(b, k), mask));
  }
  if (!multiblock_reduce<BLOCK, 2, kMinMin>(g, v, sm)) return;
  if (threadIdx.x == 0) {
    const T tau = settings[SF_TAU];
    alpha_s[b] = sm.result[0] * tau;
    alpha_z[b] = sm.result[1] * tau;
  }
}

// ------------------------------------------------------------------ calculate_mu
template <int BLOCK, class T>
__global__ void __launch_bounds__(BLOCK)
calculate_mu_kernel(MultiblockGrid g, int32_t num_ineq, RowsView<T> s_all, RowsView<T> z_all,
                    RowsView<T> finite_mask_all, const T* num_finite_bounds, T* mu) {
  __shared__ MultiblockStorage<BLOCK, 1, T> sm;
  const int64_t b = g.problem();
  T v[1] = {T(0)};
  for (int64_t k = g.first<BLOCK>(); k < num_ineq; k += g.stride<BLOCK>())
    v[0] += s_all(b, k) * z_all(b, k) * finite_mask_all(b, k);
  if (!multiblock_reduce<BLOCK, 1, kSum1>(g, v, sm)) return;
  if (threadIdx.x == 0) {
    T mu_b = T(0);
    if (num_finite_bounds[b] > T(0)) mu_b = sm.result[0] / num_finite_bounds[b];
    mu[b] = mu_b;
  }
}

// ------------------------------------------------------------------ init_guess_center
// One block per problem (nb == 1): the reference kernel in one launch.
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

// Multi-block init_guess_center, phase 1: red[b] = (min_k s, min_k z).
template <int BLOCK, class T>
__global__ void __launch_bounds__(BLOCK)
init_guess_center_min_kernel(MultiblockGrid g, int32_t num_ineq, RowsView<T> s_all, RowsView<T> z_all,
                             T* red) {
  __shared__ MultiblockStorage<BLOCK, 2, T> sm;
  const int64_t b = g.problem();
  T v[2] = {T(INFINITY), T(INFINITY)};
  for (int64_t k = g.first<BLOCK>(); k < num_ineq; k += g.stride<BLOCK>()) {
    v[0] = wmin(v[0], s_all(b, k));
    v[1] = wmin(v[1], z_all(b, k));
  }
  if (!multiblock_reduce<BLOCK, 2, kMinMin>(g, v, sm)) return;
  if (threadIdx.x == 0) {
    red[b * kMultiblockScalarsPerProblem + 0] = sm.result[0];
    red[b * kMultiblockScalarsPerProblem + 1] = sm.result[1];
  }
}

// Phase 2: shift s and z by -min, mu = clipped complementarity measure.
template <int BLOCK, class T>
__global__ void __launch_bounds__(BLOCK)
init_guess_center_shift_kernel(MultiblockGrid g, int32_t num_ineq, RowsView<T> finite_mask_all,
                               const T* num_finite_bounds, RowsView<T> s_all, RowsView<T> z_all,
                               const T* red, T* mu) {
  __shared__ MultiblockStorage<BLOCK, 1, T> sm;
  const int64_t b = g.problem();
  const T delta_s = -red[b * kMultiblockScalarsPerProblem + 0];
  const T delta_z = -red[b * kMultiblockScalarsPerProblem + 1];
  T v[1] = {T(0)};
  for (int64_t k = g.first<BLOCK>(); k < num_ineq; k += g.stride<BLOCK>()) {
    T sv = s_all(b, k) + delta_s;
    T zv = z_all(b, k) + delta_z;
    s_all(b, k) = sv;
    z_all(b, k) = zv;
    v[0] += sv * zv * finite_mask_all(b, k);
  }
  if (!multiblock_reduce<BLOCK, 1, kSum1>(g, v, sm)) return;
  if (threadIdx.x == 0) {
    T denom = num_finite_bounds[b];
    T mu_b = T(0);
    if (denom > T(0)) mu_b = sm.result[0] / denom;
    // mu must be positive here, otherwise sqrt(mu) in the projection gives z = 0.
    mu[b] = wmax(mu_b, T(1e-10));
  }
}

// Phase 3 (element-wise over (B, num_ineq)): projection onto the central path.
template <class T>
__global__ void init_guess_center_project_kernel(int32_t B, int32_t num_ineq, RowsView<T> finite_mask_all,
                                                 RowsView<T> s_all, RowsView<T> z_all, const T* red,
                                                 const T* mu) {
  const int64_t tid = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tid >= int64_t(B) * num_ineq) return;
  const int64_t b = tid / num_ineq;
  const int64_t k = tid - b * num_ineq;
  if (finite_mask_all(b, k) > T(0.5)) {
    const T delta_z = -red[b * kMultiblockScalarsPerProblem + 1];
    const T mu_b = mu[b];
    T c_bk = z_all(b, k) - delta_z;
    T z_new = (c_bk + sqrt(c_bk * c_bk + T(4.0) * mu_b)) * T(0.5);
    z_all(b, k) = z_new;
    s_all(b, k) = z_new - c_bk;
  } else {
    z_all(b, k) = T(0);
    s_all(b, k) = T(0);
  }
}

// ------------------------------------------------------------------ prepare_factor_and_predictor_step
// The element-wise part (boundary shift of z, res_s = -s * z) does not need
// the reduced mu, so one multi-block launch does everything; apply_finetune
// and mu run in the finalizing block.
template <int BLOCK, class T>
__global__ void __launch_bounds__(BLOCK)
prepare_factor_and_predictor_step_kernel(MultiblockGrid g, int32_t num_ineq,
                                         PrepareFactorAndPredictorStepArgs<T> a) {
  __shared__ MultiblockStorage<BLOCK, 1, T> sm;
  const int64_t b = g.problem();
  const bool active = a.active_mask[b];
  const T eps = MachineEps<T>::value;
  T v[1] = {T(0)};
  for (int64_t k = g.first<BLOCK>(); k < num_ineq; k += g.stride<BLOCK>()) {
    const T mask = a.finite_mask_all(b, k);
    // boundary_shift (active problems only)
    if (active) {
      if (mask > T(0.5)) {
        if (a.z_all(b, k) < eps) a.z_all(b, k) = a.z_all(b, k) + eps;
      } else {
        a.z_all(b, k) = T(0);
      }
    }
    const T sv = a.s_all(b, k);
    const T zv = a.z_all(b, k);  // this thread's own (shifted) entry
    v[0] += sv * zv * mask;
    T r = -sv * zv;
    a.res_s_all(b, k) = mask > T(0.5) ? r : T(0);
  }
  if (!multiblock_reduce<BLOCK, 1, kSum1>(g, v, sm)) return;
  if (threadIdx.x == 0) {
    apply_finetune(a.finetune_mask, a.reg_limit, a.no_primal_update, a.no_dual_update, a.settings, b);
    T mu_b = T(0);
    if (a.num_finite_bounds[b] > T(0)) mu_b = sm.result[0] / a.num_finite_bounds[b];
    a.mu[b] = mu_b;
  }
}

// ------------------------------------------------------------------ prepare_corrector_step
// One block per problem (nb == 1): the reference kernel in one launch.
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

// Multi-block prepare_corrector_step, phase 2 (phase 1 is calculate_step into
// primal_step / dual_step): sigma from the affine step lengths.
template <int BLOCK, class T>
__global__ void __launch_bounds__(BLOCK)
corrector_sigma_kernel(MultiblockGrid g, int32_t num_ineq, PrepareCorrectorStepArgs<T> a) {
  __shared__ MultiblockStorage<BLOCK, 1, T> sm;
  const int64_t b = g.problem();
  const T alpha_s = a.primal_step[b];
  const T alpha_z = a.dual_step[b];
  T v[1] = {T(0)};
  for (int64_t k = g.first<BLOCK>(); k < num_ineq; k += g.stride<BLOCK>()) {
    v[0] += (a.s_all(b, k) + alpha_s * a.step_s_all(b, k)) * (a.z_all(b, k) + alpha_z * a.step_z_all(b, k)) *
            a.finite_mask_all(b, k);
  }
  if (!multiblock_reduce<BLOCK, 1, kSum1>(g, v, sm)) return;
  if (threadIdx.x == 0) {
    T sigma = T(0);
    T denominator = a.mu[b] * a.num_finite_bounds[b];
    if (denominator > T(0)) {
      T val = wclamp(sm.result[0] / denominator, T(0), T(1));
      sigma = val * val * val;
    }
    a.sigma[b] = sigma;
  }
}

// Phase 3 (element-wise over (B, num_ineq)): corrector rhs.
template <class T>
__global__ void corrector_rhs_kernel(int32_t B, int32_t num_ineq, PrepareCorrectorStepArgs<T> a) {
  const int64_t tid = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tid >= int64_t(B) * num_ineq) return;
  const int64_t b = tid / num_ineq;
  const int64_t k = tid - b * num_ineq;
  // sigma * mu is rounded on its own (never contracted into an FMA), as in
  // the one-block kernel where it is a loop-invariant product.
  const T sigma_mu = mul_rn(a.sigma[b], a.mu[b]);
  T r = a.res_s_all(b, k) - a.step_s_all(b, k) * a.step_z_all(b, k) + sigma_mu;
  a.res_s_all(b, k) = r * a.finite_mask_all(b, k);
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
                           RowsView<T> step_z_all, const T* settings, T* alpha_s, T* alpha_z,
                           const ReductionWorkspace& ws) {
  if (B <= 0) return;
  const MbLaunch L = mb_launch(ws, B, num_ineq, 2);
  const MultiblockGrid g = multiblock_grid(ws, L.nb);
  dispatch_block(L.block, [&](auto bd) {
    constexpr int BLOCK = decltype(bd)::value;
    calculate_step_kernel<BLOCK, T><<<multiblock_dim(B, L.nb), BLOCK, 0, s>>>(
        g, num_ineq, s_all, z_all, finite_mask_all, step_s_all, step_z_all, settings, alpha_s, alpha_z);
  });
  CUPIQP_CUDA_CHECK_LAUNCH();
}

template <class T>
void launch_calculate_mu(cudaStream_t s, int32_t B, int32_t num_ineq, RowsView<T> s_all,
                         RowsView<T> z_all, RowsView<T> finite_mask_all, const T* num_finite_bounds,
                         T* mu, const ReductionWorkspace& ws) {
  if (B <= 0) return;
  const MbLaunch L = mb_launch(ws, B, num_ineq, 1);
  const MultiblockGrid g = multiblock_grid(ws, L.nb);
  dispatch_block(L.block, [&](auto bd) {
    constexpr int BLOCK = decltype(bd)::value;
    calculate_mu_kernel<BLOCK, T><<<multiblock_dim(B, L.nb), BLOCK, 0, s>>>(
        g, num_ineq, s_all, z_all, finite_mask_all, num_finite_bounds, mu);
  });
  CUPIQP_CUDA_CHECK_LAUNCH();
}

template <class T>
void launch_init_guess_center(cudaStream_t s, int32_t B, int32_t num_ineq,
                              RowsView<T> finite_mask_all, const T* num_finite_bounds,
                              RowsView<T> s_all, RowsView<T> z_all, T* mu,
                              const ReductionWorkspace& ws) {
  if (B <= 0) return;
  const MbLaunch L = mb_launch(ws, B, num_ineq, 2);
  const bool have_scratch = ws.scalars && ws.scalar_capacity >= int64_t(B) * kMultiblockScalarsPerProblem;
  if (L.nb == 1 || !have_scratch) {
    dispatch_block(reduction_block_dim(num_ineq), [&](auto bd) {
      constexpr int BLOCK = decltype(bd)::value;
      init_guess_center_kernel<BLOCK, T><<<B, BLOCK, 0, s>>>(num_ineq, finite_mask_all,
                                                            num_finite_bounds, s_all, z_all, mu);
    });
    CUPIQP_CUDA_CHECK_LAUNCH();
    return;
  }
  const MultiblockGrid g = multiblock_grid(ws, L.nb);
  T* red = reinterpret_cast<T*>(ws.scalars);
  dispatch_block(L.block, [&](auto bd) {
    constexpr int BLOCK = decltype(bd)::value;
    init_guess_center_min_kernel<BLOCK, T><<<multiblock_dim(B, L.nb), BLOCK, 0, s>>>(g, num_ineq, s_all,
                                                                                  z_all, red);
    CUPIQP_CUDA_CHECK_LAUNCH();
    init_guess_center_shift_kernel<BLOCK, T><<<multiblock_dim(B, L.nb), BLOCK, 0, s>>>(
        g, num_ineq, finite_mask_all, num_finite_bounds, s_all, z_all, red, mu);
    CUPIQP_CUDA_CHECK_LAUNCH();
  });
  const int64_t total = int64_t(B) * num_ineq;
  init_guess_center_project_kernel<T><<<grid_for(total, kElementwiseBlock), kElementwiseBlock, 0, s>>>(
      B, num_ineq, finite_mask_all, s_all, z_all, red, mu);
  CUPIQP_CUDA_CHECK_LAUNCH();
}

template <class T>
void launch_prepare_factor_and_predictor_step(cudaStream_t s, int32_t B, int32_t num_ineq,
                                              const PrepareFactorAndPredictorStepArgs<T>& a,
                                              const ReductionWorkspace& ws) {
  if (B <= 0) return;
  const MbLaunch L = mb_launch(ws, B, num_ineq, 1);
  const MultiblockGrid g = multiblock_grid(ws, L.nb);
  dispatch_block(L.block, [&](auto bd) {
    constexpr int BLOCK = decltype(bd)::value;
    prepare_factor_and_predictor_step_kernel<BLOCK, T><<<multiblock_dim(B, L.nb), BLOCK, 0, s>>>(
        g, num_ineq, a);
  });
  CUPIQP_CUDA_CHECK_LAUNCH();
}

template <class T>
void launch_prepare_corrector_step(cudaStream_t s, int32_t B, int32_t num_ineq,
                                   const PrepareCorrectorStepArgs<T>& a, const ReductionWorkspace& ws) {
  if (B <= 0) return;
  const MbLaunch L = mb_launch(ws, B, num_ineq, 2);
  if (L.nb == 1) {
    dispatch_block(reduction_block_dim(num_ineq), [&](auto bd) {
      constexpr int BLOCK = decltype(bd)::value;
      prepare_corrector_step_kernel<BLOCK, T><<<B, BLOCK, 0, s>>>(num_ineq, a);
    });
    CUPIQP_CUDA_CHECK_LAUNCH();
    return;
  }
  const MultiblockGrid g = multiblock_grid(ws, L.nb);
  dispatch_block(L.block, [&](auto bd) {
    constexpr int BLOCK = decltype(bd)::value;
    calculate_step_kernel<BLOCK, T><<<multiblock_dim(B, L.nb), BLOCK, 0, s>>>(
        g, num_ineq, a.s_all, a.z_all, a.finite_mask_all, a.step_s_all, a.step_z_all, a.settings,
        a.primal_step, a.dual_step);
    CUPIQP_CUDA_CHECK_LAUNCH();
    corrector_sigma_kernel<BLOCK, T><<<multiblock_dim(B, L.nb), BLOCK, 0, s>>>(g, num_ineq, a);
    CUPIQP_CUDA_CHECK_LAUNCH();
  });
  const int64_t total = int64_t(B) * num_ineq;
  corrector_rhs_kernel<T><<<grid_for(total, kElementwiseBlock), kElementwiseBlock, 0, s>>>(B, num_ineq, a);
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
                                         RowsView<T>, RowsView<T>, RowsView<T>, const T*, T*, T*, \
                                         const ReductionWorkspace&);                               \
  template void launch_calculate_mu<T>(cudaStream_t, int32_t, int32_t, RowsView<T>, RowsView<T>,   \
                                       RowsView<T>, const T*, T*, const ReductionWorkspace&);     \
  template void launch_init_guess_center<T>(cudaStream_t, int32_t, int32_t, RowsView<T>, const T*, \
                                            RowsView<T>, RowsView<T>, T*,                          \
                                            const ReductionWorkspace&);                            \
  template void launch_prepare_factor_and_predictor_step<T>(                                       \
      cudaStream_t, int32_t, int32_t, const PrepareFactorAndPredictorStepArgs<T>&, const ReductionWorkspace&); \
  template void launch_prepare_corrector_step<T>(cudaStream_t, int32_t, int32_t,                   \
                                                 const PrepareCorrectorStepArgs<T>&,               \
                                                 const ReductionWorkspace&);                       \
  template void launch_update_vars_after_corrector_step<T>(                                        \
      cudaStream_t, int32_t, int32_t, int32_t, int32_t, const bool*, RowsView<T>, const T*,        \
      const T*, RowsView<T>, RowsView<T>, RowsView<T>, RowsView<T>);                               \
  template void launch_run_full_newton_step<T>(cudaStream_t, int32_t, int32_t, int32_t,            \
                                               const bool*, RowsView<T>, RowsView<T>, RowsView<T>, \
                                               RowsView<T>, T*, T*);

CUPIQP_IPM_STEP_INSTANTIATE(double)
#undef CUPIQP_IPM_STEP_INSTANTIATE

}  // namespace cupiqp
