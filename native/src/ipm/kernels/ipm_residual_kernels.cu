// Initial-guess rhs and residual kernels (reference cupiqp/solver_kernels.py:
// init_guess_rhs, prepare_zu_minus_zl_and_zbu_minus_zbl, update_residual_nr,
// update_residuals_r). Version 1: mirrors the Warp kernels.
#include <algorithm>

#include "common/check.hpp"
#include "common/reduce.cuh"
#include "ipm/kernels/ipm_device_helpers.cuh"
#include "ipm/kernels/ipm_kernels.hpp"

namespace cupiqp {
namespace {

using namespace ipm_detail;

// ------------------------------------------------------------------ init_guess_rhs
template <class T>
__global__ void init_guess_rhs_kernel(Dims d, InitGuessRhsArgs<T> a) {
  const int64_t n = d.n, p = d.p, nhl = d.num_hl, nhu = d.num_hu, nxl = d.num_xl, nxu = d.num_xu;
  const int64_t width = n + p + nhl + nhu + nxl + nxu;
  const int64_t tid = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tid >= int64_t(d.B) * width) return;
  const int64_t b = tid / width;
  const int64_t t = tid - b * width;
  if (t < n) {
    a.res_x(b, t) = -a.c(b, t);
  } else if (t < n + p) {
    const int64_t k = t - n;
    a.res_y(b, k) = a.b_eq(b, k);
  } else if (t < n + p + nhl) {
    const int64_t k = t - (n + p);
    a.res_zl(b, k) = a.finite_mask_hl(b, k) > T(0.5) ? -a.h_l(b, k) : T(0);
  } else if (t < n + p + nhl + nhu) {
    const int64_t k = t - (n + p + nhl);
    a.res_zu(b, k) = a.finite_mask_hu(b, k) > T(0.5) ? a.h_u(b, k) : T(0);
  } else if (t < n + p + nhl + nhu + nxl) {
    const int64_t k = t - (n + p + nhl + nhu);
    a.res_zbl(b, k) = a.finite_mask_xl(b, k) > T(0.5) ? -a.x_l(b, k) : T(0);
  } else {
    const int64_t k = t - (n + p + nhl + nhu + nxl);
    a.res_zbu(b, k) = a.finite_mask_xu(b, k) > T(0.5) ? a.x_u(b, k) : T(0);
  }
}

// ------------------------------------------------------------------ prepare_zu_minus_zl_and_zbu_minus_zbl
template <class T>
__global__ void prepare_zu_minus_zl_and_zbu_minus_zbl_kernel(
    Dims d, RowsView<T> z_u, RowsView<T> z_l, RowsView<T> z_bl, RowsView<T> z_bu,
    RowsView<T> x_b_scaling, RowsView<T> zu_minus_zl, RowsView<T> zbu_minus_zbl) {
  const int64_t m = d.m;
  const int64_t width = m + d.n;
  const int64_t tid = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tid >= int64_t(d.B) * width) return;
  const int64_t b = tid / width;
  const int64_t i = tid - b * width;
  if (i < m) {
    T zu = T(0);
    T zl = T(0);
    if (d.num_hu > 0) zu = z_u(b, i);
    if (d.num_hl > 0) zl = z_l(b, i);
    zu_minus_zl(b, i) = zu - zl;
  } else {
    const int64_t j = i - m;
    T zbu = T(0);
    T zbl = T(0);
    if (d.num_xu > 0) zbu = z_bu(b, j);
    if (d.num_xl > 0) zbl = z_bl(b, j);
    zbu_minus_zbl(b, j) = x_b_scaling(b, j) * (zbu - zbl);
  }
}

// ------------------------------------------------------------------ update_residual_nr
template <int BLOCK, class T>
__global__ void __launch_bounds__(BLOCK) update_residual_nr_kernel(Dims d, UpdateResidualNrArgs<T> a) {
  const int64_t b = blockIdx.x;
  const int i = threadIdx.x;
  const int32_t n = d.n, p = d.p;
  const int32_t num_hl = d.num_hl, num_hu = d.num_hu, num_xl = d.num_xl, num_xu = d.num_xu;
  const T csi = a.cost_scaling_inv[b];

  if (i == 0) {
    a.info_prev_primal_res[b] = a.info_primal_res[b];
    a.info_prev_dual_res[b] = a.info_dual_res[b];
  }

  // ===================== x-segment
  T max_dual_res = T(0), sum_xPx = T(0), sum_cx = T(0);
  T max_Px = T(0), max_c = T(0), max_accum = T(0);
  for (int k = i; k < n; k += BLOCK) {
    T mpx = a.minus_Px(b, k);
    T xk = a.result_x(b, k);
    T ck = a.data_c(b, k);
    T aty = a.AT_y(b, k);
    T gtz = a.GT_zh_assembled(b, k);
    T zbk = a.zb_assembled(b, k);
    T dxi = a.delta_inv(b, k);
    T r = mpx - ck - aty - gtz - zbk;
    a.res_nr_x(b, k) = r;
    max_dual_res = wmax(max_dual_res, fabs(dxi * r));
    sum_xPx += mpx * xk;
    sum_cx += ck * xk;
    max_Px = wmax(max_Px, fabs(mpx * dxi));
    max_c = wmax(max_c, fabs(ck * dxi));
    max_accum = wmax(max_accum, fabs((aty + gtz + zbk) * dxi));
  }
  T red_dual_res = block_fmax<BLOCK>(max_dual_res);
  T red_xPx = block_sum<BLOCK>(sum_xPx);
  T red_cx = block_sum<BLOCK>(sum_cx);
  T red_Px = block_fmax<BLOCK>(max_Px);
  T red_c = block_fmax<BLOCK>(max_c);
  T red_accum = block_fmax<BLOCK>(max_accum);

  T dual_res = red_dual_res * csi;
  T half_xT_Px = T(-0.5) * red_xPx;
  T cT_x = red_cx;
  T drn_max = wmax(red_Px, wmax(red_c, red_accum));

  T bT_y = T(0), hl_zhl = T(0), hu_zhu = T(0), xl_zxl = T(0), xu_zxu = T(0);
  T primal_res = T(0);
  T primal_res_rel_norm = T(0);

  // ===================== y-segment
  if (p > 0) {
    T sum_by = T(0), max_res_y = T(0), max_Ax = T(0);
    for (int k = i; k < p; k += BLOCK) {
      T bk = a.data_b(b, k);
      T ax = a.A_x(b, k);
      T dyi = a.delta_inv(b, n + k);
      sum_by += bk * a.result_y(b, k);
      T r = -ax + bk;
      a.res_nr_y(b, k) = r;
      max_res_y = wmax(max_res_y, fabs(r * dyi));
      max_Ax = wmax(max_Ax, fabs(ax * dyi));
    }
    T red_by = block_sum<BLOCK>(sum_by);
    T red_res_y = block_fmax<BLOCK>(max_res_y);
    T red_Ax = block_fmax<BLOCK>(max_Ax);
    bT_y = red_by;
    primal_res = wmax(primal_res, red_res_y);
    primal_res_rel_norm = wmax(primal_res_rel_norm, red_Ax);
  }

  // ===================== z_l (hl) segment
  if (num_hl > 0) {
    T sum_hz = T(0), max_res = T(0), max_gx = T(0), max_s = T(0);
    for (int k = i; k < num_hl; k += BLOCK) {
      T mask = a.finite_mask_hl(b, k);
      T hl = finite_value(a.data_h_l(b, k), mask);
      T zhl = a.result_z_hl(b, k) * mask;
      sum_hz += hl * zhl;
      T gx = a.G_x(b, k);
      T sv = a.result_s_hl(b, k) * mask;
      T r = (gx - sv - hl) * mask;
      a.res_nr_z_hl(b, k) = r;
      T dzi = a.delta_inv(b, n + p + k);
      max_res = wmax(max_res, fabs(r * dzi));
      max_gx = wmax(max_gx, fabs(gx * mask * dzi));
      max_s = wmax(max_s, fabs(sv * dzi));
    }
    T red_hz = block_sum<BLOCK>(sum_hz);
    T red_res = block_fmax<BLOCK>(max_res);
    T red_gx = block_fmax<BLOCK>(max_gx);
    T red_s = block_fmax<BLOCK>(max_s);
    hl_zhl = red_hz;
    primal_res = wmax(primal_res, red_res);
    primal_res_rel_norm = wmax(primal_res_rel_norm, wmax(red_gx, red_s));
  }

  // ===================== z_u (hu) segment
  if (num_hu > 0) {
    T sum_hz = T(0), max_res = T(0), max_gx = T(0), max_s = T(0);
    for (int k = i; k < num_hu; k += BLOCK) {
      T mask = a.finite_mask_hu(b, k);
      T hu = finite_value(a.data_h_u(b, k), mask);
      T zhu = a.result_z_hu(b, k) * mask;
      sum_hz += hu * zhu;
      T gx = a.G_x(b, k);
      T sv = a.result_s_hu(b, k) * mask;
      T r = (-gx - sv + hu) * mask;
      a.res_nr_z_hu(b, k) = r;
      T dzi = a.delta_inv(b, n + p + k);
      max_res = wmax(max_res, fabs(r * dzi));
      max_gx = wmax(max_gx, fabs(gx * mask * dzi));
      max_s = wmax(max_s, fabs(sv * dzi));
    }
    T red_hz = block_sum<BLOCK>(sum_hz);
    T red_res = block_fmax<BLOCK>(max_res);
    T red_gx = block_fmax<BLOCK>(max_gx);
    T red_s = block_fmax<BLOCK>(max_s);
    hu_zhu = red_hz;
    primal_res = wmax(primal_res, red_res);
    primal_res_rel_norm = wmax(primal_res_rel_norm, wmax(red_gx, red_s));
  }

  // ===================== z_bl (xl) segment
  if (num_xl > 0) {
    T sum_xz = T(0), max_res = T(0), max_s = T(0);
    for (int k = i; k < num_xl; k += BLOCK) {
      T mask = a.finite_mask_xl(b, k);
      T xl = finite_value(a.data_x_l(b, k), mask);
      T zxl = a.result_z_xl(b, k) * mask;
      sum_xz += xl * zxl;
      T sv = a.result_s_xl(b, k) * mask;
      T r = (a.x_b_scaling(b, k) * a.result_x(b, k) - sv - xl) * mask;
      a.res_nr_z_xl(b, k) = r;
      T dbi = a.delta_b_inv(b, k);
      max_res = wmax(max_res, fabs(r * dbi));
      max_s = wmax(max_s, fabs(sv * dbi));
    }
    T red_xz = block_sum<BLOCK>(sum_xz);
    T red_res = block_fmax<BLOCK>(max_res);
    T red_s = block_fmax<BLOCK>(max_s);
    xl_zxl = red_xz;
    primal_res = wmax(primal_res, red_res);
    primal_res_rel_norm = wmax(primal_res_rel_norm, red_s);
  }

  // ===================== z_bu (xu) segment
  if (num_xu > 0) {
    T sum_xz = T(0), max_res = T(0), max_s = T(0);
    for (int k = i; k < num_xu; k += BLOCK) {
      T mask = a.finite_mask_xu(b, k);
      T xu = finite_value(a.data_x_u(b, k), mask);
      T zxu = a.result_z_xu(b, k) * mask;
      sum_xz += xu * zxu;
      T sv = a.result_s_xu(b, k) * mask;
      T r = (-a.x_b_scaling(b, k) * a.result_x(b, k) - sv + xu) * mask;
      a.res_nr_z_xu(b, k) = r;
      T dbi = a.delta_b_inv(b, k);
      max_res = wmax(max_res, fabs(r * dbi));
      max_s = wmax(max_s, fabs(sv * dbi));
    }
    T red_xz = block_sum<BLOCK>(sum_xz);
    T red_res = block_fmax<BLOCK>(max_res);
    T red_s = block_fmax<BLOCK>(max_s);
    xu_zxu = red_xz;
    primal_res = wmax(primal_res, red_res);
    primal_res_rel_norm = wmax(primal_res_rel_norm, red_s);
  }

  // ===================== finalize info
  if (i == 0) {
    T primal_obj = csi * (half_xT_Px + cT_x);
    T dual_obj = csi * (-half_xT_Px - bT_y - hu_zhu + hl_zhl - xu_zxu + xl_zxl);
    T gap = fabs(primal_obj - dual_obj);
    a.info_primal_obj[b] = primal_obj;
    a.info_dual_obj[b] = dual_obj;
    a.info_duality_gap[b] = gap;

    T gap_norm = fabs(half_xT_Px);
    gap_norm = wmax(gap_norm, fabs(cT_x));
    gap_norm = wmax(gap_norm, fabs(bT_y));
    gap_norm = wmax(gap_norm, fabs(hl_zhl));
    gap_norm = wmax(gap_norm, fabs(hu_zhu));
    gap_norm = wmax(gap_norm, fabs(xl_zxl));
    gap_norm = wmax(gap_norm, fabs(xu_zxu));
    gap_norm = wmax(csi * gap_norm, T(1));
    a.info_duality_gap_rel[b] = gap / gap_norm;

    a.info_primal_res[b] = primal_res;
    T prn = wmax(primal_res_rel_norm, a.constraints_rhs_inf_norm[b]);
    prn = wmax(prn, T(1));
    a.info_primal_res_rel[b] = primal_res / prn;

    a.info_dual_res[b] = dual_res;
    T dual_res_rel_norm = wmax(drn_max * csi, T(1));
    a.info_dual_res_rel[b] = dual_res / dual_res_rel_norm;
  }
}

// ------------------------------------------------------------------ update_residuals_r
template <int BLOCK, class T>
__global__ void __launch_bounds__(BLOCK)
update_residuals_r_kernel(int32_t n, int32_t num_duals, UpdateResidualsRArgs<T> a) {
  const int64_t b = blockIdx.x;
  const int i = threadIdx.x;
  const T rho_b = a.rho[b];
  const T delta_b = a.delta[b];

  T max_diff_x = T(0), max_res_x = T(0);
  for (int k = i; k < n; k += BLOCK) {
    T diff_x = a.result_x(b, k) - a.prox_x(b, k);
    T new_res_x = a.res_nr_x(b, k) - rho_b * diff_x;
    a.res_x(b, k) = new_res_x;
    max_diff_x = wmax(max_diff_x, fabs(diff_x));
    max_res_x = wmax(max_res_x, fabs(new_res_x) * a.dual_res_unscale_factor(b, k));
  }
  T red_diff_x = block_fmax<BLOCK>(max_diff_x);
  T red_res_x = block_fmax<BLOCK>(max_res_x);
  if (i == 0) {
    a.dual_prox_inf[b] = red_diff_x * rho_b;
    a.dual_res_reg[b] = red_res_x;
  }

  T max_diff_d = T(0), max_res_d = T(0);
  for (int k = i; k < num_duals; k += BLOCK) {
    T diff_d = a.result_duals(b, k) - a.prox_duals(b, k);
    T new_res_d = a.res_nr_duals(b, k) + delta_b * diff_d;
    a.res_duals(b, k) = new_res_d;
    max_diff_d = wmax(max_diff_d, fabs(diff_d));
    max_res_d = wmax(max_res_d, fabs(new_res_d) * a.primal_res_unscale_factor(b, k));
  }
  T red_diff_d = block_fmax<BLOCK>(max_diff_d);
  T red_res_d = block_fmax<BLOCK>(max_res_d);

  if (i == 0) {
    T primal_prox_inf = red_diff_d * delta_b;
    T primal_res_reg = red_res_d;
    a.primal_prox_inf[b] = primal_prox_inf;
    a.primal_res_reg[b] = primal_res_reg;

    if (a.primal_res_rel[b] > T(0)) {
      a.primal_res_reg_rel[b] = primal_res_reg * a.primal_res_rel[b] / a.primal_res[b];
    } else {
      a.primal_res_reg_rel[b] = primal_res_reg;
    }
    // dual_res_reg was written by this thread above.
    T dual_res_reg = a.dual_res_reg[b];
    if (a.dual_res_rel[b] > T(0)) {
      a.dual_res_reg_rel[b] = dual_res_reg * a.dual_res_rel[b] / a.dual_res[b];
    } else {
      a.dual_res_reg_rel[b] = dual_res_reg;
    }
  }
}

}  // namespace

// ================================================================== launchers

template <class T>
void launch_init_guess_rhs(cudaStream_t s, const Dims& d, const InitGuessRhsArgs<T>& a) {
  const int64_t total = int64_t(d.B) * (int64_t(d.n) + d.p + d.num_ineq());
  if (total <= 0) return;
  init_guess_rhs_kernel<T><<<grid_for(total, kElementwiseBlock), kElementwiseBlock, 0, s>>>(d, a);
  CUPIQP_CUDA_CHECK_LAUNCH();
}

template <class T>
void launch_prepare_zu_minus_zl_and_zbu_minus_zbl(cudaStream_t s, const Dims& d, RowsView<T> z_u,
                                                  RowsView<T> z_l, RowsView<T> z_bl,
                                                  RowsView<T> z_bu, RowsView<T> x_b_scaling,
                                                  RowsView<T> zu_minus_zl,
                                                  RowsView<T> zbu_minus_zbl) {
  const int64_t total = int64_t(d.B) * (int64_t(d.m) + d.n);
  if (total <= 0) return;
  prepare_zu_minus_zl_and_zbu_minus_zbl_kernel<T>
      <<<grid_for(total, kElementwiseBlock), kElementwiseBlock, 0, s>>>(
          d, z_u, z_l, z_bl, z_bu, x_b_scaling, zu_minus_zl, zbu_minus_zbl);
  CUPIQP_CUDA_CHECK_LAUNCH();
}

template <class T>
void launch_update_residual_nr(cudaStream_t s, const Dims& d, const UpdateResidualNrArgs<T>& a) {
  if (d.B <= 0) return;
  const int width = std::max(d.n, std::max(d.p, d.m));
  dispatch_block(reduction_block_dim(width), [&](auto bd) {
    constexpr int BLOCK = decltype(bd)::value;
    update_residual_nr_kernel<BLOCK, T><<<d.B, BLOCK, 0, s>>>(d, a);
  });
  CUPIQP_CUDA_CHECK_LAUNCH();
}

template <class T>
void launch_update_residuals_r(cudaStream_t s, int32_t B, int32_t n, int32_t num_duals,
                               const UpdateResidualsRArgs<T>& a) {
  if (B <= 0) return;
  dispatch_block(reduction_block_dim(std::max(n, num_duals)), [&](auto bd) {
    constexpr int BLOCK = decltype(bd)::value;
    update_residuals_r_kernel<BLOCK, T><<<B, BLOCK, 0, s>>>(n, num_duals, a);
  });
  CUPIQP_CUDA_CHECK_LAUNCH();
}

#define CUPIQP_IPM_RESIDUAL_INSTANTIATE(T)                                                         \
  template void launch_init_guess_rhs<T>(cudaStream_t, const Dims&, const InitGuessRhsArgs<T>&);  \
  template void launch_prepare_zu_minus_zl_and_zbu_minus_zbl<T>(                                   \
      cudaStream_t, const Dims&, RowsView<T>, RowsView<T>, RowsView<T>, RowsView<T>, RowsView<T>,  \
      RowsView<T>, RowsView<T>);                                                                   \
  template void launch_update_residual_nr<T>(cudaStream_t, const Dims&,                            \
                                             const UpdateResidualNrArgs<T>&);                      \
  template void launch_update_residuals_r<T>(cudaStream_t, int32_t, int32_t, int32_t,              \
                                             const UpdateResidualsRArgs<T>&);

CUPIQP_IPM_RESIDUAL_INSTANTIATE(double)
#undef CUPIQP_IPM_RESIDUAL_INSTANTIATE

}  // namespace cupiqp
