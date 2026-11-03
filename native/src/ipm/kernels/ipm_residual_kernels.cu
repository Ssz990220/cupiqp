// init_guess_rhs, prepare_zu_minus_zl_and_zbu_minus_zbl, update_residual_nr,
// update_residuals_r). Version 1: mirrors the Warp kernels.
#include <algorithm>

#include "common/check.hpp"
#include "common/multiblock.cuh"
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
// Multi-block reduction (common/multiblock.cuh): every segment loop visits
// k = j * BLOCK + i, k += nb * BLOCK (the reference mapping when nb == 1).
// The 23 reduced values, in this order:
enum NrValue : int {
  NR_MAX_DUAL_RES, NR_SUM_XPX, NR_SUM_CX, NR_MAX_PX, NR_MAX_C, NR_MAX_ACCUM,   // x
  NR_SUM_BY, NR_MAX_RES_Y, NR_MAX_AX,                                         // y
  NR_HL_SUM, NR_HL_RES, NR_HL_GX, NR_HL_S,                                    // z_l (hl)
  NR_HU_SUM, NR_HU_RES, NR_HU_GX, NR_HU_S,                                    // z_u (hu)
  NR_XL_SUM, NR_XL_RES, NR_XL_S,                                              // z_bl (xl)
  NR_XU_SUM, NR_XU_RES, NR_XU_S,                                              // z_bu (xu)
  NR_COUNT
};
constexpr ReduceKind kS = ReduceKind::Sum, kM = ReduceKind::Max;
constexpr uint64_t kNrKinds = reduce_kinds(kM, kS, kS, kM, kM, kM,  // x
                                           kS, kM, kM,              // y
                                           kS, kM, kM, kM,          // hl
                                           kS, kM, kM, kM,          // hu
                                           kS, kM, kM,              // xl
                                           kS, kM, kM);             // xu

template <int BLOCK, class T>
__global__ void __launch_bounds__(BLOCK)
update_residual_nr_kernel(MultiblockGrid g, Dims d, UpdateResidualNrArgs<T> a) {
  __shared__ MultiblockStorage<BLOCK, NR_COUNT, T> sm;
  const int64_t b = g.problem();
  const int64_t k0 = g.first<BLOCK>();
  const int64_t ks = g.stride<BLOCK>();
  const int32_t n = d.n, p = d.p;
  const int32_t num_hl = d.num_hl, num_hu = d.num_hu, num_xl = d.num_xl, num_xu = d.num_xu;

  T v[NR_COUNT];
#pragma unroll
  for (int q = 0; q < NR_COUNT; ++q) v[q] = T(0);

  // One pass over the concatenated row [x | y | hl | hu | xl | xu]: each
  // thread visits t = j * BLOCK + i, t += nb * BLOCK and handles entry t in
  // its segment (the per-entry expressions are the reference ones).
  const int64_t e_y = n, e_hl = e_y + p, e_hu = e_hl + num_hl, e_xl = e_hu + num_hu, e_xu = e_xl + num_xl;
  const int64_t total = e_xu + num_xu;
  for (int64_t t = k0; t < total; t += ks) {
    if (t < e_y) {  // x-segment
      const int64_t k = t;
      T mpx = a.minus_Px(b, k);
      T xk = a.result_x(b, k);
      T ck = a.data_c(b, k);
      T aty = a.AT_y(b, k);
      T gtz = a.GT_zh_assembled(b, k);
      T zbk = a.zb_assembled(b, k);
      T dxi = a.delta_inv(b, k);
      T r = mpx - ck - aty - gtz - zbk;
      a.res_nr_x(b, k) = r;
      v[NR_MAX_DUAL_RES] = wmax(v[NR_MAX_DUAL_RES], fabs(dxi * r));
      v[NR_SUM_XPX] += mpx * xk;
      v[NR_SUM_CX] += ck * xk;
      v[NR_MAX_PX] = wmax(v[NR_MAX_PX], fabs(mpx * dxi));
      v[NR_MAX_C] = wmax(v[NR_MAX_C], fabs(ck * dxi));
      v[NR_MAX_ACCUM] = wmax(v[NR_MAX_ACCUM], fabs((aty + gtz + zbk) * dxi));
    } else if (t < e_hl) {  // y-segment
      const int64_t k = t - e_y;
      T bk = a.data_b(b, k);
      T ax = a.A_x(b, k);
      T dyi = a.delta_inv(b, n + k);
      v[NR_SUM_BY] += bk * a.result_y(b, k);
      T r = -ax + bk;
      a.res_nr_y(b, k) = r;
      v[NR_MAX_RES_Y] = wmax(v[NR_MAX_RES_Y], fabs(r * dyi));
      v[NR_MAX_AX] = wmax(v[NR_MAX_AX], fabs(ax * dyi));
    } else if (t < e_hu) {  // z_l (hl) segment
      const int64_t k = t - e_hl;
      T mask = a.finite_mask_hl(b, k);
      T hl = finite_value(a.data_h_l(b, k), mask);
      T zhl = a.result_z_hl(b, k) * mask;
      v[NR_HL_SUM] += hl * zhl;
      T gx = a.G_x(b, k);
      T sv = a.result_s_hl(b, k) * mask;
      T r = (gx - sv - hl) * mask;
      a.res_nr_z_hl(b, k) = r;
      T dzi = a.delta_inv(b, n + p + k);
      v[NR_HL_RES] = wmax(v[NR_HL_RES], fabs(r * dzi));
      v[NR_HL_GX] = wmax(v[NR_HL_GX], fabs(gx * mask * dzi));
      v[NR_HL_S] = wmax(v[NR_HL_S], fabs(sv * dzi));
    } else if (t < e_xl) {  // z_u (hu) segment
      const int64_t k = t - e_hu;
      T mask = a.finite_mask_hu(b, k);
      T hu = finite_value(a.data_h_u(b, k), mask);
      T zhu = a.result_z_hu(b, k) * mask;
      v[NR_HU_SUM] += hu * zhu;
      T gx = a.G_x(b, k);
      T sv = a.result_s_hu(b, k) * mask;
      T r = (-gx - sv + hu) * mask;
      a.res_nr_z_hu(b, k) = r;
      T dzi = a.delta_inv(b, n + p + k);
      v[NR_HU_RES] = wmax(v[NR_HU_RES], fabs(r * dzi));
      v[NR_HU_GX] = wmax(v[NR_HU_GX], fabs(gx * mask * dzi));
      v[NR_HU_S] = wmax(v[NR_HU_S], fabs(sv * dzi));
    } else if (t < e_xu) {  // z_bl (xl) segment
      const int64_t k = t - e_xl;
      T mask = a.finite_mask_xl(b, k);
      T xl = finite_value(a.data_x_l(b, k), mask);
      T zxl = a.result_z_xl(b, k) * mask;
      v[NR_XL_SUM] += xl * zxl;
      T sv = a.result_s_xl(b, k) * mask;
      T r = (a.x_b_scaling(b, k) * a.result_x(b, k) - sv - xl) * mask;
      a.res_nr_z_xl(b, k) = r;
      T dbi = a.delta_b_inv(b, k);
      v[NR_XL_RES] = wmax(v[NR_XL_RES], fabs(r * dbi));
      v[NR_XL_S] = wmax(v[NR_XL_S], fabs(sv * dbi));
    } else {  // z_bu (xu) segment
      const int64_t k = t - e_xu;
      T mask = a.finite_mask_xu(b, k);
      T xu = finite_value(a.data_x_u(b, k), mask);
      T zxu = a.result_z_xu(b, k) * mask;
      v[NR_XU_SUM] += xu * zxu;
      T sv = a.result_s_xu(b, k) * mask;
      T r = (-a.x_b_scaling(b, k) * a.result_x(b, k) - sv + xu) * mask;
      a.res_nr_z_xu(b, k) = r;
      T dbi = a.delta_b_inv(b, k);
      v[NR_XU_RES] = wmax(v[NR_XU_RES], fabs(r * dbi));
      v[NR_XU_S] = wmax(v[NR_XU_S], fabs(sv * dbi));
    }
  }

  if (!multiblock_reduce<BLOCK, NR_COUNT, kNrKinds>(g, v, sm)) return;

  // ===================== finalize info (finalizing block). The three
  // independent groups of outputs run on lane 0 of different warps (FP64
  // chains with a division each overlap instead of running back to back).
  const T* red = sm.result;
  constexpr int NW = BLOCK / 32;
  const int tid = threadIdx.x;
  const T csi = a.cost_scaling_inv[b];

  if (tid == 0) {  // objectives and duality gap
    T half_xT_Px = T(-0.5) * red[NR_SUM_XPX];
    T cT_x = red[NR_SUM_CX];
    T bT_y = T(0), hl_zhl = T(0), hu_zhu = T(0), xl_zxl = T(0), xu_zxu = T(0);
    if (p > 0) bT_y = red[NR_SUM_BY];
    if (num_hl > 0) hl_zhl = red[NR_HL_SUM];
    if (num_hu > 0) hu_zhu = red[NR_HU_SUM];
    if (num_xl > 0) xl_zxl = red[NR_XL_SUM];
    if (num_xu > 0) xu_zxu = red[NR_XU_SUM];

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
  }

  if (tid == (NW > 1 ? 32 : 0)) {  // primal residual
    T primal_res = T(0);
    T primal_res_rel_norm = T(0);
    if (p > 0) {
      primal_res = wmax(primal_res, red[NR_MAX_RES_Y]);
      primal_res_rel_norm = wmax(primal_res_rel_norm, red[NR_MAX_AX]);
    }
    if (num_hl > 0) {
      primal_res = wmax(primal_res, red[NR_HL_RES]);
      primal_res_rel_norm = wmax(primal_res_rel_norm, wmax(red[NR_HL_GX], red[NR_HL_S]));
    }
    if (num_hu > 0) {
      primal_res = wmax(primal_res, red[NR_HU_RES]);
      primal_res_rel_norm = wmax(primal_res_rel_norm, wmax(red[NR_HU_GX], red[NR_HU_S]));
    }
    if (num_xl > 0) {
      primal_res = wmax(primal_res, red[NR_XL_RES]);
      primal_res_rel_norm = wmax(primal_res_rel_norm, red[NR_XL_S]);
    }
    if (num_xu > 0) {
      primal_res = wmax(primal_res, red[NR_XU_RES]);
      primal_res_rel_norm = wmax(primal_res_rel_norm, red[NR_XU_S]);
    }
    a.info_prev_primal_res[b] = a.info_primal_res[b];  // before it is overwritten
    a.info_primal_res[b] = primal_res;
    T prn = wmax(primal_res_rel_norm, a.constraints_rhs_inf_norm[b]);
    prn = wmax(prn, T(1));
    a.info_primal_res_rel[b] = primal_res / prn;
  }

  if (tid == (NW > 2 ? 64 : 0)) {  // dual residual
    T dual_res = red[NR_MAX_DUAL_RES] * csi;
    T drn_max = wmax(red[NR_MAX_PX], wmax(red[NR_MAX_C], red[NR_MAX_ACCUM]));
    a.info_prev_dual_res[b] = a.info_dual_res[b];  // before it is overwritten
    a.info_dual_res[b] = dual_res;
    T dual_res_rel_norm = wmax(drn_max * csi, T(1));
    a.info_dual_res_rel[b] = dual_res / dual_res_rel_norm;
  }
}

// ------------------------------------------------------------------ update_residuals_r
constexpr uint64_t kRKinds = reduce_kinds(kM, kM, kM, kM);

template <int BLOCK, class T>
__global__ void __launch_bounds__(BLOCK)
update_residuals_r_kernel(MultiblockGrid g, int32_t n, int32_t num_duals, UpdateResidualsRArgs<T> a) {
  __shared__ MultiblockStorage<BLOCK, 4, T> sm;
  const int64_t b = g.problem();
  const int64_t k0 = g.first<BLOCK>();
  const int64_t ks = g.stride<BLOCK>();
  const T rho_b = a.rho[b];
  const T delta_b = a.delta[b];

  // v = {max |diff_x|, max |res_x| * unscale, max |diff_d|, max |res_d| * unscale}
  T v[4] = {T(0), T(0), T(0), T(0)};
  // One pass over the concatenated row [x | duals].
  const int64_t total = int64_t(n) + num_duals;
  for (int64_t t = k0; t < total; t += ks) {
    if (t < n) {
      const int64_t k = t;
      T diff_x = a.result_x(b, k) - a.prox_x(b, k);
      T new_res_x = a.res_nr_x(b, k) - rho_b * diff_x;
      a.res_x(b, k) = new_res_x;
      v[0] = wmax(v[0], fabs(diff_x));
      v[1] = wmax(v[1], fabs(new_res_x) * a.dual_res_unscale_factor(b, k));
    } else {
      const int64_t k = t - n;
      T diff_d = a.result_duals(b, k) - a.prox_duals(b, k);
      T new_res_d = a.res_nr_duals(b, k) + delta_b * diff_d;
      a.res_duals(b, k) = new_res_d;
      v[2] = wmax(v[2], fabs(diff_d));
      v[3] = wmax(v[3], fabs(new_res_d) * a.primal_res_unscale_factor(b, k));
    }
  }
  if (!multiblock_reduce<BLOCK, 4, kRKinds>(g, v, sm)) return;
  if (threadIdx.x != 0) return;

  const T dual_res_reg = sm.result[1];
  a.dual_prox_inf[b] = sm.result[0] * rho_b;
  a.dual_res_reg[b] = dual_res_reg;

  T primal_prox_inf = sm.result[2] * delta_b;
  T primal_res_reg = sm.result[3];
  a.primal_prox_inf[b] = primal_prox_inf;
  a.primal_res_reg[b] = primal_res_reg;

  if (a.primal_res_rel[b] > T(0)) {
    a.primal_res_reg_rel[b] = primal_res_reg * a.primal_res_rel[b] / a.primal_res[b];
  } else {
    a.primal_res_reg_rel[b] = primal_res_reg;
  }
  if (a.dual_res_rel[b] > T(0)) {
    a.dual_res_reg_rel[b] = dual_res_reg * a.dual_res_rel[b] / a.dual_res[b];
  } else {
    a.dual_res_reg_rel[b] = dual_res_reg;
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
void launch_update_residual_nr(cudaStream_t s, const Dims& d, const UpdateResidualNrArgs<T>& a,
                               const ReductionWorkspace& ws) {
  if (d.B <= 0) return;
  const int64_t width = int64_t(d.n) + d.p + d.num_ineq();  // concatenated row
  const MbLaunch L = mb_launch(ws, d.B, width, NR_COUNT);
  const MultiblockGrid g = multiblock_grid(ws, L.nb);
  dispatch_block(L.block, [&](auto bd) {
    constexpr int BLOCK = decltype(bd)::value;
    update_residual_nr_kernel<BLOCK, T><<<multiblock_dim(d.B, L.nb), BLOCK, 0, s>>>(g, d, a);
  });
  CUPIQP_CUDA_CHECK_LAUNCH();
}

template <class T>
void launch_update_residuals_r(cudaStream_t s, int32_t B, int32_t n, int32_t num_duals,
                               const UpdateResidualsRArgs<T>& a, const ReductionWorkspace& ws) {
  if (B <= 0) return;
  const int64_t width = int64_t(n) + num_duals;  // concatenated row
  const MbLaunch L = mb_launch(ws, B, width, 4);
  const MultiblockGrid g = multiblock_grid(ws, L.nb);
  dispatch_block(L.block, [&](auto bd) {
    constexpr int BLOCK = decltype(bd)::value;
    update_residuals_r_kernel<BLOCK, T><<<multiblock_dim(B, L.nb), BLOCK, 0, s>>>(g, n, num_duals, a);
  });
  CUPIQP_CUDA_CHECK_LAUNCH();
}

#define CUPIQP_IPM_RESIDUAL_INSTANTIATE(T)                                                         \
  template void launch_init_guess_rhs<T>(cudaStream_t, const Dims&, const InitGuessRhsArgs<T>&);  \
  template void launch_prepare_zu_minus_zl_and_zbu_minus_zbl<T>(                                   \
      cudaStream_t, const Dims&, RowsView<T>, RowsView<T>, RowsView<T>, RowsView<T>, RowsView<T>,  \
      RowsView<T>, RowsView<T>);                                                                   \
  template void launch_update_residual_nr<T>(cudaStream_t, const Dims&,                            \
                                             const UpdateResidualNrArgs<T>&, const ReductionWorkspace&); \
  template void launch_update_residuals_r<T>(cudaStream_t, int32_t, int32_t, int32_t,              \
                                             const UpdateResidualsRArgs<T>&, const ReductionWorkspace&);

CUPIQP_IPM_RESIDUAL_INSTANTIATE(double)
#undef CUPIQP_IPM_RESIDUAL_INSTANTIATE

}  // namespace cupiqp
