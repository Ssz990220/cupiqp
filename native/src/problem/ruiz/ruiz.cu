// M3 ruiz: host side of RuizEquilibration (reference cupiqp/preconditioner.py
// RuizEquilibration and cupiqp/sparse/sparse_preconditioner.py
// SparseRuizEquilibration). Kernels are in ruiz_kernels.cuh.
#include "problem/ruiz/ruiz.hpp"

#include <algorithm>

#include "common/check.hpp"
#include "common/reduce.cuh"
#include "problem/ruiz/ruiz_kernels.cuh"

namespace cupiqp {

namespace {

using ruiz_detail::BlockFlags;
using ruiz_detail::Csr;

constexpr int kBlock = 256;

Csr csr_of(const CsrView<double>& M) { return Csr{M.indptr, M.indices, M.row_idx, M.nnz}; }

int32_t num_duals_of(const Dims& d) { return d.p + d.num_ineq(); }

}  // namespace

void RuizEquilibration::init(const SparseDataView<double>& data, cudaStream_t s) {
  dims_ = data.dims;
  const Dims& d = dims_;
  CUPIQP_REQUIRE(d.B >= 1 && d.n >= 0 && d.p >= 0 && d.m >= 0, "RuizEquilibration::init: invalid dims");
  const size_t B = d.B, n = d.n, W = size_t(d.n) + d.p + d.m;
  const size_t nd = num_duals_of(d);
  delta_.allocate(B * W);
  delta_inv_.allocate(B * W);
  delta_b_.allocate(B * n);
  delta_b_inv_.allocate(B * n);
  cost_scaling_.allocate(B);
  cost_scaling_inv_.allocate(B);
  x_b_scaling_.allocate(B * n);
  dual_res_unscale_.allocate(B * n);
  primal_res_unscale_.allocate(B * nd);
  delta_iter_.allocate(B * W);
  delta_b_iter_.allocate(B * n);
  conv_buf_.allocate(1);
  P_norms_.allocate(B * n);
  gamma_buf_.allocate(B);
  ones_.allocate(B);
  done_flag_.allocate(2);
  passes_run_.allocate(1);

  if (data.active_x_bound != nullptr) {
    x_b_scaling_init_ = data.active_x_bound;
  } else {
    x_b_scaling_ones_.allocate(B * n);
    x_b_scaling_init_ = x_b_scaling_ones_.data();
  }

  // Constant buffers of ones (ones_: the reference cost factor of a Ruiz pass).
  ruiz_detail::ruiz_fill_kernel<double><<<grid_for(int64_t(B)), kBlock, 0, s>>>(ones_.data(), int64_t(B), 1.0);
  CUPIQP_CUDA_CHECK_LAUNCH();
  if (!x_b_scaling_ones_.empty()) {
    ruiz_detail::ruiz_fill_kernel<double><<<grid_for(int64_t(B * n)), kBlock, 0, s>>>(
        x_b_scaling_ones_.data(), int64_t(B * n), 1.0);
    CUPIQP_CUDA_CHECK_LAUNCH();
  }
  reset(s);
  done_flag_.zero_async(s);
  passes_run_.zero_async(s);
  conv_buf_.zero_async(s);
  P_norms_.zero_async(s);
}

ScalingView<double> RuizEquilibration::view() {
  ScalingView<double> v;
  v.delta = delta_.data();
  v.delta_inv = delta_inv_.data();
  v.delta_b = delta_b_.data();
  v.delta_b_inv = delta_b_inv_.data();
  v.cost_scaling = cost_scaling_.data();
  v.cost_scaling_inv = cost_scaling_inv_.data();
  v.x_b_scaling = x_b_scaling_.data();
  v.x_b_scaling_init = x_b_scaling_init_;
  v.dual_res_unscale_factor = dual_res_unscale_.data();
  v.primal_res_unscale_factor = primal_res_unscale_.data();
  return v;
}

void RuizEquilibration::reset(cudaStream_t s) {
  const Dims& d = dims_;
  const int32_t W = d.n + d.p + d.m, nd = num_duals_of(d);
  const int32_t K = std::max<int32_t>({W, nd, 1});
  ruiz_detail::ruiz_reset_kernel<double><<<grid_for(int64_t(d.B) * K), kBlock, 0, s>>>(
      d.B, d.n, W, nd, K, delta_.data(), delta_inv_.data(), delta_b_.data(), delta_b_inv_.data(),
      cost_scaling_.data(), cost_scaling_inv_.data(), x_b_scaling_.data(), x_b_scaling_init_,
      dual_res_unscale_.data(), primal_res_unscale_.data());
  CUPIQP_CUDA_CHECK_LAUNCH();
}

// ------------------------------------------------------------ backend hooks
void RuizEquilibration::compute_kkt_norms(const SparseDataView<double>& data, const int32_t* done,
                                          cudaStream_t s) {
  const Dims& d = dims_;
  const int32_t R = std::max({d.n, d.p, d.m});
  if (R == 0) return;
  ruiz_detail::ruiz_row_norm_kernel<double><<<grid_for(int64_t(d.B) * R), kBlock, 0, s>>>(
      d.B, d.n, d.p, d.m, R, data.P.values, csr_of(data.P), data.A.values, csr_of(data.A), data.G.values,
      csr_of(data.G), x_b_scaling_.data(), delta_iter_.data(), delta_b_iter_.data(), done);
  CUPIQP_CUDA_CHECK_LAUNCH();
  const int32_t nnz_max = std::max(d.p > 0 ? data.A.nnz : 0, d.m > 0 ? data.G.nnz : 0);
  if (nnz_max > 0 && (d.p > 0 || d.m > 0)) {
    ruiz_detail::ruiz_col_norm_kernel<double><<<grid_for(int64_t(d.B) * nnz_max), kBlock, 0, s>>>(
        d.B, d.n, d.p, d.m, nnz_max, data.A.values, csr_of(data.A), data.G.values, csr_of(data.G),
        delta_iter_.data(), done);
    CUPIQP_CUDA_CHECK_LAUNCH();
  }
}

void RuizEquilibration::scale_matrices(const SparseDataView<double>& data, const double* dvec,
                                       const double* cost_factor, Blocks blocks, const int32_t* done,
                                       cudaStream_t s) {
  const Dims& d = dims_;
  const int32_t R = std::max({d.n, d.p, d.m});
  if (R == 0) return;
  if (!(blocks.P || blocks.c || blocks.A || blocks.G)) return;
  BlockFlags f{blocks.P, blocks.c, blocks.A && d.p > 0, blocks.G && d.m > 0, false, false, false, false, false};
  ruiz_detail::ruiz_scale_matrices_kernel<double><<<grid_for(int64_t(d.B) * R), kBlock, 0, s>>>(
      d.B, d.n, d.p, d.m, R, data.P.values, csr_of(data.P), data.A.values, csr_of(data.A), data.G.values,
      csr_of(data.G), data.c, dvec, cost_factor, f, done);
  CUPIQP_CUDA_CHECK_LAUNCH();
}

void RuizEquilibration::apply_cost_scaling(const SparseDataView<double>& data, const int32_t* done,
                                           cudaStream_t s) {
  const Dims& d = dims_;
  const int32_t nnz_P = data.P.nnz;
  P_norms_.zero_async(s);  // scratch: harmless after convergence
  if (nnz_P > 0) {
    ruiz_detail::ruiz_P_norms_kernel<double><<<grid_for(int64_t(d.B) * nnz_P), kBlock, 0, s>>>(
        d.B, d.n, data.P.values, csr_of(data.P), P_norms_.data(), done);
    CUPIQP_CUDA_CHECK_LAUNCH();
  }
  dispatch_block(reduction_block_dim(d.n), [&](auto blk) {
    constexpr int BLOCK = decltype(blk)::value;
    ruiz_detail::ruiz_compute_gamma_kernel<BLOCK, double><<<d.B, BLOCK, 0, s>>>(
        d.n, P_norms_.data(), data.c, gamma_buf_.data(), kMinScaling, kMaxScaling, done);
  });
  CUPIQP_CUDA_CHECK_LAUNCH();
  const int32_t K = std::max(nnz_P, d.n);
  if (K > 0) {
    ruiz_detail::ruiz_apply_gamma_kernel<double><<<grid_for(int64_t(d.B) * K), kBlock, 0, s>>>(
        d.B, d.n, nnz_P, K, data.P.values, data.c, cost_scaling_.data(), gamma_buf_.data(), done);
    CUPIQP_CUDA_CHECK_LAUNCH();
  }
}

void RuizEquilibration::scale_bounds(const SparseDataView<double>& data, const double* dvec, const double* d_b,
                                     Blocks blocks, cudaStream_t s) {
  const Dims& d = dims_;
  const int32_t W = d.n + d.p + d.m;
  if (W == 0) return;
  BlockFlags f{false, false, false, false, blocks.b && d.p > 0, blocks.h_l && d.has_h_l(),
               blocks.h_u && d.has_h_u(), blocks.x_l && d.has_x_l(), blocks.x_u && d.has_x_u()};
  if (!(f.b || f.h_l || f.h_u || f.x_l || f.x_u)) return;
  ruiz_detail::ruiz_bounds_kernel<double><<<grid_for(int64_t(d.B) * W), kBlock, 0, s>>>(
      d.B, d.n, d.p, d.m, dvec, d_b, data.b, data.h_l, data.h_u, data.x_l, data.x_u, f);
  CUPIQP_CUDA_CHECK_LAUNCH();
}

// ------------------------------------------------------------ scale_data
void RuizEquilibration::scale_data(const SparseDataView<double>& data, bool scale_cost, int max_iter,
                                   cudaStream_t s) {
  const Dims& d = dims_;
  const int32_t W = d.n + d.p + d.m;
  int32_t* done = done_flag_.data();

  ruiz_detail::ruiz_begin_kernel<<<1, 1, 0, s>>>(done, passes_run_.data(), conv_buf_.data());
  CUPIQP_CUDA_CHECK_LAUNCH();

  for (int it = 0; it < max_iter; ++it) {
    compute_kkt_norms(data, done, s);
    if (W > 0) {
      ruiz_detail::ruiz_clamp_rsqrt_kernel<double><<<grid_for(int64_t(d.B) * W), kBlock, 0, s>>>(
          d.B, d.n, W, delta_iter_.data(), delta_b_iter_.data(), kMinScaling, kMaxScaling, done);
      CUPIQP_CUDA_CHECK_LAUNCH();
    }
    scale_matrices(data, delta_iter_.data(), ones_.data(), Blocks{}, done, s);
    if (W > 0) {
      ruiz_detail::ruiz_accumulate_deltas_kernel<double><<<grid_for(int64_t(d.B) * W), kBlock, 0, s>>>(
          d.B, d.n, W, delta_.data(), delta_b_.data(), x_b_scaling_.data(), delta_iter_.data(),
          delta_b_iter_.data(), done);
      CUPIQP_CUDA_CHECK_LAUNCH();
    }
    if (scale_cost) apply_cost_scaling(data, done, s);
    dispatch_block(reduction_block_dim(W), [&](auto blk) {
      constexpr int BLOCK = decltype(blk)::value;
      ruiz_detail::ruiz_conv_check_kernel<BLOCK, double><<<d.B, BLOCK, 0, s>>>(
          d.B, d.n, W, delta_iter_.data(), delta_b_iter_.data(), conv_buf_.data(), done, passes_run_.data(),
          kConvergenceTol);
    });
    CUPIQP_CUDA_CHECK_LAUNCH();
  }

  const int32_t nd = num_duals_of(d);
  const int32_t K = W + 1 + nd;
  ruiz_detail::ruiz_finalize_kernel<double><<<grid_for(int64_t(d.B) * K), kBlock, 0, s>>>(
      d.B, d.n, d.p, d.m, d.num_hl, d.num_hu, d.num_xl, d.num_xu, delta_.data(), delta_inv_.data(),
      delta_b_.data(), delta_b_inv_.data(), cost_scaling_.data(), cost_scaling_inv_.data(), data.b, data.h_l,
      data.h_u, data.x_l, data.x_u, dual_res_unscale_.data(), primal_res_unscale_.data());
  CUPIQP_CUDA_CHECK_LAUNCH();
}

// ------------------------------------------------------------ unscale / reuse
void RuizEquilibration::unscale_data(const SparseDataView<double>& data, const UpdateMask& replaced,
                                     cudaStream_t s) {
  const Dims& d = dims_;
  // Every block is independent: each element of P, c, A, G and of every bound
  // vector depends only on its own old value and the stored factors, so a
  // block that update() overwrites can be skipped without changing the others.
  Blocks keep;
  keep.P = !replaced.P;
  keep.c = !replaced.c;
  keep.A = !replaced.A;
  keep.G = !replaced.G;
  keep.b = !replaced.b;
  keep.h_l = !replaced.h_l;
  keep.h_u = !replaced.h_u;
  keep.x_l = !replaced.x_l;
  keep.x_u = !replaced.x_u;
  scale_matrices(data, delta_inv_.data(), cost_scaling_inv_.data(), keep, nullptr, s);
  scale_bounds(data, delta_inv_.data(), delta_b_inv_.data(), keep, s);
  // x_b_scaling *= delta_b_inv * d_x_inv (always).
  if (d.n > 0) {
    ruiz_detail::ruiz_product3_kernel<double><<<grid_for(int64_t(d.B) * d.n), kBlock, 0, s>>>(
        d.B, d.n, d.n + d.p + d.m, x_b_scaling_.data(), x_b_scaling_.data(), delta_b_inv_.data(),
        delta_inv_.data());
    CUPIQP_CUDA_CHECK_LAUNCH();
  }
}

void RuizEquilibration::reuse_scaling(const SparseDataView<double>& data, cudaStream_t s) {
  scale_matrices(data, delta_.data(), cost_scaling_.data(), Blocks{}, nullptr, s);
  scale_bounds(data, delta_.data(), delta_b_.data(), Blocks{}, s);
  refresh_x_b_scaling(s);
}

void RuizEquilibration::refresh_x_b_scaling(cudaStream_t s) {
  const Dims& d = dims_;
  if (d.n == 0) return;
  ruiz_detail::ruiz_product3_kernel<double><<<grid_for(int64_t(d.B) * d.n), kBlock, 0, s>>>(
      d.B, d.n, d.n + d.p + d.m, x_b_scaling_.data(), x_b_scaling_init_, delta_b_.data(), delta_.data());
  CUPIQP_CUDA_CHECK_LAUNCH();
}

// ------------------------------------------------------------ solution
void RuizEquilibration::unscale_solution(const VariablesView<double>& r, cudaStream_t s) {
  const Dims& d = dims_;
  const int32_t K = d.num_primal() + d.num_dual();
  if (K == 0) return;
  ruiz_detail::ruiz_solution_kernel<double><<<grid_for(int64_t(d.B) * K), kBlock, 0, s>>>(
      d.B, d.n, d.p, d.m, d.num_hl, d.num_hu, d.num_xl, d.num_xu, r.primal, r.dual, delta_.data(),
      delta_inv_.data(), delta_b_.data(), delta_b_inv_.data(), cost_scaling_inv_.data());
  CUPIQP_CUDA_CHECK_LAUNCH();
}

void RuizEquilibration::scale_solution(const VariablesView<double>& r, cudaStream_t s) {
  const Dims& d = dims_;
  const int32_t K = d.num_primal() + d.num_dual();
  if (K == 0) return;
  // Same kernel with every factor replaced by its inverse (reference).
  ruiz_detail::ruiz_solution_kernel<double><<<grid_for(int64_t(d.B) * K), kBlock, 0, s>>>(
      d.B, d.n, d.p, d.m, d.num_hl, d.num_hu, d.num_xl, d.num_xu, r.primal, r.dual, delta_inv_.data(),
      delta_.data(), delta_b_inv_.data(), delta_b_.data(), cost_scaling_.data());
  CUPIQP_CUDA_CHECK_LAUNCH();
}

void RuizEquilibration::compute_constraints_rhs_inf_norm_unscaled(const SparseDataView<double>& data,
                                                                  double* out, cudaStream_t s) {
  const Dims& d = dims_;
  dispatch_block(reduction_block_dim(std::max({d.n, d.p, d.m})), [&](auto blk) {
    constexpr int BLOCK = decltype(blk)::value;
    ruiz_detail::ruiz_rhs_inf_norm_kernel<BLOCK, double><<<d.B, BLOCK, 0, s>>>(
        d.n, d.p, d.m, d.num_hl, d.num_hu, d.num_xl, d.num_xu, delta_inv_.data(), delta_b_inv_.data(), data.b,
        data.h_l, data.h_u, data.x_l, data.x_u, data.finite_mask_all, d.num_ineq(), out);
  });
  CUPIQP_CUDA_CHECK_LAUNCH();
}

}  // namespace cupiqp
