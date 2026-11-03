// M3 ruiz: sparse Ruiz equilibration (reference cupiqp/preconditioner.py
// RuizEquilibration, cupiqp/preconditioner_kernels.py,
// cupiqp/sparse/sparse_preconditioner.py and its kernels). Constants: min_scaling
// 1e-4, max_scaling 1e4, convergence_tol 1e-3 (reference defaults).
//
// INTERFACE CONTRACT (set by the integration owner).
#pragma once

#include <cuda_runtime_api.h>

#include "common/buffer.hpp"
#include "common/views.hpp"

namespace cupiqp {

class RuizEquilibration {
 public:
  // Allocate the state (all factors 1, x_b_scaling = x_b_scaling_init).
  // x_b_scaling_init aliases data.active_x_bound when box bounds exist, else a
  // buffer of ones (reference behavior: the alias follows bound updates).
  void init(const SparseDataView<double>& data, cudaStream_t s);

  ScalingView<double> view();

  // Reference scale_data: up to max_iter Ruiz passes on `data` in place, then
  // calc_scaling_inv_and_scale_bounds. The reference breaks out of the pass
  // loop on the host when the convergence measure < convergence_tol; here the
  // convergence kernel sets a device flag and later passes are no-ops, so no
  // host sync happens and the same passes run. passes_run() counts them.
  void scale_data(const SparseDataView<double>& data, bool scale_cost, int max_iter, cudaStream_t s);

  // Reference unscale_data, restricted to blocks NOT in `replaced` (their old
  // values are about to be overwritten, so unscaling them is dead work). The
  // x_b_scaling update (x_b_scaling *= delta_b_inv * d_x_inv) always runs.
  void unscale_data(const SparseDataView<double>& data, const UpdateMask& replaced, cudaStream_t s);
  void reuse_scaling(const SparseDataView<double>& data, cudaStream_t s);  // reference reuse_scaling
  void reset(cudaStream_t s);                                            // reference reset
  void refresh_x_b_scaling(cudaStream_t s);                              // reference refresh_x_b_scaling

  // Reference unscale_solution / scale_solution on a Variables buffer.
  void unscale_solution(const VariablesView<double>& result, cudaStream_t s);
  void scale_solution(const VariablesView<double>& result, cudaStream_t s);

  // Reference compute_constraints_rhs_inf_norm_unscaled -> out (B,).
  void compute_constraints_rhs_inf_norm_unscaled(const SparseDataView<double>& data, double* out,
                                                 cudaStream_t s);

  // Device int32 counter of Ruiz passes executed by the last scale_data.
  const int32_t* passes_run() const { return passes_run_.data(); }

 private:
  Dims dims_;
  CudaBuffer<double> delta_, delta_inv_, delta_b_, delta_b_inv_, cost_scaling_, cost_scaling_inv_;
  CudaBuffer<double> x_b_scaling_, x_b_scaling_ones_, dual_res_unscale_, primal_res_unscale_;
  CudaBuffer<double> delta_iter_, delta_b_iter_, conv_buf_;
  CudaBuffer<int32_t> done_flag_, passes_run_;
  const double* x_b_scaling_init_ = nullptr;
};

}  // namespace cupiqp
