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
  // x_b_scaling_init aliases data.active_x_bound whenever that buffer is
  // given (the reference SparseSolver always passes Data.active_x_bound,
  // which is all zeros when no box side exists), else a buffer of ones (the
  // reference default for active_x_bound=None). The alias follows bound
  // updates, as in the reference.
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

  const Dims& dims() const { return dims_; }

  // Constants (reference RuizEquilibration defaults).
  static constexpr double kMinScaling = 1e-4;
  static constexpr double kMaxScaling = 1e4;
  static constexpr double kConvergenceTol = 1e-3;

 private:
  // Which blocks scale_matrices / the bound kernels touch (unscale_data skips
  // the blocks that update() is about to overwrite).
  struct Blocks {
    bool P = true, c = true, A = true, G = true;
    bool b = true, h_l = true, h_u = true, x_l = true, x_u = true;
  };
  // Reference SparseRuizEquilibration hooks. `done` (nullable) turns every
  // kernel into a no-op once the device early-exit flag is set.
  void compute_kkt_norms(const SparseDataView<double>& data, const int32_t* done, cudaStream_t s);
  void scale_matrices(const SparseDataView<double>& data, const double* d, const double* cost_factor,
                      Blocks blocks, const int32_t* done, cudaStream_t s);
  void apply_cost_scaling(const SparseDataView<double>& data, const int32_t* done, cudaStream_t s);
  void scale_bounds(const SparseDataView<double>& data, const double* d, const double* d_b, Blocks blocks,
                    cudaStream_t s);

  Dims dims_;
  CudaBuffer<double> delta_, delta_inv_, delta_b_, delta_b_inv_, cost_scaling_, cost_scaling_inv_;
  CudaBuffer<double> x_b_scaling_, x_b_scaling_ones_, dual_res_unscale_, primal_res_unscale_;
  CudaBuffer<double> delta_iter_, delta_b_iter_, conv_buf_;
  CudaBuffer<double> P_norms_, gamma_buf_, ones_;  // cost-scaling workspace, (B, n), (B,), (B,)
  // done_flag_[0]: early-exit flag; done_flag_[1]: block ticket of the
  // convergence kernel (last block of the batch decides).
  CudaBuffer<int32_t> done_flag_, passes_run_;
  const double* x_b_scaling_init_ = nullptr;
};

}  // namespace cupiqp
