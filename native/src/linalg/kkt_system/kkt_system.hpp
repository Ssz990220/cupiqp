// M6 kkt_system: condensed KKT system on top of the sparse KKT matrix
// (reference cupiqp/kkt_systems.py KKTSystem, cupiqp/kkt_systems_kernels.py,
// cupiqp/kkt_solver.py KKTSolverBase finite check, and the solve() part of
// cupiqp/sparse/sparse_kkt_solver.py: rhs assembly -> cuDSS solve -> split).
//
// INTERFACE CONTRACT (set by the integration owner).
#pragma once

#include <cuda_runtime_api.h>

#include <functional>

#include "common/buffer.hpp"
#include "common/graph.hpp"
#include "common/multiblock.hpp"
#include "common/views.hpp"
#include "linalg/kkt_assembly/kkt_assembly.hpp"
#include "linalg/kkt_system/kkt_linear_solver.hpp"
#include "linalg/spmv/spmv.hpp"

namespace cupiqp {

class KktSystem {
 public:
  // Allocate x_reg, z_reg, z_reg_inv, rho/delta copies, rhs_x_bar, rhs_z_bar,
  // work buffers, m_s_all, m_z_inv_all, w_delta_inv_all, updated_rhs_z_all,
  // factor_status / solve_status (int32, zero), refinement buffers.
  // `kkt` must be initialized and `solver` initialized on kkt's pattern/values
  // (cuDSS: and analyzed) by the caller; `ops` initialized on the same data. Every
  // pointer must outlive this object. `settings` is unused (runtime values are
  // read from dev_settings on the device).
  void init(const Dims& dims, const Settings& settings, KktAssembly* kkt, KktLinearSolver* solver,
            SparseOperators* ops, DeviceSettingsView<double> dev_settings, LoopContext* loops,
            cudaStream_t s);

  // Reference update_scalings_and_factor: update_regularizations kernel, KKT
  // diagonal update (KktAssembly::update_diagonal), backend factorization.
  // cuDSS: factor_status stays 0 (reference: the cuDSS outcome is never read).
  // Reduced backend: factor_status is the Cholesky outcome (reference dense
  // backend semantics), which drives the factor-retry kernel.
  void update_scalings_and_factor(const SparseDataView<double>& data, const ScalingView<double>& scaling,
                                  bool iterative_refinement, const double* rho, const double* delta,
                                  const VariablesView<double>& vars, cudaStream_t s);

  // Reference KKTSystem.solve (forward, transpose = false): eliminate slacks
  // and duals, assemble [rhs_x_bar | rhs_y | rhs_z_bar] into the cuDSS rhs,
  // cuDSS solve, split into lhs.x / lhs.y / work_z, write solve_status
  // (finite check of that condensed solution, as the reference backend does),
  // optional iterative refinement (each correction solve rewrites
  // solve_status), recover duals and slacks. B == 1: rhs.y, lhs.x, lhs.y are
  // used as contiguous rows.
  void solve(const SparseDataView<double>& data, const ScalingView<double>& scaling,
             const VariablesView<double>& rhs, const VariablesView<double>& lhs, cudaStream_t s);

  // (B,), stable address: the backend's status when it reports one, else a
  // buffer that stays zero.
  const int32_t* factor_status() const {
    return solver_ && solver_->factor_status() ? solver_->factor_status() : factor_status_.data();
  }
  const int32_t* solve_status() const { return solve_status_.data(); }    // (B,)

 private:
  // Reference KKTSolverBase.solve of the sparse backend: fused rhs assembly
  // into the cuDSS rhs, cuDSS solve, fused split, finite check.
  void kkt_solve(const double* rhs_x, const double* rhs_y, const double* rhs_z, double* lhs_x, double* lhs_y,
                 double* lhs_z, cudaStream_t s);
  void prepare_rhs(const ScalingView<double>& scaling, const VariablesView<double>& rhs, cudaStream_t s);
  void recover_lhs(const ScalingView<double>& scaling, const VariablesView<double>& rhs,
                   const VariablesView<double>& lhs, cudaStream_t s);
  // Reference iterative_refinement on (rhs_x_bar, rhs.y, rhs_z_bar) ->
  // (lhs.x, lhs.y, work_z); arguments are passed through ir_* members so the
  // loop bodies are built once in init().
  void iterative_refinement(cudaStream_t s);
  void condensed_kkt_apply(const double* lhs_x, const double* lhs_y, const double* lhs_z, const double* rhs_x,
                           const double* rhs_y, const double* rhs_z, double* out_x, double* out_y, double* out_z,
                           bool residual, cudaStream_t s);
  void ir_evaluate_and_decide(const double* cand_x, const double* cand_y, const double* cand_z, cudaStream_t s);

  Dims dims_;
  KktAssembly* kkt_ = nullptr;
  KktLinearSolver* solver_ = nullptr;
  SparseOperators* ops_ = nullptr;
  LoopContext* loops_ = nullptr;
  DeviceSettingsView<double> dev_settings_{};
  bool use_ir_ = false;
  CudaBuffer<double> rho_, delta_, x_reg_, z_reg_, z_reg_inv_, rhs_x_bar_, rhs_z_bar_, work_x_, work_z_;
  CudaBuffer<double> m_s_all_, m_z_inv_all_, w_delta_inv_all_, updated_rhs_z_all_;
  CudaBuffer<int32_t> factor_status_, solve_status_;
  ReductionWorkspaceBuffers reduce_ws_;  // finite check (multi-block reduction)
  // Iterative refinement: condensed-operator products, residual / candidate
  // (B, n + p + m), device loop state [step, accept, correct, loop].
  CudaBuffer<double> work_Px_, work_GTz_, work_Ax_, work_Gx_, norm_buf_;
  CudaBuffer<double> ir_error_xyz_, ir_delta_xyz_, refine_rhs_norm_, refine_error_;
  CudaBuffer<int32_t> refine_state_;
  CudaPinnedBuffer<int32_t> if_flag_;  // host mirror for capture_if outside capture
  const double* active_G_row_ = nullptr;  // data.active_G_row of the current IR call
  const double* ir_rhs_y_ = nullptr;      // rhs.y of the current IR call
  double* ir_lhs_x_ = nullptr;            // lhs.x of the current IR call
  double* ir_lhs_y_ = nullptr;            // lhs.y of the current IR call
  std::function<void(cudaStream_t)> ir_correct_, ir_first_step_, ir_later_step_;
};

}  // namespace cupiqp
