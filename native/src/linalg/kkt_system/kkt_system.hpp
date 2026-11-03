// M6 kkt_system: condensed KKT system on top of the sparse KKT matrix
// (reference cupiqp/kkt_systems.py KKTSystem, cupiqp/kkt_systems_kernels.py,
// cupiqp/kkt_solver.py KKTSolverBase finite check, and the solve() part of
// cupiqp/sparse/sparse_kkt_solver.py: rhs assembly -> cuDSS solve -> split).
//
// INTERFACE CONTRACT (set by the integration owner).
#pragma once

#include <cuda_runtime_api.h>

#include "common/buffer.hpp"
#include "common/graph.hpp"
#include "common/views.hpp"
#include "linalg/cudss/cudss_solver.hpp"
#include "linalg/kkt_assembly/kkt_assembly.hpp"
#include "linalg/spmv/spmv.hpp"

namespace cupiqp {

class KktSystem {
 public:
  // Allocate x_reg, z_reg, z_reg_inv, rho/delta copies, rhs_x_bar, rhs_z_bar,
  // work buffers, m_s_all, m_z_inv_all, w_delta_inv_all, updated_rhs_z_all,
  // factor_status / solve_status (int32, zero), refinement buffers.
  void init(const Dims& dims, const Settings& settings, KktAssembly* kkt, CudssSolver* cudss,
            SparseOperators* ops, DeviceSettingsView<double> dev_settings, LoopContext* loops,
            cudaStream_t s);

  // Reference update_scalings_and_factor: update_regularizations kernel, KKT
  // diagonal update (KktAssembly::update_diagonal), cuDSS factorization.
  // factor_status stays 0 (reference: the cuDSS outcome is never read).
  void update_scalings_and_factor(const SparseDataView<double>& data, const ScalingView<double>& scaling,
                                  bool iterative_refinement, const double* rho, const double* delta,
                                  const VariablesView<double>& vars, cudaStream_t s);

  // Reference KKTSystem.solve (forward, transpose = false): eliminate slacks
  // and duals, assemble [rhs_x_bar | rhs_y | rhs_z_bar] into the cuDSS rhs,
  // cuDSS solve, split into lhs.x / lhs.y / work_z, optional iterative
  // refinement, recover duals and slacks, write solve_status (finite check).
  void solve(const SparseDataView<double>& data, const ScalingView<double>& scaling,
             const VariablesView<double>& rhs, const VariablesView<double>& lhs, cudaStream_t s);

  const int32_t* factor_status() const { return factor_status_.data(); }  // (B,)
  const int32_t* solve_status() const { return solve_status_.data(); }    // (B,)

 private:
  Dims dims_;
  KktAssembly* kkt_ = nullptr;
  CudssSolver* cudss_ = nullptr;
  SparseOperators* ops_ = nullptr;
  LoopContext* loops_ = nullptr;
  DeviceSettingsView<double> dev_settings_{};
  bool use_ir_ = false;
  CudaBuffer<double> rho_, delta_, x_reg_, z_reg_, z_reg_inv_, rhs_x_bar_, rhs_z_bar_, work_x_, work_z_;
  CudaBuffer<double> m_s_all_, m_z_inv_all_, w_delta_inv_all_, updated_rhs_z_all_;
  CudaBuffer<int32_t> factor_status_, solve_status_;
};

}  // namespace cupiqp
