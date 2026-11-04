// Reduced (Schur-complement) KKT backend: an opt-in replacement of cuDSS
// inside KktSystem (Settings::kkt_backend = KktBackend::Reduced).
//
// The condensed KKT matrix of KktAssembly,
//   K = [[P + diag(x_reg), A^T, G^T], [A, -delta I, 0], [G, 0, -diag(z_reg)]]
// (inactive z rows: diagonal -1 and a zeroed G row), is solved by eliminating
// the y / z rows. With M = [A; G] and w = -1 / diag(K) on the y / z rows
// (1 / delta, 1 / z_reg, 1 on inactive rows):
//   H  = P + diag(x_reg) + M^T diag(w) M             (dense n x n, SPD)
//   H dx = r_x + M^T (w * r_m),   d_m = w * (M dx - r_m).
// Everything is read straight from the K values buffer through index maps
// built at init, so the backend sees exactly the matrix cuDSS would factor.
// H is assembled by a deterministic sparse kernel and factored by a blocked
// right-looking Cholesky (custom kernels, no cuBLAS / cuSOLVER); solves use
// a custom one-block triangular solve. factor() writes the Cholesky outcome
// to factor_status() (reference dense backend semantics: nonzero on a
// non-positive or non-finite pivot), which drives the factor-retry kernel.
// Batch size 1. No allocation after init; capturable into CUDA graphs,
// including the conditional bodies of the whole-solve graph.
#pragma once

#include <cuda_runtime_api.h>

#include <vector>

#include "common/buffer.hpp"
#include "linalg/kkt_assembly/kkt_assembly.hpp"
#include "linalg/kkt_assembly/kkt_pattern.hpp"
#include "linalg/kkt_system/kkt_linear_solver.hpp"

namespace cupiqp {

class ReducedKktSolver : public KktLinearSolver {
 public:
  // Largest n accepted (H is dense n x n; see docs/modules/reduced_kkt_backend.md
  // for the measurements behind the limit).
  static constexpr int32_t kMaxN = 2048;

  // Build the index maps from the K pattern / slot maps of `kkt` (setup
  // time), allocate H, the weights, rhs / sol and the status. `kkt` must be
  // initialized and outlive this object; its values buffer is read in place.
  void init(KktAssembly& kkt, cudaStream_t s);
  // Same from the host maps and the device CSR column indices / values of K
  // (stable addresses, outliving this object).
  void init(const KktHostMaps& maps, const int32_t* d_indices, double* d_values, cudaStream_t s);

  double* rhs() override { return rhs_.data(); }
  double* sol() override { return sol_.data(); }
  void factor(cudaStream_t s) override;  // assemble() + cholesky()
  void solve(cudaStream_t s) override;
  const int32_t* factor_status() const override { return status_.data(); }

  // The two halves of factor(), public for tests.
  void assemble(cudaStream_t s);  // weights and the lower triangle of H
  void cholesky(cudaStream_t s);  // H -> L in place, dinv, status
  const double* H() const { return H_.data(); }  // (n, n) column-major lower
  int32_t n() const { return n_; }
  int32_t mm() const { return mm_; }
  int assembly_warps() const { return warps_; }
  int32_t assembly_tasks() const { return num_tasks_; }
  int32_t assembly_split_rows() const { return num_split_rows_; }

 private:
  int32_t n_ = 0, mm_ = 0;
  int warps_ = 0;          // warps per block (accumulators) of the assembly
  size_t assembly_smem_ = 0;
  const int32_t* Kidx_ = nullptr;
  double* K_ = nullptr;
  int32_t num_tasks_ = 0, num_split_rows_ = 0;
  CudaBuffer<int32_t> Pptr_, Pcol_, Pslot_, Dslot_, Kbase_, Mlen_, Cptr_, Crow_, Coff_, Cslot_;
  // Assembly tasks (row, CSC range, output) and the rows split over several.
  CudaBuffer<int32_t> task_row_, task_e0_, task_e1_, srow_, stask0_, stask1_;
  CudaBuffer<int64_t> task_out_;
  // Compacted nonzeros of the rows of M (indexed by K slot) and prefix counts.
  CudaBuffer<double> Cv_;
  CudaBuffer<int32_t> Cc_, pre_;
  CudaBuffer<double> w_, H_, dinv_, rhs_, sol_, ws_;
  CudaBuffer<int32_t> status_;
};

}  // namespace cupiqp
