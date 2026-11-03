// M4 kkt_assembly: sparse KKT pattern, slot maps and value updates (reference
// cupiqp/sparse/sparse_kkt_solver.py SparseKKTSolver.__init__ / _initialize_kkt_csr /
// _scatter_P_A_G / update_data / update_and_factor_kkt diagonal part, and
// cupiqp/sparse/sparse_kkt_solver_kernels.py).
//
// KKT = bmat([[P + I_n, None, None], [A, I_p, None], [G, None, I_m]]) with
// sorted columns and duplicates summed: the full P block is stored, A and G in
// the lower-left, explicit identity diagonals; every stored entry of P, A, G
// (explicit zeros included) keeps a slot. cuDSS reads the lower triangle.
//
// INTERFACE CONTRACT (set by the integration owner).
#pragma once

#include <cuda_runtime_api.h>

#include "core/buffer.hpp"
#include "core/views.hpp"
#include "cupiqp/cupiqp.hpp"

namespace cupiqp {

class KktAssembly {
 public:
  // Build the pattern and slot maps on the host, upload them, allocate
  // kkt_values (B, kkt_nnz) initialized like the reference (ones on every
  // stored entry, i.e. the bmat template values), P_diag (B, n), then scatter
  // P, A, G from `data` (masked G with active_G_row) and refresh P_diag.
  void init(const CsrPattern& P, const CsrPattern& A, const CsrPattern& G,
            const SparseDataView<double>& data, cudaStream_t s);

  int32_t dim() const { return dims_.kkt_dim(); }
  int32_t nnz() const { return nnz_; }
  const int32_t* indptr() const { return indptr_.data(); }    // device, dim + 1
  const int32_t* indices() const { return indices_.data(); }  // device, nnz
  double* values() { return values_.data(); }                 // device (B, nnz), stable address
  const double* P_diag() const { return P_diag_.data(); }     // device (B, n)

  // Reference update_data: scatter the flagged blocks (G masked by
  // active_G_row) and refresh P_diag when P is flagged.
  void scatter(const SparseDataView<double>& data, bool P, bool A, bool G, cudaStream_t s);

  // Reference update_kkt_diag_kernel: x diag = P_diag + x_reg, y diag = -delta,
  // z diag = -z_reg (active) or -1 (z_reg == 0).
  void update_diagonal(const double* x_reg, const double* delta, const double* z_reg, cudaStream_t s);

  // Host copies for tests.
  const CsrPattern& host_pattern() const { return host_; }

 private:
  Dims dims_;
  int32_t nnz_ = 0;
  CsrPattern host_;
  CudaBuffer<int32_t> indptr_, indices_;
  CudaBuffer<int32_t> diag_x_, diag_y_, diag_z_;     // KKT slots of the diagonal blocks
  CudaBuffer<int32_t> P_slots_, A_slots_, G_slots_;  // KKT slot of every stored entry
  CudaBuffer<int32_t> P_diag_src_, P_diag_dst_;      // stored P diagonal positions -> variable
  CudaBuffer<double> values_, P_diag_;
};

}  // namespace cupiqp
