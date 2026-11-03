// M1 data: problem storage of the sparse backend (reference cupiqp/data.py,
// cupiqp/data_kernels.py, cupiqp/sparse/sparse_data.py, batched_csr.py,
// csr_helpers.py and sparse_solver._check_csr).
//
// INTERFACE CONTRACT (set by the integration owner). Implementations may add
// private members and helpers; changing a public signature needs the owner.
#pragma once

#include <cuda_runtime_api.h>

#include <string>

#include "common/buffer.hpp"
#include "common/views.hpp"
#include "cupiqp/cupiqp.hpp"

namespace cupiqp {

// Throws Error(InvalidArgument) with the reference's messages when `M` is not a
// valid CSR pattern with `expected_cols` columns (expected_cols < 0: square).
void validate_csr(const CsrPattern& M, int32_t expected_cols, const std::string& name);

// Solver-owned copy of the problem (B = 1). All buffers are allocated in init()
// and keep their addresses. Values are stored unscaled after init()/set_*;
// the Ruiz module (M3) scales them in place afterwards.
class SparseData {
 public:
  // Validates patterns and block/vector consistency (P and c required; A <-> b;
  // G requires h_l or h_u; vector presence fixes num_hl/num_hu/num_xl/num_xu),
  // allocates every buffer, uploads patterns and COO row indices, copies the
  // initial device values (stream ordered) and computes the finite masks.
  void init(const CsrPattern& P, const CsrPattern* A, const CsrPattern* G, const ValuesView& v,
            cudaStream_t s);

  const Dims& dims() const { return dims_; }
  SparseDataView<double> view();

  // Host copies of the patterns (for the KKT build and tests).
  const CsrPattern& P_pattern() const { return P_host_; }
  const CsrPattern& A_pattern() const { return A_host_; }  // rows == 0 when absent
  const CsrPattern& G_pattern() const { return G_host_; }  // rows == 0 when absent

  // Copy new values (device pointers, stream ordered) into storage. Bound
  // setters require the side to have been given at setup (else InvalidArgument).
  // Applies every non-null entry of `v`; refreshes the finite masks when a bound
  // changed. Returns the mask of replaced blocks.
  UpdateMask set_values(const ValuesView& v, cudaStream_t s);
  // Same, from host memory (pinned or pageable).
  UpdateMask set_values_from_host(const ValuesView& host_v, cudaStream_t s);

  // Recompute finite_mask_all, active_G_row, active_x_bound, num_finite_bounds
  // (reference finite_bound_masks_kernel: a lower bound is finite iff
  // v > -PIQP_INF, an upper bound iff v < PIQP_INF).
  void update_finite_masks(cudaStream_t s);
  // Version 1 of the mask kernel (one thread per problem, the Warp kernel's
  // thread mapping). Same outputs as update_finite_masks; kept for parity
  // tests and timing.
  void update_finite_masks_reference(cudaStream_t s);

  // Bound sides given at init() (fixed for the lifetime of the object; true
  // even when the side has zero length because m == 0 or n == 0).
  bool has_h_l() const { return has_hl_; }
  bool has_h_u() const { return has_hu_; }
  bool has_x_l() const { return has_xl_; }
  bool has_x_u() const { return has_xu_; }
  bool initialized() const { return initialized_; }

 private:
  UpdateMask apply_values(const ValuesView& v, cudaMemcpyKind kind, cudaStream_t s);
  void check_update(const ValuesView& v) const;

  bool initialized_ = false;
  bool has_hl_ = false, has_hu_ = false, has_xl_ = false, has_xu_ = false;
  Dims dims_;
  CsrPattern P_host_, A_host_, G_host_;
  // Device storage (see SparseDataView for meaning and shapes).
  CudaBuffer<int32_t> P_indptr_, P_indices_, P_rows_, A_indptr_, A_indices_, A_rows_, G_indptr_,
      G_indices_, G_rows_;
  CudaBuffer<double> P_val_, A_val_, G_val_, c_, b_, h_l_, h_u_, x_l_, x_u_;
  CudaBuffer<double> finite_mask_all_, active_G_row_, active_x_bound_, num_finite_bounds_;
};

}  // namespace cupiqp
