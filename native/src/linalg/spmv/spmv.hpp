// M2 spmv: graph-safe sparse matrix-vector products (reference
// cupiqp/sparse/sparse_matvec.py and SparseKKTSolver.eval_*), with the
// module's own deterministic CSR kernels (spmv_kernels.cuh).
#pragma once

#include <cuda_runtime_api.h>

#include "common/buffer.hpp"
#include "common/views.hpp"

namespace cupiqp {

// y = alpha * op(M) x + beta * y for one CSR matrix (B = 1). op = N uses a
// fixed number of lanes per row of M; op = T is a gather over a CSC view of
// the pattern built in init (with a value permutation, so in-place updates of
// M.values are seen by the next call). Bitwise reproducible for every op and
// beta; agrees with the exact product to a few 1e-16 of |op(M)| |x|. A call
// allocates nothing, does not synchronize and is capturable. A matrix with
// zero rows, columns or stored entries computes y = beta * y (beta == 0 does
// not read y).
class SpMV {
 public:
  SpMV() = default;
  // For op = T, init copies the pattern to the host once (synchronizes).
  void init(const CsrView<double>& M, bool transpose);
  void operator()(cudaStream_t s, double alpha, const double* x, double beta, double* y);
  SpMV(const SpMV&) = delete;
  SpMV& operator=(const SpMV&) = delete;

  // Lengths of x and y for op(M) (0 before init).
  int32_t x_len() const { return transpose_ ? rows_ : cols_; }
  int32_t y_len() const { return transpose_ ? cols_ : rows_; }
  // True when calls run the product kernel (matrix not empty).
  bool computes_product() const { return lanes_ > 0; }
  // Threads per output entry of the product kernel (0 if unused).
  int lanes() const { return lanes_; }

 private:
  void init_pattern(const CsrView<double>& M);
  bool transpose_ = false;
  int32_t rows_ = 0, cols_ = 0, nnz_ = 0;
  // CSR view of op(M) (for op = T the CSC pattern of M with
  // perm_[k] = position of the entry in M.values).
  int lanes_ = 0;
  const int32_t* op_indptr_ = nullptr;
  const int32_t* op_indices_ = nullptr;
  const double* values_ = nullptr;
  CudaBuffer<int32_t> t_indptr_, t_indices_, t_perm_;
};

// The five products used by the KKT system and residuals (reference
// KKTSolverBase.eval_*): P x, A x, A^T y, G x, G^T z. Absent blocks (p == 0 or
// m == 0) make the corresponding call write zeros.
class SparseOperators {
 public:
  void init(const SparseDataView<double>& data);
  void P(cudaStream_t s, double alpha, const double* x, double* y);   // y = alpha P x
  void A(cudaStream_t s, double alpha, const double* x, double* y);   // y = alpha A x
  void AT(cudaStream_t s, double alpha, const double* x, double* y);  // y = alpha A^T x
  void G(cudaStream_t s, double alpha, const double* x, double* y);   // y = alpha G x
  void GT(cudaStream_t s, double alpha, const double* x, double* y);  // y = alpha G^T x

 private:
  Dims dims_;
  SpMV P_, A_, AT_, G_, GT_;
};

}  // namespace cupiqp
