// M2 spmv: graph-safe cuSPARSE SpMV operators (reference
// cupiqp/sparse/sparse_matvec.py, CUSPARSE_SPMV_ALG_DEFAULT, 32-bit indices,
// host pointer mode for alpha/beta, preallocated external buffer).
//
// INTERFACE CONTRACT (set by the integration owner).
#pragma once

#include <cusparse.h>

#include "common/buffer.hpp"
#include "common/views.hpp"

namespace cupiqp {

// RAII cuSPARSE handle.
class CusparseHandle {
 public:
  CusparseHandle();
  ~CusparseHandle();
  CusparseHandle(const CusparseHandle&) = delete;
  CusparseHandle& operator=(const CusparseHandle&) = delete;
  cusparseHandle_t get() const { return h_; }

 private:
  cusparseHandle_t h_ = nullptr;
};

// Which implementation an SpMV uses. Cusparse (default) mirrors the reference
// call exactly. Custom uses the module's own CSR kernels (spmv_kernels.cuh):
// op = N with a fixed number of lanes per row, op = T as a gather over a CSC
// view of the pattern built in init (value permutation, so in-place value
// updates are still seen). Custom is bitwise reproducible for every op and
// beta (cuSPARSE ALG_DEFAULT is not for op = T or beta != 0), agrees with the
// exact product to a few 1e-16 of |op(M)| |x|, and on the MPC shapes is 2-7x
// faster (docs/modules/spmv.md). It is opt-in.
enum class SpmvBackend { Cusparse, Custom };

// y = alpha * op(M) x + beta * y for one CSR matrix (B = 1). Descriptors and
// the workspace are created once; a call binds the handle to `s`, rebinds the
// x / y pointers and calls cusparseSpMV: no allocation, no sync, capturable.
// In-place updates of M.values are seen by the next call. A matrix with zero
// rows or columns is handled without calling cuSPARSE (y = beta * y or no-op,
// matching the reference results).
class SpMV {
 public:
  SpMV() = default;
  void init(cusparseHandle_t h, const CsrView<double>& M, bool transpose);
  // Same with an explicit backend (h may be null for SpmvBackend::Custom).
  // Custom init copies the pattern to the host once (synchronizes).
  void init(cusparseHandle_t h, const CsrView<double>& M, bool transpose, SpmvBackend backend);
  void operator()(cudaStream_t s, double alpha, const double* x, double beta, double* y);
  ~SpMV();
  SpMV(const SpMV&) = delete;
  SpMV& operator=(const SpMV&) = delete;

  // Lengths of x and y for op(M) (0 before init).
  int32_t x_len() const { return transpose_ ? rows_ : cols_; }
  int32_t y_len() const { return transpose_ ? cols_ : rows_; }
  // True when calls go to cuSPARSE (Cusparse backend, rows, cols and nnz all nonzero).
  bool uses_cusparse() const { return mat_ != nullptr; }
  // True when calls go to the custom kernels (Custom backend, matrix not empty).
  bool uses_custom() const { return custom_; }
  // Threads per output entry of the custom kernel (0 if unused).
  int lanes() const { return lanes_; }

 private:
  void release() noexcept;
  void init_custom(const CsrView<double>& M);
  cusparseHandle_t h_ = nullptr;
  cusparseSpMatDescr_t mat_ = nullptr;
  cusparseDnVecDescr_t x_ = nullptr, y_ = nullptr;
  CudaBuffer<unsigned char> buffer_;
  bool transpose_ = false;
  int32_t rows_ = 0, cols_ = 0, nnz_ = 0;
  // Custom backend: CSR view of op(M) (for op = T the CSC pattern of M with
  // perm_[k] = position of the entry in M.values).
  bool custom_ = false;
  int lanes_ = 0;
  const int32_t* op_indptr_ = nullptr;
  const int32_t* op_indices_ = nullptr;
  const double* values_ = nullptr;
  CudaBuffer<int32_t> t_indptr_, t_indices_, t_perm_;
};

// The five products used by the KKT system and residuals (reference
// KKTSolverBase.eval_*): P x, A x, A^T y, G x, G^T z. Absent blocks (p == 0 or
// m == 0) make the corresponding call write zeros (when beta == 0).
class SparseOperators {
 public:
  void init(cusparseHandle_t h, const SparseDataView<double>& data);
  void init(cusparseHandle_t h, const SparseDataView<double>& data, SpmvBackend backend);
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
