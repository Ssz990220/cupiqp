// M2 spmv: graph-safe cuSPARSE SpMV operators (reference
// cupiqp/sparse/sparse_matvec.py, CUSPARSE_SPMV_ALG_DEFAULT, 32-bit indices,
// host pointer mode for alpha/beta, preallocated external buffer).
//
// INTERFACE CONTRACT (set by the integration owner).
#pragma once

#include <cusparse.h>

#include "core/buffer.hpp"
#include "core/views.hpp"

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
  void operator()(cudaStream_t s, double alpha, const double* x, double beta, double* y);
  ~SpMV();
  SpMV(const SpMV&) = delete;
  SpMV& operator=(const SpMV&) = delete;

 private:
  cusparseHandle_t h_ = nullptr;
  cusparseSpMatDescr_t mat_ = nullptr;
  cusparseDnVecDescr_t x_ = nullptr, y_ = nullptr;
  CudaBuffer<unsigned char> buffer_;
  bool transpose_ = false;
  int32_t rows_ = 0, cols_ = 0, nnz_ = 0;
};

// The five products used by the KKT system and residuals (reference
// KKTSolverBase.eval_*): P x, A x, A^T y, G x, G^T z. Absent blocks (p == 0 or
// m == 0) make the corresponding call write zeros (when beta == 0).
class SparseOperators {
 public:
  void init(cusparseHandle_t h, const SparseDataView<double>& data);
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
