// M2 spmv: graph-safe cuSPARSE SpMV (reference cupiqp/sparse/sparse_matvec.py).
//
// Mirrors SparseMatVecProduct for B = 1: one CSR descriptor over the caller's
// indptr / indices / values (32-bit indices, base 0, compute type = value
// type), two dense-vector descriptors whose pointers are rebound on every
// call, CUSPARSE_SPMV_ALG_DEFAULT, host-pointer alpha / beta, and a workspace
// sized once by cusparseSpMV_bufferSize. The reference does not call
// cusparseSpMV_preprocess, so neither do we.
#include "linalg/spmv/spmv.hpp"

#include <algorithm>
#include <type_traits>
#include <vector>

#include "common/check.hpp"
#include "linalg/spmv/spmv_kernels.cuh"

namespace cupiqp {

namespace {
constexpr cudaDataType kValueType = CUDA_R_64F;
constexpr cusparseSpMVAlg_t kAlg = CUSPARSE_SPMV_ALG_DEFAULT;
constexpr int kCustomBlock = 128;

// Lanes per output row of the custom kernel: a power of two, at most 32, at
// most the next power of two of the mean row length, and small enough that
// rows * lanes stays within kTargetThreads (tuned on the MPC shapes on an
// RTX 4070 SUPER: G x -> 8 lanes, P x / A x / G^T z -> 32).
int choose_lanes(int64_t out_rows, int64_t nnz) {
  constexpr int64_t kTargetThreads = 32768;
  const int64_t mean = (nnz + out_rows - 1) / std::max<int64_t>(out_rows, 1);
  int lanes = 1;
  while (lanes < 32 && lanes < mean && out_rows * lanes * 2 <= kTargetThreads) lanes *= 2;
  return lanes;
}
}  // namespace

// ---------------------------------------------------------------- handle
CusparseHandle::CusparseHandle() { CUPIQP_CUSPARSE_CHECK(cusparseCreate(&h_)); }
CusparseHandle::~CusparseHandle() {
  if (h_) cusparseDestroy(h_);
}

// ---------------------------------------------------------------- SpMV
void SpMV::release() noexcept {
  if (x_) cusparseDestroyDnVec(x_);
  if (y_) cusparseDestroyDnVec(y_);
  if (mat_) cusparseDestroySpMat(mat_);
  x_ = y_ = nullptr;
  mat_ = nullptr;
  buffer_ = CudaBuffer<unsigned char>();
  custom_ = false;
  lanes_ = 0;
  op_indptr_ = op_indices_ = nullptr;
  values_ = nullptr;
  t_indptr_ = CudaBuffer<int32_t>();
  t_indices_ = CudaBuffer<int32_t>();
  t_perm_ = CudaBuffer<int32_t>();
}

SpMV::~SpMV() { release(); }

void SpMV::init(cusparseHandle_t h, const CsrView<double>& M, bool transpose) {
  init(h, M, transpose, SpmvBackend::Cusparse);
}

void SpMV::init(cusparseHandle_t h, const CsrView<double>& M, bool transpose, SpmvBackend backend) {
  CUPIQP_REQUIRE(M.rows >= 0 && M.cols >= 0 && M.nnz >= 0, "SpMV::init: negative CSR dimensions");
  release();
  h_ = h;
  transpose_ = transpose;
  rows_ = M.rows;
  cols_ = M.cols;
  nnz_ = M.nnz;
  // Without stored entries op(M) x is zero: handled without cuSPARSE.
  if (rows_ == 0 || cols_ == 0 || nnz_ == 0) return;
  CUPIQP_REQUIRE(M.indptr && M.indices && M.values, "SpMV::init: null CSR array");
  if (backend == SpmvBackend::Custom) {
    init_custom(M);
    return;
  }
  CUPIQP_REQUIRE(h != nullptr, "SpMV::init: null cuSPARSE handle");

  CUPIQP_CUSPARSE_CHECK(cusparseCreateCsr(&mat_, rows_, cols_, nnz_, const_cast<int32_t*>(M.indptr),
                                          const_cast<int32_t*>(M.indices), M.values, CUSPARSE_INDEX_32I,
                                          CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, kValueType));
  // Vector pointers are bound per call; the descriptors start unbound.
  CUPIQP_CUSPARSE_CHECK(cusparseCreateDnVec(&x_, x_len(), nullptr, kValueType));
  CUPIQP_CUSPARSE_CHECK(cusparseCreateDnVec(&y_, y_len(), nullptr, kValueType));

  const double alpha = 1.0, beta = 0.0;  // the workspace size does not depend on the scalars
  size_t bytes = 0;
  const cusparseOperation_t op = transpose_ ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE;
  CUPIQP_CUSPARSE_CHECK(
      cusparseSpMV_bufferSize(h_, op, &alpha, mat_, x_, &beta, y_, kValueType, kAlg, &bytes));
  buffer_ = CudaBuffer<unsigned char>(bytes > 0 ? bytes : 1);
}

// Custom backend setup: op = N uses M's pattern directly; op = T builds the
// CSC pattern of M on the host (counting sort, stable in the row order, so
// every column lists its rows ascending) with the value permutation.
void SpMV::init_custom(const CsrView<double>& M) {
  custom_ = true;
  values_ = M.values;
  const int64_t out_rows = transpose_ ? cols_ : rows_;
  lanes_ = choose_lanes(out_rows, nnz_);
  if (!transpose_) {
    op_indptr_ = M.indptr;
    op_indices_ = M.indices;
    return;
  }
  std::vector<int32_t> indptr(rows_ + 1), indices(nnz_);
  CUPIQP_CUDA_CHECK(cudaMemcpy(indptr.data(), M.indptr, indptr.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));
  CUPIQP_CUDA_CHECK(cudaMemcpy(indices.data(), M.indices, indices.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));
  CUPIQP_REQUIRE(indptr[0] == 0 && indptr[rows_] == nnz_, "SpMV::init: inconsistent CSR indptr");
  std::vector<int32_t> t_indptr(cols_ + 1, 0), t_indices(nnz_), t_perm(nnz_);
  for (int32_t c : indices) {
    CUPIQP_REQUIRE(c >= 0 && c < cols_, "SpMV::init: CSR column index out of range");
    ++t_indptr[c + 1];
  }
  for (int32_t c = 0; c < cols_; ++c) t_indptr[c + 1] += t_indptr[c];
  std::vector<int32_t> next(t_indptr.begin(), t_indptr.end() - 1);
  for (int32_t r = 0; r < rows_; ++r) {
    for (int32_t k = indptr[r]; k < indptr[r + 1]; ++k) {
      const int32_t pos = next[indices[k]]++;
      t_indices[pos] = r;
      t_perm[pos] = k;
    }
  }
  t_indptr_ = CudaBuffer<int32_t>(t_indptr.size());
  t_indices_ = CudaBuffer<int32_t>(t_indices.size());
  t_perm_ = CudaBuffer<int32_t>(t_perm.size());
  CUPIQP_CUDA_CHECK(cudaMemcpy(t_indptr_.data(), t_indptr.data(), t_indptr.size() * sizeof(int32_t),
                               cudaMemcpyHostToDevice));
  CUPIQP_CUDA_CHECK(cudaMemcpy(t_indices_.data(), t_indices.data(), t_indices.size() * sizeof(int32_t),
                               cudaMemcpyHostToDevice));
  CUPIQP_CUDA_CHECK(cudaMemcpy(t_perm_.data(), t_perm.data(), t_perm.size() * sizeof(int32_t),
                               cudaMemcpyHostToDevice));
  // A pageable host-to-device cudaMemcpy may return before its DMA lands;
  // calls run on caller (possibly non-blocking) streams, so wait here.
  CUPIQP_CUDA_CHECK(cudaDeviceSynchronize());
  op_indptr_ = t_indptr_.data();
  op_indices_ = t_indices_.data();
}

void SpMV::operator()(cudaStream_t s, double alpha, const double* x, double beta, double* y) {
  const int32_t ny = y_len();
  if (ny == 0) return;
  if (custom_) {
    const int32_t* perm = transpose_ ? t_perm_.data() : nullptr;
    const int64_t threads = static_cast<int64_t>(ny) * lanes_;
    const unsigned grid = static_cast<unsigned>((threads + kCustomBlock - 1) / kCustomBlock);
    auto launch = [&](auto L) {
      spmv_detail::csr_lanes_kernel<decltype(L)::value, double><<<grid, kCustomBlock, 0, s>>>(
          ny, op_indptr_, op_indices_, values_, perm, x, alpha, beta, y);
    };
    switch (lanes_) {
      case 1: launch(std::integral_constant<int, 1>{}); break;
      case 2: launch(std::integral_constant<int, 2>{}); break;
      case 4: launch(std::integral_constant<int, 4>{}); break;
      case 8: launch(std::integral_constant<int, 8>{}); break;
      case 16: launch(std::integral_constant<int, 16>{}); break;
      default: launch(std::integral_constant<int, 32>{}); break;
    }
    CUPIQP_CUDA_CHECK_LAUNCH();
    return;
  }
  if (mat_ == nullptr) {
    // op(M) has no entries: y = beta * y (cuSPARSE semantics; beta == 0 does not read y).
    if (beta == 0.0) {
      CUPIQP_CUDA_CHECK(cudaMemsetAsync(y, 0, sizeof(double) * static_cast<size_t>(ny), s));
    } else if (beta != 1.0) {
      constexpr int kBlock = 256;
      spmv_detail::scale_kernel<double><<<(ny + kBlock - 1) / kBlock, kBlock, 0, s>>>(ny, beta, y);
      CUPIQP_CUDA_CHECK_LAUNCH();
    }
    return;
  }
  CUPIQP_CUSPARSE_CHECK(cusparseSetStream(h_, s));
  CUPIQP_CUSPARSE_CHECK(cusparseDnVecSetValues(x_, const_cast<double*>(x)));
  CUPIQP_CUSPARSE_CHECK(cusparseDnVecSetValues(y_, y));
  const cusparseOperation_t op = transpose_ ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE;
  CUPIQP_CUSPARSE_CHECK(cusparseSpMV(h_, op, &alpha, mat_, x_, &beta, y_, kValueType, kAlg, buffer_.data()));
}

// ---------------------------------------------------------------- operators
void SparseOperators::init(cusparseHandle_t h, const SparseDataView<double>& data) {
  init(h, data, SpmvBackend::Cusparse);
}

void SparseOperators::init(cusparseHandle_t h, const SparseDataView<double>& data, SpmvBackend backend) {
  dims_ = data.dims;
  const int32_t n = dims_.n, p = dims_.p, m = dims_.m;
  CUPIQP_REQUIRE(dims_.B == 1, "SparseOperators: batch size must be 1");
  CUPIQP_REQUIRE(data.P.rows == n && data.P.cols == n, "SparseOperators: P must be n x n");
  // Absent blocks (p == 0 / m == 0) become empty operators: A x / G x write
  // nothing and A^T y / G^T z write n zeros, as KKTSystem does (it skips the
  // product and its work buffers stay zero).
  CsrView<double> A = data.A, G = data.G;
  if (p == 0) A = CsrView<double>{0, n, 0, nullptr, nullptr, nullptr, nullptr};
  if (m == 0) G = CsrView<double>{0, n, 0, nullptr, nullptr, nullptr, nullptr};
  CUPIQP_REQUIRE(A.rows == p && A.cols == n, "SparseOperators: A must be p x n");
  CUPIQP_REQUIRE(G.rows == m && G.cols == n, "SparseOperators: G must be m x n");
  P_.init(h, data.P, false, backend);
  A_.init(h, A, false, backend);
  AT_.init(h, A, true, backend);
  G_.init(h, G, false, backend);
  GT_.init(h, G, true, backend);
}

void SparseOperators::P(cudaStream_t s, double alpha, const double* x, double* y) { P_(s, alpha, x, 0.0, y); }
void SparseOperators::A(cudaStream_t s, double alpha, const double* x, double* y) { A_(s, alpha, x, 0.0, y); }
void SparseOperators::AT(cudaStream_t s, double alpha, const double* x, double* y) { AT_(s, alpha, x, 0.0, y); }
void SparseOperators::G(cudaStream_t s, double alpha, const double* x, double* y) { G_(s, alpha, x, 0.0, y); }
void SparseOperators::GT(cudaStream_t s, double alpha, const double* x, double* y) { GT_(s, alpha, x, 0.0, y); }

}  // namespace cupiqp
