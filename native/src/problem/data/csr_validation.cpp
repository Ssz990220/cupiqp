// CSR pattern validation (reference cupiqp/sparse/sparse_solver.py _check_csr).
#include <cstdint>
#include <string>

#include "common/check.hpp"
#include "problem/data/sparse_data.hpp"

namespace cupiqp {

void validate_csr(const CsrPattern& M, int32_t expected_cols, const std::string& name) {
  using detail::throw_invalid;
  // Shape. The reference derives rows from len(indptr) - 1.
  if (M.indptr.empty()) throw_invalid(name + " indptr must have at least one entry.");
  if (M.rows < 0 || static_cast<int64_t>(M.indptr.size()) != static_cast<int64_t>(M.rows) + 1) {
    throw_invalid(name + " indptr must have length rows + 1 = " + std::to_string(int64_t(M.rows) + 1) +
                  "; got " + std::to_string(M.indptr.size()) + ".");
  }
  const int32_t cols = expected_cols < 0 ? M.rows : expected_cols;
  if (M.cols != cols) {
    throw_invalid(name + " must have " + std::to_string(cols) + " columns; got " +
                  std::to_string(M.cols) + ".");
  }
  // Stored-entry count (int32 offsets; at most 2^31 - 1 entries).
  const int64_t nnz = static_cast<int64_t>(M.indices.size());
  if (nnz >= (int64_t(1) << 31)) {
    throw_invalid(name + " has " + std::to_string(nnz) +
                  " stored entries; at most 2**31 - 1 are supported.");
  }
  // indptr: starts at 0, never decreases, ends at nnz.
  bool ok = M.indptr.front() == 0 && static_cast<int64_t>(M.indptr.back()) == nnz;
  for (int32_t r = 0; ok && r < M.rows; ++r) ok = M.indptr[r + 1] >= M.indptr[r];
  if (!ok) {
    throw_invalid(name + " indptr must start at 0, never decrease, and end at nnz = " +
                  std::to_string(nnz) + ".");
  }
  // Column indices in range.
  for (int64_t k = 0; k < nnz; ++k) {
    if (M.indices[k] < 0 || M.indices[k] >= cols) {
      throw_invalid(name + " column indices must lie in [0, " + std::to_string(cols) + ").");
    }
  }
  // Strictly increasing within each row (sorted, no duplicates).
  for (int32_t r = 0; r < M.rows; ++r) {
    for (int32_t k = M.indptr[r] + 1; k < M.indptr[r + 1]; ++k) {
      if (M.indices[k] <= M.indices[k - 1]) {
        throw_invalid(name +
                      " column indices must be strictly increasing within each row "
                      "(sorted, no duplicates). Products and stacks of scipy sparse matrices "
                      "can leave them unsorted; call sort_indices() on the scipy matrix "
                      "before passing its pattern.");
      }
    }
  }
}

}  // namespace cupiqp
