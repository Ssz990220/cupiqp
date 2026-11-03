// Host KKT pattern build. See kkt_pattern.hpp.
#include "linalg/kkt_assembly/kkt_pattern.hpp"

#include <limits>
#include <string>

#include "common/check.hpp"

namespace cupiqp {

namespace {

int32_t pattern_rows(const CsrPattern& M) { return M.indptr.empty() ? 0 : M.rows; }

void check_block(const CsrPattern& M, int32_t n, const char* name) {
  if (pattern_rows(M) == 0) return;
  CUPIQP_REQUIRE(M.cols == n, std::string("kkt_assembly: ") + name + " must have n columns");
  CUPIQP_REQUIRE(static_cast<int64_t>(M.indptr.size()) == static_cast<int64_t>(M.rows) + 1,
                 std::string("kkt_assembly: ") + name + " indptr must have rows + 1 entries");
  CUPIQP_REQUIRE(static_cast<int64_t>(M.indices.size()) == M.nnz(),
                 std::string("kkt_assembly: ") + name + " indices must have nnz entries");
}

}  // namespace

KktHostMaps build_kkt_host_maps(const CsrPattern& P, const CsrPattern& A, const CsrPattern& G) {
  KktHostMaps h;
  const int32_t n = pattern_rows(P);
  const int32_t p = pattern_rows(A);
  const int32_t m = pattern_rows(G);
  CUPIQP_REQUIRE(n > 0, "kkt_assembly: P must have at least one row");
  check_block(P, n, "P");
  check_block(A, n, "A");
  check_block(G, n, "G");
  h.n = n;
  h.p = p;
  h.m = m;

  const int64_t dim = static_cast<int64_t>(n) + p + m;
  const int64_t nnz_P = P.nnz();
  const int64_t nnz_A = p ? A.nnz() : 0;
  const int64_t nnz_G = m ? G.nnz() : 0;
  // Upper bound: every block entry plus one diagonal per row.
  const int64_t nnz_max = nnz_P + nnz_A + nnz_G + dim;
  CUPIQP_REQUIRE(dim < std::numeric_limits<int32_t>::max(), "kkt_assembly: KKT dimension exceeds int32");
  CUPIQP_REQUIRE(nnz_max < std::numeric_limits<int32_t>::max(), "kkt_assembly: KKT nnz exceeds int32");

  CsrPattern& K = h.kkt;
  K.rows = static_cast<int32_t>(dim);
  K.cols = static_cast<int32_t>(dim);
  K.indptr.assign(static_cast<size_t>(dim) + 1, 0);
  K.indices.reserve(static_cast<size_t>(nnz_max));
  h.template_values.reserve(static_cast<size_t>(nnz_max));
  h.P_slots.resize(static_cast<size_t>(nnz_P));
  h.A_slots.resize(static_cast<size_t>(nnz_A));
  h.G_slots.resize(static_cast<size_t>(nnz_G));
  h.diag_x.resize(n);
  h.diag_y.resize(p);
  h.diag_z.resize(m);
  h.G_rows.resize(static_cast<size_t>(nnz_G));

  auto push = [&](int32_t col, double v) {
    const int32_t pos = static_cast<int32_t>(K.indices.size());
    K.indices.push_back(col);
    h.template_values.push_back(v);
    return pos;
  };

  // Rows 0..n-1: P + I_n (the diagonal is inserted in column order; a stored
  // P diagonal sums to 1 + 1 = 2).
  for (int32_t i = 0; i < n; ++i) {
    bool diag_done = false;
    for (int32_t k = P.indptr[i]; k < P.indptr[i + 1]; ++k) {
      const int32_t j = P.indices[k];
      if (!diag_done && j > i) {
        h.diag_x[i] = push(i, 1.0);
        diag_done = true;
      }
      if (j == i) {
        const int32_t pos = push(j, 1.0 + 1.0);
        h.P_slots[k] = pos;
        h.diag_x[i] = pos;
        h.P_diag_src.push_back(k);
        h.P_diag_dst.push_back(j);
        diag_done = true;
      } else {
        h.P_slots[k] = push(j, 1.0);
      }
    }
    if (!diag_done) h.diag_x[i] = push(i, 1.0);
    K.indptr[i + 1] = static_cast<int32_t>(K.indices.size());
  }
  // Rows n..n+p-1: A (columns < n) then the I_p diagonal.
  for (int32_t i = 0; i < p; ++i) {
    for (int32_t k = A.indptr[i]; k < A.indptr[i + 1]; ++k) h.A_slots[k] = push(A.indices[k], 1.0);
    h.diag_y[i] = push(n + i, 1.0);
    K.indptr[n + i + 1] = static_cast<int32_t>(K.indices.size());
  }
  // Rows n+p..n+p+m-1: G (columns < n) then the I_m diagonal.
  for (int32_t i = 0; i < m; ++i) {
    for (int32_t k = G.indptr[i]; k < G.indptr[i + 1]; ++k) {
      h.G_slots[k] = push(G.indices[k], 1.0);
      h.G_rows[k] = i;
    }
    h.diag_z[i] = push(n + p + i, 1.0);
    K.indptr[n + p + i + 1] = static_cast<int32_t>(K.indices.size());
  }
  return h;
}

}  // namespace cupiqp
