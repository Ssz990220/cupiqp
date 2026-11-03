// Host construction of the sparse KKT pattern and its index maps (reference
// SparseKKTSolver._initialize_kkt_csr, csr_diag_indices, csr_subblock_indices
// and the P-diagonal maps of SparseKKTSolver.__init__), without SciPy.
#pragma once

#include <cstdint>
#include <vector>

#include "cupiqp/cupiqp.hpp"

namespace cupiqp {

// Everything the KKT assembly needs from the setup patterns, on the host.
struct KktHostMaps {
  int32_t n = 0, p = 0, m = 0;
  // Pattern of bmat([[P + I_n, None, None], [A, I_p, None], [G, None, I_m]])
  // after sum_duplicates(): rows n + p + m, sorted columns, union pattern.
  CsrPattern kkt;
  // bmat template values when P, A, G are all-ones patterns: 2 on a P
  // diagonal stored by P (1 + 1), 1 everywhere else.
  std::vector<double> template_values;
  // KKT slot of every stored entry of P / A / G, in stored-entry order.
  std::vector<int32_t> P_slots, A_slots, G_slots;
  // KKT slot of the diagonal entry of each row of the x / y / z blocks.
  std::vector<int32_t> diag_x, diag_y, diag_z;
  // Positions in P's stored entries that lie on the diagonal (ascending) and
  // the variable (column) each one belongs to.
  std::vector<int32_t> P_diag_src, P_diag_dst;
  // Row of every stored entry of G (COO rows).
  std::vector<int32_t> G_rows;
};

// Build the KKT pattern and maps. P is n x n (full symmetric storage), A is
// p x n, G is m x n; an absent block has rows == 0 (indptr may be empty).
// Patterns must be valid CSR (sorted, no duplicates); throws InvalidArgument
// on inconsistent shapes or when the KKT nnz does not fit in int32.
KktHostMaps build_kkt_host_maps(const CsrPattern& P, const CsrPattern& A, const CsrPattern& G);

}  // namespace cupiqp
