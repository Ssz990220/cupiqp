// Test and benchmark fixtures for M5 cudss: random quasi-definite KKT systems
// laid out like the reference sparse KKT (sparse_kkt_solver.py):
//   [ P + rho I              ]
//   [ A          -delta I    ]
//   [ G                -z I  ]
// stored as CSR with sorted columns, the FULL P block (upper entries too, as
// the reference does) and the A/G blocks in the lower-left; cuDSS reads the
// lower triangle only. Includes a dense CPU solve for verification.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <set>
#include <stdexcept>
#include <vector>

namespace cupiqp_test {

struct Kkt {
  int n = 0, p = 0, m = 0, dim = 0;
  std::vector<int32_t> indptr, indices;
  std::vector<double> values;
  std::vector<int32_t> diag_slot;      // slot of (r, r) for every row
  std::vector<int32_t> offdiag_lower;  // slots strictly below the diagonal
  int nnz() const { return static_cast<int>(indices.size()); }
};

// Random pattern with roughly the requested number of entries per block.
// P is symmetric (pattern and values) with a stored diagonal.
inline Kkt make_kkt(int n, int p, int m, long nnz_p, long nnz_a, long nnz_g, uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> uni(-1.0, 1.0);
  Kkt k;
  k.n = n, k.p = p, k.m = m, k.dim = n + p + m;
  std::vector<std::set<int>> rows(k.dim);
  // P: symmetric off-diagonal pairs.
  long pairs = std::max<long>(0, (nnz_p - n) / 2);
  pairs = std::min<long>(pairs, static_cast<long>(n) * (n - 1) / 2);
  for (int r = 0; r < n; ++r) rows[r].insert(r);
  std::uniform_int_distribution<int> col_n(0, std::max(0, n - 1));
  for (long added = 0; added < pairs;) {
    int i = col_n(rng), j = col_n(rng);
    if (i == j || rows[i].count(j)) continue;
    rows[i].insert(j), rows[j].insert(i), ++added;
  }
  auto fill_block = [&](int row0, int nrows, long target) {
    if (nrows == 0) return;
    target = std::min<long>(target, static_cast<long>(nrows) * n);
    for (int r = 0; r < nrows; ++r) rows[row0 + r].insert(col_n(rng));  // no empty constraint rows
    long have = nrows;
    std::uniform_int_distribution<int> row_d(0, nrows - 1);
    while (have < target) {
      int r = row_d(rng), c = col_n(rng);
      if (rows[row0 + r].insert(c).second) ++have;
    }
    for (int r = 0; r < nrows; ++r) rows[row0 + r].insert(row0 + r);  // -delta / -z diagonal
  };
  fill_block(n, p, nnz_a);
  fill_block(n + p, m, nnz_g);

  k.indptr.push_back(0);
  for (int r = 0; r < k.dim; ++r) {
    for (int c : rows[r]) {
      if (c == r) k.diag_slot.push_back(static_cast<int32_t>(k.indices.size()));
      else if (c < r) k.offdiag_lower.push_back(static_cast<int32_t>(k.indices.size()));
      k.indices.push_back(c);
    }
    k.indptr.push_back(static_cast<int32_t>(k.indices.size()));
  }
  k.values.assign(k.indices.size(), 0.0);
  return k;
}

// Find the slot of (r, c) or -1.
inline int slot_of(const Kkt& k, int r, int c) {
  auto b = k.indices.begin() + k.indptr[r], e = k.indices.begin() + k.indptr[r + 1];
  auto it = std::lower_bound(b, e, c);
  return (it != e && *it == c) ? static_cast<int>(it - k.indices.begin()) : -1;
}

// Fill quasi-definite values: random off-diagonals, P diagonal made dominant
// (+ rho), -delta on the A block diagonal, -z on the G block diagonal. P stays
// symmetric (upper entries mirror the lower ones).
inline void fill_values(Kkt& k, uint64_t seed, double rho = 1e-6, double delta = 1e-4) {
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> uni(-1.0, 1.0), pos(0.5, 2.0);
  std::vector<double> rowabs(k.n, 0.0);
  for (int r = 0; r < k.dim; ++r)
    for (int q = k.indptr[r]; q < k.indptr[r + 1]; ++q) {
      int c = k.indices[q];
      if (c >= r) continue;
      double v = uni(rng);
      k.values[q] = v;
      if (r < k.n) {  // mirror into the upper part of P
        k.values[slot_of(k, c, r)] = v;
        rowabs[r] += std::fabs(v), rowabs[c] += std::fabs(v);
      }
    }
  for (int r = 0; r < k.dim; ++r) {
    double& d = k.values[k.diag_slot[r]];
    if (r < k.n) d = rowabs[r] + pos(rng) + rho;
    else if (r < k.n + k.p) d = -delta;
    else d = -pos(rng);
  }
}

// Dense symmetric matrix from the LOWER triangle of the CSR (upper ignored).
inline std::vector<double> dense_lower_sym(const Kkt& k) {
  std::vector<double> M(static_cast<size_t>(k.dim) * k.dim, 0.0);
  for (int r = 0; r < k.dim; ++r)
    for (int q = k.indptr[r]; q < k.indptr[r + 1]; ++q) {
      int c = k.indices[q];
      if (c > r) continue;
      M[static_cast<size_t>(r) * k.dim + c] = k.values[q];
      M[static_cast<size_t>(c) * k.dim + r] = k.values[q];
    }
  return M;
}

// Dense LU with partial pivoting (CPU reference solve).
inline std::vector<double> dense_solve(std::vector<double> M, std::vector<double> b) {
  const size_t n = b.size();
  for (size_t col = 0; col < n; ++col) {
    size_t piv = col;
    for (size_t r = col + 1; r < n; ++r)
      if (std::fabs(M[r * n + col]) > std::fabs(M[piv * n + col])) piv = r;
    if (M[piv * n + col] == 0.0) throw std::runtime_error("dense_solve: singular");
    if (piv != col) {
      for (size_t c = 0; c < n; ++c) std::swap(M[col * n + c], M[piv * n + c]);
      std::swap(b[col], b[piv]);
    }
    for (size_t r = col + 1; r < n; ++r) {
      double f = M[r * n + col] / M[col * n + col];
      if (f == 0.0) continue;
      for (size_t c = col; c < n; ++c) M[r * n + c] -= f * M[col * n + c];
      b[r] -= f * b[col];
    }
  }
  std::vector<double> x(n);
  for (size_t i = n; i-- > 0;) {
    double s = b[i];
    for (size_t c = i + 1; c < n; ++c) s -= M[i * n + c] * x[c];
    x[i] = s / M[i * n + i];
  }
  return x;
}

// ||K x - b||_inf with K the symmetric matrix of the lower triangle.
inline double residual_inf(const Kkt& k, const std::vector<double>& x, const std::vector<double>& b) {
  std::vector<double> r(b.size());
  for (size_t i = 0; i < b.size(); ++i) r[i] = -b[i];
  for (int row = 0; row < k.dim; ++row)
    for (int q = k.indptr[row]; q < k.indptr[row + 1]; ++q) {
      int c = k.indices[q];
      if (c > row) continue;
      r[row] += k.values[q] * x[c];
      if (c != row) r[c] += k.values[q] * x[row];
    }
  double m = 0;
  for (double v : r) m = std::max(m, std::fabs(v));
  return m;
}

}  // namespace cupiqp_test
