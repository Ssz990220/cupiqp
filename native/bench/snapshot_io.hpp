// Reader of the snapshot bundles written by bench/tools/export_snapshots.py
// (group_<k>.bin: magic "CPQPSNP1", int32 n, p, m, nnz_P, nnz_A, nnz_C,
// num_cases, the three CSR patterns, then per case the packed float64 values
// P, A, C, q, b, l, u).
#pragma once

#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "cupiqp/cupiqp.hpp"

namespace bench {

struct Snapshots {
  int32_t n = 0, p = 0, m = 0, nnzP = 0, nnzA = 0, nnzC = 0, cases = 0;
  cupiqp::CsrPattern P, A, C;
  std::vector<double> values;  // cases x total
  size_t total = 0;
  size_t off_P() const { return 0; }
  size_t off_A() const { return size_t(nnzP); }
  size_t off_C() const { return size_t(nnzP) + nnzA; }
  size_t off_q() const { return size_t(nnzP) + nnzA + nnzC; }
  size_t off_b() const { return off_q() + n; }
  size_t off_l() const { return off_b() + p; }
  size_t off_u() const { return off_l() + m; }
  const double* case_values(int k) const { return values.data() + total * size_t(k); }
  // Device-pointer view of one packed case stored at `base`.
  cupiqp::ValuesView view(const double* base) const {
    cupiqp::ValuesView v;
    v.P = base + off_P(); v.A = base + off_A(); v.G = base + off_C();
    v.c = base + off_q(); v.b = base + off_b(); v.h_l = base + off_l(); v.h_u = base + off_u();
    return v;
  }
};

inline void read_pattern(std::ifstream& f, cupiqp::CsrPattern& M, int rows, int cols, int nnz) {
  M.rows = rows;
  M.cols = cols;
  M.indptr.resize(rows + 1);
  M.indices.resize(nnz);
  f.read(reinterpret_cast<char*>(M.indptr.data()), sizeof(int32_t) * (rows + 1));
  f.read(reinterpret_cast<char*>(M.indices.data()), sizeof(int32_t) * nnz);
}

inline Snapshots load_snapshots(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + path);
  char magic[8];
  f.read(magic, 8);
  if (std::memcmp(magic, "CPQPSNP1", 8) != 0) throw std::runtime_error("bad magic in " + path);
  Snapshots g;
  int32_t hdr[7];
  f.read(reinterpret_cast<char*>(hdr), sizeof hdr);
  g.n = hdr[0]; g.p = hdr[1]; g.m = hdr[2]; g.nnzP = hdr[3]; g.nnzA = hdr[4]; g.nnzC = hdr[5]; g.cases = hdr[6];
  read_pattern(f, g.P, g.n, g.n, g.nnzP);
  read_pattern(f, g.A, g.p, g.n, g.nnzA);
  read_pattern(f, g.C, g.m, g.n, g.nnzC);
  g.total = size_t(g.nnzP) + g.nnzA + g.nnzC + g.n + g.p + 2 * size_t(g.m);
  g.values.resize(g.total * g.cases);
  f.read(reinterpret_cast<char*>(g.values.data()), sizeof(double) * g.values.size());
  if (!f) throw std::runtime_error("short read in " + path);
  return g;
}

}  // namespace bench
