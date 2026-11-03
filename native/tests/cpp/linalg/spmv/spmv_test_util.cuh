// Shared helpers for the spmv tests and benchmark: random host CSR matrices,
// a CPU reference product and device copies.
#pragma once

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include "common/buffer.hpp"
#include "common/views.hpp"

namespace cupiqp::spmv_test {

struct HostCsr {
  int32_t rows = 0, cols = 0;
  std::vector<int32_t> indptr, indices;
  std::vector<double> values;
  int32_t nnz() const { return static_cast<int32_t>(indices.size()); }
};

// Random CSR with sorted columns. `density` is the stored fraction, of which
// `zero_frac` are explicit zeros; `empty_row_frac` of the rows store nothing.
inline HostCsr random_csr(int32_t rows, int32_t cols, double density, uint64_t seed, double zero_frac = 0.2,
                          double empty_row_frac = 0.1) {
  std::mt19937_64 g(seed);
  std::uniform_real_distribution<double> u(0.0, 1.0);
  std::normal_distribution<double> nrm(0.0, 1.0);
  HostCsr M;
  M.rows = rows;
  M.cols = cols;
  M.indptr.assign(rows + 1, 0);
  for (int32_t r = 0; r < rows; ++r) {
    const bool empty = u(g) < empty_row_frac;
    for (int32_t c = 0; c < cols && !empty; ++c) {
      if (u(g) < density) {
        M.indices.push_back(c);
        M.values.push_back(u(g) < zero_frac ? 0.0 : nrm(g));
      }
    }
    M.indptr[r + 1] = M.nnz();
  }
  return M;
}

inline std::vector<double> random_vec(int32_t k, uint64_t seed) {
  std::mt19937_64 g(seed);
  std::normal_distribution<double> nrm(0.0, 1.0);
  std::vector<double> v(k);
  for (auto& x : v) x = nrm(g);
  return v;
}

// y = alpha op(M) x + beta y in long double, plus the magnitude sum
// |alpha| |op(M)| |x| + |beta| |y| per entry for relative error bounds.
inline void cpu_spmv(const HostCsr& M, bool transpose, double alpha, const std::vector<double>& x, double beta,
                     std::vector<double>& y, std::vector<double>* mag = nullptr) {
  const int32_t ny = transpose ? M.cols : M.rows;
  std::vector<long double> acc(ny, 0.0L), am(ny, 0.0L);
  for (int32_t r = 0; r < M.rows; ++r)
    for (int32_t k = M.indptr[r]; k < M.indptr[r + 1]; ++k) {
      const int32_t c = M.indices[k];
      const int32_t out = transpose ? c : r, in = transpose ? r : c;
      acc[out] += static_cast<long double>(M.values[k]) * x[in];
      am[out] += std::fabs(static_cast<long double>(M.values[k]) * x[in]);
    }
  if (mag) mag->assign(ny, 0.0);
  for (int32_t i = 0; i < ny; ++i) {
    const long double yb = beta == 0.0 ? 0.0L : static_cast<long double>(beta) * y[i];
    if (mag) (*mag)[i] = static_cast<double>(std::fabs(alpha) * am[i] + std::fabs(yb));
    y[i] = static_cast<double>(alpha * acc[i] + yb);
  }
}

// Device copy of a HostCsr (owns the buffers).
struct DeviceCsr {
  CudaBuffer<int32_t> indptr, indices;
  CudaBuffer<double> values;
  int32_t rows = 0, cols = 0, nnz = 0;
  explicit DeviceCsr(const HostCsr& M) : indptr(M.indptr.size()), indices(M.indices.size()), values(M.values.size()) {
    rows = M.rows;
    cols = M.cols;
    nnz = M.nnz();
    indptr.copy_from_host(M.indptr.data(), M.indptr.size(), nullptr);
    indices.copy_from_host(M.indices.data(), M.indices.size(), nullptr);
    values.copy_from_host(M.values.data(), M.values.size(), nullptr);
    CUPIQP_CUDA_CHECK(cudaDeviceSynchronize());
  }
  CsrView<double> view() {
    CsrView<double> v;
    v.rows = rows;
    v.cols = cols;
    v.nnz = nnz;
    v.indptr = indptr.data();
    v.indices = indices.data();
    v.values = values.data();
    return v;
  }
};

inline CudaBuffer<double> to_device(const std::vector<double>& h) {
  CudaBuffer<double> d(h.size());
  d.copy_from_host(h.data(), h.size(), nullptr);
  CUPIQP_CUDA_CHECK(cudaDeviceSynchronize());
  return d;
}

inline std::vector<double> to_host(const CudaBuffer<double>& d, cudaStream_t s) {
  std::vector<double> h(d.size());
  if (!h.empty())
    CUPIQP_CUDA_CHECK(cudaMemcpyAsync(h.data(), d.data(), h.size() * sizeof(double), cudaMemcpyDeviceToHost, s));
  CUPIQP_CUDA_CHECK(cudaStreamSynchronize(s));
  return h;
}

}  // namespace cupiqp::spmv_test
