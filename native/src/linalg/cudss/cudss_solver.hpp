// M5 cudss: cuDSS sparse LDL^T wrapper (reference
// cupiqp/sparse/sparse_direct_solver.py, CudssStackedSolver with B = 1).
// Configuration identical to the reference: SYMMETRIC, LOWER view, base 0,
// int32 offsets/indices, FP64 values; HYBRID_MEMORY_MODE 0,
// USE_CUDA_REGISTER_MEMORY 1, REORDERING_ALG DEFAULT, USE_SUPERPANELS 0,
// IR_N_STEPS 0, DETERMINISTIC_MODE when requested; threading layer
// libcudss_mtlayer_gomp.so.0; device memory handler on the default pool.
//
// INTERFACE CONTRACT (set by the integration owner).
#pragma once

#include <cudss.h>

#include <vector>

#include "common/buffer.hpp"
#include "cupiqp/cupiqp.hpp"

namespace cupiqp {

class CudssSolver {
 public:
  explicit CudssSolver(Counters* counters);
  ~CudssSolver();  // descriptors, data, config, handle (in that order)
  CudssSolver(const CudssSolver&) = delete;
  CudssSolver& operator=(const CudssSolver&) = delete;

  // Create handle/config/data, install the counting device memory handler,
  // allocate rhs/sol (dim), create the CSR and dense descriptors over the given
  // device buffers (which must outlive the solver and keep their addresses).
  void init(int32_t dim, int32_t nnz, const int32_t* indptr, const int32_t* indices,
            double* values, bool deterministic, cudaStream_t s);

  double* rhs() { return rhs_.data(); }  // (dim,), stable
  double* sol() { return sol_.data(); }  // (dim,), stable

  void analyze(cudaStream_t s);  // ANALYSIS (once, at setup)
  void factor(cudaStream_t s);   // FACTORIZATION, async, capturable (also in conditional bodies)
  void solve(cudaStream_t s);    // SOLVE, async, capturable in plain graphs only

 private:
  Counters* counters_;
  cudssHandle_t handle_ = nullptr;
  cudssConfig_t config_ = nullptr;
  cudssData_t data_ = nullptr;
  cudssMatrix_t A_ = nullptr, x_ = nullptr, b_ = nullptr;
  CudaBuffer<double> rhs_, sol_;
};

// Re-point every memcpy node of `graph` (recursing into child and conditional
// body graphs) whose source is pageable host memory to a pinned snapshot of
// the same bytes (FEASIBILITY.md 3.3). Snapshots are appended to `keep`, which
// must outlive the instantiated graph. Returns the number of patched nodes.
int patch_pageable_memcpy(cudaGraph_t graph, std::vector<CudaPinnedBuffer<unsigned char>>& keep);

}  // namespace cupiqp
