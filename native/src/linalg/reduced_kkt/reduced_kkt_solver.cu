// Reduced (Schur-complement) KKT backend. See reduced_kkt_solver.hpp.
#include "linalg/reduced_kkt/reduced_kkt_solver.hpp"

#include <algorithm>
#include <string>

#include "common/check.hpp"
#include "linalg/reduced_kkt/reduced_kkt_kernels.cuh"

namespace cupiqp {

namespace {

using namespace reduced_kkt;

template <class T>
void upload(CudaBuffer<T>& d, const std::vector<T>& h, cudaStream_t s) {
  d.allocate(std::max<size_t>(h.size(), 1));
  d.zero_async(s);
  if (!h.empty()) d.copy_from_host(h.data(), h.size(), s);
}

template <int W>
void set_assembly_smem(size_t bytes) {
  if (bytes > 48 * 1024)
    CUPIQP_CUDA_CHECK(cudaFuncSetAttribute(assemble_kernel<W>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                           int(bytes)));
}

}  // namespace

void ReducedKktSolver::init(KktAssembly& kkt, cudaStream_t s) {
  init(kkt.host_maps(), kkt.indices(), kkt.values(), s);
}

void ReducedKktSolver::init(const KktHostMaps& maps, const int32_t* d_indices, double* d_values, cudaStream_t s) {
  CUPIQP_REQUIRE(n_ == 0, "ReducedKktSolver::init called twice");
  CUPIQP_REQUIRE(d_indices && d_values, "ReducedKktSolver::init: null K pointer");
  const int32_t n = maps.n, p = maps.p, m = maps.m, mm = p + m, dim = n + mm;
  CUPIQP_REQUIRE(n > 0, "ReducedKktSolver::init: n must be positive");
  if (n > kMaxN)
    throw Error(ErrorCode::Unsupported, "kkt_backend = Reduced supports n <= " + std::to_string(kMaxN) +
                                            " (dense n x n Schur complement); this problem has n = " +
                                            std::to_string(n) + ". Use kkt_backend = Cudss.");
  const CsrPattern& kp = maps.kkt;
  CUPIQP_REQUIRE(kp.rows == dim && int64_t(kp.indptr.size()) == int64_t(dim) + 1,
                 "ReducedKktSolver::init: KKT pattern shape mismatch");
  n_ = n;
  mm_ = mm;
  Kidx_ = d_indices;
  K_ = d_values;

  // P + diag(x_reg): lower part (columns <= row) of the x rows of K.
  std::vector<int32_t> Pptr(n + 1, 0), Pcol, Pslot;
  for (int32_t r = 0; r < n; ++r) {
    for (int32_t e = kp.indptr[r]; e < kp.indptr[r + 1]; ++e) {
      const int32_t c = kp.indices[e];
      CUPIQP_REQUIRE(c < n, "ReducedKktSolver::init: unexpected upper entry in the KKT x rows");
      if (c <= r) {
        Pcol.push_back(c);
        Pslot.push_back(e);
      }
    }
    Pptr[r + 1] = int32_t(Pcol.size());
  }
  // y / z rows: off-diagonal entries (columns < n) first, the diagonal last.
  std::vector<int32_t> Dslot(mm), Kbase(mm), Mlen(mm);
  std::vector<int32_t> Mcount(n + 1, 0);
  for (int32_t k = 0; k < mm; ++k) {
    const int32_t r = n + k, b = kp.indptr[r], e = kp.indptr[r + 1];
    CUPIQP_REQUIRE(e > b && kp.indices[e - 1] == r, "ReducedKktSolver::init: missing y / z diagonal");
    Dslot[k] = e - 1;
    Kbase[k] = b;
    Mlen[k] = e - 1 - b;
    for (int32_t q = b; q < e - 1; ++q) {
      CUPIQP_REQUIRE(kp.indices[q] < n, "ReducedKktSolver::init: unexpected entry in the KKT y / z block");
      ++Mcount[kp.indices[q] + 1];
    }
  }
  // CSC of M: for each column, the rows k (ascending) and the offset of the
  // entry within row k's contiguous run in K.
  for (int32_t i = 0; i < n; ++i) Mcount[i + 1] += Mcount[i];
  const int32_t nnzM = Mcount[n];
  std::vector<int32_t> Crow(nnzM), Coff(nnzM), Cslot(nnzM), fill(Mcount.begin(), Mcount.end() - 1);
  for (int32_t k = 0; k < mm; ++k)
    for (int32_t q = 0; q < Mlen[k]; ++q) {
      const int32_t at = fill[kp.indices[Kbase[k] + q]]++;
      Crow[at] = k;
      Coff[at] = q;
      Cslot[at] = Kbase[k] + q;
    }
  // Assembly tasks: split column i's CSC range so that no task does much more
  // than the average work (estimated from the pattern: entry e costs
  // Coff[e] + 1 products before zeros are dropped).
  int64_t total_work = 0;
  for (int32_t e = 0; e < nnzM; ++e) total_work += Coff[e] + 1;
  const int64_t target = std::max<int64_t>(4096, total_work / std::max<int64_t>(1, 2 * int64_t(n)));
  std::vector<int32_t> task_row, task_e0, task_e1, srow, stask0, stask1;
  std::vector<int64_t> task_out;
  int64_t ws_size = 0;
  for (int32_t i = 0; i < n; ++i) {
    const int32_t t_begin = int32_t(task_row.size());
    int32_t e = Mcount[i];
    do {
      const int32_t start = e;
      int64_t wsum = 0;
      while (e < Mcount[i + 1] && (wsum == 0 || wsum + Coff[e] + 1 <= target)) wsum += Coff[e++] + 1;
      task_row.push_back(i);
      task_e0.push_back(start);
      task_e1.push_back(e);
    } while (e < Mcount[i + 1]);
    const int32_t t_end = int32_t(task_row.size());
    if (t_end - t_begin == 1) {
      task_out.push_back(-1);
    } else {
      for (int32_t t = t_begin; t < t_end; ++t) {
        task_out.push_back((ws_size << 1) | (t == t_begin ? 0 : 1));
        ws_size += i + 1;
      }
      srow.push_back(i);
      stask0.push_back(t_begin);
      stask1.push_back(t_end);
    }
  }
  num_tasks_ = int32_t(task_row.size());
  num_split_rows_ = int32_t(srow.size());
  upload(task_row_, task_row, s);
  upload(task_e0_, task_e0, s);
  upload(task_e1_, task_e1, s);
  upload(task_out_, task_out, s);
  upload(srow_, srow, s);
  upload(stask0_, stask0, s);
  upload(stask1_, stask1, s);
  ws_.allocate(std::max<int64_t>(ws_size, 1));
  ws_.zero_async(s);
  Cv_.allocate(std::max<size_t>(kp.indices.size(), 1));
  Cc_.allocate(std::max<size_t>(kp.indices.size(), 1));
  pre_.allocate(std::max<size_t>(kp.indices.size(), 1));
  Cv_.zero_async(s);
  Cc_.zero_async(s);
  pre_.zero_async(s);

  upload(Pptr_, Pptr, s);
  upload(Pcol_, Pcol, s);
  upload(Pslot_, Pslot, s);
  upload(Dslot_, Dslot, s);
  upload(Kbase_, Kbase, s);
  upload(Mlen_, Mlen, s);
  upload(Cptr_, Mcount, s);
  upload(Crow_, Crow, s);
  upload(Coff_, Coff, s);
  upload(Cslot_, Cslot, s);

  w_.allocate(std::max(mm, 1));
  H_.allocate(size_t(n) * n);
  dinv_.allocate(n);
  rhs_.allocate(dim);
  sol_.allocate(dim);
  status_.allocate(1);
  for (CudaBuffer<double>* b : {&w_, &H_, &dinv_, &rhs_, &sol_}) b->zero_async(s);
  status_.zero_async(s);

  // Assembly accumulators: (W + 1) n doubles of shared memory per block.
  int dev = 0, optin = 0;
  CUPIQP_CUDA_CHECK(cudaGetDevice(&dev));
  CUPIQP_CUDA_CHECK(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
  warps_ = 0;
  for (int W : {16, 8, 4, 2, 1})
    if (size_t(W + 1) * n * sizeof(double) <= size_t(optin)) {
      warps_ = W;
      break;
    }
  CUPIQP_REQUIRE(warps_ > 0, "ReducedKktSolver::init: n too large for the assembly shared memory");
  assembly_smem_ = size_t(warps_ + 1) * n * sizeof(double);
  switch (warps_) {
    case 16: set_assembly_smem<16>(assembly_smem_); break;
    case 8: set_assembly_smem<8>(assembly_smem_); break;
    case 4: set_assembly_smem<4>(assembly_smem_); break;
    case 2: set_assembly_smem<2>(assembly_smem_); break;
    default: set_assembly_smem<1>(assembly_smem_); break;
  }
  // Triangular-solve dataflow: ticket counter and one flag per task (2N).
  trsv_ticket_.allocate(1);
  trsv_ticket_.zero_async(s);
  trsv_flags_.allocate(size_t(2) * ((n + kTile - 1) / kTile));
  trsv_flags_.zero_async(s);
}

void ReducedKktSolver::assemble(cudaStream_t s) {
  CUPIQP_REQUIRE(n_ > 0, "ReducedKktSolver used before init()");
  if (mm_ > 0) {
    weights_kernel<<<(mm_ + 255) / 256, 256, 0, s>>>(K_, Dslot_.data(), mm_, w_.data());
    CUPIQP_CUDA_CHECK_LAUNCH();
  }
  if (mm_ > 0) {
    compact_rows_kernel<<<int((int64_t(mm_) * 32 + 255) / 256), 256, 0, s>>>(
        K_, Kidx_, Kbase_.data(), Mlen_.data(), mm_, Cv_.data(), Cc_.data(), pre_.data());
    CUPIQP_CUDA_CHECK_LAUNCH();
  }
#define CUPIQP_REDUCED_ASSEMBLE(W)                                                                             \
  assemble_kernel<W><<<num_tasks_, W * 32, assembly_smem_, s>>>(                                              \
      K_, w_.data(), Pptr_.data(), Pcol_.data(), Pslot_.data(), Kbase_.data(), Crow_.data(), Coff_.data(),     \
      Cv_.data(), Cc_.data(), pre_.data(), task_row_.data(), task_e0_.data(), task_e1_.data(), task_out_.data(), \
      n_, H_.data(), ws_.data())
  switch (warps_) {
    case 16: CUPIQP_REDUCED_ASSEMBLE(16); break;
    case 8: CUPIQP_REDUCED_ASSEMBLE(8); break;
    case 4: CUPIQP_REDUCED_ASSEMBLE(4); break;
    case 2: CUPIQP_REDUCED_ASSEMBLE(2); break;
    default: CUPIQP_REDUCED_ASSEMBLE(1); break;
  }
#undef CUPIQP_REDUCED_ASSEMBLE
  CUPIQP_CUDA_CHECK_LAUNCH();
  if (num_split_rows_ > 0) {
    reduce_split_rows_kernel<<<num_split_rows_, 256, 0, s>>>(srow_.data(), stask0_.data(), stask1_.data(),
                                                             task_out_.data(), ws_.data(), n_, H_.data());
    CUPIQP_CUDA_CHECK_LAUNCH();
  }
}

void ReducedKktSolver::cholesky(cudaStream_t s) {
  CUPIQP_REQUIRE(n_ > 0, "ReducedKktSolver used before init()");
  const int N = (n_ + kTile - 1) / kTile;
  for (int kk = 0; kk < N; ++kk) {
    const int mt = N - kk - 1;
    chol_panel_kernel<<<mt + 1, 32, 0, s>>>(H_.data(), n_, kk, dinv_.data(), status_.data());
    CUPIQP_CUDA_CHECK_LAUNCH();
    if (mt == 0) break;
    chol_update_kernel<<<dim3(mt * (kTile / kSub), mt), dim3(kTile, kSub), 0, s>>>(H_.data(), n_, kk);
    CUPIQP_CUDA_CHECK_LAUNCH();
  }
}

void ReducedKktSolver::factor(cudaStream_t s) {
  assemble(s);
  cholesky(s);
}

void ReducedKktSolver::solve(cudaStream_t s) {
  CUPIQP_REQUIRE(n_ > 0, "ReducedKktSolver used before init()");
  reduce_rhs_kernel<<<n_, kRhsBlock, 0, s>>>(K_, w_.data(), Cptr_.data(), Crow_.data(),
                                                           Cslot_.data(), rhs_.data(), n_, sol_.data());
  CUPIQP_CUDA_CHECK_LAUNCH();
  const int N = (n_ + kTile - 1) / kTile;
  trsv_flow_kernel<<<2 * N, kTile, 0, s>>>(H_.data(), dinv_.data(), n_, sol_.data(), trsv_ticket_.data(),
                                           trsv_flags_.data());
  CUPIQP_CUDA_CHECK_LAUNCH();
  if (mm_ > 0) {
    recover_kernel<<<int((int64_t(mm_) * 32 + 255) / 256), 256, 0, s>>>(K_, Kidx_, w_.data(), Kbase_.data(),
                                                                       Mlen_.data(), rhs_.data(), n_, mm_,
                                                                       sol_.data());
    CUPIQP_CUDA_CHECK_LAUNCH();
  }
}

}  // namespace cupiqp
