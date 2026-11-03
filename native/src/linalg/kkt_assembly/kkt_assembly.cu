// KktAssembly: setup (host pattern build, upload, initial scatter) and the
// stream-ordered value updates. See kkt_assembly.hpp.
#include "linalg/kkt_assembly/kkt_assembly.hpp"

#include "common/check.hpp"
#include "linalg/kkt_assembly/kkt_assembly_kernels.cuh"

namespace cupiqp {

namespace {

void upload(CudaBuffer<int32_t>& dst, const std::vector<int32_t>& src, cudaStream_t s) {
  dst.allocate(src.size());
  dst.copy_from_host(src.data(), src.size(), s);
}

}  // namespace

void KktAssembly::init(const CsrPattern& P, const CsrPattern& A, const CsrPattern& G,
                       const SparseDataView<double>& data, cudaStream_t s) {
  CUPIQP_REQUIRE(values_.empty() && indptr_.empty(), "KktAssembly::init called twice");
  maps_ = build_kkt_host_maps(P, A, G);
  dims_ = data.dims;
  CUPIQP_REQUIRE(dims_.B >= 1, "KktAssembly: batch size must be >= 1");
  CUPIQP_REQUIRE(dims_.n == maps_.n && dims_.p == maps_.p && dims_.m == maps_.m,
                 "KktAssembly: data dims do not match the P/A/G patterns");
  CUPIQP_REQUIRE(data.P.nnz == P.nnz() && data.A.nnz == (maps_.p ? A.nnz() : 0) &&
                     data.G.nnz == (maps_.m ? G.nnz() : 0),
                 "KktAssembly: data nnz do not match the P/A/G patterns");
  nnz_ = maps_.kkt.nnz();
  const int32_t B = dims_.B;

  // Pattern and maps (host vectors stay alive in maps_ for the object's life).
  upload(indptr_, maps_.kkt.indptr, s);
  upload(indices_, maps_.kkt.indices, s);
  upload(diag_x_, maps_.diag_x, s);
  upload(diag_y_, maps_.diag_y, s);
  upload(diag_z_, maps_.diag_z, s);
  upload(P_slots_, maps_.P_slots, s);
  upload(A_slots_, maps_.A_slots, s);
  upload(G_slots_, maps_.G_slots, s);
  upload(P_diag_src_, maps_.P_diag_src, s);
  upload(P_diag_dst_, maps_.P_diag_dst, s);

  // Values: the bmat template in every problem of the batch.
  values_.allocate(static_cast<size_t>(B) * nnz_);
  values_.copy_from_host(maps_.template_values.data(), maps_.template_values.size(), s);
  for (int32_t b = 1; b < B; ++b) {
    CUPIQP_CUDA_CHECK(cudaMemcpyAsync(values_.data() + static_cast<size_t>(b) * nnz_, values_.data(),
                                      sizeof(double) * nnz_, cudaMemcpyDeviceToDevice, s));
  }
  P_diag_.allocate(static_cast<size_t>(B) * dims_.n);
  P_diag_.zero_async(s);

  // Reference order: P_diag refresh, then the P, A, G scatter.
  scatter(data, true, dims_.p > 0, dims_.m > 0, s);
}

void KktAssembly::scatter(const SparseDataView<double>& data, bool P, bool A, bool G, cudaStream_t s) {
  using namespace kkt_kernels;
  const int32_t B = dims_.B;
  if (P && data.P.nnz > 0) {
    const int64_t t = static_cast<int64_t>(B) * data.P.nnz;
    scatter_values_kernel<double><<<grid_for(t), kBlock, 0, s>>>(data.P.values, P_slots_.data(),
                                                                 values_.data(), B, data.P.nnz, nnz_);
    CUPIQP_CUDA_CHECK_LAUNCH();
  }
  if (A && dims_.p > 0 && data.A.nnz > 0) {
    const int64_t t = static_cast<int64_t>(B) * data.A.nnz;
    scatter_values_kernel<double><<<grid_for(t), kBlock, 0, s>>>(data.A.values, A_slots_.data(),
                                                                 values_.data(), B, data.A.nnz, nnz_);
    CUPIQP_CUDA_CHECK_LAUNCH();
  }
  if (G && dims_.m > 0 && data.G.nnz > 0) {
    CUPIQP_REQUIRE(data.G.row_idx != nullptr && data.active_G_row != nullptr,
                   "KktAssembly::scatter: G needs row_idx and active_G_row");
    const int64_t t = static_cast<int64_t>(B) * data.G.nnz;
    scatter_masked_G_kernel<double><<<grid_for(t), kBlock, 0, s>>>(
        data.G.values, data.active_G_row, data.G.row_idx, G_slots_.data(), values_.data(), B,
        data.G.nnz, dims_.m, nnz_);
    CUPIQP_CUDA_CHECK_LAUNCH();
  }
  if (P) {
    // Reference _refresh_P_diag_buffer.
    const int32_t num = static_cast<int32_t>(P_diag_src_.size());
    if (num > 0) {
      const int64_t t = static_cast<int64_t>(B) * num;
      gather_scatter_values_kernel<double><<<grid_for(t), kBlock, 0, s>>>(
          data.P.values, P_diag_src_.data(), P_diag_dst_.data(), P_diag_.data(), B, num, data.P.nnz,
          dims_.n);
      CUPIQP_CUDA_CHECK_LAUNCH();
    }
  }
}

void KktAssembly::update_diagonal(const double* x_reg, const double* delta, const double* z_reg,
                                  cudaStream_t s) {
  using namespace kkt_kernels;
  const int64_t t = static_cast<int64_t>(dims_.B) * dims_.kkt_dim();
  if (t == 0) return;
  update_kkt_diag_kernel<double><<<grid_for(t), kBlock, 0, s>>>(
      P_diag_.data(), x_reg, delta, z_reg, diag_x_.data(), diag_y_.data(), diag_z_.data(), values_.data(),
      dims_.B, dims_.n, dims_.p, dims_.m, nnz_);
  CUPIQP_CUDA_CHECK_LAUNCH();
}

}  // namespace cupiqp
