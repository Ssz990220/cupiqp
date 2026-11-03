// SparseData (module data): problem storage of the sparse backend.
// Reference: cupiqp/data.py, cupiqp/sparse/sparse_data.py,
// cupiqp/sparse/batched_csr.py, cupiqp/sparse/csr_helpers.py.
#include <cstdint>
#include <string>
#include <vector>

#include "common/check.hpp"
#include "common/reduce.cuh"
#include "problem/data/finite_masks_kernels.cuh"
#include "problem/data/sparse_data.hpp"

namespace cupiqp {

namespace {

// Row index of every stored entry (reference csr_helpers.csr_row_indices).
std::vector<int32_t> csr_row_indices(const CsrPattern& M) {
  std::vector<int32_t> rows(static_cast<size_t>(M.nnz()));
  for (int32_t r = 0; r < M.rows; ++r) {
    for (int32_t k = M.indptr[r]; k < M.indptr[r + 1]; ++k) rows[k] = r;
  }
  return rows;
}

void upload_pattern(const CsrPattern& M, CudaBuffer<int32_t>& indptr, CudaBuffer<int32_t>& indices,
                    CudaBuffer<int32_t>& rows, cudaStream_t s) {
  const size_t nnz = static_cast<size_t>(M.nnz());
  indptr.allocate(M.indptr.size());
  indices.allocate(nnz);
  rows.allocate(nnz);
  indptr.copy_from_host(M.indptr.data(), M.indptr.size(), s);
  indices.copy_from_host(M.indices.data(), nnz, s);
  // Pageable H2D cudaMemcpyAsync returns after the source has been staged, so
  // the temporary may be released right after the call.
  const std::vector<int32_t> r = csr_row_indices(M);
  rows.copy_from_host(r.data(), nnz, s);
}

template <class T>
void copy_values(CudaBuffer<T>& dst, const T* src, size_t count, cudaMemcpyKind kind, cudaStream_t s) {
  if (count == 0) return;
  CUPIQP_CUDA_CHECK(cudaMemcpyAsync(dst.data(), src, count * sizeof(T), kind, s));
}

FiniteMaskShape mask_shape(const Dims& d) {
  FiniteMaskShape sh;
  sh.B = d.B;
  sh.m = d.m;
  sh.n = d.n;
  sh.num_hl = d.num_hl;
  sh.num_hu = d.num_hu;
  sh.num_xl = d.num_xl;
  sh.num_xu = d.num_xu;
  return sh;
}

}  // namespace

void SparseData::init(const CsrPattern& P, const CsrPattern* A, const CsrPattern* G,
                      const ValuesView& v, cudaStream_t s) {
  using detail::throw_invalid;
  if (initialized_) throw_invalid("SparseData::init may only be called once.");

  // ---- validation (no state change before it passes)
  validate_csr(P, -1, "P");
  const int32_t n = P.rows;
  if (A) validate_csr(*A, n, "A");
  if (G) validate_csr(*G, n, "G");

  // Block / vector consistency (reference SparseData.init). A null device
  // pointer cannot be told apart from an empty array, so a vector of length 0
  // may be passed as nullptr.
  const int32_t p = A ? A->rows : 0;
  const int32_t m = G ? G->rows : 0;
  if ((A == nullptr) != (v.b == nullptr) && !(A && p == 0)) {
    throw_invalid("A and b must either both be provided or both be None.");
  }
  if (G && v.h_l == nullptr && v.h_u == nullptr && m > 0) {
    throw_invalid("Either h_l or h_u must be provided when G is given.");
  }
  if (!G && (v.h_u != nullptr || v.h_l != nullptr)) {
    throw_invalid("h_l and h_u must be None when G is None.");
  }
  if (v.P == nullptr && P.nnz() > 0) throw_invalid("SparseData requires the P values.");
  if (v.c == nullptr && n > 0) throw_invalid("SparseData requires c.");
  if (A == nullptr && v.A != nullptr) throw_invalid("A values were given without an A pattern.");
  if (G == nullptr && v.G != nullptr) throw_invalid("G values were given without a G pattern.");
  if (A && v.A == nullptr && A->nnz() > 0) throw_invalid("A values are required when A is given.");
  if (G && v.G == nullptr && G->nnz() > 0) throw_invalid("G values are required when G is given.");

  // ---- structure
  P_host_ = P;
  if (A) {
    A_host_ = *A;
  } else {
    A_host_ = CsrPattern{};
    A_host_.rows = 0;
    A_host_.cols = n;
    A_host_.indptr = {0};
  }
  if (G) {
    G_host_ = *G;
  } else {
    G_host_ = CsrPattern{};
    G_host_.rows = 0;
    G_host_.cols = n;
    G_host_.indptr = {0};
  }
  has_hl_ = G && v.h_l != nullptr;
  has_hu_ = G && v.h_u != nullptr;
  has_xl_ = v.x_l != nullptr;
  has_xu_ = v.x_u != nullptr;

  Dims d;
  d.B = 1;
  d.n = n;
  d.p = p;
  d.m = m;
  d.num_hl = has_hl_ ? m : 0;
  d.num_hu = has_hu_ ? m : 0;
  d.num_xl = has_xl_ ? n : 0;
  d.num_xu = has_xu_ ? n : 0;
  d.nnz_P = P_host_.nnz();
  d.nnz_A = A_host_.nnz();
  d.nnz_G = G_host_.nnz();
  dims_ = d;

  // ---- allocation (the only place this object allocates)
  upload_pattern(P_host_, P_indptr_, P_indices_, P_rows_, s);
  upload_pattern(A_host_, A_indptr_, A_indices_, A_rows_, s);
  upload_pattern(G_host_, G_indptr_, G_indices_, G_rows_, s);
  P_val_.allocate(d.nnz_P);
  A_val_.allocate(d.nnz_A);
  G_val_.allocate(d.nnz_G);
  c_.allocate(n);
  b_.allocate(p);
  h_l_.allocate(d.num_hl);
  h_u_.allocate(d.num_hu);
  x_l_.allocate(d.num_xl);
  x_u_.allocate(d.num_xu);
  finite_mask_all_.allocate(d.num_ineq());
  active_G_row_.allocate(m);
  active_x_bound_.allocate(n);
  num_finite_bounds_.allocate(d.B);
  initialized_ = true;

  // ---- initial values (stream ordered) and masks
  apply_values(v, cudaMemcpyDeviceToDevice, s);
  update_finite_masks(s);
}

SparseDataView<double> SparseData::view() {
  SparseDataView<double> out;
  out.dims = dims_;
  auto csr = [&](const CsrPattern& h, CudaBuffer<int32_t>& ip, CudaBuffer<int32_t>& ix,
                 CudaBuffer<int32_t>& rw, CudaBuffer<double>& val) {
    CsrView<double> c;
    c.rows = h.rows;
    c.cols = h.cols;
    c.nnz = h.nnz();
    c.indptr = ip.data();
    c.indices = ix.data();
    c.row_idx = rw.data();
    c.values = val.data();
    return c;
  };
  out.P = csr(P_host_, P_indptr_, P_indices_, P_rows_, P_val_);
  out.A = csr(A_host_, A_indptr_, A_indices_, A_rows_, A_val_);
  out.G = csr(G_host_, G_indptr_, G_indices_, G_rows_, G_val_);
  out.c = c_.data();
  out.b = b_.data();
  out.h_l = h_l_.data();
  out.h_u = h_u_.data();
  out.x_l = x_l_.data();
  out.x_u = x_u_.data();
  out.finite_mask_all = finite_mask_all_.data();
  out.active_G_row = active_G_row_.data();
  out.active_x_bound = active_x_bound_.data();
  out.num_finite_bounds = num_finite_bounds_.data();
  return out;
}

void SparseData::check_update(const ValuesView& v) const {
  using detail::throw_invalid;
  if (!initialized_) throw_invalid("SparseData: init() must be called before setting values.");
  // Reference Data._set_bound message.
  auto bound = [&](const double* ptr, bool has, const char* name) {
    if (ptr != nullptr && !has) {
      throw_invalid(std::string("Cannot set ") + name + ": no " + name +
                    " block was provided at setup(). Adding a bound block requires a new setup().");
    }
  };
  bound(v.h_u, has_hu_, "h_u");
  bound(v.h_l, has_hl_, "h_l");
  bound(v.x_u, has_xu_, "x_u");
  bound(v.x_l, has_xl_, "x_l");
}

UpdateMask SparseData::apply_values(const ValuesView& v, cudaMemcpyKind kind, cudaStream_t s) {
  const Dims& d = dims_;
  UpdateMask u;
  // Reference SparseSolver._update_impl order: P, c, A, b, G, h_u, h_l, x_u, x_l.
  if (v.P) { copy_values(P_val_, v.P, d.nnz_P, kind, s); u.P = true; }
  if (v.c) { copy_values(c_, v.c, d.n, kind, s); u.c = true; }
  if (v.A) { copy_values(A_val_, v.A, d.nnz_A, kind, s); u.A = true; }
  if (v.b) { copy_values(b_, v.b, d.p, kind, s); u.b = true; }
  if (v.G) { copy_values(G_val_, v.G, d.nnz_G, kind, s); u.G = true; }
  if (v.h_u) { copy_values(h_u_, v.h_u, d.num_hu, kind, s); u.h_u = true; }
  if (v.h_l) { copy_values(h_l_, v.h_l, d.num_hl, kind, s); u.h_l = true; }
  if (v.x_u) { copy_values(x_u_, v.x_u, d.num_xu, kind, s); u.x_u = true; }
  if (v.x_l) { copy_values(x_l_, v.x_l, d.num_xl, kind, s); u.x_l = true; }
  return u;
}

UpdateMask SparseData::set_values(const ValuesView& v, cudaStream_t s) {
  check_update(v);
  const UpdateMask u = apply_values(v, cudaMemcpyDeviceToDevice, s);
  // The reference refreshes the masks after every bound setter; one refresh
  // after all copies gives the same final masks.
  if (u.any_ineq_bound() || u.any_box_bound()) update_finite_masks(s);
  return u;
}

UpdateMask SparseData::set_values_from_host(const ValuesView& host_v, cudaStream_t s) {
  check_update(host_v);
  const UpdateMask u = apply_values(host_v, cudaMemcpyHostToDevice, s);
  if (u.any_ineq_bound() || u.any_box_bound()) update_finite_masks(s);
  return u;
}

void SparseData::update_finite_masks(cudaStream_t s) {
  CUPIQP_REQUIRE(initialized_, "SparseData: init() must be called first.");
  const FiniteMaskShape sh = mask_shape(dims_);
  const int block = reduction_block_dim(static_cast<int64_t>(sh.m > sh.n ? sh.m : sh.n));
  dispatch_block(block, [&](auto bc) {
    constexpr int BLOCK = decltype(bc)::value;
    finite_bound_masks_block_kernel<BLOCK, double><<<sh.B, BLOCK, 0, s>>>(
        sh, h_l_.data(), h_u_.data(), x_l_.data(), x_u_.data(), finite_mask_all_.data(),
        active_G_row_.data(), active_x_bound_.data(), num_finite_bounds_.data());
  });
  CUPIQP_CUDA_CHECK_LAUNCH();
}

void SparseData::update_finite_masks_reference(cudaStream_t s) {
  CUPIQP_REQUIRE(initialized_, "SparseData: init() must be called first.");
  const FiniteMaskShape sh = mask_shape(dims_);
  constexpr int kThreads = 32;
  finite_bound_masks_serial_kernel<double><<<grid_for(sh.B, kThreads), kThreads, 0, s>>>(
      sh, h_l_.data(), h_u_.data(), x_l_.data(), x_u_.data(), finite_mask_all_.data(),
      active_G_row_.data(), active_x_bound_.data(), num_finite_bounds_.data());
  CUPIQP_CUDA_CHECK_LAUNCH();
}

}  // namespace cupiqp
