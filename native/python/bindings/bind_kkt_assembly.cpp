// nanobind entry points of module kkt_assembly (owned by that module).
//
// _testing.KktAssembly wraps cupiqp::KktAssembly for parity tests. Device
// arrays are passed as integer addresses (cupiqp_native._interop.dev_ptr);
// patterns as host int32 NumPy arrays. Every call runs on the given stream
// handle and synchronizes it (tests only).
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#include <cstdint>
#include <string>
#include <vector>

#include "common/check.hpp"
#include "linalg/kkt_assembly/kkt_assembly.hpp"

namespace nb = nanobind;

namespace {

using cupiqp::CsrPattern;
using IntArray = nb::ndarray<const int32_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

CsrPattern to_pattern(int32_t rows, int32_t cols, const IntArray& indptr, const IntArray& indices) {
  CsrPattern M;
  if (rows == 0) return M;
  M.rows = rows;
  M.cols = cols;
  M.indptr.assign(indptr.data(), indptr.data() + indptr.shape(0));
  M.indices.assign(indices.data(), indices.data() + indices.shape(0));
  return M;
}

template <class T>
nb::ndarray<nb::numpy, T, nb::ndim<1>> to_numpy(std::vector<T> v) {
  auto* heap = new std::vector<T>(std::move(v));
  nb::capsule owner(heap, [](void* p) noexcept { delete static_cast<std::vector<T>*>(p); });
  return nb::ndarray<nb::numpy, T, nb::ndim<1>>(heap->data(), {heap->size()}, owner);
}

template <class T>
nb::ndarray<nb::numpy, T, nb::ndim<1>> download(const T* dev, size_t count) {
  std::vector<T> h(count);
  if (count)
    CUPIQP_CUDA_CHECK(cudaMemcpy(h.data(), dev, count * sizeof(T), cudaMemcpyDeviceToHost));
  return to_numpy(std::move(h));
}

cudaStream_t as_stream(uintptr_t s) { return reinterpret_cast<cudaStream_t>(s); }

struct KktAssemblyTesting {
  cupiqp::KktAssembly kkt;
  cupiqp::Dims dims;

  // Device pointers: values of P/A/G (B = 1), COO rows of G, active_G_row.
  cupiqp::SparseDataView<double> view(uintptr_t P, uintptr_t A, uintptr_t G, uintptr_t G_rows,
                                      uintptr_t active_G_row) const {
    cupiqp::SparseDataView<double> v;
    v.dims = dims;
    v.P.rows = v.P.cols = dims.n;
    v.P.nnz = dims.nnz_P;
    v.P.values = reinterpret_cast<double*>(P);
    v.A.rows = dims.p;
    v.A.cols = dims.n;
    v.A.nnz = dims.nnz_A;
    v.A.values = reinterpret_cast<double*>(A);
    v.G.rows = dims.m;
    v.G.cols = dims.n;
    v.G.nnz = dims.nnz_G;
    v.G.row_idx = reinterpret_cast<const int32_t*>(G_rows);
    v.G.values = reinterpret_cast<double*>(G);
    v.active_G_row = reinterpret_cast<double*>(active_G_row);
    return v;
  }
  void sync(uintptr_t s) { CUPIQP_CUDA_CHECK(cudaStreamSynchronize(as_stream(s))); }
};

}  // namespace

void bind_kkt_assembly(nb::module_& m, nb::module_& testing) {
  (void)m;
  nb::class_<KktAssemblyTesting>(testing, "KktAssembly")
      .def("__init__",
           [](KktAssemblyTesting* self, int32_t n, int32_t p, int32_t m_, IntArray P_indptr,
              IntArray P_indices, IntArray A_indptr, IntArray A_indices, IntArray G_indptr,
              IntArray G_indices, uintptr_t P_val, uintptr_t A_val, uintptr_t G_val, uintptr_t G_rows,
              uintptr_t active_G_row, uintptr_t stream) {
             new (self) KktAssemblyTesting();
             CsrPattern P = to_pattern(n, n, P_indptr, P_indices);
             CsrPattern A = to_pattern(p, n, A_indptr, A_indices);
             CsrPattern G = to_pattern(m_, n, G_indptr, G_indices);
             self->dims.n = n;
             self->dims.p = p;
             self->dims.m = m_;
             self->dims.nnz_P = P.nnz();
             self->dims.nnz_A = A.nnz();
             self->dims.nnz_G = G.nnz();
             self->kkt.init(P, A, G, self->view(P_val, A_val, G_val, G_rows, active_G_row),
                            as_stream(stream));
             self->sync(stream);
           },
           nb::arg("n"), nb::arg("p"), nb::arg("m"), nb::arg("P_indptr"), nb::arg("P_indices"),
           nb::arg("A_indptr"), nb::arg("A_indices"), nb::arg("G_indptr"), nb::arg("G_indices"),
           nb::arg("P_val"), nb::arg("A_val"), nb::arg("G_val"), nb::arg("G_rows"),
           nb::arg("active_G_row"), nb::arg("stream"))
      .def("scatter",
           [](KktAssemblyTesting& self, uintptr_t P_val, uintptr_t A_val, uintptr_t G_val,
              uintptr_t G_rows, uintptr_t active_G_row, bool P, bool A, bool G, uintptr_t stream) {
             self.kkt.scatter(self.view(P_val, A_val, G_val, G_rows, active_G_row), P, A, G,
                              as_stream(stream));
             self.sync(stream);
           },
           nb::arg("P_val"), nb::arg("A_val"), nb::arg("G_val"), nb::arg("G_rows"),
           nb::arg("active_G_row"), nb::arg("P"), nb::arg("A"), nb::arg("G"), nb::arg("stream"))
      .def("update_diagonal",
           [](KktAssemblyTesting& self, uintptr_t x_reg, uintptr_t delta, uintptr_t z_reg,
              uintptr_t stream) {
             self.kkt.update_diagonal(reinterpret_cast<const double*>(x_reg),
                                      reinterpret_cast<const double*>(delta),
                                      reinterpret_cast<const double*>(z_reg), as_stream(stream));
             self.sync(stream);
           },
           nb::arg("x_reg"), nb::arg("delta"), nb::arg("z_reg"), nb::arg("stream"))
      .def_prop_ro("dim", [](const KktAssemblyTesting& self) { return self.kkt.dim(); })
      .def_prop_ro("nnz", [](const KktAssemblyTesting& self) { return self.kkt.nnz(); })
      .def_prop_ro("values_ptr",
                   [](KktAssemblyTesting& self) { return reinterpret_cast<uintptr_t>(self.kkt.values()); })
      .def("indptr", [](const KktAssemblyTesting& self) { return download(self.kkt.indptr(), self.kkt.dim() + 1); })
      .def("indices", [](const KktAssemblyTesting& self) { return download(self.kkt.indices(), self.kkt.nnz()); })
      .def("values", [](KktAssemblyTesting& self) { return download<double>(self.kkt.values(), self.kkt.nnz()); })
      .def("P_diag", [](const KktAssemblyTesting& self) { return download(self.kkt.P_diag(), self.dims.n); })
      .def("template_values",
           [](const KktAssemblyTesting& self) { return to_numpy(self.kkt.host_maps().template_values); })
      .def("host_map", [](const KktAssemblyTesting& self, const std::string& name) {
        const cupiqp::KktHostMaps& h = self.kkt.host_maps();
        const std::vector<int32_t>* v = nullptr;
        if (name == "P_slots") v = &h.P_slots;
        else if (name == "A_slots") v = &h.A_slots;
        else if (name == "G_slots") v = &h.G_slots;
        else if (name == "diag_x") v = &h.diag_x;
        else if (name == "diag_y") v = &h.diag_y;
        else if (name == "diag_z") v = &h.diag_z;
        else if (name == "P_diag_src") v = &h.P_diag_src;
        else if (name == "P_diag_dst") v = &h.P_diag_dst;
        else throw cupiqp::Error(cupiqp::ErrorCode::InvalidArgument, "unknown map " + name);
        return to_numpy(*v);
      }, nb::arg("name"));
}
