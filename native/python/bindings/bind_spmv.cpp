// nanobind entry points of module spmv (owned by that module).
//
// _testing.SpMVOp: one SpMV over caller-owned device CSR arrays (addresses as
// ints), callable eagerly or through a captured CUDA graph; custom=True
// selects SpmvBackend::Custom.
// _testing.SparseOps: SparseOperators over caller-owned P / A / G arrays.
// All calls enqueue on the given stream (default 0) and synchronize it, so
// results are readable from Python on return.
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <cstdint>
#include <memory>
#include <string>

#include "common/graph.hpp"
#include "common/handles.hpp"
#include "linalg/spmv/spmv.hpp"

namespace nb = nanobind;
using namespace cupiqp;

namespace {

template <class P>
P* as_ptr(uintptr_t a) {
  return reinterpret_cast<P*>(a);
}

CsrView<double> make_view(int32_t rows, int32_t cols, int32_t nnz, uintptr_t indptr, uintptr_t indices,
                          uintptr_t values) {
  CsrView<double> v;
  v.rows = rows;
  v.cols = cols;
  v.nnz = nnz;
  v.indptr = as_ptr<const int32_t>(indptr);
  v.indices = as_ptr<const int32_t>(indices);
  v.values = as_ptr<double>(values);
  return v;
}

void run_on(cudaStream_t s, bool graph, const std::function<void(cudaStream_t)>& fn) {
  if (!graph) {
    fn(s);
  } else {
    Counters c;
    // Capture needs a non-legacy stream; use a private one when given 0.
    CudaStream own;
    cudaStream_t cs = s;
    if (cs == nullptr) {
      own = CudaStream::create(cudaStreamNonBlocking);
      cs = own.get();
      CUPIQP_CUDA_CHECK(cudaDeviceSynchronize());
    }
    GraphExec exec;
    exec.instantiate(capture(cs, fn, &c));
    exec.launch(cs, &c);
    CUPIQP_CUDA_CHECK(cudaStreamSynchronize(cs));
  }
  CUPIQP_CUDA_CHECK(cudaStreamSynchronize(s));
}

struct SpMVOp {
  CusparseHandle handle;
  SpMV op;
  SpMVOp(int32_t rows, int32_t cols, int32_t nnz, uintptr_t indptr, uintptr_t indices, uintptr_t values,
         bool transpose, bool custom) {
    op.init(handle.get(), make_view(rows, cols, nnz, indptr, indices, values), transpose,
            custom ? SpmvBackend::Custom : SpmvBackend::Cusparse);
  }
  void call(uintptr_t x, uintptr_t y, double alpha, double beta, uintptr_t stream, bool graph) {
    run_on(as_ptr<CUstream_st>(stream), graph,
           [&](cudaStream_t s) { op(s, alpha, as_ptr<const double>(x), beta, as_ptr<double>(y)); });
  }
};

struct SparseOps {
  CusparseHandle handle;
  SparseOperators ops;
  SparseOps(int32_t n, int32_t p, int32_t m, int32_t nnz_P, uintptr_t P_indptr, uintptr_t P_indices,
            uintptr_t P_values, int32_t nnz_A, uintptr_t A_indptr, uintptr_t A_indices, uintptr_t A_values,
            int32_t nnz_G, uintptr_t G_indptr, uintptr_t G_indices, uintptr_t G_values, bool custom) {
    SparseDataView<double> d;
    d.dims.n = n;
    d.dims.p = p;
    d.dims.m = m;
    d.dims.nnz_P = nnz_P;
    d.dims.nnz_A = nnz_A;
    d.dims.nnz_G = nnz_G;
    d.P = make_view(n, n, nnz_P, P_indptr, P_indices, P_values);
    d.A = make_view(p, n, nnz_A, A_indptr, A_indices, A_values);
    d.G = make_view(m, n, nnz_G, G_indptr, G_indices, G_values);
    ops.init(handle.get(), d, custom ? SpmvBackend::Custom : SpmvBackend::Cusparse);
  }
  void apply(const std::string& which, double alpha, uintptr_t x, uintptr_t y, uintptr_t stream, bool graph) {
    auto xp = as_ptr<const double>(x);
    auto yp = as_ptr<double>(y);
    run_on(as_ptr<CUstream_st>(stream), graph, [&](cudaStream_t s) {
      if (which == "P") ops.P(s, alpha, xp, yp);
      else if (which == "A") ops.A(s, alpha, xp, yp);
      else if (which == "AT") ops.AT(s, alpha, xp, yp);
      else if (which == "G") ops.G(s, alpha, xp, yp);
      else if (which == "GT") ops.GT(s, alpha, xp, yp);
      else throw Error(ErrorCode::InvalidArgument, "SparseOps.apply: unknown operator " + which);
    });
  }
};

}  // namespace

void bind_spmv(nb::module_& m, nb::module_& testing) {
  (void)m;
  nb::class_<SpMVOp>(testing, "SpMVOp")
      .def(nb::init<int32_t, int32_t, int32_t, uintptr_t, uintptr_t, uintptr_t, bool, bool>(), nb::arg("rows"),
           nb::arg("cols"), nb::arg("nnz"), nb::arg("indptr"), nb::arg("indices"), nb::arg("values"),
           nb::arg("transpose") = false, nb::arg("custom") = false)
      .def("__call__", &SpMVOp::call, nb::arg("x"), nb::arg("y"), nb::arg("alpha") = 1.0,
           nb::arg("beta") = 0.0, nb::arg("stream") = 0, nb::arg("graph") = false)
      .def_prop_ro("uses_cusparse", [](const SpMVOp& o) { return o.op.uses_cusparse(); })
      .def_prop_ro("uses_custom", [](const SpMVOp& o) { return o.op.uses_custom(); })
      .def_prop_ro("lanes", [](const SpMVOp& o) { return o.op.lanes(); });
  nb::class_<SparseOps>(testing, "SparseOps")
      .def(nb::init<int32_t, int32_t, int32_t, int32_t, uintptr_t, uintptr_t, uintptr_t, int32_t, uintptr_t,
                    uintptr_t, uintptr_t, int32_t, uintptr_t, uintptr_t, uintptr_t, bool>(),
           nb::arg("n"), nb::arg("p"), nb::arg("m"), nb::arg("nnz_P"), nb::arg("P_indptr"), nb::arg("P_indices"),
           nb::arg("P_values"), nb::arg("nnz_A"), nb::arg("A_indptr"), nb::arg("A_indices"), nb::arg("A_values"),
           nb::arg("nnz_G"), nb::arg("G_indptr"), nb::arg("G_indices"), nb::arg("G_values"),
           nb::arg("custom") = false)
      .def("apply", &SparseOps::apply, nb::arg("which"), nb::arg("alpha"), nb::arg("x"), nb::arg("y"),
           nb::arg("stream") = 0, nb::arg("graph") = false);
}
