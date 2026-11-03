// nanobind entry points of module cudss (owned by that module).
//
// _testing.CudssSolver wraps cupiqp::CudssSolver over caller-owned device CSR
// arrays (addresses as Python ints; they must outlive the object and keep
// their addresses). rhs_ptr / sol_ptr expose the solver-owned (dim,) buffers.
// Every call enqueues on the given stream (integer handle, 0 = legacy default
// stream) and then synchronizes it, so results are readable on return.
// launch_graph() captures FACTORIZATION + SOLVE into a plain CUDA graph on
// its first call (on that stream) and replays it.
#include <nanobind/nanobind.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/string.h>

#include <cstdint>
#include <map>
#include <memory>
#include <string>

#include "common/graph.hpp"
#include "linalg/cudss/cudss_solver.hpp"

namespace nb = nanobind;
using namespace cupiqp;

namespace {

cudaStream_t as_stream(uintptr_t s) { return reinterpret_cast<cudaStream_t>(s); }

struct TestingCudss {
  Counters counters;
  CudssSolver solver{&counters};
  GraphExec graph;

  void sync(uintptr_t s) { CUPIQP_CUDA_CHECK(cudaStreamSynchronize(as_stream(s))); }
};

}  // namespace

void bind_cudss(nb::module_& m, nb::module_& testing) {
  (void)m;
  nb::class_<TestingCudss>(testing, "CudssSolver")
      .def(
          "__init__",
          [](TestingCudss* self, int32_t dim, int32_t nnz, uintptr_t indptr, uintptr_t indices, uintptr_t values,
             bool deterministic, uintptr_t stream) {
            new (self) TestingCudss();
            self->solver.init(dim, nnz, reinterpret_cast<const int32_t*>(indptr),
                              reinterpret_cast<const int32_t*>(indices), reinterpret_cast<double*>(values),
                              deterministic, as_stream(stream));
            self->sync(stream);
          },
          nb::arg("dim"), nb::arg("nnz"), nb::arg("indptr"), nb::arg("indices"), nb::arg("values"),
          nb::arg("deterministic") = false, nb::arg("stream") = 0)
      .def_prop_ro("dim", [](TestingCudss& t) { return t.solver.dim(); })
      .def_prop_ro("rhs_ptr", [](TestingCudss& t) { return reinterpret_cast<uintptr_t>(t.solver.rhs()); })
      .def_prop_ro("sol_ptr", [](TestingCudss& t) { return reinterpret_cast<uintptr_t>(t.solver.sol()); })
      .def(
          "analyze", [](TestingCudss& t, uintptr_t s) { t.solver.analyze(as_stream(s)), t.sync(s); },
          nb::arg("stream") = 0)
      .def(
          "factor", [](TestingCudss& t, uintptr_t s) { t.solver.factor(as_stream(s)), t.sync(s); },
          nb::arg("stream") = 0)
      .def(
          "solve", [](TestingCudss& t, uintptr_t s) { t.solver.solve(as_stream(s)), t.sync(s); },
          nb::arg("stream") = 0)
      .def(
          "launch_graph",
          [](TestingCudss& t, uintptr_t s) {
            if (!t.graph.valid())
              t.graph.instantiate(capture(as_stream(s), [&](cudaStream_t c) {
                t.solver.factor(c);
                t.solver.solve(c);
              }, &t.counters));
            t.graph.launch(as_stream(s), &t.counters);
            t.sync(s);
          },
          nb::arg("stream") = 0)
      .def("counters", [](TestingCudss& t) {
        const Counters& c = t.counters;
        const CudssMemory& mem = t.solver.memory();
        return std::map<std::string, int64_t>{
            {"cudss_device_allocs", c.cudss_device_allocs},
            {"cudss_device_frees", c.cudss_device_frees},
            {"cudss_analysis_calls", c.cudss_analysis_calls},
            {"cudss_factor_calls", c.cudss_factor_calls},
            {"cudss_solve_calls", c.cudss_solve_calls},
            {"graph_captures", c.graph_captures},
            {"graph_launches", c.graph_launches},
            {"zero_byte_requests", mem.zero_byte_requests()},
            {"peak_live_bytes", static_cast<int64_t>(mem.peak_live_bytes())},
        };
      });
}
