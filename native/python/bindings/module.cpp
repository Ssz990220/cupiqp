// cupiqp_native._ext: nanobind module (integration owner).
// Public solver API on the module itself; per-module parity hooks under
// cupiqp_native._ext._testing. Device arrays cross the boundary as integer
// addresses (uintptr) plus explicit sizes; cupiqp_native/_interop.py converts
// __cuda_array_interface__ objects and wraps results as Warp arrays.
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include "cupiqp/cupiqp.hpp"

namespace nb = nanobind;

void bind_data(nb::module_&, nb::module_&);
void bind_spmv(nb::module_&, nb::module_&);
void bind_ruiz(nb::module_&, nb::module_&);
void bind_kkt_assembly(nb::module_&, nb::module_&);
void bind_ipm_kernels(nb::module_&, nb::module_&);
void bind_kkt_system(nb::module_&, nb::module_&);
void bind_solver(nb::module_&, nb::module_&);

NB_MODULE(_ext, m) {
  m.doc() = "cuPIQP native CUDA solver (nanobind binding)";
  nb::exception<cupiqp::Error>(m, "Error", PyExc_RuntimeError);
  m.def("version", &cupiqp::version_string);
  m.attr("PIQP_INF") = cupiqp::PIQP_INF;
  nb::module_ testing = m.def_submodule("_testing", "Per-module parity entry points (tests only)");
  testing.def("synchronize", []() {
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) throw cupiqp::Error(cupiqp::ErrorCode::Cuda, cudaGetErrorString(e));
  });
  bind_data(m, testing);
  bind_spmv(m, testing);
  bind_ruiz(m, testing);
  bind_kkt_assembly(m, testing);
  bind_ipm_kernels(m, testing);
  bind_kkt_system(m, testing);
  bind_solver(m, testing);
}
