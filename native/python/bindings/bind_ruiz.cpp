// nanobind entry points of module ruiz (owned by that module).
//
// _testing.RuizTester wraps a RuizEquilibration over problem data held in
// caller-owned device arrays (passed as integer addresses). Every method
// synchronizes the device before and after its work, so Python may hand in
// Warp arrays written on Warp's stream and read the results right away.
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>

#include <cstdint>
#include <memory>
#include <string>

#include "common/check.hpp"
#include "common/handles.hpp"
#include "problem/ruiz/ruiz.hpp"

namespace nb = nanobind;

namespace {

using cupiqp::Dims;
using cupiqp::RuizEquilibration;
using cupiqp::SparseDataView;

void device_sync() { CUPIQP_CUDA_CHECK(cudaDeviceSynchronize()); }

int64_t get_int(const nb::dict& d, const char* key, int64_t dflt = 0) {
  return d.contains(key) ? nb::cast<int64_t>(d[key]) : dflt;
}
template <class P>
P* get_ptr(const nb::dict& d, const char* key) {
  return reinterpret_cast<P*>(static_cast<uintptr_t>(get_int(d, key, 0)));
}

class RuizTester {
 public:
  // dims: n, p, m, num_hl, num_hu, num_xl, num_xu, nnz_P, nnz_A, nnz_G.
  // ptrs: {P,A,G}_{indptr,indices,row_idx,values}, c, b, h_l, h_u, x_l, x_u,
  //       finite_mask_all, active_G_row, active_x_bound, num_finite_bounds
  //       (missing or 0 -> nullptr).
  RuizTester(const nb::dict& dims, const nb::dict& ptrs) : stream_(cupiqp::CudaStream::create()) {
    Dims d;
    d.B = 1;
    d.n = int32_t(get_int(dims, "n"));
    d.p = int32_t(get_int(dims, "p"));
    d.m = int32_t(get_int(dims, "m"));
    d.num_hl = int32_t(get_int(dims, "num_hl"));
    d.num_hu = int32_t(get_int(dims, "num_hu"));
    d.num_xl = int32_t(get_int(dims, "num_xl"));
    d.num_xu = int32_t(get_int(dims, "num_xu"));
    d.nnz_P = int32_t(get_int(dims, "nnz_P"));
    d.nnz_A = int32_t(get_int(dims, "nnz_A"));
    d.nnz_G = int32_t(get_int(dims, "nnz_G"));
    view_.dims = d;
    auto csr = [&](cupiqp::CsrView<double>& M, const char* name, int32_t rows, int32_t nnz) {
      const std::string s(name);
      M.rows = rows;
      M.cols = d.n;
      M.nnz = nnz;
      M.indptr = get_ptr<const int32_t>(ptrs, (s + "_indptr").c_str());
      M.indices = get_ptr<const int32_t>(ptrs, (s + "_indices").c_str());
      M.row_idx = get_ptr<const int32_t>(ptrs, (s + "_row_idx").c_str());
      M.values = get_ptr<double>(ptrs, (s + "_values").c_str());
    };
    csr(view_.P, "P", d.n, d.nnz_P);
    csr(view_.A, "A", d.p, d.nnz_A);
    csr(view_.G, "G", d.m, d.nnz_G);
    view_.c = get_ptr<double>(ptrs, "c");
    view_.b = get_ptr<double>(ptrs, "b");
    view_.h_l = get_ptr<double>(ptrs, "h_l");
    view_.h_u = get_ptr<double>(ptrs, "h_u");
    view_.x_l = get_ptr<double>(ptrs, "x_l");
    view_.x_u = get_ptr<double>(ptrs, "x_u");
    view_.finite_mask_all = get_ptr<double>(ptrs, "finite_mask_all");
    view_.active_G_row = get_ptr<double>(ptrs, "active_G_row");
    view_.active_x_bound = get_ptr<double>(ptrs, "active_x_bound");
    view_.num_finite_bounds = get_ptr<double>(ptrs, "num_finite_bounds");
    device_sync();
    ruiz_.init(view_, s());
    device_sync();
  }

  void scale_data(bool scale_cost, int max_iter) { run([&] { ruiz_.scale_data(view_, scale_cost, max_iter, s()); }); }
  void unscale_data(const nb::dict& replaced) {
    cupiqp::UpdateMask mk;
    auto flag = [&](const char* k) { return replaced.contains(k) && nb::cast<bool>(replaced[k]); };
    mk.P = flag("P");
    mk.c = flag("c");
    mk.A = flag("A");
    mk.b = flag("b");
    mk.G = flag("G");
    mk.h_l = flag("h_l");
    mk.h_u = flag("h_u");
    mk.x_l = flag("x_l");
    mk.x_u = flag("x_u");
    run([&] { ruiz_.unscale_data(view_, mk, s()); });
  }
  void reuse_scaling() { run([&] { ruiz_.reuse_scaling(view_, s()); }); }
  void reset() { run([&] { ruiz_.reset(s()); }); }
  void refresh_x_b_scaling() { run([&] { ruiz_.refresh_x_b_scaling(s()); }); }
  void unscale_solution(uintptr_t primal, uintptr_t dual) {
    auto v = vars(primal, dual);
    run([&] { ruiz_.unscale_solution(v, s()); });
  }
  void scale_solution(uintptr_t primal, uintptr_t dual) {
    auto v = vars(primal, dual);
    run([&] { ruiz_.scale_solution(v, s()); });
  }
  void rhs_inf_norm(uintptr_t out) {
    run([&] { ruiz_.compute_constraints_rhs_inf_norm_unscaled(view_, reinterpret_cast<double*>(out), s()); });
  }
  int passes_run() {
    int32_t h = 0;
    device_sync();
    CUPIQP_CUDA_CHECK(cudaMemcpy(&h, ruiz_.passes_run(), sizeof(h), cudaMemcpyDeviceToHost));
    return h;
  }
  // name -> (device address, element count) of every ScalingView buffer.
  nb::dict state() {
    auto v = ruiz_.view();
    const Dims& d = view_.dims;
    const int64_t W = int64_t(d.n) + d.p + d.m;
    nb::dict out;
    auto put = [&](const char* k, const void* p, int64_t n) {
      out[k] = nb::make_tuple(reinterpret_cast<uintptr_t>(p), n);
    };
    put("delta", v.delta, W);
    put("delta_inv", v.delta_inv, W);
    put("delta_b", v.delta_b, d.n);
    put("delta_b_inv", v.delta_b_inv, d.n);
    put("cost_scaling", v.cost_scaling, 1);
    put("cost_scaling_inv", v.cost_scaling_inv, 1);
    put("x_b_scaling", v.x_b_scaling, d.n);
    put("x_b_scaling_init", v.x_b_scaling_init, d.n);
    put("dual_res_unscale_factor", v.dual_res_unscale_factor, d.n);
    put("primal_res_unscale_factor", v.primal_res_unscale_factor, int64_t(d.p) + d.num_ineq());
    return out;
  }

 private:
  cudaStream_t s() const { return stream_.get(); }
  template <class F>
  void run(F&& f) {
    device_sync();
    f();
    CUPIQP_CUDA_CHECK(cudaStreamSynchronize(s()));
  }
  cupiqp::VariablesView<double> vars(uintptr_t primal, uintptr_t dual) const {
    cupiqp::VariablesView<double> v;
    v.dims = view_.dims;
    v.primal = reinterpret_cast<double*>(primal);
    v.dual = reinterpret_cast<double*>(dual);
    return v;
  }

  cupiqp::CudaStream stream_;
  SparseDataView<double> view_;
  RuizEquilibration ruiz_;
};

}  // namespace

void bind_ruiz(nb::module_& m, nb::module_& testing) {
  (void)m;
  nb::class_<RuizTester>(testing, "RuizTester")
      .def(nb::init<const nb::dict&, const nb::dict&>(), nb::arg("dims"), nb::arg("ptrs"))
      .def("scale_data", &RuizTester::scale_data, nb::arg("scale_cost"), nb::arg("max_iter"))
      .def("unscale_data", &RuizTester::unscale_data, nb::arg("replaced"))
      .def("reuse_scaling", &RuizTester::reuse_scaling)
      .def("reset", &RuizTester::reset)
      .def("refresh_x_b_scaling", &RuizTester::refresh_x_b_scaling)
      .def("unscale_solution", &RuizTester::unscale_solution, nb::arg("primal"), nb::arg("dual"))
      .def("scale_solution", &RuizTester::scale_solution, nb::arg("primal"), nb::arg("dual"))
      .def("rhs_inf_norm", &RuizTester::rhs_inf_norm, nb::arg("out"))
      .def("passes_run", &RuizTester::passes_run)
      .def("state", &RuizTester::state);
  testing.attr("RUIZ_MIN_SCALING") = RuizEquilibration::kMinScaling;
  testing.attr("RUIZ_MAX_SCALING") = RuizEquilibration::kMaxScaling;
  testing.attr("RUIZ_CONVERGENCE_TOL") = RuizEquilibration::kConvergenceTol;
}
