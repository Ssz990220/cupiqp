// nanobind entry points of the solver (ipm/solver, integration owner).
// The Python shim cupiqp_native/_solver.py builds the reference-like
// SparseSolver API on top of these. Values cross as device addresses (int),
// patterns as host int32/int64 NumPy arrays.
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <cstdint>
#include <optional>

#include "cupiqp/cupiqp.hpp"

namespace nb = nanobind;
using namespace cupiqp;

namespace {

using HostIdx = nb::ndarray<nb::numpy, const int64_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

CsrPattern pattern(int64_t rows, int64_t cols, const HostIdx& indptr, const HostIdx& indices) {
  if (static_cast<int64_t>(indptr.shape(0)) != rows + 1)
    throw Error(ErrorCode::InvalidArgument, "indptr length must be rows + 1");
  if (indptr.shape(0) > 0 && static_cast<int64_t>(indices.shape(0)) != indptr.data()[rows])
    throw Error(ErrorCode::InvalidArgument, "indices length must equal indptr[-1]");
  return CsrPattern::from_int64(rows, cols, indptr.data(), indices.data());
}

ValuesView values_from(const nb::dict& d) {
  ValuesView v;
  auto get = [&](const char* k) -> const double* {
    if (!d.contains(k)) return nullptr;
    nb::object o = d[k];
    if (o.is_none()) return nullptr;
    return reinterpret_cast<const double*>(nb::cast<uintptr_t>(o));
  };
  v.P = get("P"); v.c = get("c"); v.A = get("A"); v.b = get("b"); v.G = get("G");
  v.h_l = get("h_l"); v.h_u = get("h_u"); v.x_l = get("x_l"); v.x_u = get("x_u");
  return v;
}

nb::dict info_dict(const Info& i) {
  nb::dict d;
  d["status"] = static_cast<int32_t>(i.status);
  d["iter"] = i.iter; d["iter_total"] = i.iter_total; d["factor_retries"] = i.factor_retries;
  d["no_primal_update"] = i.no_primal_update; d["no_dual_update"] = i.no_dual_update;
#define F(x) d[#x] = i.x
  F(rho); F(delta); F(mu); F(sigma); F(primal_step); F(dual_step); F(primal_res); F(primal_res_rel);
  F(dual_res); F(dual_res_rel); F(primal_res_reg); F(primal_res_reg_rel); F(dual_res_reg);
  F(dual_res_reg_rel); F(primal_prox_inf); F(dual_prox_inf); F(prev_primal_res); F(prev_dual_res);
  F(primal_obj); F(dual_obj); F(duality_gap); F(duality_gap_rel); F(reg_limit);
#undef F
  return d;
}

}  // namespace

void bind_solver(nb::module_& m, nb::module_& testing) {
  (void)testing;
  nb::enum_<GraphMode>(m, "GraphMode")
      .value("Auto", GraphMode::Auto)
      .value("None_", GraphMode::None)
      .value("IterationGraphHostLoop", GraphMode::IterationGraphHostLoop)
      .value("SolveGraphDeviceLoop", GraphMode::SolveGraphDeviceLoop);

  nb::class_<Settings> st(m, "NativeSettings");
  st.def(nb::init<>())
#define RW(x) .def_rw(#x, &Settings::x)
      RW(rho_init) RW(delta_init) RW(eps_abs) RW(eps_rel) RW(check_duality_gap)
      RW(eps_duality_gap_abs) RW(eps_duality_gap_rel) RW(infeasibility_threshold) RW(reg_lower_limit)
      RW(reg_finetune_lower_limit) RW(reg_finetune_primal_update_threshold)
      RW(reg_finetune_dual_update_threshold) RW(max_iter) RW(max_factor_retires)
      RW(preconditioner_scale_cost) RW(preconditioner_reuse_on_update) RW(preconditioner_iter) RW(tau)
      RW(iterative_refinement_always_enabled) RW(iterative_refinement_eps_abs)
      RW(iterative_refinement_eps_rel) RW(iterative_refinement_max_iter)
      RW(iterative_refinement_min_improvement_rate) RW(iterative_refinement_static_regularization_eps)
      RW(iterative_refinement_static_regularization_rel) RW(enable_cuda_graph) RW(verbose)
      RW(compute_timings) RW(graph_mode)
#undef RW
      .def("verify", &Settings::verify);

  nb::class_<SparseSolver>(m, "NativeSparseSolver")
      .def(
          "__init__",
          [](SparseSolver* self, uintptr_t stream) {
            new (self) SparseSolver(reinterpret_cast<cudaStream_t>(stream));
          },
          nb::arg("stream") = 0)
      .def_prop_ro(
          "settings", [](SparseSolver& s) -> Settings& { return s.settings(); }, nb::rv_policy::reference_internal)
      .def(
          "setup",
          [](SparseSolver& s, int64_t n, const HostIdx& P_indptr, const HostIdx& P_indices,
             std::optional<HostIdx> A_indptr, std::optional<HostIdx> A_indices, int64_t p,
             std::optional<HostIdx> G_indptr, std::optional<HostIdx> G_indices, int64_t m,
             const nb::dict& values) {
            CsrPattern P = pattern(n, n, P_indptr, P_indices);
            std::optional<CsrPattern> A, G;
            if (A_indptr) A = pattern(p, n, *A_indptr, *A_indices);
            if (G_indptr) G = pattern(m, n, *G_indptr, *G_indices);
            ValuesView v = values_from(values);
            nb::gil_scoped_release release;
            s.setup(P, v, A ? &*A : nullptr, G ? &*G : nullptr);
          },
          nb::arg("n"), nb::arg("P_indptr"), nb::arg("P_indices"), nb::arg("A_indptr").none(),
          nb::arg("A_indices").none(), nb::arg("p"), nb::arg("G_indptr").none(),
          nb::arg("G_indices").none(), nb::arg("m"), nb::arg("values"))
      .def(
          "update",
          [](SparseSolver& s, const nb::dict& values, bool from_host) {
            ValuesView v = values_from(values);
            nb::gil_scoped_release release;
            if (from_host) s.update_from_host(v);
            else s.update(v);
          },
          nb::arg("values"), nb::arg("from_host") = false)
      .def("solve", [](SparseSolver& s) {
        nb::gil_scoped_release release;
        s.solve();
      })
      .def("result_ptrs",
           [](const SparseSolver& s) {
             ResultView r = s.result();
             nb::dict d;
             auto put = [&](const char* k, const double* p, int64_t len) {
               d[k] = nb::make_tuple(reinterpret_cast<uintptr_t>(p), len);
             };
             put("x", r.x, r.n); put("y", r.y, r.p);
             put("z_l", r.z_l, r.num_hl); put("z_u", r.z_u, r.num_hu);
             put("z_bl", r.z_bl, r.num_xl); put("z_bu", r.z_bu, r.num_xu);
             put("s_l", r.s_l, r.num_hl); put("s_u", r.s_u, r.num_hu);
             put("s_bl", r.s_bl, r.num_xl); put("s_bu", r.s_bu, r.num_xu);
             return d;
           })
      .def("input_buffers",
           [](SparseSolver& s, bool host) {
             InputBuffers ib = host ? s.host_input_buffers() : s.input_buffers();
             nb::dict d;
             auto put = [&](const char* k, double* p, int64_t len) {
               d[k] = nb::make_tuple(reinterpret_cast<uintptr_t>(p), len);
             };
             put("P", ib.P, ib.len_P); put("c", ib.c, ib.len_c); put("A", ib.A, ib.len_A);
             put("b", ib.b, ib.len_b); put("G", ib.G, ib.len_G); put("h_l", ib.h_l, ib.len_h_l);
             put("h_u", ib.h_u, ib.len_h_u); put("x_l", ib.x_l, ib.len_x_l); put("x_u", ib.x_u, ib.len_x_u);
             return d;
           },
           nb::arg("host") = false)
      .def("update_staged",
           [](SparseSolver& s, const std::vector<std::string>& blocks, bool from_host) {
             UpdateBlocks ub;
             for (const auto& k : blocks) {
               if (k == "P") ub.P = true; else if (k == "c") ub.c = true; else if (k == "A") ub.A = true;
               else if (k == "b") ub.b = true; else if (k == "G") ub.G = true; else if (k == "h_l") ub.h_l = true;
               else if (k == "h_u") ub.h_u = true; else if (k == "x_l") ub.x_l = true; else if (k == "x_u") ub.x_u = true;
               else throw Error(ErrorCode::InvalidArgument, "unknown block " + k);
             }
             nb::gil_scoped_release release;
             s.update_staged(ub, from_host);
           },
           nb::arg("blocks"), nb::arg("from_host") = false)
      .def("info", [](SparseSolver& s) {
        Info i;
        {
          nb::gil_scoped_release release;
          i = s.info();
        }
        return info_dict(i);
      })
      .def("download_x",
           [](SparseSolver& s, uintptr_t host, uintptr_t event) {
             s.download_x(reinterpret_cast<double*>(host), reinterpret_cast<cudaEvent_t>(event));
           },
           nb::arg("host"), nb::arg("event") = 0)
      .def("synchronize",
           [](SparseSolver& s) {
             nb::gil_scoped_release release;
             cudaError_t e = cudaStreamSynchronize(s.stream());
             if (e != cudaSuccess) throw Error(ErrorCode::Cuda, cudaGetErrorString(e));
           })
      .def_prop_ro("stream", [](const SparseSolver& s) { return reinterpret_cast<uintptr_t>(s.stream()); })
      .def_prop_ro("n", &SparseSolver::n)
      .def_prop_ro("p", &SparseSolver::p)
      .def_prop_ro("m", &SparseSolver::m)
      .def("counters",
           [](const SparseSolver& s) {
             Counters c = s.counters();
             nb::dict d;
             d["host_syncs"] = c.host_syncs; d["graph_launches"] = c.graph_launches;
             d["graph_captures"] = c.graph_captures;
             d["updates"] = c.updates; d["solves"] = c.solves;
             return d;
           })
      .def("timings",
           [](SparseSolver& s) {
             PhaseTimings t = s.timings();
             nb::dict d;
             d["update_ms"] = t.update_ms; d["solve_ms"] = t.solve_ms;
             return d;
           })
      .def("debug_buffers", [](const SparseSolver& s) {
        std::vector<std::tuple<std::string, uintptr_t, int64_t, int64_t, std::string>> out;
        for (const auto& b : s.debug_buffers())
          out.emplace_back(b.name, reinterpret_cast<uintptr_t>(b.ptr), b.rows, b.cols, std::string(1, b.dtype));
        return out;
      });
}
