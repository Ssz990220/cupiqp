// nanobind entry points of module data (owned by that module).
// _testing.SparseData wraps cupiqp::SparseData for parity tests: patterns are
// host integer lists, values are device (or host, for set_values_from_host)
// addresses passed as Python ints. Streams are passed as integer handles
// (0 = legacy default stream).
#include <nanobind/nanobind.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "problem/data/sparse_data.hpp"

namespace nb = nanobind;
using namespace cupiqp;

namespace {

using Addr = uintptr_t;
using AddrMap = std::map<std::string, Addr>;

CsrPattern make_pattern(int64_t cols, const std::vector<int64_t>& indptr, const std::vector<int64_t>& indices) {
  // Keep the raw shape so validate_csr sees exactly what the caller passed;
  // only narrow (range-checked) to int32.
  CsrPattern M;
  M.rows = static_cast<int32_t>(static_cast<int64_t>(indptr.size()) - 1);
  M.cols = static_cast<int32_t>(cols);
  M.indptr.reserve(indptr.size());
  for (int64_t x : indptr) {
    if (x < INT32_MIN || x > INT32_MAX) throw Error(ErrorCode::InvalidArgument, "CSR indptr entry out of int32 range");
    M.indptr.push_back(static_cast<int32_t>(x));
  }
  M.indices.reserve(indices.size());
  for (int64_t x : indices) {
    if (x < INT32_MIN || x > INT32_MAX) throw Error(ErrorCode::InvalidArgument, "CSR column index out of int32 range");
    M.indices.push_back(static_cast<int32_t>(x));
  }
  return M;
}

ValuesView to_values(const AddrMap& a) {
  ValuesView v;
  auto get = [&](const char* k) -> const double* {
    auto it = a.find(k);
    return it == a.end() ? nullptr : reinterpret_cast<const double*>(it->second);
  };
  for (const auto& kv : a) {
    static const char* names[] = {"P", "c", "A", "b", "G", "h_l", "h_u", "x_l", "x_u"};
    bool known = false;
    for (const char* nm : names) known = known || kv.first == nm;
    if (!known) throw Error(ErrorCode::InvalidArgument, "unknown value name '" + kv.first + "'");
  }
  v.P = get("P");
  v.c = get("c");
  v.A = get("A");
  v.b = get("b");
  v.G = get("G");
  v.h_l = get("h_l");
  v.h_u = get("h_u");
  v.x_l = get("x_l");
  v.x_u = get("x_u");
  return v;
}

std::map<std::string, bool> mask_dict(const UpdateMask& u) {
  return {{"P", u.P}, {"c", u.c}, {"A", u.A}, {"b", u.b}, {"G", u.G},
          {"h_l", u.h_l}, {"h_u", u.h_u}, {"x_l", u.x_l}, {"x_u", u.x_u}};
}

cudaStream_t to_stream(Addr s) { return reinterpret_cast<cudaStream_t>(s); }

struct SparseDataTester {
  std::unique_ptr<SparseData> d = std::make_unique<SparseData>();
};

}  // namespace

void bind_data(nb::module_& m, nb::module_& testing) {
  (void)m;

  testing.def(
      "validate_csr",
      [](const std::vector<int64_t>& indptr, const std::vector<int64_t>& indices, int64_t cols,
         int32_t expected_cols, const std::string& name) {
        validate_csr(make_pattern(cols, indptr, indices), expected_cols, name);
      },
      nb::arg("indptr"), nb::arg("indices"), nb::arg("cols"), nb::arg("expected_cols"), nb::arg("name"),
      "cupiqp::validate_csr on a host pattern; cols is the pattern's column count.");

  nb::class_<SparseDataTester>(testing, "SparseData")
      .def(nb::init<>())
      .def(
          "init",
          [](SparseDataTester& t, const std::vector<int64_t>& P_indptr, const std::vector<int64_t>& P_indices,
             std::optional<std::vector<int64_t>> A_indptr, std::optional<std::vector<int64_t>> A_indices,
             std::optional<std::vector<int64_t>> G_indptr, std::optional<std::vector<int64_t>> G_indices,
             const AddrMap& values, Addr stream) {
            const int64_t n = static_cast<int64_t>(P_indptr.size()) - 1;
            CsrPattern P = make_pattern(n, P_indptr, P_indices);
            std::optional<CsrPattern> A, G;
            if (A_indptr) A = make_pattern(n, *A_indptr, A_indices.value_or(std::vector<int64_t>{}));
            if (G_indptr) G = make_pattern(n, *G_indptr, G_indices.value_or(std::vector<int64_t>{}));
            t.d->init(P, A ? &*A : nullptr, G ? &*G : nullptr, to_values(values), to_stream(stream));
          },
          nb::arg("P_indptr"), nb::arg("P_indices"), nb::arg("A_indptr").none() = nb::none(),
          nb::arg("A_indices").none() = nb::none(), nb::arg("G_indptr").none() = nb::none(),
          nb::arg("G_indices").none() = nb::none(), nb::arg("values") = AddrMap{}, nb::arg("stream") = 0)
      .def(
          "set_values",
          [](SparseDataTester& t, const AddrMap& values, Addr stream) {
            return mask_dict(t.d->set_values(to_values(values), to_stream(stream)));
          },
          nb::arg("values"), nb::arg("stream") = 0)
      .def(
          "set_values_from_host",
          [](SparseDataTester& t, const AddrMap& values, Addr stream) {
            return mask_dict(t.d->set_values_from_host(to_values(values), to_stream(stream)));
          },
          nb::arg("values"), nb::arg("stream") = 0)
      .def(
          "update_finite_masks",
          [](SparseDataTester& t, Addr stream) { t.d->update_finite_masks(to_stream(stream)); },
          nb::arg("stream") = 0)
      .def(
          "update_finite_masks_reference",
          [](SparseDataTester& t, Addr stream) { t.d->update_finite_masks_reference(to_stream(stream)); },
          nb::arg("stream") = 0)
      .def("dims",
           [](SparseDataTester& t) {
             const Dims& d = t.d->dims();
             return std::map<std::string, int64_t>{
                 {"B", d.B},         {"n", d.n},          {"p", d.p},          {"m", d.m},
                 {"num_hl", d.num_hl}, {"num_hu", d.num_hu}, {"num_xl", d.num_xl}, {"num_xu", d.num_xu},
                 {"nnz_P", d.nnz_P}, {"nnz_A", d.nnz_A},  {"nnz_G", d.nnz_G},  {"num_ineq", d.num_ineq()}};
           })
      .def("has_bounds",
           [](SparseDataTester& t) {
             return std::map<std::string, bool>{{"h_l", t.d->has_h_l()},
                                                {"h_u", t.d->has_h_u()},
                                                {"x_l", t.d->has_x_l()},
                                                {"x_u", t.d->has_x_u()}};
           })
      .def("view",
           [](SparseDataTester& t) {
             SparseDataView<double> v = t.d->view();
             auto a = [](const void* p) { return reinterpret_cast<Addr>(p); };
             return AddrMap{{"P_indptr", a(v.P.indptr)},   {"P_indices", a(v.P.indices)},
                            {"P_row_idx", a(v.P.row_idx)}, {"P", a(v.P.values)},
                            {"A_indptr", a(v.A.indptr)},   {"A_indices", a(v.A.indices)},
                            {"A_row_idx", a(v.A.row_idx)}, {"A", a(v.A.values)},
                            {"G_indptr", a(v.G.indptr)},   {"G_indices", a(v.G.indices)},
                            {"G_row_idx", a(v.G.row_idx)}, {"G", a(v.G.values)},
                            {"c", a(v.c)},                 {"b", a(v.b)},
                            {"h_l", a(v.h_l)},             {"h_u", a(v.h_u)},
                            {"x_l", a(v.x_l)},             {"x_u", a(v.x_u)},
                            {"finite_mask_all", a(v.finite_mask_all)},
                            {"active_G_row", a(v.active_G_row)},
                            {"active_x_bound", a(v.active_x_bound)},
                            {"num_finite_bounds", a(v.num_finite_bounds)}};
           });
}
