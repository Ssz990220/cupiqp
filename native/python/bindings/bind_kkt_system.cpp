// nanobind entry points of module kkt_system (owned by that module).
//
// _testing.kkt_* run one kernel launcher of linalg/kkt_system on caller device
// arrays (addresses as int, 0 = absent) on the legacy default stream, B = 1
// layout (row stride = row width). Callers synchronize before reading.
#include <nanobind/nanobind.h>

#include <cstdint>

#include "linalg/kkt_system/kkt_system_kernels.hpp"

namespace nb = nanobind;
using namespace cupiqp;

namespace {

inline double* P(uintptr_t a) { return reinterpret_cast<double*>(a); }
inline int32_t* I(uintptr_t a) { return reinterpret_cast<int32_t*>(a); }
inline DMat M(uintptr_t a, int64_t ld) { return DMat{P(a), ld}; }
inline CDMat C(uintptr_t a, int64_t ld) { return CDMat{P(a), ld}; }

Dims make_dims(int32_t B, int32_t n, int32_t p, int32_t m, bool has_h_l, bool has_h_u, bool has_x_l, bool has_x_u) {
  Dims d;
  d.B = B;
  d.n = n;
  d.p = p;
  d.m = m;
  d.num_hl = has_h_l ? m : 0;
  d.num_hu = has_h_u ? m : 0;
  d.num_xl = has_x_l ? n : 0;
  d.num_xu = has_x_u ? n : 0;
  return d;
}

}  // namespace

void bind_kkt_system(nb::module_& m, nb::module_& testing) {
  (void)m;
  cudaStream_t s = nullptr;

  testing.def("kkt_update_regularizations",
              [s](int32_t n, int32_t mm, bool hl, bool hu, bool xl, bool xu, uintptr_t s_all, uintptr_t z_all,
                  uintptr_t mask, uintptr_t xbs, uintptr_t rho_in, uintptr_t delta_in, uintptr_t m_s,
                  uintptr_t m_z_inv, uintptr_t w, uintptr_t rho_out, uintptr_t delta_out, uintptr_t x_reg,
                  uintptr_t z_reg, uintptr_t z_reg_inv) {
                Dims d = make_dims(1, n, 0, mm, hl, hu, xl, xu);
                const int64_t ni = d.num_ineq();
                kkt_kernels::update_regularizations(d, C(s_all, ni), C(z_all, ni), C(mask, ni), C(xbs, n),
                                                    P(rho_in), P(delta_in), M(m_s, ni), M(m_z_inv, ni), M(w, ni),
                                                    P(rho_out), P(delta_out), M(x_reg, n), M(z_reg, mm),
                                                    M(z_reg_inv, mm), s);
              });

  testing.def("kkt_eliminate_slacks",
              [s](int32_t num_ineq, uintptr_t rhs_z, uintptr_t rhs_s, uintptr_t m_z_inv, uintptr_t out) {
                kkt_kernels::eliminate_slacks(1, num_ineq, C(rhs_z, num_ineq), C(rhs_s, num_ineq),
                                              C(m_z_inv, num_ineq), M(out, num_ineq), s);
              });

  // w_all / rhs_z_all are the full [hl | hu | xl | xu] buffers.
  testing.def("kkt_eliminate_duals",
              [s](int32_t n, int32_t mm, bool hl, bool hu, bool xl, bool xu, uintptr_t rhs_x, uintptr_t w_all,
                  uintptr_t xbs, uintptr_t rhs_z_all, uintptr_t z_reg, uintptr_t rhs_x_out, uintptr_t rhs_z_out) {
                Dims d = make_dims(1, n, 0, mm, hl, hu, xl, xu);
                const int64_t ni = d.num_ineq();
                const double* w = P(w_all);
                const double* u = P(rhs_z_all);
                auto sub = [&](const double* b, int32_t off) { return CDMat{b ? b + off : nullptr, ni}; };
                kkt_kernels::eliminate_duals(d, C(rhs_x, n), sub(w, d.off_xu()), sub(w, d.off_xl()), C(xbs, n),
                                             sub(u, d.off_xu()), sub(u, d.off_xl()), M(rhs_x_out, n),
                                             sub(w, d.off_hu()), sub(w, d.off_hl()), sub(u, d.off_hu()),
                                             sub(u, d.off_hl()), C(z_reg, mm), M(rhs_z_out, mm), s);
              });

  // Every *_all argument is a full [hl | hu | xl | xu] buffer; rhs_s_all /
  // rhs_z_all / lhs_z_all likewise (the s / z tails of Variables).
  testing.def("kkt_recover_duals",
              [s](int32_t n, int32_t mm, bool hl, bool hu, bool xl, bool xu, uintptr_t G_dx, uintptr_t lhs_x,
                  uintptr_t w_all, uintptr_t upd_rhs_z_all, uintptr_t m_z_inv_all, uintptr_t rhs_z_all,
                  uintptr_t rhs_s_all, uintptr_t lhs_z_all, uintptr_t xbs) {
                Dims d = make_dims(1, n, 0, mm, hl, hu, xl, xu);
                const int64_t ni = d.num_ineq();
                auto sub = [&](uintptr_t a, int32_t off) { return CDMat{a ? P(a) + off : nullptr, ni}; };
                auto out = [&](uintptr_t a, int32_t off) { return DMat{a ? P(a) + off : nullptr, ni}; };
                kkt_kernels::recover_duals(
                    d, C(G_dx, mm), C(lhs_x, n), sub(w_all, d.off_hu()), sub(upd_rhs_z_all, d.off_hu()),
                    out(lhs_z_all, d.off_hu()), sub(w_all, d.off_hl()), sub(upd_rhs_z_all, d.off_hl()),
                    out(lhs_z_all, d.off_hl()), sub(w_all, d.off_xu()), sub(m_z_inv_all, d.off_xu()),
                    sub(rhs_z_all, d.off_xu()), sub(rhs_s_all, d.off_xu()), out(lhs_z_all, d.off_xu()),
                    sub(w_all, d.off_xl()), sub(m_z_inv_all, d.off_xl()), sub(rhs_z_all, d.off_xl()),
                    sub(rhs_s_all, d.off_xl()), out(lhs_z_all, d.off_xl()), C(xbs, n), s);
              });

  testing.def("kkt_recover_slacks", [s](int32_t num_ineq, uintptr_t rhs_s, uintptr_t lhs_z, uintptr_t m_s,
                                        uintptr_t m_z_inv, uintptr_t lhs_s) {
    kkt_kernels::recover_slacks(1, num_ineq, C(rhs_s, num_ineq), C(lhs_z, num_ineq), C(m_s, num_ineq),
                                C(m_z_inv, num_ineq), M(lhs_s, num_ineq), s);
  });

  testing.def("kkt_condensed_combine",
              [s](int32_t n, int32_t p, int32_t mm, uintptr_t P_x, uintptr_t x_reg, uintptr_t lhs_x, uintptr_t AT_y,
                  uintptr_t GT_z, uintptr_t A_x, uintptr_t lhs_y, uintptr_t delta, uintptr_t G_x, uintptr_t active,
                  uintptr_t z_reg, uintptr_t lhs_z, uintptr_t rhs_x, uintptr_t rhs_y, uintptr_t rhs_z, bool residual,
                  uintptr_t out_x, uintptr_t out_y, uintptr_t out_z, uintptr_t norm) {
                kkt_kernels::condensed_kkt_combine(1, n, p, mm, C(P_x, n), C(x_reg, n), C(lhs_x, n), C(AT_y, n),
                                                   C(GT_z, n), C(A_x, p), C(lhs_y, p), P(delta), C(G_x, mm),
                                                   C(active, mm), C(z_reg, mm), C(lhs_z, mm), C(rhs_x, n),
                                                   C(rhs_y, p), C(rhs_z, mm), residual, M(out_x, n), M(out_y, p),
                                                   M(out_z, mm), P(norm), s);
              });

  testing.def("kkt_mul", [s](int32_t k, uintptr_t a, uintptr_t b, uintptr_t out) {
    kkt_kernels::mul(1, k, C(a, k), C(b, k), M(out, k), s);
  });

  testing.def("kkt_refinement_begin", [s](uintptr_t ints, uintptr_t rhs_norm, uintptr_t new_error, uintptr_t error,
                                          uintptr_t step, uintptr_t loop) {
    kkt_kernels::refinement_begin(I(ints), P(rhs_norm), P(new_error), P(error), I(step), I(loop), s);
  });

  testing.def("kkt_refinement_decide",
              [s](uintptr_t floats, uintptr_t ints, uintptr_t rhs_norm, uintptr_t new_error, uintptr_t error,
                  uintptr_t step, uintptr_t accept, uintptr_t correct, uintptr_t loop) {
                kkt_kernels::refinement_decide(P(floats), I(ints), P(rhs_norm), P(new_error), P(error), I(step),
                                               I(accept), I(correct), I(loop), s);
              });

  testing.def("kkt_refinement_accept",
              [s](int32_t n, int32_t p, int32_t mm, uintptr_t accept, uintptr_t cx, uintptr_t cy, uintptr_t cz,
                  uintptr_t lx, uintptr_t ly, uintptr_t lz) {
                kkt_kernels::refinement_accept(1, n, p, mm, I(accept), C(cx, n), C(cy, p), C(cz, mm), M(lx, n),
                                               M(ly, p), M(lz, mm), s);
              });

  testing.def("kkt_refinement_rhs_norm",
              [s](int32_t n, int32_t p, int32_t mm, uintptr_t rx, uintptr_t ry, uintptr_t rz, uintptr_t norm) {
                kkt_kernels::refinement_rhs_norm(1, n, p, mm, C(rx, n), C(ry, p), C(rz, mm), P(norm), s);
              });

  testing.def("kkt_refinement_add",
              [s](int32_t n, int32_t p, int32_t mm, uintptr_t lx, uintptr_t ly, uintptr_t lz, uintptr_t cx,
                  uintptr_t cy, uintptr_t cz) {
                kkt_kernels::refinement_add(1, n, p, mm, C(lx, n), C(ly, p), C(lz, mm), M(cx, n), M(cy, p),
                                            M(cz, mm), s);
              });

  testing.def("kkt_solution_finite", [s](int32_t n, int32_t p, int32_t mm, uintptr_t x, uintptr_t y, uintptr_t z,
                                         uintptr_t status) {
    kkt_kernels::solution_finite(1, n, p, mm, C(x, n), C(y, p), C(z, mm), I(status), s);
  });

  testing.def("kkt_assemble_rhs", [s](int32_t n, int32_t p, int32_t mm, uintptr_t x, uintptr_t y, uintptr_t z,
                                      uintptr_t out) {
    kkt_kernels::assemble_rhs(1, n, p, mm, C(x, n), C(y, p), C(z, mm), M(out, int64_t(n) + p + mm), s);
  });

  testing.def("kkt_split_sol", [s](int32_t n, int32_t p, int32_t mm, uintptr_t in, uintptr_t x, uintptr_t y,
                                   uintptr_t z) {
    kkt_kernels::split_sol(1, n, p, mm, C(in, int64_t(n) + p + mm), M(x, n), M(y, p), M(z, mm), s);
  });
}
