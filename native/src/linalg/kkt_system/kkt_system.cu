// M6 kkt_system: condensed KKT system (reference cupiqp/kkt_systems.py
// KKTSystem, forward direction, sparse backend, B = 1 storage with a leading
// batch axis kept in every buffer and kernel).
#include "linalg/kkt_system/kkt_system.hpp"

#include "common/check.hpp"
#include "common/graph.hpp"
#include "linalg/kkt_system/kkt_system_kernels.hpp"

namespace cupiqp {

namespace {

// Row view of a (B, k) buffer with row stride ld, starting at column `off`.
inline DMat mat(double* p, int64_t ld, int64_t off = 0) { return DMat{p ? p + off : nullptr, ld}; }
inline CDMat cmat(const double* p, int64_t ld, int64_t off = 0) { return CDMat{p ? p + off : nullptr, ld}; }

}  // namespace

void KktSystem::init(const Dims& dims, const Settings& settings, KktAssembly* kkt, KktLinearSolver* solver,
                     SparseOperators* ops, DeviceSettingsView<double> dev_settings, LoopContext* loops,
                     cudaStream_t s) {
  (void)settings;  // runtime settings are read on the device (dev_settings)
  CUPIQP_REQUIRE(kkt && solver && ops && loops, "KktSystem::init: null dependency");
  CUPIQP_REQUIRE(kkt->dim() == dims.kkt_dim(), "KktSystem::init: KKT dimension mismatch");
  dims_ = dims;
  kkt_ = kkt;
  solver_ = solver;
  ops_ = ops;
  loops_ = loops;
  dev_settings_ = dev_settings;
  use_ir_ = false;

  const size_t B = dims.B, n = dims.n, p = dims.p, m = dims.m, ni = dims.num_ineq();
  rho_.allocate(B);
  delta_.allocate(B);
  x_reg_.allocate(B * n);
  z_reg_.allocate(B * m);
  z_reg_inv_.allocate(B * m);
  rhs_x_bar_.allocate(B * n);
  rhs_z_bar_.allocate(B * m);
  work_x_.allocate(B * n);
  work_z_.allocate(B * m);
  work_Px_.allocate(B * n);
  work_GTz_.allocate(B * n);
  work_Ax_.allocate(B * p);
  work_Gx_.allocate(B * m);
  norm_buf_.allocate(1);
  m_s_all_.allocate(B * ni);
  m_z_inv_all_.allocate(B * ni);
  w_delta_inv_all_.allocate(B * ni);
  updated_rhs_z_all_.allocate(B * ni);
  factor_status_.allocate(B);
  solve_status_.allocate(B);
  ir_error_xyz_.allocate(B * (n + p + m));
  ir_delta_xyz_.allocate(B * (n + p + m));
  refine_rhs_norm_.allocate(1);
  refine_error_.allocate(1);
  refine_state_.allocate(4);
  if_flag_.allocate(1);
  if_flag_[0] = 0;
  reduce_ws_.allocate(dims.B, int64_t(n + p + m), s);

  // Reference: m_s/m_z_inv/w_delta_inv/updated_rhs_z, norm, refinement
  // buffers and statuses are wp.zeros; the rest wp.empty (zeroed here too).
  for (CudaBuffer<double>* b : {&rho_, &delta_, &x_reg_, &z_reg_, &z_reg_inv_, &rhs_x_bar_, &rhs_z_bar_, &work_x_,
                                &work_z_, &work_Px_, &work_GTz_, &work_Ax_, &work_Gx_, &norm_buf_, &m_s_all_,
                                &m_z_inv_all_, &w_delta_inv_all_, &updated_rhs_z_all_, &ir_error_xyz_,
                                &ir_delta_xyz_, &refine_rhs_norm_, &refine_error_})
    b->zero_async(s);
  factor_status_.zero_async(s);
  solve_status_.zero_async(s);
  refine_state_.zero_async(s);

  // Loop bodies of the refinement, built once (no std::function construction
  // on the hot path). They read the per-call arguments from ir_* members.
  ir_correct_ = [this](cudaStream_t st) {
    // Next candidate: ref_lhs = lhs + K^-1 ref_err.
    const int64_t n_ = dims_.n, p_ = dims_.p;
    const double* e = ir_error_xyz_.data();
    double* d = ir_delta_xyz_.data();
    kkt_solve(e, e + n_, e + n_ + p_, d, d + n_, d + n_ + p_, st);
    const int64_t W = n_ + p_ + dims_.m;
    kkt_kernels::refinement_add(dims_.B, dims_.n, dims_.p, dims_.m, cmat(ir_lhs_x_, dims_.n),
                                cmat(ir_lhs_y_, dims_.p), cmat(work_z_.data(), dims_.m), mat(d, W),
                                mat(d, W, n_), mat(d, W, n_ + p_), st);
  };
  ir_first_step_ = [this](cudaStream_t st) {
    // Step 0 evaluates the solution itself (never accepted).
    ir_evaluate_and_decide(ir_lhs_x_, ir_lhs_y_, work_z_.data(), st);
    capture_if(*loops_, st, refine_state_.data() + 2, if_flag_.data(), ir_correct_);
  };
  ir_later_step_ = [this](cudaStream_t st) {
    const int64_t n_ = dims_.n, p_ = dims_.p, W = n_ + p_ + dims_.m;
    const double* d = ir_delta_xyz_.data();
    ir_evaluate_and_decide(d, d + n_, d + n_ + p_, st);
    kkt_kernels::refinement_accept(dims_.B, dims_.n, dims_.p, dims_.m, refine_state_.data() + 1, cmat(d, W),
                                   cmat(d, W, n_), cmat(d, W, n_ + p_), mat(ir_lhs_x_, dims_.n),
                                   mat(ir_lhs_y_, dims_.p), mat(work_z_.data(), dims_.m), st);
    capture_if(*loops_, st, refine_state_.data() + 2, if_flag_.data(), ir_correct_);
  };
}

void KktSystem::update_scalings_and_factor(const SparseDataView<double>& data, const ScalingView<double>& scaling,
                                           bool iterative_refinement, const double* rho, const double* delta,
                                           const VariablesView<double>& vars, cudaStream_t s) {
  const Dims& d = dims_;
  const int64_t ni = d.num_ineq();
  // Reference _update_regularization.
  kkt_kernels::update_regularizations(
      d, cmat(vars.s_all(), vars.primal_stride()), cmat(vars.z_all(), vars.dual_stride()),
      cmat(data.finite_mask_all, ni), cmat(scaling.x_b_scaling, d.n), rho, delta, mat(m_s_all_.data(), ni),
      mat(m_z_inv_all_.data(), ni), mat(w_delta_inv_all_.data(), ni), rho_.data(), delta_.data(),
      mat(x_reg_.data(), d.n), mat(z_reg_.data(), d.m), mat(z_reg_inv_.data(), d.m), s);
  use_ir_ = iterative_refinement;
  // Reference SparseKKTSolver.update_and_factor_kkt.
  kkt_->update_diagonal(x_reg_.data(), delta_.data(), z_reg_.data(), s);
  solver_->factor(s);
}

void KktSystem::kkt_solve(const double* rhs_x, const double* rhs_y, const double* rhs_z, double* lhs_x,
                          double* lhs_y, double* lhs_z, cudaStream_t s) {
  const Dims& d = dims_;
  const int64_t W = int64_t(d.n) + d.p + d.m;
  kkt_kernels::assemble_rhs(d.B, d.n, d.p, d.m, cmat(rhs_x, d.n), cmat(rhs_y, d.p), cmat(rhs_z, d.m),
                            mat(solver_->rhs(), W), s);
  solver_->solve(s);
  kkt_kernels::split_sol(d.B, d.n, d.p, d.m, cmat(solver_->sol(), W), mat(lhs_x, d.n), mat(lhs_y, d.p),
                         mat(lhs_z, d.m), s);
  kkt_kernels::solution_finite(d.B, d.n, d.p, d.m, cmat(lhs_x, d.n), cmat(lhs_y, d.p), cmat(lhs_z, d.m),
                               solve_status_.data(), s, reduce_ws_.view());
}

void KktSystem::solve(const SparseDataView<double>& data, const ScalingView<double>& scaling,
                      const VariablesView<double>& rhs, const VariablesView<double>& lhs, cudaStream_t s) {
  prepare_rhs(scaling, rhs, s);
  // The condensed z solution goes to work_z (not needed after the recovery).
  // With B == 1 the rhs.y / lhs.x / lhs.y views are contiguous.
  kkt_solve(rhs_x_bar_.data(), rhs.y(), rhs_z_bar_.data(), lhs.x(), lhs.y(), work_z_.data(), s);
  if (use_ir_) {
    active_G_row_ = data.active_G_row;
    ir_rhs_y_ = rhs.y();
    ir_lhs_x_ = lhs.x();
    ir_lhs_y_ = lhs.y();
    iterative_refinement(s);
  }
  recover_lhs(scaling, rhs, lhs, s);
}

void KktSystem::prepare_rhs(const ScalingView<double>& scaling, const VariablesView<double>& rhs, cudaStream_t s) {
  const Dims& d = dims_;
  const int64_t ni = d.num_ineq();
  const int64_t ps = rhs.primal_stride(), ds = rhs.dual_stride();
  kkt_kernels::eliminate_slacks(d.B, d.num_ineq(), cmat(rhs.z_all(), ds), cmat(rhs.s_all(), ps),
                                cmat(m_z_inv_all_.data(), ni), mat(updated_rhs_z_all_.data(), ni), s);
  const double* w = w_delta_inv_all_.data();
  const double* u = updated_rhs_z_all_.data();
  kkt_kernels::eliminate_duals(d, cmat(rhs.x(), ps), cmat(w, ni, d.off_xu()), cmat(w, ni, d.off_xl()),
                               cmat(scaling.x_b_scaling, d.n), cmat(u, ni, d.off_xu()), cmat(u, ni, d.off_xl()),
                               mat(rhs_x_bar_.data(), d.n), cmat(w, ni, d.off_hu()), cmat(w, ni, d.off_hl()),
                               cmat(u, ni, d.off_hu()), cmat(u, ni, d.off_hl()), cmat(z_reg_.data(), d.m),
                               mat(rhs_z_bar_.data(), d.m), s);
}

void KktSystem::recover_lhs(const ScalingView<double>& scaling, const VariablesView<double>& rhs,
                            const VariablesView<double>& lhs, cudaStream_t s) {
  const Dims& d = dims_;
  if (d.m > 0) ops_->G(s, 1.0, lhs.x(), work_z_.data());
  if (d.num_ineq() == 0) return;
  const int64_t ni = d.num_ineq();
  const int64_t rps = rhs.primal_stride(), rds = rhs.dual_stride();
  const int64_t lps = lhs.primal_stride(), lds = lhs.dual_stride();
  const double* w = w_delta_inv_all_.data();
  const double* u = updated_rhs_z_all_.data();
  const double* zi = m_z_inv_all_.data();
  kkt_kernels::recover_duals(
      d, cmat(work_z_.data(), d.m), cmat(lhs.x(), lps),
      cmat(w, ni, d.off_hu()), cmat(u, ni, d.off_hu()), mat(lhs.z_u(), lds),
      cmat(w, ni, d.off_hl()), cmat(u, ni, d.off_hl()), mat(lhs.z_l(), lds),
      cmat(w, ni, d.off_xu()), cmat(zi, ni, d.off_xu()), cmat(rhs.z_bu(), rds), cmat(rhs.s_bu(), rps),
      mat(lhs.z_bu(), lds),
      cmat(w, ni, d.off_xl()), cmat(zi, ni, d.off_xl()), cmat(rhs.z_bl(), rds), cmat(rhs.s_bl(), rps),
      mat(lhs.z_bl(), lds),
      cmat(scaling.x_b_scaling, d.n), s);
  // lhs.s = (rhs.s - S lhs.z) / Z
  kkt_kernels::recover_slacks(d.B, d.num_ineq(), cmat(rhs.s_all(), rps), cmat(lhs.z_all(), lds),
                              cmat(m_s_all_.data(), ni), cmat(zi, ni), mat(lhs.s_all(), lps), s);
}

void KktSystem::condensed_kkt_apply(const double* lhs_x, const double* lhs_y, const double* lhs_z,
                                    const double* rhs_x, const double* rhs_y, const double* rhs_z, double* out_x,
                                    double* out_y, double* out_z, bool residual, cudaStream_t s) {
  const Dims& d = dims_;
  ops_->P(s, 1.0, lhs_x, work_Px_.data());
  if (d.p > 0) {
    ops_->A(s, 1.0, lhs_x, work_Ax_.data());
    ops_->AT(s, 1.0, lhs_y, work_x_.data());
  }
  if (d.m > 0) {
    // G^T (active * lhs_z): inactive rows contribute nothing. As in the
    // reference, lhs_z may alias work_z (step 0 of the refinement), in which
    // case the masking happens in place.
    kkt_kernels::mul(d.B, d.m, cmat(lhs_z, d.m), cmat(active_G_row_, d.m), mat(work_z_.data(), d.m), s);
    ops_->GT(s, 1.0, work_z_.data(), work_GTz_.data());
    ops_->G(s, 1.0, lhs_x, work_Gx_.data());
  }
  kkt_kernels::condensed_kkt_combine(
      d.B, d.n, d.p, d.m, cmat(work_Px_.data(), d.n), cmat(x_reg_.data(), d.n), cmat(lhs_x, d.n),
      cmat(work_x_.data(), d.n), cmat(work_GTz_.data(), d.n), cmat(work_Ax_.data(), d.p), cmat(lhs_y, d.p),
      delta_.data(), cmat(work_Gx_.data(), d.m), cmat(active_G_row_, d.m), cmat(z_reg_.data(), d.m),
      cmat(lhs_z, d.m), cmat(rhs_x, d.n), cmat(rhs_y, d.p), cmat(rhs_z, d.m), residual, mat(out_x, d.n),
      mat(out_y, d.p), mat(out_z, d.m), norm_buf_.data(), s);
}

void KktSystem::ir_evaluate_and_decide(const double* cand_x, const double* cand_y, const double* cand_z,
                                       cudaStream_t s) {
  // Residual of the candidate into ir_error_xyz, its norm into norm_buf
  // (zeroed by the begin kernel, then by each decision).
  const int64_t n = dims_.n, p = dims_.p;
  double* e = ir_error_xyz_.data();
  condensed_kkt_apply(cand_x, cand_y, cand_z, rhs_x_bar_.data(), ir_rhs_y_, rhs_z_bar_.data(), e, e + n,
                      e + n + p, /*residual=*/true, s);
  int32_t* st = refine_state_.data();
  kkt_kernels::refinement_decide(dev_settings_.floats, dev_settings_.ints, refine_rhs_norm_.data(),
                                 norm_buf_.data(), refine_error_.data(), st + 0, st + 1, st + 2, st + 3, s);
}

void KktSystem::iterative_refinement(cudaStream_t s) {
  const Dims& d = dims_;
  int32_t* st = refine_state_.data();
  kkt_kernels::refinement_begin(dev_settings_.ints, refine_rhs_norm_.data(), norm_buf_.data(),
                                refine_error_.data(), st + 0, st + 3, s);
  // ||rhs||_inf over the batch, for the relative tolerance.
  kkt_kernels::refinement_rhs_norm(d.B, d.n, d.p, d.m, cmat(rhs_x_bar_.data(), d.n), cmat(ir_rhs_y_, d.p),
                                   cmat(rhs_z_bar_.data(), d.m), refine_rhs_norm_.data(), s);
  capture_if(*loops_, s, st + 3, if_flag_.data(), ir_first_step_);
  capture_while(*loops_, s, st + 3, ir_later_step_);
}

}  // namespace cupiqp
