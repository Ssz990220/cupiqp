// ipm/solver: orchestration of the sparse FP64 solver (public cupiqp::SparseSolver).
//
// Mirrors the reference cupiqp/solver.py (SolverBase) and
// cupiqp/sparse/sparse_solver.py: setup order, the update() sequence,
// _run_device_solve (prelude, device-controlled IPM loop, finalization),
// _run_one_iteration, _initial_guess, _update_residuals_nr, the factor-retry
// loop and the CUDA graph modes (DESIGN.md section 4).
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include "common/buffer.hpp"
#include "common/check.hpp"
#include "common/graph.hpp"
#include "common/handles.hpp"
#include "common/reduce.cuh"
#include "common/views.hpp"
#include "cupiqp/cupiqp.hpp"
#include "ipm/kernels/ipm_kernels.hpp"
#include "linalg/kkt_assembly/kkt_assembly.hpp"
#include "linalg/kkt_system/kkt_system.hpp"
#include "linalg/reduced_kkt/reduced_kkt_solver.hpp"
#include "linalg/spmv/spmv.hpp"
#include "problem/data/sparse_data.hpp"
#include "problem/ruiz/ruiz.hpp"

namespace cupiqp {

namespace {

__global__ void fill_kernel(double* p, int64_t n, double v) {
  int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (i < n) p[i] = v;
}

void fill(cudaStream_t s, double* p, int64_t n, double v) {
  if (n <= 0) return;
  fill_kernel<<<grid_for(n), 256, 0, s>>>(p, n, v);
  CUPIQP_CUDA_CHECK_LAUNCH();
}

void zero(cudaStream_t s, void* p, size_t bytes) {
  if (bytes) CUPIQP_CUDA_CHECK(cudaMemsetAsync(p, 0, bytes, s));
}

void copy_d2d(cudaStream_t s, void* dst, const void* src, size_t bytes) {
  if (bytes) CUPIQP_CUDA_CHECK(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, s));
}

// Warm-start hook: s and z of infinite bounds are exactly 0 (as the default
// initial guess leaves them), finite ones keep the staged values.
__global__ void warm_mask_kernel(double* s_all, double* z_all, const double* finite_mask, int64_t ni) {
  int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (i >= ni) return;
  if (!(finite_mask[i] > 0.5)) {
    s_all[i] = 0.0;
    z_all[i] = 0.0;
  }
}

// Owned storage of one reference Variables object (primal and dual buffers).
struct VariablesStorage {
  CudaBuffer<double> primal, dual;
  VariablesView<double> view;
  void allocate(const Dims& d) {
    primal.allocate(static_cast<size_t>(d.B) * d.num_primal());
    dual.allocate(static_cast<size_t>(d.B) * d.num_dual());
    view.dims = d;
    view.primal = primal.data();
    view.dual = dual.data();
  }
};

constexpr int kInfoFloats = 23;
constexpr int kInfoInts = 5;

// Settings that fix coordinates, buffers or the KKT structure (reference
// _FROZEN_SETTINGS that exist natively).
struct Frozen {
  int32_t preconditioner_iter;
  bool preconditioner_scale_cost, preconditioner_reuse_on_update, iterative_refinement_always_enabled;
  bool operator==(const Frozen& o) const {
    return preconditioner_iter == o.preconditioner_iter &&
           preconditioner_scale_cost == o.preconditioner_scale_cost &&
           preconditioner_reuse_on_update == o.preconditioner_reuse_on_update &&
           iterative_refinement_always_enabled == o.iterative_refinement_always_enabled;
  }
};
Frozen frozen_of(const Settings& s) {
  return {s.preconditioner_iter, s.preconditioner_scale_cost, s.preconditioner_reuse_on_update,
          s.iterative_refinement_always_enabled};
}

}  // namespace

class SparseSolverImpl {
 public:
  explicit SparseSolverImpl(cudaStream_t user_stream)
      : stream_(user_stream ? CudaStream::borrow(user_stream) : CudaStream::create(cudaStreamDefault)),
        loops_(&counters_) {}

  ~SparseSolverImpl() {
    // Finish queued work before members (graphs, buffers) go away; the
    // stream is declared first so it is destroyed last.
    if (stream_.get()) cudaStreamSynchronize(stream_.get());
  }

  Settings settings;
  Counters counters_;

  // ------------------------------------------------------------------ setup
  void setup(const CsrPattern& P, const ValuesView& v, const CsrPattern* A, const CsrPattern* G) {
    CUPIQP_REQUIRE(!setup_done_,
                   "setup() may only be called once per solver instance; create a new solver "
                   "instance to set up a different problem.");
    CUPIQP_REQUIRE(settings.verify(), "Invalid solver settings; check the Settings field values.");
    if (P.rows > ReducedKktSolver::kMaxN)
      throw Error(ErrorCode::Unsupported, ReducedKktSolver::unsupported_n_message(P.rows));
    cudaStream_t s = stream_.get();

    data_.init(P, A, G, v, s);
    d_ = data_.dims();
    auto dv = data_.view();

    ruiz_.init(dv, s);
    if (settings.preconditioner_iter > 0)
      ruiz_.scale_data(dv, settings.preconditioner_scale_cost, settings.preconditioner_iter, s);

    allocate_state();
    allocate_inputs(v);

    ops_.init(dv);
    kkt_.init(data_.P_pattern(), data_.A_pattern(), data_.G_pattern(), dv, s);
    // Dense Schur complement on the x block; allocates everything in init.
    reduced_.init(kkt_, s);
    kkt_sys_.init(d_, settings, &kkt_, &reduced_, &ops_, dev_settings_view(), &loops_, s);

    ruiz_.compute_constraints_rhs_inf_norm_unscaled(dv, rhs_norm_.data(), s);
    sync_stream(s, &counters_);
    frozen_ = frozen_of(settings);
    setup_done_ = true;
  }

  // ------------------------------------------------------------------ update
  void update(const ValuesView& v, bool from_host) {
    require_setup("update()");
    cudaStream_t s = stream_.get();
    event_record_start();
    UpdateMask mask;
    mask.P = v.P; mask.c = v.c; mask.A = v.A; mask.b = v.b; mask.G = v.G;
    mask.h_l = v.h_l; mask.h_u = v.h_u; mask.x_l = v.x_l; mask.x_u = v.x_u;
    update_work(s, v, mask, from_host);
    event_record_update_end();
    ++counters_.updates;
  }

  // Device work of update(), the reference _update_impl sequence. Captures
  // into a CUDA graph when the inputs are device memory at fixed addresses.
  void update_work(cudaStream_t s, const ValuesView& v, const UpdateMask& mask, bool from_host) {
    auto dv = data_.view();
    fill_status(s, STATUS_UNSOLVED);  // reference: info.reset_status()
    const bool precond = settings.preconditioner_iter > 0;
    if (precond) ruiz_.unscale_data(dv, mask, s);  // replaced blocks skipped (exact)
    if (from_host) data_.set_values_from_host(v, s);
    else data_.set_values(v, s);
    const bool matrix_changed = mask.any_matrix();
    const bool ineq_bound_pattern_may_change = mask.any_ineq_bound();
    bool fresh_ruiz = false;
    if (precond) {
      const bool reuse = settings.preconditioner_reuse_on_update || !matrix_changed;
      if (reuse) {
        ruiz_.reuse_scaling(dv, s);
      } else {
        ruiz_.reset(s);
        ruiz_.scale_data(dv, settings.preconditioner_scale_cost, settings.preconditioner_iter, s);
        fresh_ruiz = true;
      }
    }
    if (mask.any_box_bound()) ruiz_.refresh_x_b_scaling(s);
    ruiz_.compute_constraints_rhs_inf_norm_unscaled(dv, rhs_norm_.data(), s);
    kkt_.scatter(dv, mask.P || fresh_ruiz, mask.A || fresh_ruiz,
                 mask.G || fresh_ruiz || ineq_bound_pattern_may_change, s);
  }

  // ------------------------------------------------------------------ staged inputs
  InputBuffers input_buffers(bool host) {
    require_setup("input_buffers()");
    InputBuffers ib;
    double** ptrs[kBlocks] = {&ib.P, &ib.c, &ib.A, &ib.b, &ib.G, &ib.h_l, &ib.h_u, &ib.x_l, &ib.x_u};
    int64_t* lens[kBlocks] = {&ib.len_P, &ib.len_c, &ib.len_A, &ib.len_b, &ib.len_G,
                              &ib.len_h_l, &ib.len_h_u, &ib.len_x_l, &ib.len_x_u};
    for (int k = 0; k < kBlocks; ++k) {
      *lens[k] = static_cast<int64_t>(in_dev_[k].size());
      *ptrs[k] = *lens[k] ? (host ? in_host_[k].data() : in_dev_[k].data()) : nullptr;
    }
    return ib;
  }

  void update_staged(const UpdateBlocks& blocks, bool from_host) {
    require_setup("update_staged()");
    cudaStream_t s = stream_.get();
    const bool want[kBlocks] = {blocks.P, blocks.c, blocks.A, blocks.b, blocks.G,
                                blocks.h_l, blocks.h_u, blocks.x_l, blocks.x_u};
    UpdateMask mask;
    bool* mk[kBlocks] = {&mask.P, &mask.c, &mask.A, &mask.b, &mask.G, &mask.h_l, &mask.h_u, &mask.x_l, &mask.x_u};
    ValuesView v;
    const double** vk[kBlocks] = {&v.P, &v.c, &v.A, &v.b, &v.G, &v.h_l, &v.h_u, &v.x_l, &v.x_u};
    uint32_t key = from_host ? 1u << kBlocks : 0u;
    for (int k = 0; k < kBlocks; ++k) {
      if (!want[k] || in_dev_[k].size() == 0) continue;  // absent blocks are ignored
      *mk[k] = true;
      *vk[k] = in_dev_[k].data();
      key |= 1u << k;
    }
    event_record_start();
    GraphExec& ge = update_graphs_[key];
    if (!ge.valid()) {
      cudaGraph_t g = capture(s, [&](cudaStream_t st) {
        if (from_host)
          for (int k = 0; k < kBlocks; ++k)
            if (*mk[k])
              CUPIQP_CUDA_CHECK(cudaMemcpyAsync(in_dev_[k].data(), in_host_[k].data(),
                                                sizeof(double) * in_dev_[k].size(), cudaMemcpyHostToDevice, st));
        update_work(st, v, mask, false);
      }, &counters_);
      ge.instantiate(g);
    }
    ge.launch(s, &counters_);
    event_record_update_end();
    ++counters_.updates;
  }

  // ------------------------------------------------------------------ solve
  void solve() { solve_impl(0u, false); }

  // Default path: warm_fields == 0 (solve_graph_, prelude_graph_, solve_prelude
  // exactly as before the hook existed). Warm path: its own graphs per key.
  void solve_impl(uint32_t warm_fields, bool warm_from_host) {
    require_setup("solve()");
    cudaStream_t s = stream_.get();
    upload_settings(s);
    GraphMode mode = effective_mode();
    if (!timing_started_) event_record_start();
    const bool warm = warm_fields != 0u;
    const uint32_t key = warm_fields | (warm_from_host ? 16u : 0u);
    warm_fields_ = warm_fields;
    warm_from_host_ = warm_from_host;

    if (mode == GraphMode::SolveGraphDeviceLoop) {
      GraphExec& g = warm ? warm_solve_graphs_[key] : solve_graph_;
      ensure_solve_graph(s, g);
      if (g.valid()) {
        g.launch(s, &counters_);
      } else {
        mode = GraphMode::IterationGraphHostLoop;  // capture failed: per-iteration graphs
      }
    }
    if (mode == GraphMode::IterationGraphHostLoop) {
      // Prelude and finalization as CUDA graphs (captured on first use), the
      // IPM loop driven from the host with one iteration graph per iteration.
      if (warm)
        launch_cached(warm_prelude_graphs_[key], s, [this](cudaStream_t st) { solve_prelude(st); });
      else
        launch_cached(prelude_graph_, s, [this](cudaStream_t st) { solve_prelude(st); });
      capture_while(loops_, s, continue_flag(), [this](cudaStream_t st) { launch_iteration_graph(st); });
      launch_cached(finalize_graph_, s, [this](cudaStream_t st) { solve_finalize(st); });
    } else if (mode == GraphMode::None) {
      run_device_solve(s, [this](cudaStream_t st) {
        run_one_iteration(st);
        if (settings.verbose) print_iteration(st);
      });
    }
    event_record_solve_end();
    ++counters_.solves;
  }

  // ------------------------------------------------------------------ warm start (experimental)
  WarmStartBuffers warm_start_buffers(bool host) {
    require_setup("warm_start_buffers()");
    ensure_warm_storage();
    WarmStartBuffers w;
    double* pr = host ? warm_host_primal_.data() : warm_in_.view.primal;
    double* du = host ? warm_host_dual_.data() : warm_in_.view.dual;
    VariablesView<double> v = warm_in_.view;
    v.primal = pr;
    v.dual = du;
    w.n = d_.n; w.p = d_.p; w.m = d_.m;
    w.num_hl = d_.num_hl; w.num_hu = d_.num_hu; w.num_xl = d_.num_xl; w.num_xu = d_.num_xu;
    w.x = v.x(); w.y = v.y();
    w.s_l = d_.num_hl ? v.s_l() : nullptr; w.s_u = d_.num_hu ? v.s_u() : nullptr;
    w.s_bl = d_.num_xl ? v.s_bl() : nullptr; w.s_bu = d_.num_xu ? v.s_bu() : nullptr;
    w.z_l = d_.num_hl ? v.z_l() : nullptr; w.z_u = d_.num_hu ? v.z_u() : nullptr;
    w.z_bl = d_.num_xl ? v.z_bl() : nullptr; w.z_bu = d_.num_xu ? v.z_bu() : nullptr;
    return w;
  }

  void solve_warm(uint32_t fields, bool from_host) {
    require_setup("solve_warm()");
    CUPIQP_REQUIRE(fields != 0u && (fields & ~uint32_t(kWarmAll)) == 0u,
                   "solve_warm(): fields must be a nonzero combination of kWarmX/Y/S/Z.");
    CUPIQP_REQUIRE(warm_allocated_, "solve_warm(): call warm_start_buffers() and write a start first.");
    solve_impl(fields, from_host);
  }

  template <class Fn>
  void launch_cached(GraphExec& ge, cudaStream_t s, Fn&& fn) {
    if (!ge.valid()) ge.instantiate(capture(s, fn, &counters_));
    ge.launch(s, &counters_);
  }

  // ------------------------------------------------------------------ accessors
  ResultView result() const {
    ResultView r;
    const auto& v = result_.view;
    r.n = d_.n; r.p = d_.p; r.m = d_.m;
    r.num_hl = d_.num_hl; r.num_hu = d_.num_hu; r.num_xl = d_.num_xl; r.num_xu = d_.num_xu;
    r.x = v.x(); r.y = v.y();
    r.s_l = d_.num_hl ? v.s_l() : nullptr; r.s_u = d_.num_hu ? v.s_u() : nullptr;
    r.s_bl = d_.num_xl ? v.s_bl() : nullptr; r.s_bu = d_.num_xu ? v.s_bu() : nullptr;
    r.z_l = d_.num_hl ? v.z_l() : nullptr; r.z_u = d_.num_hu ? v.z_u() : nullptr;
    r.z_bl = d_.num_xl ? v.z_bl() : nullptr; r.z_bu = d_.num_xu ? v.z_bu() : nullptr;
    return r;
  }

  void download_x(double* host, cudaEvent_t done) {
    require_setup("download_x()");
    cudaStream_t s = stream_.get();
    CUPIQP_CUDA_CHECK(cudaMemcpyAsync(host, result_.view.x(), sizeof(double) * d_.n,
                                      cudaMemcpyDeviceToHost, s));
    if (done) CUPIQP_CUDA_CHECK(cudaEventRecord(done, s));
  }

  Info info() {
    require_setup("info()");
    cudaStream_t s = stream_.get();
    CUPIQP_CUDA_CHECK(cudaMemcpyAsync(info_f_host_.data(), info_f_.data(),
                                      sizeof(double) * info_f_.size(), cudaMemcpyDeviceToHost, s));
    CUPIQP_CUDA_CHECK(cudaMemcpyAsync(info_i_host_.data(), info_i_.data(),
                                      sizeof(int32_t) * info_i_.size(), cudaMemcpyDeviceToHost, s));
    CUPIQP_CUDA_CHECK(cudaMemcpyAsync(info_i_host_.data() + kInfoInts * d_.B, iter_total_.data(),
                                      sizeof(int32_t), cudaMemcpyDeviceToHost, s));
    sync_stream(s, &counters_);
    const double* f = info_f_host_.data();
    const int32_t* i = info_i_host_.data();
    const int B = d_.B;
    Info out;
    double* dst[kInfoFloats] = {&out.rho, &out.delta, &out.mu, &out.sigma, &out.primal_step,
                                &out.dual_step, &out.primal_res, &out.primal_res_rel, &out.dual_res,
                                &out.dual_res_rel, &out.primal_res_reg, &out.primal_res_reg_rel,
                                &out.dual_res_reg, &out.dual_res_reg_rel, &out.primal_prox_inf,
                                &out.dual_prox_inf, &out.prev_primal_res, &out.prev_dual_res,
                                &out.primal_obj, &out.dual_obj, &out.duality_gap,
                                &out.duality_gap_rel, &out.reg_limit};
    for (int k = 0; k < kInfoFloats; ++k) *dst[k] = f[k * B];
    out.no_primal_update = i[0 * B];
    out.no_dual_update = i[1 * B];
    out.status = static_cast<Status>(i[2 * B]);
    out.iter = i[3 * B];
    out.factor_retries = i[4 * B];
    out.iter_total = i[kInfoInts * B];
    return out;
  }

  cudaStream_t stream() const { return stream_.get(); }
  const Dims& dims() const { return d_; }

  Counters counters() const {
    Counters c = counters_;
    return c;
  }

  PhaseTimings timings() {
    PhaseTimings t;
    if (!settings.compute_timings || !timing_valid_) return t;
    CUPIQP_CUDA_CHECK(cudaEventSynchronize(ev_solve_end_.get()));
    if (have_update_) CUPIQP_CUDA_CHECK(cudaEventElapsedTime(&t.update_ms, ev_start_.get(), ev_update_end_.get()));
    CUPIQP_CUDA_CHECK(cudaEventElapsedTime(&t.solve_ms, have_update_ ? ev_update_end_.get() : ev_start_.get(),
                                           ev_solve_end_.get()));
    return t;
  }

  std::vector<DebugBuffer> debug_buffers() const {
    std::vector<DebugBuffer> out;
    if (!setup_done_) return out;
    auto add = [&](const char* name, const void* p, int64_t cols, char t = 'f') {
      out.push_back({name, p, d_.B, cols, t});
    };
    const auto dv = const_cast<SparseData&>(data_).view();
    const auto sc = const_cast<RuizEquilibration&>(ruiz_).view();
    add("data.P", dv.P.values, d_.nnz_P); add("data.A", dv.A.values, d_.nnz_A);
    add("data.G", dv.G.values, d_.nnz_G); add("data.c", dv.c, d_.n); add("data.b", dv.b, d_.p);
    add("data.h_l", dv.h_l, d_.num_hl); add("data.h_u", dv.h_u, d_.num_hu);
    add("data.x_l", dv.x_l, d_.num_xl); add("data.x_u", dv.x_u, d_.num_xu);
    add("data.finite_mask_all", dv.finite_mask_all, d_.num_ineq());
    add("scaling.delta", sc.delta, d_.n + d_.p + d_.m); add("scaling.delta_b", sc.delta_b, d_.n);
    add("scaling.cost_scaling", sc.cost_scaling, 1); add("scaling.x_b_scaling", sc.x_b_scaling, d_.n);
    add("kkt.values", const_cast<KktAssembly&>(kkt_).values(), kkt_.nnz());
    add("result.primal", result_.view.primal, d_.num_primal());
    add("result.dual", result_.view.dual, d_.num_dual());
    add("prox.primal", prox_.view.primal, d_.num_primal());
    add("prox.dual", prox_.view.dual, d_.num_dual());
    add("step.primal", step_.view.primal, d_.num_primal());
    add("res.primal", res_.view.primal, d_.num_primal());
    add("res.dual", res_.view.dual, d_.num_dual());
    add("res_nr.primal", res_nr_.view.primal, d_.num_primal());
    add("res_nr.dual", res_nr_.view.dual, d_.num_dual());
    add("info.floats", info_f_.data(), kInfoFloats);
    add("info.ints", info_i_.data(), kInfoInts, 'i');
    add("constraints_rhs_inf_norm", rhs_norm_.data(), 1);
    return out;
  }

 private:
  // Declaration order = construction order; destruction runs in reverse, so
  // the stream (first) outlives everything that enqueues on it.
  CudaStream stream_;
  LoopContext loops_;
  SparseData data_;
  RuizEquilibration ruiz_;
  SparseOperators ops_;
  KktAssembly kkt_;
  ReducedKktSolver reduced_;
  KktSystem kkt_sys_;
  Dims d_;
  bool setup_done_ = false;
  Frozen frozen_{};

  VariablesStorage result_, step_, res_nr_, res_, prox_;
  CudaBuffer<double> info_f_;   // (kInfoFloats, B) structure of arrays
  CudaBuffer<int32_t> info_i_;  // (kInfoInts, B)
  CudaBuffer<int32_t> iter_total_;
  CudaPinnedBuffer<double> info_f_host_;
  CudaPinnedBuffer<int32_t> info_i_host_;
  InfoView<double> info_{};

  CudaBuffer<double> settings_f_;
  CudaBuffer<int32_t> settings_i_;
  CudaPinnedBuffer<double> settings_f_host_;
  CudaPinnedBuffer<int32_t> settings_i_host_;
  CudaEvent settings_copied_;
  bool settings_uploaded_ = false;
  std::array<double, SF_COUNT> uploaded_f_{};
  std::array<int32_t, SI_COUNT> uploaded_i_{};

  CudaBuffer<bool> active_, finetune_, dual_improved_, primal_improved_;
  CudaBuffer<int32_t> loop_state_;  // [current_iter, continue_flag]
  CudaBuffer<int32_t> needs_retry_, any_retry_;
  CudaBuffer<double> work_z_1_, work_z_2_, work_x_, rhs_norm_;
  ReductionWorkspaceBuffers reduce_ws_;  // multi-block reductions of the IPM kernels
  ReductionWorkspace rws_{};             // its view

  GraphExec iteration_graph_;
  GraphExec prelude_graph_, finalize_graph_;
  GraphExec solve_graph_;
  std::map<uint32_t, GraphExec> update_graphs_;  // keyed by replaced blocks (+ from_host bit)
  static constexpr int kBlocks = 9;              // P, c, A, b, G, h_l, h_u, x_l, x_u
  CudaBuffer<double> in_dev_[kBlocks];
  CudaPinnedBuffer<double> in_host_[kBlocks];
  bool solve_graph_failed_ = false;

  // Experimental warm-start hook (allocated on first warm_start_buffers()).
  VariablesStorage warm_in_, warm_scaled_;
  CudaPinnedBuffer<double> warm_host_primal_, warm_host_dual_;
  bool warm_allocated_ = false;
  uint32_t warm_fields_ = 0u;  // read at capture time by solve_prelude()
  bool warm_from_host_ = false;
  std::map<uint32_t, GraphExec> warm_solve_graphs_, warm_prelude_graphs_;

  CudaEvent ev_start_, ev_update_end_, ev_solve_end_;
  bool timing_started_ = false, have_update_ = false, timing_valid_ = false;

  int32_t* current_iter() { return loop_state_.data(); }
  int32_t* continue_flag() { return loop_state_.data() + 1; }

  DeviceSettingsView<double> dev_settings_view() const {
    return {settings_f_.data(), settings_i_.data()};
  }

  void require_setup(const char* where) {
    CUPIQP_REQUIRE(setup_done_, "Solver not setup yet. Call setup() first.");
    if (!(frozen_of(settings) == frozen_))
      throw Error(ErrorCode::InvalidArgument,
                  std::string("structural settings changed after setup(); they fix the data "
                              "coordinates, the buffers or the KKT structure and cannot change before ") +
                      where + ". Create a new solver to change them.");
  }

  void allocate_state() {
    const int B = d_.B;
    for (VariablesStorage* v : {&result_, &step_, &res_nr_, &res_, &prox_}) v->allocate(d_);
    info_f_.allocate(static_cast<size_t>(kInfoFloats) * B);
    info_i_.allocate(static_cast<size_t>(kInfoInts) * B);
    iter_total_.allocate(1);
    info_f_host_.allocate(info_f_.size());
    info_i_host_.allocate(info_i_.size() + 1);
    double* f = info_f_.data();
    double** fields[kInfoFloats] = {&info_.rho, &info_.delta, &info_.mu, &info_.sigma,
                                    &info_.primal_step, &info_.dual_step, &info_.primal_res,
                                    &info_.primal_res_rel, &info_.dual_res, &info_.dual_res_rel,
                                    &info_.primal_res_reg, &info_.primal_res_reg_rel,
                                    &info_.dual_res_reg, &info_.dual_res_reg_rel,
                                    &info_.primal_prox_inf, &info_.dual_prox_inf,
                                    &info_.prev_primal_res, &info_.prev_dual_res, &info_.primal_obj,
                                    &info_.dual_obj, &info_.duality_gap, &info_.duality_gap_rel,
                                    &info_.reg_limit};
    for (int k = 0; k < kInfoFloats; ++k) *fields[k] = f + static_cast<size_t>(k) * B;
    int32_t* i = info_i_.data();
    info_.no_primal_update = i;
    info_.no_dual_update = i + B;
    info_.status = i + 2 * B;
    info_.iter = i + 3 * B;
    info_.factor_retries = i + 4 * B;
    info_.iter_total = iter_total_.data();

    settings_f_.allocate(SF_COUNT);
    settings_i_.allocate(SI_COUNT);
    settings_f_host_.allocate(SF_COUNT);
    settings_i_host_.allocate(SI_COUNT);
    settings_copied_ = CudaEvent::create(false);

    active_.allocate(B); finetune_.allocate(B); dual_improved_.allocate(B); primal_improved_.allocate(B);
    loop_state_.allocate(2);
    needs_retry_.allocate(B);
    any_retry_.allocate(1);
    work_z_1_.allocate(static_cast<size_t>(B) * d_.m);
    work_z_2_.allocate(static_cast<size_t>(B) * d_.m);
    work_x_.allocate(static_cast<size_t>(B) * d_.n);
    rhs_norm_.allocate(B);
    // Widest reduced row: num_ineq (step, mu, centering) or the concatenated
    // rows of residual_nr (n + p + num_ineq) and residuals_r (n + num_dual).
    const int64_t red_width = int64_t(d_.n) + d_.p + std::max(d_.num_ineq(), d_.m);
    reduce_ws_.allocate(B, red_width, stream_.get());
    rws_ = reduce_ws_.view();

    cudaStream_t s = stream_.get();
    for (VariablesStorage* v : {&result_, &step_, &res_nr_, &res_, &prox_}) {
      v->primal.zero_async(s);
      v->dual.zero_async(s);
    }
    info_f_.zero_async(s); info_i_.zero_async(s); iter_total_.zero_async(s);
    active_.zero_async(s); finetune_.zero_async(s); dual_improved_.zero_async(s);
    primal_improved_.zero_async(s); loop_state_.zero_async(s); needs_retry_.zero_async(s);
    any_retry_.zero_async(s); work_z_1_.zero_async(s); work_z_2_.zero_async(s); work_x_.zero_async(s);
    fill_status(s, STATUS_UNSOLVED);
    ev_start_ = CudaEvent::create(true);
    ev_update_end_ = CudaEvent::create(true);
    ev_solve_end_ = CudaEvent::create(true);
  }

  // Solver-owned input buffers (device + pinned host), initialized with the
  // setup() values so that a partial update_staged() keeps the other blocks.
  void allocate_inputs(const ValuesView& v) {
    cudaStream_t s = stream_.get();
    const int64_t lens[kBlocks] = {d_.nnz_P, d_.n, d_.nnz_A, d_.p, d_.nnz_G,
                                   d_.num_hl, d_.num_hu, d_.num_xl, d_.num_xu};
    const double* src[kBlocks] = {v.P, v.c, v.A, v.b, v.G, v.h_l, v.h_u, v.x_l, v.x_u};
    for (int k = 0; k < kBlocks; ++k) {
      in_dev_[k].allocate(static_cast<size_t>(lens[k]));
      in_host_[k].allocate(static_cast<size_t>(lens[k]));
      if (lens[k] && src[k]) {
        in_dev_[k].copy_from_device(src[k], static_cast<size_t>(lens[k]), s);
        CUPIQP_CUDA_CHECK(cudaMemcpyAsync(in_host_[k].data(), src[k], sizeof(double) * lens[k],
                                          cudaMemcpyDeviceToHost, s));
      }
    }
  }

  void fill_status(cudaStream_t s, int32_t status) { set_int_kernel_launch(s, info_.status, d_.B, status); }

  static void set_int_kernel_launch(cudaStream_t s, int32_t* p, int n, int32_t v);

  // Reference DeviceSettings.upload: copy only when the values changed; the
  // host waits for the previous copy before rewriting the pinned staging.
  void upload_settings(cudaStream_t s) {
    std::array<double, SF_COUNT> f{};
    std::array<int32_t, SI_COUNT> i{};
    f[SF_RHO_INIT] = settings.rho_init;
    f[SF_DELTA_INIT] = settings.delta_init;
    f[SF_EPS_ABS] = settings.eps_abs;
    f[SF_EPS_REL] = settings.eps_rel;
    f[SF_EPS_DUALITY_GAP_ABS] = settings.eps_duality_gap_abs;
    f[SF_EPS_DUALITY_GAP_REL] = settings.eps_duality_gap_rel;
    f[SF_INFEASIBILITY_THRESHOLD] = settings.infeasibility_threshold;
    f[SF_REG_LOWER_LIMIT] = settings.reg_lower_limit;
    f[SF_REG_FINETUNE_LOWER_LIMIT] = settings.reg_finetune_lower_limit;
    f[SF_TAU] = settings.tau;
    f[SF_IR_EPS_ABS] = settings.iterative_refinement_eps_abs;
    f[SF_IR_EPS_REL] = settings.iterative_refinement_eps_rel;
    f[SF_IR_MIN_IMPROVEMENT_RATE] = settings.iterative_refinement_min_improvement_rate;
    i[SI_CHECK_DUALITY_GAP] = settings.check_duality_gap ? 1 : 0;
    i[SI_REG_FINETUNE_PRIMAL_UPDATE_THRESHOLD] = settings.reg_finetune_primal_update_threshold;
    i[SI_REG_FINETUNE_DUAL_UPDATE_THRESHOLD] = settings.reg_finetune_dual_update_threshold;
    i[SI_MAX_ITER] = settings.max_iter;
    i[SI_MAX_FACTOR_RETIRES] = settings.max_factor_retires;
    i[SI_IR_MAX_ITER] = settings.iterative_refinement_max_iter;
    if (settings_uploaded_ && f == uploaded_f_ && i == uploaded_i_) return;
    CUPIQP_REQUIRE(settings.verify(), "Invalid solver settings; check the Settings field values.");
    if (settings_uploaded_) {
      CUPIQP_CUDA_CHECK(cudaEventSynchronize(settings_copied_.get()));
      ++counters_.host_syncs;
    }
    std::memcpy(settings_f_host_.data(), f.data(), sizeof(double) * SF_COUNT);
    std::memcpy(settings_i_host_.data(), i.data(), sizeof(int32_t) * SI_COUNT);
    CUPIQP_CUDA_CHECK(cudaMemcpyAsync(settings_f_.data(), settings_f_host_.data(),
                                      sizeof(double) * SF_COUNT, cudaMemcpyHostToDevice, s));
    CUPIQP_CUDA_CHECK(cudaMemcpyAsync(settings_i_.data(), settings_i_host_.data(),
                                      sizeof(int32_t) * SI_COUNT, cudaMemcpyHostToDevice, s));
    settings_copied_.record(s);
    uploaded_f_ = f;
    uploaded_i_ = i;
    settings_uploaded_ = true;
  }

  GraphMode effective_mode() const {
    // Reference _cuda_graph_mode: verbose or graphs disabled -> no graph;
    // iterative refinement always on -> no graph (kept from the reference,
    // whose sparse backend cannot capture its solve inside the refinement's
    // conditional node).
    if (settings.verbose) return GraphMode::None;
    GraphMode m = settings.graph_mode;
    if (m == GraphMode::Auto)  // Auto: the whole solve in one CUDA graph
      m = settings.enable_cuda_graph ? GraphMode::SolveGraphDeviceLoop : GraphMode::None;
    if (m != GraphMode::None && settings.iterative_refinement_always_enabled) return GraphMode::None;
    if (m == GraphMode::SolveGraphDeviceLoop && solve_graph_failed_) return GraphMode::IterationGraphHostLoop;
    return m;
  }

  // ------------------------------------------------------------------ graphs
  void launch_iteration_graph(cudaStream_t s) {
    if (!iteration_graph_.valid()) {
      cudaGraph_t g = capture(s, [this](cudaStream_t st) { run_one_iteration(st); }, &counters_);
      iteration_graph_.instantiate(g);
    }
    iteration_graph_.launch(s, &counters_);
  }

  // Whole-solve graph: prelude, device-controlled IPM loop (conditional
  // WHILE node) and finalization in one capture. Every operation of the solve
  // is capturable (custom kernels only, no allocation after setup). If the
  // capture or instantiation fails anyway, the solver falls back to the
  // per-iteration graph for its lifetime.
  void ensure_solve_graph(cudaStream_t s, GraphExec& ge) {
    if (ge.valid() || solve_graph_failed_) return;
    try {
      cudaGraph_t g = capture(s, [this](cudaStream_t st) {
        run_device_solve(st, [this](cudaStream_t b) { run_one_iteration(b); });
      }, &counters_);
      ge.instantiate(g);
    } catch (const Error&) {
      ge.reset();
      solve_graph_failed_ = true;
    }
  }

  // ------------------------------------------------------------------ device solve
  template <class RunIteration>
  void run_device_solve(cudaStream_t s, RunIteration&& run_iteration) {
    solve_prelude(s);
    capture_while(loops_, s, continue_flag(), run_iteration);
    solve_finalize(s);
  }

  // Reset, initial guess, its residuals and the first termination check.
  void solve_prelude(cudaStream_t s) {
    reset_solve_state(s);
    zero(s, loop_state_.data(), 2 * sizeof(int32_t));
    if (warm_fields_ == 0u) initial_guess(s);
    else warm_initial_guess(s);
    update_residuals_nr(s);
    copy_d2d(s, info_.prev_primal_res, info_.primal_res, sizeof(double) * d_.B);
    copy_d2d(s, info_.prev_dual_res, info_.dual_res, sizeof(double) * d_.B);
    update_residuals_r(s);
    update_termination(s);
  }

  // Everything still active ran out of iterations; unscale the solution.
  void solve_finalize(cudaStream_t s) {
    launch_terminate_active(s, d_.B, active_.data(), current_iter(), STATUS_MAX_ITER_REACHED,
                            info_.status, info_.iter);
    copy_d2d(s, info_.iter_total, current_iter(), sizeof(int32_t));
    if (settings.preconditioner_iter > 0) ruiz_.unscale_solution(result_.view, s);
  }

  void reset_solve_state(cudaStream_t s) {
    launch_reset_solve_state<double>(
        s, d_.B, reset_solve_state_args<double>(settings_f_.data(), info_, active_.data()));
  }

  void run_one_iteration(cudaStream_t s) {
    const int B = d_.B, ni = d_.num_ineq();
    auto dv = data_.view();
    if (ni > 0) {
      PrepareFactorAndPredictorStepArgs<double> a;
      a.active_mask = active_.data();
      a.finetune_mask = finetune_.data();
      a.settings = settings_f_.data();
      a.finite_mask_all = rows(dv.finite_mask_all, ni);
      a.num_finite_bounds = dv.num_finite_bounds;
      a.s_all = rows_s_all(result_.view);
      a.z_all = rows_z_all(result_.view);
      a.reg_limit = info_.reg_limit;
      a.no_primal_update = info_.no_primal_update;
      a.no_dual_update = info_.no_dual_update;
      a.mu = info_.mu;
      a.res_s_all = rows_s_all(res_.view);
      launch_prepare_factor_and_predictor_step<double>(s, B, ni, a, rws_);
    } else {
      launch_apply_finetune<double>(s, B, finetune_.data(), info_.reg_limit, info_.no_primal_update,
                                    info_.no_dual_update, settings_f_.data());
    }

    update_and_factorize_kkt(s);

    if (ni == 0) {
      run_full_newton_step(s);
      update_residuals_nr(s);
      update_rho_delta_without_ineq(s);
    } else {
      run_predictor_corrector(s);
      update_residuals_nr(s);
      update_rho_delta_with_ineq(s);
    }
    update_residuals_r(s);
    launch_advance_iteration(s, current_iter());
    update_termination(s);
  }

  void update_termination(cudaStream_t s) {
    launch_update_termination<double>(
        s, d_.B,
        update_termination_args<double>(info_, current_iter(), dev_settings_view(), active_.data(),
                                        finetune_.data()));
    launch_update_continue_flag(s, d_.B, active_.data(), current_iter(), settings_i_.data(),
                                continue_flag());
  }

  void update_and_factorize_kkt(cudaStream_t s) {
    factor_once(s);
    capture_while(loops_, s, any_retry_.data(), [this](cudaStream_t st) { factor_once(st); });
  }

  void factor_once(cudaStream_t s) {
    kkt_sys_.update_scalings_and_factor(data_.view(), ruiz_.view(),
                                        settings.iterative_refinement_always_enabled, info_.rho,
                                        info_.delta, result_.view, s);
    FactorRetryArgs<double> a;
    a.factor_status = kkt_sys_.factor_status();
    a.current_iter = current_iter();
    a.settings_f = settings_f_.data();
    a.settings_i = settings_i_.data();
    a.active = active_.data();
    a.status = info_.status;
    a.iteration = info_.iter;
    a.retries = info_.factor_retries;
    a.rho = info_.rho;
    a.delta = info_.delta;
    a.reg_limit = info_.reg_limit;
    a.needs_retry = needs_retry_.data();
    launch_factor_retry<double>(s, d_.B, a);
    launch_any_flag(s, d_.B, needs_retry_.data(), any_retry_.data());
  }

  void kkt_solve(cudaStream_t s, const VariablesView<double>& rhs, const VariablesView<double>& lhs) {
    kkt_sys_.solve(data_.view(), ruiz_.view(), rhs, lhs, s);
  }

  void mark_solve_failures(cudaStream_t s) {
    launch_mark_failed(s, d_.B, kkt_sys_.solve_status(), current_iter(), active_.data(), info_.status,
                       info_.iter);
  }

  void initial_guess(cudaStream_t s) {
    const int B = d_.B, ni = d_.num_ineq();
    auto& r = result_.view;
    // eq(12) of Schwan et al. 2023: x = 0, y = 0, s = z = 1
    zero(s, r.primal, sizeof(double) * B * d_.n);  // x (B == 1: contiguous)
    zero(s, r.dual, sizeof(double) * B * d_.p);    // y
    fill(s, r.s_all(), static_cast<int64_t>(B) * ni, 1.0);
    fill(s, r.z_all(), static_cast<int64_t>(B) * ni, 1.0);

    update_and_factorize_kkt(s);

    auto dv = data_.view();
    InitGuessRhsArgs<double> a;
    a.c = rows(dv.c, d_.n);
    a.b_eq = rows(dv.b, d_.p);
    a.h_l = rows(dv.h_l, d_.m);
    a.h_u = rows(dv.h_u, d_.m);
    a.x_l = rows(dv.x_l, d_.n);
    a.x_u = rows(dv.x_u, d_.n);
    a.finite_mask_hl = rows(dv.finite_mask_hl(), ni);
    a.finite_mask_hu = rows(dv.finite_mask_hu(), ni);
    a.finite_mask_xl = rows(dv.finite_mask_xl(), ni);
    a.finite_mask_xu = rows(dv.finite_mask_xu(), ni);
    a.res_x = rows_x(res_.view);
    a.res_y = rows_y(res_.view);
    a.res_zl = rows_z_l(res_.view);
    a.res_zu = rows_z_u(res_.view);
    a.res_zbl = rows_z_bl(res_.view);
    a.res_zbu = rows_z_bu(res_.view);
    launch_init_guess_rhs<double>(s, d_, a);
    zero(s, res_.view.s_all(), sizeof(double) * B * ni);

    kkt_solve(s, res_.view, result_.view);

    if (ni > 0) {
      launch_init_guess_center<double>(s, B, ni, rows(dv.finite_mask_all, ni), dv.num_finite_bounds,
                                       rows_s_all(r), rows_z_all(r), info_.mu, rws_);
      launch_calculate_mu<double>(s, B, ni, rows_s_all(r), rows_z_all(r), rows(dv.finite_mask_all, ni),
                                  dv.num_finite_bounds, info_.mu, rws_);
    }
    copy_d2d(s, prox_.view.primal, r.primal, sizeof(double) * B * d_.num_primal());
    copy_d2d(s, prox_.view.dual, r.dual, sizeof(double) * B * d_.num_dual());
  }

  // Experimental hook: install the staged start (user coordinates). Parts
  // not selected in warm_fields_ come from the default initial guess.
  void warm_initial_guess(cudaStream_t s) {
    const int B = d_.B, ni = d_.num_ineq();
    const size_t np = sizeof(double) * B * d_.num_primal(), nd = sizeof(double) * B * d_.num_dual();
    auto& r = result_.view;
    auto& w = warm_scaled_.view;
    if (warm_from_host_) {
      CUPIQP_CUDA_CHECK(cudaMemcpyAsync(warm_in_.view.primal, warm_host_primal_.data(), np,
                                        cudaMemcpyHostToDevice, s));
      CUPIQP_CUDA_CHECK(cudaMemcpyAsync(warm_in_.view.dual, warm_host_dual_.data(), nd,
                                        cudaMemcpyHostToDevice, s));
    }
    copy_d2d(s, w.primal, warm_in_.view.primal, np);
    copy_d2d(s, w.dual, warm_in_.view.dual, nd);
    if (settings.preconditioner_iter > 0) ruiz_.scale_solution(w, s);
    if (warm_fields_ != kWarmAll) initial_guess(s);  // fills the parts not selected
    if (warm_fields_ & kWarmX) copy_d2d(s, r.x(), w.x(), sizeof(double) * B * d_.n);
    if (warm_fields_ & kWarmY) copy_d2d(s, r.y(), w.y(), sizeof(double) * B * d_.p);
    if (warm_fields_ & kWarmS) copy_d2d(s, r.s_all(), w.s_all(), sizeof(double) * B * ni);
    if (warm_fields_ & kWarmZ) copy_d2d(s, r.z_all(), w.z_all(), sizeof(double) * B * ni);
    if (ni > 0) {
      auto dv = data_.view();
      warm_mask_kernel<<<grid_for(int64_t(B) * ni), 256, 0, s>>>(r.s_all(), r.z_all(), dv.finite_mask_all,
                                                                 int64_t(B) * ni);
      CUPIQP_CUDA_CHECK_LAUNCH();
      launch_calculate_mu<double>(s, B, ni, rows_s_all(r), rows_z_all(r), rows(dv.finite_mask_all, ni),
                                  dv.num_finite_bounds, info_.mu, rws_);
    }
    copy_d2d(s, prox_.view.primal, r.primal, np);
    copy_d2d(s, prox_.view.dual, r.dual, nd);
  }

  void ensure_warm_storage() {
    if (warm_allocated_) return;
    cudaStream_t s = stream_.get();
    warm_in_.allocate(d_);
    warm_scaled_.allocate(d_);
    warm_host_primal_.allocate(static_cast<size_t>(d_.B) * d_.num_primal());
    warm_host_dual_.allocate(static_cast<size_t>(d_.B) * d_.num_dual());
    warm_in_.primal.zero_async(s); warm_in_.dual.zero_async(s);
    warm_scaled_.primal.zero_async(s); warm_scaled_.dual.zero_async(s);
    sync_stream(s, &counters_);
    std::memset(warm_host_primal_.data(), 0, sizeof(double) * warm_host_primal_.size());
    std::memset(warm_host_dual_.data(), 0, sizeof(double) * warm_host_dual_.size());
    warm_allocated_ = true;
  }

  void run_full_newton_step(cudaStream_t s) {
    kkt_solve(s, res_.view, step_.view);
    mark_solve_failures(s);
    launch_run_full_newton_step<double>(s, d_.B, d_.n, d_.p, active_.data(), rows_x(step_.view),
                                        rows_y(step_.view), rows_x(result_.view), rows_y(result_.view),
                                        info_.primal_step, info_.dual_step);
  }

  void run_predictor_corrector(cudaStream_t s) {
    const int B = d_.B, ni = d_.num_ineq();
    auto dv = data_.view();
    // predictor: res.s = -s * z was written by prepare_factor_and_predictor_step
    kkt_solve(s, res_.view, step_.view);
    PrepareCorrectorStepArgs<double> c;
    c.s_all = rows_s_all(result_.view);
    c.z_all = rows_z_all(result_.view);
    c.step_s_all = rows_s_all(step_.view);
    c.step_z_all = rows_z_all(step_.view);
    c.finite_mask_all = rows(dv.finite_mask_all, ni);
    c.num_finite_bounds = dv.num_finite_bounds;
    c.mu = info_.mu;
    c.settings = settings_f_.data();
    c.primal_step = info_.primal_step;
    c.dual_step = info_.dual_step;
    c.sigma = info_.sigma;
    c.res_s_all = rows_s_all(res_.view);
    launch_prepare_corrector_step<double>(s, B, ni, c, rws_);
    kkt_solve(s, res_.view, step_.view);
    mark_solve_failures(s);
    launch_calculate_step<double>(s, B, ni, rows_s_all(result_.view), rows_z_all(result_.view),
                                  rows(dv.finite_mask_all, ni), rows_s_all(step_.view),
                                  rows_z_all(step_.view), settings_f_.data(), info_.primal_step,
                                  info_.dual_step, rws_);
    launch_update_vars_after_corrector_step<double>(
        s, B, d_.n, d_.p, ni, active_.data(), rows(dv.finite_mask_all, ni), info_.primal_step,
        info_.dual_step, rows_primals_all(step_.view), rows_duals_all(step_.view),
        rows_primals_all(result_.view), rows_duals_all(result_.view));
    launch_calculate_mu<double>(s, B, ni, rows_s_all(result_.view), rows_z_all(result_.view),
                                rows(dv.finite_mask_all, ni), dv.num_finite_bounds, info_.mu, rws_);
  }

  void update_residuals_nr(cudaStream_t s) {
    const int B = d_.B;
    auto dv = data_.view();
    auto sc = ruiz_.view();
    auto& r = result_.view;
    ops_.P(s, -1.0, r.x(), res_nr_.view.x());
    if (d_.p > 0) {
      ops_.A(s, 1.0, r.x(), res_.view.y());
      ops_.AT(s, 1.0, r.y(), res_.view.x());
    } else {
      zero(s, res_.view.x(), sizeof(double) * B * d_.n);  // res.y has length 0
    }
    launch_prepare_zu_minus_zl_and_zbu_minus_zbl<double>(
        s, d_, rows_z_u(r), rows_z_l(r), rows_z_bl(r), rows_z_bu(r), rows(sc.x_b_scaling, d_.n),
        rows(work_z_1_.data(), d_.m), rows(work_x_.data(), d_.n));
    double* G_x = work_z_2_.data();
    double* GT_zu_minus_zl = step_.view.x();
    if (d_.m > 0) {
      ops_.G(s, 1.0, r.x(), G_x);
      ops_.GT(s, 1.0, work_z_1_.data(), GT_zu_minus_zl);
    } else {
      zero(s, GT_zu_minus_zl, sizeof(double) * B * d_.n);
    }
    const int ni = d_.num_ineq();
    UpdateResidualNrArgs<double> a;
    a.minus_Px = rows_x(res_nr_.view);
    a.A_x = rows_y(res_.view);
    a.AT_y = rows_x(res_.view);
    a.G_x = rows(G_x, d_.m);
    a.GT_zh_assembled = rows_x(step_.view);  // G^T (z_u - z_l) lives in step.x
    a.zb_assembled = rows(work_x_.data(), d_.n);
    a.data_c = rows(dv.c, d_.n);
    a.data_b = rows(dv.b, d_.p);
    a.data_h_l = rows(dv.h_l, d_.m);
    a.data_h_u = rows(dv.h_u, d_.m);
    a.data_x_l = rows(dv.x_l, d_.n);
    a.data_x_u = rows(dv.x_u, d_.n);
    a.finite_mask_hl = rows(dv.finite_mask_hl(), ni);
    a.finite_mask_hu = rows(dv.finite_mask_hu(), ni);
    a.finite_mask_xl = rows(dv.finite_mask_xl(), ni);
    a.finite_mask_xu = rows(dv.finite_mask_xu(), ni);
    a.result_x = rows_x(r);
    a.result_y = rows_y(r);
    a.result_z_hl = rows_z_l(r); a.result_z_hu = rows_z_u(r);
    a.result_z_xl = rows_z_bl(r); a.result_z_xu = rows_z_bu(r);
    a.result_s_hl = rows_s_l(r); a.result_s_hu = rows_s_u(r);
    a.result_s_xl = rows_s_bl(r); a.result_s_xu = rows_s_bu(r);
    a.x_b_scaling = rows(sc.x_b_scaling, d_.n);
    a.cost_scaling_inv = sc.cost_scaling_inv;
    a.delta_inv = rows(sc.delta_inv, d_.n + d_.p + d_.m);
    a.delta_b_inv = rows(sc.delta_b_inv, d_.n);
    a.constraints_rhs_inf_norm = rhs_norm_.data();
    a.res_nr_x = rows_x(res_nr_.view);
    a.res_nr_y = rows_y(res_nr_.view);
    a.res_nr_z_hl = rows_z_l(res_nr_.view); a.res_nr_z_hu = rows_z_u(res_nr_.view);
    a.res_nr_z_xl = rows_z_bl(res_nr_.view); a.res_nr_z_xu = rows_z_bu(res_nr_.view);
    a.info_primal_obj = info_.primal_obj;
    a.info_dual_obj = info_.dual_obj;
    a.info_duality_gap = info_.duality_gap;
    a.info_duality_gap_rel = info_.duality_gap_rel;
    a.info_primal_res = info_.primal_res;
    a.info_primal_res_rel = info_.primal_res_rel;
    a.info_dual_res = info_.dual_res;
    a.info_dual_res_rel = info_.dual_res_rel;
    a.info_prev_primal_res = info_.prev_primal_res;
    a.info_prev_dual_res = info_.prev_dual_res;
    launch_update_residual_nr<double>(s, d_, a, rws_);
  }

  void update_residuals_r(cudaStream_t s) {
    auto sc = ruiz_.view();
    UpdateResidualsRArgs<double> a;
    a.rho = info_.rho;
    a.delta = info_.delta;
    a.res_nr_x = rows_x(res_nr_.view);
    a.res_nr_duals = rows_duals_all(res_nr_.view);
    a.result_x = rows_x(result_.view);
    a.result_duals = rows_duals_all(result_.view);
    a.prox_x = rows_x(prox_.view);
    a.prox_duals = rows_duals_all(prox_.view);
    a.res_x = rows_x(res_.view);
    a.res_duals = rows_duals_all(res_.view);
    a.dual_res_unscale_factor = rows(sc.dual_res_unscale_factor, d_.n);
    a.primal_res_unscale_factor = rows(sc.primal_res_unscale_factor, d_.num_dual());
    a.primal_res = info_.primal_res;
    a.primal_res_rel = info_.primal_res_rel;
    a.dual_res = info_.dual_res;
    a.dual_res_rel = info_.dual_res_rel;
    a.primal_res_reg = info_.primal_res_reg;
    a.primal_res_reg_rel = info_.primal_res_reg_rel;
    a.dual_res_reg = info_.dual_res_reg;
    a.dual_res_reg_rel = info_.dual_res_reg_rel;
    a.primal_prox_inf = info_.primal_prox_inf;
    a.dual_prox_inf = info_.dual_prox_inf;
    launch_update_residuals_r<double>(s, d_.B, d_.n, d_.num_dual(), a, rws_);
  }

  void update_rho_delta_with_ineq(cudaStream_t s) {
    launch_update_rho_delta_with_ineq<double>(s, d_.B, update_rho_delta_args<double>(active_.data(), info_),
                                              dual_improved_.data(), primal_improved_.data(),
                                              settings_f_.data(), current_iter());
    launch_update_prox_vars<double>(s, d_.B, d_.n, d_.num_dual(), active_.data(), dual_improved_.data(),
                                    primal_improved_.data(), rows_x(result_.view), rows_x(prox_.view),
                                    rows_duals_all(result_.view), rows_duals_all(prox_.view));
  }

  void update_rho_delta_without_ineq(cudaStream_t s) {
    launch_update_rho_delta_without_ineq<double>(
        s, d_.B, d_.n, d_.p, update_rho_delta_args<double>(active_.data(), info_), rows_x(result_.view),
        rows_x(prox_.view), rows_y(result_.view), rows_y(prox_.view), settings_f_.data(), current_iter());
  }

  // ------------------------------------------------------------------ verbose / timing
  void print_iteration(cudaStream_t s) {
    Info h = info();
    int32_t it = 0;
    CUPIQP_CUDA_CHECK(cudaMemcpyAsync(&it, current_iter(), sizeof it, cudaMemcpyDeviceToHost, s));
    sync_stream(s, &counters_);
    std::printf("%3d   % .5e   % .5e  % .5e  % .5e  % .5e  % .3e  % .3e  % .3e  % .4f  % .4f\n", it,
                h.primal_obj, h.dual_obj, h.duality_gap, h.primal_res, h.dual_res, h.rho, h.delta, h.mu,
                h.primal_step, h.dual_step);
  }

  void event_record_start() {
    if (!settings.compute_timings) return;
    ev_start_.record(stream_.get());
    timing_started_ = true;
    have_update_ = false;
  }
  void event_record_update_end() {
    if (!settings.compute_timings) return;
    ev_update_end_.record(stream_.get());
    have_update_ = true;
  }
  void event_record_solve_end() {
    if (!settings.compute_timings) return;
    ev_solve_end_.record(stream_.get());
    timing_valid_ = true;
    timing_started_ = false;
  }
};

namespace {
__global__ void set_int_kernel(int32_t* p, int n, int32_t v) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[i] = v;
}
}  // namespace

void SparseSolverImpl::set_int_kernel_launch(cudaStream_t s, int32_t* p, int n, int32_t v) {
  if (n <= 0) return;
  set_int_kernel<<<grid_for(n), 256, 0, s>>>(p, n, v);
  CUPIQP_CUDA_CHECK_LAUNCH();
}

// ====================================================================== public API
SparseSolver::SparseSolver(cudaStream_t stream) : impl_(std::make_unique<SparseSolverImpl>(stream)) {}
SparseSolver::~SparseSolver() = default;
SparseSolver::SparseSolver(SparseSolver&&) noexcept = default;
SparseSolver& SparseSolver::operator=(SparseSolver&&) noexcept = default;

Settings& SparseSolver::settings() { return impl_->settings; }
const Settings& SparseSolver::settings() const { return impl_->settings; }
void SparseSolver::setup(const CsrPattern& P, const ValuesView& values, const CsrPattern* A,
                         const CsrPattern* G) {
  impl_->setup(P, values, A, G);
}
void SparseSolver::update(const ValuesView& values) { impl_->update(values, false); }
void SparseSolver::update_from_host(const ValuesView& host_values) { impl_->update(host_values, true); }
void SparseSolver::solve() { impl_->solve(); }
WarmStartBuffers SparseSolver::warm_start_buffers(bool host) { return impl_->warm_start_buffers(host); }
void SparseSolver::solve_warm(uint32_t fields, bool from_host) { impl_->solve_warm(fields, from_host); }
InputBuffers SparseSolver::input_buffers() { return impl_->input_buffers(false); }
InputBuffers SparseSolver::host_input_buffers() { return impl_->input_buffers(true); }
void SparseSolver::update_staged(const UpdateBlocks& blocks, bool from_host) { impl_->update_staged(blocks, from_host); }
ResultView SparseSolver::result() const { return impl_->result(); }
void SparseSolver::download_x(double* host, cudaEvent_t done) { impl_->download_x(host, done); }
Info SparseSolver::info() { return impl_->info(); }
cudaStream_t SparseSolver::stream() const { return impl_->stream(); }
int32_t SparseSolver::n() const { return impl_->dims().n; }
int32_t SparseSolver::p() const { return impl_->dims().p; }
int32_t SparseSolver::m() const { return impl_->dims().m; }
Counters SparseSolver::counters() const { return impl_->counters(); }
PhaseTimings SparseSolver::timings() { return impl_->timings(); }
std::vector<DebugBuffer> SparseSolver::debug_buffers() const { return impl_->debug_buffers(); }

}  // namespace cupiqp
