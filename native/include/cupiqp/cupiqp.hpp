// cuPIQP native: public C++ API of the sparse FP64 solver.
//
// Solves   min 1/2 x'Px + c'x   s.t.  A x = b,  h_l <= G x <= h_u,  x_l <= x <= x_u
// with the proximal interior-point method of the pinned Python/Warp cuPIQP
// (d538dda), on the GPU, without any Python dependency. The condensed KKT
// system is solved by a dense Schur complement on the x block (custom
// Cholesky), so the number of variables n is limited to 2048; setup() rejects
// larger problems with ErrorCode::Unsupported. Only the CUDA runtime is linked
// (no sparse or dense math library). See native/docs/design/DESIGN.md.
#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace cupiqp {

// Infinite bounds: a lower bound <= -PIQP_INF or an upper bound >= PIQP_INF is
// absent (one-sided tests, as in the reference finite_bound_masks_kernel).
inline constexpr double PIQP_INF = 1e20;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------
enum class ErrorCode : int32_t {
  InvalidArgument = 1,  // bad pattern, shape, missing block, wrong call order
  Unsupported = 2,      // outside milestone 1 (batching, FP32, gradients, n > 2048, ...)
  Cuda = 3,             // CUDA runtime error
  Internal = 6          // broken invariant (codes 4 and 5 belonged to removed math libraries)
};

class Error : public std::runtime_error {
 public:
  Error(ErrorCode code, const std::string& what) : std::runtime_error(what), code_(code) {}
  ErrorCode code() const noexcept { return code_; }

 private:
  ErrorCode code_;
};

// ---------------------------------------------------------------------------
// Problem description
// ---------------------------------------------------------------------------

// Host CSR pattern: 0-based, column indices strictly increasing within each row
// (sorted, no duplicates). Explicit zeros are stored entries: they keep their
// slot for the lifetime of the solver and may later become nonzero.
struct CsrPattern {
  int32_t rows = 0;
  int32_t cols = 0;
  std::vector<int32_t> indptr;   // rows + 1 entries
  std::vector<int32_t> indices;  // nnz entries
  int32_t nnz() const { return indptr.empty() ? 0 : indptr.back(); }

  // Narrowing constructor for int64 patterns; throws InvalidArgument when an
  // offset or index does not fit in int32 (nnz must be < 2^31).
  static CsrPattern from_int64(int64_t rows, int64_t cols, const int64_t* indptr,
                               const int64_t* indices);
};

// Device pointers to value arrays (double, contiguous), in the stored-entry
// order of the setup() patterns for P, A, G. In setup(), nullptr means the block
// or bound side is absent for the lifetime of the solver. In update(), nullptr
// means "keep the current value".
struct ValuesView {
  const double* P = nullptr;
  const double* c = nullptr;
  const double* A = nullptr;
  const double* b = nullptr;
  const double* G = nullptr;
  const double* h_l = nullptr;
  const double* h_u = nullptr;
  const double* x_l = nullptr;
  const double* x_u = nullptr;
};

// Which value blocks an update replaces.
struct UpdateBlocks {
  bool P = false, c = false, A = false, b = false, G = false;
  bool h_l = false, h_u = false, x_l = false, x_u = false;
  static UpdateBlocks all() { return {true, true, true, true, true, true, true, true, true}; }
};

// Solver-owned input buffers, allocated in setup() at fixed addresses for the
// lifetime of the solver (device or pinned host memory). Write new values in
// the stored-entry order of the setup() patterns, then call update_staged().
// Pointers of absent blocks / bound sides are nullptr with length 0.
struct InputBuffers {
  double *P = nullptr, *c = nullptr, *A = nullptr, *b = nullptr, *G = nullptr;
  double *h_l = nullptr, *h_u = nullptr, *x_l = nullptr, *x_u = nullptr;
  int64_t len_P = 0, len_c = 0, len_A = 0, len_b = 0, len_G = 0;
  int64_t len_h_l = 0, len_h_u = 0, len_x_l = 0, len_x_u = 0;
};

// ---------------------------------------------------------------------------
// Settings, status, results
// ---------------------------------------------------------------------------
enum class Status : int32_t {
  Unsolved = -1,
  Solved = 0,
  MaxIterReached = 1,
  PrimalInfeasible = 2,
  DualInfeasible = 3,
  NumericalIssues = 4
};
const char* status_name(Status s);  // "CUPIQP_SOLVED", ... (reference names)

enum class GraphMode : int32_t {
  Auto = 0,                    // enable_cuda_graph: SolveGraphDeviceLoop; otherwise None
  None = 1,                    // plain launches, host-driven loops
  IterationGraphHostLoop = 2,  // one IPM iteration per CUDA graph, host-driven loop
                               // (the reference default)
  SolveGraphDeviceLoop = 3     // whole solve in one graph with a device-controlled loop
};

// Same fields, names and FP64 defaults as cupiqp.Settings.for_dtype(float64),
// except the reference's deterministic-mode switch of its sparse direct
// solver: the KKT solve and the sparse products here always use fixed
// summation orders. The Python shim accepts and ignores that field
// (docs/design/DEVIATIONS.md).
struct Settings {
  double rho_init = 1e-6;
  double delta_init = 1e-4;
  double eps_abs = 1e-8;
  double eps_rel = 1e-9;
  bool check_duality_gap = true;
  double eps_duality_gap_abs = 1e-8;
  double eps_duality_gap_rel = 1e-9;
  double infeasibility_threshold = 0.9;
  double reg_lower_limit = 1e-10;
  double reg_finetune_lower_limit = 1e-13;
  int32_t reg_finetune_primal_update_threshold = 7;
  int32_t reg_finetune_dual_update_threshold = 7;
  int32_t max_iter = 250;
  int32_t max_factor_retires = 10;  // (sic) reference spelling
  bool preconditioner_scale_cost = false;
  bool preconditioner_reuse_on_update = false;
  int32_t preconditioner_iter = 10;
  double tau = 0.99;
  bool iterative_refinement_always_enabled = false;
  double iterative_refinement_eps_abs = 1e-12;
  double iterative_refinement_eps_rel = 1e-12;
  int32_t iterative_refinement_max_iter = 10;
  double iterative_refinement_min_improvement_rate = 5.0;
  double iterative_refinement_static_regularization_eps = 1e-8;
  double iterative_refinement_static_regularization_rel = 4.930380657631324e-32;
  bool enable_cuda_graph = true;
  bool verbose = false;          // only honoured with GraphMode::None
  bool compute_timings = false;  // CUDA-event phase timers (adds no host sync)
  GraphMode graph_mode = GraphMode::Auto;

  // Reference Settings.verify_settings().
  bool verify() const;
};

// Host snapshot of the per-problem diagnostics (reference Result.info).
struct Info {
  Status status = Status::Unsolved;
  int32_t iter = 0;
  int32_t iter_total = 0;
  int32_t factor_retries = 0;
  int32_t no_primal_update = 0;
  int32_t no_dual_update = 0;
  double rho = 0, delta = 0, mu = 0, sigma = 0, primal_step = 0, dual_step = 0;
  double primal_res = 0, primal_res_rel = 0, dual_res = 0, dual_res_rel = 0;
  double primal_res_reg = 0, primal_res_reg_rel = 0, dual_res_reg = 0, dual_res_reg_rel = 0;
  double primal_prox_inf = 0, dual_prox_inf = 0, prev_primal_res = 0, prev_dual_res = 0;
  double primal_obj = 0, dual_obj = 0, duality_gap = 0, duality_gap_rel = 0, reg_limit = 0;
};

// Device pointers to the solution in the user (unscaled) coordinates. Stable
// from setup() to destruction. Lengths: x n; y p; z_l/s_l m if h_l given else 0;
// z_u/s_u m if h_u given else 0; z_bl/s_bl n if x_l given else 0; z_bu/s_bu
// n if x_u given else 0. Entries of infinite bounds are exactly 0.
struct ResultView {
  const double *x = nullptr, *y = nullptr;
  const double *z_l = nullptr, *z_u = nullptr, *z_bl = nullptr, *z_bu = nullptr;
  const double *s_l = nullptr, *s_u = nullptr, *s_bl = nullptr, *s_bu = nullptr;
  int32_t n = 0, p = 0, m = 0;
  int32_t num_hl = 0, num_hu = 0, num_xl = 0, num_xu = 0;
};

// Instrumentation counters (cumulative since construction).
struct Counters {
  int64_t host_syncs = 0;           // stream/device synchronizations issued by the solver
  int64_t graph_launches = 0;       // cudaGraphLaunch calls
  int64_t graph_captures = 0;       // stream captures performed
  int64_t updates = 0;
  int64_t solves = 0;
};

// A named device span, exposed for parity tests and debugging.
struct DebugBuffer {
  std::string name;
  const void* ptr = nullptr;
  int64_t rows = 0, cols = 0;  // logical (B, k) shape
  char dtype = 'f';            // 'f' = double, 'i' = int32, 'b' = bool (uint8)
};

// Optional CUDA-event phase timings of the last update()/solve() in ms.
struct PhaseTimings {
  float update_ms = 0, scaling_ms = 0, kkt_scatter_ms = 0;
  float solve_ms = 0, factor_ms_total = 0, kkt_solve_ms_total = 0;
};

// EXPERIMENTAL initial-point hook (warm-start study, see
// native/docs/modules/warmstart_study.md). Pointers to solver-owned staging
// buffers in the user (unscaled) coordinates, laid out like ResultView.
// Allocated on the first warm_start_buffers() call and stable afterwards.
struct WarmStartBuffers {
  double *x = nullptr, *y = nullptr;
  double *z_l = nullptr, *z_u = nullptr, *z_bl = nullptr, *z_bu = nullptr;
  double *s_l = nullptr, *s_u = nullptr, *s_bl = nullptr, *s_bu = nullptr;
  int32_t n = 0, p = 0, m = 0;
  int32_t num_hl = 0, num_hu = 0, num_xl = 0, num_xu = 0;
};

// Which parts of the staged start solve_warm() uses (bit mask). Parts not
// selected come from the default initial guess (eq. 12 of Schwan et al.
// 2023), which then runs first. The proximal centers always start at the
// resulting initial point, and mu = s'z / (number of finite bounds).
enum WarmStartField : uint32_t {
  kWarmX = 1u,
  kWarmY = 2u,
  kWarmS = 4u,
  kWarmZ = 8u,
  kWarmAll = 15u
};

// ---------------------------------------------------------------------------
// Warm start from the previous solve (built on the device)
// ---------------------------------------------------------------------------

// How the inequality rows of the new QP find their rows in the previous QP.
enum class RowMatch : int32_t {
  Identity = 0,  // row i of the new QP is row i of the previous QP
  Keys = 1       // rows carry int64 keys (see WarmStartOptions and row_key_buffers())
};

// Options of solve_warm_start(). The start for QP k+1 is built on the device
// from the solver's own result of QP k:
//   x = 0;  y = y_k (or 0);  s = slack of the new bounds at x = 0, at least s_min;
//   z = max(z_k of the matching row, z_min), or max(z_unmatched, z_min) for a
//   row without a match.  s and z are 0 on infinite bounds.
// All values are in user (unscaled) coordinates. The defaults are the
// strategy recommended by the warm-start study (x0_duals, s_min 0.1, z_min 1e-4).
struct WarmStartOptions {
  double s_min = 0.1;
  double z_min = 1e-4;
  double z_unmatched = 1e-4;
  bool previous_y = true;          // false: y starts at 0
  bool cold_if_not_solved = true;  // previous status not Solved -> cold start (decided on the device)

  // Mid-solve snapshot (Gondzio and Grothey 2002): keep the last
  // snapshot_slots + 1 iterates of the solve in a ring buffer and start the
  // next QP from the iterate snapshot_back iterations before the end
  // (0 = the end-of-solve iterate). If the solve ran fewer than
  // snapshot_back + 1 iterations, the end-of-solve iterate is used.
  // Requires 0 <= snapshot_back <= snapshot_slots. Costs one device copy of
  // (y, z, s) per IPM iteration while snapshot_slots > 0.
  int32_t snapshot_slots = 0;
  int32_t snapshot_back = 0;

  // Row matching. With RowMatch::Keys the keyed rows are the inequality rows
  // [key_row_begin, key_row_begin + key_row_count) (count -1: to the last
  // row); every other row (and every box bound) uses identity. A keyed row of
  // the new QP matches the keyed row of the previous QP with the same key
  // (and the same group id when row_groups is true); a row that matches
  // itself is preferred, then the lowest previous row index.
  RowMatch row_match = RowMatch::Identity;
  bool row_groups = false;
  int32_t key_row_begin = 0;
  int32_t key_row_count = -1;
};

// Solver-owned staging of the row keys of the next QP (device or pinned
// host), stable after set_warm_start(). keys[j] and groups[j] belong to
// inequality row key_row_begin + j; groups is nullptr unless row_groups.
struct RowKeyBuffers {
  int64_t* keys = nullptr;
  int32_t* groups = nullptr;
  int32_t count = 0;
};

class SparseSolverImpl;

// Sparse FP64 solver, batch size 1. Not thread safe; one instance per thread.
class SparseSolver {
 public:
  // stream == nullptr: the solver creates and owns a stream
  // (cudaStreamDefault flags). Otherwise the stream is borrowed.
  explicit SparseSolver(cudaStream_t stream = nullptr);
  ~SparseSolver();
  SparseSolver(const SparseSolver&) = delete;
  SparseSolver& operator=(const SparseSolver&) = delete;
  SparseSolver(SparseSolver&&) noexcept;
  SparseSolver& operator=(SparseSolver&&) noexcept;

  // Mutable before setup(); runtime fields may change between solves. Changing
  // a structural field after setup() makes update()/solve() throw.
  Settings& settings();
  const Settings& settings() const;

  // Fix the structure (patterns, present blocks and bound sides), allocate all
  // memory, load and scale the values, build the KKT index maps. Blocks once.
  // Throws Unsupported when n > 2048 (dense Schur complement limit).
  // P stores the full symmetric matrix (both triangles); P and c are required;
  // A requires b and vice versa; G requires at least one of h_l / h_u.
  void setup(const CsrPattern& P, const ValuesView& values, const CsrPattern* A = nullptr,
             const CsrPattern* G = nullptr);

  // Replace numeric values (device pointers, stream ordered). Asynchronous.
  void update(const ValuesView& values);
  // Same with host pointers; pinned memory keeps it asynchronous.
  void update_from_host(const ValuesView& host_values);

  // Preallocated inputs (recommended for repeated updates). Write the new
  // values into input_buffers() (device) or host_input_buffers() (pinned host),
  // ordered on stream(), then call update_staged(). Because the input
  // addresses never change, the whole update (including the host-to-device
  // copy when from_host is true) is replayed as one CUDA graph, captured on the
  // first call for each combination of blocks. Results are identical to
  // update() with the same values.
  InputBuffers input_buffers();
  InputBuffers host_input_buffers();
  void update_staged(const UpdateBlocks& blocks = UpdateBlocks::all(), bool from_host = false);

  // Run the solver on the current data. Enqueued on stream(). With
  // GraphMode::SolveGraphDeviceLoop (what Auto selects unless graphs are off,
  // verbose is set or iterative refinement is always on) it is one graph
  // launch with no host wait; the other modes wait on one loop-state copy per
  // IPM iteration.
  void solve();

  // EXPERIMENTAL initial-point hook. warm_start_buffers(false) returns device
  // staging buffers, warm_start_buffers(true) pinned host ones. Write a start
  // (x, y, s, z in user coordinates; s and z strictly positive on finite
  // bounds, any value on infinite ones, which are zeroed), then call
  // solve_warm(fields, from_host). The start is copied (host-to-device when
  // from_host), Ruiz-scaled and installed inside the solve, so the whole
  // solve stays one CUDA graph (one graph per (fields, from_host) pair,
  // captured on first use). solve() itself is unchanged by this hook.
  WarmStartBuffers warm_start_buffers(bool host = false);
  void solve_warm(uint32_t fields = kWarmAll, bool from_host = false);

  // Warm start from the previous solve, built on the device inside the one
  // CUDA graph of the solve (no host round trip, no host sync).
  //
  //   s.set_warm_start(opts);        // once; allocates the feature's buffers
  //   for each QP of a sequence:
  //     s.update_staged(...);        // new data
  //     [write row_key_buffers(host) when opts.row_match == Keys]
  //     s.solve_warm_start(keys_from_host);
  //   s.reset_warm_start();          // before the next, unrelated sequence
  //
  // The first solve_warm_start() after set_warm_start(), reset_warm_start(),
  // solve() or solve_warm() is cold (the default initial guess) and records
  // its result; every later one starts from the previous result as described
  // in WarmStartOptions, or cold when the previous solve did not report
  // Solved (cold_if_not_solved; decided on the device). With keys_from_host
  // the keys are copied from the pinned staging inside the graph (8 bytes per
  // keyed row, plus 4 with groups); otherwise the device staging is used as
  // written on stream(). Changing the options re-captures the graphs and
  // makes the next solve_warm_start() cold. solve() is not affected.
  void set_warm_start(const WarmStartOptions& options = WarmStartOptions{});
  RowKeyBuffers row_key_buffers(bool host = true);
  void solve_warm_start(bool keys_from_host = true);
  void reset_warm_start();

  ResultView result() const;
  // Enqueue a copy of x (length n) into `host` (pinned memory recommended) and
  // record `done` (optional) after it.
  void download_x(double* host, cudaEvent_t done = nullptr);
  // Synchronize the stream and return the diagnostics.
  Info info();

  cudaStream_t stream() const;
  int32_t n() const;
  int32_t p() const;
  int32_t m() const;
  Counters counters() const;
  PhaseTimings timings();  // synchronizes when compute_timings is on
  std::vector<DebugBuffer> debug_buffers() const;

 private:
  std::unique_ptr<SparseSolverImpl> impl_;
};

// Library version, e.g. "0.1.0".
std::string version_string();

}  // namespace cupiqp
