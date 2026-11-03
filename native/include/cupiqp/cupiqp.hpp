// cuPIQP native: public C++ API of the sparse FP64 solver.
//
// Solves   min 1/2 x'Px + c'x   s.t.  A x = b,  h_l <= G x <= h_u,  x_l <= x <= x_u
// with the proximal interior-point method of the pinned Python/Warp cuPIQP
// (d538dda), on the GPU, without any Python dependency. See native/docs/DESIGN.md.
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
  Unsupported = 2,      // feature outside milestone 1 (batching, FP32, gradients, ...)
  Cuda = 3,             // CUDA runtime error
  Cusparse = 4,         // cuSPARSE error
  Cudss = 5,            // cuDSS error
  Internal = 6          // broken invariant
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
  Auto = 0,                    // IterationGraphHostLoop if enable_cuda_graph, else None
  None = 1,                    // plain launches, host-driven loops
  IterationGraphHostLoop = 2,  // one IPM iteration per CUDA graph (reference default)
  SolveGraphDeviceLoop = 3     // EXPERIMENTAL whole-solve graph (FEASIBILITY.md 3.3)
};

// Same fields, names and FP64 defaults as cupiqp.Settings.for_dtype(float64).
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
  bool use_deterministic_mode_for_cudss = false;
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
  int64_t cudss_device_allocs = 0;  // allocations requested by cuDSS (counting handler)
  int64_t cudss_device_frees = 0;
  int64_t cudss_analysis_calls = 0;
  int64_t cudss_factor_calls = 0;   // host-side cudssExecute(FACTORIZATION) calls (incl. captures)
  int64_t cudss_solve_calls = 0;
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
  // memory, load and scale the values, run the symbolic analysis. Blocks once.
  // P stores the full symmetric matrix (both triangles); P and c are required;
  // A requires b and vice versa; G requires at least one of h_l / h_u.
  void setup(const CsrPattern& P, const ValuesView& values, const CsrPattern* A = nullptr,
             const CsrPattern* G = nullptr);

  // Replace numeric values (device pointers, stream ordered). Asynchronous.
  void update(const ValuesView& values);
  // Same with host pointers; pinned memory keeps it asynchronous.
  void update_from_host(const ValuesView& host_values);

  // Run the solver on the current data. Enqueued on stream(); in the default
  // graph mode it waits on one 8-byte loop-state copy per IPM iteration.
  void solve();

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

// Library version and the linked cuDSS version, e.g. "0.1.0 (cuDSS 0.8.0)".
std::string version_string();

}  // namespace cupiqp
