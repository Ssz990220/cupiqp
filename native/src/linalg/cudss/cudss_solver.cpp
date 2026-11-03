// M5 cudss: CudssSolver (reference cupiqp/sparse/sparse_direct_solver.py,
// _CudssSolver / CudssStackedSolver with B = 1).
#include "linalg/cudss/cudss_solver.hpp"

#include <sys/stat.h>

#include <cstdlib>

#include "common/check.hpp"

#ifndef CUPIQP_CUDSS_MT_LIB
#define CUPIQP_CUDSS_MT_LIB ""
#endif

namespace cupiqp {

namespace {

bool is_file(const char* path) {
  struct stat st{};
  return path && path[0] && stat(path, &st) == 0 && S_ISREG(st.st_mode);
}

// Threading layer: the CUDSS_THREADING_LIB environment variable overrides the
// build-time path of libcudss_mtlayer_gomp.so.0. Like the reference, a path
// that is not an existing file is skipped (cuDSS then runs single threaded).
const char* threading_layer_path() {
  const char* env = std::getenv("CUDSS_THREADING_LIB");
  if (is_file(env)) return env;
  if (is_file(CUPIQP_CUDSS_MT_LIB)) return CUPIQP_CUDSS_MT_LIB;
  return nullptr;
}

template <class V>
void config_set(cudssConfig_t cfg, cudssConfigParam_t p, V v) {
  CUPIQP_CUDSS_CHECK(cudssConfigSet(cfg, p, &v, sizeof(V)));
}

}  // namespace

CudssSolver::CudssSolver(Counters* counters)
    : counters_(counters ? counters : &local_counters_), memory_(counters_) {}

// cuDSS frees its internal buffers through the handler on exec_stream_, which
// is destroyed only after this body (member destruction order).
CudssSolver::~CudssSolver() {
  if (A_) cudssMatrixDestroy(A_);
  if (x_) cudssMatrixDestroy(x_);
  if (b_) cudssMatrixDestroy(b_);
  if (data_ && handle_) cudssDataDestroy(handle_, data_);
  if (config_) cudssConfigDestroy(config_);
  if (handle_) cudssDestroy(handle_);
}

void CudssSolver::init(int32_t dim, int32_t nnz, const int32_t* indptr, const int32_t* indices,
                       double* values, bool deterministic, cudaStream_t s) {
  CUPIQP_REQUIRE(handle_ == nullptr, "CudssSolver::init called twice");
  CUPIQP_REQUIRE(dim > 0 && nnz > 0, "CudssSolver::init: dim and nnz must be positive");
  CUPIQP_REQUIRE(indptr && indices && values, "CudssSolver::init: null CSR pointer");
  dim_ = dim;
  nnz_ = nnz;

  exec_stream_ = CudaStream::create(cudaStreamNonBlocking);
  fork_ = CudaEvent::create(false);
  join_ = CudaEvent::create(false);

  CUPIQP_CUDSS_CHECK(cudssCreate(&handle_));
  CUPIQP_CUDSS_CHECK(cudssSetStream(handle_, exec_stream_.get()));
  // Workspace from the device's default memory pool (stream ordered), counted.
  cudssDeviceMemHandler_t mh = memory_.handler();
  CUPIQP_CUDSS_CHECK(cudssSetDeviceMemHandler(handle_, &mh));
  CUPIQP_CUDSS_CHECK(cudssConfigCreate(&config_));
  CUPIQP_CUDSS_CHECK(cudssDataCreate(handle_, &data_));

  rhs_.allocate(static_cast<size_t>(dim));
  sol_.allocate(static_cast<size_t>(dim));
  rhs_.zero_async(s);
  sol_.zero_async(s);

  // Symmetric matrix, only the lower triangle is read (the full P block may be
  // stored; its upper entries are ignored). Standard CSR (no row_end).
  CUPIQP_CUDSS_CHECK(cudssMatrixCreateCsr(&A_, dim, dim, nnz, const_cast<int32_t*>(indptr), nullptr,
                                          const_cast<int32_t*>(indices), values, CUDSS_R_32I,
                                          CUDSS_R_32I, CUDSS_R_64F, CUDSS_MTYPE_SYMMETRIC,
                                          CUDSS_MVIEW_LOWER, CUDSS_BASE_ZERO));
  CUPIQP_CUDSS_CHECK(cudssMatrixCreateDn(&b_, dim, 1, dim, rhs_.data(), CUDSS_R_64F, CUDSS_LAYOUT_COL_MAJOR));
  CUPIQP_CUDSS_CHECK(cudssMatrixCreateDn(&x_, dim, 1, dim, sol_.data(), CUDSS_R_64F, CUDSS_LAYOUT_COL_MAJOR));

  if (const char* lib = threading_layer_path()) CUPIQP_CUDSS_CHECK(cudssSetThreadingLayer(handle_, lib));

  // Reference configuration (sparse_direct_solver.py, _CudssSolver.__init__).
  config_set<int>(config_, CUDSS_CONFIG_HYBRID_MEMORY_MODE, 0);
  config_set<int>(config_, CUDSS_CONFIG_USE_CUDA_REGISTER_MEMORY, 1);
  config_set<cudssReorderingAlg_t>(config_, CUDSS_CONFIG_REORDERING_ALG, CUDSS_REORDERING_ALG_DEFAULT);
  config_set<int>(config_, CUDSS_CONFIG_USE_SUPERPANELS, 0);
  config_set<int>(config_, CUDSS_CONFIG_IR_N_STEPS, 0);
  if (deterministic) config_set<int>(config_, CUDSS_CONFIG_DETERMINISTIC_MODE, 1);
}

// Fork from `s` to exec_stream_, run the phase there, join back into `s`.
// Eagerly this orders the phase after all prior work on `s` and all later work
// on `s` after the phase. Under capture of `s` (thread-local mode) the event
// wait makes exec_stream_ part of the same capture and the final wait joins
// it back, so no extra nodes are created, only edges.
void CudssSolver::execute(int phase, cudaStream_t s) {
  CUPIQP_REQUIRE(handle_ != nullptr, "CudssSolver used before init()");
  cudaStream_t e = exec_stream_.get();
  CUPIQP_CUDA_CHECK(cudaEventRecord(fork_.get(), s));
  CUPIQP_CUDA_CHECK(cudaStreamWaitEvent(e, fork_.get(), 0));
  CUPIQP_CUDSS_CHECK(cudssSetStream(handle_, e));
  cudssStatus_t st = cudssExecute(handle_, phase, config_, data_, A_, x_, b_);
  // Join even on failure so a capture of `s` is not left with an unjoined fork.
  CUPIQP_CUDA_CHECK(cudaEventRecord(join_.get(), e));
  CUPIQP_CUDA_CHECK(cudaStreamWaitEvent(s, join_.get(), 0));
  CUPIQP_CUDSS_CHECK(st);
}

void CudssSolver::analyze(cudaStream_t s) {
  execute(CUDSS_PHASE_ANALYSIS, s);
  ++counters_->cudss_analysis_calls;
}

// The factorization outcome is NOT read: cudssDataGet(INFO/DIAG) waits for the
// stream and fails under capture (same as the reference). A failed
// factorization shows up only through non-finite solutions downstream.
void CudssSolver::factor(cudaStream_t s) {
  execute(CUDSS_PHASE_FACTORIZATION, s);
  ++counters_->cudss_factor_calls;
}

void CudssSolver::solve(cudaStream_t s) {
  execute(CUDSS_PHASE_SOLVE, s);
  ++counters_->cudss_solve_calls;
}

}  // namespace cupiqp
