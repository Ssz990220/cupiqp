// ipm/solver: device-side warm start from the previous solve (phase 2 of the
// warm-start study, native/docs/modules/warmstart_study.md).
//
// Owns every buffer of the feature (previous duals in user coordinates, row
// keys of the current and the previous QP, the row match, the iterate ring
// buffer, CUB sort storage) and enqueues its kernels. All work is stream
// ordered and capturable; nothing here allocates or synchronizes after
// configure(). The solver (sparse_solver.cu) decides where in the solve graph
// each step runs.
#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>

#include "common/buffer.hpp"
#include "common/views.hpp"
#include "cupiqp/cupiqp.hpp"

namespace cupiqp {

// Packed (group, key) pair of one keyed inequality row. Sorted
// lexicographically by (group, key) with CUB radix sort.
struct WarmRowKey {
  int64_t key;
  int32_t group;
  int32_t pad;
};

class WarmStartDevice {
 public:
  // Validate the options against the dimensions and (re)allocate the buffers
  // whose size depends on them. Called outside any solve. Batch size 1 only.
  void configure(const Dims& d, const WarmStartOptions& opt, cudaStream_t s);
  bool configured() const { return configured_; }
  const WarmStartOptions& options() const { return opt_; }

  // Host (pinned) or device staging of the current QP's keys.
  RowKeyBuffers key_buffers(bool host);
  bool keyed() const { return opt_.row_match == RowMatch::Keys; }
  int32_t key_count() const { return key_count_; }

  // Upload the current keys from the pinned host staging (graph node).
  void upload_keys(cudaStream_t s);
  // match[i] = row of the previous QP matching keyed row i, or -1.
  void match_rows(cudaStream_t s);
  // The current keys become the previous keys.
  void commit_keys(cudaStream_t s);

  // cond[0] = 1 when the start must be cold, cond[1] = 1 when warm (device
  // decision on the previous status).
  void decide(cudaStream_t s);
  const int32_t* cold_flag() const { return cond_.data(); }
  const int32_t* warm_flag() const { return cond_.data() + 1; }
  int32_t* host_flag() { return host_flag_.data(); }

  // Write the warm start into `r` (scaled coordinates of the new QP): x = 0,
  // y = previous y, s from the new bounds at x = 0 clamped to s_min, z =
  // max(previous z of the matched row, z_min). Zero on infinite bounds.
  void build_start(cudaStream_t s, const SparseDataView<double>& data, const ScalingView<double>& sc,
                   bool scaled, const VariablesView<double>& r);

  // Store the current iterate (y, z, s; scaled) in ring slot iter % (K + 1).
  void ring_store(cudaStream_t s, const VariablesView<double>& r, const int32_t* current_iter);

  // At the end of a solve, still in its scaling: pick the end-of-solve
  // iterate or the one `snapshot_back` iterations earlier, unscale its y and
  // z into the previous-solve storage and keep the final status.
  void record_previous(cudaStream_t s, const VariablesView<double>& r, const ScalingView<double>& sc, bool scaled,
                       const int32_t* current_iter, const int32_t* status);

  bool ring_enabled() const { return opt_.snapshot_slots > 0; }
  const double* prev_dual() const { return prev_dual_.data(); }
  const int32_t* match() const { return match_.data(); }

 private:
  Dims d_{};
  WarmStartOptions opt_{};
  bool configured_ = false;
  int32_t key_begin_ = 0, key_count_ = 0;

  CudaBuffer<double> prev_dual_;  // (p + num_ineq) user coordinates [y | z_all]
  CudaBuffer<int32_t> prev_status_;
  CudaBuffer<int32_t> cond_;       // [cold, warm]
  CudaPinnedBuffer<int32_t> host_flag_;

  // Keys (only when row_match == Keys).
  CudaBuffer<int64_t> keys_;       // current, device staging
  CudaBuffer<int32_t> groups_;
  CudaPinnedBuffer<int64_t> keys_host_;
  CudaPinnedBuffer<int32_t> groups_host_;
  CudaBuffer<WarmRowKey> prev_keys_, sorted_keys_;
  CudaBuffer<int32_t> iota_, sorted_rows_, match_;
  CudaBuffer<unsigned char> sort_temp_;
  size_t sort_temp_bytes_ = 0;

  // Ring buffer: (K + 1) slots of [y | z_all | s_all].
  CudaBuffer<double> ring_;
  int32_t slot_width_ = 0;
};

}  // namespace cupiqp
