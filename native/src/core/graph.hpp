// CUDA Graph helpers and the device-controlled loop primitive.
//
// capture_while() reproduces warp.capture_while semantics:
//  * while the stream is being captured, it records a conditional WHILE node
//    whose condition is the device int32 *cond (set by a tiny kernel before
//    the node and again at the end of the body);
//  * otherwise it runs a host loop that copies *cond to pinned memory,
//    synchronizes the stream (counted in Counters::host_syncs) and calls the
//    body while it is nonzero.
// The body receives the stream it must enqueue on. Under capture this is a
// helper stream capturing into the conditional body graph, so EVERY function
// called from a body must take the stream as a parameter and must bind library
// handles (cudssSetStream, cusparseSetStream) to it on every call.
#pragma once

#include <cuda_runtime_api.h>

#include <array>
#include <cstdint>
#include <functional>

#include "core/buffer.hpp"
#include "cupiqp/cupiqp.hpp"

namespace cupiqp {

// Owned instantiated graph.
class GraphExec {
 public:
  GraphExec() = default;
  ~GraphExec() { reset(); }
  GraphExec(const GraphExec&) = delete;
  GraphExec& operator=(const GraphExec&) = delete;
  GraphExec(GraphExec&& o) noexcept : graph_(o.graph_), exec_(o.exec_) { o.graph_ = nullptr; o.exec_ = nullptr; }
  GraphExec& operator=(GraphExec&& o) noexcept;

  // Take ownership of `graph`, optionally edit it with `patch` before
  // instantiation (see M5 patch_pageable_memcpy), then instantiate.
  void instantiate(cudaGraph_t graph, const std::function<void(cudaGraph_t)>& patch = {});
  void launch(cudaStream_t s, Counters* counters);
  bool valid() const { return exec_ != nullptr; }
  cudaGraph_t graph() const { return graph_; }
  void reset() noexcept;

 private:
  cudaGraph_t graph_ = nullptr;
  cudaGraphExec_t exec_ = nullptr;
};

// Loop context shared by one solver: helper streams for nested conditional
// body captures and the pinned mirrors of the conditions for host loops.
class LoopContext {
 public:
  static constexpr int kMaxDepth = 4;
  explicit LoopContext(Counters* counters);
  ~LoopContext();
  LoopContext(const LoopContext&) = delete;
  LoopContext& operator=(const LoopContext&) = delete;

  Counters* counters() { return counters_; }
  cudaStream_t helper(int depth) const { return helpers_[depth]; }
  int32_t* host_flag(int depth) { return host_flags_.data() + depth; }
  int depth = 0;  // current nesting of captured conditional bodies

 private:
  Counters* counters_;
  std::array<cudaStream_t, kMaxDepth> helpers_{};
  CudaPinnedBuffer<int32_t> host_flags_;
};

bool is_capturing(cudaStream_t s);

// Capture everything `fn` enqueues on `s` into a new graph (thread-local mode).
cudaGraph_t capture(cudaStream_t s, const std::function<void(cudaStream_t)>& fn, Counters* counters);

// while (*cond != 0) body(stream);  see the header comment.
void capture_while(LoopContext& ctx, cudaStream_t s, const int32_t* cond,
                   const std::function<void(cudaStream_t)>& body);

// Synchronize `s` and count it.
void sync_stream(cudaStream_t s, Counters* counters);

}  // namespace cupiqp
