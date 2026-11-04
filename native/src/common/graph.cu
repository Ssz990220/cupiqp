#include "common/graph.hpp"

#include <cuda_runtime.h>

#include <exception>

#include "common/check.hpp"

namespace cupiqp {

namespace {

__global__ void set_condition_kernel(cudaGraphConditionalHandle h, const int32_t* cond) {
  cudaGraphSetConditional(h, *cond != 0 ? 1u : 0u);
}

}  // namespace

GraphExec& GraphExec::operator=(GraphExec&& o) noexcept {
  if (this != &o) {
    reset();
    graph_ = o.graph_;
    exec_ = o.exec_;
    o.graph_ = nullptr;
    o.exec_ = nullptr;
  }
  return *this;
}

void GraphExec::instantiate(cudaGraph_t graph) {
  reset();
  graph_ = graph;
  CUPIQP_CUDA_CHECK(cudaGraphInstantiate(&exec_, graph_, 0));
}

void GraphExec::launch(cudaStream_t s, Counters* counters) {
  CUPIQP_CUDA_CHECK(cudaGraphLaunch(exec_, s));
  if (counters) ++counters->graph_launches;
}

void GraphExec::reset() noexcept {
  if (exec_) cudaGraphExecDestroy(exec_);
  if (graph_) cudaGraphDestroy(graph_);
  exec_ = nullptr;
  graph_ = nullptr;
}

LoopContext::LoopContext(Counters* counters) : counters_(counters) {
  for (auto& h : helpers_) CUPIQP_CUDA_CHECK(cudaStreamCreateWithFlags(&h, cudaStreamNonBlocking));
  host_flags_.allocate(kMaxDepth);
  for (int i = 0; i < kMaxDepth; ++i) host_flags_[i] = 0;
}

LoopContext::~LoopContext() {
  for (auto& h : helpers_)
    if (h) cudaStreamDestroy(h);
}

bool is_capturing(cudaStream_t s) {
  cudaStreamCaptureStatus st = cudaStreamCaptureStatusNone;
  CUPIQP_CUDA_CHECK(cudaStreamIsCapturing(s, &st));
  return st == cudaStreamCaptureStatusActive;
}

cudaGraph_t capture(cudaStream_t s, const std::function<void(cudaStream_t)>& fn, Counters* counters) {
  CUPIQP_CUDA_CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
  try {
    fn(s);
  } catch (...) {
    cudaGraph_t g = nullptr;
    cudaStreamEndCapture(s, &g);
    if (g) cudaGraphDestroy(g);
    cudaGetLastError();
    throw;
  }
  cudaGraph_t g = nullptr;
  CUPIQP_CUDA_CHECK(cudaStreamEndCapture(s, &g));
  if (counters) ++counters->graph_captures;
  return g;
}

void sync_stream(cudaStream_t s, Counters* counters) {
  CUPIQP_CUDA_CHECK(cudaStreamSynchronize(s));
  if (counters) ++counters->host_syncs;
}

namespace {

// Record a conditional node (WHILE or IF) on `s`, which is capturing; see the
// header for the suspend/resume scheme.
void capture_conditional(LoopContext& ctx, cudaStream_t s, const int32_t* cond,
                         cudaGraphConditionalNodeType type,
                         const std::function<void(cudaStream_t)>& body) {
  // The body is captured on the SAME stream: suspend the
  // enclosing capture (end it; the graph object stays valid), capture the body
  // into the conditional node's body graph, then resume the enclosing capture
  // after the node with cudaStreamBeginCaptureToGraph. Streams that joined the
  // enclosing capture through events leave it
  // when it is suspended, so they can join the body capture; a helper-stream
  // body would merge two captures and break.
  CUPIQP_REQUIRE(ctx.depth < LoopContext::kMaxDepth, "capture_while nesting too deep");
  cudaStreamCaptureStatus status;
  cudaGraph_t graph = nullptr;
  cudaStreamCaptureMode mode = cudaStreamCaptureModeThreadLocal;
  CUPIQP_CUDA_CHECK(cudaStreamGetCaptureInfo(s, &status, nullptr, &graph, nullptr, nullptr, nullptr));

  cudaGraphConditionalHandle handle;
  CUPIQP_CUDA_CHECK(cudaGraphConditionalHandleCreate(&handle, graph, 0, 0));
  // Condition for the first test, recorded in the enclosing capture.
  set_condition_kernel<<<1, 1, 0, s>>>(handle, cond);
  CUPIQP_CUDA_CHECK_LAUNCH();

  const cudaGraphNode_t* deps = nullptr;
  const cudaGraphEdgeData* edges = nullptr;
  size_t ndeps = 0;
  CUPIQP_CUDA_CHECK(cudaStreamGetCaptureInfo(s, &status, nullptr, &graph, &deps, &edges, &ndeps));

  cudaGraphNodeParams params{};
  params.type = cudaGraphNodeTypeConditional;
  params.conditional.handle = handle;
  params.conditional.type = type;
  params.conditional.size = 1;
  cudaGraphNode_t node;
  CUPIQP_CUDA_CHECK(cudaGraphAddNode(&node, graph, deps, edges, ndeps, &params));
  cudaGraph_t body_graph = params.conditional.phGraph_out[0];

  // Suspend the enclosing capture.
  cudaGraph_t suspended = nullptr;
  CUPIQP_CUDA_CHECK(cudaStreamEndCapture(s, &suspended));
  CUPIQP_REQUIRE(suspended == graph, "capture_while: unexpected graph on suspend");

  CUPIQP_CUDA_CHECK(cudaStreamBeginCaptureToGraph(s, body_graph, nullptr, nullptr, 0, mode));
  ++ctx.depth;
  std::exception_ptr err;
  try {
    body(s);
    if (type == cudaGraphCondTypeWhile) {  // re-evaluate for the next pass
      set_condition_kernel<<<1, 1, 0, s>>>(handle, cond);
      CUPIQP_CUDA_CHECK_LAUNCH();
    }
  } catch (...) {
    err = std::current_exception();
  }
  --ctx.depth;
  cudaGraph_t out = nullptr;
  cudaError_t end_err = cudaStreamEndCapture(s, &out);
  // Resume the enclosing capture after the conditional node.
  CUPIQP_CUDA_CHECK(cudaStreamBeginCaptureToGraph(s, graph, &node, nullptr, 1, mode));
  if (err) {
    cudaGetLastError();
    std::rethrow_exception(err);
  }
  CUPIQP_CUDA_CHECK(end_err);
}

}  // namespace

void capture_while(LoopContext& ctx, cudaStream_t s, const int32_t* cond,
                   const std::function<void(cudaStream_t)>& body) {
  if (!is_capturing(s)) {
    int32_t* flag = ctx.host_flag(0);
    for (;;) {
      CUPIQP_CUDA_CHECK(cudaMemcpyAsync(flag, cond, sizeof(int32_t), cudaMemcpyDeviceToHost, s));
      sync_stream(s, ctx.counters());
      if (*flag == 0) break;
      body(s);
    }
    return;
  }

  capture_conditional(ctx, s, cond, cudaGraphCondTypeWhile, body);
}

void capture_if(LoopContext& ctx, cudaStream_t s, const int32_t* cond, int32_t* host_flag,
                const std::function<void(cudaStream_t)>& body) {
  if (!is_capturing(s)) {
    CUPIQP_CUDA_CHECK(cudaMemcpyAsync(host_flag, cond, sizeof(int32_t), cudaMemcpyDeviceToHost, s));
    sync_stream(s, ctx.counters());
    if (*host_flag != 0) body(s);
    return;
  }
  capture_conditional(ctx, s, cond, cudaGraphCondTypeIf, body);
}

}  // namespace cupiqp
