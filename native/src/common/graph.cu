#include "common/graph.hpp"

#include <cuda_runtime.h>

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

void GraphExec::instantiate(cudaGraph_t graph, const std::function<void(cudaGraph_t)>& patch) {
  reset();
  graph_ = graph;
  if (patch) patch(graph_);
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

  CUPIQP_REQUIRE(ctx.depth < LoopContext::kMaxDepth, "capture_while nesting too deep");
  cudaStreamCaptureStatus status;
  cudaGraph_t graph = nullptr;
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
  params.conditional.type = cudaGraphCondTypeWhile;
  params.conditional.size = 1;
  cudaGraphNode_t node;
  CUPIQP_CUDA_CHECK(cudaGraphAddNode(&node, graph, deps, edges, ndeps, &params));
  CUPIQP_CUDA_CHECK(cudaStreamUpdateCaptureDependencies(s, &node, nullptr, 1, cudaStreamSetCaptureDependencies));

  cudaGraph_t body_graph = params.conditional.phGraph_out[0];
  cudaStream_t h = ctx.helper(ctx.depth);
  CUPIQP_CUDA_CHECK(cudaStreamBeginCaptureToGraph(h, body_graph, nullptr, nullptr, 0,
                                                  cudaStreamCaptureModeThreadLocal));
  ++ctx.depth;
  try {
    body(h);
    set_condition_kernel<<<1, 1, 0, h>>>(handle, cond);
    CUPIQP_CUDA_CHECK_LAUNCH();
  } catch (...) {
    --ctx.depth;
    cudaGraph_t dummy;
    cudaStreamEndCapture(h, &dummy);
    cudaGetLastError();
    throw;
  }
  --ctx.depth;
  cudaGraph_t out = nullptr;
  CUPIQP_CUDA_CHECK(cudaStreamEndCapture(h, &out));
}

}  // namespace cupiqp
