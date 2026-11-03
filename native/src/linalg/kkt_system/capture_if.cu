#include "linalg/kkt_system/capture_if.hpp"

#include <cuda_runtime.h>

#include "common/check.hpp"

namespace cupiqp {
namespace {

__global__ void set_if_condition_kernel(cudaGraphConditionalHandle h, const int32_t* cond) {
  cudaGraphSetConditional(h, *cond != 0 ? 1u : 0u);
}

}  // namespace

void capture_if(LoopContext& ctx, cudaStream_t s, const int32_t* cond, int32_t* host_flag,
                const std::function<void(cudaStream_t)>& body) {
  if (!is_capturing(s)) {
    CUPIQP_CUDA_CHECK(cudaMemcpyAsync(host_flag, cond, sizeof(int32_t), cudaMemcpyDeviceToHost, s));
    sync_stream(s, ctx.counters());
    if (*host_flag != 0) body(s);
    return;
  }

  CUPIQP_REQUIRE(ctx.depth < LoopContext::kMaxDepth, "capture_if nesting too deep");
  cudaStreamCaptureStatus status;
  cudaGraph_t graph = nullptr;
  CUPIQP_CUDA_CHECK(cudaStreamGetCaptureInfo(s, &status, nullptr, &graph, nullptr, nullptr, nullptr));

  cudaGraphConditionalHandle handle;
  CUPIQP_CUDA_CHECK(cudaGraphConditionalHandleCreate(&handle, graph, 0, 0));
  set_if_condition_kernel<<<1, 1, 0, s>>>(handle, cond);
  CUPIQP_CUDA_CHECK_LAUNCH();

  const cudaGraphNode_t* deps = nullptr;
  const cudaGraphEdgeData* edges = nullptr;
  size_t ndeps = 0;
  CUPIQP_CUDA_CHECK(cudaStreamGetCaptureInfo(s, &status, nullptr, &graph, &deps, &edges, &ndeps));

  cudaGraphNodeParams params{};
  params.type = cudaGraphNodeTypeConditional;
  params.conditional.handle = handle;
  params.conditional.type = cudaGraphCondTypeIf;
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
