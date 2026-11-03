// M5 cudss: patch_pageable_memcpy (FEASIBILITY.md 3.3).
//
// cuDSS 0.8 SOLVE issues two 8-byte host-to-device copies from pageable host
// memory. A plain graph accepts such memcpy nodes, but a conditional node body
// does not (cudaGraphInstantiate returns cudaErrorInvalidValue). This pass
// re-points each of them to a pinned snapshot of the same bytes taken when the
// pass runs. The snapshot is correct only while the pageable bytes would not
// change between capture and replay; with cuDSS 0.8.0 they are an internal
// address that stays fixed after ANALYSIS (verified by the module tests).
#include <cuda_runtime.h>

#include <cstring>
#include <vector>

#include "common/check.hpp"
#include "linalg/cudss/cudss_solver.hpp"

namespace cupiqp {

namespace {

bool is_pageable_host(const void* p) {
  cudaPointerAttributes at{};
  cudaError_t e = cudaPointerGetAttributes(&at, p);
  if (e != cudaSuccess) {
    cudaGetLastError();  // older runtimes report unregistered host memory as an error
    return true;
  }
  return at.type == cudaMemoryTypeUnregistered;
}

int patch_graph(cudaGraph_t graph, std::vector<CudaPinnedBuffer<unsigned char>>& keep) {
  size_t count = 0;
  CUPIQP_CUDA_CHECK(cudaGraphGetNodes(graph, nullptr, &count));
  std::vector<cudaGraphNode_t> nodes(count);
  if (count) CUPIQP_CUDA_CHECK(cudaGraphGetNodes(graph, nodes.data(), &count));
  int patched = 0;
  for (cudaGraphNode_t node : nodes) {
    cudaGraphNodeType type;
    CUPIQP_CUDA_CHECK(cudaGraphNodeGetType(node, &type));
    if (type == cudaGraphNodeTypeGraph) {
      cudaGraph_t child = nullptr;
      CUPIQP_CUDA_CHECK(cudaGraphChildGraphNodeGetGraph(node, &child));
      patched += patch_graph(child, keep);
    } else if (type == cudaGraphNodeTypeConditional) {
      cudaGraphNodeParams np{};
      CUPIQP_CUDA_CHECK(cudaGraphNodeGetParams(node, &np));
      for (unsigned i = 0; i < np.conditional.size; ++i)
        if (np.conditional.phGraph_out && np.conditional.phGraph_out[i])
          patched += patch_graph(np.conditional.phGraph_out[i], keep);
    } else if (type == cudaGraphNodeTypeMemcpy) {
      cudaMemcpy3DParms mp{};
      CUPIQP_CUDA_CHECK(cudaGraphMemcpyNodeGetParams(node, &mp));
      // Only copies whose source is on the host: the direction decides first.
      // (On WSL cudaPointerGetAttributes can report cudaMalloc memory as
      // "unregistered", so a device-to-device copy must never be inspected.)
      const bool host_source = mp.kind == cudaMemcpyHostToDevice || mp.kind == cudaMemcpyHostToHost ||
                               mp.kind == cudaMemcpyDefault;
      if (!host_source || mp.srcArray != nullptr || mp.srcPtr.ptr == nullptr ||
          !is_pageable_host(mp.srcPtr.ptr))
        continue;
      // Captured cudaMemcpyAsync calls are 1-D: width bytes at srcPtr.ptr + srcPos.x.
      CUPIQP_REQUIRE(mp.extent.height <= 1 && mp.extent.depth <= 1 && mp.srcPos.y == 0 && mp.srcPos.z == 0,
                     "patch_pageable_memcpy: unsupported multi-dimensional pageable copy");
      const size_t bytes = mp.extent.width;
      const auto* src = static_cast<const unsigned char*>(mp.srcPtr.ptr) + mp.srcPos.x;
      CudaPinnedBuffer<unsigned char> snap(bytes);
      std::memcpy(snap.data(), src, bytes);
      mp.srcPtr = make_cudaPitchedPtr(snap.data(), bytes, bytes, 1);
      mp.srcPos = make_cudaPos(0, 0, 0);
      CUPIQP_CUDA_CHECK(cudaGraphMemcpyNodeSetParams(node, &mp));
      keep.push_back(std::move(snap));
      ++patched;
    }
  }
  return patched;
}

}  // namespace

int patch_pageable_memcpy(cudaGraph_t graph, std::vector<CudaPinnedBuffer<unsigned char>>& keep) {
  CUPIQP_REQUIRE(graph != nullptr, "patch_pageable_memcpy: null graph");
  return patch_graph(graph, keep);
}

}  // namespace cupiqp
