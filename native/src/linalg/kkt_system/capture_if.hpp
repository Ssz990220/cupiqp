// M6 kkt_system: capture_if, the one-shot counterpart of capture_while
// (common/graph.hpp), reproducing warp.capture_if semantics:
//  * while `s` is being captured, record a conditional IF node whose
//    condition is the device int32 *cond (read by a tiny kernel enqueued
//    right before the node); the body is captured on the LoopContext helper
//    stream of the current depth;
//  * otherwise copy *cond into `host_flag` (pinned), synchronize `s` (counted
//    in Counters::host_syncs) and call the body once if it is nonzero.
// Kept local to this module; see docs/modules/kkt_system.md (requested move
// to common/graph.hpp).
#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>
#include <functional>

#include "common/graph.hpp"

namespace cupiqp {

void capture_if(LoopContext& ctx, cudaStream_t s, const int32_t* cond, int32_t* host_flag,
                const std::function<void(cudaStream_t)>& body);

}  // namespace cupiqp
