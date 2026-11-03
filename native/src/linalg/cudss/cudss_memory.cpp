#include "linalg/cudss/cudss_memory.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdio>

#include "common/check.hpp"

namespace cupiqp {

CudssMemory::CudssMemory(Counters* counters) : counters_(counters) {
  int dev = 0;
  CUPIQP_CUDA_CHECK(cudaGetDevice(&dev));
  CUPIQP_CUDA_CHECK(cudaDeviceGetDefaultMemPool(&pool_, dev));
}

cudssDeviceMemHandler_t CudssMemory::handler() {
  cudssDeviceMemHandler_t h{};
  h.ctx = this;
  h.device_alloc = &CudssMemory::alloc_cb;
  h.device_free = &CudssMemory::free_cb;
  std::snprintf(h.name, CUDSS_ALLOCATOR_NAME_LEN, "cupiqp_counting_pool");
  return h;
}

int CudssMemory::alloc_cb(void* ctx, void** ptr, size_t size, cudaStream_t s) {
  return static_cast<CudssMemory*>(ctx)->alloc(ptr, size, s);
}

int CudssMemory::free_cb(void* ctx, void* ptr, size_t size, cudaStream_t s) {
  return static_cast<CudssMemory*>(ctx)->free(ptr, size, s);
}

// Callbacks run on the host thread that called cudssExecute / cudssData*,
// possibly while `s` is being captured: a pool allocation then becomes a
// graph memory node (none happen in steady state, see the header).
int CudssMemory::alloc(void** ptr, size_t size, cudaStream_t s) {
  if (size == 0) {
    *ptr = nullptr;
    ++zero_byte_requests_;
    return 0;
  }
  ++counters_->cudss_device_allocs;
  if (cudaMallocFromPoolAsync(ptr, size, pool_, s) != cudaSuccess) return 1;
  largest_request_ = std::max(largest_request_, size);
  live_bytes_ += size;
  peak_live_bytes_ = std::max(peak_live_bytes_, live_bytes_);
  return 0;
}

int CudssMemory::free(void* ptr, size_t size, cudaStream_t s) {
  if (ptr == nullptr) {
    ++zero_byte_frees_;
    return 0;
  }
  ++counters_->cudss_device_frees;
  live_bytes_ -= std::min(live_bytes_, size);
  return cudaFreeAsync(ptr, s) == cudaSuccess ? 0 : 1;
}

}  // namespace cupiqp
