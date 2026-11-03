// M5 cudss: device memory handler installed into the cuDSS handle.
//
// Every device request made by cuDSS goes through CudssMemory::alloc/free.
// Requests are served from the device's default memory pool
// (cudaMallocFromPoolAsync / cudaFreeAsync on the stream cuDSS passes),
// exactly as the reference's `cudss.set_async_workspace_allocator(handle,
// default_pool)`, and counted in Counters::cudss_device_allocs /
// cudss_device_frees.
//
// Zero-byte requests: cuDSS 0.8.0 asks for 0 bytes once per eager FACTORIZATION
// or SOLVE call and once per capture of them (measured; see
// docs/modules/cudss.md). They are answered with a null pointer here without
// touching the pool (the pool would return null as well) and are counted in
// zero_byte_requests(), NOT in Counters::cudss_device_allocs, since no memory
// is allocated. Every request with a nonzero size is a real allocation and is
// counted in Counters.
//
// DESIGN 10.1 asks for a setup-preallocated arena if cuDSS requests memory in
// steady state. Measured with cuDSS 0.8.0.10: after ANALYSIS and the first
// FACTORIZATION + SOLVE there are no nonzero requests (eager, captured, or
// replayed), so no arena is installed. The counters make any regression
// (e.g. another cuDSS version) visible.
#pragma once

#include <cudss.h>

#include <cstddef>
#include <cstdint>

#include "cupiqp/cupiqp.hpp"

namespace cupiqp {

class CudssMemory {
 public:
  explicit CudssMemory(Counters* counters);
  CudssMemory(const CudssMemory&) = delete;
  CudssMemory& operator=(const CudssMemory&) = delete;

  // Handler to pass to cudssSetDeviceMemHandler. `this` is its context and
  // must outlive the cuDSS handle and data.
  cudssDeviceMemHandler_t handler();

  // Statistics, cumulative since construction.
  int64_t zero_byte_requests() const { return zero_byte_requests_; }
  int64_t zero_byte_frees() const { return zero_byte_frees_; }
  size_t live_bytes() const { return live_bytes_; }  // currently held by cuDSS
  size_t peak_live_bytes() const { return peak_live_bytes_; }
  size_t largest_request() const { return largest_request_; }

 private:
  static int alloc_cb(void* ctx, void** ptr, size_t size, cudaStream_t s);
  static int free_cb(void* ctx, void* ptr, size_t size, cudaStream_t s);
  int alloc(void** ptr, size_t size, cudaStream_t s);
  int free(void* ptr, size_t size, cudaStream_t s);

  Counters* counters_;
  cudaMemPool_t pool_ = nullptr;
  int64_t zero_byte_requests_ = 0, zero_byte_frees_ = 0;
  size_t live_bytes_ = 0, peak_live_bytes_ = 0, largest_request_ = 0;
};

}  // namespace cupiqp
