// RAII wrappers for CUDA streams and events: move-only, released in the
// destructor, never copied.
#pragma once

#include <cuda_runtime_api.h>

#include <utility>

#include "common/check.hpp"

namespace cupiqp {

class CudaStream {
 public:
  CudaStream() = default;
  // Create and own a stream (cudaStreamDefault: blocking w.r.t. the legacy stream).
  static CudaStream create(unsigned flags = cudaStreamDefault) {
    CudaStream s;
    CUPIQP_CUDA_CHECK(cudaStreamCreateWithFlags(&s.s_, flags));
    s.owned_ = true;
    return s;
  }
  // Borrow a caller stream (never destroyed here).
  static CudaStream borrow(cudaStream_t s) {
    CudaStream b;
    b.s_ = s;
    b.owned_ = false;
    return b;
  }
  ~CudaStream() { reset(); }
  CudaStream(const CudaStream&) = delete;
  CudaStream& operator=(const CudaStream&) = delete;
  CudaStream(CudaStream&& o) noexcept : s_(std::exchange(o.s_, nullptr)), owned_(std::exchange(o.owned_, false)) {}
  CudaStream& operator=(CudaStream&& o) noexcept {
    if (this != &o) {
      reset();
      s_ = std::exchange(o.s_, nullptr);
      owned_ = std::exchange(o.owned_, false);
    }
    return *this;
  }
  cudaStream_t get() const { return s_; }
  bool owned() const { return owned_; }
  void reset() noexcept {
    if (owned_ && s_) cudaStreamDestroy(s_);
    s_ = nullptr;
    owned_ = false;
  }

 private:
  cudaStream_t s_ = nullptr;
  bool owned_ = false;
};

class CudaEvent {
 public:
  CudaEvent() = default;
  static CudaEvent create(bool timing = false) {
    CudaEvent e;
    CUPIQP_CUDA_CHECK(cudaEventCreateWithFlags(&e.e_, timing ? cudaEventDefault : cudaEventDisableTiming));
    return e;
  }
  ~CudaEvent() {
    if (e_) cudaEventDestroy(e_);
  }
  CudaEvent(const CudaEvent&) = delete;
  CudaEvent& operator=(const CudaEvent&) = delete;
  CudaEvent(CudaEvent&& o) noexcept : e_(std::exchange(o.e_, nullptr)) {}
  CudaEvent& operator=(CudaEvent&& o) noexcept {
    if (this != &o) {
      if (e_) cudaEventDestroy(e_);
      e_ = std::exchange(o.e_, nullptr);
    }
    return *this;
  }
  cudaEvent_t get() const { return e_; }
  void record(cudaStream_t s) { CUPIQP_CUDA_CHECK(cudaEventRecord(e_, s)); }

 private:
  cudaEvent_t e_ = nullptr;
};

}  // namespace cupiqp
