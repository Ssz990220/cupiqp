// RAII device and pinned-host buffers. Allocation happens only at setup; the
// address of a buffer never changes after construction (no resize).
#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <utility>

#include "common/check.hpp"

namespace cupiqp {

template <class T>
class CudaBuffer {
 public:
  CudaBuffer() = default;
  explicit CudaBuffer(size_t count) { allocate(count); }
  ~CudaBuffer() { release(); }
  CudaBuffer(const CudaBuffer&) = delete;
  CudaBuffer& operator=(const CudaBuffer&) = delete;
  CudaBuffer(CudaBuffer&& o) noexcept : ptr_(o.ptr_), count_(o.count_) {
    o.ptr_ = nullptr;
    o.count_ = 0;
  }
  CudaBuffer& operator=(CudaBuffer&& o) noexcept {
    if (this != &o) {
      release();
      ptr_ = std::exchange(o.ptr_, nullptr);
      count_ = std::exchange(o.count_, 0);
    }
    return *this;
  }

  // Allocate once. A zero count yields a null pointer (valid empty buffer).
  void allocate(size_t count) {
    CUPIQP_REQUIRE(ptr_ == nullptr, "CudaBuffer::allocate called twice");
    count_ = count;
    if (count > 0) CUPIQP_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr_), count * sizeof(T)));
  }
  void zero_async(cudaStream_t s) {
    if (count_) CUPIQP_CUDA_CHECK(cudaMemsetAsync(ptr_, 0, count_ * sizeof(T), s));
  }
  void copy_from_host(const T* host, size_t count, cudaStream_t s) {
    CUPIQP_REQUIRE(count <= count_, "CudaBuffer::copy_from_host overflow");
    if (count) CUPIQP_CUDA_CHECK(cudaMemcpyAsync(ptr_, host, count * sizeof(T), cudaMemcpyHostToDevice, s));
  }
  void copy_from_device(const T* dev, size_t count, cudaStream_t s) {
    CUPIQP_REQUIRE(count <= count_, "CudaBuffer::copy_from_device overflow");
    if (count) CUPIQP_CUDA_CHECK(cudaMemcpyAsync(ptr_, dev, count * sizeof(T), cudaMemcpyDeviceToDevice, s));
  }

  T* data() { return ptr_; }
  const T* data() const { return ptr_; }
  size_t size() const { return count_; }
  bool empty() const { return count_ == 0; }

 private:
  void release() noexcept {
    if (ptr_) cudaFree(ptr_);
    ptr_ = nullptr;
    count_ = 0;
  }
  T* ptr_ = nullptr;
  size_t count_ = 0;
};

template <class T>
class CudaPinnedBuffer {
 public:
  CudaPinnedBuffer() = default;
  explicit CudaPinnedBuffer(size_t count) { allocate(count); }
  ~CudaPinnedBuffer() {
    if (ptr_) cudaFreeHost(ptr_);
  }
  CudaPinnedBuffer(const CudaPinnedBuffer&) = delete;
  CudaPinnedBuffer& operator=(const CudaPinnedBuffer&) = delete;
  CudaPinnedBuffer(CudaPinnedBuffer&& o) noexcept
      : ptr_(std::exchange(o.ptr_, nullptr)), count_(std::exchange(o.count_, 0)) {}
  CudaPinnedBuffer& operator=(CudaPinnedBuffer&& o) noexcept {
    if (this != &o) {
      if (ptr_) cudaFreeHost(ptr_);
      ptr_ = std::exchange(o.ptr_, nullptr);
      count_ = std::exchange(o.count_, 0);
    }
    return *this;
  }
  void allocate(size_t count) {
    CUPIQP_REQUIRE(ptr_ == nullptr, "CudaPinnedBuffer::allocate called twice");
    count_ = count;
    if (count > 0) CUPIQP_CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&ptr_), count * sizeof(T)));
  }
  T* data() { return ptr_; }
  const T* data() const { return ptr_; }
  T& operator[](size_t i) { return ptr_[i]; }
  const T& operator[](size_t i) const { return ptr_[i]; }
  size_t size() const { return count_; }

 private:
  T* ptr_ = nullptr;
  size_t count_ = 0;
};

}  // namespace cupiqp
