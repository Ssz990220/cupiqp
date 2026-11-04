// Error-check macros. Every CUDA runtime call outside kernels goes through
// one of these; failures throw cupiqp::Error with the call site.
#pragma once

#include <cuda_runtime_api.h>

#include <sstream>
#include <string>

#include "cupiqp/cupiqp.hpp"

namespace cupiqp::detail {

[[noreturn]] inline void throw_error(ErrorCode code, const char* expr, const char* file, int line,
                                     const std::string& detail) {
  std::ostringstream os;
  os << file << ":" << line << ": " << expr << " failed: " << detail;
  throw Error(code, os.str());
}

[[noreturn]] inline void throw_invalid(const std::string& what) {
  throw Error(ErrorCode::InvalidArgument, what);
}

}  // namespace cupiqp::detail

#define CUPIQP_CUDA_CHECK(expr)                                                              \
  do {                                                                                       \
    cudaError_t cupiqp_e_ = (expr);                                                          \
    if (cupiqp_e_ != cudaSuccess) {                                                          \
      ::cupiqp::detail::throw_error(::cupiqp::ErrorCode::Cuda, #expr, __FILE__, __LINE__,    \
                                    std::string(cudaGetErrorName(cupiqp_e_)) + " (" +        \
                                        cudaGetErrorString(cupiqp_e_) + ")");                \
    }                                                                                        \
  } while (0)

// Check the last kernel launch without synchronizing.
#define CUPIQP_CUDA_CHECK_LAUNCH() CUPIQP_CUDA_CHECK(cudaGetLastError())

#define CUPIQP_REQUIRE(cond, msg)                                                            \
  do {                                                                                       \
    if (!(cond)) ::cupiqp::detail::throw_invalid(msg);                                       \
  } while (0)
