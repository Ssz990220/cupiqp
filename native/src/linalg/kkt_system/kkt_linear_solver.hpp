// Linear-solver interface of KktSystem: factor and solve the quasi-definite
// KKT matrix held in KktAssembly's values buffer. The implementation is
// ReducedKktSolver (linalg/reduced_kkt): dense Schur complement on the x
// block, H = P + diag(x_reg) + [A; G]^T W [A; G], with a custom tiled
// Cholesky; tests may substitute their own. An implementation reads the KKT
// values at a fixed address, solves from rhs() into sol() (layout
// [x | y | z], length n + p + m), enqueues on the given stream only,
// allocates nothing after init and is capturable into CUDA graphs.
#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>

namespace cupiqp {

class KktLinearSolver {
 public:
  virtual ~KktLinearSolver() = default;
  virtual double* rhs() = 0;  // (dim,), stable address
  virtual double* sol() = 0;  // (dim,), stable address
  // Numeric factorization of the current KKT values.
  virtual void factor(cudaStream_t s) = 0;
  // sol = K^-1 rhs with the last factorization.
  virtual void solve(cudaStream_t s) = 0;
  // Device int32 (B,) outcome of the last factor(): 0 on success, nonzero
  // when the matrix was found not numerically positive definite. nullptr when
  // the implementation does not report one (KktSystem then reports 0).
  virtual const int32_t* factor_status() const { return nullptr; }
};

}  // namespace cupiqp
