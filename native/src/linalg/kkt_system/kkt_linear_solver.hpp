// Linear-solver backend of KktSystem: factor and solve the quasi-definite KKT
// matrix held in KktAssembly's values buffer. Two implementations:
//   CudssSolver       (linalg/cudss): sparse LDL^T, the reference backend;
//   ReducedKktSolver  (linalg/reduced_kkt): dense Schur complement on the x
//                     block, H = P + diag(x_reg) + [A; G]^T W [A; G], with a
//                     custom tiled Cholesky (Settings::kkt_backend = Reduced).
// Both read the KKT values at a fixed address, solve from rhs() into sol()
// (layout [x | y | z], length n + p + m), enqueue on the given stream only,
// allocate nothing after init and are capturable into CUDA graphs.
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
  // the backend does not report one (cuDSS: the outcome cannot be read on the
  // device without a host sync, as in the reference).
  virtual const int32_t* factor_status() const { return nullptr; }
};

}  // namespace cupiqp
