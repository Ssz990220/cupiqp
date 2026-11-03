// M8 placeholder: replaced during integration (wave 3).
#include "core/check.hpp"
#include "cupiqp/cupiqp.hpp"

namespace cupiqp {

class SparseSolverImpl {};

SparseSolver::SparseSolver(cudaStream_t) {
  throw Error(ErrorCode::Unsupported, "SparseSolver is not integrated yet (M8 pending)");
}
SparseSolver::~SparseSolver() = default;
SparseSolver::SparseSolver(SparseSolver&&) noexcept = default;
SparseSolver& SparseSolver::operator=(SparseSolver&&) noexcept = default;

}  // namespace cupiqp
