// Public API pieces that do not depend on the solver implementation.
#include <cudss.h>

#include <limits>
#include <string>

#include "core/check.hpp"
#include "cupiqp/cupiqp.hpp"

namespace cupiqp {

const char* status_name(Status s) {
  switch (s) {
    case Status::Unsolved: return "CUPIQP_UNSOLVED";
    case Status::Solved: return "CUPIQP_SOLVED";
    case Status::MaxIterReached: return "CUPIQP_MAX_ITER_REACHED";
    case Status::PrimalInfeasible: return "CUPIQP_PRIMAL_INFEASIBLE";
    case Status::DualInfeasible: return "CUPIQP_DUAL_INFEASIBLE";
    case Status::NumericalIssues: return "CUPIQP_NUMERICAL_ISSUES";
  }
  return "CUPIQP_UNKNOWN";
}

bool Settings::verify() const {
  // Reference Settings.verify_settings() (fields that exist for FP64 solves).
  return rho_init > 0 && delta_init > 0 && eps_abs > 0 && eps_rel >= 0 &&
         eps_duality_gap_abs > 0 && eps_duality_gap_rel >= 0 && infeasibility_threshold >= 0 &&
         reg_lower_limit > 0 && reg_finetune_primal_update_threshold >= 0 &&
         reg_finetune_dual_update_threshold >= 0 && max_iter > 0 && max_factor_retires > 0 &&
         preconditioner_iter >= 0 && tau > 0 && tau <= 1 && iterative_refinement_eps_abs > 0 &&
         iterative_refinement_eps_rel >= 0 && iterative_refinement_max_iter >= 0 &&
         iterative_refinement_min_improvement_rate >= 1.0 &&
         iterative_refinement_static_regularization_eps > 0 &&
         iterative_refinement_static_regularization_rel >= 0;
}

CsrPattern CsrPattern::from_int64(int64_t rows, int64_t cols, const int64_t* indptr,
                                  const int64_t* indices) {
  constexpr int64_t kMax = std::numeric_limits<int32_t>::max();
  CUPIQP_REQUIRE(rows >= 0 && cols >= 0 && rows <= kMax && cols <= kMax,
                 "CSR shape does not fit in int32");
  CsrPattern out;
  out.rows = static_cast<int32_t>(rows);
  out.cols = static_cast<int32_t>(cols);
  const int64_t nnz = rows >= 0 ? indptr[rows] : 0;
  CUPIQP_REQUIRE(nnz < (int64_t(1) << 31),
                 "CSR has " + std::to_string(nnz) + " stored entries; at most 2**31 - 1 are supported.");
  out.indptr.resize(static_cast<size_t>(rows) + 1);
  for (int64_t i = 0; i <= rows; ++i) {
    CUPIQP_REQUIRE(indptr[i] >= 0 && indptr[i] <= kMax, "CSR indptr entry out of int32 range");
    out.indptr[i] = static_cast<int32_t>(indptr[i]);
  }
  out.indices.resize(static_cast<size_t>(nnz));
  for (int64_t k = 0; k < nnz; ++k) {
    CUPIQP_REQUIRE(indices[k] >= 0 && indices[k] <= kMax, "CSR column index out of int32 range");
    out.indices[k] = static_cast<int32_t>(indices[k]);
  }
  return out;
}

std::string version_string() {
  int major = 0, minor = 0, patch = 0;
  cudssGetProperty(MAJOR_VERSION, &major);
  cudssGetProperty(MINOR_VERSION, &minor);
  cudssGetProperty(PATCH_LEVEL, &patch);
  return "0.1.0 (cuDSS " + std::to_string(major) + "." + std::to_string(minor) + "." +
         std::to_string(patch) + ")";
}

}  // namespace cupiqp
