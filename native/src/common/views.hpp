// Shared device-buffer views: the frozen contract between modules (M0).
//
// All views are plain structs of device pointers, cheap to pass by value into
// kernels. Per-problem arrays are row-major (B, k) with row stride k; milestone 1
// always has B == 1 but every kernel takes B. Layouts follow the reference
// (cupiqp/results.py Variables, cupiqp/settings.py DeviceSettings,
// cupiqp/typedef.py status codes). Do not change without the integration owner.
#pragma once

#include <cstdint>

namespace cupiqp {

// ---------------------------------------------------------------- constants
inline constexpr int32_t STATUS_UNSOLVED = -1;
inline constexpr int32_t STATUS_SOLVED = 0;
inline constexpr int32_t STATUS_MAX_ITER_REACHED = 1;
inline constexpr int32_t STATUS_PRIMAL_INFEASIBLE = 2;
inline constexpr int32_t STATUS_DUAL_INFEASIBLE = 3;
inline constexpr int32_t STATUS_NUMERICAL_ISSUES = 4;

// Index of each runtime setting in DeviceSettingsView::floats / ::ints
// (reference SettingsFloatIdx / SettingsIntIdx).
enum SettingsFloatIdx : int32_t {
  SF_RHO_INIT = 0,
  SF_DELTA_INIT = 1,
  SF_EPS_ABS = 2,
  SF_EPS_REL = 3,
  SF_EPS_DUALITY_GAP_ABS = 4,
  SF_EPS_DUALITY_GAP_REL = 5,
  SF_INFEASIBILITY_THRESHOLD = 6,
  SF_REG_LOWER_LIMIT = 7,
  SF_REG_FINETUNE_LOWER_LIMIT = 8,
  SF_TAU = 9,
  SF_IR_EPS_ABS = 10,
  SF_IR_EPS_REL = 11,
  SF_IR_MIN_IMPROVEMENT_RATE = 12,
  SF_COUNT = 13
};
enum SettingsIntIdx : int32_t {
  SI_CHECK_DUALITY_GAP = 0,
  SI_REG_FINETUNE_PRIMAL_UPDATE_THRESHOLD = 1,
  SI_REG_FINETUNE_DUAL_UPDATE_THRESHOLD = 2,
  SI_MAX_ITER = 3,
  SI_MAX_FACTOR_RETIRES = 4,
  SI_IR_MAX_ITER = 5,
  SI_COUNT = 6
};

// ---------------------------------------------------------------- dimensions
struct Dims {
  int32_t B = 1;
  int32_t n = 0;  // variables
  int32_t p = 0;  // equality rows
  int32_t m = 0;  // inequality rows (rows of G; 0 if G absent)
  // Bound sides present at setup: full length (m or n) when present, else 0.
  int32_t num_hl = 0, num_hu = 0, num_xl = 0, num_xu = 0;
  int32_t nnz_P = 0, nnz_A = 0, nnz_G = 0;
  bool has_h_l() const { return num_hl > 0; }
  bool has_h_u() const { return num_hu > 0; }
  bool has_x_l() const { return num_xl > 0; }
  bool has_x_u() const { return num_xu > 0; }
  // Packed inequality layout [hl | hu | xl | xu].
  int32_t num_ineq() const { return num_hl + num_hu + num_xl + num_xu; }
  int32_t off_hl() const { return 0; }
  int32_t off_hu() const { return num_hl; }
  int32_t off_xl() const { return num_hl + num_hu; }
  int32_t off_xu() const { return num_hl + num_hu + num_xl; }
  int32_t num_primal() const { return n + num_ineq(); }  // [x | s_all]
  int32_t num_dual() const { return p + num_ineq(); }    // [y | z_all]
  int32_t kkt_dim() const { return n + p + m; }
};

// ---------------------------------------------------------------- problem data
// Device CSR matrix with B value rows sharing one pattern.
template <class T>
struct CsrView {
  int32_t rows = 0, cols = 0, nnz = 0;
  const int32_t* indptr = nullptr;    // rows + 1
  const int32_t* indices = nullptr;   // nnz
  const int32_t* row_idx = nullptr;   // nnz, row of every stored entry (COO rows)
  T* values = nullptr;                // (B, nnz)
};

// SparseData (M1) in solver-owned storage. Scaled in place by M3.
// Vectors of absent sides are nullptr. Masks: 1.0 finite / 0.0 infinite.
template <class T>
struct SparseDataView {
  Dims dims;
  CsrView<T> P, A, G;              // A / G have rows == 0 when absent
  T* c = nullptr;                  // (B, n)
  T* b = nullptr;                  // (B, p)
  T* h_l = nullptr;                // (B, m) if has_h_l
  T* h_u = nullptr;                // (B, m) if has_h_u
  T* x_l = nullptr;                // (B, n) if has_x_l
  T* x_u = nullptr;                // (B, n) if has_x_u
  T* finite_mask_all = nullptr;    // (B, num_ineq), layout [hl | hu | xl | xu]
  T* active_G_row = nullptr;       // (B, m): 1 if the row has any finite bound
  T* active_x_bound = nullptr;     // (B, n): 1 if the variable has any finite box bound
  T* num_finite_bounds = nullptr;  // (B,)
  // Convenience sub-views of finite_mask_all (nullptr when the side is absent).
  T* finite_mask_hl() const { return dims.num_hl ? finite_mask_all + dims.off_hl() : nullptr; }
  T* finite_mask_hu() const { return dims.num_hu ? finite_mask_all + dims.off_hu() : nullptr; }
  T* finite_mask_xl() const { return dims.num_xl ? finite_mask_all + dims.off_xl() : nullptr; }
  T* finite_mask_xu() const { return dims.num_xu ? finite_mask_all + dims.off_xu() : nullptr; }
};

// ---------------------------------------------------------------- variables
// Reference Variables: primal [x | s_l | s_u | s_bl | s_bu], dual [y | z_l | z_u | z_bl | z_bu].
// With B > 1 the sub-arrays are strided column slices; with B == 1 they are contiguous.
template <class T>
struct VariablesView {
  Dims dims;
  T* primal = nullptr;  // (B, n + num_ineq)
  T* dual = nullptr;    // (B, p + num_ineq)
  int32_t primal_stride() const { return dims.num_primal(); }
  int32_t dual_stride() const { return dims.num_dual(); }
  T* x() const { return primal; }
  T* s_all() const { return primal + dims.n; }
  T* s_l() const { return primal + dims.n + dims.off_hl(); }
  T* s_u() const { return primal + dims.n + dims.off_hu(); }
  T* s_bl() const { return primal + dims.n + dims.off_xl(); }
  T* s_bu() const { return primal + dims.n + dims.off_xu(); }
  T* y() const { return dual; }
  T* z_all() const { return dual + dims.p; }
  T* z_l() const { return dual + dims.p + dims.off_hl(); }
  T* z_u() const { return dual + dims.p + dims.off_hu(); }
  T* z_bl() const { return dual + dims.p + dims.off_xl(); }
  T* z_bu() const { return dual + dims.p + dims.off_xu(); }
};

// ---------------------------------------------------------------- info
// Reference Result.info, structure of arrays, each (B,).
template <class T>
struct InfoView {
  T *rho, *delta, *mu, *sigma, *primal_step, *dual_step;
  T *primal_res, *primal_res_rel, *dual_res, *dual_res_rel;
  T *primal_res_reg, *primal_res_reg_rel, *dual_res_reg, *dual_res_reg_rel;
  T *primal_prox_inf, *dual_prox_inf, *prev_primal_res, *prev_dual_res;
  T *primal_obj, *dual_obj, *duality_gap, *duality_gap_rel, *reg_limit;
  int32_t *no_primal_update, *no_dual_update, *status, *iter, *factor_retries;
  int32_t* iter_total;  // (1,)
};

// ---------------------------------------------------------------- scaling
// Ruiz state (M3), reference RuizEquilibration attributes.
template <class T>
struct ScalingView {
  T* delta = nullptr;                      // (B, n + p + m): [D_x | D_y | D_z]
  T* delta_inv = nullptr;                  // (B, n + p + m)
  T* delta_b = nullptr;                    // (B, n)
  T* delta_b_inv = nullptr;                // (B, n)
  T* cost_scaling = nullptr;               // (B,)
  T* cost_scaling_inv = nullptr;           // (B,)
  T* x_b_scaling = nullptr;                // (B, n) = x_b_scaling_init * delta_b * D_x
  const T* x_b_scaling_init = nullptr;     // (B, n): data.active_x_bound (zeros without box bounds)
  T* dual_res_unscale_factor = nullptr;    // (B, n)
  T* primal_res_unscale_factor = nullptr;  // (B, p + num_ineq)
};

// ---------------------------------------------------------------- update mask
// Which blocks an update() replaces (non-null ValuesView entries).
struct UpdateMask {
  bool P = false, c = false, A = false, b = false, G = false;
  bool h_l = false, h_u = false, x_l = false, x_u = false;
  bool any_matrix() const { return P || A || G; }
  bool any_ineq_bound() const { return h_l || h_u; }
  bool any_box_bound() const { return x_l || x_u; }
  static UpdateMask all() { return {true, true, true, true, true, true, true, true, true}; }
};

// ---------------------------------------------------------------- settings
template <class T>
struct DeviceSettingsView {
  const T* floats = nullptr;       // SF_COUNT
  const int32_t* ints = nullptr;   // SI_COUNT
};

}  // namespace cupiqp
