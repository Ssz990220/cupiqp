// M6 kkt_system: launchers of the condensed KKT kernels (forward direction).
//
// Version 1 ports of the Warp kernels in cupiqp/kkt_systems_kernels.py
// (update_regularizations, eliminate_slacks, eliminate_duals, recover_duals
// with box_rhs_eliminated = false, recover_slacks, condensed_kkt_combine and
// the five refinement kernels), the finite check of cupiqp/solver_kernels.py
// (create_solution_finite_kernel) and the fused rhs assembly / split that
// replace the three wp.copy calls each of SparseKKTSolver.solve.
//
// Thread mapping follows the reference: a launch of Warp dim (B, W) becomes a
// 1-D grid over B * W threads with b = tid / W, t = tid % W. The finite check
// is a per-problem reduction (reference wp.launch_tiled with
// reduction_block_dim(n + p + m)), split over several blocks per problem
// when a ReductionWorkspace is given and B is small.
//
// Every per-problem array is passed as a RowMat (pointer + row stride), so the
// strided column views of Variables (B > 1) need no copies. With B == 1 the
// stride is never used. Every launcher enqueues on `s` only; a launcher with
// no work (zero threads) launches nothing.
#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>

#include "common/multiblock.hpp"
#include "common/views.hpp"

namespace cupiqp {

// Row-major (B, k) view with row stride `ld`.
template <class T>
struct RowMat {
  T* ptr = nullptr;
  int64_t ld = 0;
  __host__ __device__ T& operator()(int64_t b, int64_t i) const { return ptr[b * ld + i]; }
};
using DMat = RowMat<double>;
using CDMat = RowMat<const double>;

namespace kkt_kernels {

// Reference create_update_regularizations_kernel(n, m, has_*), dim (B, n + m).
// Inputs: s_all, z_all, finite_mask_all (B, num_ineq); x_b_scaling (B, n);
// rho_in, delta_in (B,). Outputs: m_s_all, m_z_inv_all, w_delta_inv_all
// (B, num_ineq); rho_out, delta_out (B,); x_reg (B, n); z_reg, z_reg_inv (B, m).
void update_regularizations(const Dims& d, CDMat s_all, CDMat z_all, CDMat finite_mask_all, CDMat x_b_scaling,
                            const double* rho_in, const double* delta_in, DMat m_s_all, DMat m_z_inv_all,
                            DMat w_delta_inv_all, double* rho_out, double* delta_out, DMat x_reg, DMat z_reg,
                            DMat z_reg_inv, cudaStream_t s);

// Reference create_eliminate_slacks_kernel, dim (B, num_ineq):
// out = -m_z_inv * rhs_s + rhs_z.
void eliminate_slacks(int32_t B, int32_t num_ineq, CDMat rhs_z_all, CDMat rhs_s_all, CDMat m_z_inv_all,
                      DMat updated_rhs_z_all, cudaStream_t s);

// Reference create_eliminate_duals_kernel(n, m, has_*), dim (B, n + m). The
// w_* / rhs_z_* blocks are column slices of w_delta_inv_all /
// updated_rhs_z_all ([hl | hu | xl | xu]); an absent side is never read.
void eliminate_duals(const Dims& d, CDMat rhs_x, CDMat w_bu_delta_inv, CDMat w_bl_delta_inv, CDMat x_b_scaling,
                     CDMat rhs_z_bu, CDMat rhs_z_bl, DMat rhs_x_updated, CDMat w_u_delta_inv,
                     CDMat w_l_delta_inv, CDMat rhs_z_u, CDMat rhs_z_l, CDMat z_reg, DMat rhs_z_updated,
                     cudaStream_t s);

// Reference create_recover_duals_kernel(num_hu, num_hl, num_xu, num_xl,
// box_rhs_eliminated=False), dim (B, num_ineq).
void recover_duals(const Dims& d, CDMat G_dx, CDMat lhs_x, CDMat w_u_delta_inv, CDMat rhs_z_u, DMat lhs_z_u,
                   CDMat w_l_delta_inv, CDMat rhs_z_l, DMat lhs_z_l, CDMat w_bu_delta_inv, CDMat m_z_bu_inv,
                   CDMat rhs_z_bu, CDMat rhs_s_bu, DMat lhs_z_bu, CDMat w_bl_delta_inv, CDMat m_z_bl_inv,
                   CDMat rhs_z_bl, CDMat rhs_s_bl, DMat lhs_z_bl, CDMat x_b_scaling, cudaStream_t s);

// Reference create_recover_slacks_kernel, dim (B, num_ineq):
// lhs_s = m_z_inv * ((-m_s) * lhs_z + rhs_s).
void recover_slacks(int32_t B, int32_t num_ineq, CDMat rhs_s_all, CDMat lhs_z_all, CDMat m_s_all,
                    CDMat m_z_inv_all, DMat lhs_s_all, cudaStream_t s);

// Reference create_condensed_kkt_combine_kernel, dim (B, n + p + m). With
// residual the kernel writes rhs - K lhs and folds max |.| into norm_out[0]
// (Warp atomic_max semantics: CAS loop with fmax, so a NaN entry is ignored).
void condensed_kkt_combine(int32_t B, int32_t n, int32_t p, int32_t m, CDMat P_x, CDMat x_reg, CDMat lhs_x,
                           CDMat AT_y, CDMat GT_z, CDMat A_x, CDMat lhs_y, const double* delta, CDMat G_x,
                           CDMat active_G_row, CDMat z_reg, CDMat lhs_z, CDMat rhs_x, CDMat rhs_y, CDMat rhs_z,
                           bool residual, DMat out_x, DMat out_y, DMat out_z, double* norm_out, cudaStream_t s);

// wp.map(wp.mul, a, b, out=out), dim (B, k): out = a * b (out may alias a).
void mul(int32_t B, int32_t k, CDMat a, CDMat b, DMat out, cudaStream_t s);

// Reference refinement kernels (dim 1 or (B, n + p + m)).
void refinement_begin(const int32_t* settings_i, double* rhs_norm, double* new_error, double* error, int32_t* step,
                      int32_t* loop, cudaStream_t s);
void refinement_decide(const double* settings_f, const int32_t* settings_i, const double* rhs_norm,
                       double* new_error, double* error, int32_t* step, int32_t* accept, int32_t* correct,
                       int32_t* loop, cudaStream_t s);
void refinement_accept(int32_t B, int32_t n, int32_t p, int32_t m, const int32_t* accept, CDMat cand_x,
                       CDMat cand_y, CDMat cand_z, DMat lhs_x, DMat lhs_y, DMat lhs_z, cudaStream_t s);
void refinement_rhs_norm(int32_t B, int32_t n, int32_t p, int32_t m, CDMat rhs_x, CDMat rhs_y, CDMat rhs_z,
                         double* rhs_norm, cudaStream_t s);
void refinement_add(int32_t B, int32_t n, int32_t p, int32_t m, CDMat lhs_x, CDMat lhs_y, CDMat lhs_z,
                    DMat cand_x, DMat cand_y, DMat cand_z, cudaStream_t s);

// Reference create_solution_finite_kernel: status[b] = 1 if any entry of
// row b of x, y or z is not finite, else 0. One block per problem, or
// several (multi-block reduction, common/multiblock.hpp) when a workspace is
// given and B is small.
void solution_finite(int32_t B, int32_t n, int32_t p, int32_t m, CDMat x, CDMat y, CDMat z, int32_t* status,
                     cudaStream_t s, const ReductionWorkspace& ws = ReductionWorkspace{});

// Fused [x | y | z] -> out (B, n + p + m) (replaces three wp.copy calls).
void assemble_rhs(int32_t B, int32_t n, int32_t p, int32_t m, CDMat x, CDMat y, CDMat z, DMat out,
                  cudaStream_t s);
// Fused in (B, n + p + m) -> [x | y | z] (replaces three wp.copy calls).
void split_sol(int32_t B, int32_t n, int32_t p, int32_t m, CDMat in, DMat x, DMat y, DMat z, cudaStream_t s);

}  // namespace kkt_kernels
}  // namespace cupiqp
