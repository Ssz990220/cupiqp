# IPM iteration profile (25 iterations, medians per iteration)

Total per iteration: **1436.7 us** (GPU span 1321.8 us + host round trip 109.5 us)

| Category | us / iteration | share |
|---|---|---|
| cuDSS factorization | 813.6 | 56.6% |
| cuDSS solve | 211.3 | 14.7% |
| SpMV | 84.2 | 5.9% |
| IPM kernels | 42.1 | 2.9% |
| KKT system kernels | 22.3 | 1.6% |
| memset | 13.3 | 0.9% |
| memcpy | 4.3 | 0.3% |
| graph condition | 0.9 | 0.1% |
| GPU idle inside the iteration graph | 135.3 | 9.4% |
| Host round trip (flag D2H + sync + next graph launch) | 109.5 | 7.6% |

## Largest idle gaps inside the iteration (us per iteration)

| us | after | before | note |
|---|---|---|---|
| 41.6 | set_condition_kernel | eliminate_slacks_kernel |  |
| 36.1 | csr_partition_kernel | [CUDA memset] |  |
| 28.2 | assemble_rhs_kernel | [CUDA memset] | stream switch |
| 4.7 | [CUDA memset] | [CUDA memset] |  |
| 4.1 | [CUDA memcpy Host-to-Device] | perm_ker |  |
| 3.0 | [CUDA memset] | csrmv_v3_transpose_kernel |  |
| 2.8 | [CUDA memset] | csrmv_v3_kernel |  |
| 2.0 | [CUDA memset] | [CUDA memcpy Host-to-Device] |  |
| 2.0 | [CUDA memcpy Device-to-Device] | split_sol_kernel | stream switch |
| 1.9 | [CUDA memcpy Host-to-Device] | [CUDA memcpy Host-to-Device] |  |

## Our kernels (per call)

| kernel | calls / iteration | median us | us / iteration | grid | block | regs |
|---|---|---|---|---|---|---|
| calculate_step_kernel | 2.0 | 4.64 | 9.2 | 1 | 128 | 48 |
| update_residual_nr_kernel | 1.0 | 6.50 | 6.5 | 1 | 128 | 96 |
| solution_finite_kernel | 2.0 | 2.42 | 4.8 | 1 | 128 | 40 |
| update_residuals_r_kernel | 1.0 | 4.28 | 4.3 | 1 | 128 | 40 |
| prepare_factor_and_predictor_step_kernel | 1.0 | 3.80 | 3.8 | 1 | 128 | 40 |
| corrector_sigma_kernel | 1.0 | 3.12 | 3.1 | 1 | 128 | 44 |
| update_regularizations_kernel | 1.0 | 2.96 | 3.0 | 11 | 256 | 32 |
| calculate_mu_kernel | 1.0 | 2.74 | 2.7 | 1 | 128 | 40 |
| eliminate_duals_kernel | 2.0 | 1.26 | 2.5 | 11 | 256 | 30 |
| recover_duals_kernel | 2.0 | 1.16 | 2.3 | 20 | 256 | 34 |
| split_sol_kernel | 2.0 | 1.09 | 2.2 | 11 | 256 | 16 |
| recover_slacks_kernel | 2.0 | 1.09 | 2.2 | 20 | 256 | 24 |
| eliminate_slacks_kernel | 2.0 | 1.06 | 2.1 | 20 | 256 | 20 |
| assemble_rhs_kernel | 2.0 | 0.97 | 1.9 | 11 | 256 | 16 |
| update_termination_kernel | 1.0 | 1.71 | 1.7 | 1 | 256 | 31 |
| update_vars_after_corrector_step_kernel | 1.0 | 1.42 | 1.4 | 41 | 256 | 21 |
| update_rho_delta_with_ineq_kernel | 1.0 | 1.26 | 1.3 | 1 | 256 | 34 |
| update_prox_vars_kernel | 1.0 | 1.22 | 1.2 | 21 | 256 | 16 |
| update_kkt_diag_kernel | 1.0 | 1.22 | 1.2 | 11 | 256 | 16 |
| corrector_rhs_kernel | 1.0 | 1.19 | 1.2 | 20 | 256 | 26 |
| update_continue_flag_kernel | 1.0 | 1.09 | 1.1 | 1 | 32 | 26 |
| prepare_zu_minus_zl_and_zbu_minus_zbl_kernel | 1.0 | 1.03 | 1.0 | 11 | 256 | 22 |
| any_flag_kernel | 1.0 | 0.97 | 1.0 | 1 | 32 | 26 |
| factor_retry_kernel | 1.0 | 0.90 | 0.9 | 1 | 256 | 22 |
| mark_failed_kernel | 1.0 | 0.87 | 0.9 | 1 | 256 | 16 |
| advance_iteration_kernel | 1.0 | 0.77 | 0.8 | 1 | 1 | 16 |
