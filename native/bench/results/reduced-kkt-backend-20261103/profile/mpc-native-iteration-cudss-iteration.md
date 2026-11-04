# IPM iteration profile (25 iterations, medians per iteration)

Total per iteration: **1407.4 us** (GPU span 1294.5 us + host round trip 112.3 us)

| Category | us / iteration | share |
|---|---|---|
| cuDSS factorization | 794.2 | 56.4% |
| cuDSS solve | 205.3 | 14.6% |
| SpMV | 81.8 | 5.8% |
| IPM kernels | 41.0 | 2.9% |
| KKT system kernels | 21.7 | 1.5% |
| memset | 13.0 | 0.9% |
| memcpy | 4.2 | 0.3% |
| graph condition | 0.9 | 0.1% |
| GPU idle inside the iteration graph | 133.0 | 9.4% |
| Host round trip (flag D2H + sync + next graph launch) | 112.3 | 8.0% |

## Largest idle gaps inside the iteration (us per iteration)

| us | after | before | note |
|---|---|---|---|
| 40.8 | set_condition_kernel | eliminate_slacks_kernel |  |
| 39.3 | csr_partition_kernel | [CUDA memset] |  |
| 28.0 | assemble_rhs_kernel | [CUDA memset] | stream switch |
| 4.4 | [CUDA memset] | [CUDA memset] |  |
| 3.9 | [CUDA memcpy Host-to-Device] | perm_ker |  |
| 3.0 | [CUDA memset] | csrmv_v3_transpose_kernel |  |
| 2.7 | [CUDA memset] | csrmv_v3_kernel |  |
| 2.0 | [CUDA memset] | [CUDA memcpy Host-to-Device] |  |
| 1.9 | [CUDA memcpy Device-to-Device] | split_sol_kernel | stream switch |
| 1.8 | [CUDA memcpy Host-to-Device] | [CUDA memcpy Host-to-Device] |  |

## Our kernels (per call)

| kernel | calls / iteration | median us | us / iteration | grid | block | regs |
|---|---|---|---|---|---|---|
| calculate_step_kernel | 2.0 | 4.52 | 9.0 | 1 | 128 | 48 |
| update_residual_nr_kernel | 1.0 | 6.28 | 6.3 | 1 | 128 | 96 |
| solution_finite_kernel | 2.0 | 2.35 | 4.7 | 1 | 128 | 40 |
| update_residuals_r_kernel | 1.0 | 4.17 | 4.2 | 1 | 128 | 40 |
| prepare_factor_and_predictor_step_kernel | 1.0 | 3.71 | 3.7 | 1 | 128 | 40 |
| corrector_sigma_kernel | 1.0 | 3.04 | 3.0 | 1 | 128 | 44 |
| update_regularizations_kernel | 1.0 | 2.89 | 2.9 | 11 | 256 | 32 |
| calculate_mu_kernel | 1.0 | 2.67 | 2.7 | 1 | 128 | 40 |
| eliminate_duals_kernel | 2.0 | 1.24 | 2.5 | 11 | 256 | 30 |
| recover_duals_kernel | 2.0 | 1.13 | 2.2 | 20 | 256 | 34 |
| recover_slacks_kernel | 2.0 | 1.07 | 2.2 | 20 | 256 | 24 |
| split_sol_kernel | 2.0 | 1.07 | 2.1 | 11 | 256 | 16 |
| eliminate_slacks_kernel | 2.0 | 1.04 | 2.1 | 20 | 256 | 20 |
| assemble_rhs_kernel | 2.0 | 0.94 | 1.9 | 11 | 256 | 16 |
| update_termination_kernel | 1.0 | 1.66 | 1.6 | 1 | 256 | 31 |
| update_vars_after_corrector_step_kernel | 1.0 | 1.35 | 1.4 | 41 | 256 | 21 |
| update_rho_delta_with_ineq_kernel | 1.0 | 1.23 | 1.3 | 1 | 256 | 34 |
| update_prox_vars_kernel | 1.0 | 1.19 | 1.2 | 21 | 256 | 16 |
| update_kkt_diag_kernel | 1.0 | 1.19 | 1.2 | 11 | 256 | 16 |
| corrector_rhs_kernel | 1.0 | 1.16 | 1.2 | 20 | 256 | 26 |
| update_continue_flag_kernel | 1.0 | 1.07 | 1.1 | 1 | 32 | 26 |
| prepare_zu_minus_zl_and_zbu_minus_zbl_kernel | 1.0 | 1.00 | 1.0 | 11 | 256 | 22 |
| any_flag_kernel | 1.0 | 0.91 | 0.9 | 1 | 32 | 26 |
| factor_retry_kernel | 1.0 | 0.91 | 0.9 | 1 | 256 | 22 |
| mark_failed_kernel | 1.0 | 0.85 | 0.8 | 1 | 256 | 16 |
| advance_iteration_kernel | 1.0 | 0.75 | 0.8 | 1 | 1 | 16 |
