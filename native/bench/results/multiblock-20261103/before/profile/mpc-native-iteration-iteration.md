# IPM iteration profile (25 iterations, medians per iteration)

Total per iteration: **1480.3 us** (GPU span 1368.8 us + host round trip 110.3 us)

| Category | us / iteration | share |
|---|---|---|
| cuDSS factorization | 789.4 | 53.3% |
| cuDSS solve | 204.8 | 13.8% |
| IPM kernels | 125.2 | 8.5% |
| SpMV | 81.9 | 5.5% |
| KKT system kernels | 21.4 | 1.4% |
| memset | 12.7 | 0.9% |
| memcpy | 4.2 | 0.3% |
| graph condition | 0.9 | 0.1% |
| GPU idle inside the iteration graph | 129.5 | 8.7% |
| Host round trip (flag D2H + sync + next graph launch) | 110.3 | 7.4% |

## Largest idle gaps inside the iteration (us per iteration)

| us | after | before | note |
|---|---|---|---|
| 39.7 | set_condition_kernel | eliminate_slacks_kernel |  |
| 35.4 | csr_partition_kernel | [CUDA memset] |  |
| 28.4 | assemble_rhs_kernel | [CUDA memset] | stream switch |
| 3.9 | [CUDA memcpy Host-to-Device] | perm_ker |  |
| 3.3 | [CUDA memset] | [CUDA memset] |  |
| 2.9 | [CUDA memset] | csrmv_v3_transpose_kernel |  |
| 2.6 | [CUDA memset] | csrmv_v3_kernel |  |
| 1.9 | [CUDA memcpy Device-to-Device] | split_sol_kernel | stream switch |
| 1.9 | [CUDA memcpy Host-to-Device] | [CUDA memcpy Host-to-Device] |  |
| 1.8 | [CUDA memset] | [CUDA memcpy Host-to-Device] |  |

## Our kernels (per call)

| kernel | calls / iteration | median us | us / iteration | grid | block | regs |
|---|---|---|---|---|---|---|
| prepare_corrector_step_kernel | 1.0 | 32.92 | 32.9 | 1 | 256 | 48 |
| update_residual_nr_kernel | 1.0 | 28.45 | 28.4 | 1 | 256 | 80 |
| calculate_step_kernel | 1.0 | 23.13 | 22.5 | 1 | 256 | 48 |
| prepare_factor_and_predictor_step_kernel | 1.0 | 13.44 | 13.4 | 1 | 256 | 48 |
| update_residuals_r_kernel | 1.0 | 12.37 | 12.4 | 1 | 256 | 48 |
| solution_finite_kernel | 2.0 | 2.23 | 4.5 | 1 | 256 | 30 |
| calculate_mu_kernel | 1.0 | 3.97 | 4.0 | 1 | 256 | 64 |
| update_regularizations_kernel | 1.0 | 2.90 | 2.9 | 11 | 256 | 32 |
| eliminate_duals_kernel | 2.0 | 1.23 | 2.4 | 11 | 256 | 30 |
| recover_duals_kernel | 2.0 | 1.10 | 2.2 | 20 | 256 | 34 |
| split_sol_kernel | 2.0 | 1.07 | 2.2 | 11 | 256 | 16 |
| recover_slacks_kernel | 2.0 | 1.07 | 2.1 | 20 | 256 | 24 |
| eliminate_slacks_kernel | 2.0 | 1.04 | 2.1 | 20 | 256 | 20 |
| assemble_rhs_kernel | 2.0 | 0.94 | 1.9 | 11 | 256 | 16 |
| update_termination_kernel | 1.0 | 1.64 | 1.6 | 1 | 256 | 31 |
| update_vars_after_corrector_step_kernel | 1.0 | 1.42 | 1.4 | 41 | 256 | 21 |
| update_rho_delta_with_ineq_kernel | 1.0 | 1.23 | 1.3 | 1 | 256 | 34 |
| update_prox_vars_kernel | 1.0 | 1.20 | 1.2 | 21 | 256 | 16 |
| update_kkt_diag_kernel | 1.0 | 1.20 | 1.2 | 11 | 256 | 16 |
| update_continue_flag_kernel | 1.0 | 1.07 | 1.1 | 1 | 32 | 26 |
| prepare_zu_minus_zl_and_zbu_minus_zbl_kernel | 1.0 | 1.01 | 1.0 | 11 | 256 | 22 |
| any_flag_kernel | 1.0 | 0.91 | 0.9 | 1 | 32 | 26 |
| factor_retry_kernel | 1.0 | 0.88 | 0.9 | 1 | 256 | 22 |
| mark_failed_kernel | 1.0 | 0.85 | 0.8 | 1 | 256 | 16 |
| advance_iteration_kernel | 1.0 | 0.76 | 0.8 | 1 | 1 | 16 |
