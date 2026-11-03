# IPM iteration profile (25 iterations, medians per iteration)

Total per iteration: **1510.5 us** (GPU span 1400.4 us + host round trip 111.1 us)

| Category | us / iteration | share |
|---|---|---|
| cuDSS factorization | 807.4 | 53.4% |
| cuDSS solve | 207.8 | 13.8% |
| IPM kernels | 127.7 | 8.5% |
| SpMV | 83.6 | 5.5% |
| KKT system kernels | 21.9 | 1.4% |
| memset | 13.1 | 0.9% |
| memcpy | 4.3 | 0.3% |
| graph condition | 0.9 | 0.1% |
| GPU idle inside the iteration graph | 132.6 | 8.8% |
| Host round trip (flag D2H + sync + next graph launch) | 111.1 | 7.4% |

## Largest idle gaps inside the iteration (us per iteration)

| us | after | before | note |
|---|---|---|---|
| 43.3 | set_condition_kernel | eliminate_slacks_kernel |  |
| 34.8 | csr_partition_kernel | [CUDA memset] |  |
| 26.9 | assemble_rhs_kernel | [CUDA memset] | stream switch |
| 5.4 | [CUDA memset] | [CUDA memset] |  |
| 3.9 | [CUDA memcpy Host-to-Device] | perm_ker |  |
| 3.0 | [CUDA memset] | csrmv_v3_transpose_kernel |  |
| 2.7 | [CUDA memset] | csrmv_v3_kernel |  |
| 2.0 | [CUDA memset] | [CUDA memcpy Host-to-Device] |  |
| 2.0 | [CUDA memcpy Device-to-Device] | split_sol_kernel | stream switch |
| 1.9 | [CUDA memcpy Host-to-Device] | [CUDA memcpy Host-to-Device] |  |

## Our kernels (per call)

| kernel | calls / iteration | median us | us / iteration | grid | block | regs |
|---|---|---|---|---|---|---|
| prepare_corrector_step_kernel | 1.0 | 33.59 | 33.6 | 1 | 256 | 48 |
| update_residual_nr_kernel | 1.0 | 29.03 | 29.0 | 1 | 256 | 80 |
| calculate_step_kernel | 1.0 | 23.59 | 23.0 | 1 | 256 | 48 |
| prepare_factor_and_predictor_step_kernel | 1.0 | 13.69 | 13.7 | 1 | 256 | 48 |
| update_residuals_r_kernel | 1.0 | 12.63 | 12.6 | 1 | 256 | 48 |
| solution_finite_kernel | 2.0 | 2.31 | 4.6 | 1 | 256 | 30 |
| calculate_mu_kernel | 1.0 | 4.05 | 4.1 | 1 | 256 | 64 |
| update_regularizations_kernel | 1.0 | 2.96 | 2.9 | 11 | 256 | 32 |
| eliminate_duals_kernel | 2.0 | 1.25 | 2.5 | 11 | 256 | 30 |
| recover_duals_kernel | 2.0 | 1.12 | 2.3 | 20 | 256 | 34 |
| split_sol_kernel | 2.0 | 1.09 | 2.2 | 11 | 256 | 16 |
| recover_slacks_kernel | 2.0 | 1.09 | 2.2 | 20 | 256 | 24 |
| eliminate_slacks_kernel | 2.0 | 1.06 | 2.1 | 20 | 256 | 20 |
| assemble_rhs_kernel | 2.0 | 0.96 | 1.9 | 11 | 256 | 16 |
| update_termination_kernel | 1.0 | 1.67 | 1.7 | 1 | 256 | 31 |
| update_vars_after_corrector_step_kernel | 1.0 | 1.45 | 1.4 | 41 | 256 | 21 |
| update_rho_delta_with_ineq_kernel | 1.0 | 1.25 | 1.3 | 1 | 256 | 34 |
| update_prox_vars_kernel | 1.0 | 1.22 | 1.2 | 21 | 256 | 16 |
| update_kkt_diag_kernel | 1.0 | 1.22 | 1.2 | 11 | 256 | 16 |
| update_continue_flag_kernel | 1.0 | 1.09 | 1.1 | 1 | 32 | 26 |
| prepare_zu_minus_zl_and_zbu_minus_zbl_kernel | 1.0 | 1.03 | 1.0 | 11 | 256 | 22 |
| any_flag_kernel | 1.0 | 0.93 | 0.9 | 1 | 32 | 26 |
| factor_retry_kernel | 1.0 | 0.90 | 0.9 | 1 | 256 | 22 |
| mark_failed_kernel | 1.0 | 0.87 | 0.9 | 1 | 256 | 16 |
| advance_iteration_kernel | 1.0 | 0.77 | 0.8 | 1 | 1 | 16 |
