# IPM iteration profile (25 iterations, medians per iteration)

Total per iteration: **1333.7 us** (GPU span 1223.9 us + host round trip 110.5 us)

| Category | us / iteration | share |
|---|---|---|
| cuDSS factorization | 709.7 | 53.2% |
| cuDSS solve | 245.2 | 18.4% |
| SpMV | 57.0 | 4.3% |
| IPM kernels | 43.7 | 3.3% |
| KKT system kernels | 22.9 | 1.7% |
| memset | 13.0 | 1.0% |
| memcpy | 4.4 | 0.3% |
| graph condition | 0.9 | 0.1% |
| GPU idle inside the iteration graph | 126.4 | 9.5% |
| Host round trip (flag D2H + sync + next graph launch) | 110.5 | 8.3% |

## Largest idle gaps inside the iteration (us per iteration)

| us | after | before | note |
|---|---|---|---|
| 38.3 | set_condition_kernel | eliminate_slacks_kernel |  |
| 35.8 | csr_partition_kernel | [CUDA memset] |  |
| 28.1 | assemble_rhs_kernel | [CUDA memset] | stream switch |
| 4.0 | [CUDA memcpy Host-to-Device] | perm_ker |  |
| 3.7 | [CUDA memset] | [CUDA memset] |  |
| 3.3 | [CUDA memset] | csrmv_v3_transpose_kernel |  |
| 2.7 | [CUDA memset] | csrmv_v3_kernel |  |
| 2.0 | [CUDA memcpy Device-to-Device] | split_sol_kernel | stream switch |
| 1.9 | [CUDA memset] | [CUDA memcpy Host-to-Device] |  |
| 1.9 | [CUDA memcpy Host-to-Device] | [CUDA memcpy Host-to-Device] |  |

## Our kernels (per call)

| kernel | calls / iteration | median us | us / iteration | grid | block | regs |
|---|---|---|---|---|---|---|
| calculate_step_kernel | 2.0 | 4.77 | 9.5 | 1 | 128 | 48 |
| update_residual_nr_kernel | 1.0 | 7.16 | 7.1 | 1 | 128 | 96 |
| solution_finite_kernel | 2.0 | 2.48 | 5.0 | 1 | 128 | 40 |
| update_residuals_r_kernel | 1.0 | 4.58 | 4.6 | 1 | 128 | 40 |
| prepare_factor_and_predictor_step_kernel | 1.0 | 3.83 | 3.8 | 1 | 128 | 40 |
| corrector_sigma_kernel | 1.0 | 3.19 | 3.2 | 1 | 128 | 44 |
| update_regularizations_kernel | 1.0 | 3.06 | 3.1 | 26 | 256 | 32 |
| calculate_mu_kernel | 1.0 | 2.84 | 2.8 | 1 | 128 | 40 |
| eliminate_duals_kernel | 2.0 | 1.26 | 2.5 | 26 | 256 | 30 |
| recover_duals_kernel | 2.0 | 1.19 | 2.4 | 50 | 256 | 34 |
| recover_slacks_kernel | 2.0 | 1.19 | 2.4 | 50 | 256 | 24 |
| eliminate_slacks_kernel | 2.0 | 1.13 | 2.3 | 50 | 256 | 20 |
| split_sol_kernel | 2.0 | 1.10 | 2.2 | 27 | 256 | 16 |
| assemble_rhs_kernel | 2.0 | 0.97 | 1.9 | 27 | 256 | 16 |
| update_termination_kernel | 1.0 | 1.68 | 1.7 | 1 | 256 | 31 |
| update_vars_after_corrector_step_kernel | 1.0 | 1.51 | 1.5 | 101 | 256 | 21 |
| corrector_rhs_kernel | 1.0 | 1.32 | 1.3 | 50 | 256 | 26 |
| update_rho_delta_with_ineq_kernel | 1.0 | 1.26 | 1.3 | 1 | 256 | 34 |
| update_prox_vars_kernel | 1.0 | 1.26 | 1.3 | 52 | 256 | 16 |
| update_kkt_diag_kernel | 1.0 | 1.16 | 1.2 | 27 | 256 | 16 |
| update_continue_flag_kernel | 1.0 | 1.10 | 1.1 | 1 | 32 | 26 |
| prepare_zu_minus_zl_and_zbu_minus_zbl_kernel | 1.0 | 1.03 | 1.0 | 26 | 256 | 22 |
| any_flag_kernel | 1.0 | 0.97 | 1.0 | 1 | 32 | 26 |
| factor_retry_kernel | 1.0 | 0.93 | 0.9 | 1 | 256 | 22 |
| mark_failed_kernel | 1.0 | 0.87 | 0.9 | 1 | 256 | 16 |
| advance_iteration_kernel | 1.0 | 0.77 | 0.8 | 1 | 1 | 16 |
