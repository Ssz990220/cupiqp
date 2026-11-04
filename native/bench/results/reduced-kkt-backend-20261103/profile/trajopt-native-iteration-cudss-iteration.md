# IPM iteration profile (25 iterations, medians per iteration)

Total per iteration: **1316.2 us** (GPU span 1207.2 us + host round trip 108.3 us)

| Category | us / iteration | share |
|---|---|---|
| cuDSS factorization | 699.3 | 53.1% |
| cuDSS solve | 243.7 | 18.5% |
| SpMV | 56.6 | 4.3% |
| IPM kernels | 43.5 | 3.3% |
| KKT system kernels | 22.7 | 1.7% |
| memset | 12.9 | 1.0% |
| memcpy | 4.4 | 0.3% |
| graph condition | 1.0 | 0.1% |
| GPU idle inside the iteration graph | 123.8 | 9.4% |
| Host round trip (flag D2H + sync + next graph launch) | 108.3 | 8.2% |

## Largest idle gaps inside the iteration (us per iteration)

| us | after | before | note |
|---|---|---|---|
| 37.1 | set_condition_kernel | eliminate_slacks_kernel |  |
| 35.9 | csr_partition_kernel | [CUDA memset] |  |
| 26.2 | assemble_rhs_kernel | [CUDA memset] | stream switch |
| 4.0 | [CUDA memcpy Host-to-Device] | perm_ker |  |
| 3.8 | [CUDA memset] | [CUDA memset] |  |
| 3.0 | [CUDA memset] | csrmv_v3_transpose_kernel |  |
| 2.7 | [CUDA memset] | csrmv_v3_kernel |  |
| 2.0 | [CUDA memcpy Device-to-Device] | split_sol_kernel | stream switch |
| 1.9 | [CUDA memset] | [CUDA memcpy Host-to-Device] |  |
| 1.9 | [CUDA memcpy Host-to-Device] | [CUDA memcpy Host-to-Device] |  |

## Our kernels (per call)

| kernel | calls / iteration | median us | us / iteration | grid | block | regs |
|---|---|---|---|---|---|---|
| calculate_step_kernel | 2.0 | 4.73 | 9.4 | 1 | 128 | 48 |
| update_residual_nr_kernel | 1.0 | 7.13 | 7.1 | 1 | 128 | 96 |
| solution_finite_kernel | 2.0 | 2.50 | 5.0 | 1 | 128 | 40 |
| update_residuals_r_kernel | 1.0 | 4.51 | 4.5 | 1 | 128 | 40 |
| prepare_factor_and_predictor_step_kernel | 1.0 | 3.87 | 3.9 | 1 | 128 | 40 |
| corrector_sigma_kernel | 1.0 | 3.13 | 3.2 | 1 | 128 | 44 |
| update_regularizations_kernel | 1.0 | 3.04 | 3.1 | 26 | 256 | 32 |
| calculate_mu_kernel | 1.0 | 2.81 | 2.8 | 1 | 128 | 40 |
| eliminate_duals_kernel | 2.0 | 1.25 | 2.5 | 26 | 256 | 30 |
| recover_duals_kernel | 2.0 | 1.18 | 2.4 | 50 | 256 | 34 |
| recover_slacks_kernel | 2.0 | 1.18 | 2.4 | 50 | 256 | 24 |
| eliminate_slacks_kernel | 2.0 | 1.10 | 2.2 | 50 | 256 | 20 |
| split_sol_kernel | 2.0 | 1.09 | 2.2 | 27 | 256 | 16 |
| assemble_rhs_kernel | 2.0 | 0.96 | 1.9 | 27 | 256 | 16 |
| update_termination_kernel | 1.0 | 1.70 | 1.7 | 1 | 256 | 31 |
| update_vars_after_corrector_step_kernel | 1.0 | 1.47 | 1.5 | 101 | 256 | 21 |
| update_rho_delta_with_ineq_kernel | 1.0 | 1.25 | 1.3 | 1 | 256 | 34 |
| corrector_rhs_kernel | 1.0 | 1.28 | 1.3 | 50 | 256 | 26 |
| update_prox_vars_kernel | 1.0 | 1.25 | 1.3 | 52 | 256 | 16 |
| update_kkt_diag_kernel | 1.0 | 1.18 | 1.2 | 27 | 256 | 16 |
| update_continue_flag_kernel | 1.0 | 1.09 | 1.1 | 1 | 32 | 26 |
| prepare_zu_minus_zl_and_zbu_minus_zbl_kernel | 1.0 | 1.02 | 1.0 | 26 | 256 | 22 |
| any_flag_kernel | 1.0 | 0.96 | 1.0 | 1 | 32 | 26 |
| factor_retry_kernel | 1.0 | 0.93 | 0.9 | 1 | 256 | 22 |
| mark_failed_kernel | 1.0 | 0.86 | 0.9 | 1 | 256 | 16 |
| advance_iteration_kernel | 1.0 | 0.77 | 0.8 | 1 | 1 | 16 |
