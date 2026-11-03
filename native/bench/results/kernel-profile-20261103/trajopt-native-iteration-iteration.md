# IPM iteration profile (25 iterations, medians per iteration)

Total per iteration: **1520.0 us** (GPU span 1409.2 us + host round trip 108.2 us)

| Category | us / iteration | share |
|---|---|---|
| cuDSS factorization | 694.6 | 45.7% |
| IPM kernels | 248.4 | 16.3% |
| cuDSS solve | 242.8 | 16.0% |
| SpMV | 56.0 | 3.7% |
| KKT system kernels | 25.5 | 1.7% |
| memset | 12.7 | 0.8% |
| memcpy | 4.3 | 0.3% |
| graph condition | 0.9 | 0.1% |
| GPU idle inside the iteration graph | 126.4 | 8.3% |
| Host round trip (flag D2H + sync + next graph launch) | 108.2 | 7.1% |

## Largest idle gaps inside the iteration (us per iteration)

| us | after | before | note |
|---|---|---|---|
| 39.6 | set_condition_kernel | eliminate_slacks_kernel |  |
| 34.7 | csr_partition_kernel | [CUDA memset] |  |
| 27.7 | assemble_rhs_kernel | [CUDA memset] | stream switch |
| 3.9 | [CUDA memcpy Host-to-Device] | perm_ker |  |
| 3.6 | [CUDA memset] | [CUDA memset] |  |
| 2.9 | [CUDA memset] | csrmv_v3_transpose_kernel |  |
| 2.7 | [CUDA memset] | csrmv_v3_kernel |  |
| 2.0 | [CUDA memcpy Device-to-Device] | split_sol_kernel | stream switch |
| 1.9 | [CUDA memset] | [CUDA memcpy Host-to-Device] |  |
| 1.9 | [CUDA memcpy Host-to-Device] | [CUDA memcpy Host-to-Device] |  |

## Our kernels (per call)

| kernel | calls / iteration | median us | us / iteration | grid | block | regs |
|---|---|---|---|---|---|---|
| prepare_corrector_step_kernel | 1.0 | 74.11 | 74.1 | 1 | 256 | 48 |
| update_residual_nr_kernel | 1.0 | 51.96 | 52.0 | 1 | 256 | 80 |
| calculate_step_kernel | 1.0 | 53.48 | 51.8 | 1 | 256 | 48 |
| prepare_factor_and_predictor_step_kernel | 1.0 | 27.77 | 27.8 | 1 | 256 | 48 |
| update_residuals_r_kernel | 1.0 | 22.82 | 22.8 | 1 | 256 | 48 |
| solution_finite_kernel | 2.0 | 4.04 | 8.1 | 1 | 256 | 30 |
| calculate_mu_kernel | 1.0 | 7.12 | 7.1 | 1 | 256 | 64 |
| update_regularizations_kernel | 1.0 | 3.02 | 3.0 | 26 | 256 | 32 |
| eliminate_duals_kernel | 2.0 | 1.21 | 2.4 | 26 | 256 | 30 |
| recover_duals_kernel | 2.0 | 1.18 | 2.3 | 50 | 256 | 34 |
| recover_slacks_kernel | 2.0 | 1.14 | 2.3 | 50 | 256 | 24 |
| eliminate_slacks_kernel | 2.0 | 1.10 | 2.2 | 50 | 256 | 20 |
| split_sol_kernel | 2.0 | 1.08 | 2.2 | 27 | 256 | 16 |
| assemble_rhs_kernel | 2.0 | 0.95 | 1.9 | 27 | 256 | 16 |
| update_termination_kernel | 1.0 | 1.65 | 1.6 | 1 | 256 | 31 |
| update_vars_after_corrector_step_kernel | 1.0 | 1.49 | 1.5 | 101 | 256 | 21 |
| update_rho_delta_with_ineq_kernel | 1.0 | 1.24 | 1.3 | 1 | 256 | 34 |
| update_prox_vars_kernel | 1.0 | 1.24 | 1.2 | 52 | 256 | 16 |
| update_kkt_diag_kernel | 1.0 | 1.14 | 1.2 | 27 | 256 | 16 |
| update_continue_flag_kernel | 1.0 | 1.05 | 1.1 | 1 | 32 | 26 |
| prepare_zu_minus_zl_and_zbu_minus_zbl_kernel | 1.0 | 1.02 | 1.0 | 26 | 256 | 22 |
| any_flag_kernel | 1.0 | 0.92 | 0.9 | 1 | 32 | 26 |
| factor_retry_kernel | 1.0 | 0.89 | 0.9 | 1 | 256 | 22 |
| mark_failed_kernel | 1.0 | 0.86 | 0.8 | 1 | 256 | 16 |
| advance_iteration_kernel | 1.0 | 0.76 | 0.8 | 1 | 1 | 16 |
