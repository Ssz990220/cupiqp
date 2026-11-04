# IPM iteration profile (25 iterations, medians per iteration)

Total per iteration: **721.6 us** (GPU span 610.6 us + host round trip 110.7 us)

| Category | us / iteration | share |
|---|---|---|
| reduced: Cholesky | 201.1 | 27.9% |
| reduced: solve | 190.6 | 26.4% |
| SpMV | 62.1 | 8.6% |
| reduced: assembly | 57.3 | 7.9% |
| IPM kernels | 47.6 | 6.6% |
| KKT system kernels | 24.4 | 3.4% |
| memset | 5.8 | 0.8% |
| graph condition | 1.0 | 0.1% |
| GPU idle inside the iteration graph | 21.1 | 2.9% |
| Host round trip (flag D2H + sync + next graph launch) | 110.7 | 15.3% |

## Largest idle gaps inside the iteration (us per iteration)

| us | after | before | note |
|---|---|---|---|
| 5.5 | set_condition_kernel | eliminate_slacks_kernel |  |
| 4.0 | csr_partition_kernel | [CUDA memset] |  |
| 3.2 | [CUDA memset] | csrmv_v3_transpose_kernel |  |
| 2.9 | [CUDA memset] | csrmv_v3_kernel |  |
| 2.3 | chol_panel_kernel | chol_update_kernel |  |
| 0.4 | csrmv_v3_kernel | csr_partition_kernel |  |
| 0.3 | chol_update_kernel | chol_panel_kernel |  |
| 0.2 | eliminate_slacks_kernel | eliminate_duals_kernel |  |
| 0.2 | csrmv_v3_transpose_kernel | prepare_zu_minus_zl_and_zbu_minus_zbl_kernel |  |
| 0.2 | csrmv_v3_transpose_kernel | update_residual_nr_kernel |  |

## Our kernels (per call)

| kernel | calls / iteration | median us | us / iteration | grid | block | regs |
|---|---|---|---|---|---|---|
| chol_panel_kernel | 8.0 | 21.68 | 166.9 | 1 | 32 | 164 |
| trsv_kernel | 2.0 | 70.68 | 141.3 | 1 | 512 | 102 |
| assemble_kernel | 1.0 | 55.87 | 55.8 | 602 | 512 | 40 |
| chol_update_kernel | 7.0 | 4.94 | 34.2 | 4 | 32 | 36 |
| reduce_rhs_kernel | 2.0 | 10.86 | 21.5 | 241 | 128 | 38 |
| recover_kernel | 2.0 | 9.20 | 18.4 | 807 | 256 | 40 |
| calculate_step_kernel | 2.0 | 5.16 | 10.3 | 1 | 128 | 48 |
| update_residual_nr_kernel | 1.0 | 7.83 | 7.8 | 1 | 128 | 96 |
| solution_finite_kernel | 2.0 | 2.71 | 5.4 | 1 | 128 | 40 |
| compact_rows_kernel | 1.0 | 4.94 | 4.9 | 807 | 256 | 40 |
| update_residuals_r_kernel | 1.0 | 4.94 | 4.9 | 1 | 128 | 40 |
| reduce_split_rows_kernel | 1.0 | 4.36 | 4.4 | 215 | 256 | 40 |
| prepare_factor_and_predictor_step_kernel | 1.0 | 4.20 | 4.2 | 1 | 128 | 40 |
| corrector_sigma_kernel | 1.0 | 3.46 | 3.5 | 1 | 128 | 44 |
| update_regularizations_kernel | 1.0 | 3.37 | 3.4 | 26 | 256 | 32 |
| calculate_mu_kernel | 1.0 | 3.11 | 3.1 | 1 | 128 | 40 |
| eliminate_duals_kernel | 2.0 | 1.31 | 2.6 | 26 | 256 | 30 |
| recover_duals_kernel | 2.0 | 1.31 | 2.6 | 50 | 256 | 34 |
| recover_slacks_kernel | 2.0 | 1.28 | 2.6 | 50 | 256 | 24 |
| eliminate_slacks_kernel | 2.0 | 1.23 | 2.5 | 50 | 256 | 20 |
| assemble_rhs_kernel | 2.0 | 1.03 | 2.1 | 27 | 256 | 16 |
| split_sol_kernel | 2.0 | 0.96 | 1.9 | 27 | 256 | 16 |
| update_termination_kernel | 1.0 | 1.86 | 1.8 | 1 | 256 | 31 |
| update_vars_after_corrector_step_kernel | 1.0 | 1.64 | 1.6 | 101 | 256 | 21 |
| update_rho_delta_with_ineq_kernel | 1.0 | 1.38 | 1.4 | 1 | 256 | 34 |
| weights_kernel | 1.0 | 1.41 | 1.4 | 26 | 256 | 16 |
| corrector_rhs_kernel | 1.0 | 1.41 | 1.4 | 50 | 256 | 26 |
| update_prox_vars_kernel | 1.0 | 1.38 | 1.4 | 52 | 256 | 16 |
| update_kkt_diag_kernel | 1.0 | 1.28 | 1.3 | 27 | 256 | 16 |
| update_continue_flag_kernel | 1.0 | 1.19 | 1.2 | 1 | 32 | 26 |
| prepare_zu_minus_zl_and_zbu_minus_zbl_kernel | 1.0 | 1.12 | 1.1 | 26 | 256 | 22 |
| any_flag_kernel | 1.0 | 1.03 | 1.0 | 1 | 32 | 26 |
| factor_retry_kernel | 1.0 | 0.99 | 1.0 | 1 | 256 | 22 |
| mark_failed_kernel | 1.0 | 0.93 | 0.9 | 1 | 256 | 16 |
| advance_iteration_kernel | 1.0 | 0.87 | 0.9 | 1 | 1 | 16 |
