# IPM iteration profile (25 iterations, medians per iteration)

Total per iteration: **627.9 us** (GPU span 515.1 us + host round trip 113.0 us)

| Category | us / iteration | share |
|---|---|---|
| reduced: Cholesky | 140.7 | 22.4% |
| reduced: solve | 129.4 | 20.6% |
| SpMV | 89.8 | 14.3% |
| reduced: assembly | 56.5 | 9.0% |
| IPM kernels | 45.2 | 7.2% |
| KKT system kernels | 23.4 | 3.7% |
| memset | 5.5 | 0.9% |
| graph condition | 1.0 | 0.2% |
| GPU idle inside the iteration graph | 23.4 | 3.7% |
| Host round trip (flag D2H + sync + next graph launch) | 113.0 | 18.0% |

## Largest idle gaps inside the iteration (us per iteration)

| us | after | before | note |
|---|---|---|---|
| 6.6 | csr_partition_kernel | [CUDA memset] |  |
| 5.5 | set_condition_kernel | eliminate_slacks_kernel |  |
| 3.2 | [CUDA memset] | csrmv_v3_transpose_kernel |  |
| 2.8 | [CUDA memset] | csrmv_v3_kernel |  |
| 2.2 | chol_panel_kernel | chol_update_kernel |  |
| 0.4 | csrmv_v3_kernel | csr_partition_kernel |  |
| 0.2 | eliminate_slacks_kernel | eliminate_duals_kernel |  |
| 0.2 | chol_update_kernel | chol_panel_kernel |  |
| 0.2 | csrmv_v3_transpose_kernel | prepare_zu_minus_zl_and_zbu_minus_zbl_kernel |  |
| 0.2 | csrmv_v3_transpose_kernel | update_residual_nr_kernel |  |

## Our kernels (per call)

| kernel | calls / iteration | median us | us / iteration | grid | block | regs |
|---|---|---|---|---|---|---|
| chol_panel_kernel | 6.0 | 21.41 | 122.2 | 1 | 32 | 164 |
| trsv_kernel | 2.0 | 46.89 | 93.8 | 1 | 512 | 102 |
| assemble_kernel | 1.0 | 55.17 | 55.4 | 424 | 512 | 40 |
| chol_update_kernel | 5.0 | 3.04 | 18.5 | 4 | 32 | 36 |
| reduce_rhs_kernel | 2.0 | 9.06 | 18.1 | 169 | 128 | 38 |
| recover_kernel | 2.0 | 5.70 | 11.4 | 324 | 256 | 40 |
| calculate_step_kernel | 2.0 | 4.97 | 9.9 | 1 | 128 | 48 |
| update_residual_nr_kernel | 1.0 | 6.97 | 7.0 | 1 | 128 | 96 |
| solution_finite_kernel | 2.0 | 2.60 | 5.2 | 1 | 128 | 40 |
| update_residuals_r_kernel | 1.0 | 4.59 | 4.6 | 1 | 128 | 40 |
| prepare_factor_and_predictor_step_kernel | 1.0 | 4.12 | 4.1 | 1 | 128 | 40 |
| compact_rows_kernel | 1.0 | 4.05 | 4.1 | 324 | 256 | 40 |
| corrector_sigma_kernel | 1.0 | 3.36 | 3.3 | 1 | 128 | 44 |
| update_regularizations_kernel | 1.0 | 3.20 | 3.2 | 11 | 256 | 32 |
| calculate_mu_kernel | 1.0 | 2.91 | 2.9 | 1 | 128 | 40 |
| eliminate_duals_kernel | 2.0 | 1.31 | 2.6 | 11 | 256 | 30 |
| recover_duals_kernel | 2.0 | 1.24 | 2.4 | 20 | 256 | 34 |
| recover_slacks_kernel | 2.0 | 1.17 | 2.3 | 20 | 256 | 24 |
| eliminate_slacks_kernel | 2.0 | 1.14 | 2.3 | 20 | 256 | 20 |
| assemble_rhs_kernel | 2.0 | 1.01 | 2.1 | 11 | 256 | 16 |
| reduce_split_rows_kernel | 1.0 | 1.96 | 2.0 | 127 | 256 | 40 |
| split_sol_kernel | 2.0 | 0.95 | 1.9 | 11 | 256 | 16 |
| update_termination_kernel | 1.0 | 1.84 | 1.8 | 1 | 256 | 31 |
| update_vars_after_corrector_step_kernel | 1.0 | 1.49 | 1.5 | 41 | 256 | 21 |
| update_rho_delta_with_ineq_kernel | 1.0 | 1.36 | 1.4 | 1 | 256 | 34 |
| weights_kernel | 1.0 | 1.36 | 1.4 | 11 | 256 | 16 |
| update_prox_vars_kernel | 1.0 | 1.33 | 1.3 | 21 | 256 | 16 |
| update_kkt_diag_kernel | 1.0 | 1.30 | 1.3 | 11 | 256 | 16 |
| corrector_rhs_kernel | 1.0 | 1.27 | 1.3 | 20 | 256 | 26 |
| update_continue_flag_kernel | 1.0 | 1.17 | 1.2 | 1 | 32 | 26 |
| prepare_zu_minus_zl_and_zbu_minus_zbl_kernel | 1.0 | 1.11 | 1.1 | 11 | 256 | 22 |
| any_flag_kernel | 1.0 | 1.01 | 1.0 | 1 | 32 | 26 |
| factor_retry_kernel | 1.0 | 0.98 | 1.0 | 1 | 256 | 22 |
| mark_failed_kernel | 1.0 | 0.92 | 0.9 | 1 | 256 | 16 |
| advance_iteration_kernel | 1.0 | 0.82 | 0.8 | 1 | 1 | 16 |
