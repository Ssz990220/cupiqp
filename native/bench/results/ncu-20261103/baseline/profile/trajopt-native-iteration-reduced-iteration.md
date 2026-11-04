# IPM iteration profile (25 iterations, medians per iteration)

Total per iteration: **713.5 us** (GPU span 613.4 us + host round trip 100.1 us)

| Category | us / iteration | share |
|---|---|---|
| reduced: Cholesky | 202.1 | 28.3% |
| reduced: solve | 182.0 | 25.5% |
| reduced: assembly | 66.7 | 9.3% |
| SpMV | 62.3 | 8.7% |
| IPM kernels | 47.8 | 6.7% |
| KKT system kernels | 24.5 | 3.4% |
| memset | 5.8 | 0.8% |
| graph condition | 1.0 | 0.1% |
| GPU idle inside the iteration graph | 21.2 | 3.0% |
| Host round trip (flag D2H + sync + next graph launch) | 100.1 | 14.0% |

## Largest idle gaps inside the iteration (us per iteration)

| us | after | before | note |
|---|---|---|---|
| 5.6 | set_condition_kernel | eliminate_slacks_kernel |  |
| 4.1 | csr_partition_kernel | [CUDA memset] |  |
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
| chol_panel_kernel | 8.0 | 21.80 | 167.7 | 1 | 32 | 164 |
| trsv_kernel | 2.0 | 70.93 | 141.9 | 1 | 512 | 102 |
| assemble_kernel | 1.0 | 55.83 | 55.7 | 602 | 512 | 40 |
| chol_update_kernel | 7.0 | 4.93 | 34.3 | 4 | 32 | 36 |
| reduce_rhs_kernel | 2.0 | 10.90 | 21.8 | 241 | 128 | 38 |
| recover_kernel | 2.0 | 9.26 | 18.5 | 807 | 256 | 40 |
| calculate_step_kernel | 2.0 | 5.19 | 10.4 | 1 | 128 | 48 |
| update_residual_nr_kernel | 1.0 | 7.84 | 7.8 | 1 | 128 | 96 |
| solution_finite_kernel | 2.0 | 2.69 | 5.4 | 1 | 128 | 40 |
| compact_rows_kernel | 1.0 | 4.97 | 5.0 | 807 | 256 | 40 |
| update_residuals_r_kernel | 1.0 | 4.97 | 5.0 | 1 | 128 | 40 |
| reduce_split_rows_kernel | 1.0 | 4.39 | 4.4 | 215 | 256 | 40 |
| prepare_factor_and_predictor_step_kernel | 1.0 | 4.19 | 4.2 | 1 | 128 | 40 |
| corrector_sigma_kernel | 1.0 | 3.48 | 3.5 | 1 | 128 | 44 |
| update_regularizations_kernel | 1.0 | 3.35 | 3.4 | 26 | 256 | 32 |
| calculate_mu_kernel | 1.0 | 3.13 | 3.1 | 1 | 128 | 40 |
| eliminate_duals_kernel | 2.0 | 1.34 | 2.7 | 26 | 256 | 30 |
| recover_duals_kernel | 2.0 | 1.32 | 2.6 | 50 | 256 | 34 |
| recover_slacks_kernel | 2.0 | 1.29 | 2.6 | 50 | 256 | 24 |
| eliminate_slacks_kernel | 2.0 | 1.24 | 2.5 | 50 | 256 | 20 |
| assemble_rhs_kernel | 2.0 | 1.03 | 2.1 | 27 | 256 | 16 |
| split_sol_kernel | 2.0 | 1.00 | 2.0 | 27 | 256 | 16 |
| update_termination_kernel | 1.0 | 1.87 | 1.8 | 1 | 256 | 31 |
| update_vars_after_corrector_step_kernel | 1.0 | 1.65 | 1.6 | 101 | 256 | 21 |
| corrector_rhs_kernel | 1.0 | 1.45 | 1.4 | 50 | 256 | 26 |
| weights_kernel | 1.0 | 1.42 | 1.4 | 26 | 256 | 16 |
| update_rho_delta_with_ineq_kernel | 1.0 | 1.35 | 1.4 | 1 | 256 | 34 |
| update_prox_vars_kernel | 1.0 | 1.39 | 1.4 | 52 | 256 | 16 |
| update_kkt_diag_kernel | 1.0 | 1.29 | 1.3 | 27 | 256 | 16 |
| update_continue_flag_kernel | 1.0 | 1.19 | 1.2 | 1 | 32 | 26 |
| prepare_zu_minus_zl_and_zbu_minus_zbl_kernel | 1.0 | 1.13 | 1.1 | 26 | 256 | 22 |
| any_flag_kernel | 1.0 | 1.03 | 1.0 | 1 | 32 | 26 |
| factor_retry_kernel | 1.0 | 1.00 | 1.0 | 1 | 256 | 22 |
| mark_failed_kernel | 1.0 | 0.94 | 1.0 | 1 | 256 | 16 |
| advance_iteration_kernel | 1.0 | 0.84 | 0.8 | 1 | 1 | 16 |
