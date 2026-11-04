# IPM iteration profile (25 iterations, medians per iteration)

Total per iteration: **604.8 us** (GPU span 502.5 us + host round trip 102.4 us)

| Category | us / iteration | share |
|---|---|---|
| reduced: Cholesky | 166.1 | 27.5% |
| reduced: solve | 112.5 | 18.6% |
| SpMV | 62.4 | 10.3% |
| reduced: assembly | 60.8 | 10.0% |
| IPM kernels | 47.9 | 7.9% |
| KKT system kernels | 24.5 | 4.1% |
| memset | 5.8 | 1.0% |
| graph condition | 1.0 | 0.2% |
| GPU idle inside the iteration graph | 21.3 | 3.5% |
| Host round trip (flag D2H + sync + next graph launch) | 102.4 | 16.9% |

## Largest idle gaps inside the iteration (us per iteration)

| us | after | before | note |
|---|---|---|---|
| 5.6 | set_condition_kernel | eliminate_slacks_kernel |  |
| 4.0 | csr_partition_kernel | [CUDA memset] |  |
| 3.3 | [CUDA memset] | csrmv_v3_transpose_kernel |  |
| 2.9 | [CUDA memset] | csrmv_v3_kernel |  |
| 2.8 | chol_panel_kernel | chol_update_kernel |  |
| 0.4 | csrmv_v3_kernel | csr_partition_kernel |  |
| 0.3 | chol_update_kernel | chol_panel_kernel |  |
| 0.2 | eliminate_slacks_kernel | eliminate_duals_kernel |  |
| 0.2 | csrmv_v3_transpose_kernel | prepare_zu_minus_zl_and_zbu_minus_zbl_kernel |  |
| 0.2 | csrmv_v3_transpose_kernel | update_residual_nr_kernel |  |

## Our kernels (per call)

| kernel | calls / iteration | median us | us / iteration | grid | block | regs |
|---|---|---|---|---|---|---|
| chol_panel_kernel | 8.0 | 17.29 | 131.7 | 1 | 32 | 164 |
| trsv_flow_kernel | 2.0 | 39.63 | 79.2 | 16 | 32 | 168 |
| assemble_kernel | 1.0 | 49.92 | 50.1 | 602 | 512 | 40 |
| chol_update_kernel | 7.0 | 4.97 | 34.3 | 4 | 32 | 36 |
| recover_kernel | 2.0 | 9.06 | 18.1 | 807 | 256 | 40 |
| reduce_rhs_kernel | 2.0 | 7.61 | 15.2 | 241 | 128 | 40 |
| calculate_step_kernel | 2.0 | 5.22 | 10.4 | 1 | 128 | 48 |
| update_residual_nr_kernel | 1.0 | 7.84 | 7.8 | 1 | 128 | 96 |
| solution_finite_kernel | 2.0 | 2.69 | 5.4 | 1 | 128 | 40 |
| update_residuals_r_kernel | 1.0 | 4.97 | 5.0 | 1 | 128 | 40 |
| compact_rows_kernel | 1.0 | 4.97 | 5.0 | 807 | 256 | 40 |
| reduce_split_rows_kernel | 1.0 | 4.42 | 4.4 | 215 | 256 | 40 |
| prepare_factor_and_predictor_step_kernel | 1.0 | 4.19 | 4.2 | 1 | 128 | 40 |
| corrector_sigma_kernel | 1.0 | 3.48 | 3.5 | 1 | 128 | 44 |
| update_regularizations_kernel | 1.0 | 3.35 | 3.4 | 26 | 256 | 32 |
| calculate_mu_kernel | 1.0 | 3.10 | 3.1 | 1 | 128 | 40 |
| eliminate_duals_kernel | 2.0 | 1.35 | 2.7 | 26 | 256 | 30 |
| recover_slacks_kernel | 2.0 | 1.32 | 2.6 | 50 | 256 | 24 |
| recover_duals_kernel | 2.0 | 1.32 | 2.6 | 50 | 256 | 34 |
| eliminate_slacks_kernel | 2.0 | 1.24 | 2.5 | 50 | 256 | 20 |
| assemble_rhs_kernel | 2.0 | 1.03 | 2.1 | 27 | 256 | 16 |
| split_sol_kernel | 2.0 | 0.97 | 2.0 | 27 | 256 | 16 |
| update_termination_kernel | 1.0 | 1.84 | 1.8 | 1 | 256 | 31 |
| update_vars_after_corrector_step_kernel | 1.0 | 1.65 | 1.7 | 101 | 256 | 21 |
| corrector_rhs_kernel | 1.0 | 1.42 | 1.4 | 50 | 256 | 26 |
| weights_kernel | 1.0 | 1.42 | 1.4 | 26 | 256 | 16 |
| update_rho_delta_with_ineq_kernel | 1.0 | 1.39 | 1.4 | 1 | 256 | 34 |
| update_prox_vars_kernel | 1.0 | 1.39 | 1.4 | 52 | 256 | 16 |
| update_kkt_diag_kernel | 1.0 | 1.29 | 1.3 | 27 | 256 | 16 |
| update_continue_flag_kernel | 1.0 | 1.19 | 1.2 | 1 | 32 | 26 |
| prepare_zu_minus_zl_and_zbu_minus_zbl_kernel | 1.0 | 1.16 | 1.2 | 26 | 256 | 22 |
| any_flag_kernel | 1.0 | 1.03 | 1.0 | 1 | 32 | 26 |
| factor_retry_kernel | 1.0 | 1.00 | 1.0 | 1 | 256 | 22 |
| mark_failed_kernel | 1.0 | 0.94 | 0.9 | 1 | 256 | 16 |
| advance_iteration_kernel | 1.0 | 0.87 | 0.9 | 1 | 1 | 16 |
