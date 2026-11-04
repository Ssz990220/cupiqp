# IPM iteration profile (25 iterations, medians per iteration)

Total per iteration: **541.8 us** (GPU span 441.2 us + host round trip 100.8 us)

| Category | us / iteration | share |
|---|---|---|
| reduced: Cholesky | 115.5 | 21.3% |
| SpMV | 91.7 | 16.9% |
| reduced: solve | 81.1 | 15.0% |
| reduced: assembly | 55.6 | 10.3% |
| IPM kernels | 45.9 | 8.5% |
| KKT system kernels | 23.8 | 4.4% |
| memset | 5.6 | 1.0% |
| graph condition | 1.0 | 0.2% |
| GPU idle inside the iteration graph | 20.8 | 3.8% |
| Host round trip (flag D2H + sync + next graph launch) | 100.8 | 18.6% |

## Largest idle gaps inside the iteration (us per iteration)

| us | after | before | note |
|---|---|---|---|
| 5.6 | set_condition_kernel | eliminate_slacks_kernel |  |
| 4.1 | csr_partition_kernel | [CUDA memset] |  |
| 3.3 | [CUDA memset] | csrmv_v3_transpose_kernel |  |
| 2.9 | [CUDA memset] | csrmv_v3_kernel |  |
| 2.2 | chol_panel_kernel | chol_update_kernel |  |
| 0.4 | csrmv_v3_kernel | csr_partition_kernel |  |
| 0.2 | eliminate_slacks_kernel | eliminate_duals_kernel |  |
| 0.2 | csrmv_v3_transpose_kernel | prepare_zu_minus_zl_and_zbu_minus_zbl_kernel |  |
| 0.2 | chol_update_kernel | chol_panel_kernel |  |
| 0.2 | csrmv_v3_transpose_kernel | update_residual_nr_kernel |  |

## Our kernels (per call)

| kernel | calls / iteration | median us | us / iteration | grid | block | regs |
|---|---|---|---|---|---|---|
| chol_panel_kernel | 6.0 | 17.24 | 96.8 | 1 | 32 | 164 |
| trsv_flow_kernel | 2.0 | 30.71 | 61.4 | 12 | 32 | 168 |
| assemble_kernel | 1.0 | 48.19 | 48.3 | 424 | 512 | 40 |
| chol_update_kernel | 5.0 | 3.08 | 18.7 | 4 | 32 | 36 |
| reduce_rhs_kernel | 2.0 | 5.30 | 10.6 | 169 | 128 | 40 |
| calculate_step_kernel | 2.0 | 5.04 | 10.1 | 1 | 128 | 48 |
| recover_kernel | 2.0 | 4.53 | 9.1 | 324 | 256 | 40 |
| update_residual_nr_kernel | 1.0 | 7.07 | 7.1 | 1 | 128 | 96 |
| solution_finite_kernel | 2.0 | 2.62 | 5.2 | 1 | 128 | 40 |
| update_residuals_r_kernel | 1.0 | 4.66 | 4.7 | 1 | 128 | 40 |
| prepare_factor_and_predictor_step_kernel | 1.0 | 4.14 | 4.1 | 1 | 128 | 40 |
| compact_rows_kernel | 1.0 | 4.08 | 4.1 | 324 | 256 | 40 |
| corrector_sigma_kernel | 1.0 | 3.41 | 3.4 | 1 | 128 | 44 |
| update_regularizations_kernel | 1.0 | 3.25 | 3.2 | 11 | 256 | 32 |
| calculate_mu_kernel | 1.0 | 2.96 | 3.0 | 1 | 128 | 40 |
| eliminate_duals_kernel | 2.0 | 1.35 | 2.7 | 11 | 256 | 30 |
| recover_duals_kernel | 2.0 | 1.25 | 2.5 | 20 | 256 | 34 |
| recover_slacks_kernel | 2.0 | 1.22 | 2.4 | 20 | 256 | 24 |
| eliminate_slacks_kernel | 2.0 | 1.17 | 2.3 | 20 | 256 | 20 |
| assemble_rhs_kernel | 2.0 | 1.03 | 2.1 | 11 | 256 | 16 |
| reduce_split_rows_kernel | 1.0 | 1.99 | 2.0 | 127 | 256 | 40 |
| split_sol_kernel | 2.0 | 0.96 | 1.9 | 11 | 256 | 16 |
| update_termination_kernel | 1.0 | 1.86 | 1.9 | 1 | 256 | 31 |
| update_vars_after_corrector_step_kernel | 1.0 | 1.51 | 1.5 | 41 | 256 | 21 |
| update_rho_delta_with_ineq_kernel | 1.0 | 1.38 | 1.4 | 1 | 256 | 34 |
| weights_kernel | 1.0 | 1.38 | 1.4 | 11 | 256 | 16 |
| update_prox_vars_kernel | 1.0 | 1.35 | 1.3 | 21 | 256 | 16 |
| update_kkt_diag_kernel | 1.0 | 1.32 | 1.3 | 11 | 256 | 16 |
| corrector_rhs_kernel | 1.0 | 1.28 | 1.3 | 20 | 256 | 26 |
| update_continue_flag_kernel | 1.0 | 1.19 | 1.2 | 1 | 32 | 26 |
| prepare_zu_minus_zl_and_zbu_minus_zbl_kernel | 1.0 | 1.12 | 1.1 | 11 | 256 | 22 |
| any_flag_kernel | 1.0 | 1.03 | 1.0 | 1 | 32 | 26 |
| factor_retry_kernel | 1.0 | 1.00 | 1.0 | 1 | 256 | 22 |
| mark_failed_kernel | 1.0 | 0.93 | 0.9 | 1 | 256 | 16 |
| advance_iteration_kernel | 1.0 | 0.83 | 0.8 | 1 | 1 | 16 |
