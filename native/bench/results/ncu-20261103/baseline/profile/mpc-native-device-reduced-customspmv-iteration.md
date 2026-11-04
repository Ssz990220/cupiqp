# IPM iteration profile (25 iterations, medians per iteration)

Total per iteration: **519.4 us** (GPU span 521.0 us + host round trip -1.3 us)

| Category | us / iteration | share |
|---|---|---|
| reduced: Cholesky | 146.2 | 28.2% |
| reduced: solve | 126.1 | 24.3% |
| reduced: assembly | 64.1 | 12.3% |
| IPM kernels | 50.4 | 9.7% |
| KKT system kernels | 27.2 | 5.2% |
| SpMV | 25.0 | 4.8% |
| graph condition | 2.5 | 0.5% |
| GPU idle inside the iteration graph | 79.2 | 15.2% |
| Host round trip (flag D2H + sync + next graph launch) | -1.3 | -0.3% |

## Largest idle gaps inside the iteration (us per iteration)

| us | after | before | note |
|---|---|---|---|
| 71.5 | set_condition_kernel | prepare_factor_and_predictor_step_kernel |  |
| 3.1 | set_condition_kernel | prepare_factor_and_predictor_step_kernel | stream switch |
| 2.0 | set_condition_kernel | eliminate_slacks_kernel |  |
| 0.4 | chol_panel_kernel | chol_update_kernel |  |
| 0.2 | chol_update_kernel | chol_panel_kernel |  |
| 0.1 | csr_lanes_kernel | csr_lanes_kernel |  |
| 0.1 | trsv_kernel | recover_kernel |  |
| 0.1 | split_sol_kernel | solution_finite_kernel |  |
| 0.1 | solution_finite_kernel | csr_lanes_kernel |  |
| 0.1 | assemble_rhs_kernel | reduce_rhs_kernel |  |

## Our kernels (per call)

| kernel | calls / iteration | median us | us / iteration | grid | block | regs |
|---|---|---|---|---|---|---|
| chol_panel_kernel | 6.0 | 22.10 | 126.3 | 1 | 32 | 164 |
| trsv_kernel | 2.0 | 47.80 | 95.6 | 1 | 512 | 102 |
| assemble_kernel | 1.0 | 56.08 | 56.3 | 424 | 512 | 40 |
| csr_lanes_kernel | 7.0 | 4.22 | 25.0 | 43 | 128 | 40 |
| chol_update_kernel | 5.0 | 3.32 | 19.9 | 4 | 32 | 36 |
| reduce_rhs_kernel | 2.0 | 9.21 | 18.4 | 169 | 128 | 38 |
| recover_kernel | 2.0 | 6.05 | 12.1 | 324 | 256 | 40 |
| calculate_step_kernel | 2.0 | 5.31 | 10.6 | 1 | 128 | 48 |
| update_residual_nr_kernel | 1.0 | 7.70 | 7.7 | 1 | 128 | 96 |
| solution_finite_kernel | 2.0 | 2.85 | 5.7 | 1 | 128 | 40 |
| update_residuals_r_kernel | 1.0 | 4.90 | 4.9 | 1 | 128 | 40 |
| prepare_factor_and_predictor_step_kernel | 1.0 | 4.35 | 4.4 | 1 | 128 | 40 |
| compact_rows_kernel | 1.0 | 4.32 | 4.3 | 324 | 256 | 40 |
| corrector_sigma_kernel | 1.0 | 3.67 | 3.7 | 1 | 128 | 44 |
| update_regularizations_kernel | 1.0 | 3.51 | 3.5 | 11 | 256 | 32 |
| calculate_mu_kernel | 1.0 | 3.22 | 3.2 | 1 | 128 | 40 |
| eliminate_duals_kernel | 2.0 | 1.59 | 3.2 | 11 | 256 | 30 |
| recover_duals_kernel | 2.0 | 1.48 | 3.0 | 20 | 256 | 34 |
| recover_slacks_kernel | 2.0 | 1.39 | 2.8 | 20 | 256 | 24 |
| eliminate_slacks_kernel | 2.0 | 1.32 | 2.7 | 20 | 256 | 20 |
| assemble_rhs_kernel | 2.0 | 1.26 | 2.5 | 11 | 256 | 16 |
| split_sol_kernel | 2.0 | 1.19 | 2.4 | 11 | 256 | 16 |
| reduce_split_rows_kernel | 1.0 | 2.19 | 2.2 | 127 | 256 | 40 |
| update_termination_kernel | 1.0 | 2.13 | 2.1 | 1 | 256 | 31 |
| update_rho_delta_with_ineq_kernel | 1.0 | 1.64 | 1.7 | 1 | 256 | 34 |
| update_vars_after_corrector_step_kernel | 1.0 | 1.71 | 1.7 | 41 | 256 | 21 |
| weights_kernel | 1.0 | 1.61 | 1.6 | 11 | 256 | 16 |
| update_prox_vars_kernel | 1.0 | 1.51 | 1.5 | 21 | 256 | 16 |
| corrector_rhs_kernel | 1.0 | 1.51 | 1.5 | 20 | 256 | 26 |
| update_kkt_diag_kernel | 1.0 | 1.51 | 1.5 | 11 | 256 | 16 |
| update_continue_flag_kernel | 1.0 | 1.39 | 1.4 | 1 | 32 | 26 |
| prepare_zu_minus_zl_and_zbu_minus_zbl_kernel | 1.0 | 1.35 | 1.4 | 11 | 256 | 22 |
| any_flag_kernel | 1.0 | 1.26 | 1.3 | 1 | 32 | 26 |
| factor_retry_kernel | 1.0 | 1.22 | 1.2 | 1 | 256 | 22 |
| mark_failed_kernel | 1.0 | 1.16 | 1.1 | 1 | 256 | 16 |
| advance_iteration_kernel | 1.0 | 1.03 | 1.0 | 1 | 1 | 16 |
