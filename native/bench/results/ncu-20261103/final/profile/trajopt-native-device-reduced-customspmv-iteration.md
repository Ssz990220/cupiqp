# IPM iteration profile (25 iterations, medians per iteration)

Total per iteration: **542.0 us** (GPU span 544.0 us + host round trip -1.4 us)

| Category | us / iteration | share |
|---|---|---|
| reduced: Cholesky | 170.9 | 31.5% |
| reduced: solve | 114.6 | 21.1% |
| reduced: assembly | 61.5 | 11.4% |
| IPM kernels | 52.3 | 9.6% |
| SpMV | 29.0 | 5.3% |
| KKT system kernels | 28.1 | 5.2% |
| graph condition | 2.5 | 0.5% |
| GPU idle inside the iteration graph | 84.5 | 15.6% |
| Host round trip (flag D2H + sync + next graph launch) | -1.4 | -0.3% |

## Largest idle gaps inside the iteration (us per iteration)

| us | after | before | note |
|---|---|---|---|
| 76.9 | set_condition_kernel | prepare_factor_and_predictor_step_kernel |  |
| 3.4 | set_condition_kernel | prepare_factor_and_predictor_step_kernel | stream switch |
| 2.0 | set_condition_kernel | eliminate_slacks_kernel |  |
| 0.5 | chol_panel_kernel | chol_update_kernel |  |
| 0.3 | chol_update_kernel | chol_panel_kernel |  |
| 0.1 | csr_lanes_kernel | csr_lanes_kernel |  |
| 0.1 | assemble_rhs_kernel | reduce_rhs_kernel |  |
| 0.1 | split_sol_kernel | solution_finite_kernel |  |
| 0.1 | solution_finite_kernel | csr_lanes_kernel |  |
| 0.1 | reduce_rhs_kernel | trsv_flow_kernel |  |

## Our kernels (per call)

| kernel | calls / iteration | median us | us / iteration | grid | block | regs |
|---|---|---|---|---|---|---|
| chol_panel_kernel | 8.0 | 17.66 | 134.8 | 1 | 32 | 164 |
| trsv_flow_kernel | 2.0 | 40.22 | 80.5 | 16 | 32 | 168 |
| assemble_kernel | 1.0 | 50.10 | 50.1 | 602 | 512 | 40 |
| chol_update_kernel | 7.0 | 5.17 | 36.1 | 4 | 32 | 36 |
| csr_lanes_kernel | 7.0 | 3.75 | 28.9 | 61 | 128 | 40 |
| recover_kernel | 2.0 | 9.33 | 18.7 | 807 | 256 | 40 |
| reduce_rhs_kernel | 2.0 | 7.78 | 15.6 | 241 | 128 | 40 |
| calculate_step_kernel | 2.0 | 5.42 | 10.8 | 1 | 128 | 48 |
| update_residual_nr_kernel | 1.0 | 8.55 | 8.5 | 1 | 128 | 96 |
| solution_finite_kernel | 2.0 | 2.94 | 5.9 | 1 | 128 | 40 |
| update_residuals_r_kernel | 1.0 | 5.20 | 5.2 | 1 | 128 | 40 |
| compact_rows_kernel | 1.0 | 5.13 | 5.1 | 807 | 256 | 40 |
| reduce_split_rows_kernel | 1.0 | 4.62 | 4.6 | 215 | 256 | 40 |
| prepare_factor_and_predictor_step_kernel | 1.0 | 4.42 | 4.4 | 1 | 128 | 40 |
| corrector_sigma_kernel | 1.0 | 3.68 | 3.7 | 1 | 128 | 44 |
| update_regularizations_kernel | 1.0 | 3.58 | 3.6 | 26 | 256 | 32 |
| calculate_mu_kernel | 1.0 | 3.36 | 3.4 | 1 | 128 | 40 |
| eliminate_duals_kernel | 2.0 | 1.58 | 3.2 | 26 | 256 | 30 |
| recover_duals_kernel | 2.0 | 1.55 | 3.1 | 50 | 256 | 34 |
| recover_slacks_kernel | 2.0 | 1.52 | 3.0 | 50 | 256 | 24 |
| eliminate_slacks_kernel | 2.0 | 1.42 | 2.8 | 50 | 256 | 20 |
| assemble_rhs_kernel | 2.0 | 1.26 | 2.5 | 27 | 256 | 16 |
| split_sol_kernel | 2.0 | 1.20 | 2.4 | 27 | 256 | 16 |
| update_termination_kernel | 1.0 | 2.13 | 2.1 | 1 | 256 | 31 |
| update_vars_after_corrector_step_kernel | 1.0 | 1.87 | 1.9 | 101 | 256 | 21 |
| update_rho_delta_with_ineq_kernel | 1.0 | 1.65 | 1.7 | 1 | 256 | 34 |
| weights_kernel | 1.0 | 1.65 | 1.7 | 26 | 256 | 16 |
| corrector_rhs_kernel | 1.0 | 1.61 | 1.6 | 50 | 256 | 26 |
| update_prox_vars_kernel | 1.0 | 1.58 | 1.6 | 52 | 256 | 16 |
| update_kkt_diag_kernel | 1.0 | 1.52 | 1.5 | 27 | 256 | 16 |
| update_continue_flag_kernel | 1.0 | 1.39 | 1.4 | 1 | 32 | 26 |
| prepare_zu_minus_zl_and_zbu_minus_zbl_kernel | 1.0 | 1.39 | 1.4 | 26 | 256 | 22 |
| any_flag_kernel | 1.0 | 1.26 | 1.3 | 1 | 32 | 26 |
| factor_retry_kernel | 1.0 | 1.23 | 1.2 | 1 | 256 | 22 |
| mark_failed_kernel | 1.0 | 1.13 | 1.1 | 1 | 256 | 16 |
| advance_iteration_kernel | 1.0 | 1.06 | 1.1 | 1 | 1 | 16 |
