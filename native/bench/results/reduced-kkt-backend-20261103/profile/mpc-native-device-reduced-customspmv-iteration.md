# IPM iteration profile (25 iterations, medians per iteration)

Total per iteration: **522.7 us** (GPU span 524.1 us + host round trip -1.3 us)

| Category | us / iteration | share |
|---|---|---|
| reduced: Cholesky | 144.7 | 27.7% |
| reduced: solve | 131.3 | 25.1% |
| reduced: assembly | 56.7 | 10.8% |
| IPM kernels | 50.0 | 9.6% |
| KKT system kernels | 26.9 | 5.2% |
| SpMV | 24.5 | 4.7% |
| graph condition | 2.5 | 0.5% |
| GPU idle inside the iteration graph | 87.5 | 16.7% |
| Host round trip (flag D2H + sync + next graph launch) | -1.3 | -0.2% |

## Largest idle gaps inside the iteration (us per iteration)

| us | after | before | note |
|---|---|---|---|
| 79.3 | set_condition_kernel | prepare_factor_and_predictor_step_kernel |  |
| 3.4 | set_condition_kernel | prepare_factor_and_predictor_step_kernel | stream switch |
| 2.0 | set_condition_kernel | eliminate_slacks_kernel |  |
| 0.4 | chol_panel_kernel | chol_update_kernel |  |
| 0.2 | chol_update_kernel | chol_panel_kernel |  |
| 0.1 | csr_lanes_kernel | csr_lanes_kernel |  |
| 0.1 | split_sol_kernel | solution_finite_kernel |  |
| 0.1 | assemble_rhs_kernel | reduce_rhs_kernel |  |
| 0.1 | reduce_rhs_kernel | trsv_kernel |  |
| 0.1 | csr_lanes_kernel | recover_duals_kernel |  |

## Our kernels (per call)

| kernel | calls / iteration | median us | us / iteration | grid | block | regs |
|---|---|---|---|---|---|---|
| chol_panel_kernel | 6.0 | 21.86 | 125.0 | 1 | 32 | 164 |
| trsv_kernel | 2.0 | 47.25 | 94.5 | 1 | 512 | 102 |
| assemble_kernel | 1.0 | 55.12 | 55.6 | 424 | 512 | 40 |
| csr_lanes_kernel | 7.0 | 4.14 | 24.5 | 43 | 128 | 40 |
| chol_update_kernel | 5.0 | 3.28 | 19.7 | 4 | 32 | 36 |
| reduce_rhs_kernel | 2.0 | 9.21 | 18.5 | 169 | 128 | 38 |
| recover_kernel | 2.0 | 5.93 | 11.9 | 324 | 256 | 40 |
| calculate_step_kernel | 2.0 | 5.23 | 10.5 | 1 | 128 | 48 |
| update_residual_nr_kernel | 1.0 | 7.68 | 7.7 | 1 | 128 | 96 |
| solution_finite_kernel | 2.0 | 2.82 | 5.6 | 1 | 128 | 40 |
| update_residuals_r_kernel | 1.0 | 4.91 | 4.9 | 1 | 128 | 40 |
| prepare_factor_and_predictor_step_kernel | 1.0 | 4.37 | 4.4 | 1 | 128 | 40 |
| compact_rows_kernel | 1.0 | 4.27 | 4.3 | 324 | 256 | 40 |
| corrector_sigma_kernel | 1.0 | 3.63 | 3.6 | 1 | 128 | 44 |
| update_regularizations_kernel | 1.0 | 3.44 | 3.4 | 11 | 256 | 32 |
| calculate_mu_kernel | 1.0 | 3.19 | 3.2 | 1 | 128 | 40 |
| eliminate_duals_kernel | 2.0 | 1.56 | 3.2 | 11 | 256 | 30 |
| recover_duals_kernel | 2.0 | 1.47 | 2.9 | 20 | 256 | 34 |
| recover_slacks_kernel | 2.0 | 1.40 | 2.8 | 20 | 256 | 24 |
| eliminate_slacks_kernel | 2.0 | 1.34 | 2.7 | 20 | 256 | 20 |
| assemble_rhs_kernel | 2.0 | 1.24 | 2.5 | 11 | 256 | 16 |
| split_sol_kernel | 2.0 | 1.18 | 2.4 | 11 | 256 | 16 |
| reduce_split_rows_kernel | 1.0 | 2.13 | 2.1 | 127 | 256 | 40 |
| update_termination_kernel | 1.0 | 2.10 | 2.1 | 1 | 256 | 31 |
| update_vars_after_corrector_step_kernel | 1.0 | 1.69 | 1.7 | 41 | 256 | 21 |
| update_rho_delta_with_ineq_kernel | 1.0 | 1.62 | 1.7 | 1 | 256 | 34 |
| weights_kernel | 1.0 | 1.59 | 1.6 | 11 | 256 | 16 |
| update_prox_vars_kernel | 1.0 | 1.50 | 1.5 | 21 | 256 | 16 |
| corrector_rhs_kernel | 1.0 | 1.50 | 1.5 | 20 | 256 | 26 |
| update_kkt_diag_kernel | 1.0 | 1.50 | 1.5 | 11 | 256 | 16 |
| update_continue_flag_kernel | 1.0 | 1.37 | 1.4 | 1 | 32 | 26 |
| prepare_zu_minus_zl_and_zbu_minus_zbl_kernel | 1.0 | 1.34 | 1.3 | 11 | 256 | 22 |
| any_flag_kernel | 1.0 | 1.24 | 1.2 | 1 | 32 | 26 |
| factor_retry_kernel | 1.0 | 1.21 | 1.2 | 1 | 256 | 22 |
| mark_failed_kernel | 1.0 | 1.15 | 1.1 | 1 | 256 | 16 |
| advance_iteration_kernel | 1.0 | 1.05 | 1.0 | 1 | 1 | 16 |
