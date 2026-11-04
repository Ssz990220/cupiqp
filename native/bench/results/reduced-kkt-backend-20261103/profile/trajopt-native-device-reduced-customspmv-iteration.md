# IPM iteration profile (25 iterations, medians per iteration)

Total per iteration: **653.1 us** (GPU span 654.4 us + host round trip -1.4 us)

| Category | us / iteration | share |
|---|---|---|
| reduced: Cholesky | 205.7 | 31.5% |
| reduced: solve | 193.2 | 29.6% |
| reduced: assembly | 57.1 | 8.7% |
| IPM kernels | 52.1 | 8.0% |
| SpMV | 28.7 | 4.4% |
| KKT system kernels | 27.8 | 4.3% |
| graph condition | 2.5 | 0.4% |
| GPU idle inside the iteration graph | 87.4 | 13.4% |
| Host round trip (flag D2H + sync + next graph launch) | -1.4 | -0.2% |

## Largest idle gaps inside the iteration (us per iteration)

| us | after | before | note |
|---|---|---|---|
| 79.1 | set_condition_kernel | prepare_factor_and_predictor_step_kernel |  |
| 3.6 | set_condition_kernel | prepare_factor_and_predictor_step_kernel | stream switch |
| 2.0 | set_condition_kernel | eliminate_slacks_kernel |  |
| 0.7 | chol_panel_kernel | chol_update_kernel |  |
| 0.3 | chol_update_kernel | chol_panel_kernel |  |
| 0.1 | csr_lanes_kernel | csr_lanes_kernel |  |
| 0.1 | solution_finite_kernel | csr_lanes_kernel |  |
| 0.1 | assemble_rhs_kernel | reduce_rhs_kernel |  |
| 0.1 | trsv_kernel | recover_kernel |  |
| 0.1 | split_sol_kernel | solution_finite_kernel |  |

## Our kernels (per call)

| kernel | calls / iteration | median us | us / iteration | grid | block | regs |
|---|---|---|---|---|---|---|
| chol_panel_kernel | 8.0 | 22.04 | 170.0 | 1 | 32 | 164 |
| trsv_kernel | 2.0 | 71.27 | 142.6 | 1 | 512 | 102 |
| assemble_kernel | 1.0 | 55.46 | 55.8 | 602 | 512 | 40 |
| chol_update_kernel | 7.0 | 5.11 | 35.7 | 4 | 32 | 36 |
| csr_lanes_kernel | 7.0 | 3.69 | 28.6 | 61 | 128 | 40 |
| reduce_rhs_kernel | 2.0 | 11.02 | 22.0 | 241 | 128 | 38 |
| recover_kernel | 2.0 | 9.48 | 19.0 | 807 | 256 | 40 |
| calculate_step_kernel | 2.0 | 5.40 | 10.8 | 1 | 128 | 48 |
| update_residual_nr_kernel | 1.0 | 8.55 | 8.5 | 1 | 128 | 96 |
| solution_finite_kernel | 2.0 | 2.92 | 5.8 | 1 | 128 | 40 |
| update_residuals_r_kernel | 1.0 | 5.11 | 5.1 | 1 | 128 | 40 |
| compact_rows_kernel | 1.0 | 5.11 | 5.1 | 807 | 256 | 40 |
| reduce_split_rows_kernel | 1.0 | 4.53 | 4.5 | 215 | 256 | 40 |
| prepare_factor_and_predictor_step_kernel | 1.0 | 4.43 | 4.4 | 1 | 128 | 40 |
| corrector_sigma_kernel | 1.0 | 3.63 | 3.6 | 1 | 128 | 44 |
| update_regularizations_kernel | 1.0 | 3.57 | 3.6 | 26 | 256 | 32 |
| calculate_mu_kernel | 1.0 | 3.34 | 3.3 | 1 | 128 | 40 |
| eliminate_duals_kernel | 2.0 | 1.57 | 3.2 | 26 | 256 | 30 |
| recover_duals_kernel | 2.0 | 1.51 | 3.0 | 50 | 256 | 34 |
| recover_slacks_kernel | 2.0 | 1.48 | 3.0 | 50 | 256 | 24 |
| eliminate_slacks_kernel | 2.0 | 1.41 | 2.8 | 50 | 256 | 20 |
| assemble_rhs_kernel | 2.0 | 1.25 | 2.5 | 27 | 256 | 16 |
| split_sol_kernel | 2.0 | 1.19 | 2.4 | 27 | 256 | 16 |
| update_termination_kernel | 1.0 | 2.12 | 2.1 | 1 | 256 | 31 |
| update_vars_after_corrector_step_kernel | 1.0 | 1.86 | 1.9 | 101 | 256 | 21 |
| update_rho_delta_with_ineq_kernel | 1.0 | 1.64 | 1.7 | 1 | 256 | 34 |
| weights_kernel | 1.0 | 1.64 | 1.6 | 26 | 256 | 16 |
| corrector_rhs_kernel | 1.0 | 1.61 | 1.6 | 50 | 256 | 26 |
| update_prox_vars_kernel | 1.0 | 1.54 | 1.6 | 52 | 256 | 16 |
| update_kkt_diag_kernel | 1.0 | 1.48 | 1.5 | 27 | 256 | 16 |
| update_continue_flag_kernel | 1.0 | 1.38 | 1.4 | 1 | 32 | 26 |
| prepare_zu_minus_zl_and_zbu_minus_zbl_kernel | 1.0 | 1.38 | 1.4 | 26 | 256 | 22 |
| any_flag_kernel | 1.0 | 1.25 | 1.3 | 1 | 32 | 26 |
| factor_retry_kernel | 1.0 | 1.19 | 1.2 | 1 | 256 | 22 |
| mark_failed_kernel | 1.0 | 1.12 | 1.1 | 1 | 256 | 16 |
| advance_iteration_kernel | 1.0 | 1.06 | 1.1 | 1 | 1 | 16 |
