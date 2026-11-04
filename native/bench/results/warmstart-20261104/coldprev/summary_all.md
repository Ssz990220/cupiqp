| strategy | run | QPs | iter med | iter mean | iter max | ms med | ms mean | ms max | speedup (med ms) | fail | KKT pass | max dx_inf |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cold | all | 99 | 29 | 28.8 | 33 | 12.09 | 12.07 | 13.72 | 1.00 | 0 | 99/99 | 0.0e+00 |
| cold_settings@rho=1e-8,delta=1e-6 | all | 99 | 29 | 28.9 | 33 | 12.10 | 12.08 | 13.70 | 1.00 | 0 | 99/99 | 3.8e-06 |
| cold_settings@rho=1e-4,delta=1e-2 | all | 99 | 29 | 28.6 | 31 | 12.09 | 11.98 | 13.15 | 1.00 | 0 | 99/99 | 3.8e-06 |
| cold_settings@eps_abs=1e-5,eps_duality_gap_abs=1e-5 | all | 99 | 28 | 28.2 | 32 | 11.69 | 11.77 | 13.61 | 1.03 | 0 | 36/99 | 3.4e-06 |
| full_prev | all | 99 | 12 | 13.5 | 40 | 5.03 | 5.61 | 16.31 | 2.40 | 0 | 99/99 | 2.8e-06 |
| full_prev_xdata:s_min=1e-1,z_min=1e-4 | all | 99 | 9 | 10.1 | 26 | 3.83 | 4.25 | 10.66 | 3.16 | 0 | 99/99 | 6.6e-06 |
| x0_prev_slack | all | 99 | 12 | 13.5 | 39 | 5.39 | 5.61 | 15.91 | 2.24 | 0 | 99/99 | 2.8e-06 |
| x0_duals:s_min=1e-4,z_min=1e-4 | all | 99 | 12 | 13.4 | 37 | 5.03 | 5.57 | 15.39 | 2.40 | 0 | 99/99 | 2.8e-06 |
| x0_duals:s_min=1e-1,z_min=1e-4 | all | 99 | 10 | 10.0 | 25 | 4.19 | 4.21 | 10.26 | 2.89 | 0 | 99/99 | 3.7e-06 |
| x0_duals:s_min=1e-1,z_min=1e-4,map=label | all | 99 | 10 | 12.6 | 25 | 4.20 | 5.27 | 10.27 | 2.88 | 0 | 99/99 | 3.7e-06 |
| x0_duals:s_min=1e-1,z_min=1e-4,map=position | all | 99 | 10 | 10.4 | 25 | 4.18 | 4.37 | 10.25 | 2.89 | 0 | 99/99 | 3.7e-06 |
| x0_duals:s_min=1e-1,z_min=1e-4@rho=1e-8,delta=1e-6 | all | 99 | 10 | 10.0 | 25 | 4.19 | 4.20 | 10.50 | 2.89 | 0 | 99/99 | 3.7e-06 |
| x0_duals_center:mu0=1e-4,beta=0.1 | all | 99 | 11 | 13.1 | 41 | 4.59 | 5.46 | 16.73 | 2.63 | 0 | 99/99 | 4.0e-06 |
| active_set:small=1e-3,big=1e-2 | all | 99 | 10 | 11.0 | 24 | 4.22 | 4.65 | 9.86 | 2.87 | 0 | 99/99 | 3.9e-06 |
| default_xy_prev_z | all | 99 | 12 | 13.5 | 25 | 5.33 | 5.96 | 10.74 | 2.27 | 0 | 99/99 | 3.9e-06 |
| gg_data:back=8,s_min=1e-2,z_min=1e-8 | all | 99 | 9 | 9.8 | 23 | 3.79 | 4.15 | 9.50 | 3.19 | 0 | 99/99 | 3.7e-06 |

| strategy | run | fixed ms (intercept) | ms per IPM iteration |
|---|---|---:|---:|
| cold | all | 0.61 | 0.398 |
| cold_settings@rho=1e-8,delta=1e-6 | all | 0.70 | 0.393 |
| cold_settings@rho=1e-4,delta=1e-2 | all | 0.92 | 0.386 |
| cold_settings@eps_abs=1e-5,eps_duality_gap_abs=1e-5 | all | 0.16 | 0.412 |
| full_prev | all | 0.19 | 0.402 |
| full_prev_xdata:s_min=1e-1,z_min=1e-4 | all | 0.17 | 0.404 |
| x0_prev_slack | all | 0.17 | 0.404 |
| x0_duals:s_min=1e-4,z_min=1e-4 | all | 0.13 | 0.407 |
| x0_duals:s_min=1e-1,z_min=1e-4 | all | 0.17 | 0.404 |
| x0_duals:s_min=1e-1,z_min=1e-4,map=label | all | 0.16 | 0.404 |
| x0_duals:s_min=1e-1,z_min=1e-4,map=position | all | 0.16 | 0.405 |
| x0_duals:s_min=1e-1,z_min=1e-4@rho=1e-8,delta=1e-6 | all | 0.15 | 0.407 |
| x0_duals_center:mu0=1e-4,beta=0.1 | all | 0.17 | 0.404 |
| active_set:small=1e-3,big=1e-2 | all | 0.15 | 0.408 |
| default_xy_prev_z | all | 0.41 | 0.409 |
| gg_data:back=8,s_min=1e-2,z_min=1e-8 | all | 0.18 | 0.405 |
