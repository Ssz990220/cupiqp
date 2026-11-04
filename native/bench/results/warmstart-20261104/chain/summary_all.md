| strategy | run | QPs | iter med | iter mean | iter max | ms med | ms mean | ms max | speedup (med ms) | fail | KKT pass | max dx_inf |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cold | all | 99 | 29 | 28.8 | 33 | 12.03 | 11.98 | 13.64 | 1.00 | 0 | 99/99 | 0.0e+00 |
| cold_settings@rho=1e-8,delta=1e-6 | all | 99 | 29 | 28.9 | 33 | 12.03 | 12.02 | 13.65 | 1.00 | 0 | 99/99 | 3.8e-06 |
| cold_settings@rho=1e-4,delta=1e-2 | all | 99 | 29 | 28.6 | 31 | 12.03 | 11.92 | 12.85 | 1.00 | 0 | 99/99 | 3.8e-06 |
| cold_settings@eps_abs=1e-5,eps_duality_gap_abs=1e-5 | all | 99 | 28 | 28.2 | 32 | 11.69 | 11.75 | 13.25 | 1.03 | 0 | 36/99 | 3.4e-06 |
| full_prev | all | 99 | 12 | 13.4 | 37 | 5.01 | 5.55 | 15.08 | 2.40 | 0 | 99/99 | 2.8e-06 |
| full_prev_xdata:s_min=1e-1,z_min=1e-4 | all | 99 | 9 | 10.1 | 26 | 3.79 | 4.23 | 10.64 | 3.17 | 0 | 99/99 | 6.6e-06 |
| x0_prev_slack | all | 99 | 12 | 13.4 | 40 | 5.00 | 5.57 | 16.30 | 2.41 | 0 | 99/99 | 2.8e-06 |
| x0_duals:s_min=1e-4,z_min=1e-4 | all | 99 | 12 | 13.4 | 37 | 5.02 | 5.54 | 15.10 | 2.40 | 0 | 99/99 | 2.8e-06 |
| x0_duals:s_min=1e-1,z_min=1e-4 | all | 99 | 10 | 10.0 | 25 | 4.17 | 4.19 | 10.25 | 2.89 | 0 | 99/99 | 3.7e-06 |
| x0_duals:s_min=1e-1,z_min=1e-4,map=label | all | 99 | 10 | 12.6 | 25 | 4.18 | 5.25 | 10.26 | 2.88 | 0 | 99/99 | 3.7e-06 |
| x0_duals:s_min=1e-1,z_min=1e-4,map=position | all | 99 | 10 | 10.4 | 25 | 4.18 | 4.34 | 10.25 | 2.88 | 0 | 99/99 | 3.7e-06 |
| x0_duals:s_min=1e-1,z_min=1e-4@rho=1e-8,delta=1e-6 | all | 99 | 10 | 10.0 | 25 | 4.17 | 4.18 | 10.24 | 2.89 | 0 | 99/99 | 3.7e-06 |
| x0_duals_center:mu0=1e-4,beta=0.1 | all | 99 | 11 | 13.1 | 41 | 4.57 | 5.44 | 16.71 | 2.63 | 0 | 99/99 | 4.0e-06 |
| active_set:small=1e-3,big=1e-2 | all | 99 | 10 | 11.0 | 24 | 4.56 | 4.60 | 9.85 | 2.64 | 0 | 99/99 | 3.9e-06 |
| default_xy_prev_z | all | 99 | 12 | 13.6 | 25 | 5.29 | 5.94 | 10.74 | 2.27 | 0 | 99/99 | 3.9e-06 |
| gg_data:back=8,s_min=1e-2,z_min=1e-8 | all | 99 | 9 | 9.8 | 23 | 3.77 | 4.10 | 9.44 | 3.19 | 0 | 99/99 | 3.7e-06 |

| strategy | run | fixed ms (intercept) | ms per IPM iteration |
|---|---|---:|---:|
| cold | all | 0.51 | 0.398 |
| cold_settings@rho=1e-8,delta=1e-6 | all | 0.35 | 0.404 |
| cold_settings@rho=1e-4,delta=1e-2 | all | 1.03 | 0.381 |
| cold_settings@eps_abs=1e-5,eps_duality_gap_abs=1e-5 | all | 0.67 | 0.394 |
| full_prev | all | 0.14 | 0.405 |
| full_prev_xdata:s_min=1e-1,z_min=1e-4 | all | 0.14 | 0.404 |
| x0_prev_slack | all | 0.14 | 0.405 |
| x0_duals:s_min=1e-4,z_min=1e-4 | all | 0.14 | 0.404 |
| x0_duals:s_min=1e-1,z_min=1e-4 | all | 0.15 | 0.405 |
| x0_duals:s_min=1e-1,z_min=1e-4,map=label | all | 0.13 | 0.405 |
| x0_duals:s_min=1e-1,z_min=1e-4,map=position | all | 0.13 | 0.405 |
| x0_duals:s_min=1e-1,z_min=1e-4@rho=1e-8,delta=1e-6 | all | 0.14 | 0.404 |
| x0_duals_center:mu0=1e-4,beta=0.1 | all | 0.15 | 0.404 |
| active_set:small=1e-3,big=1e-2 | all | 0.17 | 0.402 |
| default_xy_prev_z | all | 0.34 | 0.413 |
| gg_data:back=8,s_min=1e-2,z_min=1e-8 | all | 0.14 | 0.404 |
