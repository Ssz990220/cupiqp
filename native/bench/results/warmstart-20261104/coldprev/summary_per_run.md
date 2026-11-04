| strategy | run | QPs | iter med | iter mean | iter max | ms med | ms mean | ms max | speedup (med ms) | fail | KKT pass | max dx_inf |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cold | 29-79b3-place-6f87a360 | 36 | 29 | 29.2 | 33 | 12.09 | 12.21 | 13.72 | 1.00 | 0 | 36/36 | 0.0e+00 |
| cold_settings@rho=1e-8,delta=1e-6 | 29-79b3-place-6f87a360 | 36 | 29 | 29.4 | 33 | 12.10 | 12.30 | 13.70 | 1.00 | 0 | 36/36 | 3.8e-06 |
| cold_settings@rho=1e-4,delta=1e-2 | 29-79b3-place-6f87a360 | 36 | 29 | 28.6 | 31 | 12.09 | 11.94 | 12.91 | 1.00 | 0 | 36/36 | 3.8e-06 |
| cold_settings@eps_abs=1e-5,eps_duality_gap_abs=1e-5 | 29-79b3-place-6f87a360 | 36 | 28 | 28.7 | 32 | 11.91 | 11.98 | 13.61 | 1.02 | 0 | 15/36 | 3.4e-06 |
| full_prev | 29-79b3-place-6f87a360 | 36 | 11 | 11.9 | 24 | 4.62 | 5.01 | 9.89 | 2.62 | 0 | 36/36 | 2.8e-06 |
| full_prev_xdata:s_min=1e-1,z_min=1e-4 | 29-79b3-place-6f87a360 | 36 | 8 | 8.6 | 21 | 3.39 | 3.64 | 8.68 | 3.57 | 0 | 36/36 | 2.5e-06 |
| x0_prev_slack | 29-79b3-place-6f87a360 | 36 | 11 | 11.9 | 24 | 4.62 | 4.99 | 9.85 | 2.62 | 0 | 36/36 | 2.8e-06 |
| x0_duals:s_min=1e-4,z_min=1e-4 | 29-79b3-place-6f87a360 | 36 | 12 | 11.9 | 24 | 5.00 | 4.99 | 9.84 | 2.42 | 0 | 36/36 | 2.8e-06 |
| x0_duals:s_min=1e-1,z_min=1e-4 | 29-79b3-place-6f87a360 | 36 | 8 | 8.5 | 17 | 3.39 | 3.61 | 7.03 | 3.57 | 0 | 36/36 | 3.7e-06 |
| x0_duals:s_min=1e-1,z_min=1e-4,map=label | 29-79b3-place-6f87a360 | 36 | 8 | 8.5 | 17 | 3.39 | 3.61 | 7.03 | 3.57 | 0 | 36/36 | 3.7e-06 |
| x0_duals:s_min=1e-1,z_min=1e-4,map=position | 29-79b3-place-6f87a360 | 36 | 8 | 8.5 | 17 | 3.38 | 3.62 | 7.05 | 3.58 | 0 | 36/36 | 3.7e-06 |
| x0_duals:s_min=1e-1,z_min=1e-4@rho=1e-8,delta=1e-6 | 29-79b3-place-6f87a360 | 36 | 8 | 8.5 | 18 | 3.20 | 3.60 | 7.45 | 3.78 | 0 | 36/36 | 3.7e-06 |
| x0_duals_center:mu0=1e-4,beta=0.1 | 29-79b3-place-6f87a360 | 36 | 10 | 11.4 | 39 | 4.21 | 4.78 | 15.89 | 2.88 | 0 | 36/36 | 4.0e-06 |
| active_set:small=1e-3,big=1e-2 | 29-79b3-place-6f87a360 | 36 | 10 | 10.3 | 17 | 4.21 | 4.36 | 7.04 | 2.87 | 0 | 36/36 | 3.9e-06 |
| default_xy_prev_z | 29-79b3-place-6f87a360 | 36 | 11 | 12.1 | 25 | 4.90 | 5.40 | 10.72 | 2.47 | 0 | 36/36 | 3.9e-06 |
| gg_data:back=8,s_min=1e-2,z_min=1e-8 | 29-79b3-place-6f87a360 | 36 | 8 | 8.6 | 21 | 3.40 | 3.66 | 8.65 | 3.55 | 0 | 36/36 | 3.7e-06 |
| cold | 55-5d59-pick-d59c9e60 | 63 | 29 | 28.5 | 30 | 12.09 | 11.98 | 13.01 | 1.00 | 0 | 63/63 | 0.0e+00 |
| cold_settings@rho=1e-8,delta=1e-6 | 55-5d59-pick-d59c9e60 | 63 | 29 | 28.6 | 31 | 12.09 | 11.96 | 12.89 | 1.00 | 0 | 63/63 | 3.6e-07 |
| cold_settings@rho=1e-4,delta=1e-2 | 55-5d59-pick-d59c9e60 | 63 | 29 | 28.7 | 31 | 12.09 | 12.01 | 13.15 | 1.00 | 0 | 63/63 | 4.9e-07 |
| cold_settings@eps_abs=1e-5,eps_duality_gap_abs=1e-5 | 55-5d59-pick-d59c9e60 | 63 | 28 | 27.9 | 30 | 11.69 | 11.65 | 12.53 | 1.03 | 0 | 21/63 | 2.7e-06 |
| full_prev | 55-5d59-pick-d59c9e60 | 63 | 13 | 14.3 | 40 | 5.43 | 5.95 | 16.31 | 2.23 | 0 | 63/63 | 1.2e-06 |
| full_prev_xdata:s_min=1e-1,z_min=1e-4 | 55-5d59-pick-d59c9e60 | 63 | 10 | 11.0 | 26 | 4.21 | 4.60 | 10.66 | 2.87 | 0 | 63/63 | 6.6e-06 |
| x0_prev_slack | 55-5d59-pick-d59c9e60 | 63 | 13 | 14.3 | 39 | 5.43 | 5.96 | 15.91 | 2.23 | 0 | 63/63 | 1.2e-06 |
| x0_duals:s_min=1e-4,z_min=1e-4 | 55-5d59-pick-d59c9e60 | 63 | 13 | 14.2 | 37 | 5.40 | 5.90 | 15.39 | 2.24 | 0 | 63/63 | 1.0e-06 |
| x0_duals:s_min=1e-1,z_min=1e-4 | 55-5d59-pick-d59c9e60 | 63 | 10 | 10.8 | 25 | 4.21 | 4.56 | 10.26 | 2.87 | 0 | 63/63 | 1.0e-06 |
| x0_duals:s_min=1e-1,z_min=1e-4,map=label | 55-5d59-pick-d59c9e60 | 63 | 16 | 15.0 | 25 | 6.65 | 6.22 | 10.27 | 1.82 | 0 | 63/63 | 1.0e-06 |
| x0_duals:s_min=1e-1,z_min=1e-4,map=position | 55-5d59-pick-d59c9e60 | 63 | 11 | 11.5 | 25 | 4.59 | 4.80 | 10.25 | 2.63 | 0 | 63/63 | 1.0e-06 |
| x0_duals:s_min=1e-1,z_min=1e-4@rho=1e-8,delta=1e-6 | 55-5d59-pick-d59c9e60 | 63 | 10 | 10.8 | 25 | 4.22 | 4.55 | 10.50 | 2.87 | 0 | 63/63 | 9.8e-07 |
| x0_duals_center:mu0=1e-4,beta=0.1 | 55-5d59-pick-d59c9e60 | 63 | 11 | 14.1 | 41 | 4.61 | 5.84 | 16.73 | 2.62 | 0 | 63/63 | 5.7e-07 |
| active_set:small=1e-3,big=1e-2 | 55-5d59-pick-d59c9e60 | 63 | 12 | 11.4 | 24 | 4.99 | 4.81 | 9.86 | 2.42 | 0 | 63/63 | 1.0e-06 |
| default_xy_prev_z | 55-5d59-pick-d59c9e60 | 63 | 13 | 14.3 | 25 | 5.71 | 6.27 | 10.74 | 2.12 | 0 | 63/63 | 1.2e-06 |
| gg_data:back=8,s_min=1e-2,z_min=1e-8 | 55-5d59-pick-d59c9e60 | 63 | 9 | 10.5 | 23 | 3.80 | 4.43 | 9.50 | 3.18 | 0 | 63/63 | 3.2e-07 |

| strategy | run | fixed ms (intercept) | ms per IPM iteration |
|---|---|---:|---:|
| cold | 29-79b3-place-6f87a360 | 0.43 | 0.403 |
| cold_settings@rho=1e-8,delta=1e-6 | 29-79b3-place-6f87a360 | 0.65 | 0.396 |
| cold_settings@rho=1e-4,delta=1e-2 | 29-79b3-place-6f87a360 | 0.88 | 0.387 |
| cold_settings@eps_abs=1e-5,eps_duality_gap_abs=1e-5 | 29-79b3-place-6f87a360 | -0.36 | 0.431 |
| full_prev | 29-79b3-place-6f87a360 | 0.26 | 0.397 |
| full_prev_xdata:s_min=1e-1,z_min=1e-4 | 29-79b3-place-6f87a360 | 0.16 | 0.405 |
| x0_prev_slack | 29-79b3-place-6f87a360 | 0.16 | 0.404 |
| x0_duals:s_min=1e-4,z_min=1e-4 | 29-79b3-place-6f87a360 | 0.17 | 0.404 |
| x0_duals:s_min=1e-1,z_min=1e-4 | 29-79b3-place-6f87a360 | 0.16 | 0.404 |
| x0_duals:s_min=1e-1,z_min=1e-4,map=label | 29-79b3-place-6f87a360 | 0.16 | 0.404 |
| x0_duals:s_min=1e-1,z_min=1e-4,map=position | 29-79b3-place-6f87a360 | 0.19 | 0.402 |
| x0_duals:s_min=1e-1,z_min=1e-4@rho=1e-8,delta=1e-6 | 29-79b3-place-6f87a360 | 0.16 | 0.404 |
| x0_duals_center:mu0=1e-4,beta=0.1 | 29-79b3-place-6f87a360 | 0.18 | 0.403 |
| active_set:small=1e-3,big=1e-2 | 29-79b3-place-6f87a360 | 0.26 | 0.397 |
| default_xy_prev_z | 29-79b3-place-6f87a360 | 0.46 | 0.407 |
| gg_data:back=8,s_min=1e-2,z_min=1e-8 | 29-79b3-place-6f87a360 | 0.25 | 0.397 |
| cold | 55-5d59-pick-d59c9e60 | 0.41 | 0.406 |
| cold_settings@rho=1e-8,delta=1e-6 | 55-5d59-pick-d59c9e60 | 0.86 | 0.388 |
| cold_settings@rho=1e-4,delta=1e-2 | 55-5d59-pick-d59c9e60 | 0.96 | 0.385 |
| cold_settings@eps_abs=1e-5,eps_duality_gap_abs=1e-5 | 55-5d59-pick-d59c9e60 | 0.59 | 0.397 |
| full_prev | 55-5d59-pick-d59c9e60 | 0.17 | 0.404 |
| full_prev_xdata:s_min=1e-1,z_min=1e-4 | 55-5d59-pick-d59c9e60 | 0.19 | 0.403 |
| x0_prev_slack | 55-5d59-pick-d59c9e60 | 0.17 | 0.404 |
| x0_duals:s_min=1e-4,z_min=1e-4 | 55-5d59-pick-d59c9e60 | 0.12 | 0.408 |
| x0_duals:s_min=1e-1,z_min=1e-4 | 55-5d59-pick-d59c9e60 | 0.19 | 0.403 |
| x0_duals:s_min=1e-1,z_min=1e-4,map=label | 55-5d59-pick-d59c9e60 | 0.16 | 0.404 |
| x0_duals:s_min=1e-1,z_min=1e-4,map=position | 55-5d59-pick-d59c9e60 | 0.15 | 0.406 |
| x0_duals:s_min=1e-1,z_min=1e-4@rho=1e-8,delta=1e-6 | 55-5d59-pick-d59c9e60 | 0.15 | 0.407 |
| x0_duals_center:mu0=1e-4,beta=0.1 | 55-5d59-pick-d59c9e60 | 0.16 | 0.404 |
| active_set:small=1e-3,big=1e-2 | 55-5d59-pick-d59c9e60 | 0.12 | 0.411 |
| default_xy_prev_z | 55-5d59-pick-d59c9e60 | 0.38 | 0.411 |
| gg_data:back=8,s_min=1e-2,z_min=1e-8 | 55-5d59-pick-d59c9e60 | 0.16 | 0.406 |
