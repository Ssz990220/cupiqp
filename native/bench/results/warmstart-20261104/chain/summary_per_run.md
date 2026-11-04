| strategy | run | QPs | iter med | iter mean | iter max | ms med | ms mean | ms max | speedup (med ms) | fail | KKT pass | max dx_inf |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cold | 29-79b3-place-6f87a360 | 36 | 29 | 29.2 | 33 | 12.03 | 12.17 | 13.64 | 1.00 | 0 | 36/36 | 0.0e+00 |
| cold_settings@rho=1e-8,delta=1e-6 | 29-79b3-place-6f87a360 | 36 | 29 | 29.4 | 33 | 12.03 | 12.23 | 13.65 | 1.00 | 0 | 36/36 | 3.8e-06 |
| cold_settings@rho=1e-4,delta=1e-2 | 29-79b3-place-6f87a360 | 36 | 29 | 28.6 | 31 | 12.02 | 11.89 | 12.83 | 1.00 | 0 | 36/36 | 3.8e-06 |
| cold_settings@eps_abs=1e-5,eps_duality_gap_abs=1e-5 | 29-79b3-place-6f87a360 | 36 | 28 | 28.7 | 32 | 12.03 | 11.92 | 13.25 | 1.00 | 0 | 15/36 | 3.4e-06 |
| full_prev | 29-79b3-place-6f87a360 | 36 | 11 | 11.9 | 24 | 4.58 | 4.97 | 9.84 | 2.63 | 0 | 36/36 | 2.8e-06 |
| full_prev_xdata:s_min=1e-1,z_min=1e-4 | 29-79b3-place-6f87a360 | 36 | 8 | 8.6 | 21 | 3.36 | 3.62 | 8.59 | 3.58 | 0 | 36/36 | 2.5e-06 |
| x0_prev_slack | 29-79b3-place-6f87a360 | 36 | 11 | 11.9 | 24 | 4.57 | 4.96 | 9.82 | 2.63 | 0 | 36/36 | 2.8e-06 |
| x0_duals:s_min=1e-4,z_min=1e-4 | 29-79b3-place-6f87a360 | 36 | 12 | 11.9 | 24 | 4.97 | 4.95 | 9.79 | 2.42 | 0 | 36/36 | 2.8e-06 |
| x0_duals:s_min=1e-1,z_min=1e-4 | 29-79b3-place-6f87a360 | 36 | 8 | 8.5 | 17 | 3.36 | 3.59 | 7.00 | 3.58 | 0 | 36/36 | 3.7e-06 |
| x0_duals:s_min=1e-1,z_min=1e-4,map=label | 29-79b3-place-6f87a360 | 36 | 8 | 8.5 | 17 | 3.35 | 3.58 | 6.99 | 3.59 | 0 | 36/36 | 3.7e-06 |
| x0_duals:s_min=1e-1,z_min=1e-4,map=position | 29-79b3-place-6f87a360 | 36 | 8 | 8.5 | 17 | 3.36 | 3.58 | 7.01 | 3.58 | 0 | 36/36 | 3.7e-06 |
| x0_duals:s_min=1e-1,z_min=1e-4@rho=1e-8,delta=1e-6 | 29-79b3-place-6f87a360 | 36 | 8 | 8.5 | 18 | 3.16 | 3.58 | 7.39 | 3.80 | 0 | 36/36 | 3.7e-06 |
| x0_duals_center:mu0=1e-4,beta=0.1 | 29-79b3-place-6f87a360 | 36 | 10 | 11.4 | 39 | 4.17 | 4.74 | 15.83 | 2.88 | 0 | 36/36 | 4.0e-06 |
| active_set:small=1e-3,big=1e-2 | 29-79b3-place-6f87a360 | 36 | 10 | 10.3 | 17 | 4.17 | 4.29 | 6.98 | 2.89 | 0 | 36/36 | 3.9e-06 |
| default_xy_prev_z | 29-79b3-place-6f87a360 | 36 | 11 | 12.1 | 25 | 4.87 | 5.34 | 10.68 | 2.47 | 0 | 36/36 | 3.9e-06 |
| gg_data:back=8,s_min=1e-2,z_min=1e-8 | 29-79b3-place-6f87a360 | 36 | 8 | 8.6 | 21 | 3.36 | 3.60 | 8.59 | 3.58 | 0 | 36/36 | 3.7e-06 |
| cold | 55-5d59-pick-d59c9e60 | 63 | 29 | 28.5 | 30 | 12.03 | 11.87 | 12.46 | 1.00 | 0 | 63/63 | 0.0e+00 |
| cold_settings@rho=1e-8,delta=1e-6 | 55-5d59-pick-d59c9e60 | 63 | 29 | 28.6 | 31 | 12.03 | 11.90 | 12.85 | 1.00 | 0 | 63/63 | 3.6e-07 |
| cold_settings@rho=1e-4,delta=1e-2 | 55-5d59-pick-d59c9e60 | 63 | 29 | 28.7 | 31 | 12.04 | 11.94 | 12.85 | 1.00 | 0 | 63/63 | 4.9e-07 |
| cold_settings@eps_abs=1e-5,eps_duality_gap_abs=1e-5 | 55-5d59-pick-d59c9e60 | 63 | 28 | 27.9 | 30 | 11.69 | 11.65 | 12.50 | 1.03 | 0 | 21/63 | 2.7e-06 |
| full_prev | 55-5d59-pick-d59c9e60 | 63 | 13 | 14.2 | 37 | 5.40 | 5.89 | 15.08 | 2.23 | 0 | 63/63 | 1.2e-06 |
| full_prev_xdata:s_min=1e-1,z_min=1e-4 | 55-5d59-pick-d59c9e60 | 63 | 10 | 11.0 | 26 | 4.19 | 4.58 | 10.64 | 2.87 | 0 | 63/63 | 6.6e-06 |
| x0_prev_slack | 55-5d59-pick-d59c9e60 | 63 | 13 | 14.3 | 40 | 5.39 | 5.92 | 16.30 | 2.23 | 0 | 63/63 | 1.2e-06 |
| x0_duals:s_min=1e-4,z_min=1e-4 | 55-5d59-pick-d59c9e60 | 63 | 13 | 14.2 | 37 | 5.39 | 5.88 | 15.10 | 2.23 | 0 | 63/63 | 1.0e-06 |
| x0_duals:s_min=1e-1,z_min=1e-4 | 55-5d59-pick-d59c9e60 | 63 | 10 | 10.8 | 25 | 4.19 | 4.54 | 10.25 | 2.87 | 0 | 63/63 | 1.0e-06 |
| x0_duals:s_min=1e-1,z_min=1e-4,map=label | 55-5d59-pick-d59c9e60 | 63 | 16 | 15.0 | 25 | 6.61 | 6.20 | 10.26 | 1.82 | 0 | 63/63 | 1.0e-06 |
| x0_duals:s_min=1e-1,z_min=1e-4,map=position | 55-5d59-pick-d59c9e60 | 63 | 11 | 11.5 | 25 | 4.59 | 4.78 | 10.25 | 2.62 | 0 | 63/63 | 1.0e-06 |
| x0_duals:s_min=1e-1,z_min=1e-4@rho=1e-8,delta=1e-6 | 55-5d59-pick-d59c9e60 | 63 | 10 | 10.8 | 25 | 4.19 | 4.52 | 10.24 | 2.87 | 0 | 63/63 | 9.8e-07 |
| x0_duals_center:mu0=1e-4,beta=0.1 | 55-5d59-pick-d59c9e60 | 63 | 11 | 14.1 | 41 | 4.60 | 5.84 | 16.71 | 2.62 | 0 | 63/63 | 5.7e-07 |
| active_set:small=1e-3,big=1e-2 | 55-5d59-pick-d59c9e60 | 63 | 12 | 11.4 | 24 | 4.99 | 4.78 | 9.85 | 2.41 | 0 | 63/63 | 1.0e-06 |
| default_xy_prev_z | 55-5d59-pick-d59c9e60 | 63 | 13 | 14.4 | 25 | 5.70 | 6.28 | 10.74 | 2.11 | 0 | 63/63 | 1.2e-06 |
| gg_data:back=8,s_min=1e-2,z_min=1e-8 | 55-5d59-pick-d59c9e60 | 63 | 9 | 10.5 | 23 | 3.78 | 4.38 | 9.44 | 3.19 | 0 | 63/63 | 3.2e-07 |

| strategy | run | fixed ms (intercept) | ms per IPM iteration |
|---|---|---:|---:|
| cold | 29-79b3-place-6f87a360 | 0.69 | 0.393 |
| cold_settings@rho=1e-8,delta=1e-6 | 29-79b3-place-6f87a360 | 0.19 | 0.409 |
| cold_settings@rho=1e-4,delta=1e-2 | 29-79b3-place-6f87a360 | 1.74 | 0.355 |
| cold_settings@eps_abs=1e-5,eps_duality_gap_abs=1e-5 | 29-79b3-place-6f87a360 | 0.65 | 0.393 |
| full_prev | 29-79b3-place-6f87a360 | 0.10 | 0.408 |
| full_prev_xdata:s_min=1e-1,z_min=1e-4 | 29-79b3-place-6f87a360 | 0.14 | 0.402 |
| x0_prev_slack | 29-79b3-place-6f87a360 | 0.11 | 0.406 |
| x0_duals:s_min=1e-4,z_min=1e-4 | 29-79b3-place-6f87a360 | 0.15 | 0.402 |
| x0_duals:s_min=1e-1,z_min=1e-4 | 29-79b3-place-6f87a360 | 0.19 | 0.399 |
| x0_duals:s_min=1e-1,z_min=1e-4,map=label | 29-79b3-place-6f87a360 | 0.15 | 0.402 |
| x0_duals:s_min=1e-1,z_min=1e-4,map=position | 29-79b3-place-6f87a360 | 0.14 | 0.403 |
| x0_duals:s_min=1e-1,z_min=1e-4@rho=1e-8,delta=1e-6 | 29-79b3-place-6f87a360 | 0.14 | 0.402 |
| x0_duals_center:mu0=1e-4,beta=0.1 | 29-79b3-place-6f87a360 | 0.15 | 0.402 |
| active_set:small=1e-3,big=1e-2 | 29-79b3-place-6f87a360 | 0.13 | 0.404 |
| default_xy_prev_z | 29-79b3-place-6f87a360 | 0.30 | 0.416 |
| gg_data:back=8,s_min=1e-2,z_min=1e-8 | 29-79b3-place-6f87a360 | 0.14 | 0.402 |
| cold | 55-5d59-pick-d59c9e60 | 0.52 | 0.397 |
| cold_settings@rho=1e-8,delta=1e-6 | 55-5d59-pick-d59c9e60 | 0.41 | 0.401 |
| cold_settings@rho=1e-4,delta=1e-2 | 55-5d59-pick-d59c9e60 | 0.71 | 0.392 |
| cold_settings@eps_abs=1e-5,eps_duality_gap_abs=1e-5 | 55-5d59-pick-d59c9e60 | 0.33 | 0.407 |
| full_prev | 55-5d59-pick-d59c9e60 | 0.15 | 0.404 |
| full_prev_xdata:s_min=1e-1,z_min=1e-4 | 55-5d59-pick-d59c9e60 | 0.15 | 0.403 |
| x0_prev_slack | 55-5d59-pick-d59c9e60 | 0.16 | 0.404 |
| x0_duals:s_min=1e-4,z_min=1e-4 | 55-5d59-pick-d59c9e60 | 0.15 | 0.404 |
| x0_duals:s_min=1e-1,z_min=1e-4 | 55-5d59-pick-d59c9e60 | 0.14 | 0.406 |
| x0_duals:s_min=1e-1,z_min=1e-4,map=label | 55-5d59-pick-d59c9e60 | 0.14 | 0.404 |
| x0_duals:s_min=1e-1,z_min=1e-4,map=position | 55-5d59-pick-d59c9e60 | 0.15 | 0.404 |
| x0_duals:s_min=1e-1,z_min=1e-4@rho=1e-8,delta=1e-6 | 55-5d59-pick-d59c9e60 | 0.15 | 0.404 |
| x0_duals_center:mu0=1e-4,beta=0.1 | 55-5d59-pick-d59c9e60 | 0.16 | 0.404 |
| active_set:small=1e-3,big=1e-2 | 55-5d59-pick-d59c9e60 | 0.20 | 0.401 |
| default_xy_prev_z | 55-5d59-pick-d59c9e60 | 0.36 | 0.412 |
| gg_data:back=8,s_min=1e-2,z_min=1e-8 | 55-5d59-pick-d59c9e60 | 0.15 | 0.404 |
