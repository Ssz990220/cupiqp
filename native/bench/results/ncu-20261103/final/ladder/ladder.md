## mpc

| row | KKT@1e-6 | iters p50/max/mean | e2e p50 / P95 ms | update p50 | solve p50 | before e2e p50 | change |
|---|---|---|---|---|---|---|---|
| 3-native-default | 1920/1920 | 24.5/28/24.60 | 35.89 / 39.31 | 0.89 | 34.96 | 36.03 | -0.4% |
| 7-native-default-reduced | 1920/1920 | 24.5/28/24.59 | 12.32 / 13.33 | 0.89 | 11.39 | 14.24 | -13.5% |
| 7d-native-default-reduced-devloop | 1920/1920 | 24.5/28/24.58 | 10.88 / 11.71 | 0.91 | 9.93 | 12.74 | -14.6% |
| 8-staged-devloop-customspmv-reduced | 1920/1920 | 24.5/28/24.58 | 8.98 / 9.64 | 0.75 | 8.05 | 10.84 | -17.1% |

## trajopt

| row | KKT@1e-6 | iters p50/max/mean | e2e p50 / P95 ms | update p50 | solve p50 | before e2e p50 | change |
|---|---|---|---|---|---|---|---|
| 3-native-default | 960/960 | 30.0/36/29.81 | 39.82 / 46.00 | 0.60 | 39.21 | 39.98 | -0.4% |
| 7-native-default-reduced | 960/960 | 30.0/36/29.82 | 16.22 / 18.35 | 0.59 | 15.58 | 19.39 | -16.3% |
| 7d-native-default-reduced-devloop | 960/960 | 30.0/36/29.81 | 14.41 / 16.27 | 0.61 | 13.77 | 17.61 | -18.1% |
| 8-staged-devloop-customspmv-reduced | 960/960 | 30.0/36/29.79 | 12.86 / 14.51 | 0.43 | 12.37 | 15.96 | -19.4% |
