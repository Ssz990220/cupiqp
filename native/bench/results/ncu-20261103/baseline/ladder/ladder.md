## mpc

| row | KKT@1e-6 | iters p50/max/mean | e2e p50 / P95 ms | update p50 | solve p50 |
|---|---|---|---|---|---|
| 3-native-default | 1920/1920 | 24.5/28/24.60 | 36.03 / 39.44 | 0.89 | 35.08 |
| 7-native-default-reduced | 1920/1920 | 24.5/28/24.59 | 14.24 / 15.50 | 0.89 | 13.31 |
| 7d-native-default-reduced-devloop | 1920/1920 | 24.5/28/24.59 | 12.74 / 13.71 | 0.90 | 11.80 |
| 8-staged-devloop-customspmv-reduced | 1920/1920 | 24.5/28/24.58 | 10.84 / 11.65 | 0.75 | 10.03 |

## trajopt

| row | KKT@1e-6 | iters p50/max/mean | e2e p50 / P95 ms | update p50 | solve p50 |
|---|---|---|---|---|---|
| 3-native-default | 960/960 | 30.0/36/29.82 | 39.98 / 46.35 | 0.61 | 39.36 |
| 7-native-default-reduced | 960/960 | 30.0/36/29.82 | 19.39 / 21.95 | 0.60 | 18.74 |
| 7d-native-default-reduced-devloop | 960/960 | 30.0/36/29.82 | 17.61 / 19.95 | 0.64 | 16.92 |
| 8-staged-devloop-customspmv-reduced | 960/960 | 29.5/36/29.79 | 15.96 / 18.05 | 0.44 | 15.51 |
