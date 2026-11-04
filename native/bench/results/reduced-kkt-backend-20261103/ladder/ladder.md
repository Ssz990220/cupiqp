# Optimization ladder (same session, gpu_resident, 20 repeats)

## MPC

| Step | KKT@1e-6 | iters p50/max/mean | end-to-end p50 / P95 (ms) | update p50 | solve p50 | vs previous | vs reference |
|---|---|---|---|---|---|---|---|
| Reference (Warp, harness protocol) | 1920/1920 | 24.5/28/24.59 | 41.18 / 44.72 | 3.17 | 37.82 |  | 1.00x |
| Native default, before multi-block reductions | 1920/1920 | 24.5/28/24.59 | 37.92 / 41.18 | 0.90 | 36.98 | -7.9% | 1.09x |
| + multi-block reductions (current default) | 1920/1920 | 24.5/28/24.60 | 35.39 / 38.82 | 0.90 | 34.46 | -6.7% | 1.16x |
| + staged preallocated inputs (update_staged) | 1920/1920 | 24.5/28/24.60 | 35.20 / 38.65 | 0.75 | 34.40 | -0.6% | 1.17x |
| + whole-solve CUDA graph (opt-in) | 1920/1920 | 24.5/28/24.60 | 31.60 / 34.32 | 0.75 | 30.79 | -10.2% | 1.30x |
| + custom SpMV (opt-in) | 1920/1920 | 24.5/28/24.60 | 29.93 / 32.75 | 0.75 | 29.13 | -5.3% | 1.38x |
| Native default + reduced KKT backend (opt-in) | 1920/1920 | 24.5/28/24.59 | 14.40 / 15.55 | 0.90 | 13.46 | -59.3% (vs step 3) | 2.86x |
| staged + whole-solve graph + custom SpMV + reduced KKT backend | 1920/1920 | 24.5/28/24.58 | 10.81 / 11.62 | 0.75 | 10.00 | -63.9% (vs step 6) | 3.81x |

## TRAJOPT

| Step | KKT@1e-6 | iters p50/max/mean | end-to-end p50 / P95 (ms) | update p50 | solve p50 | vs previous | vs reference |
|---|---|---|---|---|---|---|---|
| Reference (Warp, harness protocol) | 960/960 | 30.0/36/29.82 | 50.51 / 58.68 | 4.03 | 46.36 |  | 1.00x |
| Native default, before multi-block reductions | 960/960 | 30.0/38/29.83 | 46.48 / 54.16 | 0.61 | 45.85 | -8.0% | 1.09x |
| + multi-block reductions (current default) | 960/960 | 30.0/36/29.83 | 40.31 / 46.30 | 0.60 | 39.68 | -13.3% | 1.25x |
| + staged preallocated inputs (update_staged) | 960/960 | 30.0/36/29.82 | 40.00 / 46.05 | 0.42 | 39.56 | -0.8% | 1.26x |
| + whole-solve CUDA graph (opt-in) | 960/960 | 30.0/36/29.83 | 35.29 / 40.57 | 0.43 | 34.85 | -11.8% | 1.43x |
| + custom SpMV (opt-in) | 960/960 | 30.0/36/29.82 | 33.98 / 39.58 | 0.43 | 33.54 | -3.7% | 1.49x |
| Native default + reduced KKT backend (opt-in) | 960/960 | 30.0/36/29.81 | 19.25 / 21.81 | 0.61 | 18.60 | -52.3% (vs step 3) | 2.62x |
| staged + whole-solve graph + custom SpMV + reduced KKT backend | 960/960 | 29.5/36/29.79 | 15.88 / 18.00 | 0.42 | 15.41 | -53.3% (vs step 6) | 3.18x |

