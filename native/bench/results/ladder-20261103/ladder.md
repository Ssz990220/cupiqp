# Optimization ladder (same session, gpu_resident, 20 repeats)

## MPC

| Step | KKT@1e-6 | iters p50/max | end-to-end p50 / P95 (ms) | update p50 | solve p50 | vs previous | vs reference |
|---|---|---|---|---|---|---|---|
| Reference (Warp, harness protocol) | 1920/1920 | 24.5/28 | 41.43 / 45.23 | 3.18 | 38.18 |  | 1.00x |
| Native default, before multi-block reductions | 1920/1920 | 24.5/28 | 37.67 / 41.22 | 0.91 | 36.72 | -9.1% | 1.10x |
| + multi-block reductions (current default) | 1920/1920 | 24.5/28 | 35.79 / 39.04 | 0.91 | 34.85 | -5.0% | 1.16x |
| + staged preallocated inputs (update_staged) | 1920/1920 | 24.5/28 | 35.33 / 38.85 | 0.75 | 34.52 | -1.3% | 1.17x |
| + whole-solve CUDA graph (opt-in) | 1920/1920 | 24.5/28 | 31.41 / 34.39 | 0.75 | 30.59 | -11.1% | 1.32x |
| + custom SpMV (opt-in) | 1920/1920 | 24.5/28 | 29.92 / 32.71 | 0.75 | 29.12 | -4.7% | 1.38x |
| *Estimate*: + reduced KKT solve (not integrated) | - | - | *~16.0* | | | *-47%* | *2.59x* |

## TRAJOPT

| Step | KKT@1e-6 | iters p50/max | end-to-end p50 / P95 (ms) | update p50 | solve p50 | vs previous | vs reference |
|---|---|---|---|---|---|---|---|
| Reference (Warp, harness protocol) | 960/960 | 30.0/36 | 50.47 / 58.32 | 4.07 | 46.22 |  | 1.00x |
| Native default, before multi-block reductions | 960/960 | 30.0/36 | 46.19 / 53.88 | 0.61 | 45.56 | -8.5% | 1.09x |
| + multi-block reductions (current default) | 960/960 | 30.0/38 | 39.84 / 46.26 | 0.61 | 39.20 | -13.8% | 1.27x |
| + staged preallocated inputs (update_staged) | 960/960 | 30.0/37 | 39.73 / 46.78 | 0.43 | 39.28 | -0.3% | 1.27x |
| + whole-solve CUDA graph (opt-in) | 960/960 | 30.0/36 | 35.12 / 41.29 | 0.43 | 34.69 | -11.6% | 1.44x |
| + custom SpMV (opt-in) | 960/960 | 30.0/36 | 33.65 / 40.04 | 0.43 | 33.21 | -4.2% | 1.50x |
| *Estimate*: + reduced KKT solve (not integrated) | - | - | *~23.8* | | | *-29%* | *2.12x* |

Estimate: measured best step minus ({'mpc': 547, 'trajopt': 318} us per factorization) x (iterations + 1), from native/bench/results/reduced-kkt-20261103; not an end-to-end measurement.
