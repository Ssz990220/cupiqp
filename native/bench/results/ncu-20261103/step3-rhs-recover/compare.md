## microbench (graph replay median, us)

| dataset | state | op | before | after | change |
|---|---|---|---|---|---|
| mpc | case0_k12 | reduced_assemble | 65.1 | 65.5 | +0.6% |
| mpc | case0_k12 | reduced_factor | 181.2 | 180.1 | -0.6% |
| mpc | case0_k12 | reduced_solve | 47.7 | 42.0 | -12.0% |
| mpc | case0_k12 | reduced_iter_1f2s | 273.2 | 260.9 | -4.5% |
| mpc | case0_k25 | reduced_assemble | 64.5 | 59.4 | -7.9% |
| mpc | case0_k25 | reduced_factor | 165.9 | 164.8 | -0.7% |
| mpc | case0_k25 | reduced_solve | 44.0 | 38.8 | -11.7% |
| mpc | case0_k25 | reduced_iter_1f2s | 250.8 | 238.6 | -4.8% |
| trajopt | case0_k12 | reduced_assemble | 62.3 | 62.4 | +0.0% |
| trajopt | case0_k12 | reduced_factor | 216.1 | 216.1 | +0.0% |
| trajopt | case0_k12 | reduced_solve | 57.0 | 53.2 | -6.6% |
| trajopt | case0_k12 | reduced_iter_1f2s | 325.4 | 318.5 | -2.1% |
| trajopt | case0_k25 | reduced_assemble | 62.2 | 61.4 | -1.2% |
| trajopt | case0_k25 | reduced_factor | 216.8 | 213.9 | -1.3% |
| trajopt | case0_k25 | reduced_solve | 57.0 | 53.1 | -6.8% |
| trajopt | case0_k25 | reduced_iter_1f2s | 324.5 | 315.2 | -2.9% |

## synthetic sweep

| case | n | p | m | factor us | solve us | iter us (before -> after, change) | res before/after | cw before/after |
|---|---|---|---|---|---|---|---|---|
| T5_x6u3_block | 51 | 36 | 108 | 34.8 -> 34.8 | 17.4 -> 17.4 | 65.4 -> 65.5 (+0.2%) | 9.3e-16 / 1.1e-15 | 3.2e-13 / 2.2e-13 |
| T10_x6u3_block | 96 | 66 | 198 | 54.3 -> 54.3 | 21.4 -> 21.5 | 92.0 -> 92.2 (+0.2%) | 7.2e-16 / 7.7e-16 | 5.3e-13 / 5.2e-13 |
| T20_x6u3_block | 186 | 126 | 378 | 116.7 -> 116.7 | 33.7 -> 33.7 | 180.9 -> 180.2 (-0.4%) | 9.6e-16 / 9.8e-16 | 1.1e-12 / 1.0e-12 |
| T40_x6u3_block | 366 | 246 | 738 | 274.4 -> 273.4 | 58.4 -> 58.4 | 387.1 -> 386.9 (-0.0%) | 8.7e-16 / 6.8e-16 | 1.1e-12 / 9.8e-13 |
| T80_x6u3_block | 726 | 486 | 1458 | 712.6 -> 710.4 | 106.4 -> 106.5 | 922.1 -> 920.5 (-0.2%) | 1.4e-15 / 1.6e-15 | 1.9e-12 / 1.6e-12 |
| T20_x2u1_block | 62 | 42 | 126 | 34.8 -> 34.8 | 17.4 -> 17.4 | 65.3 -> 65.5 (+0.3%) | 4.2e-14 / 2.3e-14 | 7.5e-13 / 8.0e-13 |
| T20_x4u2_block | 124 | 84 | 252 | 73.5 -> 73.4 | 25.3 -> 25.3 | 118.8 -> 119.4 (+0.5%) | 4.6e-15 / 4.2e-15 | 1.1e-12 / 9.2e-13 |
| T20_x12u4_block | 332 | 252 | 672 | 245.5 -> 244.7 | 54.9 -> 54.9 | 350.2 -> 349.9 (-0.1%) | 4.3e-16 / 4.5e-16 | 6.2e-13 / 5.8e-13 |
| T20_x20u8_block | 580 | 420 | 1176 | 529.0 -> 527.3 | 89.1 -> 89.0 | 702.5 -> 700.4 (-0.3%) | 5.7e-16 / 4.9e-16 | 8.6e-13 / 7.4e-13 |
| T20_x32u8_block | 832 | 672 | 1680 | 929.6 -> 926.7 | 120.8 -> 119.8 | 1167.1 -> 1163.2 (-0.3%) | 3.7e-16 / 3.5e-16 | 3.4e-13 / 3.3e-13 |
| T20_x8u4_banded | 248 | 168 | 504 | 166.8 -> 165.9 | 42.0 -> 42.0 | 245.8 -> 245.7 (-0.0%) | 1.5e-15 / 1.3e-15 | 1.3e-12 / 1.9e-12 |
| T20_x8u4_sparse0.2 | 248 | 168 | 504 | 165.9 -> 165.7 | 41.9 -> 42.0 | 244.7 -> 244.7 (+0.0%) | 1.8e-15 / 1.9e-15 | 2.3e-12 / 2.1e-12 |
| T20_x8u4_sparse0.5 | 248 | 168 | 504 | 166.9 -> 166.5 | 42.0 -> 42.0 | 245.8 -> 245.6 (-0.1%) | 7.6e-16 / 7.2e-16 | 1.2e-12 / 1.1e-12 |
| T20_x8u4_global0.05 | 248 | 168 | 504 | 167.7 -> 166.9 | 42.0 -> 42.0 | 247.4 -> 246.8 (-0.2%) | 6.2e-16 / 5.8e-16 | 7.7e-13 / 7.2e-13 |
| T20_x8u4_global0.3 | 248 | 168 | 126 | 170.5 -> 170.0 | 42.0 -> 42.0 | 249.9 -> 248.8 (-0.4%) | 2.7e-15 / 2.8e-15 | 6.5e-13 / 6.7e-13 |
| T20_x8u4_rows120 | 248 | 168 | 2520 | 172.7 -> 171.9 | 44.0 -> 44.0 | 256.7 -> 256.0 (-0.3%) | 9.0e-17 / 1.1e-16 | 1.1e-13 / 1.3e-13 |
| edge_p0 | 248 | 0 | 504 | 166.9 -> 166.5 | 42.0 -> 42.0 | 245.8 -> 245.8 (+0.0%) | 8.4e-17 / 9.1e-17 | 6.9e-16 / 8.5e-16 |
| edge_m0 | 248 | 168 | 0 | 165.8 -> 164.9 | 41.6 -> 41.5 | 243.7 -> 243.7 (+0.0%) | 3.0e-15 / 3.8e-15 | 6.1e-12 / 4.0e-12 |
| edge_boxonly | 248 | 0 | 0 | 159.7 -> 159.7 | 39.8 -> 39.8 | 236.5 -> 236.3 (-0.1%) | 1.1e-16 / 1.1e-16 | 2.7e-16 / 2.7e-16 |
| edge_inactive0.9 | 248 | 168 | 504 | 166.9 -> 166.6 | 42.0 -> 41.8 | 245.8 -> 245.7 (-0.0%) | 3.5e-15 / 2.7e-15 | 3.4e-12 / 3.4e-12 |
| edge_zeros0.8 | 248 | 168 | 504 | 166.9 -> 166.7 | 42.0 -> 42.0 | 246.3 -> 245.8 (-0.2%) | 1.4e-15 / 1.6e-15 | 1.7e-12 / 1.7e-12 |
| edge_n1 | 1 | 0 | 3 | 16.0 -> 15.9 | 9.2 -> 9.2 | 29.7 -> 30.3 (+2.0%) | 4.3e-17 / 4.3e-17 | 5.6e-17 / 5.6e-17 |
| edge_n33 | 33 | 27 | 12 | 33.7 -> 33.5 | 17.4 -> 17.4 | 63.5 -> 63.5 (+0.0%) | 3.0e-15 / 3.5e-15 | 5.3e-13 / 5.1e-13 |
| near_limit_T100_x16u4 | 2016 | 1616 | 808 | 6199.3 -> 6173.5 | 279.5 -> 279.3 | 6755.3 -> 6745.1 (-0.2%) | 1.9e-15 / 1.7e-15 | 3.6e-12 / 4.2e-12 |
| near_limit_T50_x30u10_sparse | 2030 | 1530 | 1020 | 6469.2 -> 6460.4 | 285.2 -> 284.7 | 7036.9 -> 7025.7 (-0.2%) | 1.9e-15 / 2.7e-15 | 6.4e-12 / 5.6e-12 |
| ill_T20_x6u3_block | 186 | 126 | 378 | 117.5 -> 116.7 | 33.8 -> 33.8 | 181.2 -> 181.2 (-0.0%) | 3.1e-17 / 4.4e-17 | 1.7e-11 / 2.7e-11 |
| ill_T40_x12u4_sparse | 652 | 492 | 1312 | 610.9 -> 610.3 | 97.9 -> 97.2 | 802.8 -> 801.8 (-0.1%) | 2.5e-17 / 2.6e-17 | 3.4e-12 / 4.9e-12 |
| ill_T80_x6u3_global | 726 | 486 | 324 | 713.5 -> 713.4 | 105.5 -> 106.0 | 921.4 -> 921.6 (+0.0%) | 3.8e-11 / 2.7e-11 | 3.6e-07 / 3.0e-07 |

sweep iteration time: 0 faster, 28 same (within 3%), 0 slower

## real-state accuracy (all IPM iterations of case 0)

- mpc res_random: before max 1.94e-18 median 9.07e-25; after max 2.36e-18 median 7.20e-25; after/before per state median 1.00 max 1.63
- mpc cw_random: before max 8.18e-06 median 4.45e-07; after max 4.39e-06 median 5.07e-07; after/before per state median 1.00 max 1.84
- mpc res_residual: before max 7.40e-16 median 9.92e-18; after max 7.04e-16 median 8.94e-18; after/before per state median 0.90 max 1.93
- mpc cw_residual: before max 7.22e-06 median 3.01e-09; after max 1.19e-05 median 2.78e-09; after/before per state median 1.00 max 1.64
- trajopt res_random: before max 3.26e-18 median 5.57e-24; after max 3.02e-18 median 8.57e-24; after/before per state median 0.99 max 1.84
- trajopt cw_random: before max 4.45e-06 median 6.09e-08; after max 5.93e-06 median 7.95e-08; after/before per state median 1.00 max 1.60
- trajopt res_residual: before max 4.33e-16 median 1.76e-20; after max 4.08e-16 median 1.39e-20; after/before per state median 1.08 max 3.03
- trajopt cw_residual: before max 1.37e-06 median 6.58e-11; after max 1.13e-06 median 5.28e-11; after/before per state median 0.99 max 3.91
