## microbench (graph replay median, us)

| dataset | state | op | before | after | change |
|---|---|---|---|---|---|
| mpc | case0_k12 | reduced_assemble | 65.5 | 57.3 | -12.5% |
| mpc | case0_k12 | reduced_factor | 180.1 | 173.7 | -3.5% |
| mpc | case0_k12 | reduced_solve | 42.0 | 42.7 | +1.8% |
| mpc | case0_k12 | reduced_iter_1f2s | 260.9 | 255.6 | -2.0% |
| mpc | case0_k25 | reduced_assemble | 59.4 | 53.1 | -10.6% |
| mpc | case0_k25 | reduced_factor | 164.8 | 160.7 | -2.5% |
| mpc | case0_k25 | reduced_solve | 38.8 | 38.9 | +0.2% |
| mpc | case0_k25 | reduced_iter_1f2s | 238.6 | 234.5 | -1.7% |
| trajopt | case0_k12 | reduced_assemble | 62.4 | 56.3 | -9.7% |
| trajopt | case0_k12 | reduced_factor | 216.1 | 210.8 | -2.4% |
| trajopt | case0_k12 | reduced_solve | 53.2 | 53.2 | +0.0% |
| trajopt | case0_k12 | reduced_iter_1f2s | 318.5 | 314.4 | -1.3% |
| trajopt | case0_k25 | reduced_assemble | 61.4 | 57.0 | -7.3% |
| trajopt | case0_k25 | reduced_factor | 213.9 | 211.6 | -1.0% |
| trajopt | case0_k25 | reduced_solve | 53.1 | 53.2 | +0.3% |
| trajopt | case0_k25 | reduced_iter_1f2s | 315.2 | 313.9 | -0.4% |

## synthetic sweep

| case | n | p | m | factor us | solve us | iter us (before -> after, change) | res before/after | cw before/after |
|---|---|---|---|---|---|---|---|---|
| T5_x6u3_block | 51 | 36 | 108 | 34.8 -> 34.8 | 17.4 -> 17.4 | 65.5 -> 65.5 (+0.0%) | 1.1e-15 / 1.1e-15 | 2.2e-13 / 2.2e-13 |
| T10_x6u3_block | 96 | 66 | 198 | 54.3 -> 54.3 | 21.5 -> 21.5 | 92.2 -> 92.2 (+0.0%) | 7.7e-16 / 7.7e-16 | 5.2e-13 / 5.2e-13 |
| T20_x6u3_block | 186 | 126 | 378 | 116.7 -> 118.8 | 33.7 -> 33.8 | 180.2 -> 181.2 (+0.6%) | 9.8e-16 / 9.8e-16 | 1.0e-12 / 1.0e-12 |
| T40_x6u3_block | 366 | 246 | 738 | 273.4 -> 274.4 | 58.4 -> 58.4 | 386.9 -> 387.9 (+0.3%) | 6.8e-16 / 6.8e-16 | 9.8e-13 / 9.8e-13 |
| T80_x6u3_block | 726 | 486 | 1458 | 710.4 -> 713.3 | 106.5 -> 106.5 | 920.5 -> 923.6 (+0.3%) | 1.6e-15 / 1.6e-15 | 1.6e-12 / 1.6e-12 |
| T20_x2u1_block | 62 | 42 | 126 | 34.8 -> 34.8 | 17.4 -> 17.4 | 65.5 -> 65.5 (+0.0%) | 2.3e-14 / 2.3e-14 | 8.0e-13 / 8.0e-13 |
| T20_x4u2_block | 124 | 84 | 252 | 73.4 -> 73.5 | 25.3 -> 25.4 | 119.4 -> 119.4 (+0.0%) | 4.2e-15 / 4.2e-15 | 9.2e-13 / 9.2e-13 |
| T20_x12u4_block | 332 | 252 | 672 | 244.7 -> 245.5 | 54.9 -> 55.0 | 349.9 -> 350.9 (+0.3%) | 4.5e-16 / 4.5e-16 | 5.8e-13 / 5.8e-13 |
| T20_x20u8_block | 580 | 420 | 1176 | 527.3 -> 528.0 | 89.0 -> 89.1 | 700.4 -> 701.4 (+0.1%) | 4.9e-16 / 4.9e-16 | 7.4e-13 / 7.4e-13 |
| T20_x32u8_block | 832 | 672 | 1680 | 926.7 -> 918.5 | 119.8 -> 121.9 | 1163.2 -> 1155.1 (-0.7%) | 3.5e-16 / 3.5e-16 | 3.3e-13 / 3.3e-13 |
| T20_x8u4_banded | 248 | 168 | 504 | 165.9 -> 166.8 | 42.0 -> 42.0 | 245.7 -> 246.3 (+0.2%) | 1.3e-15 / 1.3e-15 | 1.9e-12 / 1.9e-12 |
| T20_x8u4_sparse0.2 | 248 | 168 | 504 | 165.7 -> 165.9 | 42.0 -> 42.0 | 244.7 -> 245.6 (+0.4%) | 1.9e-15 / 1.9e-15 | 2.1e-12 / 2.1e-12 |
| T20_x8u4_sparse0.5 | 248 | 168 | 504 | 166.5 -> 166.9 | 42.0 -> 42.0 | 245.6 -> 245.8 (+0.1%) | 7.2e-16 / 7.2e-16 | 1.1e-12 / 1.1e-12 |
| T20_x8u4_global0.05 | 248 | 168 | 504 | 166.9 -> 166.9 | 42.0 -> 42.0 | 246.8 -> 246.8 (+0.0%) | 5.8e-16 / 5.8e-16 | 7.2e-13 / 7.2e-13 |
| T20_x8u4_global0.3 | 248 | 168 | 126 | 170.0 -> 170.8 | 42.0 -> 42.0 | 248.8 -> 249.9 (+0.4%) | 2.8e-15 / 2.8e-15 | 6.7e-13 / 6.7e-13 |
| T20_x8u4_rows120 | 248 | 168 | 2520 | 171.9 -> 171.9 | 44.0 -> 44.0 | 256.0 -> 255.7 (-0.1%) | 1.1e-16 / 1.1e-16 | 1.3e-13 / 1.3e-13 |
| edge_p0 | 248 | 0 | 504 | 166.5 -> 166.8 | 42.0 -> 42.0 | 245.8 -> 246.5 (+0.3%) | 9.1e-17 / 9.1e-17 | 8.5e-16 / 8.5e-16 |
| edge_m0 | 248 | 168 | 0 | 164.9 -> 165.7 | 41.5 -> 41.0 | 243.7 -> 244.3 (+0.3%) | 3.8e-15 / 3.8e-15 | 4.0e-12 / 4.0e-12 |
| edge_boxonly | 248 | 0 | 0 | 159.7 -> 160.3 | 39.8 -> 39.9 | 236.3 -> 236.5 (+0.1%) | 1.1e-16 / 1.1e-16 | 2.7e-16 / 2.7e-16 |
| edge_inactive0.9 | 248 | 168 | 504 | 166.6 -> 166.9 | 41.8 -> 41.9 | 245.7 -> 245.8 (+0.0%) | 2.7e-15 / 2.7e-15 | 3.4e-12 / 3.4e-12 |
| edge_zeros0.8 | 248 | 168 | 504 | 166.7 -> 166.9 | 42.0 -> 42.0 | 245.8 -> 246.4 (+0.2%) | 1.6e-15 / 1.6e-15 | 1.7e-12 / 1.7e-12 |
| edge_n1 | 1 | 0 | 3 | 15.9 -> 16.1 | 9.2 -> 9.2 | 30.3 -> 30.5 (+0.8%) | 4.3e-17 / 4.3e-17 | 5.6e-17 / 5.6e-17 |
| edge_n33 | 33 | 27 | 12 | 33.5 -> 33.7 | 17.4 -> 17.4 | 63.5 -> 63.5 (+0.0%) | 3.5e-15 / 3.5e-15 | 5.1e-13 / 5.1e-13 |
| near_limit_T100_x16u4 | 2016 | 1616 | 808 | 6173.5 -> 6177.8 | 279.3 -> 279.6 | 6745.1 -> 6745.9 (+0.0%) | 1.7e-15 / 1.7e-15 | 4.2e-12 / 4.2e-12 |
| near_limit_T50_x30u10_sparse | 2030 | 1530 | 1020 | 6460.4 -> 6453.1 | 284.7 -> 285.4 | 7025.7 -> 7022.6 (-0.0%) | 2.7e-15 / 2.7e-15 | 5.6e-12 / 5.6e-12 |
| ill_T20_x6u3_block | 186 | 126 | 378 | 116.7 -> 117.3 | 33.8 -> 33.8 | 181.2 -> 181.2 (+0.0%) | 4.4e-17 / 4.4e-17 | 2.7e-11 / 2.7e-11 |
| ill_T40_x12u4_sparse | 652 | 492 | 1312 | 610.3 -> 610.0 | 97.2 -> 97.3 | 801.8 -> 802.8 (+0.1%) | 2.6e-17 / 2.6e-17 | 4.9e-12 / 4.9e-12 |
| ill_T80_x6u3_global | 726 | 486 | 324 | 713.4 -> 712.6 | 106.0 -> 106.1 | 921.6 -> 920.6 (-0.1%) | 2.7e-11 / 2.7e-11 | 3.0e-07 / 3.0e-07 |

sweep iteration time: 0 faster, 28 same (within 3%), 0 slower

## real-state accuracy (all IPM iterations of case 0)

- mpc res_random: before max 2.36e-18 median 7.20e-25; after max 1.94e-18 median 7.20e-25; after/before per state median 1.00 max 2.18
- mpc cw_random: before max 4.39e-06 median 5.07e-07; after max 5.51e-06 median 5.07e-07; after/before per state median 1.00 max 1.97
- mpc res_residual: before max 7.04e-16 median 8.94e-18; after max 1.14e-15 median 9.67e-18; after/before per state median 0.98 max 1.62
- mpc cw_residual: before max 1.19e-05 median 2.78e-09; after max 6.62e-06 median 2.33e-09; after/before per state median 1.01 max 1.76
- trajopt res_random: before max 3.02e-18 median 8.57e-24; after max 3.03e-18 median 7.26e-24; after/before per state median 0.99 max 1.64
- trajopt cw_random: before max 5.93e-06 median 7.95e-08; after max 9.05e-06 median 8.06e-08; after/before per state median 1.00 max 1.53
- trajopt res_residual: before max 4.08e-16 median 1.39e-20; after max 4.49e-16 median 2.52e-20; after/before per state median 1.00 max 1.81
- trajopt cw_residual: before max 1.13e-06 median 5.28e-11; after max 1.56e-06 median 6.93e-11; after/before per state median 1.06 max 2.11
