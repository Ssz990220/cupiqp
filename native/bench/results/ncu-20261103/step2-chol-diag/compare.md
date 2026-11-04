## microbench (graph replay median, us)

| dataset | state | op | before | after | change |
|---|---|---|---|---|---|
| mpc | case0_k12 | reduced_assemble | 65.2 | 65.1 | -0.2% |
| mpc | case0_k12 | reduced_factor | 207.9 | 181.2 | -12.8% |
| mpc | case0_k12 | reduced_solve | 47.8 | 47.7 | -0.2% |
| mpc | case0_k12 | reduced_iter_1f2s | 299.9 | 273.2 | -8.9% |
| mpc | case0_k25 | reduced_assemble | 65.2 | 64.5 | -1.0% |
| mpc | case0_k25 | reduced_factor | 190.5 | 165.9 | -12.9% |
| mpc | case0_k25 | reduced_solve | 43.9 | 44.0 | +0.2% |
| mpc | case0_k25 | reduced_iter_1f2s | 274.4 | 250.8 | -8.6% |
| trajopt | case0_k12 | reduced_assemble | 61.4 | 62.3 | +1.5% |
| trajopt | case0_k12 | reduced_factor | 247.5 | 216.1 | -12.7% |
| trajopt | case0_k12 | reduced_solve | 56.3 | 57.0 | +1.2% |
| trajopt | case0_k12 | reduced_iter_1f2s | 355.3 | 325.4 | -8.4% |
| trajopt | case0_k25 | reduced_assemble | 61.4 | 62.2 | +1.3% |
| trajopt | case0_k25 | reduced_factor | 247.8 | 216.8 | -12.5% |
| trajopt | case0_k25 | reduced_solve | 56.3 | 57.0 | +1.1% |
| trajopt | case0_k25 | reduced_iter_1f2s | 355.3 | 324.5 | -8.7% |

## synthetic sweep

| case | n | p | m | factor us | solve us | iter us (before -> after, change) | res before/after | cw before/after |
|---|---|---|---|---|---|---|---|---|
| T5_x6u3_block | 51 | 36 | 108 | 43.6 -> 34.8 | 17.4 -> 17.4 | 73.7 -> 65.4 (-11.2%) | 1.2e-15 / 9.3e-16 | 3.0e-13 / 3.2e-13 |
| T10_x6u3_block | 96 | 66 | 198 | 66.6 -> 54.3 | 21.3 -> 21.4 | 104.5 -> 92.0 (-11.9%) | 7.5e-16 / 7.2e-16 | 4.7e-13 / 5.3e-13 |
| T20_x6u3_block | 186 | 126 | 378 | 142.1 -> 116.7 | 33.8 -> 33.7 | 205.8 -> 180.9 (-12.1%) | 1.1e-15 / 9.6e-16 | 1.0e-12 / 1.1e-12 |
| T40_x6u3_block | 366 | 246 | 738 | 323.6 -> 274.4 | 58.4 -> 58.4 | 436.9 -> 387.1 (-11.4%) | 7.6e-16 / 8.7e-16 | 1.2e-12 / 1.1e-12 |
| T80_x6u3_block | 726 | 486 | 1458 | 806.9 -> 712.6 | 106.5 -> 106.4 | 1016.8 -> 922.1 (-9.3%) | 1.2e-15 / 1.4e-15 | 2.2e-12 / 1.9e-12 |
| T20_x2u1_block | 62 | 42 | 126 | 43.6 -> 34.8 | 17.4 -> 17.4 | 73.5 -> 65.3 (-11.1%) | 3.5e-14 / 4.2e-14 | 5.7e-13 / 7.5e-13 |
| T20_x4u2_block | 124 | 84 | 252 | 90.1 -> 73.5 | 25.3 -> 25.3 | 136.0 -> 118.8 (-12.7%) | 4.0e-15 / 4.6e-15 | 9.3e-13 / 1.1e-12 |
| T20_x12u4_block | 332 | 252 | 672 | 290.8 -> 245.5 | 54.3 -> 54.9 | 395.3 -> 350.2 (-11.4%) | 4.8e-16 / 4.3e-16 | 6.9e-13 / 6.2e-13 |
| T20_x20u8_block | 580 | 420 | 1176 | 607.1 -> 529.0 | 89.1 -> 89.1 | 781.0 -> 702.5 (-10.1%) | 6.2e-16 / 5.7e-16 | 5.5e-13 / 8.6e-13 |
| T20_x32u8_block | 832 | 672 | 1680 | 1036.2 -> 929.6 | 120.8 -> 120.8 | 1273.5 -> 1167.1 (-8.4%) | 3.2e-16 / 3.7e-16 | 4.3e-13 / 3.4e-13 |
| T20_x8u4_banded | 248 | 168 | 504 | 199.7 -> 166.8 | 42.0 -> 42.0 | 278.5 -> 245.8 (-11.8%) | 1.5e-15 / 1.5e-15 | 1.4e-12 / 1.3e-12 |
| T20_x8u4_sparse0.2 | 248 | 168 | 504 | 198.7 -> 165.9 | 42.0 -> 41.9 | 278.4 -> 244.7 (-12.1%) | 2.1e-15 / 1.8e-15 | 2.7e-12 / 2.3e-12 |
| T20_x8u4_sparse0.5 | 248 | 168 | 504 | 199.7 -> 166.9 | 42.0 -> 42.0 | 278.5 -> 245.8 (-11.8%) | 1.1e-15 / 7.6e-16 | 1.3e-12 / 1.2e-12 |
| T20_x8u4_global0.05 | 248 | 168 | 504 | 200.6 -> 167.7 | 42.0 -> 42.0 | 280.3 -> 247.4 (-11.7%) | 3.8e-16 / 6.2e-16 | 4.7e-13 / 7.7e-13 |
| T20_x8u4_global0.3 | 248 | 168 | 126 | 203.5 -> 170.5 | 42.0 -> 42.0 | 282.6 -> 249.9 (-11.6%) | 2.5e-15 / 2.7e-15 | 7.2e-13 / 6.5e-13 |
| T20_x8u4_rows120 | 248 | 168 | 2520 | 205.6 -> 172.7 | 44.0 -> 44.0 | 289.5 -> 256.7 (-11.3%) | 1.0e-16 / 9.0e-17 | 1.3e-13 / 1.1e-13 |
| edge_p0 | 248 | 0 | 504 | 199.7 -> 166.9 | 42.0 -> 42.0 | 279.3 -> 245.8 (-12.0%) | 8.6e-17 / 8.4e-17 | 7.3e-16 / 6.9e-16 |
| edge_m0 | 248 | 168 | 0 | 198.7 -> 165.8 | 41.0 -> 41.6 | 276.5 -> 243.7 (-11.9%) | 3.7e-15 / 3.0e-15 | 3.3e-12 / 6.1e-12 |
| edge_boxonly | 248 | 0 | 0 | 193.2 -> 159.7 | 39.9 -> 39.8 | 269.3 -> 236.5 (-12.2%) | 1.3e-16 / 1.1e-16 | 3.4e-16 / 2.7e-16 |
| edge_inactive0.9 | 248 | 168 | 504 | 199.7 -> 166.9 | 42.0 -> 42.0 | 279.4 -> 245.8 (-12.0%) | 2.1e-15 / 3.5e-15 | 3.2e-12 / 3.4e-12 |
| edge_zeros0.8 | 248 | 168 | 504 | 199.7 -> 166.9 | 42.0 -> 42.0 | 279.5 -> 246.3 (-11.9%) | 1.4e-15 / 1.4e-15 | 2.4e-12 / 1.7e-12 |
| edge_n1 | 1 | 0 | 3 | 19.5 -> 16.0 | 9.2 -> 9.2 | 33.8 -> 29.7 (-12.1%) | 4.3e-17 / 4.3e-17 | 5.6e-17 / 5.6e-17 |
| edge_n33 | 33 | 27 | 12 | 41.7 -> 33.7 | 17.4 -> 17.4 | 71.7 -> 63.5 (-11.4%) | 3.4e-15 / 3.0e-15 | 8.9e-13 / 5.3e-13 |
| near_limit_T100_x16u4 | 2016 | 1616 | 808 | 6448.1 -> 6199.3 | 279.6 -> 279.5 | 7014.0 -> 6755.3 (-3.7%) | 2.1e-15 / 1.9e-15 | 3.2e-12 / 3.6e-12 |
| near_limit_T50_x30u10_sparse | 2030 | 1530 | 1020 | 6732.6 -> 6469.2 | 285.3 -> 285.2 | 7296.0 -> 7036.9 (-3.6%) | 2.4e-15 / 1.9e-15 | 7.8e-12 / 6.4e-12 |
| ill_T20_x6u3_block | 186 | 126 | 378 | 142.3 -> 117.5 | 33.8 -> 33.8 | 206.4 -> 181.2 (-12.2%) | 3.3e-17 / 3.1e-17 | 3.0e-11 / 1.7e-11 |
| ill_T40_x12u4_sparse | 652 | 492 | 1312 | 697.3 -> 610.9 | 97.9 -> 97.9 | 889.8 -> 802.8 (-9.8%) | 2.0e-17 / 2.5e-17 | 5.5e-12 / 3.4e-12 |
| ill_T80_x6u3_global | 726 | 486 | 324 | 807.9 -> 713.5 | 110.6 -> 105.5 | 1015.7 -> 921.4 (-9.3%) | 3.4e-11 / 3.8e-11 | 3.6e-07 / 3.6e-07 |

sweep iteration time: 28 faster, 0 same (within 3%), 0 slower

## real-state accuracy (all IPM iterations of case 0)

- mpc res_random: before max 1.67e-18 median 1.03e-24; after max 1.94e-18 median 9.07e-25; after/before per state median 1.00 max 1.37
- mpc cw_random: before max 5.63e-06 median 4.10e-07; after max 8.18e-06 median 4.45e-07; after/before per state median 1.00 max 2.11
- mpc res_residual: before max 7.21e-16 median 7.63e-18; after max 7.40e-16 median 9.92e-18; after/before per state median 1.06 max 2.38
- mpc cw_residual: before max 9.53e-06 median 3.12e-09; after max 7.22e-06 median 3.01e-09; after/before per state median 0.97 max 2.69
- trajopt res_random: before max 2.98e-18 median 7.49e-24; after max 3.26e-18 median 5.57e-24; after/before per state median 0.97 max 1.33
- trajopt cw_random: before max 4.94e-06 median 8.48e-08; after max 4.45e-06 median 6.09e-08; after/before per state median 1.00 max 1.73
- trajopt res_residual: before max 3.88e-16 median 2.47e-20; after max 4.33e-16 median 1.76e-20; after/before per state median 0.81 max 2.29
- trajopt cw_residual: before max 1.71e-06 median 6.34e-11; after max 1.37e-06 median 6.58e-11; after/before per state median 1.08 max 1.51
