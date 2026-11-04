## microbench (graph replay median, us)

| dataset | state | op | before | after | change |
|---|---|---|---|---|---|
| mpc | case0_k12 | reduced_assemble | 65.2 | 65.2 | +0.0% |
| mpc | case0_k12 | reduced_factor | 208.7 | 207.9 | -0.4% |
| mpc | case0_k12 | reduced_solve | 64.5 | 47.8 | -25.9% |
| mpc | case0_k12 | reduced_iter_1f2s | 304.1 | 299.9 | -1.4% |
| mpc | case0_k25 | reduced_assemble | 59.2 | 65.2 | +10.0% |
| mpc | case0_k25 | reduced_factor | 188.4 | 190.5 | +1.1% |
| mpc | case0_k25 | reduced_solve | 58.4 | 43.9 | -24.8% |
| mpc | case0_k25 | reduced_iter_1f2s | 303.1 | 274.4 | -9.5% |
| trajopt | case0_k12 | reduced_assemble | 62.1 | 61.4 | -1.1% |
| trajopt | case0_k12 | reduced_factor | 248.8 | 247.5 | -0.5% |
| trajopt | case0_k12 | reduced_solve | 85.6 | 56.3 | -34.2% |
| trajopt | case0_k12 | reduced_iter_1f2s | 416.4 | 355.3 | -14.7% |
| trajopt | case0_k25 | reduced_assemble | 62.2 | 61.4 | -1.3% |
| trajopt | case0_k25 | reduced_factor | 249.5 | 247.8 | -0.7% |
| trajopt | case0_k25 | reduced_solve | 85.8 | 56.3 | -34.4% |
| trajopt | case0_k25 | reduced_iter_1f2s | 415.6 | 355.3 | -14.5% |

## synthetic sweep

| case | n | p | m | factor us | solve us | iter us (before -> after, change) | res before/after | cw before/after |
|---|---|---|---|---|---|---|---|---|
| T5_x6u3_block | 51 | 36 | 108 | 47.1 -> 43.6 | 18.4 -> 17.4 | 78.8 -> 73.7 (-6.5%) | 8.9e-16 / 1.2e-15 | 3.7e-13 / 3.0e-13 |
| T10_x6u3_block | 96 | 66 | 198 | 72.7 -> 66.6 | 27.2 -> 21.3 | 121.9 -> 104.5 (-14.3%) | 8.6e-16 / 7.5e-16 | 5.3e-13 / 4.7e-13 |
| T20_x6u3_block | 186 | 126 | 378 | 154.4 -> 142.1 | 55.3 -> 33.8 | 261.1 -> 205.8 (-21.2%) | 9.6e-16 / 1.1e-15 | 1.0e-12 / 1.0e-12 |
| T40_x6u3_block | 366 | 246 | 738 | 352.3 -> 323.6 | 128.0 -> 58.4 | 603.1 -> 436.9 (-27.6%) | 1.1e-15 / 7.6e-16 | 1.1e-12 / 1.2e-12 |
| T80_x6u3_block | 726 | 486 | 1458 | 802.6 -> 806.9 | 295.6 -> 106.5 | 1388.5 -> 1016.8 (-26.8%) | 1.2e-15 / 1.2e-15 | 1.5e-12 / 2.2e-12 |
| T20_x2u1_block | 62 | 42 | 126 | 43.0 -> 43.6 | 17.4 -> 17.4 | 73.7 -> 73.5 (-0.4%) | 3.8e-14 / 3.5e-14 | 9.0e-13 / 5.7e-13 |
| T20_x4u2_block | 124 | 84 | 252 | 89.9 -> 90.1 | 31.7 -> 25.3 | 148.5 -> 136.0 (-8.4%) | 4.2e-15 / 4.0e-15 | 1.1e-12 / 9.3e-13 |
| T20_x12u4_block | 332 | 252 | 672 | 289.8 -> 290.8 | 105.0 -> 54.3 | 496.6 -> 395.3 (-20.4%) | 5.6e-16 / 4.8e-16 | 6.9e-13 / 6.9e-13 |
| T20_x20u8_block | 580 | 420 | 1176 | 604.2 -> 607.1 | 221.2 -> 89.1 | 1042.4 -> 781.0 (-25.1%) | 5.7e-16 / 6.2e-16 | 4.8e-13 / 5.5e-13 |
| T20_x32u8_block | 832 | 672 | 1680 | 1033.2 -> 1036.2 | 361.5 -> 120.8 | 1750.0 -> 1273.5 (-27.2%) | 3.1e-16 / 3.2e-16 | 2.7e-13 / 4.3e-13 |
| T20_x8u4_banded | 248 | 168 | 504 | 198.7 -> 199.7 | 71.7 -> 42.0 | 337.9 -> 278.5 (-17.6%) | 1.3e-15 / 1.5e-15 | 1.3e-12 / 1.4e-12 |
| T20_x8u4_sparse0.2 | 248 | 168 | 504 | 198.6 -> 198.7 | 71.5 -> 42.0 | 336.7 -> 278.4 (-17.3%) | 1.8e-15 / 2.1e-15 | 1.9e-12 / 2.7e-12 |
| T20_x8u4_sparse0.5 | 248 | 168 | 504 | 198.7 -> 199.7 | 71.7 -> 42.0 | 337.7 -> 278.5 (-17.5%) | 1.4e-15 / 1.1e-15 | 9.5e-13 / 1.3e-12 |
| T20_x8u4_global0.05 | 248 | 168 | 504 | 199.7 -> 200.6 | 71.7 -> 42.0 | 337.9 -> 280.3 (-17.1%) | 3.7e-16 / 3.8e-16 | 8.8e-13 / 4.7e-13 |
| T20_x8u4_global0.3 | 248 | 168 | 126 | 202.8 -> 203.5 | 71.5 -> 42.0 | 341.0 -> 282.6 (-17.1%) | 2.8e-15 / 2.5e-15 | 7.0e-13 / 7.2e-13 |
| T20_x8u4_rows120 | 248 | 168 | 2520 | 204.8 -> 205.6 | 73.7 -> 44.0 | 347.7 -> 289.5 (-16.7%) | 8.8e-17 / 1.0e-16 | 9.9e-14 / 1.3e-13 |
| edge_p0 | 248 | 0 | 504 | 198.7 -> 199.7 | 71.7 -> 42.0 | 337.7 -> 279.3 (-17.3%) | 1.1e-16 / 8.6e-17 | 1.1e-15 / 7.3e-16 |
| edge_m0 | 248 | 168 | 0 | 197.6 -> 198.7 | 70.7 -> 41.0 | 335.4 -> 276.5 (-17.6%) | 2.2e-15 / 3.7e-15 | 5.0e-12 / 3.3e-12 |
| edge_boxonly | 248 | 0 | 0 | 192.5 -> 193.2 | 69.4 -> 39.9 | 327.7 -> 269.3 (-17.8%) | 1.3e-16 / 1.3e-16 | 3.4e-16 / 3.4e-16 |
| edge_inactive0.9 | 248 | 168 | 504 | 199.5 -> 199.7 | 71.7 -> 42.0 | 337.9 -> 279.4 (-17.3%) | 3.3e-15 / 2.1e-15 | 3.7e-12 / 3.2e-12 |
| edge_zeros0.8 | 248 | 168 | 504 | 199.4 -> 199.7 | 71.7 -> 42.0 | 337.9 -> 279.5 (-17.3%) | 1.3e-15 / 1.4e-15 | 1.9e-12 / 2.4e-12 |
| edge_n1 | 1 | 0 | 3 | 19.5 -> 19.5 | 9.2 -> 9.2 | 33.8 -> 33.8 (+0.0%) | 4.3e-17 / 4.3e-17 | 5.6e-17 / 5.6e-17 |
| edge_n33 | 33 | 27 | 12 | 41.6 -> 41.7 | 16.1 -> 17.4 | 68.6 -> 71.7 (+4.5%) | 3.4e-15 / 3.4e-15 | 4.3e-13 / 8.9e-13 |
| near_limit_T100_x16u4 | 2016 | 1616 | 808 | 6442.5 -> 6448.1 | 1516.5 -> 279.6 | 9477.9 -> 7014.0 (-26.0%) | 2.5e-15 / 2.1e-15 | 2.7e-12 / 3.2e-12 |
| near_limit_T50_x30u10_sparse | 2030 | 1530 | 1020 | 6718.5 -> 6732.6 | 1548.9 -> 285.3 | 9810.9 -> 7296.0 (-25.6%) | 2.2e-15 / 2.4e-15 | 4.6e-12 / 7.8e-12 |
| ill_T20_x6u3_block | 186 | 126 | 378 | 141.3 -> 142.3 | 50.9 -> 33.8 | 240.5 -> 206.4 (-14.2%) | 3.7e-17 / 3.3e-17 | 2.1e-11 / 3.0e-11 |
| ill_T40_x12u4_sparse | 652 | 492 | 1312 | 696.3 -> 697.3 | 257.0 -> 97.9 | 1207.9 -> 889.8 (-26.3%) | 2.7e-17 / 2.0e-17 | 6.2e-12 / 5.5e-12 |
| ill_T80_x6u3_global | 726 | 486 | 324 | 806.9 -> 807.9 | 296.6 -> 110.6 | 1396.7 -> 1015.7 (-27.3%) | 3.8e-11 / 3.4e-11 | 2.9e-07 / 3.6e-07 |

sweep iteration time: 25 faster, 2 same (within 3%), 1 slower

## real-state accuracy (all IPM iterations of case 0)

- mpc res_random: before max 2.35e-18 median 1.18e-24; after max 1.67e-18 median 1.03e-24; after/before per state median 0.96 max 1.71
- mpc cw_random: before max 7.15e-06 median 4.10e-07; after max 5.63e-06 median 4.10e-07; after/before per state median 1.00 max 1.47
- mpc res_residual: before max 1.01e-15 median 1.09e-17; after max 7.21e-16 median 7.63e-18; after/before per state median 0.91 max 2.25
- mpc cw_residual: before max 4.95e-06 median 1.58e-09; after max 9.53e-06 median 3.12e-09; after/before per state median 1.01 max 1.98
- trajopt res_random: before max 3.03e-18 median 8.92e-24; after max 2.98e-18 median 7.49e-24; after/before per state median 0.99 max 1.55
- trajopt cw_random: before max 8.23e-06 median 8.05e-08; after max 4.94e-06 median 8.48e-08; after/before per state median 0.96 max 2.10
- trajopt res_residual: before max 3.78e-16 median 2.00e-20; after max 3.88e-16 median 2.47e-20; after/before per state median 1.07 max 4.99
- trajopt cw_residual: before max 1.70e-06 median 7.53e-11; after max 1.71e-06 median 6.34e-11; after/before per state median 1.00 max 1.78
