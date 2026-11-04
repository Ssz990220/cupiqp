## microbench (graph replay median, us)

| dataset | state | op | before | after | change |
|---|---|---|---|---|---|
| mpc | case0_k12 | reduced_assemble | 65.2 | 57.3 | -12.1% |
| mpc | case0_k12 | reduced_factor | 208.7 | 173.7 | -16.8% |
| mpc | case0_k12 | reduced_solve | 64.5 | 42.7 | -33.8% |
| mpc | case0_k12 | reduced_iter_1f2s | 304.1 | 255.6 | -16.0% |
| mpc | case0_k25 | reduced_assemble | 59.2 | 53.1 | -10.4% |
| mpc | case0_k25 | reduced_factor | 188.4 | 160.7 | -14.7% |
| mpc | case0_k25 | reduced_solve | 58.4 | 38.9 | -33.3% |
| mpc | case0_k25 | reduced_iter_1f2s | 303.1 | 234.5 | -22.6% |
| trajopt | case0_k12 | reduced_assemble | 62.1 | 56.3 | -9.3% |
| trajopt | case0_k12 | reduced_factor | 248.8 | 210.8 | -15.3% |
| trajopt | case0_k12 | reduced_solve | 85.6 | 53.2 | -37.8% |
| trajopt | case0_k12 | reduced_iter_1f2s | 416.4 | 314.4 | -24.5% |
| trajopt | case0_k25 | reduced_assemble | 62.2 | 57.0 | -8.5% |
| trajopt | case0_k25 | reduced_factor | 249.5 | 211.6 | -15.2% |
| trajopt | case0_k25 | reduced_solve | 85.8 | 53.2 | -37.9% |
| trajopt | case0_k25 | reduced_iter_1f2s | 415.6 | 313.9 | -24.5% |

## synthetic sweep

| case | n | p | m | factor us | solve us | iter us (before -> after, change) | res before/after | cw before/after |
|---|---|---|---|---|---|---|---|---|
| T5_x6u3_block | 51 | 36 | 108 | 47.1 -> 34.8 | 18.4 -> 17.4 | 78.8 -> 65.5 (-16.9%) | 8.9e-16 / 1.1e-15 | 3.7e-13 / 2.2e-13 |
| T10_x6u3_block | 96 | 66 | 198 | 72.7 -> 54.3 | 27.2 -> 21.5 | 121.9 -> 92.2 (-24.4%) | 8.6e-16 / 7.7e-16 | 5.3e-13 / 5.2e-13 |
| T20_x6u3_block | 186 | 126 | 378 | 154.4 -> 118.8 | 55.3 -> 33.8 | 261.1 -> 181.2 (-30.6%) | 9.6e-16 / 9.8e-16 | 1.0e-12 / 1.0e-12 |
| T40_x6u3_block | 366 | 246 | 738 | 352.3 -> 274.4 | 128.0 -> 58.4 | 603.1 -> 387.9 (-35.7%) | 1.1e-15 / 6.8e-16 | 1.1e-12 / 9.8e-13 |
| T80_x6u3_block | 726 | 486 | 1458 | 802.6 -> 713.3 | 295.6 -> 106.5 | 1388.5 -> 923.6 (-33.5%) | 1.2e-15 / 1.6e-15 | 1.5e-12 / 1.6e-12 |
| T20_x2u1_block | 62 | 42 | 126 | 43.0 -> 34.8 | 17.4 -> 17.4 | 73.7 -> 65.5 (-11.1%) | 3.8e-14 / 2.3e-14 | 9.0e-13 / 8.0e-13 |
| T20_x4u2_block | 124 | 84 | 252 | 89.9 -> 73.5 | 31.7 -> 25.4 | 148.5 -> 119.4 (-19.6%) | 4.2e-15 / 4.2e-15 | 1.1e-12 / 9.2e-13 |
| T20_x12u4_block | 332 | 252 | 672 | 289.8 -> 245.5 | 105.0 -> 55.0 | 496.6 -> 350.9 (-29.3%) | 5.6e-16 / 4.5e-16 | 6.9e-13 / 5.8e-13 |
| T20_x20u8_block | 580 | 420 | 1176 | 604.2 -> 528.0 | 221.2 -> 89.1 | 1042.4 -> 701.4 (-32.7%) | 5.7e-16 / 4.9e-16 | 4.8e-13 / 7.4e-13 |
| T20_x32u8_block | 832 | 672 | 1680 | 1033.2 -> 918.5 | 361.5 -> 121.9 | 1750.0 -> 1155.1 (-34.0%) | 3.1e-16 / 3.5e-16 | 2.7e-13 / 3.3e-13 |
| T20_x8u4_banded | 248 | 168 | 504 | 198.7 -> 166.8 | 71.7 -> 42.0 | 337.9 -> 246.3 (-27.1%) | 1.3e-15 / 1.3e-15 | 1.3e-12 / 1.9e-12 |
| T20_x8u4_sparse0.2 | 248 | 168 | 504 | 198.6 -> 165.9 | 71.5 -> 42.0 | 336.7 -> 245.6 (-27.0%) | 1.8e-15 / 1.9e-15 | 1.9e-12 / 2.1e-12 |
| T20_x8u4_sparse0.5 | 248 | 168 | 504 | 198.7 -> 166.9 | 71.7 -> 42.0 | 337.7 -> 245.8 (-27.2%) | 1.4e-15 / 7.2e-16 | 9.5e-13 / 1.1e-12 |
| T20_x8u4_global0.05 | 248 | 168 | 504 | 199.7 -> 166.9 | 71.7 -> 42.0 | 337.9 -> 246.8 (-27.0%) | 3.7e-16 / 5.8e-16 | 8.8e-13 / 7.2e-13 |
| T20_x8u4_global0.3 | 248 | 168 | 126 | 202.8 -> 170.8 | 71.5 -> 42.0 | 341.0 -> 249.9 (-26.7%) | 2.8e-15 / 2.8e-15 | 7.0e-13 / 6.7e-13 |
| T20_x8u4_rows120 | 248 | 168 | 2520 | 204.8 -> 171.9 | 73.7 -> 44.0 | 347.7 -> 255.7 (-26.5%) | 8.8e-17 / 1.1e-16 | 9.9e-14 / 1.3e-13 |
| edge_p0 | 248 | 0 | 504 | 198.7 -> 166.8 | 71.7 -> 42.0 | 337.7 -> 246.5 (-27.0%) | 1.1e-16 / 9.1e-17 | 1.1e-15 / 8.5e-16 |
| edge_m0 | 248 | 168 | 0 | 197.6 -> 165.7 | 70.7 -> 41.0 | 335.4 -> 244.3 (-27.2%) | 2.2e-15 / 3.8e-15 | 5.0e-12 / 4.0e-12 |
| edge_boxonly | 248 | 0 | 0 | 192.5 -> 160.3 | 69.4 -> 39.9 | 327.7 -> 236.5 (-27.8%) | 1.3e-16 / 1.1e-16 | 3.4e-16 / 2.7e-16 |
| edge_inactive0.9 | 248 | 168 | 504 | 199.5 -> 166.9 | 71.7 -> 41.9 | 337.9 -> 245.8 (-27.3%) | 3.3e-15 / 2.7e-15 | 3.7e-12 / 3.4e-12 |
| edge_zeros0.8 | 248 | 168 | 504 | 199.4 -> 166.9 | 71.7 -> 42.0 | 337.9 -> 246.4 (-27.1%) | 1.3e-15 / 1.6e-15 | 1.9e-12 / 1.7e-12 |
| edge_n1 | 1 | 0 | 3 | 19.5 -> 16.1 | 9.2 -> 9.2 | 33.8 -> 30.5 (-9.6%) | 4.3e-17 / 4.3e-17 | 5.6e-17 / 5.6e-17 |
| edge_n33 | 33 | 27 | 12 | 41.6 -> 33.7 | 16.1 -> 17.4 | 68.6 -> 63.5 (-7.5%) | 3.4e-15 / 3.5e-15 | 4.3e-13 / 5.1e-13 |
| near_limit_T100_x16u4 | 2016 | 1616 | 808 | 6442.5 -> 6177.8 | 1516.5 -> 279.6 | 9477.9 -> 6745.9 (-28.8%) | 2.5e-15 / 1.7e-15 | 2.7e-12 / 4.2e-12 |
| near_limit_T50_x30u10_sparse | 2030 | 1530 | 1020 | 6718.5 -> 6453.1 | 1548.9 -> 285.4 | 9810.9 -> 7022.6 (-28.4%) | 2.2e-15 / 2.7e-15 | 4.6e-12 / 5.6e-12 |
| ill_T20_x6u3_block | 186 | 126 | 378 | 141.3 -> 117.3 | 50.9 -> 33.8 | 240.5 -> 181.2 (-24.6%) | 3.7e-17 / 4.4e-17 | 2.1e-11 / 2.7e-11 |
| ill_T40_x12u4_sparse | 652 | 492 | 1312 | 696.3 -> 610.0 | 257.0 -> 97.3 | 1207.9 -> 802.8 (-33.5%) | 2.7e-17 / 2.6e-17 | 6.2e-12 / 4.9e-12 |
| ill_T80_x6u3_global | 726 | 486 | 324 | 806.9 -> 712.6 | 296.6 -> 106.1 | 1396.7 -> 920.6 (-34.1%) | 3.8e-11 / 2.7e-11 | 2.9e-07 / 3.0e-07 |

sweep iteration time: 28 faster, 0 same (within 3%), 0 slower

## real-state accuracy (all IPM iterations of case 0)

- mpc res_random: before max 2.35e-18 median 1.18e-24; after max 1.94e-18 median 7.20e-25; after/before per state median 1.00 max 1.50
- mpc cw_random: before max 7.15e-06 median 4.10e-07; after max 5.51e-06 median 5.07e-07; after/before per state median 1.00 max 1.84
- mpc res_residual: before max 1.01e-15 median 1.09e-17; after max 1.14e-15 median 9.67e-18; after/before per state median 0.92 max 1.42
- mpc cw_residual: before max 4.95e-06 median 1.58e-09; after max 6.62e-06 median 2.33e-09; after/before per state median 1.01 max 2.24
- trajopt res_random: before max 3.03e-18 median 8.92e-24; after max 3.03e-18 median 7.26e-24; after/before per state median 1.00 max 1.55
- trajopt cw_random: before max 8.23e-06 median 8.05e-08; after max 9.05e-06 median 8.06e-08; after/before per state median 1.00 max 2.10
- trajopt res_residual: before max 3.78e-16 median 2.00e-20; after max 4.49e-16 median 2.52e-20; after/before per state median 1.11 max 2.60
- trajopt cw_residual: before max 1.70e-06 median 7.53e-11; after max 1.56e-06 median 6.93e-11; after/before per state median 1.08 max 2.32
