| kernel | grid x block | n | us | SM % | mem % | FP64 pipe % (max SM) | issue % (max SM) | occ ach / th % | regs | spill ld/st | sectors/req (gld) | bank confl ld/st | top stalls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| weights_kernel | 11 x 256 | 9 | 2.46 | 2.3 | 8.3 | 12.6 | 1.1 | 14.8 / 100.0 | 16 | 0/0 | 17.45 | 0/0 | long_scoreboard 48%; imc_miss 24%; short_scoreboard 17%; no_instructions 4% |
| compact_rows_kernel | 324 x 256 | 9 | 7.97 | 14.3 | 60.0 | 19.4 | 13.1 | 62.6 / 100.0 | 40 | 0/0 | 4.45 | 0/0 | long_scoreboard 85%; imc_miss 5%; wait 4%; short_scoreboard 3% |
| assemble_kernel<16> | 424 x 512 | 9 | 57.76 | 18.4 | 26.7 | 30.5 | 20.5 | 90.5 / 100.0 | 40 | 0/0 | 5.71 | 3055/11440 | barrier 49%; long_scoreboard 19%; short_scoreboard 9%; lg_throttle 7% |
| reduce_split_rows_kernel | 127 x 256 | 9 | 3.65 | 5.3 | 21.0 | 8.5 | 4.0 | 21.5 / 100.0 | 40 | 0/0 | 3.04 | 0/0 | long_scoreboard 65%; imc_miss 16%; drain 7%; wait 4% |
| chol_panel_kernel | 6 x 32 | 3 | 19.23 | 4.3 | 1.7 | 43.4 | 2.4 | 2.1 / 20.8 | 164 | 0/0 | 6.05 | 0/0 | short_scoreboard 65%; wait 16%; imc_miss 7%; long_scoreboard 5% |
| chol_update_kernel | 100 x 256 | 3 | 5.89 | 29.4 | 9.0 | 57.6 | 7.9 | 19.2 / 100.0 | 36 | 0/0 | 10.74 | 0/0 | short_scoreboard 52%; long_scoreboard 23%; imc_miss 12%; wait 6% |
| chol_panel_kernel | 5 x 32 | 2 | 19.22 | 3.6 | 1.6 | 43.5 | 2.4 | 2.1 / 20.8 | 164 | 0/0 | 5.89 | 0/0 | short_scoreboard 65%; wait 17%; imc_miss 6%; long_scoreboard 6% |
| chol_update_kernel | 64 x 256 | 2 | 5.82 | 19.2 | 5.9 | 58.0 | 8.0 | 18.7 / 100.0 | 36 | 0/0 | 10.33 | 0/0 | short_scoreboard 50%; long_scoreboard 24%; imc_miss 12%; wait 6% |
| chol_panel_kernel | 4 x 32 | 2 | 19.25 | 2.8 | 1.5 | 43.4 | 2.4 | 2.1 / 20.8 | 164 | 0/0 | 5.64 | 0/0 | short_scoreboard 71%; wait 12%; imc_miss 6%; long_scoreboard 6% |
| chol_update_kernel | 36 x 256 | 2 | 4.14 | 15.5 | 4.8 | 41.1 | 5.7 | 16.2 / 100.0 | 36 | 0/0 | 9.69 | 0/0 | short_scoreboard 49%; long_scoreboard 25%; imc_miss 12%; wait 7% |
| chol_panel_kernel | 3 x 32 | 2 | 19.41 | 2.0 | 1.3 | 43.1 | 2.3 | 2.1 / 20.8 | 164 | 0/0 | 5.20 | 0/0 | short_scoreboard 64%; wait 13%; long_scoreboard 7%; imc_miss 5% |
| chol_update_kernel | 16 x 256 | 2 | 4.10 | 6.7 | 2.2 | 41.4 | 5.7 | 15.8 / 100.0 | 36 | 0/0 | 8.47 | 0/0 | short_scoreboard 49%; long_scoreboard 21%; imc_miss 12%; wait 5% |
| chol_panel_kernel | 2 x 32 | 2 | 19.44 | 1.2 | 4.1 | 43.0 | 2.3 | 2.1 / 20.8 | 164 | 0/0 | 4.17 | 0/0 | short_scoreboard 62%; wait 19%; imc_miss 8%; no_instructions 6% |
| chol_update_kernel | 4 x 256 | 2 | 3.89 | 0.9 | 0.9 | 43.4 | 6.0 | 13.9 / 100.0 | 36 | 0/0 | 4.72 | 0/0 | short_scoreboard 35%; imc_miss 16%; long_scoreboard 16%; wait 16% |
| chol_panel_kernel | 1 x 32 | 2 | 12.98 | 0.7 | 1.5 | 38.5 | 2.8 | 2.1 / 20.8 | 164 | 0/0 | 1.89 | 0/0 | short_scoreboard 57%; wait 17%; no_instructions 11%; imc_miss 9% |
| reduce_rhs_kernel | 169 x 128 | 2 | 9.26 | 26.1 | 49.6 | 37.7 | 4.0 | 23.5 / 100.0 | 40 | 0/0 | 5.89 | 0/0 | long_scoreboard 68%; barrier 12%; short_scoreboard 10%; imc_miss 6% |
| trsv_flow_kernel | 12 x 32 | 2 | 36.06 | 0.5 | 1.5 | 4.0 | 1.1 | 2.1 / 25.0 | 168 | 0/0 | 7.57 | 0/0 | long_scoreboard 57%; branch_resolving 20%; wait 9%; short_scoreboard 6% |
| recover_kernel | 324 x 256 | 2 | 6.75 | 36.6 | 23.3 | 39.0 | 8.3 | 79.2 / 100.0 | 40 | 0/0 | 2.15 | 0/0 | long_scoreboard 46%; short_scoreboard 28%; tex_throttle 12%; imc_miss 5% |
