| kernel | grid x block | n | us | SM % | mem % | FP64 pipe % (max SM) | issue % (max SM) | occ ach / th % | regs | spill ld/st | sectors/req (gld) | bank confl ld/st | top stalls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| weights_kernel | 11 x 256 | 9 | 2.56 | 2.2 | 8.1 | 12.2 | 1.1 | 14.6 / 100.0 | 16 | 0/0 | 17.45 | 0/0 | long_scoreboard 48%; imc_miss 24%; short_scoreboard 16%; wait 5% |
| compact_rows_kernel | 324 x 256 | 9 | 7.94 | 14.4 | 60.3 | 19.6 | 13.2 | 62.5 / 100.0 | 40 | 0/0 | 4.45 | 0/0 | long_scoreboard 84%; imc_miss 5%; short_scoreboard 4%; wait 3% |
| assemble_kernel<16> | 424 x 512 | 9 | 64.67 | 16.6 | 22.3 | 29.0 | 17.5 | 91.2 / 100.0 | 40 | 0/0 | 5.71 | 2785/10588 | barrier 53%; long_scoreboard 21%; short_scoreboard 7%; lg_throttle 6% |
| reduce_split_rows_kernel | 127 x 256 | 9 | 3.74 | 5.1 | 20.5 | 8.2 | 3.9 | 21.4 / 100.0 | 40 | 0/0 | 3.04 | 0/0 | long_scoreboard 65%; imc_miss 17%; drain 6%; wait 5% |
| chol_panel_kernel | 6 x 32 | 3 | 25.95 | 3.7 | 1.6 | 36.9 | 1.8 | 2.1 / 20.8 | 164 | 0/0 | 6.05 | 0/0 | short_scoreboard 61%; wait 15%; no_instructions 12%; long_scoreboard 6% |
| chol_update_kernel | 100 x 256 | 3 | 5.89 | 29.3 | 8.9 | 57.4 | 7.9 | 19.5 / 100.0 | 36 | 0/0 | 10.74 | 0/0 | short_scoreboard 50%; long_scoreboard 22%; imc_miss 10%; wait 7% |
| chol_panel_kernel | 5 x 32 | 2 | 25.95 | 3.1 | 1.5 | 36.9 | 1.8 | 2.1 / 20.8 | 164 | 0/0 | 5.89 | 0/0 | short_scoreboard 61%; no_instructions 13%; wait 13%; long_scoreboard 7% |
| chol_update_kernel | 64 x 256 | 2 | 5.82 | 19.2 | 5.9 | 58.0 | 8.0 | 19.1 / 100.0 | 36 | 0/0 | 10.33 | 0/0 | short_scoreboard 49%; long_scoreboard 22%; imc_miss 12%; wait 6% |
| chol_panel_kernel | 4 x 32 | 2 | 26.05 | 2.4 | 1.2 | 36.7 | 1.8 | 2.1 / 20.8 | 164 | 0/0 | 5.64 | 0/0 | short_scoreboard 56%; no_instructions 17%; wait 12%; long_scoreboard 7% |
| chol_update_kernel | 36 x 256 | 2 | 4.27 | 15.0 | 4.7 | 39.8 | 5.5 | 15.9 / 100.0 | 36 | 0/0 | 9.69 | 0/0 | short_scoreboard 47%; long_scoreboard 27%; imc_miss 14%; wait 5% |
| chol_panel_kernel | 3 x 32 | 2 | 26.16 | 1.7 | 1.3 | 36.6 | 1.8 | 2.1 / 20.8 | 164 | 0/0 | 5.20 | 0/0 | short_scoreboard 59%; no_instructions 15%; wait 13%; imc_miss 5% |
| chol_update_kernel | 16 x 256 | 2 | 4.00 | 6.9 | 2.2 | 42.2 | 5.8 | 15.8 / 100.0 | 36 | 0/0 | 8.47 | 0/0 | short_scoreboard 47%; long_scoreboard 20%; imc_miss 16%; wait 9% |
| chol_panel_kernel | 2 x 32 | 2 | 26.13 | 1.1 | 1.3 | 36.6 | 1.8 | 2.1 / 20.8 | 164 | 0/0 | 4.17 | 0/0 | short_scoreboard 56%; no_instructions 19%; wait 10%; long_scoreboard 7% |
| chol_update_kernel | 4 x 256 | 2 | 3.92 | 0.9 | 0.9 | 43.2 | 6.0 | 13.8 / 100.0 | 36 | 0/0 | 4.72 | 0/0 | short_scoreboard 33%; imc_miss 24%; long_scoreboard 18%; wait 12% |
| chol_panel_kernel | 1 x 32 | 2 | 18.78 | 0.6 | 1.4 | 33.1 | 2.1 | 2.1 / 20.8 | 164 | 0/0 | 1.89 | 0/0 | short_scoreboard 53%; no_instructions 25%; wait 12%; imc_miss 6% |
| reduce_rhs_kernel | 169 x 128 | 2 | 12.75 | 19.2 | 65.8 | 27.7 | 3.5 | 24.4 / 100.0 | 38 | 0/0 | 10.64 | 0/9 | long_scoreboard 66%; barrier 17%; short_scoreboard 6%; imc_miss 4% |
| trsv_kernel<512> | 1 x 512 | 2 | 60.13 | 0.4 | 1.0 | 23.4 | 5.2 | 33.2 / 33.3 | 102 | 0/0 | 12.61 | 2560/0 | barrier 74%; short_scoreboard 10%; long_scoreboard 6%; mio_throttle 3% |
| recover_kernel | 324 x 256 | 2 | 10.32 | 31.1 | 61.8 | 32.3 | 6.0 | 72.3 / 100.0 | 40 | 0/0 | 4.80 | 0/0 | long_scoreboard 69%; short_scoreboard 18%; tex_throttle 4%; imc_miss 3% |
