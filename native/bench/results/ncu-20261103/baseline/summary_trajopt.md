| kernel | grid x block | n | us | SM % | mem % | FP64 pipe % (max SM) | issue % (max SM) | occ ach / th % | regs | spill ld/st | sectors/req (gld) | bank confl ld/st | top stalls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| weights_kernel | 26 x 256 | 8 | 2.62 | 5.3 | 18.5 | 11.8 | 1.0 | 15.3 / 100.0 | 16 | 0/0 | 17.68 | 0/0 | long_scoreboard 51%; imc_miss 23%; short_scoreboard 15%; wait 4% |
| compact_rows_kernel | 807 x 256 | 8 | 7.68 | 17.9 | 56.6 | 20.8 | 21.7 | 83.1 / 100.0 | 40 | 0/0 | 2.71 | 0/0 | long_scoreboard 78%; imc_miss 6%; wait 5%; short_scoreboard 5% |
| assemble_kernel<16> | 602 x 512 | 8 | 62.18 | 40.8 | 33.2 | 51.2 | 32.9 | 86.8 / 100.0 | 40 | 0/0 | 4.12 | 5820/38523 | long_scoreboard 32%; short_scoreboard 22%; barrier 14%; wait 11% |
| reduce_split_rows_kernel | 215 x 256 | 8 | 6.91 | 5.1 | 19.0 | 27.2 | 6.4 | 35.1 / 100.0 | 40 | 0/0 | 3.05 | 0/0 | long_scoreboard 66%; imc_miss 13%; drain 12%; wait 3% |
| chol_panel_kernel | 8 x 32 | 2 | 26.02 | 5.0 | 1.5 | 36.8 | 1.8 | 2.1 / 20.8 | 164 | 0/0 | 6.37 | 0/0 | short_scoreboard 59%; wait 15%; no_instructions 13%; long_scoreboard 8% |
| chol_update_kernel | 196 x 256 | 2 | 9.58 | 34.6 | 10.5 | 70.4 | 9.7 | 35.0 / 100.0 | 36 | 0/0 | 11.66 | 0/0 | short_scoreboard 60%; long_scoreboard 19%; imc_miss 6%; wait 6% |
| chol_panel_kernel | 7 x 32 | 2 | 26.00 | 4.4 | 1.5 | 36.8 | 1.8 | 2.1 / 20.8 | 164 | 0/0 | 6.31 | 0/0 | short_scoreboard 59%; no_instructions 14%; wait 12%; long_scoreboard 8% |
| chol_update_kernel | 144 x 256 | 2 | 7.73 | 32.1 | 9.8 | 65.6 | 9.0 | 28.1 / 100.0 | 36 | 0/0 | 11.51 | 0/0 | short_scoreboard 58%; long_scoreboard 19%; imc_miss 8%; wait 5% |
| chol_panel_kernel | 6 x 32 | 2 | 25.95 | 3.7 | 1.5 | 36.9 | 1.8 | 2.1 / 20.8 | 164 | 0/0 | 6.23 | 0/0 | short_scoreboard 61%; wait 13%; no_instructions 12%; long_scoreboard 8% |
| chol_update_kernel | 100 x 256 | 2 | 5.89 | 29.9 | 9.1 | 57.7 | 7.9 | 19.4 / 100.0 | 36 | 0/0 | 11.32 | 0/0 | short_scoreboard 52%; long_scoreboard 24%; imc_miss 12%; wait 8% |
| chol_panel_kernel | 5 x 32 | 2 | 26.06 | 3.0 | 1.5 | 36.7 | 1.8 | 2.1 / 20.8 | 164 | 0/0 | 6.11 | 0/0 | short_scoreboard 60%; no_instructions 13%; wait 13%; long_scoreboard 6% |
| chol_update_kernel | 64 x 256 | 2 | 5.82 | 19.8 | 6.1 | 58.2 | 8.0 | 18.9 / 100.0 | 36 | 0/0 | 11.04 | 0/0 | short_scoreboard 48%; long_scoreboard 20%; imc_miss 15%; wait 6% |
| chol_panel_kernel | 4 x 32 | 2 | 26.14 | 2.4 | 1.4 | 36.7 | 1.8 | 2.1 / 20.8 | 164 | 0/0 | 5.93 | 0/0 | short_scoreboard 57%; wait 15%; no_instructions 14%; long_scoreboard 6% |
| chol_update_kernel | 36 x 256 | 2 | 4.14 | 16.2 | 5.0 | 41.0 | 5.7 | 16.0 / 100.0 | 36 | 0/0 | 10.60 | 0/0 | short_scoreboard 50%; long_scoreboard 23%; imc_miss 13%; wait 7% |
| chol_panel_kernel | 3 x 32 | 2 | 26.24 | 1.7 | 1.4 | 36.5 | 1.8 | 2.1 / 20.8 | 164 | 0/0 | 5.60 | 0/0 | short_scoreboard 58%; no_instructions 16%; wait 13%; long_scoreboard 6% |
| chol_update_kernel | 16 x 256 | 2 | 4.03 | 7.6 | 2.4 | 42.2 | 5.8 | 15.8 / 100.0 | 36 | 0/0 | 9.80 | 0/0 | short_scoreboard 52%; long_scoreboard 18%; imc_miss 11%; wait 7% |
| chol_panel_kernel | 2 x 32 | 2 | 26.06 | 1.1 | 1.1 | 36.7 | 1.8 | 2.1 / 20.8 | 164 | 0/0 | 4.83 | 0/0 | short_scoreboard 55%; wait 19%; no_instructions 14%; imc_miss 5% |
| chol_update_kernel | 4 x 256 | 1 | 3.87 | 1.7 | 1.0 | 43.7 | 6.0 | 14.8 / 100.0 | 36 | 0/0 | 7.61 | 0/0 | long_scoreboard 35%; short_scoreboard 30%; imc_miss 20%; wait 15% |
| chol_panel_kernel | 1 x 32 | 1 | 18.82 | 0.6 | 1.3 | 33.0 | 2.1 | 2.1 / 20.8 | 164 | 0/0 | 2.88 | 0/0 | short_scoreboard 51%; wait 20%; no_instructions 17%; branch_resolving 7% |
| reduce_rhs_kernel | 241 x 128 | 2 | 14.64 | 13.3 | 39.7 | 31.9 | 4.0 | 30.7 / 100.0 | 38 | 0/0 | 10.66 | 0/10 | long_scoreboard 70%; barrier 10%; short_scoreboard 6%; imc_miss 5% |
| trsv_kernel<512> | 1 x 512 | 2 | 89.26 | 0.5 | 0.9 | 26.4 | 5.3 | 33.3 / 33.3 | 102 | 0/0 | 14.68 | 3648/0 | barrier 70%; short_scoreboard 14%; long_scoreboard 8%; wait 3% |
| recover_kernel | 807 x 256 | 2 | 11.23 | 57.7 | 40.8 | 60.4 | 11.7 | 86.1 / 100.0 | 40 | 0/0 | 2.65 | 0/0 | long_scoreboard 48%; short_scoreboard 31%; tex_throttle 10%; imc_miss 3% |
