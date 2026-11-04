| kernel | grid x block | n | us | SM % | mem % | FP64 pipe % (max SM) | issue % (max SM) | occ ach / th % | regs | spill ld/st | sectors/req (gld) | bank confl ld/st | top stalls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| weights_kernel | 26 x 256 | 8 | 2.69 | 5.2 | 18.1 | 11.6 | 1.0 | 14.5 / 100.0 | 16 | 0/0 | 17.68 | 0/0 | long_scoreboard 50%; imc_miss 23%; short_scoreboard 17%; no_instructions 4% |
| compact_rows_kernel | 807 x 256 | 8 | 7.65 | 18.0 | 56.7 | 20.4 | 21.8 | 81.9 / 100.0 | 40 | 0/0 | 2.71 | 0/0 | long_scoreboard 77%; wait 6%; imc_miss 6%; short_scoreboard 5% |
| assemble_kernel<16> | 602 x 512 | 8 | 55.14 | 45.4 | 37.5 | 56.9 | 39.6 | 85.9 / 100.0 | 40 | 0/0 | 4.12 | 6230/40815 | short_scoreboard 31%; long_scoreboard 16%; barrier 14%; wait 12% |
| reduce_split_rows_kernel | 215 x 256 | 8 | 6.85 | 5.2 | 19.0 | 27.4 | 6.4 | 34.5 / 100.0 | 40 | 0/0 | 3.05 | 0/0 | long_scoreboard 65%; drain 14%; imc_miss 12%; wait 3% |
| chol_panel_kernel | 8 x 32 | 2 | 19.30 | 5.9 | 1.7 | 43.3 | 2.3 | 2.1 / 20.8 | 164 | 0/0 | 6.37 | 0/0 | short_scoreboard 69%; wait 14%; imc_miss 7%; long_scoreboard 5% |
| chol_update_kernel | 196 x 256 | 2 | 9.63 | 34.5 | 10.5 | 70.1 | 9.6 | 34.3 / 100.0 | 36 | 0/0 | 11.66 | 0/0 | short_scoreboard 61%; long_scoreboard 20%; imc_miss 7%; wait 4% |
| chol_panel_kernel | 7 x 32 | 2 | 19.33 | 5.1 | 1.8 | 43.2 | 2.3 | 2.1 / 20.8 | 164 | 0/0 | 6.31 | 0/0 | short_scoreboard 68%; wait 14%; imc_miss 7%; long_scoreboard 6% |
| chol_update_kernel | 144 x 256 | 2 | 7.71 | 32.2 | 9.8 | 65.9 | 9.1 | 28.3 / 100.0 | 36 | 0/0 | 11.51 | 0/0 | short_scoreboard 57%; long_scoreboard 20%; imc_miss 7%; wait 7% |
| chol_panel_kernel | 6 x 32 | 2 | 19.23 | 4.3 | 1.7 | 43.5 | 2.4 | 2.1 / 20.8 | 164 | 0/0 | 6.23 | 0/0 | short_scoreboard 67%; wait 15%; imc_miss 8%; long_scoreboard 4% |
| chol_update_kernel | 100 x 256 | 2 | 5.90 | 29.8 | 9.1 | 57.5 | 7.9 | 19.6 / 100.0 | 36 | 0/0 | 11.32 | 0/0 | short_scoreboard 52%; long_scoreboard 20%; imc_miss 11%; wait 7% |
| chol_panel_kernel | 5 x 32 | 2 | 19.34 | 3.5 | 1.6 | 43.2 | 2.3 | 2.1 / 20.8 | 164 | 0/0 | 6.11 | 0/0 | short_scoreboard 64%; wait 17%; imc_miss 6%; long_scoreboard 6% |
| chol_update_kernel | 64 x 256 | 2 | 5.84 | 19.8 | 6.0 | 58.1 | 8.0 | 18.9 / 100.0 | 36 | 0/0 | 11.04 | 0/0 | short_scoreboard 49%; long_scoreboard 22%; imc_miss 13%; wait 7% |
| chol_panel_kernel | 4 x 32 | 2 | 19.33 | 2.8 | 1.6 | 43.2 | 2.3 | 2.1 / 20.8 | 164 | 0/0 | 5.93 | 0/0 | short_scoreboard 64%; wait 16%; long_scoreboard 7%; imc_miss 6% |
| chol_update_kernel | 36 x 256 | 2 | 4.11 | 16.2 | 5.0 | 41.1 | 5.7 | 16.1 / 100.0 | 36 | 0/0 | 10.60 | 0/0 | short_scoreboard 52%; long_scoreboard 21%; imc_miss 13%; wait 5% |
| chol_panel_kernel | 3 x 32 | 2 | 19.36 | 2.0 | 1.5 | 43.1 | 2.3 | 2.1 / 20.8 | 164 | 0/0 | 5.60 | 0/0 | short_scoreboard 62%; wait 19%; imc_miss 5%; long_scoreboard 5% |
| chol_update_kernel | 16 x 256 | 2 | 4.02 | 7.6 | 2.4 | 42.2 | 5.8 | 16.1 / 100.0 | 36 | 0/0 | 9.80 | 0/0 | short_scoreboard 48%; long_scoreboard 21%; imc_miss 14%; wait 10% |
| chol_panel_kernel | 2 x 32 | 2 | 19.33 | 1.2 | 1.4 | 43.2 | 2.3 | 2.1 / 20.8 | 164 | 0/0 | 4.83 | 0/0 | short_scoreboard 53%; wait 28%; imc_miss 9%; no_instructions 7% |
| chol_update_kernel | 4 x 256 | 1 | 4.06 | 1.6 | 0.9 | 41.5 | 5.7 | 13.6 / 100.0 | 36 | 0/0 | 7.61 | 0/0 | short_scoreboard 39%; no_instructions 22%; long_scoreboard 17%; wait 17% |
| chol_panel_kernel | 1 x 32 | 1 | 12.93 | 0.7 | 1.6 | 38.6 | 2.8 | 2.1 / 20.8 | 164 | 0/0 | 2.88 | 0/0 | short_scoreboard 50%; wait 27%; imc_miss 12%; no_instructions 12% |
| reduce_rhs_kernel | 241 x 128 | 2 | 12.61 | 15.3 | 33.3 | 36.7 | 3.7 | 30.9 / 100.0 | 40 | 0/0 | 6.00 | 0/0 | long_scoreboard 65%; barrier 12%; short_scoreboard 11%; imc_miss 6% |
| trsv_flow_kernel | 16 x 32 | 2 | 44.80 | 0.7 | 1.6 | 4.1 | 1.1 | 2.1 / 25.0 | 168 | 0/0 | 8.80 | 0/0 | long_scoreboard 58%; branch_resolving 22%; wait 8%; short_scoreboard 5% |
| recover_kernel | 807 x 256 | 2 | 10.82 | 58.6 | 42.4 | 67.6 | 12.9 | 85.7 / 100.0 | 40 | 0/0 | 2.39 | 0/0 | long_scoreboard 46%; short_scoreboard 35%; tex_throttle 10%; imc_miss 4% |
