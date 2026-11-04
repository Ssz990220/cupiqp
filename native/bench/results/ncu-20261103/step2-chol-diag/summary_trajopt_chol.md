| kernel | grid x block | n | us | SM % | mem % | FP64 pipe % (max SM) | issue % (max SM) | occ ach / th % | regs | spill ld/st | sectors/req (gld) | bank confl ld/st | top stalls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| chol_panel_kernel | 8 x 32 | 1 | 19.26 | 5.9 | 1.7 | 43.3 | 2.4 | 2.1 / 20.8 | 164 | 0/0 | 6.37 | 0/0 | short_scoreboard 68%; wait 15%; long_scoreboard 7%; imc_miss 5% |
| chol_update_kernel | 196 x 256 | 1 | 9.63 | 34.6 | 10.5 | 70.3 | 9.7 | 34.8 / 100.0 | 36 | 0/0 | 11.66 | 0/0 | short_scoreboard 59%; long_scoreboard 20%; imc_miss 8%; wait 5% |
| chol_panel_kernel | 7 x 32 | 1 | 19.20 | 5.1 | 1.8 | 43.5 | 2.4 | 2.1 / 20.8 | 164 | 0/0 | 6.31 | 0/0 | short_scoreboard 66%; wait 18%; imc_miss 7%; long_scoreboard 4% |
| chol_update_kernel | 144 x 256 | 1 | 7.74 | 32.0 | 9.7 | 65.5 | 9.0 | 28.2 / 100.0 | 36 | 0/0 | 11.51 | 0/0 | short_scoreboard 57%; long_scoreboard 20%; imc_miss 10%; wait 5% |
| chol_panel_kernel | 6 x 32 | 1 | 19.23 | 4.3 | 1.7 | 43.5 | 2.4 | 2.1 / 20.8 | 164 | 0/0 | 6.23 | 0/0 | short_scoreboard 68%; wait 16%; imc_miss 5%; long_scoreboard 5% |
| chol_update_kernel | 100 x 256 | 1 | 5.86 | 30.0 | 9.1 | 57.8 | 8.0 | 19.4 / 100.0 | 36 | 0/0 | 11.32 | 0/0 | short_scoreboard 53%; long_scoreboard 25%; imc_miss 9%; wait 7% |
| chol_panel_kernel | 5 x 32 | 1 | 19.23 | 3.6 | 1.6 | 43.4 | 2.4 | 2.1 / 20.8 | 164 | 0/0 | 6.11 | 0/0 | short_scoreboard 66%; wait 17%; imc_miss 6%; long_scoreboard 5% |
| chol_update_kernel | 64 x 256 | 1 | 5.82 | 19.8 | 6.0 | 58.1 | 8.0 | 18.8 / 100.0 | 36 | 0/0 | 11.04 | 0/0 | short_scoreboard 53%; long_scoreboard 19%; imc_miss 12%; wait 6% |
| chol_panel_kernel | 4 x 32 | 1 | 19.33 | 2.8 | 1.5 | 43.2 | 2.3 | 2.1 / 20.8 | 164 | 0/0 | 5.93 | 0/0 | short_scoreboard 62%; wait 20%; imc_miss 7%; long_scoreboard 7% |
| chol_update_kernel | 36 x 256 | 1 | 4.13 | 16.3 | 5.0 | 41.2 | 5.7 | 16.1 / 100.0 | 36 | 0/0 | 10.60 | 0/0 | short_scoreboard 45%; long_scoreboard 20%; imc_miss 13%; wait 10% |
| chol_panel_kernel | 3 x 32 | 1 | 19.36 | 2.0 | 1.4 | 43.1 | 2.3 | 2.1 / 20.8 | 164 | 0/0 | 5.60 | 0/0 | short_scoreboard 67%; wait 12%; imc_miss 6%; long_scoreboard 6% |
| chol_update_kernel | 16 x 256 | 1 | 4.00 | 7.7 | 2.4 | 42.5 | 5.9 | 16.0 / 100.0 | 36 | 0/0 | 9.80 | 0/0 | short_scoreboard 52%; long_scoreboard 17%; imc_miss 13%; wait 10% |
| chol_panel_kernel | 2 x 32 | 1 | 19.20 | 1.2 | 1.2 | 43.5 | 2.4 | 2.1 / 20.8 | 164 | 0/0 | 4.83 | 0/0 | short_scoreboard 67%; wait 12%; imc_miss 9%; no_instructions 6% |
| chol_update_kernel | 4 x 256 | 1 | 3.90 | 1.7 | 1.0 | 43.6 | 6.0 | 14.8 / 100.0 | 36 | 0/0 | 7.61 | 0/0 | short_scoreboard 38%; long_scoreboard 33%; imc_miss 14%; wait 10% |
| chol_panel_kernel | 1 x 32 | 1 | 12.93 | 0.7 | 1.6 | 38.6 | 2.8 | 2.1 / 20.8 | 164 | 0/0 | 2.88 | 0/0 | short_scoreboard 74%; no_instructions 11%; imc_miss 7%; wait 7% |
