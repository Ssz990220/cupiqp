| kernel | grid x block | n | us | SM % | mem % | FP64 pipe % (max SM) | issue % (max SM) | occ ach / th % | regs | spill ld/st | sectors/req (gld) | bank confl ld/st | top stalls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| reduce_rhs_kernel | 169 x 128 | 2 | 12.77 | 19.2 | 65.9 | 27.7 | 3.5 | 24.4 / 100.0 | 38 | 0/0 | 10.64 | 0/12 | long_scoreboard 65%; barrier 17%; short_scoreboard 6%; imc_miss 4% |
| trsv_flow_kernel | 12 x 32 | 2 | 36.16 | 0.5 | 1.5 | 4.0 | 1.1 | 2.1 / 25.0 | 168 | 0/0 | 7.63 | 0/0 | long_scoreboard 59%; branch_resolving 19%; wait 8%; short_scoreboard 6% |
| recover_kernel | 324 x 256 | 2 | 10.59 | 30.2 | 60.4 | 31.5 | 5.8 | 73.1 / 100.0 | 40 | 0/0 | 4.80 | 0/0 | long_scoreboard 69%; short_scoreboard 19%; tex_throttle 3%; imc_miss 3% |
