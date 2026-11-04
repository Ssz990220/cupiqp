| kernel | grid x block | n | us | SM % | mem % | FP64 pipe % (max SM) | issue % (max SM) | occ ach / th % | regs | spill ld/st | sectors/req (gld) | bank confl ld/st | top stalls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| reduce_rhs_kernel | 241 x 128 | 2 | 14.75 | 13.2 | 39.0 | 31.7 | 4.0 | 31.1 / 100.0 | 38 | 0/0 | 10.66 | 0/16 | long_scoreboard 72%; barrier 9%; short_scoreboard 7%; imc_miss 4% |
| trsv_flow_kernel | 16 x 32 | 2 | 44.82 | 0.7 | 4.8 | 4.1 | 1.1 | 2.1 / 25.0 | 168 | 0/0 | 8.85 | 0/0 | long_scoreboard 56%; branch_resolving 22%; wait 9%; short_scoreboard 6% |
| recover_kernel | 807 x 256 | 2 | 11.42 | 56.6 | 40.8 | 59.4 | 11.5 | 85.0 / 100.0 | 40 | 0/0 | 2.65 | 0/0 | long_scoreboard 50%; short_scoreboard 31%; tex_throttle 9%; imc_miss 3% |
