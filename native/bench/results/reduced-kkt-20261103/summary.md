
### Timing, mpc (29 timed states; median over states of the per-state median, us)

| op | eager us | graph us | graph min..max over states |
|---|---:|---:|---|
| cudss_factor | 883 | 816 | 812..830 |
| cudss_solve | 178 | 119 | 118..152 |
| cudss_iter_1f2s | 1125 | 1040 | 1034..1049 |
| assemble_a_dense_syrk | 580 | 574 | 573..575 |
| assemble_b_sparse | 193 | 191 | 191..192 |
| assemble_c_sparse_skip0 | 129 | 124 | 122..134 |
| copyH | 3 | 3 | 3..3 |
| copyH_potrf | 465 | 456 | 455..472 |
| copyH_tiled_chol | 270 | 245 | 244..263 |
| reduced_solve_recover | 72 | 65 | 65..66 |
| reduced_iter_1a1f2s_a_dense_syrk | 1176 | 1153 | 1150..1153 |
| reduced_iter_1a1f2s_b_sparse | 790 | 770 | 769..771 |
| reduced_iter_1a1f2s_c_sparse_skip0 | 722 | 701 | 698..712 |
| reduced_iter_1a1f2s_b_sparse_tiled_chol | 595 | 561 | 561..562 |
| reduced_iter_1a1f2s_c_sparse_skip0_tiled_chol | 527 | 493 | 490..503 |

Per-iteration speedup vs cuDSS (graph, median): 
- reduced_iter_1a1f2s_a_dense_syrk: 1153 us, cuDSS/reduced = 0.90x
- reduced_iter_1a1f2s_b_sparse: 770 us, cuDSS/reduced = 1.35x
- reduced_iter_1a1f2s_c_sparse_skip0: 701 us, cuDSS/reduced = 1.48x
- reduced_iter_1a1f2s_b_sparse_tiled_chol: 561 us, cuDSS/reduced = 1.85x
- reduced_iter_1a1f2s_c_sparse_skip0_tiled_chol: 493 us, cuDSS/reduced = 2.11x

### Timing, trajopt (104 timed states; median over states of the per-state median, us)

| op | eager us | graph us | graph min..max over states |
|---|---:|---:|---|
| cudss_factor | 821 | 758 | 701..794 |
| cudss_solve | 194 | 137 | 121..171 |
| cudss_iter_1f2s | 1074 | 995 | 967..1025 |
| assemble_a_dense_syrk | 2778 | 2773 | 2767..2780 |
| assemble_b_sparse | 160 | 159 | 157..162 |
| assemble_c_sparse_skip0 | 154 | 153 | 152..155 |
| copyH | 3 | 3 | 3..4 |
| copyH_potrf | 670 | 661 | 659..678 |
| copyH_tiled_chol | 369 | 339 | 338..356 |
| reduced_solve_recover | 102 | 96 | 95..96 |
| reduced_iter_1a1f2s_a_dense_syrk | 3640 | 3617 | 3608..3626 |
| reduced_iter_1a1f2s_b_sparse | 1022 | 1003 | 999..1005 |
| reduced_iter_1a1f2s_c_sparse_skip0 | 1016 | 997 | 994..999 |
| reduced_iter_1a1f2s_b_sparse_tiled_chol | 722 | 689 | 680..690 |
| reduced_iter_1a1f2s_c_sparse_skip0_tiled_chol | 716 | 677 | 674..678 |

Per-iteration speedup vs cuDSS (graph, median): 
- reduced_iter_1a1f2s_a_dense_syrk: 3617 us, cuDSS/reduced = 0.28x
- reduced_iter_1a1f2s_b_sparse: 1003 us, cuDSS/reduced = 0.99x
- reduced_iter_1a1f2s_c_sparse_skip0: 997 us, cuDSS/reduced = 1.00x
- reduced_iter_1a1f2s_b_sparse_tiled_chol: 689 us, cuDSS/reduced = 1.44x
- reduced_iter_1a1f2s_c_sparse_skip0_tiled_chol: 677 us, cuDSS/reduced = 1.47x

### Accuracy over all states (normwise condensed residual; cw = componentwise backward error)

| dataset | config | rhs | states | potrf fails | max res_A | max res_B | max res_B_ir | pass | max cw_A | max cw_B | max cw_B_ir | cw pass | max reldiff(A,B) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mpc | a_dense_syrk+potrf | random | 97 | 0 | 6.1e-18 | 9.6e-18 | 6.0e-20 | 97/97 | 2.8e-05 | 1.8e-05 | 8.2e-10 | 97/97 | 6.6e-13 |
| mpc | a_dense_syrk+potrf | residual | 97 | 0 | 1.0e-15 | 1.7e-15 | 3.1e-17 | 97/97 | 3.1e-05 | 2.5e-05 | 3.0e-10 | 96/97 | 9.1e-05 |
| mpc | b_sparse+potrf | random | 97 | 0 | 6.1e-18 | 3.1e-18 | 2.4e-20 | 97/97 | 2.7e-05 | 1.3e-05 | 5.8e-10 | 97/97 | 5.0e-13 |
| mpc | b_sparse+potrf | residual | 97 | 0 | 1.0e-15 | 1.9e-15 | 1.7e-17 | 97/97 | 1.6e-05 | 1.6e-05 | 5.6e-10 | 97/97 | 8.6e-05 |
| mpc | b_sparse+tiled_chol | random | 97 | 0 | 5.8e-18 | 2.7e-18 | 3.4e-20 | 97/97 | 2.0e-05 | 1.0e-05 | 6.3e-10 | 97/97 | 9.9e-13 |
| mpc | b_sparse+tiled_chol | residual | 97 | 0 | 1.3e-15 | 1.9e-15 | 2.3e-17 | 97/97 | 1.8e-05 | 3.5e-05 | 3.4e-10 | 97/97 | 1.3e-04 |
| trajopt | a_dense_syrk+potrf | random | 359 | 0 | 7.5e-18 | 1.5e-17 | 1.2e-19 | 359/359 | 1.1e-02 | 1.5e-02 | 4.9e-04 | 329/359 | 1.6e-09 |
| trajopt | a_dense_syrk+potrf | residual | 359 | 0 | 4.2e-15 | 6.3e-15 | 3.1e-17 | 356/359 | 1.0e+00 | 3.6e-02 | 7.8e-04 | 357/359 | 1.4e-01 |
| trajopt | b_sparse+potrf | random | 359 | 0 | 8.3e-18 | 4.5e-18 | 1.2e-19 | 359/359 | 1.1e-02 | 1.3e-02 | 4.4e-04 | 356/359 | 8.3e-10 |
| trajopt | b_sparse+potrf | residual | 359 | 0 | 4.1e-15 | 1.7e-15 | 2.4e-17 | 359/359 | 1.0e+00 | 4.2e-02 | 1.1e-03 | 359/359 | 3.0e-01 |
| trajopt | b_sparse+tiled_chol | random | 359 | 0 | 7.5e-18 | 4.3e-18 | 1.2e-19 | 359/359 | 1.1e-02 | 8.6e-03 | 2.8e-04 | 355/359 | 9.8e-10 |
| trajopt | b_sparse+tiled_chol | residual | 359 | 0 | 4.0e-15 | 1.6e-15 | 2.6e-17 | 359/359 | 1.0e+00 | 2.3e-02 | 1.3e-03 | 359/359 | 2.1e-01 |

### Last iteration of each solve (config b_sparse+potrf)

| label | case | k | rho_f | delta_f | rhs | potrf | res_A | res_B | res_B_ir | cw_A | cw_B | cw_B_ir | reldiff | L diag ratio^2 |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mpc | 0 | 25 | 1e-10 | 1e-10 | random | 0 | 4.8e-34 | 3.2e-34 | 3.7e-39 | 6.5e-06 | 4.0e-06 | 6.0e-11 | 2.4e-14 | 7.6e+10 |
| mpc | 0 | 25 | 1e-10 | 1e-10 | residual | 0 | 1.6e-25 | 1.4e-25 | 4.7e-30 | 7.9e-06 | 8.6e-06 | 1.4e-10 | 8.4e-06 | 7.6e+10 |
| mpc | 21 | 22 | 1e-10 | 1e-10 | random | 0 | 1.8e-33 | 7.7e-34 | 1.7e-38 | 1.1e-05 | 4.6e-06 | 8.3e-11 | 2.9e-14 | 1.4e+11 |
| mpc | 21 | 22 | 1e-10 | 1e-10 | residual | 0 | 5.6e-25 | 2.1e-25 | 4.8e-30 | 1.6e-05 | 1.6e-05 | 5.6e-10 | 1.1e-05 | 1.4e+11 |
| mpc | 42 | 26 | 1e-10 | 1e-10 | random | 0 | 7.6e-34 | 3.8e-34 | 1.3e-38 | 2.7e-05 | 1.3e-05 | 5.8e-10 | 5.0e-13 | 9.3e+10 |
| mpc | 42 | 26 | 1e-10 | 1e-10 | residual | 0 | 4.3e-28 | 4.9e-28 | 3.6e-32 | 4.4e-06 | 7.5e-06 | 4.1e-10 | 2.9e-07 | 9.3e+10 |
| mpc | 63 | 24 | 1e-10 | 1e-10 | random | 0 | 4.3e-34 | 6.9e-34 | 3.7e-38 | 7.8e-07 | 9.9e-07 | 2.2e-11 | 1.7e-13 | 6.0e+10 |
| mpc | 63 | 24 | 1e-10 | 1e-10 | residual | 0 | 3.1e-25 | 2.5e-25 | 5.6e-30 | 3.8e-06 | 3.5e-06 | 2.3e-10 | 8.6e-05 | 6.0e+10 |
| trajopt_g0 | 0 | 25 | 1e-10 | 1e-10 | random | 0 | 5.9e-33 | 6.5e-33 | 1.5e-37 | 5.6e-06 | 7.4e-06 | 6.7e-11 | 8.4e-12 | 8.3e+10 |
| trajopt_g0 | 0 | 25 | 1e-10 | 1e-10 | residual | 0 | 4.6e-26 | 1.2e-25 | 2.9e-31 | 2.8e-06 | 4.4e-06 | 3.5e-11 | 2.4e-06 | 8.3e+10 |
| trajopt_g0 | 4 | 29 | 1e-10 | 1e-10 | random | 0 | 2.2e-33 | 2.2e-33 | 3.2e-37 | 3.6e-06 | 2.7e-06 | 9.4e-11 | 2.2e-11 | 5.7e+10 |
| trajopt_g0 | 4 | 29 | 1e-10 | 1e-10 | residual | 0 | 4.3e-27 | 3.1e-27 | 1.6e-31 | 2.9e-03 | 1.6e-06 | 5.6e-11 | 2.2e-05 | 5.7e+10 |
| trajopt_g1 | 0 | 29 | 1e-11 | 5e-11 | random | 0 | 3.8e-33 | 5.1e-33 | 1.1e-36 | 1.3e-05 | 1.1e-05 | 1.4e-09 | 1.2e-11 | 4.2e+11 |
| trajopt_g1 | 0 | 29 | 1e-11 | 5e-11 | residual | 0 | 2.4e-26 | 3.6e-26 | 1.3e-29 | 5.7e-03 | 2.0e-05 | 1.9e-09 | 3.7e-05 | 4.2e+11 |
| trajopt_g1 | 4 | 30 | 1e-10 | 1e-10 | random | 0 | 5.1e-31 | 4.7e-31 | 3.5e-35 | 3.2e-06 | 1.4e-06 | 3.2e-11 | 2.1e-11 | 2.0e+11 |
| trajopt_g1 | 4 | 30 | 1e-10 | 1e-10 | residual | 0 | 6.7e-25 | 5.2e-25 | 7.9e-29 | 1.9e-03 | 1.6e-06 | 6.8e-11 | 1.0e-04 | 2.0e+11 |
| trajopt_g2 | 0 | 34 | 1e-10 | 1e-10 | random | 0 | 6.4e-33 | 5.9e-33 | 2.2e-37 | 1.4e-05 | 2.4e-05 | 1.1e-09 | 5.7e-11 | 2.1e+11 |
| trajopt_g2 | 0 | 34 | 1e-10 | 1e-10 | residual | 0 | 1.2e-27 | 9.7e-28 | 7.0e-32 | 5.8e-02 | 1.7e-05 | 1.4e-09 | 1.7e-05 | 2.1e+11 |
| trajopt_g2 | 4 | 30 | 1e-10 | 1e-10 | random | 0 | 5.6e-33 | 6.3e-33 | 1.1e-36 | 1.3e-05 | 4.4e-06 | 4.2e-10 | 3.9e-10 | 4.4e+09 |
| trajopt_g2 | 4 | 30 | 1e-10 | 1e-10 | residual | 0 | 3.4e-27 | 4.2e-27 | 5.2e-31 | 9.5e-04 | 1.5e-06 | 3.8e-10 | 1.3e-04 | 4.4e+09 |
| trajopt_g3 | 0 | 26 | 1e-13 | 1e-13 | random | 0 | 3.1e-34 | 6.0e-34 | 1.3e-39 | 6.7e-04 | 8.8e-04 | 1.3e-09 | 2.2e-15 | 7.8e+13 |
| trajopt_g3 | 0 | 26 | 1e-13 | 1e-13 | residual | 0 | 1.7e-23 | 5.0e-24 | 1.2e-27 | 1.1e-03 | 2.7e-05 | 7.6e-09 | 1.9e-04 | 7.8e+13 |
| trajopt_g3 | 4 | 29 | 1e-10 | 1e-10 | random | 0 | 1.9e-31 | 8.4e-32 | 2.0e-36 | 2.4e-06 | 1.4e-06 | 4.0e-11 | 2.3e-10 | 1.4e+10 |
| trajopt_g3 | 4 | 29 | 1e-10 | 1e-10 | residual | 0 | 1.6e-27 | 7.3e-28 | 2.0e-32 | 7.6e-02 | 9.3e-07 | 3.1e-11 | 1.7e-06 | 1.4e+10 |
| trajopt_g4 | 0 | 26 | 1e-10 | 1e-10 | random | 0 | 9.5e-34 | 2.0e-33 | 6.7e-38 | 2.7e-06 | 3.4e-06 | 1.5e-10 | 6.8e-12 | 5.6e+10 |
| trajopt_g4 | 0 | 26 | 1e-10 | 1e-10 | residual | 0 | 3.4e-28 | 4.5e-28 | 3.2e-33 | 6.3e-05 | 1.4e-05 | 1.2e-10 | 4.0e-07 | 5.6e+10 |
| trajopt_g4 | 4 | 31 | 1e-10 | 1e-10 | random | 0 | 6.5e-33 | 7.1e-33 | 8.8e-37 | 1.0e-05 | 9.5e-06 | 2.9e-10 | 4.4e-11 | 8.1e+10 |
| trajopt_g4 | 4 | 31 | 1e-10 | 1e-10 | residual | 0 | 5.2e-25 | 3.3e-25 | 2.2e-29 | 2.1e-01 | 7.6e-05 | 1.0e-08 | 2.6e-03 | 8.1e+10 |
| trajopt_g5 | 0 | 34 | 1e-13 | 1e-13 | random | 0 | 1.8e-34 | 1.5e-34 | 1.1e-35 | 1.1e-02 | 1.3e-02 | 4.4e-04 | 2.0e-13 | 1.3e+14 |
| trajopt_g5 | 0 | 34 | 1e-13 | 1e-13 | residual | 0 | 1.0e-22 | 1.7e-22 | 9.2e-24 | 2.1e-01 | 4.2e-02 | 1.1e-03 | 3.0e-01 | 1.3e+14 |
| trajopt_g5 | 4 | 36 | 1e-12 | 5e-12 | random | 0 | 8.6e-33 | 6.5e-33 | 2.2e-35 | 1.1e-04 | 1.5e-04 | 8.9e-07 | 3.2e-10 | 8.6e+11 |
| trajopt_g5 | 4 | 36 | 1e-12 | 5e-12 | residual | 0 | 4.2e-26 | 5.3e-26 | 1.4e-28 | 3.4e-03 | 4.8e-04 | 9.9e-07 | 4.2e-03 | 8.6e+11 |

### Worst over cases per iteration index (config b_sparse+potrf, both rhs)

| dataset | k | max res_A | max res_B | max cw_A | max cw_B | max cw_B_ir | max reldiff | max L ratio^2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mpc | 1 | 3.5e-16 | 3.2e-16 | 3.9e-13 | 2.3e-13 | 4.4e-16 | 1.0e-13 | 1.1e+04 |
| mpc | 2 | 8.1e-17 | 2.9e-17 | 1.5e-12 | 1.6e-12 | 5.4e-16 | 7.6e-14 | 1.5e+05 |
| mpc | 3 | 4.7e-17 | 1.8e-17 | 6.0e-12 | 3.6e-12 | 4.0e-16 | 2.9e-14 | 3.7e+05 |
| mpc | 4 | 1.5e-16 | 1.4e-16 | 1.2e-11 | 7.5e-12 | 4.0e-16 | 1.3e-13 | 2.2e+06 |
| mpc | 5 | 2.7e-16 | 3.6e-16 | 3.8e-11 | 1.7e-11 | 6.2e-16 | 3.2e-13 | 2.0e+07 |
| mpc | 6 | 8.3e-16 | 2.1e-16 | 1.3e-10 | 6.9e-11 | 4.3e-16 | 9.2e-13 | 1.7e+08 |
| mpc | 7 | 1.0e-15 | 1.9e-15 | 5.9e-10 | 2.4e-10 | 5.4e-16 | 4.0e-12 | 3.4e+09 |
| mpc | 8 | 4.2e-16 | 1.7e-16 | 1.6e-08 | 5.7e-09 | 4.4e-16 | 5.2e-12 | 1.8e+10 |
| mpc | 9 | 5.5e-17 | 4.3e-17 | 1.1e-07 | 3.9e-08 | 5.9e-16 | 1.7e-11 | 2.7e+10 |
| mpc | 10 | 2.7e-17 | 2.7e-17 | 3.9e-07 | 2.4e-07 | 4.9e-16 | 5.3e-11 | 3.0e+10 |
| mpc | 11 | 3.6e-17 | 1.4e-17 | 1.6e-06 | 8.9e-07 | 8.7e-16 | 5.0e-10 | 5.3e+10 |
| mpc | 12 | 6.9e-18 | 6.3e-18 | 2.9e-06 | 3.5e-06 | 5.7e-16 | 1.1e-09 | 8.8e+10 |
| mpc | 13 | 4.6e-18 | 2.8e-18 | 1.1e-05 | 5.9e-06 | 6.5e-16 | 4.1e-09 | 9.8e+10 |
| mpc | 14 | 3.4e-18 | 3.2e-18 | 4.3e-06 | 3.5e-06 | 1.5e-15 | 3.2e-08 | 1.1e+11 |
| mpc | 15 | 1.1e-17 | 8.2e-18 | 1.9e-05 | 6.0e-06 | 2.3e-13 | 9.8e-08 | 1.2e+11 |
| mpc | 16 | 2.0e-17 | 2.0e-17 | 6.6e-06 | 5.5e-06 | 6.9e-14 | 4.8e-07 | 1.4e+11 |
| mpc | 17 | 2.5e-18 | 1.0e-18 | 8.8e-06 | 5.8e-06 | 1.8e-13 | 3.8e-07 | 1.4e+11 |
| mpc | 18 | 1.7e-19 | 9.6e-20 | 6.6e-06 | 8.7e-06 | 1.1e-13 | 1.7e-07 | 1.4e+11 |
| mpc | 19 | 9.3e-20 | 9.3e-20 | 5.8e-06 | 4.8e-06 | 4.3e-14 | 2.4e-07 | 1.4e+11 |
| mpc | 20 | 3.6e-21 | 8.7e-21 | 1.0e-05 | 4.6e-06 | 4.2e-13 | 8.9e-08 | 1.4e+11 |
| mpc | 21 | 2.8e-21 | 6.4e-22 | 1.3e-05 | 6.7e-06 | 1.1e-12 | 2.6e-07 | 1.4e+11 |
| mpc | 22 | 1.1e-22 | 8.9e-23 | 1.6e-05 | 1.6e-05 | 5.6e-10 | 1.1e-05 | 1.4e+11 |
| mpc | 23 | 4.0e-25 | 3.0e-25 | 4.3e-06 | 5.0e-06 | 5.6e-11 | 4.7e-05 | 7.0e+10 |
| mpc | 24 | 3.1e-25 | 2.5e-25 | 3.8e-06 | 4.2e-06 | 2.3e-10 | 8.6e-05 | 7.0e+10 |
| mpc | 25 | 1.6e-25 | 1.4e-25 | 7.9e-06 | 8.6e-06 | 1.4e-10 | 8.4e-06 | 7.6e+10 |
| mpc | 26 | 4.3e-28 | 4.9e-28 | 2.7e-05 | 1.3e-05 | 5.8e-10 | 2.9e-07 | 9.3e+10 |
| trajopt | 1 | 4.7e-16 | 2.3e-16 | 2.5e-12 | 8.0e-13 | 1.5e-15 | 2.6e-13 | 1.5e+03 |
| trajopt | 2 | 1.1e-15 | 2.7e-16 | 1.7e-11 | 7.9e-13 | 1.4e-15 | 1.9e-13 | 1.3e+04 |
| trajopt | 3 | 2.1e-16 | 5.4e-16 | 1.9e-11 | 2.3e-12 | 2.3e-15 | 4.1e-13 | 4.7e+04 |
| trajopt | 4 | 2.8e-16 | 2.2e-16 | 8.4e-11 | 5.3e-12 | 8.2e-16 | 2.9e-13 | 6.2e+04 |
| trajopt | 5 | 6.7e-16 | 1.6e-15 | 3.3e-10 | 1.9e-11 | 9.5e-16 | 1.7e-12 | 2.3e+05 |
| trajopt | 6 | 4.1e-15 | 1.7e-15 | 7.2e-10 | 1.0e-10 | 7.1e-16 | 2.2e-11 | 1.1e+06 |
| trajopt | 7 | 1.6e-15 | 1.5e-15 | 1.9e-09 | 1.6e-10 | 1.3e-15 | 3.1e-12 | 1.0e+07 |
| trajopt | 8 | 2.4e-16 | 2.2e-16 | 1.7e-06 | 9.6e-10 | 6.1e-16 | 2.9e-11 | 3.2e+08 |
| trajopt | 9 | 5.5e-17 | 1.7e-16 | 7.3e-05 | 2.4e-09 | 1.3e-15 | 3.5e-11 | 1.1e+09 |
| trajopt | 10 | 2.3e-17 | 8.9e-17 | 2.1e-04 | 6.5e-09 | 1.5e-15 | 2.1e-10 | 4.4e+09 |
| trajopt | 11 | 1.5e-17 | 1.1e-16 | 1.5e-03 | 5.4e-08 | 1.5e-15 | 2.1e-09 | 1.1e+10 |
| trajopt | 12 | 4.2e-16 | 4.8e-16 | 6.2e-03 | 2.0e-07 | 9.7e-15 | 4.9e-07 | 1.8e+10 |
| trajopt | 13 | 1.6e-17 | 2.4e-17 | 1.2e-02 | 1.1e-07 | 2.1e-15 | 1.6e-08 | 3.0e+10 |
| trajopt | 14 | 2.1e-15 | 1.5e-15 | 2.3e-02 | 1.3e-06 | 4.7e-13 | 1.5e-06 | 3.1e+10 |
| trajopt | 15 | 4.5e-16 | 2.8e-16 | 1.8e-03 | 6.6e-07 | 2.0e-13 | 5.6e-07 | 4.0e+10 |
| trajopt | 16 | 2.5e-16 | 3.7e-16 | 4.0e-03 | 1.2e-06 | 8.1e-13 | 3.9e-06 | 5.1e+10 |
| trajopt | 17 | 6.8e-17 | 8.7e-17 | 2.6e-03 | 3.9e-06 | 3.4e-13 | 1.7e-06 | 6.8e+10 |
| trajopt | 18 | 2.3e-16 | 3.0e-17 | 1.8e-02 | 1.9e-06 | 5.0e-13 | 3.2e-05 | 9.1e+10 |
| trajopt | 19 | 4.6e-18 | 2.3e-18 | 4.2e-03 | 3.2e-06 | 3.0e-14 | 7.9e-07 | 1.1e+11 |
| trajopt | 20 | 3.0e-17 | 4.3e-18 | 1.6e-02 | 2.5e-06 | 1.0e-12 | 9.9e-06 | 1.5e+11 |
| trajopt | 21 | 2.4e-18 | 2.5e-19 | 3.6e-02 | 4.2e-05 | 8.0e-14 | 6.0e-07 | 1.5e+12 |
| trajopt | 22 | 4.6e-19 | 6.9e-20 | 1.6e-01 | 2.0e-04 | 4.2e-13 | 3.9e-07 | 1.6e+13 |
| trajopt | 23 | 7.5e-20 | 2.9e-20 | 1.0e-01 | 5.2e-04 | 8.3e-13 | 1.6e-07 | 7.8e+13 |
| trajopt | 24 | 2.0e-21 | 3.4e-21 | 3.2e-01 | 1.8e-03 | 1.4e-11 | 6.7e-07 | 1.0e+14 |
| trajopt | 25 | 1.9e-22 | 1.2e-22 | 8.8e-01 | 1.5e-03 | 6.7e-11 | 3.9e-06 | 1.1e+14 |
| trajopt | 26 | 3.0e-23 | 3.3e-23 | 1.0e+00 | 1.3e-03 | 7.6e-09 | 1.9e-04 | 1.1e+14 |
| trajopt | 27 | 7.4e-23 | 2.3e-23 | 1.0e+00 | 2.0e-03 | 9.8e-10 | 2.0e-05 | 1.1e+14 |
| trajopt | 28 | 3.8e-23 | 1.6e-23 | 4.0e-01 | 1.2e-03 | 9.0e-09 | 6.8e-05 | 1.1e+14 |
| trajopt | 29 | 5.6e-23 | 2.3e-23 | 1.7e-01 | 2.0e-03 | 1.4e-08 | 1.7e-03 | 1.1e+14 |
| trajopt | 30 | 1.1e-24 | 1.6e-24 | 1.7e-01 | 1.4e-03 | 1.8e-09 | 1.3e-04 | 1.1e+14 |
| trajopt | 31 | 5.2e-25 | 1.4e-24 | 2.1e-01 | 1.3e-03 | 1.0e-07 | 2.6e-03 | 1.1e+14 |
| trajopt | 32 | 1.5e-23 | 3.5e-23 | 4.4e-02 | 1.1e-03 | 9.6e-06 | 3.3e-02 | 1.2e+14 |
| trajopt | 33 | 1.9e-23 | 2.2e-23 | 3.9e-02 | 3.3e-03 | 1.6e-04 | 9.6e-03 | 1.3e+14 |
| trajopt | 34 | 1.0e-22 | 1.7e-22 | 2.1e-01 | 4.2e-02 | 1.1e-03 | 3.0e-01 | 1.3e+14 |
| trajopt | 35 | 2.0e-27 | 2.5e-27 | 7.6e-03 | 1.8e-05 | 5.9e-09 | 1.6e-05 | 8.6e+10 |
| trajopt | 36 | 4.2e-26 | 5.3e-26 | 3.4e-03 | 4.8e-04 | 9.9e-07 | 4.2e-03 | 8.6e+11 |

mpc: numerically nonzero stored [A; G] entries per state: min 41485, max 44120

trajopt: numerically nonzero stored [A; G] entries per state: min 146007, max 149062
