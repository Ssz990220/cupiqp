
### 29-79b3-place-6f87a360 (36 transitions)

| row group / side | rows (median) | alpha med | alpha min..max | err(alpha=1) med | err(alpha=1) max | shift C x_k max |
|---|---:|---:|---:|---:|---:|---:|
| Q(position)/l | 240 | 1.0000 | 1.000..1.000 | 8.8e-17 | 2.0e-15 | 1.2e-01 |
| Q(position)/u | 240 | 1.0000 | 1.000..1.000 | 9.9e-17 | 1.1e-15 | 1.2e-01 |
| collision/l | 751 | 0.9947 | 0.937..1.002 | 7.7e-08 | 9.3e-02 | 1.7e-01 |
| dQ(velocity)/u | 448 | 0.9996 | 0.414..1.000 | 7.6e-06 | 4.4e-01 | 8.9e-01 |
| ddQ(acceleration)/u | 448 | 1.0000 | 0.085..1.000 | 2.5e-04 | 9.9e+00 | 1.3e+01 |
| delta_t/l | 1 | 0.0000 | 0.000..0.000 | 5.0e-06 | 1.2e-01 | 1.2e-01 |
| delta_t/u | 1 | 0.0000 | 0.000..0.000 | 5.0e-06 | 1.2e-01 | 1.2e-01 |
| eef/l | 4064 | 0.9999 | 0.996..1.162 | 9.5e-06 | 1.1e+00 | 4.8e+00 |
| eef/u | 4064 | 0.9999 | 0.996..1.162 | 9.5e-06 | 1.1e+00 | 4.8e+00 |

| row group | active in k+1 (mean) | of those active in k (mean %) | min % |
|---|---:|---:|---:|
| Q(position) | 0.0 | nan | nan |
| collision | 0.9 | 97.1 | 0.0 |
| dQ(velocity) | 2.7 | 99.4 | 80.0 |
| ddQ(acceleration) | 0.1 | 50.0 | 0.0 |
| dddQ(jerk) | 0.0 | nan | nan |
| delta_t | 0.0 | nan | nan |
| eef | 102.5 | 93.5 | 52.6 |
| all | 106.2 | 93.7 | 53.8 |

collision rows active in k and k+1, matched by (sample, pair): 33, by position: 33
collision rows per QP 752: raw label at the same position: mean 679; raw label found anywhere: mean 679; (sample, pair) found: mean 750
all rows active in k and k+1: (sample, pair) map 3580, raw label map 3580, of 3823 active in k+1
relative dual change with raw-label map: median 0.054, max 0.958
relative dual change |z_k+1 - z_k|/|z_k+1| ((sample, pair) map): median 0.054, max 0.958; by position: median 0.054; y: median 0.003, max 0.919

### 55-5d59-pick-d59c9e60 (63 transitions)

| row group / side | rows (median) | alpha med | alpha min..max | err(alpha=1) med | err(alpha=1) max | shift C x_k max |
|---|---:|---:|---:|---:|---:|---:|
| Q(position)/l | 240 | 1.0000 | 1.000..1.000 | 9.2e-17 | 2.0e-15 | 1.3e-01 |
| Q(position)/u | 240 | 1.0000 | 1.000..1.000 | 9.5e-17 | 1.1e-15 | 1.3e-01 |
| collision/l | 752 | 1.0006 | 0.847..1.028 | 1.8e-07 | 1.0e-01 | 1.3e-01 |
| dQ(velocity)/u | 448 | 0.9918 | 0.196..1.000 | 8.8e-04 | 4.6e-01 | 7.8e-01 |
| ddQ(acceleration)/u | 448 | 0.9243 | 0.020..1.000 | 2.8e-02 | 1.1e+01 | 1.5e+01 |
| delta_t/l | 1 | 0.0000 | 0.000..0.000 | 6.3e-04 | 2.0e-01 | 2.0e-01 |
| delta_t/u | 1 | 0.0000 | 0.000..0.000 | 6.3e-04 | 2.0e-01 | 2.0e-01 |
| eef/l | 4064 | 1.0006 | 0.999..1.171 | 1.3e-05 | 1.3e+00 | 4.8e+00 |
| eef/u | 4064 | 1.0006 | 0.999..1.171 | 1.3e-05 | 1.3e+00 | 4.8e+00 |

| row group | active in k+1 (mean) | of those active in k (mean %) | min % |
|---|---:|---:|---:|
| Q(position) | 0.0 | nan | nan |
| collision | 2.8 | 94.6 | 0.0 |
| dQ(velocity) | 3.3 | 95.3 | 0.0 |
| ddQ(acceleration) | 1.8 | 91.9 | 0.0 |
| dddQ(jerk) | 0.0 | nan | nan |
| delta_t | 0.0 | 100.0 | 100.0 |
| eef | 146.7 | 90.3 | 25.0 |
| all | 154.6 | 90.4 | 25.7 |

collision rows active in k and k+1, matched by (sample, pair): 164, by position: 161
collision rows per QP 752: raw label at the same position: mean 447; raw label found anywhere: mean 447; (sample, pair) found: mean 750
all rows active in k and k+1: (sample, pair) map 8881, raw label map 8829, of 9739 active in k+1
relative dual change with raw-label map: median 0.586, max 1.161
relative dual change |z_k+1 - z_k|/|z_k+1| ((sample, pair) map): median 0.030, max 1.161; by position: median 0.030; y: median 0.008, max 0.552
