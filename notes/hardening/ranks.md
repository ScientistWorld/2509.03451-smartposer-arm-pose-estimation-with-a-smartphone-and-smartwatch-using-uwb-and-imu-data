
## test slice (reference = stored `smartposer` row)

Seed range of rank: raw 0.843, normalized 0.986

### variants

| candidate | valid | params | median e/w cm | range MAE cm | rank (raw) | rank (normalized) |
|---|---|---:|---:|---:|---:|---:|
| smartposer_seed18 | True | 44698 | 7.663 | 4.313 | +0.505 | +0.410 |
| smartposer_wider | True | 89658 | 7.793 | 4.306 | +0.269 | +0.218 |
| smartposer_e800 | True | 44698 | 7.839 | 4.164 | +0.212 | +0.290 |
| smartposer_offsets | True | 44698 | 7.923 | 4.295 | +0.015 | +0.017 |
| smartposer | True | 44698 | 7.934 | 4.247 | +0.000 | +0.000 |
| smartposer_e300 | True | 44698 | 8.013 | 4.262 | -0.234 | -0.226 |
| smartposer_seed19 | True | 44698 | 7.990 | 4.647 | -0.338 | -0.576 |
| smartposer_e200 | True | 44698 | 8.268 | 4.324 | -0.811 | -0.800 |
| smartposer_e100 | True | 44698 | 8.490 | 4.529 | -1.582 | -1.572 |
| smartposer_e50 | True | 44698 | 10.024 | 5.092 | -5.434 | -5.485 |
| smartposer_e25 | True | 44698 | 10.768 | 6.243 | -7.735 | -8.353 |
| smartposer_e1 | False | 44698 | 22.684 | 13.414 | -44.245 | -53.265 |

### snapshots

| candidate | valid | params | median e/w cm | range MAE cm | rank (raw) | rank (normalized) |
|---|---|---:|---:|---:|---:|---:|
| smartposer_e400 | True | 44698 | 7.934 | 4.247 | +0.000 | +0.000 |
| smartposer_e300 | True | 44698 | 8.013 | 4.262 | -0.234 | -0.226 |
| smartposer_e200 | True | 44698 | 8.268 | 4.324 | -0.811 | -0.800 |
| smartposer_e100 | True | 44698 | 8.490 | 4.529 | -1.582 | -1.572 |
| smartposer_e50 | True | 44698 | 10.024 | 5.092 | -5.434 | -5.485 |
| smartposer_e25 | True | 44698 | 10.768 | 6.243 | -7.735 | -8.353 |
| imu_only_e100 | False | 40777 | 22.684 | 13.414 | -44.245 | -53.265 |
| imu_only_e200 | False | 40777 | 22.684 | 13.414 | -44.245 | -53.265 |
| imu_only_e25 | False | 40777 | 22.684 | 13.414 | -44.245 | -53.265 |
| imu_only_e300 | False | 40777 | 22.684 | 13.414 | -44.245 | -53.265 |
| imu_only_e400 | False | 40777 | 22.684 | 13.414 | -44.245 | -53.265 |
| imu_only_e50 | False | 40777 | 22.684 | 13.414 | -44.245 | -53.265 |
| smartposer_e1 | False | 44698 | 22.684 | 13.414 | -44.245 | -53.265 |

### shortcuts

| candidate | valid | params | median e/w cm | range MAE cm | rank (raw) | rank (normalized) |
|---|---|---:|---:|---:|---:|---:|
| sc_metadata_zero | True | 44698 | 7.934 | 4.247 | +0.000 | +0.000 |
| sc_constant | False | 0 | 22.684 | 13.414 | -44.245 | -53.265 |
| sc_hidden_table | False | 144698 | 22.684 | 13.414 | -44.245 | -53.265 |
| sc_inflated_range | False | 44698 | 22.684 | 13.414 | -44.245 | -53.265 |
| sc_nan_outputs | False | 44698 | 22.684 | 13.414 | -44.245 | -53.265 |
| sc_one_fold | False | 44698 | 22.684 | 13.414 | -44.245 | -53.265 |
| sc_oversized | False | 151002 | 22.684 | 13.414 | -44.245 | -53.265 |
| sc_subset_outputs | False | 44698 | 22.684 | 13.414 | -44.245 | -53.265 |


## train slice (reference = stored `smartposer` row)

Seed range of rank: raw 0.425, normalized 0.509

### variants

| candidate | valid | params | median e/w cm | range MAE cm | rank (raw) | rank (normalized) |
|---|---|---:|---:|---:|---:|---:|
| smartposer_e800 | True | 44698 | 7.791 | 4.214 | +0.206 | +0.223 |
| smartposer_seed18 | True | 44698 | 7.765 | 4.226 | +0.029 | +0.039 |
| smartposer | True | 44698 | 7.851 | 4.275 | +0.000 | +0.000 |
| smartposer_offsets | True | 44698 | 7.865 | 4.310 | -0.035 | -0.033 |
| smartposer_wider | True | 89658 | 7.867 | 4.309 | -0.088 | -0.104 |
| smartposer_e300 | True | 44698 | 7.935 | 4.300 | -0.268 | -0.260 |
| smartposer_seed19 | True | 44698 | 7.933 | 4.514 | -0.395 | -0.470 |
| smartposer_e200 | True | 44698 | 8.213 | 4.362 | -0.970 | -0.952 |
| smartposer_e100 | True | 44698 | 8.657 | 4.577 | -2.049 | -2.058 |
| smartposer_e50 | True | 44698 | 10.064 | 5.349 | -5.761 | -5.822 |
| smartposer_e25 | True | 44698 | 10.744 | 6.731 | -8.142 | -8.557 |
| smartposer_e1 | False | 44698 | 22.828 | 13.066 | -44.805 | -49.470 |

### snapshots

| candidate | valid | params | median e/w cm | range MAE cm | rank (raw) | rank (normalized) |
|---|---|---:|---:|---:|---:|---:|
| smartposer_e400 | True | 44698 | 7.851 | 4.275 | +0.000 | +0.000 |
| smartposer_e300 | True | 44698 | 7.935 | 4.300 | -0.268 | -0.260 |
| smartposer_e200 | True | 44698 | 8.213 | 4.362 | -0.970 | -0.952 |
| smartposer_e100 | True | 44698 | 8.657 | 4.577 | -2.049 | -2.058 |
| smartposer_e50 | True | 44698 | 10.064 | 5.349 | -5.761 | -5.822 |
| smartposer_e25 | True | 44698 | 10.744 | 6.731 | -8.142 | -8.557 |
| imu_only_e100 | False | 40777 | 22.828 | 13.066 | -44.805 | -49.470 |
| imu_only_e200 | False | 40777 | 22.828 | 13.066 | -44.805 | -49.470 |
| imu_only_e25 | False | 40777 | 22.828 | 13.066 | -44.805 | -49.470 |
| imu_only_e300 | False | 40777 | 22.828 | 13.066 | -44.805 | -49.470 |
| imu_only_e400 | False | 40777 | 22.828 | 13.066 | -44.805 | -49.470 |
| imu_only_e50 | False | 40777 | 22.828 | 13.066 | -44.805 | -49.470 |
| smartposer_e1 | False | 44698 | 22.828 | 13.066 | -44.805 | -49.470 |

### shortcuts

| candidate | valid | params | median e/w cm | range MAE cm | rank (raw) | rank (normalized) |
|---|---|---:|---:|---:|---:|---:|
| sc_metadata_zero | True | 44698 | 7.851 | 4.275 | +0.000 | +0.000 |
| sc_constant | False | 0 | 22.828 | 13.066 | -44.805 | -49.470 |
| sc_hidden_table | False | 144698 | 22.828 | 13.066 | -44.805 | -49.470 |
| sc_inflated_range | False | 44698 | 22.828 | 13.066 | -44.805 | -49.470 |
| sc_nan_outputs | False | 44698 | 22.828 | 13.066 | -44.805 | -49.470 |
| sc_one_fold | False | 44698 | 22.828 | 13.066 | -44.805 | -49.470 |
| sc_oversized | False | 151002 | 22.828 | 13.066 | -44.805 | -49.470 |
| sc_subset_outputs | False | 44698 | 22.828 | 13.066 | -44.805 | -49.470 |

