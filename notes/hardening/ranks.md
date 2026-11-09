
## test slice (reference = stored `smartposer` row)

Seed range of rank: raw 0.208, normalized 0.215

### variants

| candidate | valid | params | median e/w cm | range MAE cm | rank (raw) | rank (normalized) |
|---|---|---:|---:|---:|---:|---:|
| smartposer_ens_seed18_19 | True | 89396 | 7.635 | 4.037 | +0.940 | +0.573 |
| smartposer | True | 44698 | 7.789 | 4.239 | +0.000 | +0.000 |
| smartposer_seed19 | True | 44698 | 7.800 | 4.249 | -0.059 | -0.042 |
| smartposer_seed18 | True | 44698 | 7.884 | 4.239 | -0.208 | -0.215 |
| smartposer_e600 | True | 44698 | 7.882 | 4.295 | -0.296 | -0.212 |
| smartposer_x2epochs | True | 44698 | 7.934 | 4.151 | -0.338 | -0.471 |
| smartposer_wider | True | 89658 | 7.907 | 4.237 | -0.435 | -0.433 |
| smartposer_e400 | True | 44698 | 8.034 | 4.355 | -0.745 | -0.563 |
| smartposer_e200 | True | 44698 | 8.341 | 4.495 | -1.764 | -1.338 |
| smartposer_e100 | True | 44698 | 8.999 | 4.841 | -4.220 | -3.156 |
| smartposer_e50 | True | 44698 | 10.118 | 5.759 | -9.060 | -6.292 |
| smartposer_e25 | True | 44698 | 10.760 | 7.059 | -13.817 | -8.698 |
| smartposer_e1 | False | 44698 | 22.919 | 13.366 | -62.876 | -48.429 |

### snapshots

| candidate | valid | params | median e/w cm | range MAE cm | rank (raw) | rank (normalized) |
|---|---|---:|---:|---:|---:|---:|
| smartposer_e800 | True | 44698 | 7.789 | 4.239 | +0.000 | +0.000 |
| smartposer_e600 | True | 44698 | 7.882 | 4.295 | -0.296 | -0.212 |
| smartposer_e400 | True | 44698 | 8.034 | 4.355 | -0.745 | -0.563 |
| smartposer_e200 | True | 44698 | 8.341 | 4.495 | -1.764 | -1.338 |
| smartposer_e100 | True | 44698 | 8.999 | 4.841 | -4.220 | -3.156 |
| smartposer_e50 | True | 44698 | 10.118 | 5.759 | -9.060 | -6.292 |
| smartposer_e25 | True | 44698 | 10.760 | 7.059 | -13.817 | -8.698 |
| imu_only_e100 | False | 40777 | 22.919 | 13.366 | -62.876 | -48.429 |
| imu_only_e200 | False | 40777 | 22.919 | 13.366 | -62.876 | -48.429 |
| imu_only_e25 | False | 40777 | 22.919 | 13.366 | -62.876 | -48.429 |
| imu_only_e400 | False | 40777 | 22.919 | 13.366 | -62.876 | -48.429 |
| imu_only_e50 | False | 40777 | 22.919 | 13.366 | -62.876 | -48.429 |
| imu_only_e600 | False | 40777 | 22.919 | 13.366 | -62.876 | -48.429 |
| imu_only_e800 | False | 40777 | 22.919 | 13.366 | -62.876 | -48.429 |
| smartposer_e1 | False | 44698 | 22.919 | 13.366 | -62.876 | -48.429 |

### shortcuts

| candidate | valid | params | median e/w cm | range MAE cm | rank (raw) | rank (normalized) |
|---|---|---:|---:|---:|---:|---:|
| sc_metadata_zero | True | 44698 | 7.789 | 4.239 | +0.000 | +0.000 |
| sc_constant | False | 0 | 22.919 | 13.366 | -62.876 | -48.429 |
| sc_hidden_table | False | 144698 | 22.919 | 13.366 | -62.876 | -48.429 |
| sc_inflated_range | False | 44698 | 22.919 | 13.366 | -62.876 | -48.429 |
| sc_list_table | False | None | 22.919 | 13.366 | -62.876 | -48.429 |
| sc_nan_outputs | False | 44698 | 22.919 | 13.366 | -62.876 | -48.429 |
| sc_one_fold | False | 44698 | 22.919 | 13.366 | -62.876 | -48.429 |
| sc_oversized | False | 151002 | 22.919 | 13.366 | -62.876 | -48.429 |
| sc_subset_outputs | False | 44698 | 22.919 | 13.366 | -62.876 | -48.429 |


## train slice (reference = stored `smartposer` row)

Seed range of rank: raw 0.287, normalized 0.211

### variants

| candidate | valid | params | median e/w cm | range MAE cm | rank (raw) | rank (normalized) |
|---|---|---:|---:|---:|---:|---:|
| smartposer_ens_seed18_19 | True | 89396 | 7.506 | 3.967 | +1.279 | +0.647 |
| smartposer_seed19 | True | 44698 | 7.653 | 4.209 | +0.287 | +0.156 |
| smartposer_seed18 | True | 44698 | 7.750 | 4.134 | +0.159 | -0.054 |
| smartposer | True | 44698 | 7.721 | 4.261 | +0.000 | +0.000 |
| smartposer_x2epochs | True | 44698 | 7.793 | 4.166 | -0.082 | -0.208 |
| smartposer_wider | True | 89658 | 7.781 | 4.252 | -0.234 | -0.215 |
| smartposer_e600 | True | 44698 | 7.822 | 4.309 | -0.334 | -0.220 |
| smartposer_e400 | True | 44698 | 8.018 | 4.359 | -0.934 | -0.673 |
| smartposer_e200 | True | 44698 | 8.246 | 4.462 | -1.765 | -1.234 |
| smartposer_e100 | True | 44698 | 9.092 | 4.791 | -4.504 | -3.128 |
| smartposer_e50 | True | 44698 | 10.307 | 5.823 | -9.955 | -6.195 |
| smartposer_e25 | True | 44698 | 10.984 | 7.413 | -15.582 | -8.555 |
| smartposer_e1 | False | 44698 | 22.828 | 13.066 | -62.589 | -42.339 |

### snapshots

| candidate | valid | params | median e/w cm | range MAE cm | rank (raw) | rank (normalized) |
|---|---|---:|---:|---:|---:|---:|
| smartposer_e800 | True | 44698 | 7.721 | 4.261 | +0.000 | +0.000 |
| smartposer_e600 | True | 44698 | 7.822 | 4.309 | -0.334 | -0.220 |
| smartposer_e400 | True | 44698 | 8.018 | 4.359 | -0.934 | -0.673 |
| smartposer_e200 | True | 44698 | 8.246 | 4.462 | -1.765 | -1.234 |
| smartposer_e100 | True | 44698 | 9.092 | 4.791 | -4.504 | -3.128 |
| smartposer_e50 | True | 44698 | 10.307 | 5.823 | -9.955 | -6.195 |
| smartposer_e25 | True | 44698 | 10.984 | 7.413 | -15.582 | -8.555 |
| imu_only_e100 | False | 40777 | 22.828 | 13.066 | -62.589 | -42.339 |
| imu_only_e200 | False | 40777 | 22.828 | 13.066 | -62.589 | -42.339 |
| imu_only_e25 | False | 40777 | 22.828 | 13.066 | -62.589 | -42.339 |
| imu_only_e400 | False | 40777 | 22.828 | 13.066 | -62.589 | -42.339 |
| imu_only_e50 | False | 40777 | 22.828 | 13.066 | -62.589 | -42.339 |
| imu_only_e600 | False | 40777 | 22.828 | 13.066 | -62.589 | -42.339 |
| imu_only_e800 | False | 40777 | 22.828 | 13.066 | -62.589 | -42.339 |
| smartposer_e1 | False | 44698 | 22.828 | 13.066 | -62.589 | -42.339 |

### shortcuts

| candidate | valid | params | median e/w cm | range MAE cm | rank (raw) | rank (normalized) |
|---|---|---:|---:|---:|---:|---:|
| sc_metadata_zero | True | 44698 | 7.721 | 4.261 | +0.000 | +0.000 |
| sc_constant | False | 0 | 22.828 | 13.066 | -62.589 | -42.339 |
| sc_hidden_table | False | 144698 | 22.828 | 13.066 | -62.589 | -42.339 |
| sc_inflated_range | False | 44698 | 22.828 | 13.066 | -62.589 | -42.339 |
| sc_list_table | False | None | 22.828 | 13.066 | -62.589 | -42.339 |
| sc_nan_outputs | False | 44698 | 22.828 | 13.066 | -62.589 | -42.339 |
| sc_one_fold | False | 44698 | 22.828 | 13.066 | -62.589 | -42.339 |
| sc_oversized | False | 151002 | 22.828 | 13.066 | -62.589 | -42.339 |
| sc_subset_outputs | False | 44698 | 22.828 | 13.066 | -62.589 | -42.339 |

