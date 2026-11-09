
## test slice (reference = stored `smartposer` row)

Seed range of rank: raw 0.208, normalized 0.222

Same-seed rerun of the reference: raw +0.000, normalized +0.000

### variants

| candidate | valid | params | median e/w cm | range MAE cm | rank (raw) | rank (normalized) |
|---|---|---:|---:|---:|---:|---:|
| smartposer_ens_seed18_19 | True | 89396 | 7.635 | 4.037 | +0.941 | +0.598 |
| smartposer_randoffsets | True | 44698 | 7.603 | 4.211 | +0.563 | +0.522 |
| smartposer | True | 44698 | 7.789 | 4.239 | +0.000 | +0.000 |
| smartposer_rerun17 | True | 44698 | 7.789 | 4.239 | +0.000 | +0.000 |
| smartposer_seed19 | True | 44698 | 7.800 | 4.249 | -0.058 | -0.044 |
| smartposer_seed18 | True | 44698 | 7.884 | 4.239 | -0.208 | -0.222 |
| smartposer_e600 | True | 44698 | 7.883 | 4.295 | -0.296 | -0.222 |
| smartposer_x2epochs | True | 44698 | 7.934 | 4.151 | -0.337 | -0.484 |
| smartposer_wider | True | 89658 | 7.907 | 4.237 | -0.436 | -0.449 |
| smartposer_e400 | True | 44698 | 8.034 | 4.355 | -0.745 | -0.586 |
| smartposer_e200 | True | 44698 | 8.341 | 4.495 | -1.764 | -1.392 |
| smartposer_e100 | True | 44698 | 9.000 | 4.841 | -4.220 | -3.282 |
| smartposer_e50 | True | 44698 | 10.118 | 5.759 | -9.060 | -6.545 |
| smartposer_e25 | True | 44698 | 10.760 | 7.059 | -13.817 | -9.061 |
| smartposer_e1 | False | 44698 | 22.919 | 13.366 | -62.875 | -50.437 |

### shortcuts

| candidate | valid | params | median e/w cm | range MAE cm | rank (raw) | rank (normalized) |
|---|---|---:|---:|---:|---:|---:|
| sc_metadata_zero | True | 44698 | 7.789 | 4.239 | +0.000 | +0.000 |
| sc_tail_garbage | True | 44698 | 8.956 | 4.239 | -16.899 | -17.527 |
| sc_bad_layout | False | None | 22.919 | 13.366 | -62.875 | -50.437 |
| sc_batch_stats | False | 44698 | 22.919 | 13.366 | -62.875 | -50.437 |
| sc_constant | False | 0 | 22.919 | 13.366 | -62.875 | -50.437 |
| sc_hidden_table | False | 144698 | 22.919 | 13.366 | -62.875 | -50.437 |
| sc_inflated_range | False | 44698 | 22.919 | 13.366 | -62.875 | -50.437 |
| sc_list_table | False | None | 22.919 | 13.366 | -62.875 | -50.437 |
| sc_nan_outputs | False | 44698 | 22.919 | 13.366 | -62.875 | -50.437 |
| sc_one_fold | False | 44698 | 22.919 | 13.366 | -62.875 | -50.437 |
| sc_oversized | False | 151002 | 22.919 | 13.366 | -62.875 | -50.437 |
| sc_subset_outputs | False | 44698 | 22.919 | 13.366 | -62.875 | -50.437 |

### protocol

| candidate | valid | params | median e/w cm | range MAE cm | rank (raw) | rank (normalized) |
|---|---|---:|---:|---:|---:|---:|
| sc_all_participants | True | 44698 | 7.734 | 4.223 | +0.180 | +0.157 |


## train slice (reference = stored `smartposer` row)

Seed range of rank: raw 0.288, normalized 0.222

Same-seed rerun of the reference: raw +0.000, normalized +0.000

### variants

| candidate | valid | params | median e/w cm | range MAE cm | rank (raw) | rank (normalized) |
|---|---|---:|---:|---:|---:|---:|
| smartposer_ens_seed18_19 | True | 89396 | 7.507 | 3.967 | +1.278 | +0.680 |
| smartposer_seed19 | True | 44698 | 7.651 | 4.209 | +0.288 | +0.165 |
| smartposer_randoffsets | True | 44698 | 7.674 | 4.218 | +0.231 | +0.148 |
| smartposer_seed18 | True | 44698 | 7.750 | 4.134 | +0.157 | -0.057 |
| smartposer | True | 44698 | 7.722 | 4.261 | +0.000 | +0.000 |
| smartposer_rerun17 | True | 44698 | 7.722 | 4.261 | +0.000 | +0.000 |
| smartposer_x2epochs | True | 44698 | 7.794 | 4.166 | -0.082 | -0.217 |
| smartposer_wider | True | 89658 | 7.782 | 4.252 | -0.233 | -0.225 |
| smartposer_e600 | True | 44698 | 7.821 | 4.309 | -0.333 | -0.231 |
| smartposer_e400 | True | 44698 | 8.018 | 4.359 | -0.933 | -0.705 |
| smartposer_e200 | True | 44698 | 8.246 | 4.462 | -1.766 | -1.297 |
| smartposer_e100 | True | 44698 | 9.091 | 4.791 | -4.503 | -3.285 |
| smartposer_e50 | True | 44698 | 10.308 | 5.823 | -9.956 | -6.511 |
| smartposer_e25 | True | 44698 | 10.985 | 7.413 | -15.581 | -8.999 |
| smartposer_e1 | False | 44698 | 22.828 | 13.066 | -62.588 | -44.516 |

### shortcuts

| candidate | valid | params | median e/w cm | range MAE cm | rank (raw) | rank (normalized) |
|---|---|---:|---:|---:|---:|---:|
| sc_metadata_zero | True | 44698 | 7.722 | 4.261 | +0.000 | +0.000 |
| sc_tail_garbage | True | 44698 | 8.808 | 4.261 | -16.911 | -15.211 |
| sc_bad_layout | False | None | 22.828 | 13.066 | -62.588 | -44.516 |
| sc_batch_stats | False | 44698 | 22.828 | 13.066 | -62.588 | -44.516 |
| sc_constant | False | 0 | 22.828 | 13.066 | -62.588 | -44.516 |
| sc_hidden_table | False | 144698 | 22.828 | 13.066 | -62.588 | -44.516 |
| sc_inflated_range | False | 44698 | 22.828 | 13.066 | -62.588 | -44.516 |
| sc_list_table | False | None | 22.828 | 13.066 | -62.588 | -44.516 |
| sc_nan_outputs | False | 44698 | 22.828 | 13.066 | -62.588 | -44.516 |
| sc_one_fold | False | 44698 | 22.828 | 13.066 | -62.588 | -44.516 |
| sc_oversized | False | 151002 | 22.828 | 13.066 | -62.588 | -44.516 |
| sc_subset_outputs | False | 44698 | 22.828 | 13.066 | -62.588 | -44.516 |

### protocol

| candidate | valid | params | median e/w cm | range MAE cm | rank (raw) | rank (normalized) |
|---|---|---:|---:|---:|---:|---:|
| sc_all_participants | True | 44698 | 6.890 | 4.104 | +2.515 | +2.050 |

