Coverage: 10 versions on 5 lists (1070 series), 85 runs; every run has every series; series with an error: 0; fitting and scoring took 25.9 hours in all; largest GPU memory of a run 0.8 GB.

#### M-Eva (180 series): mean VUS-PR, trained / untrained, for the three score types. The first three rows are from Sections 7.9 and 7.10; `subjepa`, `straight` and `motionjepa` share the untrained control of `lewm`. Minutes: fitting and scoring of the trained run.

| version | `pred` | `embed` | `combined` | minutes |
|---|---|---|---|---|
| lejepa (default) | 0.351 / 0.347 | 0.353 / 0.352 | 0.352 / 0.351 | 46 |
| vicreg | 0.360 / 0.347 | 0.359 / 0.352 | 0.360 / 0.351 | 47 |
| lewm | 0.386 / 0.369 | 0.367 / 0.366 | 0.385 / 0.373 | 23 |
| ststjepa | 0.356 / 0.347 | 0.352 / 0.352 | 0.356 / 0.351 | 39 |
| hepa | 0.414 / 0.396 | 0.377 / 0.378 | 0.407 / 0.401 | 58 |
| subjepa | 0.380 / 0.369 | 0.364 / 0.366 | 0.381 / 0.373 | 26 |
| straight | 0.427 / 0.369 | 0.364 / 0.366 | 0.381 / 0.373 | 24 |
| motionjepa | 0.384 / 0.369 | 0.366 / 0.366 | 0.385 / 0.373 | 31 |
| njepa | 0.272 / 0.135 | 0.295 / 0.287 | 0.293 / 0.286 | 46 |
| gmmjepa | 0.271 / 0.135 | 0.279 / 0.287 | 0.284 / 0.286 | 39 |
| variational | 0.297 / 0.171 | 0.290 / 0.287 | 0.313 / 0.287 | 47 |
| phijepa | 0.299 / 0.120 | 0.284 / 0.287 | 0.303 / 0.282 | 45 |
| harjepa | 0.346 / 0.347 | 0.352 / 0.352 | 0.351 / 0.351 | 39 |
| no network (raw patches) |  | 0.345 |  |  |
| PCA (window 1), re-run | 0.307 |  |  |  |

#### M-Tuning (20 series): mean VUS-PR, trained / untrained, for the three score types. The first three rows are from Sections 7.9 and 7.10; `subjepa`, `straight` and `motionjepa` share the untrained control of `lewm`. Minutes: fitting and scoring of the trained run.

| version | `pred` | `embed` | `combined` | minutes |
|---|---|---|---|---|
| lejepa (default) | 0.363 / 0.353 | 0.373 / 0.375 | 0.368 / 0.368 | 6 |
| vicreg | 0.390 / 0.353 | 0.387 / 0.375 | 0.389 / 0.368 | 6 |
| lewm | 0.441 / 0.391 | 0.386 / 0.388 | 0.414 / 0.396 | 4 |
| ststjepa | 0.370 / 0.353 | 0.376 / 0.375 | 0.374 / 0.368 | 4 |
| hepa | 0.481 / 0.433 | 0.418 / 0.407 | 0.450 / 0.424 | 7 |
| subjepa | 0.426 / 0.391 | 0.385 / 0.388 | 0.403 / 0.396 | 3 |
| straight | 0.444 / 0.391 | 0.389 / 0.388 | 0.404 / 0.396 | 3 |
| motionjepa | 0.434 / 0.391 | 0.386 / 0.388 | 0.410 / 0.396 | 4 |
| njepa | 0.296 / 0.159 | 0.325 / 0.319 | 0.313 / 0.314 | 5 |
| gmmjepa | 0.350 / 0.159 | 0.321 / 0.319 | 0.332 / 0.314 | 5 |
| variational | 0.372 / 0.198 | 0.325 / 0.319 | 0.359 / 0.319 | 5 |
| phijepa | 0.383 / 0.169 | 0.328 / 0.319 | 0.358 / 0.316 | 5 |
| harjepa | 0.349 / 0.353 | 0.376 / 0.375 | 0.368 / 0.368 | 5 |
| no network (raw patches) |  | 0.374 |  |  |
| PCA (window 1), re-run | 0.323 |  |  |  |

#### U-Eva (350 series): mean VUS-PR, trained / untrained, for the three score types. The first three rows are from Sections 7.9 and 7.10; `subjepa`, `straight` and `motionjepa` share the untrained control of `lewm`. Minutes: fitting and scoring of the trained run.

| version | `pred` | `embed` | `combined` | minutes |
|---|---|---|---|---|
| lejepa (default) | 0.378 / 0.346 | 0.461 / 0.467 | 0.455 / 0.458 | 49 |
| vicreg | 0.417 / 0.346 | 0.462 / 0.467 | 0.463 / 0.458 | 51 |
| lewm | 0.394 / 0.309 | 0.462 / 0.472 | 0.482 / 0.473 | 24 |
| ststjepa | 0.386 / 0.346 | 0.467 / 0.467 | 0.465 / 0.458 | 41 |
| hepa | 0.418 / 0.373 | 0.509 / 0.513 | 0.523 / 0.513 | 62 |
| subjepa | 0.397 / 0.309 | 0.466 / 0.472 | 0.482 / 0.473 | 23 |
| straight | 0.398 / 0.309 | 0.468 / 0.472 | 0.484 / 0.473 | 22 |
| motionjepa | 0.388 / 0.309 | 0.462 / 0.472 | 0.480 / 0.473 | 29 |
| njepa | 0.326 / 0.141 | 0.331 / 0.288 | 0.380 / 0.294 | 44 |
| gmmjepa | 0.324 / 0.141 | 0.306 / 0.288 | 0.354 / 0.294 | 40 |
| variational | 0.323 / 0.119 | 0.301 / 0.288 | 0.343 / 0.290 | 46 |
| phijepa | 0.316 / 0.082 | 0.301 / 0.288 | 0.349 / 0.277 | 43 |
| harjepa | 0.348 / 0.346 | 0.466 / 0.467 | 0.458 / 0.458 | 41 |
| no network (raw patches) |  | 0.465 |  |  |
| Sub-PCA, re-run | 0.417 |  |  |  |

#### U-Tuning (48 series): mean VUS-PR, trained / untrained, for the three score types. The first three rows are from Sections 7.9 and 7.10; `subjepa`, `straight` and `motionjepa` share the untrained control of `lewm`. Minutes: fitting and scoring of the trained run.

| version | `pred` | `embed` | `combined` | minutes |
|---|---|---|---|---|
| lejepa (default) | 0.337 / 0.304 | 0.449 / 0.459 | 0.443 / 0.447 | 7 |
| vicreg | 0.410 / 0.304 | 0.449 / 0.459 | 0.449 / 0.447 | 5 |
| lewm | 0.398 / 0.304 | 0.435 / 0.437 | 0.454 / 0.439 | 3 |
| ststjepa | 0.346 / 0.306 | 0.457 / 0.459 | 0.450 / 0.446 | 5 |
| hepa | 0.338 / 0.293 | 0.487 / 0.498 | 0.507 / 0.497 | 7 |
| subjepa | 0.401 / 0.304 | 0.433 / 0.437 | 0.450 / 0.439 | 3 |
| straight | 0.396 / 0.304 | 0.435 / 0.437 | 0.452 / 0.439 | 3 |
| motionjepa | 0.393 / 0.304 | 0.434 / 0.437 | 0.454 / 0.439 | 3 |
| njepa | 0.302 / 0.127 | 0.315 / 0.282 | 0.369 / 0.282 | 5 |
| gmmjepa | 0.318 / 0.127 | 0.302 / 0.282 | 0.383 / 0.282 | 5 |
| variational | 0.317 / 0.122 | 0.295 / 0.282 | 0.372 / 0.286 | 6 |
| phijepa | 0.315 / 0.084 | 0.286 / 0.282 | 0.372 / 0.274 | 5 |
| harjepa | 0.306 / 0.304 | 0.458 / 0.459 | 0.446 / 0.447 | 5 |
| no network (raw patches) |  | 0.458 |  |  |
| Sub-PCA, re-run | 0.326 |  |  |  |

#### U-Rest (472 series): mean VUS-PR, trained / untrained, for the three score types. The first three rows are from Sections 7.9 and 7.10; `subjepa`, `straight` and `motionjepa` share the untrained control of `lewm`. Minutes: fitting and scoring of the trained run.

| version | `pred` | `embed` | `combined` | minutes |
|---|---|---|---|---|
| lejepa (default) | 0.177 / 0.146 | 0.432 / 0.435 | 0.401 / 0.403 | 48 |
| vicreg | 0.242 / 0.146 | 0.428 / 0.435 | 0.397 / 0.403 | 46 |
| lewm | 0.213 / 0.143 | 0.438 / 0.442 | 0.442 / 0.436 | 28 |
| ststjepa | 0.172 / 0.148 | 0.434 / 0.435 | 0.406 / 0.403 | 36 |
| hepa | 0.134 / 0.097 | 0.451 / 0.439 | 0.431 / 0.416 | 57 |
| subjepa | 0.203 / 0.143 | 0.438 / 0.442 | 0.440 / 0.436 | 24 |
| straight | 0.232 / 0.143 | 0.442 / 0.442 | 0.443 / 0.436 | 22 |
| motionjepa | 0.201 / 0.143 | 0.438 / 0.442 | 0.440 / 0.436 | 28 |
| njepa | 0.108 / 0.048 | 0.417 / 0.396 | 0.436 / 0.404 | 41 |
| gmmjepa | 0.133 / 0.048 | 0.406 / 0.396 | 0.436 / 0.404 | 38 |
| variational | 0.186 / 0.047 | 0.406 / 0.396 | 0.422 / 0.397 | 44 |
| phijepa | 0.154 / 0.036 | 0.401 / 0.396 | 0.430 / 0.371 | 44 |
| harjepa | 0.149 / 0.146 | 0.434 / 0.435 | 0.402 / 0.403 | 38 |
| no network (raw patches) |  | 0.545 |  |  |
| Sub-PCA, re-run | 0.146 |  |  |  |

#### Tests A of the plan: each new version, trained, `pred`, against the default version `lejepa` (trained). Cell: mean difference in VUS-PR, series better / worse, Holm-corrected p over all 100 tests (two-sided Wilcoxon signed-rank); a reading is given where p < 0.05. All numbers: `results/tsb_more_tests.csv`.

| version | M-Eva | M-Tuning | U-Eva | U-Tuning | U-Rest |
|---|---|---|---|---|---|
| ststjepa | +0.005, 106 / 67, p 0.138 | +0.007, 15 / 5, p 0.539 | +0.008, 184 / 157, p 0.475 | +0.009, 25 / 19, p 1.000 | -0.005, 195 / 259, p 0.045 **behind** |
| hepa | +0.064, 121 / 59, p <0.001 **ahead** | +0.118, 14 / 6, p 1.000 | +0.040, 212 / 134, p <0.001 **ahead** | +0.002, 26 / 21, p 1.000 | -0.043, 186 / 285, p <0.001 **behind** |
| subjepa | +0.029, 142 / 38, p <0.001 **ahead** | +0.063, 18 / 2, p 0.053 | +0.019, 205 / 139, p 0.012 **ahead** | +0.065, 30 / 17, p 0.475 | +0.025, 263 / 206, p 0.088 |
| straight | +0.077, 142 / 38, p <0.001 **ahead** | +0.081, 18 / 2, p 0.014 **ahead** | +0.020, 216 / 129, p <0.001 **ahead** | +0.060, 35 / 12, p 0.045 **ahead** | +0.055, 294 / 175, p <0.001 **ahead** |
| motionjepa | +0.033, 140 / 40, p <0.001 **ahead** | +0.071, 15 / 5, p 0.093 | +0.010, 197 / 148, p 0.311 | +0.057, 31 / 17, p 0.581 | +0.024, 264 / 205, p 0.278 |
| njepa | -0.079, 54 / 126, p <0.001 **behind** | -0.067, 5 / 15, p 1.000 | -0.052, 150 / 199, p 0.045 **behind** | -0.035, 21 / 27, p 1.000 | -0.070, 157 / 315, p <0.001 **behind** |
| gmmjepa | -0.080, 65 / 115, p <0.001 **behind** | -0.013, 5 / 15, p 1.000 | -0.054, 182 / 167, p 1.000 | -0.018, 25 / 23, p 1.000 | -0.044, 225 / 245, p 0.279 |
| variational | -0.053, 79 / 101, p 0.298 | +0.009, 11 / 9, p 1.000 | -0.055, 193 / 157, p 1.000 | -0.019, 24 / 24, p 1.000 | +0.009, 284 / 186, p 0.024 **ahead** |
| phijepa | -0.051, 82 / 98, p 0.613 | +0.020, 9 / 11, p 1.000 | -0.062, 190 / 159, p 1.000 | -0.021, 24 / 24, p 1.000 | -0.023, 261 / 210, p 1.000 |
| harjepa | -0.004, 65 / 108, p 0.029 **behind** | -0.014, 10 / 10, p 1.000 | -0.030, 104 / 235, p <0.001 **behind** | -0.030, 14 / 32, p 0.085 | -0.028, 106 / 351, p <0.001 **behind** |

#### Tests B of the plan: each new version, trained, `pred`, against its untrained control. Cell: mean difference in VUS-PR, series better / worse, Holm-corrected p over all 100 tests (two-sided Wilcoxon signed-rank); a reading is given where p < 0.05. All numbers: `results/tsb_more_tests.csv`.

| version | M-Eva | M-Tuning | U-Eva | U-Tuning | U-Rest |
|---|---|---|---|---|---|
| ststjepa | +0.009, 110 / 62, p <0.001 **training helps** | +0.017, 9 / 10, p 1.000 | +0.040, 246 / 83, p <0.001 **training helps** | +0.040, 34 / 9, p 0.004 **training helps** | +0.024, 330 / 102, p <0.001 **training helps** |
| hepa | +0.019, 90 / 86, p 1.000 | +0.048, 9 / 10, p 1.000 | +0.045, 216 / 126, p <0.001 **training helps** | +0.045, 24 / 23, p 1.000 | +0.037, 291 / 176, p <0.001 **training helps** |
| subjepa | +0.011, 122 / 56, p 0.025 **training helps** | +0.036, 15 / 5, p 1.000 | +0.088, 245 / 99, p <0.001 **training helps** | +0.098, 28 / 20, p 0.155 | +0.059, 277 / 188, p <0.001 **training helps** |
| straight | +0.059, 132 / 47, p <0.001 **training helps** | +0.053, 15 / 5, p 1.000 | +0.089, 254 / 92, p <0.001 **training helps** | +0.092, 35 / 10, p 0.003 **training helps** | +0.089, 357 / 101, p <0.001 **training helps** |
| motionjepa | +0.015, 120 / 58, p 0.016 **training helps** | +0.043, 14 / 6, p 1.000 | +0.079, 238 / 106, p <0.001 **training helps** | +0.090, 28 / 19, p 0.344 | +0.058, 288 / 176, p <0.001 **training helps** |
| njepa | +0.137, 130 / 50, p <0.001 **training helps** | +0.137, 14 / 6, p 0.422 | +0.185, 262 / 88, p <0.001 **training helps** | +0.175, 37 / 11, p <0.001 **training helps** | +0.060, 285 / 187, p <0.001 **training helps** |
| gmmjepa | +0.136, 135 / 45, p <0.001 **training helps** | +0.191, 15 / 5, p 0.085 | +0.183, 290 / 60, p <0.001 **training helps** | +0.191, 38 / 10, p <0.001 **training helps** | +0.085, 364 / 108, p <0.001 **training helps** |
| variational | +0.127, 133 / 47, p <0.001 **training helps** | +0.173, 16 / 4, p 0.034 **training helps** | +0.204, 301 / 49, p <0.001 **training helps** | +0.195, 39 / 9, p <0.001 **training helps** | +0.139, 405 / 67, p <0.001 **training helps** |
| phijepa | +0.180, 155 / 25, p <0.001 **training helps** | +0.214, 18 / 2, p 0.029 **training helps** | +0.234, 311 / 39, p <0.001 **training helps** | +0.232, 41 / 7, p <0.001 **training helps** | +0.118, 394 / 78, p <0.001 **training helps** |
| harjepa | -0.000, 88 / 86, p 1.000 | -0.004, 11 / 9, p 1.000 | +0.003, 224 / 105, p <0.001 **training helps** | +0.002, 31 / 14, p 0.095 | +0.002, 306 / 130, p <0.001 **training helps** |

#### Tally of the 100 tests

| tests | ahead | behind | not separated | training helps |
|---|---|---|---|---|
| A: against the default | 11 | 9 | 30 | 0 |
| B: against the untrained control | 0 | 0 | 14 | 36 |

#### All twenty-six versions: the three highest means of trained `pred`, the place of each new version among the twenty-six, and the highest of the 156 cells (version, trained or not, score type). The cell is picked after the results.

| list | versions | highest `pred`, trained | places of the ten | highest cell |
|---|---|---|---|---|
| M-Eva | 26 | straight 0.427, hepa 0.414, lewm 0.386 | ststjepa 7, hepa 2, subjepa 5, straight 1, motionjepa 4, njepa 24, gmmjepa 25, variational 18, phijepa 17, harjepa 12 | 0.436 (cfjepa, untrained, embed) |
| M-Tuning | 26 | hepa 0.481, straight 0.444, lewm 0.441 | ststjepa 13, hepa 1, subjepa 5, straight 2, motionjepa 4, njepa 26, gmmjepa 19, variational 11, phijepa 7, harjepa 20 | 0.528 (cfjepa, trained, embed) |
| U-Eva | 26 | hepa 0.418, vicreg 0.417, straight 0.398 | ststjepa 8, hepa 1, subjepa 4, straight 3, motionjepa 7, njepa 18, gmmjepa 19, variational 20, phijepa 22, harjepa 15 | 0.530 (tsjepa, trained, embed) |
| U-Tuning | 26 | vicreg 0.410, subjepa 0.401, lewm 0.398 | ststjepa 7, hepa 10, subjepa 2, straight 4, motionjepa 5, njepa 24, gmmjepa 16, variational 18, phijepa 19, harjepa 21 | 0.508 (cfjepa, trained, embed) |
| U-Rest | 26 | lenepa 0.342, vicreg 0.242, straight 0.232 | ststjepa 12, hepa 23, subjepa 6, straight 3, motionjepa 7, njepa 26, gmmjepa 24, variational 8, phijepa 16, harjepa 17 | 0.511 (tjepa, trained, combined) |

#### Not in the plan: mean VUS-PR of trained `pred` per source dataset, all 200 multivariate series (the lists pooled); `lewm`, `subjepa`, `motionjepa` and `straight` share one network

| dataset | series | lejepa | lewm | subjepa | motionjepa | straight | hepa | straight better / worse than lejepa | hepa better / worse than lejepa |
|---|---|---|---|---|---|---|---|---|---|
| SVDB | 31 | 0.168 | 0.270 | 0.240 | 0.260 | 0.244 | 0.347 | 30 / 1 | 31 / 0 |
| Exathlon | 27 | 0.898 | 0.901 | 0.903 | 0.902 | 0.899 | 0.903 | 23 / 4 | 27 / 0 |
| SMAP | 27 | 0.286 | 0.376 | 0.370 | 0.373 | 0.373 | 0.517 | 23 / 4 | 20 / 7 |
| GHL | 25 | 0.015 | 0.014 | 0.014 | 0.024 | 0.348 | 0.045 | 24 / 1 | 12 / 13 |
| SMD | 22 | 0.434 | 0.436 | 0.428 | 0.435 | 0.450 | 0.398 | 13 / 9 | 8 / 14 |
| MSL | 16 | 0.399 | 0.438 | 0.441 | 0.439 | 0.439 | 0.444 | 13 / 3 | 11 / 5 |
| MITDB | 13 | 0.074 | 0.121 | 0.108 | 0.111 | 0.113 | 0.169 | 11 / 2 | 13 / 0 |
| TAO | 13 | 0.841 | 0.870 | 0.871 | 0.872 | 0.858 | 0.771 | 12 / 1 | 0 / 13 |
| OPPORTUNITY | 8 | 0.128 | 0.134 | 0.132 | 0.127 | 0.132 | 0.130 | 3 / 5 | 3 / 5 |
| CATSv2 | 6 | 0.161 | 0.171 | 0.158 | 0.141 | 0.136 | 0.139 | 1 / 5 | 2 / 4 |
| LTDB | 5 | 0.295 | 0.409 | 0.405 | 0.399 | 0.360 | 0.478 | 3 / 2 | 4 / 1 |
| SWaT | 2 | 0.357 | 0.338 | 0.358 | 0.348 | 0.403 | 0.370 | 2 / 0 | 1 / 1 |
| CreditCard | 1 | 0.123 | 0.092 | 0.093 | 0.093 | 0.095 | 0.098 | 0 / 1 | 0 / 1 |
| PSM | 1 | 0.197 | 0.207 | 0.330 | 0.265 | 0.274 | 0.200 | 1 / 0 | 1 / 0 |
| Genesis | 1 | 0.030 | 0.211 | 0.027 | 0.036 | 0.020 | 0.014 | 0 / 1 | 0 / 1 |
| GECCO | 1 | 0.039 | 0.035 | 0.039 | 0.037 | 0.329 | 0.048 | 1 / 0 | 1 / 0 |
| Daphnet | 1 | 0.438 | 0.334 | 0.351 | 0.297 | 0.415 | 0.508 | 0 / 1 | 1 / 0 |

#### Not in the plan: mean VUS-PR of trained `pred` per source dataset, all 870 univariate series (the lists pooled); `lewm`, `subjepa`, `motionjepa` and `straight` share one network

| dataset | series | lejepa | lewm | subjepa | motionjepa | straight | hepa | straight better / worse than lejepa | hepa better / worse than lejepa |
|---|---|---|---|---|---|---|---|---|---|
| YAHOO | 259 | 0.215 | 0.208 | 0.193 | 0.201 | 0.240 | 0.142 | 139 / 117 | 79 / 179 |
| UCR | 228 | 0.036 | 0.089 | 0.073 | 0.070 | 0.076 | 0.065 | 162 / 66 | 133 / 95 |
| WSD | 111 | 0.265 | 0.383 | 0.416 | 0.378 | 0.412 | 0.184 | 77 / 31 | 33 / 76 |
| SMD | 38 | 0.506 | 0.689 | 0.662 | 0.688 | 0.681 | 0.679 | 33 / 5 | 32 / 6 |
| Exathlon | 32 | 0.923 | 0.787 | 0.844 | 0.796 | 0.878 | 0.948 | 16 / 16 | 26 / 6 |
| OPPORTUNITY | 29 | 0.804 | 0.568 | 0.556 | 0.546 | 0.571 | 0.806 | 7 / 22 | 22 / 7 |
| NAB | 28 | 0.276 | 0.289 | 0.289 | 0.286 | 0.307 | 0.325 | 23 / 5 | 23 / 5 |
| Stock | 20 | 0.719 | 0.694 | 0.693 | 0.696 | 0.690 | 0.668 | 0 / 20 | 0 / 20 |
| SVDB | 20 | 0.128 | 0.299 | 0.256 | 0.288 | 0.233 | 0.410 | 19 / 1 | 20 / 0 |
| SMAP | 19 | 0.448 | 0.488 | 0.495 | 0.480 | 0.485 | 0.560 | 12 / 6 | 10 / 8 |
| IOPS | 17 | 0.375 | 0.328 | 0.379 | 0.312 | 0.371 | 0.257 | 8 / 8 | 0 / 16 |
| TODS | 15 | 0.487 | 0.488 | 0.486 | 0.481 | 0.500 | 0.492 | 10 / 5 | 10 / 5 |
| NEK | 9 | 0.947 | 0.932 | 0.932 | 0.932 | 0.938 | 0.885 | 6 / 3 | 3 / 6 |
| MSL | 9 | 0.388 | 0.410 | 0.400 | 0.406 | 0.415 | 0.360 | 5 / 3 | 3 / 5 |
| MGAB | 9 | 0.008 | 0.182 | 0.164 | 0.174 | 0.072 | 0.035 | 9 / 0 | 9 / 0 |
| LTDB | 9 | 0.316 | 0.413 | 0.412 | 0.420 | 0.377 | 0.443 | 6 / 3 | 8 / 1 |
| MITDB | 8 | 0.101 | 0.168 | 0.168 | 0.169 | 0.155 | 0.196 | 7 / 1 | 8 / 0 |
| SED | 3 | 0.048 | 0.067 | 0.056 | 0.064 | 0.118 | 0.048 | 3 / 0 | 2 / 1 |
| TAO | 3 | 0.911 | 0.889 | 0.888 | 0.889 | 0.886 | 0.876 | 0 / 3 | 0 / 3 |
| CATSv2 | 1 | 0.259 | 0.339 | 0.339 | 0.329 | 0.389 | 0.392 | 1 / 0 | 1 / 0 |
| SWaT | 1 | 0.428 | 0.413 | 0.521 | 0.392 | 0.430 | 0.367 | 1 / 0 | 0 / 1 |
| Daphnet | 1 | 0.462 | 0.424 | 0.447 | 0.423 | 0.450 | 0.488 | 0 / 1 | 1 / 0 |
| Power | 1 | 0.074 | 0.077 | 0.082 | 0.078 | 0.088 | 0.125 | 1 / 0 | 1 / 0 |

#### Not in the plan: `straight` with two more seeds (trained, `pred`, mean VUS-PR; difference to the default version with seed 42, series better / worse, uncorrected p)

| list | lejepa, seed 42 | straight, seed 42 | straight, seed 0 | seed 0 against lejepa | straight, seed 1 | seed 1 against lejepa |
|---|---|---|---|---|---|---|
| M-Eva | 0.351 | 0.427 | 0.422 | +0.071, 135 / 45, p <0.001 | 0.423 | +0.072, 133 / 47, p <0.001 |
| M-Tuning | 0.363 | 0.444 | 0.439 | +0.076, 16 / 4, p 0.003 | 0.443 | +0.080, 16 / 4, p 0.002 |
| U-Eva | 0.378 | 0.398 | 0.422 | +0.044, 225 / 120, p <0.001 | 0.417 | +0.039, 240 / 105, p <0.001 |
| U-Tuning | 0.337 | 0.396 | 0.397 | +0.061, 31 / 16, p 0.006 | 0.382 | +0.045, 33 / 13, p 0.006 |
| U-Rest | 0.177 | 0.232 | 0.253 | +0.076, 349 / 116, p <0.001 | 0.211 | +0.033, 313 / 154, p <0.001 |

#### Small benchmarks (protocol of Section 7.8, 5 seeds): mean ROC-AUC, trained / zero-epoch control

| version | synthetic `pred` | synthetic `embed` | synthetic `combined` | SKAB `pred` | SKAB `embed` | SKAB `combined` | NAB `pred` | NAB `embed` | NAB `combined` |
|---|---|---|---|---|---|---|---|---|---|
| lejepa (default) | 0.941 / 0.798 | 0.928 / 0.932 | 0.936 / 0.877 | 0.825 / 0.831 | 0.825 / 0.828 | 0.825 / 0.829 | 0.628 / 0.671 | 0.627 / 0.619 | 0.629 / 0.658 |
| vicreg | 0.935 / 0.798 | 0.928 / 0.932 | 0.933 / 0.877 | 0.829 / 0.831 | 0.830 / 0.828 | 0.829 / 0.829 | 0.576 / 0.671 | 0.587 / 0.619 | 0.590 / 0.658 |
| lewm | 0.933 / 0.799 | 0.940 / 0.943 | 0.948 / 0.888 | 0.726 / 0.810 | 0.879 / 0.873 | 0.846 / 0.868 | 0.622 / 0.661 | 0.643 / 0.642 | 0.656 / 0.662 |
| ststjepa | 0.907 / 0.795 | 0.921 / 0.932 | 0.916 / 0.874 | 0.829 / 0.831 | 0.822 / 0.828 | 0.825 / 0.829 | 0.700 / 0.683 | 0.633 / 0.619 | 0.661 / 0.660 |
| hepa | 0.838 / 0.658 | 0.940 / 0.940 | 0.922 / 0.863 | 0.780 / 0.874 | 0.864 / 0.876 | 0.858 / 0.871 | 0.668 / 0.689 | 0.640 / 0.658 | 0.671 / 0.709 |
| subjepa | 0.946 / 0.799 | 0.942 / 0.943 | 0.951 / 0.888 | 0.762 / 0.810 | 0.852 / 0.873 | 0.835 / 0.868 | 0.640 / 0.661 | 0.643 / 0.642 | 0.658 / 0.662 |
| straight | 0.894 / 0.799 | 0.938 / 0.943 | 0.936 / 0.888 | 0.632 / 0.810 | 0.756 / 0.873 | 0.701 / 0.868 | 0.622 / 0.661 | 0.646 / 0.642 | 0.665 / 0.662 |
| motionjepa | 0.936 / 0.799 | 0.940 / 0.943 | 0.949 / 0.888 | 0.795 / 0.810 | 0.881 / 0.873 | 0.853 / 0.868 | 0.627 / 0.661 | 0.642 / 0.642 | 0.656 / 0.662 |
| njepa | 0.879 / 0.519 | 0.932 / 0.891 | 0.951 / 0.879 | 0.464 / 0.422 | 0.738 / 0.749 | 0.609 / 0.710 | 0.674 / 0.521 | 0.628 / 0.552 | 0.667 / 0.536 |
| gmmjepa | 0.896 / 0.519 | 0.900 / 0.891 | 0.905 / 0.879 | 0.673 / 0.422 | 0.771 / 0.749 | 0.798 / 0.710 | 0.664 / 0.521 | 0.582 / 0.552 | 0.630 / 0.536 |
| variational | 0.885 / 0.483 | 0.895 / 0.891 | 0.895 / 0.885 | 0.728 / 0.449 | 0.718 / 0.749 | 0.740 / 0.698 | 0.616 / 0.513 | 0.581 / 0.552 | 0.606 / 0.541 |
| phijepa | 0.885 / 0.527 | 0.904 / 0.891 | 0.899 / 0.877 | 0.603 / 0.570 | 0.560 / 0.749 | 0.589 / 0.750 | 0.589 / 0.473 | 0.592 / 0.552 | 0.604 / 0.531 |
| harjepa | 0.814 / 0.798 | 0.931 / 0.932 | 0.882 / 0.877 | 0.833 / 0.831 | 0.830 / 0.828 | 0.830 / 0.829 | 0.695 / 0.671 | 0.621 / 0.619 | 0.660 / 0.658 |

#### Small benchmarks: the 60 tests of the plan on `block[pred]` (two-sided Wilcoxon signed-rank over series, Holm over all of them)

|  | A | B | group | series | ROC-AUC A / B | A − B | A / B better | p | p (Holm) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | ststjepa, trained | its untrained control | synthetic | 15 | 0.907 / 0.795 | +0.112 | 15 / 0 | <0.001 | 0.003 |
| 2 | ststjepa, trained | its untrained control | SKAB | 16 | 0.829 / 0.831 | -0.002 | 0 / 16 | <0.001 | 0.002 |
| 3 | ststjepa, trained | its untrained control | NAB | 7 | 0.700 / 0.683 | +0.016 | 5 / 2 | 0.375 | 1.000 |
| 4 | hepa, trained | its untrained control | synthetic | 15 | 0.838 / 0.658 | +0.180 | 11 / 4 | 0.007 | 0.255 |
| 5 | hepa, trained | its untrained control | SKAB | 16 | 0.780 / 0.874 | -0.095 | 0 / 16 | <0.001 | 0.002 |
| 6 | hepa, trained | its untrained control | NAB | 7 | 0.668 / 0.689 | -0.021 | 2 / 5 | 0.469 | 1.000 |
| 7 | subjepa, trained | its untrained control | synthetic | 15 | 0.946 / 0.799 | +0.147 | 15 / 0 | <0.001 | 0.003 |
| 8 | subjepa, trained | its untrained control | SKAB | 16 | 0.762 / 0.810 | -0.048 | 5 / 11 | 0.011 | 0.396 |
| 9 | subjepa, trained | its untrained control | NAB | 7 | 0.640 / 0.661 | -0.021 | 3 / 4 | 0.812 | 1.000 |
| 10 | straight, trained | its untrained control | synthetic | 15 | 0.894 / 0.799 | +0.095 | 12 / 3 | 0.007 | 0.255 |
| 11 | straight, trained | its untrained control | SKAB | 16 | 0.632 / 0.810 | -0.178 | 4 / 12 | 0.011 | 0.396 |
| 12 | straight, trained | its untrained control | NAB | 7 | 0.622 / 0.661 | -0.039 | 3 / 4 | 0.578 | 1.000 |
| 13 | motionjepa, trained | its untrained control | synthetic | 15 | 0.936 / 0.799 | +0.137 | 15 / 0 | <0.001 | 0.003 |
| 14 | motionjepa, trained | its untrained control | SKAB | 16 | 0.795 / 0.810 | -0.014 | 7 / 9 | 0.562 | 1.000 |
| 15 | motionjepa, trained | its untrained control | NAB | 7 | 0.627 / 0.661 | -0.033 | 3 / 4 | 0.578 | 1.000 |
| 16 | njepa, trained | its untrained control | synthetic | 15 | 0.879 / 0.519 | +0.360 | 15 / 0 | <0.001 | 0.003 |
| 17 | njepa, trained | its untrained control | SKAB | 16 | 0.464 / 0.422 | +0.041 | 9 / 7 | 0.528 | 1.000 |
| 18 | njepa, trained | its untrained control | NAB | 7 | 0.674 / 0.521 | +0.153 | 7 / 0 | 0.016 | 0.531 |
| 19 | gmmjepa, trained | its untrained control | synthetic | 15 | 0.896 / 0.519 | +0.378 | 15 / 0 | <0.001 | 0.003 |
| 20 | gmmjepa, trained | its untrained control | SKAB | 16 | 0.673 / 0.422 | +0.251 | 16 / 0 | <0.001 | 0.002 |
| 21 | gmmjepa, trained | its untrained control | NAB | 7 | 0.664 / 0.521 | +0.143 | 6 / 1 | 0.078 | 1.000 |
| 22 | variational, trained | its untrained control | synthetic | 15 | 0.885 / 0.483 | +0.402 | 15 / 0 | <0.001 | 0.003 |
| 23 | variational, trained | its untrained control | SKAB | 16 | 0.728 / 0.449 | +0.279 | 14 / 2 | <0.001 | 0.018 |
| 24 | variational, trained | its untrained control | NAB | 7 | 0.616 / 0.513 | +0.103 | 6 / 1 | 0.156 | 1.000 |
| 25 | phijepa, trained | its untrained control | synthetic | 15 | 0.885 / 0.527 | +0.358 | 15 / 0 | <0.001 | 0.003 |
| 26 | phijepa, trained | its untrained control | SKAB | 16 | 0.603 / 0.570 | +0.033 | 9 / 7 | 0.231 | 1.000 |
| 27 | phijepa, trained | its untrained control | NAB | 7 | 0.589 / 0.473 | +0.116 | 6 / 1 | 0.156 | 1.000 |
| 28 | harjepa, trained | its untrained control | synthetic | 15 | 0.814 / 0.798 | +0.016 | 14 / 1 | <0.001 | 0.035 |
| 29 | harjepa, trained | its untrained control | SKAB | 16 | 0.833 / 0.831 | +0.002 | 8 / 8 | 0.562 | 1.000 |
| 30 | harjepa, trained | its untrained control | NAB | 7 | 0.695 / 0.671 | +0.024 | 5 / 2 | 0.109 | 1.000 |
| 31 | ststjepa, trained | lejepa, trained | synthetic | 15 | 0.907 / 0.941 | -0.034 | 6 / 9 | 0.035 | 1.000 |
| 32 | ststjepa, trained | lejepa, trained | SKAB | 16 | 0.829 / 0.825 | +0.004 | 14 / 2 | <0.001 | 0.013 |
| 33 | ststjepa, trained | lejepa, trained | NAB | 7 | 0.700 / 0.628 | +0.071 | 5 / 2 | 0.156 | 1.000 |
| 34 | hepa, trained | lejepa, trained | synthetic | 15 | 0.838 / 0.941 | -0.104 | 1 / 14 | <0.001 | 0.006 |
| 35 | hepa, trained | lejepa, trained | SKAB | 16 | 0.780 / 0.825 | -0.045 | 5 / 11 | 0.039 | 1.000 |
| 36 | hepa, trained | lejepa, trained | NAB | 7 | 0.668 / 0.628 | +0.040 | 4 / 3 | 0.578 | 1.000 |
| 37 | subjepa, trained | lejepa, trained | synthetic | 15 | 0.946 / 0.941 | +0.004 | 11 / 4 | 0.169 | 1.000 |
| 38 | subjepa, trained | lejepa, trained | SKAB | 16 | 0.762 / 0.825 | -0.063 | 2 / 14 | <0.001 | 0.007 |
| 39 | subjepa, trained | lejepa, trained | NAB | 7 | 0.640 / 0.628 | +0.011 | 4 / 3 | 0.812 | 1.000 |
| 40 | straight, trained | lejepa, trained | synthetic | 15 | 0.894 / 0.941 | -0.047 | 1 / 14 | 0.003 | 0.131 |
| 41 | straight, trained | lejepa, trained | SKAB | 16 | 0.632 / 0.825 | -0.193 | 3 / 13 | <0.001 | 0.024 |
| 42 | straight, trained | lejepa, trained | NAB | 7 | 0.622 / 0.628 | -0.006 | 3 / 4 | 0.812 | 1.000 |
| 43 | motionjepa, trained | lejepa, trained | synthetic | 15 | 0.936 / 0.941 | -0.005 | 7 / 8 | 0.639 | 1.000 |
| 44 | motionjepa, trained | lejepa, trained | SKAB | 16 | 0.795 / 0.825 | -0.029 | 7 / 9 | 0.065 | 1.000 |
| 45 | motionjepa, trained | lejepa, trained | NAB | 7 | 0.627 / 0.628 | -0.001 | 4 / 3 | 0.938 | 1.000 |
| 46 | njepa, trained | lejepa, trained | synthetic | 15 | 0.879 / 0.941 | -0.063 | 5 / 10 | 0.095 | 1.000 |
| 47 | njepa, trained | lejepa, trained | SKAB | 16 | 0.464 / 0.825 | -0.361 | 0 / 16 | <0.001 | 0.002 |
| 48 | njepa, trained | lejepa, trained | NAB | 7 | 0.674 / 0.628 | +0.046 | 3 / 4 | 0.688 | 1.000 |
| 49 | gmmjepa, trained | lejepa, trained | synthetic | 15 | 0.896 / 0.941 | -0.045 | 4 / 11 | 0.026 | 0.844 |
| 50 | gmmjepa, trained | lejepa, trained | SKAB | 16 | 0.673 / 0.825 | -0.152 | 0 / 16 | <0.001 | 0.002 |
| 51 | gmmjepa, trained | lejepa, trained | NAB | 7 | 0.664 / 0.628 | +0.036 | 4 / 3 | 0.297 | 1.000 |
| 52 | variational, trained | lejepa, trained | synthetic | 15 | 0.885 / 0.941 | -0.057 | 3 / 12 | 0.048 | 1.000 |
| 53 | variational, trained | lejepa, trained | SKAB | 16 | 0.728 / 0.825 | -0.097 | 7 / 9 | 0.159 | 1.000 |
| 54 | variational, trained | lejepa, trained | NAB | 7 | 0.616 / 0.628 | -0.012 | 3 / 4 | 0.578 | 1.000 |
| 55 | phijepa, trained | lejepa, trained | synthetic | 15 | 0.885 / 0.941 | -0.057 | 4 / 11 | 0.055 | 1.000 |
| 56 | phijepa, trained | lejepa, trained | SKAB | 16 | 0.603 / 0.825 | -0.222 | 0 / 16 | <0.001 | 0.002 |
| 57 | phijepa, trained | lejepa, trained | NAB | 7 | 0.589 / 0.628 | -0.039 | 3 / 4 | 0.688 | 1.000 |
| 58 | harjepa, trained | lejepa, trained | synthetic | 15 | 0.814 / 0.941 | -0.127 | 0 / 15 | <0.001 | 0.003 |
| 59 | harjepa, trained | lejepa, trained | SKAB | 16 | 0.833 / 0.825 | +0.008 | 14 / 2 | 0.002 | 0.067 |
| 60 | harjepa, trained | lejepa, trained | NAB | 7 | 0.695 / 0.628 | +0.066 | 3 / 4 | 0.578 | 1.000 |
