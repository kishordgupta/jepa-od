Coverage: `scjepa`, trained and untrained, on 5 lists (1070 series); every run has every series; series with an error: 0; fitting and scoring took 3.3 hours in all; largest GPU memory of a run 1.7 GB.

#### Mean VUS-PR, trained / untrained (`straight` shares the untrained control of `lewm`; the ablation was run trained only); the other rows are from Sections 7.9 to 7.11

| version | score | M-Eva | M-Tuning | U-Eva | U-Tuning | U-Rest |
|---|---|---|---|---|---|---|
| lejepa (default) | `pred` | 0.351 / 0.347 | 0.363 / 0.353 | 0.378 / 0.346 | 0.337 / 0.304 | 0.177 / 0.146 |
| lejepa (default) | `embed` | 0.353 / 0.352 | 0.373 / 0.375 | 0.461 / 0.467 | 0.449 / 0.459 | 0.432 / 0.435 |
| lejepa (default) | `combined` | 0.352 / 0.351 | 0.368 / 0.368 | 0.455 / 0.458 | 0.443 / 0.447 | 0.401 / 0.403 |
| lewm | `pred` | 0.386 / 0.369 | 0.441 / 0.391 | 0.394 / 0.309 | 0.398 / 0.304 | 0.213 / 0.143 |
| lewm | `embed` | 0.367 / 0.366 | 0.386 / 0.388 | 0.462 / 0.472 | 0.435 / 0.437 | 0.438 / 0.442 |
| lewm | `combined` | 0.385 / 0.373 | 0.414 / 0.396 | 0.482 / 0.473 | 0.454 / 0.439 | 0.442 / 0.436 |
| straight | `pred` | 0.427 / 0.369 | 0.444 / 0.391 | 0.398 / 0.309 | 0.396 / 0.304 | 0.232 / 0.143 |
| straight | `embed` | 0.364 / 0.366 | 0.389 / 0.388 | 0.468 / 0.472 | 0.435 / 0.437 | 0.442 / 0.442 |
| straight | `combined` | 0.381 / 0.373 | 0.404 / 0.396 | 0.484 / 0.473 | 0.452 / 0.439 | 0.443 / 0.436 |
| hepa | `pred` | 0.414 / 0.396 | 0.481 / 0.433 | 0.418 / 0.373 | 0.338 / 0.293 | 0.134 / 0.097 |
| hepa | `embed` | 0.377 / 0.378 | 0.418 / 0.407 | 0.509 / 0.513 | 0.487 / 0.498 | 0.451 / 0.439 |
| hepa | `combined` | 0.407 / 0.401 | 0.450 / 0.424 | 0.523 / 0.513 | 0.507 / 0.497 | 0.431 / 0.416 |
| scjepa | `pred` | 0.122 / 0.107 | 0.184 / 0.164 | 0.150 / 0.116 | 0.150 / 0.118 | 0.132 / 0.056 |
| scjepa | `embed` | 0.190 / 0.167 | 0.244 / 0.237 | 0.325 / 0.318 | 0.316 / 0.307 | 0.529 / 0.529 |
| scjepa | `combined` | 0.166 / 0.152 | 0.221 / 0.209 | 0.305 / 0.298 | 0.303 / 0.301 | 0.517 / 0.517 |
| scjepa without normalisation (ablation) | `pred` | 0.259 | 0.306 | 0.352 | 0.301 | 0.121 |
| scjepa without normalisation (ablation) | `embed` | 0.346 | 0.388 | 0.475 | 0.437 | 0.307 |
| scjepa without normalisation (ablation) | `combined` | 0.351 | 0.394 | 0.475 | 0.430 | 0.297 |
| no network (raw patches) | `embed` | 0.345 | 0.374 | 0.465 | 0.458 | 0.545 |
| PCA / Sub-PCA, re-run |  | 0.307 | 0.323 | 0.417 | 0.326 | 0.146 |

#### The ten tests of the addendum: A = `scjepa`, trained, `pred` (two-sided Wilcoxon signed-rank over series, Holm over the ten; CI: bootstrap over series)

|  | against | list | series | VUS-PR A / B | A − B [95 % CI] | A / B better | p | p (Holm) | reading |
|---|---|---|---|---|---|---|---|---|---|
| 1 | the default version `lejepa`, trained | M-Eva | 180 | 0.122 / 0.351 | -0.229 [-0.277, -0.183] | 33 / 147 | <0.001 | <0.001 | behind |
| 2 | the default version `lejepa`, trained | M-Tuning | 20 | 0.184 / 0.363 | -0.179 [-0.307, -0.070] | 4 / 16 | 0.002 | 0.010 | behind |
| 3 | the default version `lejepa`, trained | U-Eva | 350 | 0.150 / 0.378 | -0.229 [-0.267, -0.191] | 114 / 235 | <0.001 | <0.001 | behind |
| 4 | the default version `lejepa`, trained | U-Tuning | 48 | 0.150 / 0.337 | -0.187 [-0.290, -0.091] | 17 / 31 | 0.002 | 0.012 | behind |
| 5 | the default version `lejepa`, trained | U-Rest | 472 | 0.132 / 0.177 | -0.046 [-0.071, -0.021] | 230 / 242 | 0.016 | 0.033 | behind |
| 6 | its untrained control | M-Eva | 180 | 0.122 / 0.107 | +0.015 [+0.004, +0.027] | 110 / 70 | 0.009 | 0.028 | training helps |
| 7 | its untrained control | M-Tuning | 20 | 0.184 / 0.164 | +0.019 [-0.005, +0.049] | 12 / 8 | 0.261 | 0.261 | not separated |
| 8 | its untrained control | U-Eva | 350 | 0.150 / 0.116 | +0.033 [+0.019, +0.048] | 252 / 98 | <0.001 | <0.001 | training helps |
| 9 | its untrained control | U-Tuning | 48 | 0.150 / 0.118 | +0.032 [+0.002, +0.067] | 32 / 16 | 0.005 | 0.019 | training helps |
| 10 | its untrained control | U-Rest | 472 | 0.132 / 0.056 | +0.075 [+0.062, +0.090] | 380 / 92 | <0.001 | <0.001 | training helps |

#### Place of `scjepa` among the 27 trained versions, per score type

| list | `pred` | `embed` | `combined` |
|---|---|---|---|
| M-Eva | 0.122, place 27 of 27 (first: straight 0.427) | 0.190, place 27 of 27 (first: cfjepa 0.434) | 0.166, place 27 of 27 (first: cfjepa 0.431) |
| M-Tuning | 0.184, place 27 of 27 (first: hepa 0.481) | 0.244, place 27 of 27 (first: cfjepa 0.528) | 0.221, place 27 of 27 (first: cfjepa 0.525) |
| U-Eva | 0.150, place 27 of 27 (first: hepa 0.418) | 0.325, place 19 of 27 (first: tsjepa 0.530) | 0.305, place 27 of 27 (first: cfjepa 0.524) |
| U-Tuning | 0.150, place 27 of 27 (first: vicreg 0.410) | 0.316, place 18 of 27 (first: cfjepa 0.508) | 0.303, place 27 of 27 (first: hepa 0.507) |
| U-Rest | 0.132, place 26 of 27 (first: lenepa 0.342) | 0.529, place 1 of 27 (first: scjepa 0.529) | 0.517, place 1 of 27 (first: scjepa 0.517) |

#### Not in the plan: `scjepa` without the normalisation of the two halves (`variant_params={'instance_norm': False}`), trained, `pred`. Differences: mean, series better / worse, two-sided Wilcoxon signed-rank p, not corrected.

| list | without / with normalisation | difference | without normalisation / default version | difference to the default |
|---|---|---|---|---|
| M-Eva | 0.259 / 0.122 | +0.137, 141 / 39, p <0.001 | 0.259 / 0.351 | -0.092, 83 / 97, p 0.001 |
| M-Tuning | 0.306 / 0.184 | +0.122, 15 / 5, p 0.008 | 0.306 / 0.363 | -0.057, 10 / 10, p 0.571 |
| U-Eva | 0.352 / 0.150 | +0.203, 251 / 98, p <0.001 | 0.352 / 0.378 | -0.026, 183 / 164, p 0.433 |
| U-Tuning | 0.301 / 0.150 | +0.151, 32 / 16, p 0.001 | 0.301 / 0.337 | -0.036, 22 / 26, p 0.373 |
| U-Rest | 0.121 / 0.132 | -0.011, 191 / 281, p <0.001 | 0.121 / 0.177 | -0.057, 166 / 302, p <0.001 |

#### Not in the plan: mean VUS-PR of trained `pred` per source dataset, all 200 multivariate series (the lists pooled)

| dataset | series | lejepa | lewm | straight | hepa | scjepa | scjepa better / worse than lejepa |
|---|---|---|---|---|---|---|---|
| SVDB | 31 | 0.168 | 0.270 | 0.244 | 0.347 | 0.137 | 10 / 21 |
| Exathlon | 27 | 0.898 | 0.901 | 0.899 | 0.903 | 0.120 | 0 / 27 |
| SMAP | 27 | 0.286 | 0.376 | 0.373 | 0.517 | 0.045 | 1 / 26 |
| GHL | 25 | 0.015 | 0.014 | 0.348 | 0.045 | 0.010 | 9 / 16 |
| SMD | 22 | 0.434 | 0.436 | 0.450 | 0.398 | 0.063 | 1 / 21 |
| MSL | 16 | 0.399 | 0.438 | 0.439 | 0.444 | 0.086 | 0 / 16 |
| MITDB | 13 | 0.074 | 0.121 | 0.113 | 0.169 | 0.076 | 6 / 7 |
| TAO | 13 | 0.841 | 0.870 | 0.858 | 0.771 | 0.806 | 3 / 10 |
| OPPORTUNITY | 8 | 0.128 | 0.134 | 0.132 | 0.130 | 0.038 | 3 / 5 |
| CATSv2 | 6 | 0.161 | 0.171 | 0.136 | 0.139 | 0.045 | 1 / 5 |
| LTDB | 5 | 0.295 | 0.409 | 0.360 | 0.478 | 0.210 | 1 / 4 |
| SWaT | 2 | 0.357 | 0.338 | 0.403 | 0.370 | 0.144 | 0 / 2 |
| CreditCard | 1 | 0.123 | 0.092 | 0.095 | 0.098 | 0.078 | 0 / 1 |
| PSM | 1 | 0.197 | 0.207 | 0.274 | 0.200 | 0.142 | 0 / 1 |
| Genesis | 1 | 0.030 | 0.211 | 0.020 | 0.014 | 0.085 | 1 / 0 |
| GECCO | 1 | 0.039 | 0.035 | 0.329 | 0.048 | 0.052 | 1 / 0 |
| Daphnet | 1 | 0.438 | 0.334 | 0.415 | 0.508 | 0.091 | 0 / 1 |
| all | 200 | 0.352 | 0.392 | 0.429 | 0.421 | 0.128 | 37 / 163 |

#### Not in the plan: mean VUS-PR of trained `pred` per source dataset, all 870 univariate series (the lists pooled)

| dataset | series | lejepa | lewm | straight | hepa | scjepa | scjepa better / worse than lejepa |
|---|---|---|---|---|---|---|---|
| YAHOO | 259 | 0.215 | 0.208 | 0.240 | 0.142 | 0.120 | 121 / 137 |
| UCR | 228 | 0.036 | 0.089 | 0.076 | 0.065 | 0.023 | 124 / 104 |
| WSD | 111 | 0.265 | 0.383 | 0.412 | 0.184 | 0.272 | 50 / 61 |
| SMD | 38 | 0.506 | 0.689 | 0.681 | 0.679 | 0.051 | 6 / 32 |
| Exathlon | 32 | 0.923 | 0.787 | 0.878 | 0.948 | 0.132 | 0 / 32 |
| OPPORTUNITY | 29 | 0.804 | 0.568 | 0.571 | 0.806 | 0.052 | 1 / 28 |
| NAB | 28 | 0.276 | 0.289 | 0.307 | 0.325 | 0.154 | 3 / 25 |
| Stock | 20 | 0.719 | 0.694 | 0.690 | 0.668 | 0.694 | 2 / 18 |
| SVDB | 20 | 0.128 | 0.299 | 0.233 | 0.410 | 0.200 | 14 / 6 |
| SMAP | 19 | 0.448 | 0.488 | 0.485 | 0.560 | 0.239 | 7 / 12 |
| IOPS | 17 | 0.375 | 0.328 | 0.371 | 0.257 | 0.107 | 4 / 13 |
| TODS | 15 | 0.487 | 0.488 | 0.500 | 0.492 | 0.451 | 7 / 8 |
| NEK | 9 | 0.947 | 0.932 | 0.938 | 0.885 | 0.275 | 0 / 9 |
| MSL | 9 | 0.388 | 0.410 | 0.415 | 0.360 | 0.197 | 2 / 7 |
| MGAB | 9 | 0.008 | 0.182 | 0.072 | 0.035 | 0.094 | 9 / 0 |
| LTDB | 9 | 0.316 | 0.413 | 0.377 | 0.443 | 0.266 | 3 / 6 |
| MITDB | 8 | 0.101 | 0.168 | 0.155 | 0.196 | 0.104 | 4 / 4 |
| SED | 3 | 0.048 | 0.067 | 0.118 | 0.048 | 0.186 | 3 / 0 |
| TAO | 3 | 0.911 | 0.889 | 0.886 | 0.876 | 0.864 | 0 / 3 |
| CATSv2 | 1 | 0.259 | 0.339 | 0.389 | 0.392 | 0.131 | 0 / 1 |
| SWaT | 1 | 0.428 | 0.413 | 0.430 | 0.367 | 0.098 | 0 / 1 |
| Daphnet | 1 | 0.462 | 0.424 | 0.450 | 0.488 | 0.095 | 0 / 1 |
| Power | 1 | 0.074 | 0.077 | 0.088 | 0.125 | 0.120 | 1 / 0 |
| all | 870 | 0.267 | 0.296 | 0.308 | 0.260 | 0.140 | 361 / 508 |

#### Small benchmarks (protocol of Section 7.8, 5 seeds): mean ROC-AUC, trained / zero-epoch control

| version | synthetic `pred` | synthetic `embed` | synthetic `combined` | SKAB `pred` | SKAB `embed` | SKAB `combined` | NAB `pred` | NAB `embed` | NAB `combined` |
|---|---|---|---|---|---|---|---|---|---|
| lejepa (default) | 0.941 / 0.798 | 0.928 / 0.932 | 0.936 / 0.877 | 0.825 / 0.831 | 0.825 / 0.828 | 0.825 / 0.829 | 0.628 / 0.671 | 0.627 / 0.619 | 0.629 / 0.658 |
| scjepa | 0.916 / 0.556 | 0.942 / 0.945 | 0.945 / 0.899 | 0.482 / 0.472 | 0.475 / 0.501 | 0.482 / 0.497 | 0.576 / 0.489 | 0.530 / 0.527 | 0.545 / 0.523 |
| scjepa without normalisation (ablation, trained) | 0.923 | 0.950 | 0.943 | 0.782 | 0.663 | 0.630 | 0.646 | 0.659 | 0.663 |

#### Small benchmarks: the six tests of the addendum, A = `scjepa`, trained, `block[pred]` (Holm over the six)

|  | B | group | series | ROC-AUC A / B | A − B | A / B better | p | p (Holm) |
|---|---|---|---|---|---|---|---|---|
| 1 | its untrained control | synthetic | 15 | 0.916 / 0.556 | +0.360 | 15 / 0 | <0.001 | <0.001 |
| 2 | its untrained control | SKAB | 16 | 0.482 / 0.472 | +0.010 | 9 / 7 | 0.464 | 0.469 |
| 3 | its untrained control | NAB | 7 | 0.576 / 0.489 | +0.087 | 5 / 2 | 0.219 | 0.469 |
| 4 | lejepa, trained | synthetic | 15 | 0.916 / 0.941 | -0.025 | 1 / 14 | <0.001 | <0.001 |
| 5 | lejepa, trained | SKAB | 16 | 0.482 / 0.825 | -0.343 | 0 / 16 | <0.001 | <0.001 |
| 6 | lejepa, trained | NAB | 7 | 0.576 / 0.628 | -0.052 | 2 / 5 | 0.156 | 0.469 |
