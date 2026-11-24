Coverage: TSB-AD-M-Eva 180 series, TSB-AD-U-Eva 350 series; every run has every series.

#### TSB-AD-M-Eva (180 series): mean VUS-PR of the three score types, trained / zero-epoch control (the default score is `pred`; minutes: fitting and scoring the whole set once, trained, on a shared H100)

| version | pred | embed | combined | AUC-PR of pred | AUC-ROC of pred | minutes |
|---|---|---|---|---|---|---|
| lejepa | 0.351 / 0.347 | 0.353 / 0.352 | 0.352 / 0.351 | 0.329 / 0.325 | 0.712 / 0.699 | 46 |
| ijepa | 0.288 / 0.135 | 0.288 / 0.287 | 0.301 / 0.286 | 0.250 / 0.090 | 0.719 / 0.519 | 38 |
| tsjepa | 0.333 / 0.174 | 0.342 / 0.342 | 0.347 / 0.344 | 0.303 / 0.134 | 0.724 / 0.474 | 36 |
| tjepa | 0.306 / 0.177 | 0.280 / 0.302 | 0.306 / 0.296 | 0.264 / 0.127 | 0.735 / 0.536 | 43 |
| vjepa | 0.279 / 0.135 | 0.279 / 0.287 | 0.293 / 0.286 | 0.241 / 0.090 | 0.718 / 0.519 | 40 |
| vjepa21 | 0.292 / 0.144 | 0.275 / 0.287 | 0.289 / 0.284 | 0.249 / 0.096 | 0.722 / 0.502 | 42 |
| cjepa | 0.294 / 0.135 | 0.286 / 0.287 | 0.303 / 0.286 | 0.257 / 0.090 | 0.720 / 0.519 | 58 |
| sjepa | 0.293 / 0.135 | 0.286 / 0.287 | 0.300 / 0.285 | 0.255 / 0.091 | 0.722 / 0.525 | 42 |
| salt | 0.308 / 0.137 | 0.353 / 0.352 | 0.359 / 0.353 | 0.267 / 0.090 | 0.722 / 0.542 | 39 |
| vicreg | 0.360 / 0.347 | 0.359 / 0.352 | 0.360 / 0.351 | 0.338 / 0.325 | 0.721 / 0.699 | 47 |
| kerjepa | 0.350 / 0.347 | 0.353 / 0.352 | 0.352 / 0.351 | 0.328 / 0.325 | 0.711 / 0.699 | 47 |
| lpjepa | 0.353 / 0.347 | 0.358 / 0.354 | 0.355 / 0.352 | 0.333 / 0.324 | 0.696 / 0.697 | 44 |
| lambdajepa | 0.355 / 0.347 | 0.352 / 0.352 | 0.354 / 0.351 | 0.333 / 0.325 | 0.718 / 0.699 | 47 |
| lenepa | 0.346 / 0.351 | 0.378 / 0.372 | 0.386 / 0.377 | 0.302 / 0.307 | 0.670 / 0.685 | 29 |
| lewm | 0.386 / 0.369 | 0.367 / 0.366 | 0.385 / 0.373 | 0.360 / 0.341 | 0.730 / 0.699 | 23 |
| cfjepa | 0.255 / 0.211 | 0.434 / 0.436 | 0.431 / 0.435 | 0.207 / 0.163 | 0.671 / 0.573 | 108 |

#### TSB-AD-U-Eva (350 series): mean VUS-PR of the three score types, trained / zero-epoch control (the default score is `pred`; minutes: fitting and scoring the whole set once, trained, on a shared H100)

| version | pred | embed | combined | AUC-PR of pred | AUC-ROC of pred | minutes |
|---|---|---|---|---|---|---|
| lejepa | 0.378 / 0.346 | 0.461 / 0.467 | 0.455 / 0.458 | 0.360 / 0.328 | 0.769 / 0.712 | 49 |
| ijepa | 0.315 / 0.141 | 0.299 / 0.288 | 0.347 / 0.294 | 0.293 / 0.102 | 0.778 / 0.569 | 41 |
| tsjepa | 0.354 / 0.128 | 0.530 / 0.522 | 0.511 / 0.514 | 0.324 / 0.097 | 0.753 / 0.402 | 43 |
| tjepa | 0.369 / 0.191 | 0.375 / 0.369 | 0.427 / 0.374 | 0.338 / 0.150 | 0.771 / 0.564 | 44 |
| vjepa | 0.305 / 0.141 | 0.298 / 0.288 | 0.341 / 0.294 | 0.282 / 0.102 | 0.771 / 0.569 | 40 |
| vjepa21 | 0.295 / 0.226 | 0.293 / 0.288 | 0.328 / 0.309 | 0.268 / 0.190 | 0.768 / 0.643 | 44 |
| cjepa | 0.317 / 0.141 | 0.300 / 0.288 | 0.349 / 0.294 | 0.295 / 0.102 | 0.780 / 0.569 | 63 |
| sjepa | 0.313 / 0.149 | 0.300 / 0.288 | 0.345 / 0.295 | 0.292 / 0.110 | 0.777 / 0.581 | 45 |
| salt | 0.337 / 0.088 | 0.467 / 0.467 | 0.476 / 0.462 | 0.315 / 0.051 | 0.775 / 0.350 | 39 |
| vicreg | 0.417 / 0.346 | 0.462 / 0.467 | 0.463 / 0.458 | 0.399 / 0.328 | 0.786 / 0.712 | 51 |
| kerjepa | 0.379 / 0.346 | 0.458 / 0.467 | 0.453 / 0.458 | 0.361 / 0.328 | 0.767 / 0.712 | 47 |
| lpjepa | 0.367 / 0.345 | 0.432 / 0.455 | 0.415 / 0.437 | 0.351 / 0.326 | 0.740 / 0.708 | 48 |
| lambdajepa | 0.392 / 0.346 | 0.425 / 0.467 | 0.430 / 0.458 | 0.376 / 0.328 | 0.768 / 0.712 | 51 |
| lenepa | 0.356 / 0.320 | 0.496 / 0.499 | 0.490 / 0.482 | 0.301 / 0.275 | 0.714 / 0.695 | 30 |
| lewm | 0.394 / 0.309 | 0.462 / 0.472 | 0.482 / 0.473 | 0.367 / 0.289 | 0.788 / 0.693 | 24 |
| cfjepa | 0.335 / 0.193 | 0.530 / 0.499 | 0.524 / 0.491 | 0.296 / 0.154 | 0.718 / 0.568 | 119 |

#### Next to the detectors TSB-AD published (same files, same VUS-PR code; their numbers are taken from the TSB-AD repository, not re-run). Place: where the mean VUS-PR falls among the published means (23 detectors on TSB-AD-M, 32 on TSB-AD-U). Mean rank: the detector is ranked among the published ones on each series (1 = best) and the ranks are averaged; in brackets its place by that mean rank. Last columns: difference to the best published detector of the set, series better / worse, and the p-value of a two-sided Wilcoxon signed-rank test (not corrected). Published top five by mean VUS-PR: TSB-AD-M: CNN 0.313, OmniAnomaly 0.312, PCA 0.310, LSTMAD 0.307, USAD 0.304; TSB-AD-U: Sub-PCA 0.423, KShapeAD 0.401, POLY 0.390, Series2Graph 0.388, MOMENT (FT) 0.386. Best published mean ranks, without ours: TSB-AD-M: CNN 7.2, KMeansAD 8.2, LSTMAD 8.3; TSB-AD-U: POLY 12.0, Series2Graph 13.0, CNN 13.1

| detector | TSB-AD-M: VUS-PR | TSB-AD-M: place | TSB-AD-M: mean rank | TSB-AD-M: vs best published | TSB-AD-U: VUS-PR | TSB-AD-U: place | TSB-AD-U: mean rank | TSB-AD-U: vs best published |
|---|---|---|---|---|---|---|---|---|
| default detector: lejepa, trained, `pred` | 0.351 | 1 of 24 | 7.7 (2.) | +0.038, 87 / 93 series, p 0.112 | 0.378 | 7 of 33 | 13.3 (2.) | -0.045, 191 / 159 series, p 0.245 |
| lejepa, untrained, `pred` | 0.347 | 1 of 24 | 8.5 (2.) | +0.034, 87 / 93 series, p 0.191 | 0.346 | 11 of 33 | 15.8 (14.) | -0.078, 158 / 192 series, p <0.001 |
| lejepa, trained, `embed` | 0.353 | 1 of 24 | 7.9 (2.) | +0.040, 92 / 88 series, p 0.061 | 0.461 | 1 of 33 | 10.2 (1.) | +0.038, 205 / 145 series, p 0.046 |
| lejepa, untrained, `embed` | 0.352 | 1 of 24 | 7.8 (2.) | +0.039, 91 / 89 series, p 0.059 | 0.467 | 1 of 33 | 10.1 (1.) | +0.043, 202 / 148 series, p 0.030 |
| lejepa, trained, `combined` | 0.352 | 1 of 24 | 7.7 (1.) | +0.039, 90 / 90 series, p 0.059 | 0.455 | 1 of 33 | 10.3 (1.) | +0.031, 207 / 143 series, p 0.035 |
| no network: raw patches, `embed` | 0.345 | 1 of 24 | 8.2 (2.) | +0.032, 84 / 96 series, p 0.307 | 0.465 | 1 of 33 | 10.8 (1.) | +0.042, 182 / 168 series, p 0.179 |
| ijepa, trained, `pred` | 0.288 | 8 of 24 | 9.6 (4.) | -0.025, 74 / 106 series, p 0.018 | 0.315 | 15 of 33 | 13.7 (4.) | -0.109, 179 / 171 series, p <0.001 |
| tsjepa, trained, `pred` | 0.333 | 1 of 24 | 7.7 (1.) | +0.020, 93 / 87 series, p 0.198 | 0.354 | 9 of 33 | 14.2 (7.) | -0.069, 165 / 185 series, p <0.001 |
| tjepa, trained, `pred` | 0.306 | 5 of 24 | 9.4 (4.) | -0.007, 83 / 97 series, p 0.457 | 0.369 | 7 of 33 | 13.2 (2.) | -0.055, 185 / 165 series, p 0.073 |
| vjepa, trained, `pred` | 0.279 | 8 of 24 | 9.7 (4.) | -0.034, 71 / 109 series, p 0.005 | 0.305 | 15 of 33 | 14.1 (7.) | -0.118, 177 / 173 series, p <0.001 |
| vjepa21, trained, `pred` | 0.292 | 8 of 24 | 9.8 (4.) | -0.021, 73 / 107 series, p 0.006 | 0.295 | 17 of 33 | 14.8 (9.) | -0.128, 163 / 187 series, p <0.001 |
| cjepa, trained, `pred` | 0.294 | 8 of 24 | 9.4 (4.) | -0.019, 82 / 98 series, p 0.075 | 0.317 | 15 of 33 | 13.8 (7.) | -0.107, 181 / 169 series, p <0.001 |
| sjepa, trained, `pred` | 0.293 | 8 of 24 | 9.4 (4.) | -0.020, 86 / 94 series, p 0.229 | 0.313 | 15 of 33 | 14.0 (7.) | -0.110, 181 / 169 series, p <0.001 |
| salt, trained, `pred` | 0.308 | 4 of 24 | 9.3 (4.) | -0.005, 77 / 103 series, p 0.065 | 0.337 | 13 of 33 | 13.4 (2.) | -0.086, 183 / 167 series, p 0.003 |
| vicreg, trained, `pred` | 0.360 | 1 of 24 | 7.4 (1.) | +0.047, 103 / 77 series, p 0.002 | 0.417 | 2 of 33 | 11.6 (1.) | -0.007, 207 / 143 series, p 0.578 |
| kerjepa, trained, `pred` | 0.350 | 1 of 24 | 7.8 (2.) | +0.037, 89 / 91 series, p 0.115 | 0.379 | 6 of 33 | 13.2 (2.) | -0.044, 195 / 155 series, p 0.261 |
| lpjepa, trained, `pred` | 0.353 | 1 of 24 | 7.6 (1.) | +0.040, 95 / 85 series, p 0.026 | 0.367 | 7 of 33 | 14.2 (7.) | -0.056, 179 / 171 series, p 0.005 |
| lambdajepa, trained, `pred` | 0.355 | 1 of 24 | 7.6 (1.) | +0.042, 93 / 87 series, p 0.028 | 0.392 | 3 of 33 | 13.0 (2.) | -0.031, 194 / 156 series, p 0.428 |
| lenepa, trained, `pred` | 0.346 | 1 of 24 | 9.1 (4.) | +0.033, 77 / 103 series, p 0.596 | 0.356 | 9 of 33 | 13.8 (7.) | -0.067, 159 / 191 series, p 0.003 |
| lewm, trained, `pred` | 0.386 | 1 of 24 | 6.7 (1.) | +0.073, 118 / 62 series, p <0.001 | 0.394 | 3 of 33 | 12.0 (1.) | -0.029, 180 / 170 series, p 0.117 |
| cfjepa, trained, `pred` | 0.255 | 12 of 24 | 10.9 (7.) | -0.058, 64 / 116 series, p 0.003 | 0.335 | 13 of 33 | 15.1 (12.) | -0.088, 151 / 199 series, p <0.001 |
| lejepa, trained, `pred`, constant channels at unit scale | 0.340 | 1 of 24 | 8.1 (2.) | +0.027, 82 / 98 series, p 0.345 |  |  |  |  |
| picked after the results: cfjepa, trained, `embed` | 0.434 | 1 of 24 | 7.0 (1.) | +0.121, 120 / 60 series, p <0.001 | 0.530 | 1 of 33 | 8.2 (1.) | +0.106, 257 / 93 series, p <0.001 |
| picked after the results: cfjepa, untrained, `embed` | 0.436 | 1 of 24 | 7.0 (1.) | +0.123, 118 / 62 series, p <0.001 | 0.499 | 1 of 33 | 9.2 (1.) | +0.075, 257 / 93 series, p <0.001 |

#### Paired tests on VUS-PR of the default score `pred` (two-sided Wilcoxon signed-rank over series; Holm correction over the 62 tests; CI: bootstrap over series)

|  | A | B | set | VUS-PR A / B | A − B [95 % CI] | A / B better | p | p (Holm) |
|---|---|---|---|---|---|---|---|---|
| 1 | lejepa, trained | lejepa, untrained | TSB-AD-M | 0.351 / 0.347 | +0.004 [+0.000, +0.008] | 116 / 62 | <0.001 | 0.003 |
| 2 | lejepa, trained | lejepa, untrained | TSB-AD-U | 0.378 / 0.346 | +0.032 [+0.022, +0.044] | 242 / 106 | <0.001 | <0.001 |
| 3 | ijepa, trained | ijepa, untrained | TSB-AD-M | 0.288 / 0.135 | +0.153 [+0.120, +0.186] | 144 / 36 | <0.001 | <0.001 |
| 4 | ijepa, trained | ijepa, untrained | TSB-AD-U | 0.315 / 0.141 | +0.173 [+0.149, +0.197] | 283 / 67 | <0.001 | <0.001 |
| 5 | tsjepa, trained | tsjepa, untrained | TSB-AD-M | 0.333 / 0.174 | +0.159 [+0.113, +0.205] | 138 / 42 | <0.001 | <0.001 |
| 6 | tsjepa, trained | tsjepa, untrained | TSB-AD-U | 0.354 / 0.128 | +0.227 [+0.193, +0.261] | 281 / 69 | <0.001 | <0.001 |
| 7 | tjepa, trained | tjepa, untrained | TSB-AD-M | 0.306 / 0.177 | +0.129 [+0.085, +0.174] | 136 / 44 | <0.001 | <0.001 |
| 8 | tjepa, trained | tjepa, untrained | TSB-AD-U | 0.369 / 0.191 | +0.178 [+0.151, +0.206] | 281 / 69 | <0.001 | <0.001 |
| 9 | vjepa, trained | vjepa, untrained | TSB-AD-M | 0.279 / 0.135 | +0.144 [+0.112, +0.177] | 144 / 36 | <0.001 | <0.001 |
| 10 | vjepa, trained | vjepa, untrained | TSB-AD-U | 0.305 / 0.141 | +0.164 [+0.140, +0.187] | 281 / 69 | <0.001 | <0.001 |
| 11 | vjepa21, trained | vjepa21, untrained | TSB-AD-M | 0.292 / 0.144 | +0.148 [+0.116, +0.184] | 143 / 37 | <0.001 | <0.001 |
| 12 | vjepa21, trained | vjepa21, untrained | TSB-AD-U | 0.295 / 0.226 | +0.070 [+0.040, +0.099] | 255 / 95 | <0.001 | <0.001 |
| 13 | cjepa, trained | cjepa, untrained | TSB-AD-M | 0.294 / 0.135 | +0.159 [+0.124, +0.195] | 146 / 34 | <0.001 | <0.001 |
| 14 | cjepa, trained | cjepa, untrained | TSB-AD-U | 0.317 / 0.141 | +0.175 [+0.151, +0.199] | 285 / 65 | <0.001 | <0.001 |
| 15 | sjepa, trained | sjepa, untrained | TSB-AD-M | 0.293 / 0.135 | +0.158 [+0.125, +0.194] | 152 / 28 | <0.001 | <0.001 |
| 16 | sjepa, trained | sjepa, untrained | TSB-AD-U | 0.313 / 0.149 | +0.165 [+0.142, +0.188] | 287 / 63 | <0.001 | <0.001 |
| 17 | salt, trained | salt, untrained | TSB-AD-M | 0.308 / 0.137 | +0.171 [+0.135, +0.209] | 148 / 32 | <0.001 | <0.001 |
| 18 | salt, trained | salt, untrained | TSB-AD-U | 0.337 / 0.088 | +0.249 [+0.218, +0.281] | 304 / 46 | <0.001 | <0.001 |
| 19 | vicreg, trained | vicreg, untrained | TSB-AD-M | 0.360 / 0.347 | +0.013 [+0.007, +0.020] | 128 / 46 | <0.001 | <0.001 |
| 20 | vicreg, trained | vicreg, untrained | TSB-AD-U | 0.417 / 0.346 | +0.071 [+0.052, +0.091] | 258 / 85 | <0.001 | <0.001 |
| 21 | kerjepa, trained | kerjepa, untrained | TSB-AD-M | 0.350 / 0.347 | +0.003 [+0.000, +0.007] | 104 / 70 | 0.006 | 0.104 |
| 22 | kerjepa, trained | kerjepa, untrained | TSB-AD-U | 0.379 / 0.346 | +0.034 [+0.022, +0.046] | 230 / 110 | <0.001 | <0.001 |
| 23 | lpjepa, trained | lpjepa, untrained | TSB-AD-M | 0.353 / 0.347 | +0.006 [+0.001, +0.012] | 111 / 66 | 0.009 | 0.150 |
| 24 | lpjepa, trained | lpjepa, untrained | TSB-AD-U | 0.367 / 0.345 | +0.022 [+0.008, +0.036] | 230 / 111 | <0.001 | <0.001 |
| 25 | lambdajepa, trained | lambdajepa, untrained | TSB-AD-M | 0.355 / 0.347 | +0.008 [+0.004, +0.013] | 126 / 47 | <0.001 | <0.001 |
| 26 | lambdajepa, trained | lambdajepa, untrained | TSB-AD-U | 0.392 / 0.346 | +0.047 [+0.032, +0.062] | 235 / 107 | <0.001 | <0.001 |
| 27 | lenepa, trained | lenepa, untrained | TSB-AD-M | 0.346 / 0.351 | -0.005 [-0.021, +0.011] | 106 / 73 | 0.043 | 0.638 |
| 28 | lenepa, trained | lenepa, untrained | TSB-AD-U | 0.356 / 0.320 | +0.036 [+0.013, +0.059] | 229 / 118 | <0.001 | <0.001 |
| 29 | lewm, trained | lewm, untrained | TSB-AD-M | 0.386 / 0.369 | +0.018 [+0.007, +0.030] | 112 / 66 | <0.001 | 0.015 |
| 30 | lewm, trained | lewm, untrained | TSB-AD-U | 0.394 / 0.309 | +0.085 [+0.064, +0.108] | 239 / 104 | <0.001 | <0.001 |
| 31 | cfjepa, trained | cfjepa, untrained | TSB-AD-M | 0.255 / 0.211 | +0.045 [+0.021, +0.068] | 109 / 71 | <0.001 | <0.001 |
| 32 | cfjepa, trained | cfjepa, untrained | TSB-AD-U | 0.335 / 0.193 | +0.143 [+0.117, +0.170] | 258 / 90 | <0.001 | <0.001 |
| 33 | ijepa, trained | lejepa, trained | TSB-AD-M | 0.288 / 0.351 | -0.063 [-0.094, -0.032] | 71 / 109 | <0.001 | 0.003 |
| 34 | ijepa, trained | lejepa, trained | TSB-AD-U | 0.315 / 0.378 | -0.064 [-0.095, -0.032] | 193 / 157 | 0.284 | 1.000 |
| 35 | tsjepa, trained | lejepa, trained | TSB-AD-M | 0.333 / 0.351 | -0.018 [-0.039, +0.002] | 89 / 91 | 0.145 | 1.000 |
| 36 | tsjepa, trained | lejepa, trained | TSB-AD-U | 0.354 / 0.378 | -0.024 [-0.043, -0.005] | 129 / 219 | <0.001 | <0.001 |
| 37 | tjepa, trained | lejepa, trained | TSB-AD-M | 0.306 / 0.351 | -0.045 [-0.085, -0.005] | 92 / 88 | 0.547 | 1.000 |
| 38 | tjepa, trained | lejepa, trained | TSB-AD-U | 0.369 / 0.378 | -0.009 [-0.035, +0.016] | 201 / 147 | 0.121 | 1.000 |
| 39 | vjepa, trained | lejepa, trained | TSB-AD-M | 0.279 / 0.351 | -0.072 [-0.102, -0.043] | 70 / 110 | <0.001 | <0.001 |
| 40 | vjepa, trained | lejepa, trained | TSB-AD-U | 0.305 / 0.378 | -0.073 [-0.104, -0.042] | 179 / 171 | 0.065 | 0.910 |
| 41 | vjepa21, trained | lejepa, trained | TSB-AD-M | 0.292 / 0.351 | -0.059 [-0.088, -0.029] | 69 / 111 | <0.001 | 0.002 |
| 42 | vjepa21, trained | lejepa, trained | TSB-AD-U | 0.295 / 0.378 | -0.083 [-0.114, -0.051] | 166 / 184 | 0.004 | 0.075 |
| 43 | cjepa, trained | lejepa, trained | TSB-AD-M | 0.294 / 0.351 | -0.057 [-0.088, -0.027] | 70 / 110 | <0.001 | 0.011 |
| 44 | cjepa, trained | lejepa, trained | TSB-AD-U | 0.317 / 0.378 | -0.062 [-0.093, -0.030] | 182 / 168 | 0.246 | 1.000 |
| 45 | sjepa, trained | lejepa, trained | TSB-AD-M | 0.293 / 0.351 | -0.057 [-0.088, -0.027] | 68 / 112 | <0.001 | 0.004 |
| 46 | sjepa, trained | lejepa, trained | TSB-AD-U | 0.313 / 0.378 | -0.065 [-0.096, -0.033] | 181 / 169 | 0.149 | 1.000 |
| 47 | salt, trained | lejepa, trained | TSB-AD-M | 0.308 / 0.351 | -0.043 [-0.070, -0.016] | 68 / 112 | <0.001 | 0.012 |
| 48 | salt, trained | lejepa, trained | TSB-AD-U | 0.337 / 0.378 | -0.041 [-0.069, -0.014] | 191 / 158 | 0.702 | 1.000 |
| 49 | vicreg, trained | lejepa, trained | TSB-AD-M | 0.360 / 0.351 | +0.009 [+0.005, +0.014] | 111 / 63 | <0.001 | 0.002 |
| 50 | vicreg, trained | lejepa, trained | TSB-AD-U | 0.417 / 0.378 | +0.039 [+0.024, +0.054] | 241 / 101 | <0.001 | <0.001 |
| 51 | kerjepa, trained | lejepa, trained | TSB-AD-M | 0.350 / 0.351 | -0.001 [-0.002, +0.001] | 80 / 93 | 0.020 | 0.313 |
| 52 | kerjepa, trained | lejepa, trained | TSB-AD-U | 0.379 / 0.378 | +0.001 [-0.001, +0.004] | 154 / 177 | 0.611 | 1.000 |
| 53 | lpjepa, trained | lejepa, trained | TSB-AD-M | 0.353 / 0.351 | +0.003 [-0.001, +0.006] | 106 / 71 | 0.108 | 1.000 |
| 54 | lpjepa, trained | lejepa, trained | TSB-AD-U | 0.367 / 0.378 | -0.011 [-0.021, -0.002] | 153 / 190 | 0.110 | 1.000 |
| 55 | lambdajepa, trained | lejepa, trained | TSB-AD-M | 0.355 / 0.351 | +0.004 [+0.001, +0.008] | 113 / 59 | <0.001 | <0.001 |
| 56 | lambdajepa, trained | lejepa, trained | TSB-AD-U | 0.392 / 0.378 | +0.014 [+0.005, +0.024] | 214 / 125 | <0.001 | <0.001 |
| 57 | lenepa, trained | lejepa, trained | TSB-AD-M | 0.346 / 0.351 | -0.005 [-0.031, +0.020] | 81 / 99 | 0.466 | 1.000 |
| 58 | lenepa, trained | lejepa, trained | TSB-AD-U | 0.356 / 0.378 | -0.022 [-0.056, +0.011] | 174 / 174 | 0.960 | 1.000 |
| 59 | lewm, trained | lejepa, trained | TSB-AD-M | 0.386 / 0.351 | +0.036 [+0.025, +0.047] | 135 / 45 | <0.001 | <0.001 |
| 60 | lewm, trained | lejepa, trained | TSB-AD-U | 0.394 / 0.378 | +0.016 [-0.005, +0.037] | 199 / 145 | 0.002 | 0.049 |
| 61 | cfjepa, trained | lejepa, trained | TSB-AD-M | 0.255 / 0.351 | -0.095 [-0.140, -0.051] | 67 / 113 | <0.001 | 0.013 |
| 62 | cfjepa, trained | lejepa, trained | TSB-AD-U | 0.335 / 0.378 | -0.043 [-0.071, -0.015] | 155 / 194 | 0.125 | 1.000 |

#### TSB-AD-M-Eva by source dataset: mean VUS-PR of the default version `lejepa` (the last column is the best of the published detectors on that dataset, a different detector from row to row)

| dataset | series | trained `pred` | untrained `pred` | trained `embed` | untrained `embed` | raw patches | published CNN | best published of the dataset |
|---|---|---|---|---|---|---|---|---|
| SVDB | 28 | 0.161 | 0.147 | 0.176 | 0.178 | 0.158 | 0.186 | 0.352 |
| Exathlon | 25 | 0.911 | 0.911 | 0.911 | 0.911 | 0.893 | 0.681 | 0.949 |
| SMAP | 25 | 0.297 | 0.294 | 0.301 | 0.300 | 0.267 | 0.191 | 0.380 |
| GHL | 23 | 0.015 | 0.016 | 0.015 | 0.015 | 0.022 | 0.024 | 0.065 |
| SMD | 20 | 0.397 | 0.404 | 0.401 | 0.404 | 0.364 | 0.353 | 0.364 |
| MSL | 14 | 0.407 | 0.404 | 0.404 | 0.403 | 0.388 | 0.352 | 0.435 |
| MITDB | 11 | 0.084 | 0.078 | 0.071 | 0.071 | 0.066 | 0.140 | 0.140 |
| TAO | 11 | 0.816 | 0.817 | 0.812 | 0.812 | 0.842 | 0.998 | 1.000 |
| OPPORTUNITY | 7 | 0.139 | 0.139 | 0.140 | 0.139 | 0.145 | 0.158 | 0.299 |
| CATSv2 | 5 | 0.166 | 0.110 | 0.168 | 0.142 | 0.252 | 0.082 | 0.132 |
| LTDB | 4 | 0.300 | 0.301 | 0.296 | 0.298 | 0.285 | 0.328 | 0.444 |
| SWaT | 2 | 0.357 | 0.376 | 0.368 | 0.360 | 0.383 | 0.405 | 0.575 |
| CreditCard | 1 | 0.123 | 0.123 | 0.122 | 0.122 | 0.132 | 0.022 | 0.103 |
| PSM | 1 | 0.197 | 0.181 | 0.196 | 0.200 | 0.255 | 0.219 | 0.280 |
| Genesis | 1 | 0.030 | 0.014 | 0.032 | 0.024 | 0.023 | 0.101 | 0.891 |
| GECCO | 1 | 0.039 | 0.037 | 0.039 | 0.039 | 0.539 | 0.033 | 0.202 |
| Daphnet | 1 | 0.438 | 0.436 | 0.429 | 0.424 | 0.370 | 0.208 | 0.340 |
| all series | 180 | 0.351 | 0.347 | 0.353 | 0.352 | 0.345 | 0.313 | 0.433 |
| mean of the dataset means | 180 | 0.287 | 0.281 | 0.287 | 0.285 | 0.317 | 0.264 | 0.409 |

#### TSB-AD-U-Eva by source dataset: mean VUS-PR of the default version `lejepa` (the last column is the best of the published detectors on that dataset, a different detector from row to row)

| dataset | series | trained `pred` | untrained `pred` | trained `embed` | untrained `embed` | raw patches | published Sub-PCA | best published of the dataset |
|---|---|---|---|---|---|---|---|---|
| UCR | 70 | 0.037 | 0.031 | 0.101 | 0.114 | 0.168 | 0.124 | 0.381 |
| SMD | 33 | 0.525 | 0.428 | 0.641 | 0.677 | 0.669 | 0.449 | 0.750 |
| YAHOO | 30 | 0.209 | 0.196 | 0.624 | 0.624 | 0.825 | 0.137 | 0.813 |
| Exathlon | 30 | 0.926 | 0.922 | 0.924 | 0.925 | 0.919 | 0.927 | 0.927 |
| OPPORTUNITY | 27 | 0.792 | 0.804 | 0.632 | 0.612 | 0.328 | 0.914 | 0.914 |
| NAB | 23 | 0.294 | 0.291 | 0.301 | 0.302 | 0.276 | 0.436 | 0.484 |
| WSD | 20 | 0.266 | 0.130 | 0.596 | 0.593 | 0.642 | 0.094 | 0.500 |
| SVDB | 18 | 0.125 | 0.111 | 0.109 | 0.109 | 0.088 | 0.522 | 0.821 |
| SMAP | 17 | 0.440 | 0.358 | 0.473 | 0.474 | 0.358 | 0.518 | 0.644 |
| IOPS | 15 | 0.398 | 0.245 | 0.509 | 0.532 | 0.583 | 0.227 | 0.377 |
| TODS | 13 | 0.471 | 0.465 | 0.740 | 0.730 | 0.816 | 0.539 | 0.761 |
| Stock | 8 | 0.745 | 0.743 | 0.754 | 0.754 | 0.765 | 0.840 | 0.998 |
| NEK | 8 | 0.955 | 0.954 | 0.965 | 0.966 | 0.937 | 0.905 | 0.905 |
| MGAB | 8 | 0.009 | 0.004 | 0.005 | 0.005 | 0.005 | 0.005 | 0.440 |
| LTDB | 8 | 0.329 | 0.333 | 0.303 | 0.306 | 0.269 | 0.556 | 0.832 |
| MSL | 7 | 0.392 | 0.334 | 0.480 | 0.479 | 0.295 | 0.507 | 0.554 |
| MITDB | 7 | 0.114 | 0.098 | 0.089 | 0.091 | 0.077 | 0.363 | 0.692 |
| SED | 2 | 0.050 | 0.048 | 0.061 | 0.059 | 0.062 | 0.031 | 0.892 |
| TAO | 2 | 0.971 | 0.971 | 0.971 | 0.971 | 0.971 | 0.930 | 1.000 |
| CATSv2 | 1 | 0.259 | 0.187 | 0.229 | 0.305 | 0.259 | 0.265 | 0.403 |
| SWaT | 1 | 0.428 | 0.405 | 0.595 | 0.626 | 0.378 | 0.393 | 0.682 |
| Daphnet | 1 | 0.462 | 0.328 | 0.486 | 0.489 | 0.472 | 0.422 | 0.520 |
| Power | 1 | 0.074 | 0.104 | 0.071 | 0.073 | 0.080 | 0.076 | 0.392 |
| all series | 350 | 0.378 | 0.346 | 0.461 | 0.467 | 0.465 | 0.423 | 0.659 |
| mean of the dataset means | 350 | 0.403 | 0.369 | 0.463 | 0.470 | 0.445 | 0.443 | 0.682 |

#### Seeds, and fitting on the whole series instead of the training prefix: mean VUS-PR

| run of lejepa, trained | TSB-AD-M: pred | TSB-AD-M: embed | TSB-AD-U: pred | TSB-AD-U: embed |
|---|---|---|---|---|
| seed 42 (the class default) | 0.351 | 0.353 | 0.378 | 0.461 |
| seed 0 | 0.352 | 0.354 | 0.377 | 0.461 |
| seed 1 | 0.351 | 0.353 | 0.376 | 0.459 |
| seed 42, fitted on the whole series | 0.332 | 0.337 | 0.392 | 0.450 |

#### Channels that are constant in the training prefix. The detector divides by the standard deviation of the training prefix plus 1e-8, so such a channel is multiplied by 1e8 when it moves later. "Unit scale" divides these channels by 1 instead; everything else is the same run. "With such a channel": the series where a channel is constant in the prefix and moves later. Mean VUS-PR, and the number of series where unit scale is better / worse.

| set | lejepa | series | n | `pred`: as is / unit scale | `pred`: better / worse | `embed`: as is / unit scale | `embed`: better / worse |
|---|---|---|---|---|---|---|---|
| TSB-AD-M | trained | with such a channel | 63 | 0.328 / 0.298 | 24 / 32 | 0.328 / 0.297 | 24 / 31 |
| TSB-AD-M | trained | without | 117 | 0.363 / 0.363 | 0 / 0 | 0.366 / 0.366 | 0 / 0 |
| TSB-AD-M | trained | all | 180 | 0.351 / 0.340 | 24 / 32 | 0.353 / 0.342 | 24 / 31 |
| TSB-AD-M | untrained | with such a channel | 63 | 0.328 / 0.296 | 28 / 32 | 0.328 / 0.297 | 25 / 31 |
| TSB-AD-M | untrained | without | 117 | 0.357 / 0.357 | 0 / 0 | 0.366 / 0.366 | 0 / 0 |
| TSB-AD-M | untrained | all | 180 | 0.347 / 0.335 | 28 / 32 | 0.352 / 0.342 | 25 / 31 |
| TSB-AD-U | untrained | with such a channel | 3 | 0.377 / 0.349 | 1 / 2 | 0.992 / 0.992 | 0 / 0 |
| TSB-AD-U | untrained | without | 347 | 0.345 / 0.345 | 0 / 0 | 0.462 / 0.462 | 0 / 0 |
| TSB-AD-U | untrained | all | 350 | 0.346 / 0.345 | 1 / 2 | 0.467 / 0.467 | 0 / 0 |
