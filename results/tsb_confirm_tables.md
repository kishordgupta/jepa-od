Coverage: M-Tuning 20 series, U-Tuning 48 series, U-Rest 472 series; 33 runs per list; every run has every series, and no run log has an error.

#### M-Tuning (20 series): mean VUS-PR of the three score types, trained / zero-epoch control (minutes: fitting and scoring the whole list once, trained, on a shared H100). Without a network (raw patches, `embed`): 0.374. Reference (PCA (window 1), re-run): 0.323.

| version | pred | embed | combined | minutes |
|---|---|---|---|---|
| lejepa | 0.363 / 0.353 | 0.373 / 0.375 | 0.368 / 0.368 | 6 |
| ijepa | 0.359 / 0.159 | 0.328 / 0.319 | 0.355 / 0.314 | 5 |
| tsjepa | 0.343 / 0.236 | 0.393 / 0.377 | 0.378 / 0.377 | 4 |
| tjepa | 0.357 / 0.171 | 0.330 / 0.339 | 0.348 / 0.329 | 5 |
| vjepa | 0.349 / 0.159 | 0.328 / 0.319 | 0.346 / 0.314 | 4 |
| vjepa21 | 0.345 / 0.182 | 0.320 / 0.319 | 0.336 / 0.319 | 5 |
| cjepa | 0.371 / 0.159 | 0.330 / 0.319 | 0.357 / 0.314 | 7 |
| sjepa | 0.376 / 0.164 | 0.327 / 0.319 | 0.353 / 0.318 | 5 |
| salt | 0.335 / 0.166 | 0.375 / 0.375 | 0.385 / 0.380 | 4 |
| vicreg | 0.390 / 0.353 | 0.387 / 0.375 | 0.389 / 0.368 | 6 |
| kerjepa | 0.363 / 0.353 | 0.372 / 0.375 | 0.368 / 0.368 | 5 |
| lpjepa | 0.378 / 0.353 | 0.379 / 0.375 | 0.378 / 0.368 | 5 |
| lambdajepa | 0.372 / 0.353 | 0.367 / 0.375 | 0.370 / 0.368 | 6 |
| lenepa | 0.365 / 0.368 | 0.417 / 0.403 | 0.418 / 0.399 | 3 |
| lewm | 0.441 / 0.391 | 0.386 / 0.388 | 0.414 / 0.396 | 4 |
| cfjepa | 0.348 / 0.248 | 0.528 / 0.513 | 0.525 / 0.509 | 23 |

#### U-Tuning (48 series): mean VUS-PR of the three score types, trained / zero-epoch control (minutes: fitting and scoring the whole list once, trained, on a shared H100). Without a network (raw patches, `embed`): 0.458. Reference (Sub-PCA, re-run): 0.326.

| version | pred | embed | combined | minutes |
|---|---|---|---|---|
| lejepa | 0.337 / 0.304 | 0.449 / 0.459 | 0.443 / 0.447 | 7 |
| ijepa | 0.305 / 0.127 | 0.287 / 0.282 | 0.381 / 0.282 | 5 |
| tsjepa | 0.329 / 0.134 | 0.497 / 0.489 | 0.493 / 0.482 | 5 |
| tjepa | 0.345 / 0.121 | 0.400 / 0.394 | 0.453 / 0.395 | 5 |
| vjepa | 0.295 / 0.127 | 0.287 / 0.282 | 0.379 / 0.282 | 5 |
| vjepa21 | 0.306 / 0.141 | 0.295 / 0.282 | 0.338 / 0.312 | 5 |
| cjepa | 0.301 / 0.127 | 0.288 / 0.282 | 0.382 / 0.282 | 7 |
| sjepa | 0.302 / 0.134 | 0.288 / 0.282 | 0.385 / 0.283 | 5 |
| salt | 0.332 / 0.078 | 0.458 / 0.459 | 0.468 / 0.458 | 5 |
| vicreg | 0.410 / 0.304 | 0.449 / 0.459 | 0.449 / 0.447 | 5 |
| kerjepa | 0.336 / 0.304 | 0.446 / 0.459 | 0.441 / 0.447 | 6 |
| lpjepa | 0.317 / 0.300 | 0.430 / 0.452 | 0.412 / 0.438 | 6 |
| lambdajepa | 0.367 / 0.304 | 0.402 / 0.459 | 0.405 / 0.447 | 6 |
| lenepa | 0.342 / 0.272 | 0.472 / 0.473 | 0.478 / 0.476 | 4 |
| lewm | 0.398 / 0.304 | 0.435 / 0.437 | 0.454 / 0.439 | 3 |
| cfjepa | 0.328 / 0.204 | 0.508 / 0.464 | 0.504 / 0.448 | 23 |

#### U-Rest (472 series): mean VUS-PR of the three score types, trained / zero-epoch control (minutes: fitting and scoring the whole list once, trained, on a shared H100). Without a network (raw patches, `embed`): 0.545. Reference (Sub-PCA, re-run): 0.146.

| version | pred | embed | combined | minutes |
|---|---|---|---|---|
| lejepa | 0.177 / 0.146 | 0.432 / 0.435 | 0.401 / 0.403 | 48 |
| ijepa | 0.144 / 0.048 | 0.404 / 0.396 | 0.430 / 0.404 | 38 |
| tsjepa | 0.171 / 0.043 | 0.469 / 0.466 | 0.457 / 0.464 | 39 |
| tjepa | 0.160 / 0.055 | 0.490 / 0.491 | 0.511 / 0.483 | 40 |
| vjepa | 0.135 / 0.048 | 0.403 / 0.396 | 0.425 / 0.404 | 37 |
| vjepa21 | 0.133 / 0.063 | 0.398 / 0.396 | 0.424 / 0.425 | 40 |
| cjepa | 0.147 / 0.048 | 0.402 / 0.396 | 0.433 / 0.404 | 59 |
| sjepa | 0.141 / 0.048 | 0.403 / 0.396 | 0.432 / 0.403 | 42 |
| salt | 0.146 / 0.040 | 0.435 / 0.435 | 0.436 / 0.432 | 35 |
| vicreg | 0.242 / 0.146 | 0.428 / 0.435 | 0.397 / 0.403 | 46 |
| kerjepa | 0.177 / 0.146 | 0.429 / 0.435 | 0.399 / 0.403 | 44 |
| lpjepa | 0.179 / 0.143 | 0.376 / 0.392 | 0.341 / 0.359 | 43 |
| lambdajepa | 0.216 / 0.146 | 0.395 / 0.435 | 0.367 / 0.403 | 49 |
| lenepa | 0.342 / 0.213 | 0.446 / 0.447 | 0.454 / 0.436 | 28 |
| lewm | 0.213 / 0.143 | 0.438 / 0.442 | 0.442 / 0.436 | 28 |
| cfjepa | 0.163 / 0.095 | 0.333 / 0.245 | 0.324 / 0.236 | 148 |

#### The 18 tests fixed in the plan (VUS-PR; two-sided Wilcoxon signed-rank over series; Holm correction over the 18; CI: bootstrap over series). Reading: "holds" when A − B is positive, as on the evaluation list, and the corrected p is below 0.05; "same direction" when only the sign agrees; "not confirmed" when A − B is negative.

|  | A | B | list | series | VUS-PR A / B | A − B [95 % CI] | A / B better | A − B on the evaluation list | p | p (Holm) | reading |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `cfjepa`, trained, `embed` | `lejepa`, trained, `pred` | M-Tuning | 20 | 0.528 / 0.363 | +0.165 [+0.049, +0.295] | 14 / 6 | +0.083 | 0.053 | 0.264 | same direction |
| 1 | `cfjepa`, trained, `embed` | `lejepa`, trained, `pred` | U-Tuning | 48 | 0.508 / 0.337 | +0.172 [+0.082, +0.265] | 35 / 12 | +0.152 | <0.001 | 0.005 | holds |
| 1 | `cfjepa`, trained, `embed` | `lejepa`, trained, `pred` | U-Rest | 472 | 0.333 / 0.177 | +0.156 [+0.127, +0.185] | 329 / 139 | +0.152 | <0.001 | <0.001 | holds |
| 2 | `cfjepa`, trained, `embed` | raw patches, `embed` | M-Tuning | 20 | 0.528 / 0.374 | +0.154 [+0.040, +0.281] | 14 / 6 | +0.089 | 0.044 | 0.264 | same direction |
| 2 | `cfjepa`, trained, `embed` | raw patches, `embed` | U-Tuning | 48 | 0.508 / 0.458 | +0.050 [-0.051, +0.145] | 29 / 18 | +0.064 | 0.169 | 0.338 | same direction |
| 2 | `cfjepa`, trained, `embed` | raw patches, `embed` | U-Rest | 472 | 0.333 / 0.545 | -0.212 [-0.250, -0.174] | 151 / 312 | +0.064 | <0.001 | <0.001 | not confirmed |
| 3 | `vicreg`, trained, `pred` | `lejepa`, trained, `pred` | M-Tuning | 20 | 0.390 / 0.363 | +0.027 [+0.006, +0.054] | 13 / 7 | +0.009 | 0.048 | 0.264 | same direction |
| 3 | `vicreg`, trained, `pred` | `lejepa`, trained, `pred` | U-Tuning | 48 | 0.410 / 0.337 | +0.073 [+0.028, +0.130] | 37 / 6 | +0.039 | <0.001 | <0.001 | holds |
| 3 | `vicreg`, trained, `pred` | `lejepa`, trained, `pred` | U-Rest | 472 | 0.242 / 0.177 | +0.065 [+0.050, +0.081] | 375 / 83 | +0.039 | <0.001 | <0.001 | holds |
| 4 | `lambdajepa`, trained, `pred` | `lejepa`, trained, `pred` | M-Tuning | 20 | 0.372 / 0.363 | +0.009 [+0.001, +0.019] | 14 / 6 | +0.004 | 0.048 | 0.264 | same direction |
| 4 | `lambdajepa`, trained, `pred` | `lejepa`, trained, `pred` | U-Tuning | 48 | 0.367 / 0.337 | +0.031 [+0.004, +0.076] | 29 / 15 | +0.014 | 0.005 | 0.044 | holds |
| 4 | `lambdajepa`, trained, `pred` | `lejepa`, trained, `pred` | U-Rest | 472 | 0.216 / 0.177 | +0.039 [+0.027, +0.053] | 314 / 135 | +0.014 | <0.001 | <0.001 | holds |
| 5 | `lewm`, trained, `pred` | `lejepa`, trained, `pred` | M-Tuning | 20 | 0.441 / 0.363 | +0.078 [+0.035, +0.126] | 16 / 4 | +0.036 | <0.001 | 0.005 | holds |
| 5 | `lewm`, trained, `pred` | `lejepa`, trained, `pred` | U-Tuning | 48 | 0.398 / 0.337 | +0.062 [+0.019, +0.111] | 32 / 15 | +0.016 | 0.007 | 0.050 | same direction |
| 5 | `lewm`, trained, `pred` | `lejepa`, trained, `pred` | U-Rest | 472 | 0.213 / 0.177 | +0.035 [+0.018, +0.054] | 280 / 188 | +0.016 | <0.001 | <0.001 | holds |
| 6 | `lejepa`, trained, `pred` | `lejepa`, untrained, `pred` | M-Tuning | 20 | 0.363 / 0.353 | +0.010 [-0.003, +0.031] | 11 / 9 | +0.004 | 0.869 | 0.869 | same direction |
| 6 | `lejepa`, trained, `pred` | `lejepa`, untrained, `pred` | U-Tuning | 48 | 0.337 / 0.304 | +0.033 [+0.010, +0.063] | 35 / 10 | +0.032 | <0.001 | <0.001 | holds |
| 6 | `lejepa`, trained, `pred` | `lejepa`, untrained, `pred` | U-Rest | 472 | 0.177 / 0.146 | +0.031 [+0.024, +0.038] | 367 / 92 | +0.032 | <0.001 | <0.001 | holds |

#### Further comparisons named in the plan, outside the 18 tests

| A | B | list | series | VUS-PR A / B | A − B [95 % CI] | A / B better | A − B on the evaluation list | p (not corrected) |
|---|---|---|---|---|---|---|---|---|
| `kerjepa`, trained, `pred` | `lejepa`, trained, `pred` | M-Tuning | 20 | 0.363 / 0.363 | -0.000 [-0.003, +0.003] | 11 / 9 | -0.001 | 1.000 |
| `kerjepa`, trained, `pred` | `lejepa`, trained, `pred` | U-Tuning | 48 | 0.336 / 0.337 | -0.000 [-0.013, +0.010] | 25 / 18 | +0.001 | 0.310 |
| `kerjepa`, trained, `pred` | `lejepa`, trained, `pred` | U-Rest | 472 | 0.177 / 0.177 | +0.000 [-0.003, +0.004] | 171 / 264 | +0.001 | <0.001 |
| `cfjepa`, trained, `embed` | `cfjepa`, untrained, `embed` | M-Tuning | 20 | 0.528 / 0.513 | +0.015 [-0.005, +0.045] | 8 / 12 | -0.002 | 0.784 |
| `cfjepa`, trained, `embed` | `cfjepa`, untrained, `embed` | U-Tuning | 48 | 0.508 / 0.464 | +0.044 [+0.005, +0.090] | 26 / 21 | +0.031 | 0.128 |
| `cfjepa`, trained, `embed` | `cfjepa`, untrained, `embed` | U-Rest | 472 | 0.333 / 0.245 | +0.088 [+0.068, +0.110] | 288 / 168 | +0.031 | <0.001 |
| `cfjepa`, trained, `embed` | `lejepa`, trained, `embed` | M-Tuning | 20 | 0.528 / 0.373 | +0.156 [+0.046, +0.277] | 14 / 6 | +0.081 | 0.053 |
| `cfjepa`, trained, `embed` | `lejepa`, trained, `embed` | U-Tuning | 48 | 0.508 / 0.449 | +0.059 [-0.029, +0.147] | 29 / 18 | +0.068 | 0.122 |
| `cfjepa`, trained, `embed` | `lejepa`, trained, `embed` | U-Rest | 472 | 0.333 / 0.432 | -0.099 [-0.131, -0.068] | 171 / 292 | +0.068 | <0.001 |
| `tsjepa`, trained, `embed` | `lejepa`, trained, `pred` | M-Tuning | 20 | 0.393 / 0.363 | +0.030 [-0.031, +0.099] | 9 / 11 | -0.009 | 0.729 |
| `tsjepa`, trained, `embed` | `lejepa`, trained, `pred` | U-Tuning | 48 | 0.497 / 0.337 | +0.160 [+0.068, +0.254] | 31 / 16 | +0.152 | 0.002 |
| `tsjepa`, trained, `embed` | `lejepa`, trained, `pred` | U-Rest | 472 | 0.469 / 0.177 | +0.291 [+0.260, +0.324] | 374 / 93 | +0.152 | <0.001 |

#### Mean VUS-PR of the sixteen trained versions, score `pred`, on the evaluation lists (section 7.9) and on the three lists of the confirmation. Rank correlation: Spearman, over the sixteen means.

| version | M-Eva | M-Tuning | U-Eva | U-Tuning | U-Rest |
|---|---|---|---|---|---|
| lejepa | 0.351 | 0.363 | 0.378 | 0.337 | 0.177 |
| ijepa | 0.288 | 0.359 | 0.315 | 0.305 | 0.144 |
| tsjepa | 0.333 | 0.343 | 0.354 | 0.329 | 0.171 |
| tjepa | 0.306 | 0.357 | 0.369 | 0.345 | 0.160 |
| vjepa | 0.279 | 0.349 | 0.305 | 0.295 | 0.135 |
| vjepa21 | 0.292 | 0.345 | 0.295 | 0.306 | 0.133 |
| cjepa | 0.294 | 0.371 | 0.317 | 0.301 | 0.147 |
| sjepa | 0.293 | 0.376 | 0.313 | 0.302 | 0.141 |
| salt | 0.308 | 0.335 | 0.337 | 0.332 | 0.146 |
| vicreg | 0.360 | 0.390 | 0.417 | 0.410 | 0.242 |
| kerjepa | 0.350 | 0.363 | 0.379 | 0.336 | 0.177 |
| lpjepa | 0.353 | 0.378 | 0.367 | 0.317 | 0.179 |
| lambdajepa | 0.355 | 0.372 | 0.392 | 0.367 | 0.216 |
| lenepa | 0.346 | 0.365 | 0.356 | 0.342 | 0.342 |
| lewm | 0.386 | 0.441 | 0.394 | 0.398 | 0.213 |
| cfjepa | 0.255 | 0.348 | 0.335 | 0.328 | 0.163 |
| rank correlation with the evaluation list |  | 0.64 |  | 0.90 | 0.86 |
| three highest | lewm, vicreg, lambdajepa | lewm, vicreg, lpjepa | vicreg, lewm, lambdajepa | vicreg, lewm, lambdajepa | lenepa, vicreg, lambdajepa |

#### Mean VUS-PR of the sixteen trained versions, score `embed`, on the evaluation lists (section 7.9) and on the three lists of the confirmation. Rank correlation: Spearman, over the sixteen means.

| version | M-Eva | M-Tuning | U-Eva | U-Tuning | U-Rest |
|---|---|---|---|---|---|
| lejepa | 0.353 | 0.373 | 0.461 | 0.449 | 0.432 |
| ijepa | 0.288 | 0.328 | 0.299 | 0.287 | 0.404 |
| tsjepa | 0.342 | 0.393 | 0.530 | 0.497 | 0.469 |
| tjepa | 0.280 | 0.330 | 0.375 | 0.400 | 0.490 |
| vjepa | 0.279 | 0.328 | 0.298 | 0.287 | 0.403 |
| vjepa21 | 0.275 | 0.320 | 0.293 | 0.295 | 0.398 |
| cjepa | 0.286 | 0.330 | 0.300 | 0.288 | 0.402 |
| sjepa | 0.286 | 0.327 | 0.300 | 0.288 | 0.403 |
| salt | 0.353 | 0.375 | 0.467 | 0.458 | 0.435 |
| vicreg | 0.359 | 0.387 | 0.462 | 0.449 | 0.428 |
| kerjepa | 0.353 | 0.372 | 0.458 | 0.446 | 0.429 |
| lpjepa | 0.358 | 0.379 | 0.432 | 0.430 | 0.376 |
| lambdajepa | 0.352 | 0.367 | 0.425 | 0.402 | 0.395 |
| lenepa | 0.378 | 0.417 | 0.496 | 0.472 | 0.446 |
| lewm | 0.367 | 0.386 | 0.462 | 0.435 | 0.438 |
| cfjepa | 0.434 | 0.528 | 0.530 | 0.508 | 0.333 |
| rank correlation with the evaluation list |  | 0.88 |  | 0.95 | 0.39 |
| three highest | cfjepa, lenepa, lewm | cfjepa, lenepa, tsjepa | tsjepa, cfjepa, lenepa | cfjepa, tsjepa, lenepa | tjepa, tsjepa, lenepa |

#### The reference detectors: TSB-AD's code of today against TSB-AD's published per-series VUS-PR, on the evaluation lists

| detector | evaluation list | series | published mean | re-run mean | series within 0.01 | series more than 0.1 apart | largest difference | rank correlation |
|---|---|---|---|---|---|---|---|---|
| PCA (window 1) | M-Eva | 180 | 0.310 | 0.307 | 150 | 0 | 0.073 | 0.996 |
| Sub-PCA | U-Eva | 350 | 0.423 | 0.417 | 317 | 17 | 0.545 | 0.984 |

#### Against the re-run reference detector of the set, on the evaluation lists and on the three lists of the confirmation (the same series, the same VUS-PR code on both sides)

| A | B: reference, re-run | list | series | VUS-PR A / B | A − B [95 % CI] | A / B better | p (not corrected) |
|---|---|---|---|---|---|---|---|
| `lejepa`, trained, `pred` | PCA (window 1) | M-Eva | 180 | 0.351 / 0.307 | +0.044 [+0.014, +0.075] | 129 / 51 | <0.001 |
| `lejepa`, trained, `embed` | PCA (window 1) | M-Eva | 180 | 0.353 / 0.307 | +0.046 [+0.015, +0.078] | 121 / 59 | <0.001 |
| `vicreg`, trained, `pred` | PCA (window 1) | M-Eva | 180 | 0.360 / 0.307 | +0.053 [+0.022, +0.085] | 127 / 53 | <0.001 |
| `cfjepa`, trained, `embed` | PCA (window 1) | M-Eva | 180 | 0.434 / 0.307 | +0.127 [+0.080, +0.175] | 127 / 53 | <0.001 |
| `lejepa`, trained, `pred` | PCA (window 1) | M-Tuning | 20 | 0.363 / 0.323 | +0.040 [-0.046, +0.121] | 13 / 7 | 0.231 |
| `lejepa`, trained, `embed` | PCA (window 1) | M-Tuning | 20 | 0.373 / 0.323 | +0.049 [-0.038, +0.133] | 12 / 8 | 0.231 |
| `vicreg`, trained, `pred` | PCA (window 1) | M-Tuning | 20 | 0.390 / 0.323 | +0.067 [-0.023, +0.155] | 13 / 7 | 0.143 |
| `cfjepa`, trained, `embed` | PCA (window 1) | M-Tuning | 20 | 0.528 / 0.323 | +0.205 [+0.045, +0.377] | 15 / 5 | 0.027 |
| `lejepa`, trained, `pred` | Sub-PCA | U-Eva | 350 | 0.378 / 0.417 | -0.039 [-0.061, -0.016] | 200 / 149 | 0.920 |
| `lejepa`, trained, `embed` | Sub-PCA | U-Eva | 350 | 0.461 / 0.417 | +0.044 [+0.008, +0.081] | 216 / 133 | 0.003 |
| `vicreg`, trained, `pred` | Sub-PCA | U-Eva | 350 | 0.417 / 0.417 | -0.000 [-0.028, +0.028] | 216 / 133 | 0.155 |
| `cfjepa`, trained, `embed` | Sub-PCA | U-Eva | 350 | 0.530 / 0.417 | +0.113 [+0.083, +0.143] | 267 / 83 | <0.001 |
| `lejepa`, trained, `pred` | Sub-PCA | U-Tuning | 48 | 0.337 / 0.326 | +0.011 [-0.047, +0.073] | 30 / 18 | 0.226 |
| `lejepa`, trained, `embed` | Sub-PCA | U-Tuning | 48 | 0.449 / 0.326 | +0.123 [+0.031, +0.223] | 30 / 18 | 0.016 |
| `vicreg`, trained, `pred` | Sub-PCA | U-Tuning | 48 | 0.410 / 0.326 | +0.084 [+0.004, +0.170] | 32 / 16 | 0.028 |
| `cfjepa`, trained, `embed` | Sub-PCA | U-Tuning | 48 | 0.508 / 0.326 | +0.182 [+0.091, +0.280] | 35 / 13 | <0.001 |
| `lejepa`, trained, `pred` | Sub-PCA | U-Rest | 472 | 0.177 / 0.146 | +0.031 [+0.013, +0.050] | 307 / 165 | <0.001 |
| `lejepa`, trained, `embed` | Sub-PCA | U-Rest | 472 | 0.432 / 0.146 | +0.286 [+0.251, +0.321] | 374 / 98 | <0.001 |
| `vicreg`, trained, `pred` | Sub-PCA | U-Rest | 472 | 0.242 / 0.146 | +0.096 [+0.072, +0.121] | 340 / 132 | <0.001 |
| `cfjepa`, trained, `embed` | Sub-PCA | U-Rest | 472 | 0.333 / 0.146 | +0.187 [+0.160, +0.215] | 362 / 110 | <0.001 |

#### M-Tuning by source dataset: mean VUS-PR (all trained)

| dataset | series | default: `lejepa` `pred` | `lejepa` `embed` | `vicreg` `pred` | `cfjepa` `embed` | raw patches | PCA (window 1), re-run |
|---|---|---|---|---|---|---|---|
| SVDB | 3 | 0.231 | 0.253 | 0.334 | 0.707 | 0.244 | 0.108 |
| Exathlon | 2 | 0.742 | 0.742 | 0.742 | 0.743 | 0.739 | 0.954 |
| GHL | 2 | 0.019 | 0.019 | 0.019 | 0.023 | 0.018 | 0.015 |
| MITDB | 2 | 0.018 | 0.032 | 0.040 | 0.157 | 0.017 | 0.066 |
| MSL | 2 | 0.345 | 0.341 | 0.347 | 0.628 | 0.342 | 0.130 |
| SMAP | 2 | 0.153 | 0.151 | 0.152 | 0.548 | 0.168 | 0.048 |
| SMD | 2 | 0.798 | 0.799 | 0.790 | 0.619 | 0.781 | 0.490 |
| TAO | 2 | 0.981 | 0.980 | 0.981 | 0.962 | 0.979 | 1.000 |
| CATSv2 | 1 | 0.140 | 0.135 | 0.164 | 0.119 | 0.294 | 0.352 |
| LTDB | 1 | 0.272 | 0.384 | 0.454 | 0.930 | 0.337 | 0.233 |
| OPPORTUNITY | 1 | 0.044 | 0.047 | 0.045 | 0.032 | 0.032 | 0.153 |
| all series | 20 | 0.363 | 0.373 | 0.390 | 0.528 | 0.374 | 0.323 |
| mean of the dataset means | 20 | 0.340 | 0.353 | 0.370 | 0.497 | 0.359 | 0.323 |

#### U-Tuning by source dataset: mean VUS-PR (all trained)

| dataset | series | default: `lejepa` `pred` | `lejepa` `embed` | `vicreg` `pred` | `cfjepa` `embed` | raw patches | Sub-PCA, re-run |
|---|---|---|---|---|---|---|---|
| UCR | 6 | 0.015 | 0.018 | 0.024 | 0.231 | 0.057 | 0.109 |
| YAHOO | 5 | 0.290 | 0.594 | 0.295 | 0.223 | 0.766 | 0.076 |
| WSD | 5 | 0.221 | 0.515 | 0.505 | 0.445 | 0.520 | 0.209 |
| NAB | 5 | 0.195 | 0.208 | 0.196 | 0.474 | 0.216 | 0.215 |
| SMD | 5 | 0.379 | 0.638 | 0.613 | 0.872 | 0.669 | 0.196 |
| SMAP | 2 | 0.523 | 0.606 | 0.560 | 0.956 | 0.618 | 0.645 |
| TODS | 2 | 0.589 | 0.965 | 0.750 | 0.932 | 0.778 | 0.512 |
| Stock | 2 | 0.663 | 0.676 | 0.666 | 0.556 | 0.697 | 0.721 |
| SVDB | 2 | 0.154 | 0.150 | 0.197 | 0.542 | 0.115 | 0.389 |
| Exathlon | 2 | 0.876 | 0.882 | 0.974 | 0.998 | 0.883 | 0.836 |
| IOPS | 2 | 0.203 | 0.231 | 0.262 | 0.348 | 0.363 | 0.160 |
| OPPORTUNITY | 2 | 0.964 | 0.954 | 0.965 | 0.966 | 0.596 | 0.971 |
| MSL | 2 | 0.374 | 0.387 | 0.386 | 0.388 | 0.358 | 0.572 |
| NEK | 1 | 0.882 | 0.912 | 0.887 | 0.685 | 0.904 | 0.949 |
| TAO | 1 | 0.791 | 0.786 | 0.789 | 0.512 | 0.786 | 0.656 |
| MITDB | 1 | 0.007 | 0.007 | 0.010 | 0.034 | 0.004 | 0.037 |
| MGAB | 1 | 0.005 | 0.003 | 0.006 | 0.006 | 0.004 | 0.003 |
| LTDB | 1 | 0.215 | 0.225 | 0.216 | 0.214 | 0.230 | 0.220 |
| SED | 1 | 0.044 | 0.045 | 0.049 | 0.117 | 0.051 | 0.028 |
| all series | 48 | 0.337 | 0.449 | 0.410 | 0.508 | 0.458 | 0.326 |
| mean of the dataset means | 48 | 0.389 | 0.463 | 0.439 | 0.500 | 0.453 | 0.395 |

#### U-Rest by source dataset: mean VUS-PR (all trained)

| dataset | series | default: `lejepa` `pred` | `lejepa` `embed` | `vicreg` `pred` | `cfjepa` `embed` | raw patches | Sub-PCA, re-run |
|---|---|---|---|---|---|---|---|
| YAHOO | 224 | 0.215 | 0.598 | 0.235 | 0.301 | 0.812 | 0.145 |
| UCR | 152 | 0.036 | 0.085 | 0.068 | 0.239 | 0.107 | 0.110 |
| WSD | 86 | 0.267 | 0.582 | 0.516 | 0.557 | 0.603 | 0.145 |
| Stock | 10 | 0.709 | 0.717 | 0.711 | 0.575 | 0.728 | 0.732 |
| all series | 472 | 0.177 | 0.432 | 0.242 | 0.333 | 0.545 | 0.146 |
| mean of the dataset means | 472 | 0.307 | 0.495 | 0.382 | 0.418 | 0.563 | 0.283 |

#### Not in the plan: the four source datasets of U-Rest, on the evaluation list and on U-Rest (mean VUS-PR, all trained). The last column is the best of the 32 published detectors on the series of that dataset in the evaluation list; nothing is published for U-Rest.

| dataset | list | series | default: `lejepa` `pred` | `lejepa` `embed` | `vicreg` `pred` | `cfjepa` `embed` | raw patches | Sub-PCA, re-run | best published on the dataset |
|---|---|---|---|---|---|---|---|---|---|
| Stock | U-Eva | 8 | 0.745 | 0.754 | 0.746 | 0.642 | 0.765 | 0.763 | 0.998 (SR) |
| Stock | U-Rest | 10 | 0.709 | 0.717 | 0.711 | 0.575 | 0.728 | 0.732 |  |
| UCR | U-Eva | 70 | 0.037 | 0.101 | 0.082 | 0.258 | 0.168 | 0.124 | 0.381 (KMeansAD) |
| UCR | U-Rest | 152 | 0.036 | 0.085 | 0.068 | 0.239 | 0.107 | 0.110 |  |
| WSD | U-Eva | 20 | 0.266 | 0.596 | 0.545 | 0.558 | 0.642 | 0.123 | 0.500 (MOMENT (FT)) |
| WSD | U-Rest | 86 | 0.267 | 0.582 | 0.516 | 0.557 | 0.603 | 0.145 |  |
| YAHOO | U-Eva | 30 | 0.209 | 0.624 | 0.211 | 0.216 | 0.825 | 0.155 | 0.813 (TimesFM) |
| YAHOO | U-Rest | 224 | 0.215 | 0.598 | 0.235 | 0.301 | 0.812 | 0.145 |  |

#### Not in the plan: the three score types of the default version `lejepa` (trained), the control without a network, and the highest of the 96 cells (version, trained or not, score type) of each list. Differences: mean, series better / worse, two-sided Wilcoxon signed-rank p, not corrected.

| list | series | `pred` | `embed` | `combined` | raw patches | `embed` − `pred` | `combined` − `pred` | highest of the 96 cells |
|---|---|---|---|---|---|---|---|---|
| M-Eva | 180 | 0.351 | 0.353 | 0.352 | 0.345 | +0.002, 87 / 89 series, p 0.832 | +0.002, 87 / 89 series, p 0.955 | 0.436 (cfjepa, untrained, embed) |
| M-Tuning | 20 | 0.363 | 0.373 | 0.368 | 0.374 | +0.010, 11 / 9 series, p 0.648 | +0.005, 11 / 9 series, p 0.674 | 0.528 (cfjepa, trained, embed) |
| U-Eva | 350 | 0.378 | 0.461 | 0.455 | 0.465 | +0.083, 208 / 137 series, p <0.001 | +0.077, 211 / 134 series, p <0.001 | 0.530 (tsjepa, trained, embed) |
| U-Tuning | 48 | 0.337 | 0.449 | 0.443 | 0.458 | +0.113, 28 / 18 series, p 0.002 | +0.106, 27 / 18 series, p <0.001 | 0.508 (cfjepa, trained, embed) |
| U-Rest | 472 | 0.177 | 0.432 | 0.401 | 0.545 | +0.255, 360 / 103 series, p <0.001 | +0.224, 359 / 104 series, p <0.001 | 0.511 (tjepa, trained, combined) |
