Coverage: 5 seeds; series per group: synthetic 15, SKAB 16, NAB 7; every version has every series.

#### JEPA versions of `TSJEPAOD`: mean ROC-AUC, trained / zero-epoch control, 5 seeds (block masking unless stated; the default score is `pred`; in the last row of each group the AP column is that of `random[pred]`)

| group | version | pred | embed | combined | forecast[pred] | AP of block[pred] | fit time (s) |
|---|---|---|---|---|---|---|---|
| synthetic | lejepa | 0.941 / 0.798 | 0.928 / 0.932 | 0.936 / 0.877 | 0.946 / 0.786 | 0.560 / 0.345 | 14.9 |
| synthetic | ijepa | 0.885 / 0.519 | 0.904 / 0.891 | 0.901 / 0.879 | 0.878 / 0.509 | 0.431 / 0.050 | 3.6 |
| synthetic | tsjepa | 0.931 / 0.485 | 0.923 / 0.925 | 0.935 / 0.907 | 0.936 / 0.466 | 0.454 / 0.059 | 7.5 |
| synthetic | tjepa | 0.928 / 0.488 | 0.925 / 0.938 | 0.935 / 0.928 | 0.926 / 0.558 | 0.516 / 0.058 | 5.0 |
| synthetic | tsjepa, masking of its paper (random, 70 %) | 0.804 / 0.494 | 0.924 / 0.925 | 0.870 / 0.909 |  | 0.331 / 0.055 | 6.8 |
| SKAB | lejepa | 0.825 / 0.831 | 0.825 / 0.828 | 0.825 / 0.829 | 0.857 / 0.859 | 0.626 / 0.637 | 50.8 |
| SKAB | ijepa | 0.431 / 0.422 | 0.530 / 0.749 | 0.472 / 0.710 | 0.588 / 0.444 | 0.339 / 0.323 | 9.9 |
| SKAB | tsjepa | 0.442 / 0.561 | 0.689 / 0.736 | 0.646 / 0.738 | 0.477 / 0.489 | 0.349 / 0.403 | 32.7 |
| SKAB | tjepa | 0.671 / 0.426 | 0.659 / 0.617 | 0.677 / 0.566 | 0.557 / 0.484 | 0.485 / 0.316 | 12.5 |
| SKAB | tsjepa, masking of its paper (random, 70 %) | 0.538 / 0.576 | 0.698 / 0.736 | 0.659 / 0.738 |  | 0.413 / 0.409 | 33.7 |
| NAB | lejepa | 0.628 / 0.671 | 0.627 / 0.619 | 0.629 / 0.658 | 0.585 / 0.664 | 0.283 / 0.322 | 19.5 |
| NAB | ijepa | 0.614 / 0.521 | 0.589 / 0.552 | 0.609 / 0.536 | 0.545 / 0.546 | 0.239 / 0.159 | 5.8 |
| NAB | tsjepa | 0.585 / 0.483 | 0.711 / 0.720 | 0.677 / 0.721 | 0.578 / 0.523 | 0.179 / 0.165 | 11.8 |
| NAB | tjepa | 0.684 / 0.530 | 0.604 / 0.564 | 0.658 / 0.552 | 0.603 / 0.493 | 0.262 / 0.163 | 6.9 |
| NAB | tsjepa, masking of its paper (random, 70 %) | 0.556 / 0.496 | 0.714 / 0.720 | 0.676 / 0.722 |  | 0.170 / 0.171 | 12.0 |

#### Paired tests on the default score `block[pred]` (two-sided Wilcoxon signed-rank over series; Holm correction over the 18 tests, which were fixed in advance; CI: bootstrap over series)

|  | A | B | group | series | ROC-AUC A / B | A − B [95 % CI] | A / B better | p | p (Holm) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | ijepa, trained | ijepa, untrained | synthetic | 15 | 0.885 / 0.519 | +0.366 [+0.306, +0.426] | 15 / 0 | <0.001 | <0.001 |
| 2 | ijepa, trained | ijepa, untrained | SKAB | 16 | 0.431 / 0.422 | +0.009 [-0.099, +0.120] | 7 / 9 | 1.000 | 1.000 |
| 3 | ijepa, trained | ijepa, untrained | NAB | 7 | 0.614 / 0.521 | +0.093 [-0.028, +0.192] | 6 / 1 | 0.219 | 1.000 |
| 4 | tsjepa, trained | tsjepa, untrained | synthetic | 15 | 0.931 / 0.485 | +0.447 [+0.381, +0.512] | 15 / 0 | <0.001 | <0.001 |
| 5 | tsjepa, trained | tsjepa, untrained | SKAB | 16 | 0.442 / 0.561 | -0.120 [-0.194, -0.038] | 3 / 13 | 0.018 | 0.201 |
| 6 | tsjepa, trained | tsjepa, untrained | NAB | 7 | 0.585 / 0.483 | +0.102 [+0.004, +0.252] | 6 / 1 | 0.109 | 0.656 |
| 7 | tjepa, trained | tjepa, untrained | synthetic | 15 | 0.928 / 0.488 | +0.440 [+0.360, +0.513] | 15 / 0 | <0.001 | <0.001 |
| 8 | tjepa, trained | tjepa, untrained | SKAB | 16 | 0.671 / 0.426 | +0.245 [+0.180, +0.311] | 16 / 0 | <0.001 | <0.001 |
| 9 | tjepa, trained | tjepa, untrained | NAB | 7 | 0.684 / 0.530 | +0.154 [+0.050, +0.275] | 6 / 1 | 0.047 | 0.422 |
| 10 | ijepa, trained | lejepa, trained | synthetic | 15 | 0.885 / 0.941 | -0.056 [-0.112, -0.006] | 3 / 12 | 0.041 | 0.413 |
| 11 | ijepa, trained | lejepa, trained | SKAB | 16 | 0.431 / 0.825 | -0.394 [-0.473, -0.309] | 0 / 16 | <0.001 | <0.001 |
| 12 | ijepa, trained | lejepa, trained | NAB | 7 | 0.614 / 0.628 | -0.014 [-0.089, +0.068] | 3 / 4 | 0.812 | 1.000 |
| 13 | tsjepa, trained | lejepa, trained | synthetic | 15 | 0.931 / 0.941 | -0.010 [-0.018, -0.001] | 3 / 12 | 0.073 | 0.584 |
| 14 | tsjepa, trained | lejepa, trained | SKAB | 16 | 0.442 / 0.825 | -0.383 [-0.436, -0.331] | 0 / 16 | <0.001 | <0.001 |
| 15 | tsjepa, trained | lejepa, trained | NAB | 7 | 0.585 / 0.628 | -0.043 [-0.074, -0.011] | 2 / 5 | 0.078 | 0.584 |
| 16 | tjepa, trained | lejepa, trained | synthetic | 15 | 0.928 / 0.941 | -0.013 [-0.042, +0.011] | 8 / 7 | 0.599 | 1.000 |
| 17 | tjepa, trained | lejepa, trained | SKAB | 16 | 0.671 / 0.825 | -0.154 [-0.186, -0.123] | 0 / 16 | <0.001 | <0.001 |
| 18 | tjepa, trained | lejepa, trained | NAB | 7 | 0.684 / 0.628 | +0.056 [+0.003, +0.118] | 3 / 4 | 0.578 | 1.000 |
