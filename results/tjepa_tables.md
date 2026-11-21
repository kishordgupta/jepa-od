Reference check: 144 JEPAOD rows of this run against results/tabular_results.csv: 106 identical, 125 within 0.001, largest absolute ROC-AUC difference 0.021. Datasets with a difference of 0.001 or more: 22_magic.gamma (0.021), 33_skin (0.019), 31_satimage-2 (0.003).

#### T-JEPA family, 16 ADBench datasets, mean of 3 seeds: mean ROC-AUC / AP per score

| variant | pred | embed | combined | emb+KNN | combined ROC-AUC by seed | fit time (s) |
|---|---|---|---|---|---|---|
| JEPAOD | 0.749 / 0.422 | 0.750 / 0.432 | 0.760 / 0.438 | 0.675 / 0.321 | 0.769 / 0.735 / 0.776 | 28.1 |
| TJEPAOD | 0.762 / 0.390 | 0.763 / 0.410 | 0.782 / 0.412 | 0.681 / 0.369 | 0.811 / 0.759 / 0.778 | 68.9 |
| TJEPAOD-nonorm | 0.765 / 0.381 | 0.761 / 0.452 | 0.777 / 0.444 | 0.761 / 0.386 | 0.782 / 0.749 / 0.801 | 68.4 |
| TJEPAOD-noreg | 0.766 / 0.393 | 0.770 / 0.425 | 0.782 / 0.410 | 0.698 / 0.375 | 0.803 / 0.737 / 0.806 | 65.9 |
| TJEPAOD-repoopt | 0.725 / 0.411 | 0.770 / 0.429 | 0.752 / 0.428 | 0.685 / 0.352 | 0.755 / 0.733 / 0.767 | 143.8 |
| JEPAOD-untrained | 0.696 / 0.406 | 0.746 / 0.414 | 0.750 / 0.435 | 0.757 / 0.397 | 0.745 / 0.743 / 0.761 | 2.4 |
| TJEPAOD-untrained | 0.466 / 0.190 | 0.770 / 0.435 | 0.641 / 0.324 | 0.667 / 0.332 | 0.653 / 0.615 / 0.654 | 3.8 |

#### All detectors ranked by mean ROC-AUC (20 columns; PCA has 15 datasets)

| detector | mean ROC-AUC | mean AP | avg rank | wins |
|---|---|---|---|---|
| KNN | 0.786 | 0.449 | 6.97 | 3 |
| TJEPAOD[combined] | 0.782 | 0.412 | 9.38 | 1 |
| TJEPAOD-noreg[combined] | 0.782 | 0.410 | 8.69 | 0 |
| TJEPAOD-nonorm[combined] | 0.777 | 0.444 | 8.22 | 0 |
| TJEPAOD-untrained[embed] | 0.770 | 0.435 | 8.75 | 0 |
| IForest | 0.767 | 0.477 | 7.16 | 2 |
| TJEPAOD[embed] | 0.763 | 0.410 | 9.25 | 1 |
| TJEPAOD-nonorm[embed] | 0.761 | 0.452 | 9.47 | 0 |
| JEPAOD[combined] | 0.760 | 0.438 | 10.41 | 0 |
| JEPAOD-untrained[combined] | 0.750 | 0.435 | 11.47 | 0 |
| AutoEncoder | 0.746 | 0.375 | 12.06 | 0 |
| DeepSVDD | 0.737 | 0.358 | 11.06 | 1 |
| DIF | 0.733 | 0.383 | 10.50 | 1 |
| PCA | 0.731 | 0.442 | 12.60 | 1 |
| LUNAR | 0.729 | 0.357 | 9.44 | 3 |
| HBOS | 0.716 | 0.400 | 12.84 | 0 |
| COPOD | 0.710 | 0.377 | 12.44 | 0 |
| OCSVM | 0.701 | 0.340 | 12.81 | 0 |
| LOF | 0.701 | 0.339 | 11.62 | 2 |
| ECOD | 0.697 | 0.380 | 14.41 | 0 |

#### Per-dataset ROC-AUC (mean of seeds; best of the row in bold)

| dataset | JEPAOD[combined] | TJEPAOD[combined] | TJEPAOD[embed] | TJEPAOD-nonorm[combined] | TJEPAOD-noreg[combined] | TJEPAOD-untrained[embed] | JEPAOD-untrained[combined] | AutoEncoder | IForest | KNN |
|---|---|---|---|---|---|---|---|---|---|---|
| fault | 0.590 | 0.451 | 0.442 | 0.564 | 0.449 | 0.465 | 0.443 | 0.651 | 0.554 | **0.719** |
| InternetAds | 0.586 | 0.639 | 0.637 | 0.595 | 0.631 | 0.641 | 0.592 | 0.543 | 0.667 | **0.687** |
| Ionosphere | 0.803 | 0.769 | 0.802 | 0.772 | 0.771 | 0.799 | 0.813 | 0.917 | 0.823 | **0.919** |
| magic.gamma | 0.684 | 0.685 | 0.584 | 0.748 | 0.761 | 0.713 | 0.701 | 0.732 | 0.725 | **0.811** |
| musk | 1.000 | 0.874 | 0.786 | 0.999 | 0.868 | 0.991 | 1.000 | 0.923 | **1.000** | 0.534 |
| Pima | 0.658 | 0.653 | 0.667 | 0.669 | 0.662 | **0.671** | 0.645 | 0.638 | 0.659 | 0.629 |
| satimage-2 | **0.993** | 0.976 | 0.969 | 0.979 | 0.979 | 0.971 | 0.976 | 0.950 | 0.989 | 0.948 |
| skin | 0.748 | **0.858** | 0.782 | 0.768 | 0.771 | 0.785 | 0.610 | 0.636 | 0.664 | 0.717 |
| speech | 0.491 | 0.505 | 0.505 | 0.496 | 0.491 | 0.505 | 0.485 | 0.490 | **0.515** | 0.497 |
| thyroid | 0.887 | 0.930 | 0.960 | 0.957 | 0.937 | 0.971 | 0.937 | 0.943 | **0.981** | 0.956 |
| vowels | 0.791 | 0.883 | 0.852 | 0.826 | 0.858 | 0.822 | 0.785 | 0.931 | 0.752 | **0.975** |
| Waveform | 0.634 | 0.870 | **0.900** | 0.828 | 0.858 | 0.689 | 0.677 | 0.641 | 0.707 | 0.757 |
| WBC | **0.995** | 0.987 | 0.982 | **0.995** | 0.984 | 0.981 | 0.994 | 0.978 | **0.995** | **0.995** |
| Wilt | 0.649 | 0.661 | 0.544 | 0.492 | 0.684 | 0.529 | 0.631 | 0.575 | 0.453 | **0.701** |
| wine | 0.759 | 0.825 | 0.844 | 0.823 | 0.852 | 0.839 | 0.764 | 0.589 | 0.847 | **0.995** |
| cardio | 0.897 | 0.953 | **0.955** | 0.927 | 0.953 | 0.948 | 0.944 | 0.797 | 0.935 | 0.736 |
| **mean** | 0.760 | 0.782 | 0.763 | 0.777 | 0.782 | 0.770 | 0.750 | 0.746 | 0.767 | 0.786 |

#### Paired comparisons over the 16 datasets (two-sided Wilcoxon signed-rank on the per-dataset means, not corrected for multiple comparisons)

|  | A | B | ROC-AUC A / B | datasets A / B better | p (ROC-AUC) | AP A / B | p (AP) |
|---|---|---|---|---|---|---|---|
| 1 | TJEPAOD[combined] | JEPAOD[combined] | 0.782 / 0.760 | 10 / 6 | 0.298 | 0.412 / 0.438 | 0.940 |
| 2 | TJEPAOD[embed] | JEPAOD[embed] | 0.763 / 0.750 | 10 / 6 | 0.562 | 0.410 / 0.432 | 0.860 |
| 3 | TJEPAOD[pred] | JEPAOD[pred] | 0.762 / 0.749 | 10 / 6 | 0.562 | 0.390 / 0.422 | 0.980 |
| 4 | TJEPAOD-nonorm[combined] | TJEPAOD[combined] | 0.777 / 0.782 | 8 / 8 | 0.860 | 0.444 / 0.412 | 0.821 |
| 5 | TJEPAOD-noreg[combined] | TJEPAOD[combined] | 0.782 / 0.782 | 7 / 9 | 0.821 | 0.410 / 0.412 | 0.632 |
| 6 | TJEPAOD-repoopt[combined] | TJEPAOD[combined] | 0.752 / 0.782 | 6 / 10 | 0.348 | 0.428 / 0.412 | 0.464 |
| 7 | TJEPAOD[embed] | TJEPAOD-untrained[embed] | 0.763 / 0.770 | 8 / 8 | 0.860 | 0.410 / 0.435 | 0.669 |
| 8 | JEPAOD[embed] | JEPAOD-untrained[embed] | 0.750 / 0.746 | 7 / 9 | 0.940 | 0.432 / 0.414 | 0.159 |
| 9 | TJEPAOD[pred] | TJEPAOD-untrained[pred] | 0.762 / 0.466 | 16 / 0 | <0.001 | 0.390 / 0.190 | 0.001 |
| 10 | JEPAOD[pred] | JEPAOD-untrained[pred] | 0.749 / 0.696 | 10 / 5 | 0.088 | 0.422 / 0.406 | 0.334 |
| 11 | JEPAOD[combined] | JEPAOD-untrained[combined] | 0.760 / 0.750 | 9 / 7 | 0.860 | 0.438 / 0.435 | 0.980 |
| 12 | TJEPAOD[combined] | TJEPAOD-untrained[embed] | 0.782 / 0.770 | 7 / 9 | 1.000 | 0.412 / 0.435 | 0.860 |
| 13 | JEPAOD[combined] | TJEPAOD-untrained[embed] | 0.760 / 0.770 | 6 / 10 | 0.252 | 0.438 / 0.435 | 0.860 |
| 14 | TJEPAOD[emb+KNN] | TJEPAOD[embed] | 0.681 / 0.763 | 6 / 10 | 0.211 | 0.369 / 0.410 | 0.495 |
| 15 | JEPAOD-untrained[emb+KNN] | JEPAOD[emb+KNN] | 0.757 / 0.675 | 13 / 2 | 0.005 | 0.397 / 0.321 | 0.003 |
| 16 | TJEPAOD[combined] | KNN | 0.782 / 0.786 | 7 / 9 | 0.706 | 0.412 / 0.449 | 0.433 |
| 17 | TJEPAOD[combined] | IForest | 0.782 / 0.767 | 5 / 11 | 0.821 | 0.412 / 0.477 | 0.348 |
| 18 | TJEPAOD-untrained[embed] | PCA | 0.790 / 0.731 | 12 / 3 | 0.008 | 0.442 / 0.442 | 0.303 |
