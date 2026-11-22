# Comment for https://github.com/yzhao062/pyod/issues/795

**Status: draft, not posted.** It is written for the author to post. Every number is from
`results/extended_tables.md` and `results/ts_control_tables.md` (`REPORT.md` section 7.6). The version
table is from `results/ts_variant_tables.md` (section 7.7). At the author's choice it ends with an offer to drop
the tabular detector and narrow the proposal to `TSJEPA` as a family with a selectable JEPA version.

## Body

**Update: three things I found after opening this issue. The second and third weaken the case I made above.**

**1. Prior work I did not cite.** T-JEPA (Thimonier et al., ICLR 2025, [arXiv:2410.05016](https://arxiv.org/abs/2410.05016)) already applies JEPA to tabular data, as representation learning for supervised tasks. It has no outlier score, but it is the closest prior architecture and I should have cited it. I re-implemented its training recipe and scored it the same way as the proposed detector. On the benchmark below the two cannot be separated (0.732 / 0.318 against 0.736 / 0.353).

**2. The ADBench ranking above does not hold on the full suite.** The table above covered 16 of the 47 classical ADBench datasets with 3 seeds. I have now run all 47 with 5 seeds: same protocol, default parameters, standalone prototype, mean ROC-AUC / AP.

| detector | ROC-AUC | AP |
|---|---|---|
| IForest | 0.763 | 0.384 |
| AutoEncoder | 0.748 | 0.333 |
| COPOD | 0.746 | 0.346 |
| ECOD | 0.742 | 0.359 |
| HBOS | 0.737 | 0.341 |
| **JEPA** | 0.736 | 0.353 |
| DIF | 0.734 | 0.317 |
| KNN | 0.732 | 0.360 |
| DeepSVDD | 0.729 | 0.317 |
| LUNAR | 0.696 | 0.289 |

JEPA is mid-table. IForest is better on 35 of the 47 datasets (Wilcoxon signed-rank, p = 0.010 after a Holm correction over 17 planned comparisons), and JEPA is not ahead of AutoEncoder. Mean fit time on one thread is 119 s for JEPA against 2.9 s for AutoEncoder and 0.1 s for IForest. The 16-dataset table is correct for those datasets, but the ranking it suggested did not carry over to the other 31. (OCSVM and LOF are below LUNAR. PCA raised an error on every seed of 5 datasets and is left out.)

**3. Training is not what produces the tabular result.** I added a control with zero training epochs: the same network at its random initialisation, scored the same way. On the 47 datasets it reaches 0.751 / 0.379 and is better than the trained detector on 32 of them. The difference is not significant after correction (p = 0.055 before). For `TSJEPA` the same control gives a mixed picture: training is what makes the default score work on `generate_ts_data` (0.941 against 0.798 untrained, 15 series, 5 seeds) and adds nothing measurable on SKAB (0.825 against 0.831) or NAB (0.628 against 0.671).

**Proposal.** Given these numbers I am happy to drop the tabular `JEPA` detector and narrow this proposal to `TSJEPA` only, as one detector with a `variant` parameter that selects which JEPA trains it: LeJEPA with SIGReg (the default, and the detector described above), I-JEPA with an EMA target, the recipe of TS-JEPA (Ennadir et al., [arXiv:2509.25449](https://arxiv.org/abs/2509.25449)), or the T-JEPA recipe on patch tokens. On the same benchmarks no version beats the default (default score, ROC-AUC, trained / untrained, 5 seeds):

| version | `generate_ts_data`, 15 series | SKAB valve1, 16 files | NAB realKnownCause, 7 series |
|---|---|---|---|
| LeJEPA (default) | 0.941 / 0.798 | 0.825 / 0.831 | 0.628 / 0.671 |
| I-JEPA | 0.885 / 0.519 | 0.431 / 0.422 | 0.614 / 0.521 |
| TS-JEPA | 0.931 / 0.485 | 0.442 / 0.561 | 0.585 / 0.483 |
| T-JEPA | 0.928 / 0.488 | 0.671 / 0.426 | 0.684 / 0.530 |

The supported strength is anomalies that break temporal structure, the `generate_ts_data` column. There is no advantage on SKAB or NAB, and the three EMA versions fail under SKAB's regime shift. The `variant` parameter is in the standalone prototype; the branch linked above does not have it yet. How to weigh this is your call.

Scripts, raw results and the full write-up are in sections 7.5 to 7.7 of the [report](https://github.com/kishordgupta/jepa-od/blob/main/REPORT.md).

AI disclosure: Claude Code ran these benchmarks and drafted this comment.
