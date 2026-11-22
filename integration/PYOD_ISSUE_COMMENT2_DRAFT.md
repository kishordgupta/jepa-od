# Second comment for https://github.com/yzhao062/pyod/issues/795

**Status: posted on 2026-11-22 as https://github.com/yzhao062/pyod/issues/795#issuecomment-7077175222** at the author's request, from the author's account.
The body below is the text as posted. It was written after the twelve further versions were ported to the
branch `feature/ts-jepa` and names the three versions that match the default (`kerjepa`, `vicreg`,
`lambdajepa`). Every number is from `results/ts_versions_tables.md` and `results/ts_versions_timing.txt`
(`REPORT.md` section 7.8).

The draft had one more paragraph, saying that the branch also holds the nine other versions and that none
of them showed a benefit. The author had it removed before posting, so the comment does not say that the
branch has sixteen `variant` values.

## Body

**Update: the branch for the narrowed proposal is ready, and three more versions match the default.**

The `TSJEPA`-only branch is [`feature/ts-jepa`](https://github.com/kishordgupta/pyod/tree/feature/ts-jepa): two commits on `development` at 3.6.9, no tabular detector, with the `variant` parameter described above.

Since the last comment I re-implemented further published JEPA versions as `variant` values and ran them through the same benchmark (5 seeds, the paired tests fixed before the runs, Holm correction). Three of them match the default and are the ones I would add to the four above. Default score, ROC-AUC, trained models:

| `variant` | what replaces SIGReg | `generate_ts_data`, 15 series | SKAB valve1, 16 files | NAB realKnownCause, 7 series |
|---|---|---|---|---|
| `lejepa` (default) | | 0.941 | 0.825 | 0.628 |
| `kerjepa` | closed-form kernel discrepancy, KerJEPA ([arXiv:2512.19605](https://arxiv.org/abs/2512.19605)) | 0.950 | 0.829 | 0.633 |
| `vicreg` | VICReg variance and covariance terms, as in EB-JEPA ([arXiv:2602.03604](https://arxiv.org/abs/2602.03604)) | 0.935 | 0.829 | 0.576 |
| `lambdajepa` | Gaussian divergence, lambda-JEPA ([arXiv:2609.35288](https://arxiv.org/abs/2609.35288)) | 0.932 | 0.828 | 0.592 |

- `kerjepa` is ahead of the default in all three columns, by 0.005 to 0.008. Only the SKAB difference survives the correction (16 of 16 files, p = 0.002), and those files share their fitted models, so I read it as level with the default, not better.
- `vicreg` and `lambdajepa` are level on SKAB and within 0.01 on the synthetic series (the 0.010 that `lambdajepa` loses there is significant, p = 0.030). Both are lower on NAB, by 0.052 and 0.036; seven series can neither confirm nor rule that out.
- All three fit in about half the time of the default (2.6 to 2.7 s against 5.0 s on one series).
- No version beats the default, so the default stays `lejepa`. The untrained control is unchanged: 0.798 / 0.831 / 0.671, so training still earns the score on the synthetic series only.

What the three add is a finding about the design. They change only the regulariser and keep the default's shared encoder, whose outputs are the prediction targets without a normalisation step. All three hold 0.83 on SKAB, where every version that normalises its targets falls to between 0.21 and 0.67. That points to the un-normalised target, not to SIGReg, as what keeps the score under SKAB's regime shift.

Checks on the branch: its 55 detector tests pass, the whole `pyod/test` directory gives 1603 passed and 0 failed on my machine (six modules that need `pythresh`, `combo`, `statsmodels` or `xgboost` could not run there), and the port returns the same scores as the prototype that produced the numbers above. PyOD's CI has not run on it.

Scripts, raw results and the full write-up are in sections 4.9 and 7.8 of the [report](https://github.com/kishordgupta/jepa-od/blob/main/REPORT.md).

AI disclosure: Claude Code implemented and benchmarked these versions and drafted this comment.
