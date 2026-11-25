# TSB-AD confirmation on series that no run has used: plan

Written on 2026-11-24, before any detector was run on these series. Only their file names and lengths had been
read. The results are in `results/tsb_confirm_tables.md` and REPORT.md section 7.10.

## Why

Section 7.9 compares sixteen versions and three score types on TSB-AD's evaluation lists (180 multivariate and 350
univariate series). Some of its conclusions were drawn after looking at those results, above all the `cfjepa`
encoder with the `embed` score, the highest of 96 cells per list. This plan fixes what is run and tested on series
that played no part in that.

## Series

| list | series | timestamps | source datasets |
|---|---|---|---|
| M-Tuning | 20 multivariate (`TSB-AD-M-Tuning.csv`) | 1.96 M | 11 |
| U-Tuning | 48 univariate (`TSB-AD-U-Tuning.csv`) | 2.26 M | 19 |
| U-Rest | 472 univariate: `TSB-AD-U-Eva-Full.csv` without the 350 of `TSB-AD-U-Eva.csv` | 13.34 M | 4 (YAHOO 224, UCR 152, WSD 86, Stock 10) |

The three lists and the two evaluation lists are disjoint, and together they are all 200 + 870 series of TSB-AD.
TSB-AD's authors chose the hyperparameters of their detectors on the two tuning lists. U-Rest is not in TSB-AD's
published tables at all; it is added here because 48 series give little power. Its sources are not mixed like
those of the evaluation list, so its results are also given per source dataset and as the mean of the dataset means.

## Runs

The code and protocol of section 7.9, unchanged: `benchmarks/bench_tsb.py` with the class defaults, fitted on the
training prefix, seed 42, and `benchmarks/eval_tsb.py` for VUS-PR. All sixteen versions, trained and untrained, and
the control without a network (raw patches), on each of the three lists: 99 runs. Nothing is tuned and no default
is changed.

## Tests fixed in advance

Six comparisons, each on each of the three lists: 18 two-sided Wilcoxon signed-rank tests over series on VUS-PR,
Holm-corrected over the 18. The expected direction is the one found on the evaluation lists.

| | A | B | A − B on the evaluation lists (M / U) |
|---|---|---|---|
| 1 | `cfjepa`, trained, `embed` | `lejepa`, trained, `pred` (the default detector) | +0.083 / +0.152 |
| 2 | `cfjepa`, trained, `embed` | raw patches, `embed` (no network) | +0.089 / +0.064 |
| 3 | `vicreg`, trained, `pred` | `lejepa`, trained, `pred` | +0.009 / +0.039 |
| 4 | `lambdajepa`, trained, `pred` | `lejepa`, trained, `pred` | +0.004 / +0.014 |
| 5 | `lewm`, trained, `pred` | `lejepa`, trained, `pred` | +0.036 / +0.016 |
| 6 | `lejepa`, trained, `pred` | `lejepa`, untrained, `pred` | +0.004 / +0.032 |

Reading of a test: "holds" when A − B has the expected sign and the Holm-corrected p is below 0.05; "same
direction" when only the sign agrees; "not confirmed" when the sign is the other way.

Power: with 20 series, only large and consistent differences can be significant on M-Tuning. Differences below
0.01 (comparisons 3, 4 and 6 on the multivariate list) cannot be expected to; there the sign is all that can be read.

## Reported without a test

- The tables of all sixteen versions and three score types per list, trained and untrained.
- The rank correlation between the sixteen version means on the evaluation list and on each of these lists.
- `kerjepa` against `lejepa` (level on the evaluation lists), `cfjepa` `embed` trained against untrained
  (−0.002 / +0.031), `cfjepa` `embed` against `lejepa` `embed` (+0.081 / +0.068), and `tsjepa` `embed` against
  the default detector (−0.009 / +0.152).

## Reference detectors

TSB-AD publishes per-series numbers for its evaluation lists only. As a reference on the other lists, two detectors
of TSB-AD's own code (the clone at commit 303dffb) are re-run here with the hyperparameters TSB-AD lists as optimal:

- univariate: Sub-PCA (`periodicity=1, n_components=None`), the detector with the best published mean;
- multivariate: PCA (`n_components=0.25`), with a window of one timestamp. Its published mean is 0.003 below the
  best one. The window is not in TSB-AD's list; one timestamp is the setting that reproduced a published value in
  section 7.9.

Both are first re-run on the evaluation list and compared with the published column, and that agreement is
reported whatever it is. The re-run is the reference on every list, the evaluation lists included, so that the
reference and our detector are measured with the same code. Compared with it, without correction: the default
detector, `lejepa` `embed`, `vicreg` `pred` and `cfjepa` `embed`.

The best published multivariate detectors (CNN, OmniAnomaly) are neural and are not re-run. So whether the
multivariate lead over everything published holds on other series is not tested here.

## Not done

No tuning, no further seeds, no change to the detector or its defaults.
