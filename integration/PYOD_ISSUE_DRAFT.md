# Issue for yzhao062/pyod

**Status: posted on 2026-11-21 as https://github.com/yzhao062/pyod/issues/795.** PyOD's
`CONTRIBUTING.md` asks the contributor to open the issue, and the author posted it. The title and
body below are the text as posted. Two lines under "AI disclosure" are the author's own wording.

## What happens next

Per `CONTRIBUTING.md`:

1. The maintainer aims to answer within 14 days, which is 2026-12-05.
2. If the issue gets the `accepted` label and is assigned to the author, open the pull request
   against `development` within 14 days of the assignment and link the issue. A pull request opened
   earlier may be closed without review.
3. Until then nothing else is opened on PyOD. The CI-only pull request in the fork stays a draft and
   is not merged.

Every number in the body was recomputed from `results/tabular_results.csv`, `results/ts_results.csv`
and `results/ts_results_jepa_final.csv`, or taken from the bundled-data table in
`integration/PYOD_PR_CHECKLIST.md` and the CI logs of the run linked in the body.

## Found after posting (2026-11-21)

Three things in `REPORT.md` sections 7.5 and 7.6 are not in the posted text, and the second and third
weaken its case. They were added to the issue as a comment on 2026-11-21:
https://github.com/yzhao062/pyod/issues/795#issuecomment-7072235903
(text in `integration/PYOD_ISSUE_COMMENT_DRAFT.md`). The comment offers to narrow the proposal to `TSJEPA`.

1. **Prior work.** T-JEPA (Thimonier et al., ICLR 2025, arXiv 2410.05016) already applies JEPA to tabular
   data, as representation learning for supervised tasks. The issue does not cite it. This repository now
   has `TJEPAOD`, a re-implementation of its recipe scored like `JEPAOD`. On the full ADBench suite the two
   cannot be separated: ROC-AUC 0.732 / AP 0.318 against 0.736 / 0.353.
2. **Full ADBench suite.** The table in the issue covers 16 of the 47 classical ADBench datasets with
   3 seeds. On all 47 with 5 seeds the proposed tabular detector reaches 0.736 / 0.353: below AutoEncoder
   (0.748 / 0.333, not significant) and significantly below IForest (0.763 / 0.384; IForest is better on 35
   of 47 datasets, corrected p = 0.010). The table in the issue is correct for its 16 datasets. The ranking
   it suggests does not hold on the other 31.
3. **Untrained control.** With zero training epochs the tabular network scores 0.751 / 0.379 on the 47
   datasets and is better than the trained detector on 32 of them (not significant after correction). For
   `TSJEPA`, training produces the result on synthetic series (0.941 against 0.798 untrained) and adds
   nothing measurable on SKAB (0.825 against 0.831) or NAB (0.628 against 0.671).

Since then the prototype's `TSJEPAOD` has a `variant` option that selects the JEPA version (`REPORT.md`
sections 4.8 and 7.7). The issue and the port branch describe the detector without it.

## Title

New detector proposal: JEPA (masked latent prediction) for tabular data and time series

## Body

### Summary

I would like to contribute two experimental detectors based on Joint-Embedding Predictive Architectures (JEPA) [1, 2]:

- `pyod.models.jepa.JEPA` for tabular data, built on `BaseDeepLearningDetector`
- `pyod.models.ts_jepa.TSJEPA` for time series, following the `LSTMAD` pattern

Per `CONTRIBUTING.md` I am opening this issue before any pull request. I searched issues and pull requests for "JEPA" and "joint embedding" and found nothing related.

### Method

A JEPA hides part of the input and predicts the representation of the hidden part from the visible part. Normal samples have structure the predictor can learn (dependencies between features, temporal continuity), so their hidden parts are predictable in latent space. An outlier breaks that structure. The outlier score is the latent prediction error, averaged over several masks.

- **Tokens.** Features (grouped when there are more than `max_tokens`) or time patches are the tokens of a small transformer encoder with an I-JEPA style predictor [2].
- **Collapse prevention.** SIGReg from LeJEPA [3] by default. An EMA teacher as in I-JEPA is available with `target_mode='ema'`.
- **Scores.** `score='pred'` is the latent prediction error. `score='embed'` is a per-token Mahalanobis distance to the training embeddings. `JEPA` defaults to `'combined'` and `TSJEPA` to `'pred'`.

Relation to detectors already in PyOD: `AutoEncoder` and `VAE` measure error in input space, and `LSTMAD` measures a raw-space forecasting error. JEPA measures error in latent space. It needs no dependency beyond PyTorch, which PyOD's deep detectors already use.

### Inclusion criterion

I am not claiming the rule-of-thumb criterion. I-JEPA [2] is from 2023, but SIGReg [3] is from November 2025 and this adaptation to outlier detection is not published. **I commit to maintaining both detectors for at least two years.**

### Evidence

All detectors use default parameters and are fitted without labels. Values are ROC-AUC / average precision. The benchmark scripts, raw results and a full report are in https://github.com/kishordgupta/jepa-od.

**PyOD bundled data** (`notebooks/data/`, one stratified 60/40 split, seed 42, run with the code on the branch):

| data | JEPA | AutoEncoder | IForest | KNN |
|---|---|---|---|---|
| cardio | 0.919 / 0.483 | 0.739 / 0.296 | 0.945 / 0.631 | 0.726 / 0.337 |
| pima | 0.657 / 0.489 | 0.575 / 0.434 | 0.638 / 0.462 | 0.579 / 0.419 |
| shuttle | 0.887 / 0.485 | 0.991 / 0.843 | 0.997 / 0.980 | 0.740 / 0.205 |

**ADBench, 16 datasets, stratified 60/40 split, 3 seeds, mean over datasets.** These runs used the standalone prototype in that repository. The PyOD port produced identical scores in the five configurations I compared.

| detector | ROC-AUC | AP |
|---|---|---|
| KNN | 0.786 | 0.449 |
| IForest | 0.767 | 0.477 |
| **JEPA** | 0.760 | 0.438 |
| AutoEncoder | 0.746 | 0.375 |
| DeepSVDD | 0.737 | 0.358 |
| DIF | 0.733 | 0.383 |
| LUNAR | 0.729 | 0.357 |

Six further detectors (PCA, HBOS, COPOD, OCSVM, LOF, ECOD) had a mean ROC-AUC between 0.697 and 0.731. PCA's mean AP of 0.442, over the 15 datasets it ran on, is slightly above JEPA's.

**Time series** (point-wise scores, `TSJEPA` defaults), compared with `LSTMAD`, `AnomalyTransformer`, `MatrixProfile`, `SpectralResidual` and `TimeSeriesOD` around IForest, KNN and AutoEncoder:

| data | TSJEPA | LSTMAD | best other detector |
|---|---|---|---|
| `generate_ts_data`, 3 configurations x 2 seeds | 0.952 / 0.587 | 0.866 / 0.657 | SpectralResidual 0.819 / 0.582 |
| SKAB valve1, 16 files, fitted on the anomaly-free recording | 0.827 / 0.637 | 0.853 / 0.757 | TimeSeriesOD(KNN) 0.994 / 0.991 |
| NAB realKnownCause, 7 series, fitted and scored on the same series | 0.622 / 0.282 | 0.556 / 0.164 | TimeSeriesOD(IForest) 0.825 / 0.486 |

### Known limits

- **Rarely the single best detector.** On ADBench it is first or tied on 3 of 16 datasets (skin, musk, WBC) and trails the best detector by more than 0.10 ROC-AUC on 8. The largest gaps are on wine and vowels, where LOF and KNN win.
- **Sustained level shifts.** On NAB, windowed IForest and AutoEncoder are clearly better. A prediction error fires at the transition into a shifted segment, not throughout it.
- **Speed.** `JEPA` took 19.7 s per ADBench split on a laptop CPU against 2.3 s for `AutoEncoder`, and 220 s on shuttle. `TSJEPA` fitted faster than `LSTMAD` in all three time-series groups, for example 7 s against 121 s per NAB series.
- **Evaluation scope.** 16 ADBench datasets, no TSB-AD, CPU only.

### What is ready

Branch: https://github.com/kishordgupta/pyod/tree/feature/jepa-od, two commits on top of `development` at 9680a8b.

- `pyod/models/jepa.py`, `pyod/models/ts_jepa.py` and the shared `pyod/models/_jepa_utils.py`
- `pyod/test/test_jepa.py` (27 tests) and `pyod/test/test_ts_jepa.py` (17 tests), both added to the torch skip list in `pyod/test/conftest.py`
- `examples/jepa_example.py` and `examples/ts_jepa_example.py`
- API docs, bibliography entries, and table rows in `README.rst` and `docs/index.rst`

PyOD's `Testing` workflow passes on the branch: all 12 jobs, with 1714 passed and 18 skipped in each test job (Python 3.9 to 3.13 on Ubuntu and Windows, 3.9 on macOS). I ran it through a CI-only draft pull request inside my fork: https://github.com/kishordgupta/pyod/pull/1. Coveralls reports 98.36% patch coverage there.

Not included, because I understand these are yours to decide: `CHANGES.txt`, entries in `pyod/utils/knowledge/algorithms.json`, and the detector counts in the README and docs.

### AI disclosure

- Tools that contributed: Claude Code wrote the implementation, tests, docs and benchmark scripts.  i only provided the direction and some cases sample psedocode.
- Checks that actually ran: the CI run above. Locally (Python 3.11.18, torch 2.15.0, macOS, CPU): the 44 new tests, 106 tests when related PyOD suites are included, and both examples.
- Did a person read the full diff? - not thoroughly but i checked as humanly possible

### Questions

1. Is a JEPA detector in scope for PyOD as an experimental model?
2. If so, would you prefer one pull request or two? The commits are separable: `JEPA` with `_jepa_utils.py` first, then `TSJEPA`.
3. Is there further evidence you would want before a pull request, for example the full ADBench set or TSB-AD?

### References

[1] Y. LeCun. A Path Towards Autonomous Machine Intelligence. OpenReview, 2022.
[2] M. Assran et al. Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture. CVPR 2023.
[3] R. Balestriero and Y. LeCun. LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics. arXiv:2511.08544, 2025.
