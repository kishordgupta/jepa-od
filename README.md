# JEPA-OD: Joint-Embedding Predictive Architectures for Outlier Detection

Research prototype of a family of PyOD-compatible detectors that use a
Joint-Embedding Predictive Architecture (JEPA) as the anomaly model:

| class | data | PyOD base class | file |
|---|---|---|---|
| `jepa_od.JEPAOD` | tabular rows | `pyod.models.base_dl.BaseDeepLearningDetector` | `jepa_od/jepa.py` |
| `jepa_od.TJEPAOD` | tabular rows, T-JEPA training recipe | `jepa_od.JEPAOD` | `jepa_od/tjepa.py` |
| `jepa_od.TSJEPAOD` | uni/multivariate time series; `variant` selects the JEPA version | `pyod.models.base.BaseDetector` (PyOD time-series API) | `jepa_od/ts_jepa.py` |

See `REPORT.md` for the literature study, design rationale, benchmark results and the
integration guide for PyOD / DeepOD.

## Idea in one paragraph

A JEPA hides part of an input, encodes the visible part, and trains a small predictor to
output the *representation* of the hidden part (not its raw values). Normal samples have
dependencies the predictor can model: a row's masked features are predictable from the others,
a time window's masked patches are predictable from the surrounding patches. An outlier breaks
those dependencies, so its **latent prediction error** is high. Because the error is measured
in representation space, unpredictable noise in the raw values is not penalised (unlike
AutoEncoder reconstruction error). Representation collapse, the classic JEPA failure mode, is
prevented with **SIGReg** from LeJEPA (Balestriero & LeCun, 2025): random 1-D projections of the
embeddings are pushed to N(0, 1). This also makes the embedding distribution of normal data a
known isotropic Gaussian, which yields a second score, the **Mahalanobis distance of each token
embedding** to the training Gaussian, usable alone (`score='embed'`) or fused (`score='combined'`).

## The family

| | `JEPAOD` | `TJEPAOD` | `TSJEPAOD` |
|---|---|---|---|
| origin | this project, built on I-JEPA and LeJEPA | training recipe of T-JEPA (Thimonier et al., ICLR 2025, [arXiv 2410.05016](https://arxiv.org/abs/2410.05016)) | this project; twenty-seven selectable JEPA versions (below) |
| collapse prevention | SIGReg, one shared encoder | EMA target encoder and a [REG] token | SIGReg by default, an EMA target encoder in the other versions |
| context / targets | 70 % of the tokens visible, one mask per row | 20-40 % visible, four target subsets per row | block or causal patch masks |
| encoder output | not normalised | LayerNorm'ed | not normalised (`lejepa`, `ijepa`), normalised (`tsjepa`, `tjepa`) |
| scores | `pred`, `embed`, `combined` | the same, inherited | the same, per timestamp |

`TJEPAOD` is a re-implementation written from the T-JEPA paper and the defaults of its
[released code](https://github.com/jose-melo/t-jepa). It is not the authors' code, and T-JEPA itself
has no outlier score: the paper evaluates supervised models on the embeddings. The scores are the
ones of `JEPAOD`. τ-JEPA, a working paper on fraud detection, is cited in `REPORT.md` and is not
implemented, because its text could not be read (login required).

### JEPA versions of `TSJEPAOD`

`TSJEPAOD(variant=...)` selects which JEPA trains the time-series network: twenty-seven versions, the
four below, twelve more in the second table, ten more in the third, and `'scjepa'`. Scores, model sizes, training
budget and the PyOD interface are the same for all of them. [LEADERBOARD.md](LEADERBOARD.md) has
every version on all of TSB-AD with nine evaluation measures.

| `variant` | network | target and collapse prevention | loss | source |
|---|---|---|---|---|
| `'lejepa'` (default) | per-patch linear tokenizer, learned positions, encoder without a final norm | one shared encoder with SIGReg | squared error | [LeJEPA](https://arxiv.org/abs/2511.08544) |
| `'ijepa'` | the same network | EMA target encoder, normalised targets | squared error | I-JEPA |
| `'tsjepa'` | convolutional patch tokenizer, sinusoidal positions, encoder with a final LayerNorm | EMA copy of the whole encoder, normalised targets | absolute error | [TS-JEPA](https://arxiv.org/abs/2509.25449) |
| `'tjepa'` | linear tokenizer, sinusoidal positions, [REG] token, post-norm encoder with a normalised output | EMA copy of the whole encoder | squared error | [T-JEPA](https://arxiv.org/abs/2410.05016), a tabular method applied to patch tokens |

TS-JEPA and T-JEPA are re-implemented from their papers and the defaults of their released code
([TS_JEPA](https://github.com/Sennadir/TS_JEPA), [t-jepa](https://github.com/jose-melo/t-jepa)); they
are not the authors' code. Each version contributes its network, target mechanism and loss. The
masking is the detector's own (`mask_mode`, `mask_ratio`); TS-JEPA's paper masks 70 % of the patches
at random, which is `mask_mode='random', mask_ratio=0.7`. None of the papers defines an anomaly
score. `REPORT.md` section 4.8 lists the differences, section 7.7 has the comparison.

Twelve further versions (added in 0.4.0; `REPORT.md` section 4.9 has the details, section 7.8 the results):

| `variant` | what trains the network | source |
|---|---|---|
| `'vjepa'` | `'ijepa'` with an absolute-error loss | [V-JEPA](https://arxiv.org/abs/2404.08471) |
| `'vjepa21'` | also predicts the visible patches, at several encoder depths | [V-JEPA 2.1](https://arxiv.org/abs/2603.14482) |
| `'cjepa'` | `'ijepa'` plus VICReg terms on the context embeddings | [C-JEPA](https://arxiv.org/abs/2410.19560) |
| `'sjepa'` | `'ijepa'` with a Gaussian latent for the predictor and a KL term | [stochastic-latent JEPA](https://arxiv.org/abs/2611.09302) |
| `'salt'` | a teacher trained by masked reconstruction, then frozen, provides the targets | [SALT](https://arxiv.org/abs/2509.24317) |
| `'vicreg'` | `'lejepa'` with VICReg variance and covariance terms in place of SIGReg | [EB-JEPA](https://arxiv.org/abs/2602.03604) |
| `'kerjepa'` | a closed-form kernel discrepancy in place of SIGReg, standardised targets | [KerJEPA](https://arxiv.org/abs/2512.19605) |
| `'lpjepa'` | non-negative embeddings matched to a rectified Laplace distribution | [Rectified LpJEPA](https://arxiv.org/abs/2602.01456) |
| `'lambdajepa'` | a Gaussian divergence (SACReg) in place of SIGReg | [lambda-JEPA](https://arxiv.org/abs/2609.35288) |
| `'lenepa'` | a causal Transformer predicts the embedding of the next patch; SIGReg per window; no masks | [LeNEPA](https://arxiv.org/abs/2607.00958) |
| `'lewm'` | next-patch embedding prediction with SIGReg per patch position; no masks; the paper's action input is removed | [LeWorldModel](https://arxiv.org/abs/2603.19312) |
| `'cfjepa'` | crops as context, forward prediction of three zones, convolutional encoder; no masks | [CF-JEPA](https://arxiv.org/abs/2606.07031) |

All twelve are written from the papers, not from released code. Where a paper leaves a constant or a
step open, it was fixed here before the benchmark was run; `REPORT.md` section 4.9 names each such
choice, and `variant_params` overrides any constant, e.g.
`TSJEPAOD(variant='vicreg', variant_params={'cov_weight': 40.0})`. `'lenepa'`, `'lewm'` and `'cfjepa'`
do not hide patches, so `mask_mode` and `mask_ratio` do not apply to them.

Ten more versions (added in 0.5.0; `REPORT.md` section 4.10 has the details and every choice that is
ours, section 7.11 the results). They are in this package only, not in the PyOD branch:

| `variant` | what trains the network | source |
|---|---|---|
| `'ststjepa'` | the targets come from an EMA copy of the tokenizer, not of the encoder; a reconstruction term on the predictions | [STST-JEPA](https://arxiv.org/abs/2607.06629) (EEG) |
| `'hepa'` | a causal encoder and a horizon-conditioned predictor predict the embedding of the next 1, 2, 4, ... patches; SIGReg on the predictions; no masks | [HEPA](https://arxiv.org/abs/2605.11130) (event prediction in time series) |
| `'subjepa'` | `'lewm'` with SIGReg in frozen random subspaces; no masks | [Sub-JEPA](https://arxiv.org/abs/2605.09241) |
| `'straight'` | next-patch prediction with a stop-gradient and a curvature term on the embedding path; no masks | [temporal straightening](https://arxiv.org/abs/2603.12231) |
| `'motionjepa'` | `'lewm'` plus an embedding of the difference of consecutive patches (DISReg); no masks | [MotionJEPA](https://arxiv.org/abs/2609.23881) |
| `'njepa'` | `'ijepa'` with a second predictor whose mask tokens carry Gaussian noise | [N-JEPA](https://arxiv.org/abs/2507.15216) |
| `'gmmjepa'` | `'ijepa'` plus a cluster head trained on the soft posteriors of a frozen Gaussian mixture of the patches | [GMM-anchored JEPA](https://arxiv.org/abs/2602.09040) (speech) |
| `'variational'` | the predictor returns a Gaussian; `pred` is a negative log-likelihood | [VJEPA](https://arxiv.org/abs/2601.14354) |
| `'phijepa'` | two prediction stages, towards a corrupted and then the clean view; cosine distance | [PhiJEPA](https://arxiv.org/abs/2611.27516) |
| `'harjepa'` | an EMA target that is not normalised; VICReg terms and a norm term on the predictions | [HAR-JEPA](https://arxiv.org/abs/2607.16350) (wearable sensors) |

Several of these papers leave more open than the earlier twelve did, `'variational'` most of all (its
paper is a framework without fixed networks or constants); `REPORT.md` section 4.10 says what is the
paper's and what is ours for each.

`'scjepa'` (added in 0.5.1; `REPORT.md` sections 4.11 and 7.13) is
[SC-JEPA](https://arxiv.org/abs/2602.04643), first published as MTS-JEPA, the one JEPA of the survey
that its authors built for anomalies in multivariate series. The first half of a window predicts the
second half, one variable at a time, as distributions over the prototypes of a soft codebook, for
every patch and for one time-averaged view of the half; EMA copies of the encoder and the codebook give
the targets, and a decoder reconstructs the first half. The objective and all its constants are those
of the paper's Appendix B.1. (Version 0.5.0 said here that the paper gives no loss weights. That was
wrong: only its main text had been read.) Ours are the mapping of the paper's two consecutive windows
onto the two halves of the detector's window, the tokenizer and decoder, and the scores, because the
paper trains a classifier on labels instead.

## What the benchmarks show

All 47 datasets of the ADBench classical suite, 5 seeds, mean ROC-AUC / average precision
(`REPORT.md`, section 7.6; fit times on one thread):

| detector | ROC-AUC | AP | fit time |
|---|---|---|---|
| untrained T-JEPA network, `embed` score | 0.768 | 0.371 | 16 s |
| `IForest` (PyOD) | 0.763 | 0.384 | 0.1 s |
| untrained `JEPAOD` network, `combined` score | 0.751 | 0.379 | 11 s |
| `AutoEncoder` (PyOD) | 0.748 | 0.333 | 3 s |
| `JEPAOD` | 0.736 | 0.353 | 119 s |
| `TJEPAOD` | 0.732 | 0.318 | 240 s |
| `KNN` (PyOD) | 0.732 | 0.360 | 0.2 s |

- The trained tabular detectors are mid-table: 9th and 11th of the 16 columns compared.
  `JEPAOD` is significantly behind `IForest` (`IForest` is better on 35 of 47 datasets, corrected
  p = 0.010) and is not ahead of `AutoEncoder`. `TJEPAOD` and `JEPAOD` cannot be separated.
- A network with **zero training epochs** scores at least as well as the trained ones (for
  `JEPAOD` the untrained network is better on 32 of 47 datasets; not significant after correction).
  The benchmark gives no evidence that JEPA training helps on tabular data.
- The untrained score is level with `IForest` and more than a hundred times slower, so it is an
  observation, not a detector to recommend.
- An earlier sweep on 16 of these datasets with 3 seeds (sections 7.1 and 7.5) had `JEPAOD` as the
  best deep detector (0.760) and `TJEPAOD` at 0.782. Those rankings did not carry over to the other
  31 datasets.

Time series, `TSJEPAOD` at its defaults against the same zero-epoch control (5 seeds, ROC-AUC of
the default `pred` score):

| benchmark | trained | untrained |
|---|---|---|
| synthetic series (15) | 0.941 | 0.798 |
| SKAB (16 files) | 0.825 | 0.831 |
| NAB (7 series) | 0.628 | 0.671 |

Training is what makes the prediction-error score work on the synthetic series, where the
anomalies break temporal structure. On the two real benchmarks it adds nothing measurable.

The JEPA versions of `TSJEPAOD` on the same benchmarks (ROC-AUC of the default `pred` score, trained /
untrained; `REPORT.md`, section 7.7):

| `variant` | synthetic series | SKAB | NAB |
|---|---|---|---|
| `'lejepa'` (default) | 0.941 / 0.798 | 0.825 / 0.831 | 0.628 / 0.671 |
| `'ijepa'` | 0.885 / 0.519 | 0.431 / 0.422 | 0.614 / 0.521 |
| `'tsjepa'` | 0.931 / 0.485 | 0.442 / 0.561 | 0.585 / 0.483 |
| `'tjepa'` | 0.928 / 0.488 | 0.671 / 0.426 | 0.684 / 0.530 |

No version beats the default. The three EMA versions normalise their targets or encoder output and
fail on SKAB, where the test recordings are in a different operating regime than the training
recording; `tjepa` with `encoder_norm=False` recovers there (0.836, checked on SKAB only). The lead
of `tjepa` on NAB is not significant. The EMA versions fit about two to four times faster than
`lejepa` on the synthetic series (indicative timings).

The twelve further versions on the same benchmarks (`REPORT.md`, section 7.8; 72 paired tests fixed before
the runs):

| `variant` | synthetic series | SKAB | NAB |
|---|---|---|---|
| `'kerjepa'` | 0.950 / 0.798 | 0.829 / 0.831 | 0.633 / 0.671 |
| `'vicreg'` | 0.935 / 0.798 | 0.829 / 0.831 | 0.576 / 0.671 |
| `'lambdajepa'` | 0.932 / 0.798 | 0.828 / 0.831 | 0.592 / 0.671 |
| `'lpjepa'` | 0.898 / 0.792 | 0.828 / 0.825 | 0.600 / 0.678 |
| `'lewm'` | 0.933 / 0.799 | 0.726 / 0.810 | 0.622 / 0.661 |
| `'salt'` | 0.919 / 0.531 | 0.605 / 0.421 | 0.676 / 0.488 |
| `'vjepa21'` | 0.860 / 0.455 | 0.619 / 0.458 | 0.600 / 0.479 |
| `'cjepa'` | 0.888 / 0.519 | 0.419 / 0.422 | 0.612 / 0.521 |
| `'sjepa'` | 0.885 / 0.493 | 0.436 / 0.396 | 0.612 / 0.547 |
| `'vjepa'` | 0.874 / 0.519 | 0.459 / 0.422 | 0.614 / 0.521 |
| `'cfjepa'` | 0.816 / 0.602 | 0.206 / 0.475 | 0.689 / 0.503 |
| `'lenepa'` | 0.758 / 0.809 | 0.496 / 0.467 | 0.434 / 0.589 |

* `kerjepa` is level with the default in all three groups (ahead by 0.005 to 0.008; only the SKAB
  difference is significant after correction) and fits in about half the time. No other version is
  ahead of the default with significance anywhere.
* On SKAB the split follows the normalisation of the prediction target, not the regulariser: the
  four versions that swap SIGReg for another term on a shared, un-normalised encoder score 0.828 to
  0.829; every version that normalises its targets scores 0.21 to 0.62.
* `cjepa`, `sjepa` and `vjepa` behave like `ijepa`. `salt` and `cfjepa` are numerically ahead on NAB
  (not significant). The `embed` score of the `cfjepa` network has the highest NAB score of any
  version, 0.797 trained and 0.792 untrained.
* The prediction-error score of `lenepa` and `cfjepa` does not work in this detector; their `embed`
  score does (0.939 / 0.849 / 0.657 and 0.853 / 0.853 / 0.797).

### TSB-AD: 530 longer series

The PyOD port of the time-series detector on the two evaluation lists of
[TSB-AD](https://github.com/TheDatumOrg/TSB-AD) (180 multivariate and 350 univariate series, 37.7 million
timestamps), at its class defaults, untuned, on a GPU (`REPORT.md`, section 7.9; 62 paired tests fixed in
advance). Mean VUS-PR, the measure TSB-AD ranks by:

| | TSB-AD-M (180 series) | TSB-AD-U (350 series) |
|---|---|---|
| best published detector | 0.313 (CNN) | 0.423 (Sub-PCA) |
| `'lejepa'`, the default, trained / untrained, score `pred` | 0.351 / 0.347 | 0.378 / 0.346 |
| the same, score `embed` | 0.353 / 0.352 | 0.461 / 0.467 |
| no network: Mahalanobis distance of raw patches | 0.345 | 0.465 |
| `'vicreg'`, trained / untrained, `pred` | 0.360 / 0.347 | 0.417 / 0.346 |
| `'lambdajepa'`, trained / untrained, `pred` | 0.355 / 0.347 | 0.392 / 0.346 |
| `'lewm'`, trained / untrained, `pred` | 0.386 / 0.369 | 0.394 / 0.309 |
| `'cfjepa'`, trained / untrained, `embed` (picked after the results) | 0.434 / 0.436 | 0.530 / 0.499 |

* The default detector is above all 23 published means on the multivariate list and seventh of 33 on the
  univariate one. The lead on the multivariate list is a lead in the mean, not per series (better than CNN
  on 87 of 180 series, p = 0.11), and 0.011 of it comes from an accident: a channel that is constant in
  the training prefix is multiplied by 1e8 when it moves later.
* The published numbers are TSB-AD's own, tuned by its authors, and were not re-run, except PCA and
  Sub-PCA in the confirmation below.
* Training adds a significant but small part of the default score (+0.004 and +0.032). The untrained
  network places the same on the multivariate list, and the `embed` score is matched by a Mahalanobis
  distance on raw patches with no network.
* `vicreg`, `lambdajepa` and `lewm` are significantly ahead of the default on both lists; `kerjepa` is
  level with it. The eight versions with an EMA or frozen-teacher target need training (their untrained
  `pred` score is 0.09 to 0.23) and mostly end below the untrained default.

### TSB-AD confirmation: 540 series that no run had used

The conclusions above were drawn after looking at the evaluation lists. They were then tested on the rest
of TSB-AD, with the plan (`results/tsb_confirm_plan.md`: lists, 99 runs, 18 tests) committed before any
run: the two tuning lists, and the 472 univariate series that are in neither the evaluation nor the tuning
list (`REPORT.md`, section 7.10). Mean VUS-PR:

| | M-Tuning (20 series) | U-Tuning (48 series) | U-Rest (472 series) |
|---|---|---|---|
| reference of TSB-AD's own code, re-run: PCA / Sub-PCA | 0.323 | 0.326 | 0.146 |
| `'lejepa'`, the default, trained / untrained, score `pred` | 0.363 / 0.353 | 0.337 / 0.304 | 0.177 / 0.146 |
| the same, score `embed` | 0.373 / 0.375 | 0.449 / 0.459 | 0.432 / 0.435 |
| the same, score `combined` | 0.368 / 0.368 | 0.443 / 0.447 | 0.401 / 0.403 |
| no network: Mahalanobis distance of raw patches | 0.374 | 0.458 | 0.545 |
| `'vicreg'`, trained, `pred` | 0.390 | 0.410 | 0.242 |
| `'lambdajepa'`, trained, `pred` | 0.372 | 0.367 | 0.216 |
| `'lewm'`, trained, `pred` | 0.441 | 0.398 | 0.213 |
| `'cfjepa'`, trained / untrained, `embed` | 0.528 / 0.513 | 0.508 / 0.464 | 0.333 / 0.245 |

* Ten of the 18 tests hold after a Holm correction, seven have the expected sign without being significant
  (five of them on the 20 multivariate series), one is reversed.
* `vicreg`, `lambdajepa` and `lewm` are ahead of the default again on all three lists, by more than on the
  evaluation lists; `kerjepa` is level again. Training again adds 0.03 to the default score on univariate
  series.
* `cfjepa` with the `embed` score is ahead of the default detector again, and it is not the best
  configuration in general: on U-Rest it is 0.21 below the control without a network, which there is
  above all sixteen versions with every score type. Per source dataset the numbers repeat those of the
  evaluation list: it gains on UCR, the ECG sets, SMAP and NAB and loses on YAHOO, which is 224 of the
  472 series of U-Rest.
* On univariate series the default score `pred` is 0.08 to 0.22 below `combined` on the evaluation list
  and on both new lists; on multivariate series the score types are level. The default was not changed.
* U-Rest has four source datasets only (YAHOO, UCR, WSD, Stock), and Sub-PCA is not the strong published
  detector on them, so its numbers say little about the standing among published detectors. TSB-AD's best
  multivariate detectors (CNN, OmniAnomaly) were not re-run.
* The highest scores come from the `embed` score of the `cfjepa` encoder, on the multivariate list with
  or without training. That cell was picked on the evaluation series and needs a run on other series.
* The default version takes 46 and 49 minutes for a whole list on one shared H100. At the class defaults
  a GPU is slower than 8 CPU threads; it is about 3 times faster with a larger network
  (`results/gpu/gpu_check.json`).

### Ten more versions (0.5.0) on all of TSB-AD

`REPORT.md` section 7.11. The ten versions of 0.5.0 were run on all five lists (1070 series, 85 runs,
no failed series). The 100 tests were fixed and pushed before any of their results was read
(`results/tsb_more_plan.md`).

Mean VUS-PR of the default score `pred`, trained:

| | M-Eva (180) | M-Tuning (20) | U-Eva (350) | U-Tuning (48) | U-Rest (472) |
|---|---|---|---|---|---|
| `lejepa` (default) | 0.351 | 0.363 | 0.378 | 0.337 | 0.177 |
| `lewm` (0.4.0) | 0.386 | 0.441 | 0.394 | 0.398 | 0.213 |
| `straight` | 0.427 | 0.444 | 0.398 | 0.396 | 0.232 |
| `hepa` | 0.414 | 0.481 | 0.418 | 0.338 | 0.134 |
| `subjepa` | 0.380 | 0.426 | 0.397 | 0.401 | 0.203 |
| `motionjepa` | 0.384 | 0.434 | 0.388 | 0.393 | 0.201 |
| `ststjepa` | 0.356 | 0.370 | 0.386 | 0.346 | 0.172 |
| `harjepa` | 0.346 | 0.349 | 0.348 | 0.306 | 0.149 |
| `variational` | 0.297 | 0.372 | 0.323 | 0.317 | 0.186 |
| `phijepa` | 0.299 | 0.383 | 0.316 | 0.315 | 0.154 |
| `njepa` | 0.272 | 0.296 | 0.326 | 0.302 | 0.108 |
| `gmmjepa` | 0.271 | 0.350 | 0.324 | 0.318 | 0.133 |

* Against the default: 11 of the 50 tests ahead, 9 behind, 30 not separated. Training helps in 36 of
  50 tests against the untrained control and hurts in none.
* `straight` is ahead of the default on all five lists, with each of three seeds. Its lead over `lewm`,
  whose network it has, is
  one source dataset: on the 25 GHL series it has 0.348 and every other version 0.045 or less; without
  them it has 0.441 and `lewm` 0.446. On a GHL series its embeddings have collapsed to an effective rank
  of 1.7 of 64 (stop-gradient is its only protection), which is what makes it insensitive to the drift
  after the training prefix there, and it is the weakest of that network's versions on SKAB (0.632).
* `hepa` is ahead on both evaluation lists and behind on U-Rest. It gains on the ECG sets and SMAP and
  loses on YAHOO, WSD, IOPS, Stock and TAO.
* The four new versions with an EMA target and normalised targets (`njepa`, `gmmjepa`, `variational`,
  `phijepa`) need training and end below the default on the evaluation lists, like the earlier versions
  of that kind.
* `ststjepa` and `harjepa` have an EMA target that is not normalised, and they keep SKAB (0.829, 0.833;
  default 0.825): of the 19 versions that hide patches, the seven with targets that are not normalised
  have 0.825 to 0.833 there and the twelve others 0.42 to 0.73. So the normalisation of the targets,
  not the EMA teacher, fails under that regime shift.
* For several of the ten, parts of the recipe are ours, so a result says how this implementation does
  at this budget. One seed, except `straight` (three); no tuning. The default was not changed.

### All nine TSB-AD measures: the leaderboard

`REPORT.md` section 7.12 and [LEADERBOARD.md](LEADERBOARD.md). Every run of sections 7.9 to 7.11
(289 runs of a list with those of section 7.13, 184,520 score vectors, all 200 multivariate and 870 univariate series) now has the nine
measures of TSB-AD, computed with `jepa_od.metrics`.

* AUC-PR and Standard-F1 rank the 27 versions like VUS-PR (rank correlation 0.95 to 1.00), the
  event-based and range-based F1 nearly so (0.86 to 0.89). VUS-ROC and AUC-ROC do not (0.26 to 0.55).
* PA-F1 turns the multivariate table around (rank correlation −0.43): TSB-AD's own PCA has the highest
  PA-F1 of all rows (0.791) and is 18th of 29 by VUS-PR, and `scjepa`, last by VUS-PR, has the
  second-highest PA-F1 of the versions.
* The comparisons the report rests on do not depend on the measure: `vicreg`, `lewm` and `straight` are
  above the default on all nine measures on both sets, and training the default helps on all nine.
  "The EMA versions are worse" holds for the precision-recall measures only.
* On the 870 univariate series the control without a network (raw patches, `embed`) is first on four
  of nine measures, VUS-PR included (0.508; the best version has 0.495).
* The default version is 10th of 27 by VUS-PR on both sets.

### SC-JEPA (`scjepa`, 0.5.1)

`REPORT.md` section 7.13. The recipe of SC-JEPA with the paper's constants, run on the small benchmarks
and on all 1070 TSB-AD series (ten runs, no failed series); 16 tests fixed and pushed before the runs.

| mean VUS-PR, trained | M-Eva (180) | M-Tuning (20) | U-Eva (350) | U-Tuning (48) | U-Rest (472) |
|---|---|---|---|---|---|
| `lejepa` (default), `pred` | 0.351 | 0.363 | 0.378 | 0.337 | 0.177 |
| `scjepa`, `pred` | 0.122 | 0.184 | 0.150 | 0.150 | 0.132 |
| `scjepa`, `embed` | 0.190 | 0.244 | 0.325 | 0.316 | 0.529 |
| no network (raw patches), `embed` | 0.345 | 0.374 | 0.465 | 0.458 | 0.545 |

* With the default score it is behind the default version on all five lists and the weakest of the 27
  versions. Training helps a little (0.015 to 0.075 on four lists).
* The reason is the paper's normalisation: every half-window is normalised by its own mean and scale, so
  a change of level never reaches the network. It loses most on Exathlon, OPPORTUNITY, NEK, SMD, SMAP
  and MSL and is level or ahead on sets with shape anomalies (MGAB, UCR, WSD). On SKAB it is at chance
  (0.482), and at 0.782 with the normalisation switched off (`variant_params={'instance_norm': False}`,
  an ablation, not the paper's method). On TSB-AD that switch raises `pred` to 0.259, 0.306, 0.352 and
  0.301 on the first four lists, still 0.03 to 0.09 below the default, and lowers it to 0.121 on the fifth.
* Its `embed` score, on the code distributions, has the highest value of any version on the 472 further
  univariate series (0.529) and is low on the other lists.
* The paper uses the model with a classifier trained on labels, to predict anomalies in the next window.
  Nothing here tests that; the result is about the recipe as an unsupervised detector at this budget.

## Install

```bash
pip install -e .            # installs numpy, scipy, scikit-learn, torch, pyod
pip install pandas          # only needed for the benchmarks
```

## Usage

```python
from pyod.utils.data import generate_data, generate_ts_data
from jepa_od import JEPAOD, TJEPAOD, TSJEPAOD

# tabular: identical API to any PyOD detector
X_train, X_test, y_train, y_test = generate_data(n_features=20, random_state=42)
clf = JEPAOD(epoch_num=30, contamination=0.1, verbose=0).fit(X_train)
scores = clf.decision_function(X_test)      # higher = more abnormal
labels = clf.predict(X_test)                # 0/1 using threshold_
proba  = clf.predict_proba(X_test)

# the T-JEPA member of the family: same API, same scores
tj = TJEPAOD(epoch_num=30, contamination=0.1, verbose=0).fit(X_train)
tj_scores = tj.decision_function(X_test)
emb = tj.token_embeddings(X_test)           # (n_samples, n_tokens, d_model), also on JEPAOD

# time series: per-timestamp scores, PyOD masked-score workflow
X_train, X_test, y_train, y_test = generate_ts_data(n_channels=3, random_state=42)
ts = TSJEPAOD(window_size=100, patch_len=10).fit(X_train)   # defaults: 10 epochs, <=5000 windows
ts_scores = ts.decision_function(X_test)    # shape (n_timestamps,)

# the same detector trained with another JEPA version: 'lejepa' (default), 'ijepa', 'tsjepa', 'tjepa',
# 'vjepa', 'vjepa21', 'cjepa', 'sjepa', 'salt', 'vicreg', 'kerjepa', 'lpjepa', 'lambdajepa', 'lenepa',
# 'lewm', 'cfjepa', 'ststjepa', 'hepa', 'subjepa', 'straight', 'motionjepa', 'njepa', 'gmmjepa',
# 'variational', 'phijepa', 'harjepa', 'scjepa'
ts2 = TSJEPAOD(window_size=100, patch_len=10, variant='tsjepa').fit(X_train)

# the nine evaluation measures of the TSB-AD benchmark for a score vector
from jepa_od.metrics import ts_metrics, acf_window
measures = ts_metrics(y_test, ts_scores, window=acf_window(X_test[:, 0]))
# {'VUS-PR': ..., 'VUS-ROC': ..., 'AUC-PR': ..., 'AUC-ROC': ..., 'Standard-F1': ..., 'PA-F1': ...,
#  'Event-based-F1': ..., 'R-based-F1': ..., 'Affiliation-F': ...}
```

### Evaluation measures (`jepa_od.metrics`)

`ts_metrics(labels, scores, window)` returns the nine measures TSB-AD reports for a score vector:
VUS-PR and VUS-ROC (range-aware, averaged over label extensions up to `window`), AUC-PR, AUC-ROC, and
the best F over thresholds in five forms (point-wise, point-adjusted, event-based, range-based,
affiliation). Each is also a function of its own (`vus`, `pa_f1`, `range_f1`, `affiliation_f1`, ...),
and `acf_window` gives the dominant period TSB-AD uses as the window.

The definitions follow TSB-AD's evaluation code, edge cases included, so the numbers can be put next
to that benchmark's. They are computed with numpy, where TSB-AD's functions
loop in Python over thresholds, anomalies and timestamps: on 1000 random cases and on 120 saved score
vectors of up to 40,000 timestamps the largest difference to TSB-AD's code is below 3e-15 for every
measure, and the time is 28 and 55 times shorter (`benchmarks/check_metrics_tsb.py`; longer series
were not compared, because TSB-AD's code is slow on them). The five F measures use a
threshold chosen with the labels, as TSB-AD reports them: they say what a detector could reach, not
what it reaches with its own threshold.

Key options (`JEPAOD` and `TSJEPAOD`): `mask_mode` (`random` / `block` / `forecast` / `mixed`),
`mask_ratio`, `n_mask_samples`, `score` (`pred` / `embed` / `combined`; tabular default `combined`, time-series default `pred`),
`target_mode` (`JEPAOD` only: `sigreg` default, or `ema` for an I-JEPA style EMA teacher),
`recon_weight` (optional MTS-JEPA style reconstruction anchor), `d_model`, `n_layers`.

`TSJEPAOD` additionally has `variant` (the JEPA version, see above), `variant_params` (overrides of a
version's constants), `patch_len`, `step`,
`score_aggregation` (`mean`/`max`), `epochs` and `max_train_windows` (the training budget; keep it small on
long, contaminated series). `mask_mode='forecast'` makes it causal (only past context), suitable for online
monitoring. `encoder_norm` must stay off for `JEPAOD` and for the `lejepa` version of `TSJEPAOD` (their
default): a final LayerNorm saturates extreme or shifted inputs and inverts the scores (REPORT.md,
section 4.6). `epochs=0` gives the untrained control of any version.

`TJEPAOD` keeps `score`, `n_mask_samples`, `d_model`, `n_layers` and `max_tokens` and replaces the masking and
collapse options with T-JEPA's: `min_ctx_share` / `max_ctx_share`, `min_tgt_share` / `max_tgt_share`,
`n_pred_masks`, `n_reg_tokens`, `ema_momentum`, `warmup_epochs`, and the predictor sizes `pred_dim`,
`pred_layers`, `pred_heads`. Its `encoder_norm` defaults to `True`, as in T-JEPA, and the failure above does
not occur with it (REPORT.md, section 7.5). `epoch_num=0` on either tabular class gives the untrained control.

## Layout

```
jepa_od/          package (detectors, shared networks and regularisers, metrics.py)
tests/            PyOD-style unittest suites
benchmarks/       bench_tabular.py (ADBench), bench_ts.py (synthetic, NAB, SKAB),
                  bench_tjepa.py (T-JEPA family and untrained controls), toy_two_features.py
                  bench_extended.py (47 ADBench datasets, 5 seeds, untrained controls),
                  bench_ts_jepa_final.py (TSJEPAOD defaults; --epochs 0 is the untrained control)
results/          CSV results + logs produced by the benchmarks
integration/      PyOD port: patches, hand-off notes, drafted knowledge-base entries
examples/         runnable example
data/             benchmark datasets, not in the repository (see below)
```

## Tests

```bash
python -m unittest discover -s tests
```

## Reproducing the benchmarks

The datasets (ADBench, NAB, SKAB) are third-party and are fetched on demand into `data/`:

```bash
python benchmarks/download_data.py
python benchmarks/bench_tabular.py      # writes results/tabular_results*.csv
python benchmarks/bench_ts.py           # writes results/ts_results*.csv
python benchmarks/bench_tjepa.py        # T-JEPA family and controls, writes results/tjepa_results.csv
python benchmarks/toy_two_features.py   # writes results/toy_two_features.*
```

The extended benchmark of section 7.6 (about 2.5 hours on 13 cores; it resumes if interrupted):

```bash
python benchmarks/download_data.py --all-adbench     # the other 31 ADBench datasets, 75 MB
python benchmarks/bench_extended.py --workers 13     # writes results/extended_parts/
python benchmarks/bench_ts_jepa_final.py --seeds 5 --threads 2 --out results/ts_control_trained.csv
python benchmarks/bench_ts_jepa_final.py --seeds 5 --threads 2 --epochs 0 --tag TSJEPAOD-untrained --out results/ts_control_untrained.csv
python benchmarks/make_extended_tables.py            # tables and REPORT.md section 7.6
```

The JEPA versions of section 7.7 (`V` is `ijepa`, `tsjepa` or `tjepa`):

```bash
python benchmarks/bench_ts_jepa_final.py --seeds 5 --threads 2 --variant V --out results/ts_variants/V_trained.csv
python benchmarks/bench_ts_jepa_final.py --seeds 5 --threads 2 --variant V --epochs 0 --out results/ts_variants/V_untrained.csv
python benchmarks/diag_ts_variant_norm.py            # SKAB without the normalised encoder output
python benchmarks/make_ts_variant_tables.py          # tables and REPORT.md section 7.7
```

The header of `bench_ts_jepa_final.py` has the two extra commands for the masking of the TS-JEPA paper.

The twelve further versions of section 7.8 (24 runs, six at a time, two threads each), and the ten of
section 7.11 with `SET=ten` (17 runs):

```bash
sh benchmarks/run_ts_versions.sh 6                   # writes results/ts_variants/V_{trained,untrained}.csv
python benchmarks/make_ts_versions_tables.py         # tables and REPORT.md section 7.8
python benchmarks/time_ts_versions.py                # fit time of all sixteen versions on one series
```

The TSB-AD benchmark of section 7.9 needs a machine with a GPU, a checkout of the PyOD branch
`feature/ts-jepa`, a clone of TSB-AD and its two data archives (`TSB-AD-M.zip`, `TSB-AD-U.zip` from
https://www.thedatum.org/datasets/), laid out as the header of `run_tsb.sh` describes. The committed run
was 75 runs of a whole list, twelve at a time on four shared H100s (38 hours of fitting and scoring in all):

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=<pyod checkout> python benchmarks/gpu_check.py results/gpu/gpu_check.json \
    small default large                              # the CPU / GPU comparison of all sixteen versions
ROOT=<work folder> GPUS=0,1 JOBS=6 sh benchmarks/run_tsb.sh    # scores; resumes if interrupted
python -I benchmarks/eval_tsb.py --tsb <TSB-AD clone> --data <folder with TSB-AD-M/ and TSB-AD-U/> \
    --results <work folder>/results/tsb --jobs 64    # VUS-PR and the other measures per series
python benchmarks/download_data.py --tsb-published   # TSB-AD's published per-series tables
python benchmarks/make_tsb_tables.py                 # tables and REPORT.md section 7.9
```

`run_tsb.sh` uses a GPU only while no other job computes on it and moves when one arrives. The per-series
measures and run logs of the committed run are in `results/tsb/`; the score vectors (16 GB) are not.

The confirmation of section 7.10 uses the same folder layout (99 runs, 19 hours of fitting and scoring):

```bash
QUEUE=confirm ROOT=<work folder> GPUS=0,1 JOBS=6 sh benchmarks/run_tsb.sh   # the three further lists
python -I benchmarks/bench_tsb_ref.py --tsb <TSB-AD clone> --data <folder with TSB-AD-M/ and TSB-AD-U/> \
    --suite U --split Rest --detector Sub_PCA --out <work folder>/results/tsb   # also Eva and Tuning; for
    # the multivariate lists: --suite M --detector PCA --params slidingWindow=1
python -I benchmarks/eval_tsb.py --tsb <TSB-AD clone> --data <folder with TSB-AD-M/ and TSB-AD-U/> \
    --results <work folder>/results/tsb --jobs 64
python benchmarks/make_tsb_confirm_tables.py         # tables and REPORT.md section 7.10
```

The ten versions of section 7.11 and the nine measures of section 7.12 use the same folder layout, with a
copy of this repository in `<work folder>/jepa-od` (85 runs, 26 hours of fitting and scoring; the nine
measures of all runs take a quarter of an hour on 72 processes):

```bash
QUEUE=more ROOT=<work folder> GPUS=0,1 JOBS=6 sh benchmarks/run_tsb.sh      # the ten versions, five lists
python -I benchmarks/check_metrics_tsb.py --tsb <TSB-AD clone> --results <work folder>/results/tsb \
    --data <folder with TSB-AD-M/ and TSB-AD-U/>     # jepa_od/metrics.py against TSB-AD's own code
python -I benchmarks/eval_tsb.py --tsb <TSB-AD clone> --data <folder with TSB-AD-M/ and TSB-AD-U/> \
    --results <work folder>/results/tsb --jobs 64    # the nine measures of every saved score vector
SET=ten sh benchmarks/run_ts_versions.sh 5           # the small benchmarks of the ten versions
python benchmarks/make_tsb_more_tables.py            # tables and REPORT.md section 7.11
QUEUE=scjepa ROOT=<work folder> GPUS=0,1 JOBS=5 sh benchmarks/run_tsb.sh    # scjepa (section 7.13), then
QUEUE=scjepa_nonorm ROOT=<work folder> GPUS=0,1 JOBS=5 sh benchmarks/run_tsb.sh    # its ablation,
SET=scjepa sh benchmarks/run_ts_versions.sh 2        # eval_tsb.py as above, and
python benchmarks/make_tsb_scjepa_tables.py          # tables and REPORT.md section 7.13
python benchmarks/make_leaderboard.py                # LEADERBOARD.md and REPORT.md section 7.12
python -I benchmarks/diag_ts_collapse.py --pyod <pyod checkout> <a TSB-AD csv file>   # finding 4 of 7.11
```

Result tables for the committed runs are in `results/report_tables.md`, `results/tjepa_tables.md`,
`results/extended_tables.md`, `results/ts_control_tables.md`, `results/ts_variant_tables.md`,
`results/ts_versions_tables.md`, `results/tsb_tables.md`, `results/tsb_confirm_tables.md`,
`results/tsb_more_tables.md`, `results/tsb_scjepa_tables.md` and `LEADERBOARD.md`
(`benchmarks/make_tjepa_tables.py` builds `results/tjepa_tables.md`; the committed run was five parallel jobs with
3 threads each, see the header of `bench_tjepa.py`). Runs are deterministic for a fixed seed and thread count; a different thread count
changes single results (by up to 0.08 ROC-AUC between one and three threads) and leaves the means over datasets within 0.002.

## Status

Research prototype. `JEPAOD` and `TSJEPAOD` are ported to PyOD as `pyod.models.jepa.JEPA` and
`pyod.models.ts_jepa.TSJEPA`: two commits on top of PyOD 3.6.8, on the branch
[`feature/jepa-od`](https://github.com/kishordgupta/pyod/tree/feature/jepa-od) of the PyOD fork and
as patches in `integration/pyod-patches/`. `integration/PYOD_PR_CHECKLIST.md` has the file map, the checks that
ran and the steps to a pull request. `TJEPAOD` is in this repository only.

A second branch, [`feature/ts-jepa`](https://github.com/kishordgupta/pyod/tree/feature/ts-jepa), holds the
narrowed proposal: `TSJEPA` alone, with the `variant` option, as three commits on top of PyOD 3.6.9 (patches in
`integration/pyod-patches-ts-jepa/`). The first commit has the first four versions, the second the twelve
further ones, the third makes six tests pass on a machine with a GPU. It is ready for a pull request if the
maintainer accepts. The TSB-AD benchmark of sections 7.9 and 7.10 ran this branch; the four speed and scaling issues it
showed on long series are listed in `integration/PYOD_PR_CHECKLIST.md` and are not changed in the branch.

`JEPA_VERSIONS_SURVEY.md` lists the further JEPA recipes that fit the `variant` family and the ones
that do not. Twelve of them are implemented in this repository (version 0.4.0, 16 versions in all) and,
since 2026-11-22, in the PyOD branch, where `TSJEPA` returns the same scores as `TSJEPAOD` for all sixteen.
Version 0.5.0 (2026-11-25) adds ten more versions and `jepa_od.metrics` to this repository only, and
0.5.1 (2026-11-28) adds `scjepa`; the PyOD branch is unchanged and has none of them.
The first comment on the PyOD issue describes four versions. A second one, posted on 2026-11-22, links the
branch and names the three versions level with the default (`kerjepa`, `vicreg`, `lambdajepa`); its text is
in `integration/PYOD_ISSUE_COMMENT2_DRAFT.md`. It does not mention the nine other versions in the branch.

The port was written before the extended benchmark. On the full ADBench suite the tabular detector is
not better than detectors PyOD already has, and `REPORT.md` (verdict in section 1) no longer argues for
adding it on performance grounds.

## License

BSD 2-Clause, the same license as PyOD.
