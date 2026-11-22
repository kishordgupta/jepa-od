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
| origin | this project, built on I-JEPA and LeJEPA | training recipe of T-JEPA (Thimonier et al., ICLR 2025, [arXiv 2410.05016](https://arxiv.org/abs/2410.05016)) | this project; four selectable JEPA versions (below) |
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

`TSJEPAOD(variant=...)` selects which JEPA trains the time-series network. Masking, scores, model
sizes, training budget and the PyOD interface are the same for all four.

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

# the same detector trained with another JEPA version: 'lejepa' (default), 'ijepa', 'tsjepa', 'tjepa'
ts2 = TSJEPAOD(window_size=100, patch_len=10, variant='tsjepa').fit(X_train)
```

Key options (`JEPAOD` and `TSJEPAOD`): `mask_mode` (`random` / `block` / `forecast` / `mixed`),
`mask_ratio`, `n_mask_samples`, `score` (`pred` / `embed` / `combined`; tabular default `combined`, time-series default `pred`),
`target_mode` (`JEPAOD` only: `sigreg` default, or `ema` for an I-JEPA style EMA teacher),
`recon_weight` (optional MTS-JEPA style reconstruction anchor), `d_model`, `n_layers`.

`TSJEPAOD` additionally has `variant` (the JEPA version, see above), `patch_len`, `step`,
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
jepa_od/          package (detectors + shared network / regulariser)
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

Result tables for the committed runs are in `results/report_tables.md`, `results/tjepa_tables.md`,
`results/extended_tables.md`, `results/ts_control_tables.md` and `results/ts_variant_tables.md`
(`benchmarks/make_tjepa_tables.py` builds the latter; the committed run was five parallel jobs with
3 threads each, see the header of `bench_tjepa.py`). Runs are deterministic for a fixed seed and thread count; a different thread count
changes single results (by up to 0.08 ROC-AUC between one and three threads) and leaves the means over datasets within 0.002.

## Status

Research prototype. `JEPAOD` and `TSJEPAOD` are ported to PyOD as `pyod.models.jepa.JEPA` and
`pyod.models.ts_jepa.TSJEPA`: two commits on top of PyOD 3.6.8, on the branch
[`feature/jepa-od`](https://github.com/kishordgupta/pyod/tree/feature/jepa-od) of the PyOD fork and
as patches in `integration/pyod-patches/`. `integration/PYOD_PR_CHECKLIST.md` has the file map, the checks that
ran and the steps to a pull request. `TJEPAOD` and the `variant` option of `TSJEPAOD` are in this
repository only and are not part of the port, whose `TSJEPA` still has `target_mode`.

The port was written before the extended benchmark. On the full ADBench suite the tabular detector is
not better than detectors PyOD already has, and `REPORT.md` (verdict in section 1) no longer argues for
adding it on performance grounds.

## License

BSD 2-Clause, the same license as PyOD.
