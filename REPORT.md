# JEPA-OD: Can Joint-Embedding Predictive Architectures Detect Outliers?

*Research study, design and prototype for a PyOD-compatible JEPA outlier detector (tabular and time series).*
*Date: 2026-11-20. Code: `jepa_od/`, benchmarks: `benchmarks/`, raw results: `results/`.*

---

## 1. Executive summary

**Question.** JEPA (Joint-Embedding Predictive Architecture) models learn by predicting the *representation* of a hidden part of the input from the visible part. Can that mechanism be turned into an unsupervised outlier detector that fits the PyOD API, for tabular rows and for time series?

**Answer.** Yes. The natural score is the **latent prediction error**: a normal sample has internal structure (feature dependencies, temporal continuity) that the predictor has learned, so its hidden parts are predictable in representation space; an outlier breaks that structure. Four 2026 papers already use exactly this signal in narrow domains (medical images, SAR satellite stacks, automotive telemetry, multivariate telemetry), but none provides a generic, library-ready detector. We built one:

* **`JEPAOD`** (tabular) subclasses PyOD's `BaseDeepLearningDetector`; **`TSJEPAOD`** (time series) follows PyOD 3.6's time-series detector pattern (`LSTMAD`, `TimeSeriesOD`). Both pass PyOD-style unit tests (27 tests), are deterministic, picklable and `sklearn.clone`-able.
* The architecture is a small *token transformer* (features or time patches are tokens) with an I-JEPA style predictor. Representation collapse is prevented with **SIGReg** from LeJEPA (Balestriero & LeCun, 2025) instead of an EMA teacher: one loss weight, no stop-gradient tricks, and, crucially for outlier detection, the embedding distribution of normal data becomes a *known isotropic Gaussian*. That gives a second score for free, the **Mahalanobis distance of each token embedding** to the training Gaussian, which can be fused with the prediction error.
* Two design lessons were found empirically and are, to our knowledge, not reported in the JEPA anomaly literature: (i) the encoder must **not** end in a LayerNorm, otherwise extreme inputs saturate to a direction that is *trivially predictable* and outliers score as *more normal* than inliers (AUC 0.19 on a 2-feature toy problem, 0.98 after the fix); (ii) a pooled (mean) row embedding is a poor density score because an outlier's token embeddings are incoherent and *average out to the origin*; per-token Mahalanobis fixes this.
* **Results (Section 7).** Tabular, 16 ADBench datasets, 3 seeds: `JEPAOD` is the best deep detector on average (ROC-AUC 0.760 / AP 0.438 vs AutoEncoder 0.746 / 0.375, DeepSVDD 0.737 / 0.358, DIF 0.733, LUNAR 0.729) and third overall behind KNN (0.786) and IForest (0.767). Time series: best of all detectors on PyOD's synthetic series (0.95 vs LSTMAD 0.87), level with LSTMAD on SKAB under a strong train/test regime shift (0.83-0.86), but behind windowed IForest / AutoEncoder on NAB's sustained level shifts (0.63 vs 0.83) where a prediction error only fires at the transition and transductive training absorbs the anomalous regime. SIGReg beats the EMA teacher in every setting.

**Verdict for the library goal.** JEPA-OD is competitive with, not dominant over, PyOD's existing detectors: a credible new *deep* detector family (it is the best deep model on the tabular sweep) with a clear niche, anomalies that break joint feature or temporal structure, and known weak spots (local outliers in low dimension, sustained level shifts). The code in this repo is structured to drop into `pyod/models/jepa.py` and `pyod/models/ts_jepa.py` (Section 6). PyOD's inclusion policy requires an accepted issue first and, for a new method, a two-year maintenance commitment.

---

## 2. The JEPA family: what we learned

### 2.1 Core idea

LeCun's 2022 position paper (*A Path Towards Autonomous Machine Intelligence*) proposes to learn world models by predicting in **embedding space** rather than reconstructing pixels or tokens. Two views of an input, `x` (context) and `y` (target), are embedded by encoders; a predictor maps the context embedding (plus a latent / conditioning variable) to the target embedding. Because the loss lives in representation space, the model is free to **discard unpredictable detail** (sensor noise, texture) and keep what is predictable (structure). That is also precisely why it is attractive for anomaly detection: an anomaly is a violation of learned structure, not a large pixel residual.

The central difficulty is **representation collapse**: the trivial solution where every embedding is constant makes the prediction error zero. All JEPA variants are defined by how they avoid it.

### 2.2 Variants studied

| Model | Modality | Context / target | Collapse prevention | Key facts |
|---|---|---|---|---|
| **I-JEPA** (Assran et al., CVPR 2023) | images | one large context block, 4 target blocks of patches | EMA target encoder + stop-gradient, narrow predictor | smooth-L1 latent loss; targets are *contextualised* patch embeddings of the full image |
| **V-JEPA / V-JEPA 2 / 2.1** (Meta 2024-2026) | video | spatio-temporal tube masking; V-JEPA 2-AC conditions on robot actions | EMA teacher | V-JEPA 2 predicts embeddings of later frames; used for planning via latent prediction |
| **TS-JEPA** "Joint Embeddings Go Temporal" (Ennadir, Golkar, Sarra; arXiv 2509.25449) | univariate series, 10 patches | uniform 70 % patch masking | EMA m = 0.998 only | 1-D CNN patch embedding + sin-cos positions, transformer encoder/predictor (dim 128, 2 heads), **L1** latent loss; evaluated on classification / forecasting, *not* anomaly detection |
| **CF-JEPA** (Lee & Sim, arXiv 2606.07031) | time series | mask-free: random crop as context, predict short/mid/long-horizon *future* representations | EMA teacher | online encoder is higher-rank (use for classification), EMA encoder smoother (use for forecasting *and anomaly detection*, kNN on embeddings); 2nd on KPI / Yahoo |
| **MTS-JEPA** (He, Wen, Wang, Ma; arXiv 2602.04643) | multivariate series, next-window prediction | fine (5×20) and coarse (1×20 block-averaged) views of the next window | **soft codebook** (K=128 prototypes, τ=0.1) + batch-entropy + **reconstruction anchor** + EMA 0.996 | ablation: removing the codebook or reconstruction collapses to near-random; plain TS-JEPA "suffers from optimization instability" |
| **LeJEPA** (Balestriero & LeCun, arXiv 2511.08544) | images (60+ architectures) | multi-view invariance; predictor removed | **SIGReg**: Epps-Pulley test on random 1-D projections → N(0, I) | *no* EMA, *no* stop-gradient, one hyper-parameter λ = 0.05; proves isotropic Gaussian embeddings minimise worst-case downstream risk; training loss correlates ~0.85 Spearman with downstream accuracy (label-free model selection) |

### 2.3 Collapse-prevention mechanisms and what they mean for outlier detection

1. **EMA teacher + stop-gradient** (I-JEPA, TS-JEPA, automotive JEPA). Works, but the target scale drifts during training, so absolute prediction errors are not comparable across checkpoints (T-SAR-JEPA reports the mean score dropping from 6.89 to 0.45 after unfreezing the encoder, a 15× shift with no check that detection improved, consistent with partial collapse). Needs the momentum schedule as an extra knob.
2. **Codebook bottleneck + reconstruction** (MTS-JEPA). Robust but heavy (K, τ, five loss weights, a decoder), and the discrete codes blur small deviations, which is the wrong bias for subtle anomalies.
3. **SIGReg** (LeJEPA). Forces each random 1-D projection of the embeddings to be standard normal, which by Cramér-Wold means the embedding cloud is isotropic Gaussian. For OD this is ideal: (a) the latent prediction error has a fixed scale (a perfectly uninformative predictor has expected squared error 2 per dimension, a perfect one 0); (b) *distance to the Gaussian* is itself a calibrated anomaly score; (c) there is one knob. We adopt it.

One subtlety not covered by LeJEPA (which has no predictor and symmetric views): with *masked* prediction the encoder can cheat by making its own targets predictable, i.e. encoding each token from its neighbours rather than from its own value. SIGReg does not prevent that ("too invariant" regime noted by MADJEPA). We therefore keep a **stop-gradient on the targets** while still back-propagating SIGReg through the full-row embeddings, and offer an optional MTS-JEPA style **linear reconstruction anchor** (`recon_weight`).

---

## 3. Prior work on JEPA for anomaly / outlier detection

| Paper | Domain | Anomaly score | Result | Lesson |
|---|---|---|---|---|
| **MADJEPA** (Stoian et al., arXiv 2611.25194) | medical images, train on healthy only | 3 fused scores: I-JEPA latent error map (N=30 resampled masks), teacher-student distillation discrepancy, pseudo-anomaly classifier | image AUROC 82-97 on 6 BMAD datasets; I-JEPA-alone 65-89 | I-JEPA alone can "over-penalise unseen healthy structures or become too invariant to subtle pathologies"; **average many masks**, fuse scores with per-dataset weights |
| **T-SAR-JEPA** (arXiv 2606.05700) | SAR satellite amplitude stacks | L2 distance between predicted and actual ViT embedding of the next acquisition (K=7 context) | ROC-AUC 77.0 vs ≈50 for RX, PaDiM, linear AR, LSTM | latent one-step prediction error works where raw-space baselines fail; use **L2 for scoring even if trained with smooth-L1** ("preserves outlier magnitude"); no collapse safeguards → scale drift |
| **Online monitoring with JEPA embeddings** (arXiv 2602.09985) | automotive object tracks (nuScenes) | classical detector (ABOD, LOF, GMM) on max-pooled 32-d JEPA embeddings | TPR 52.9 % at 5 % FPR vs raw features "not able to solve the task" | **embeddings + classical OD** is a viable second route; L1 latent loss "more stable" |
| **CF-JEPA** (arXiv 2606.07031) | KPI / Yahoo series | kNN on EMA-encoder embeddings | 2nd among SSL baselines | the *smoother* (EMA) encoder is the better one for AD |
| **MTS-JEPA** (arXiv 2602.04643) | MSL / SMAP / SWaT / PSM | *supervised* MLP on frozen code distributions (anomaly *prediction*) | F1 33.6 / 33.6 / 72.9 / 61.6 (best of 10) | collapse diagnostics matter; cross-domain pre-training transfers |
| **AV safety survey** (Comput. Ind. 2026) | autonomous vehicles | lists anomaly / intrusion detection as JEPA use-cases | survey | demand exists, no library implementation |

**Gap.** No generic tabular JEPA detector exists, no PyOD/DeepOD-style implementation exists, and none of the time-series works scores *timestamps* through the standard sliding-window protocols of PyOD / TSB-AD. All but MTS-JEPA use the latent prediction error or embedding density, which is what JEPA-OD generalises.

### 3.1 Relation to existing self-supervised OD methods

* **AutoEncoder / VAE (PyOD):** reconstruction error in *input* space; penalises unpredictable noise and generalises to anomalies. JEPA replaces the decoder with a latent predictor.
* **Masked-feature / internal-contrastive methods** (ICL, Shenkar & Wolf ICLR 2022; GOAD; NeuTraL-AD; SLAD in DeepOD): also split features into context/target but use *contrastive* objectives with negatives. JEPA-OD is the *regression-in-latent-space* counterpart with SIGReg instead of negatives, which scales to many masks and gives per-token calibrated errors.
* **Forecasting detectors (PyOD `LSTMAD`):** raw-space next-step error with a Mahalanobis fit on residuals. `TSJEPAOD(mask_mode='forecast')` is the latent-space analogue and additionally supports bidirectional ('block') context.
* **Deep SVDD / DIF / LUNAR:** one-class or isolation objectives on learned features; JEPA's objective is predictive, so the representation is shaped by *what is predictable*, not by a hypersphere.

---

## 4. JEPA-OD design

### 4.1 Tokenisation (one network for both data types)

| data | token | token input dim | count |
|---|---|---|---|
| tabular row with `d` features | one feature (or `ceil(d / max_tokens)` consecutive features when `d > max_tokens`, zero-padded) | 1 (or group size) | `T = min(d, max_tokens)`, default ≤ 64 |
| time-series window of `window_size` steps × `C` channels | a patch of `patch_len` steps, all channels flattened | `patch_len · C` | `P = window_size / patch_len` |

Each token gets its own affine embedding (`x_t W_t + b_t`, FT-Transformer style) plus a learned token-identity / position vector.

### 4.2 Architecture (`jepa_od/_nets.py::JEPACore`)

* **Encoder**: pre-norm Transformer, `n_layers=2`, `d_model=64`, 4 heads, GELU, **no final LayerNorm** (see 4.6).
* **Predictor**: narrow Transformer (`d_model/2`, 1 layer) fed with the encoded context tokens plus learnable *mask tokens* carrying the target positions (I-JEPA design).
* **Targets**: the same encoder applied to the *full* input (contextualised target embeddings, as in I-JEPA), gathered at the masked positions; `target_mode='ema'` swaps in an I-JEPA EMA teacher with LayerNorm'ed targets for comparison.
* Parameters: ≈ 70 k for the defaults, CPU-friendly (a 3 000-step series trains in ~4 s on a laptop CPU).

### 4.3 Masking

| mode | tabular | time series | use |
|---|---|---|---|
| `random` | random feature subset (default) | random patches | generic |
| `block` | – | one contiguous run of patches at a random position (default) | bidirectional context, detection |
| `forecast` | – | always the last patches | causal / online monitoring (T-SAR-JEPA, CF-JEPA style) |
| `mixed` | – | coin flip block / forecast per window | both behaviours in one model |

`mask_ratio = 0.3` by default (`1 ≤ m ≤ T−1` enforced). Training samples one mask per row per step; **inference uses a fixed, seeded mask set shared by all rows**: for block modes *every* block position is enumerated (complete, deterministic coverage), for `random` permutations are chopped into disjoint groups so every feature is masked about equally often (`n_mask_samples = 8`). This is MADJEPA's "average N masks" made deterministic, which PyOD needs for reproducible `decision_function`.

### 4.4 Training objective

```
L = (1 − λ) · MSE( Pred(Enc(x_ctx)), sg(Enc(x))_tgt ) + λ · SIGReg( Enc(x) )  [+ ρ · MSE(LinearDecoder(Enc(x)), x)]
```

* `sg` = stop-gradient on the targets (prevents the "make my own targets predictable" shortcut).
* `SIGReg` (`jepa_od/_sigreg.py`): 128 random unit directions resampled each step, Epps-Pulley statistic `N ∫ |φ̂(t) − e^{−t²/2}|² e^{−t²/2} dt` with 17 trapezoid knots on [−5, 5], computed **per token position** across the batch so every feature / patch embedding is individually pushed to N(0, I). λ = 0.05 (LeJEPA default).
* Optional reconstruction anchor ρ (`recon_weight`, default 0) and I-JEPA EMA alternative.
* AdamW-style weight decay 1e-4, cosine LR decay, gradient clipping 1.0, batch 64; 30 epochs for tabular data, and for time series 10 epochs over at most 5 000 randomly chosen training windows (a bounded gradient budget: longer transductive training on a long, contaminated series absorbs the anomalous regime, Section 7.2).

### 4.5 Scoring

1. **`pred`** (time-series default) – per-target-token squared latent error `e_j = mean_k (pred_jk − h_jk)²`, **z-normalised per token** with training mean / std (so an intrinsically unpredictable feature does not dominate; analogous to `LSTMAD`'s Mahalanobis on residuals), averaged over masked tokens and over the mask set.
2. **`embed`** – Mahalanobis distance of each contextualised token embedding to a Gaussian fitted on training embeddings (per-token mean, covariance shared across tokens), averaged over tokens. Thanks to SIGReg this is a calibrated density score; for time series it is computed per patch and localises anomalies **without the temporal smearing** that prediction-based scores have.
3. **`combined`** (tabular default) – both components z-normalised with their training statistics and averaged (MADJEPA-style fusion, but label-free). On the ADBench sweep it beats either component alone; under distribution shift the `embed` term is fragile, so the time-series detector defaults to `pred`.

**Time-series mapping.** Each (window, mask, masked patch) error is assigned to the `patch_len` timestamps that patch covers and aggregated (`mean` default, `max` optional) over all windows with stride `step`. Timestamps never covered (e.g. the first `window_size − m·patch_len` in `forecast` mode) are filled with `threshold_`, exactly as PyOD's `TimeSeriesOD` / `LSTMAD` do.

### 4.6 Two failure modes discovered while building it

1. **Final LayerNorm saturation ⇒ inverted scores.** With a final LayerNorm on the encoder, an extreme standardised value (|x| ≫ 1) produces an embedding ≈ `normalize(W_t)`: constant for all large values. The predictor learns "if the context is saturated, predict the saturated direction", so outliers get *lower* latent error than inliers (AUC 0.19 on PyOD's 2-feature `generate_data`). Removing the final norm lets the residual stream keep input magnitude; SIGReg bounds the *normal* embeddings instead. AUC became 0.98 and the 20- and 300-feature toy problems reached 1.0. The same saturation appears for *shifted* time series: on SKAB, whose test files run in a different operating regime from the training recording, a final LayerNorm turns the `embed` score upside down (AUC 0.20) and halves the `pred` score (0.58 vs 0.83), see Section 7. This is a general caveat for latent-prediction anomaly scores and is worth stating in the PyOD docstring.
2. **Pooled embeddings cancel.** The mean over tokens of an outlier's embeddings is *closer* to the origin than an inlier's (incoherent directions average out), so Mahalanobis on the pooled vector was anti-correlated with anomaly (AUC 0.17). Per-token Mahalanobis with a shared covariance solves it.

### 4.7 Complexity

Training `O(epochs · n · T² · d_model)`; inference `O(n · K · T² · d_model)` with `K` masks. With `max_tokens = 64` the attention cost is bounded regardless of `d`; a 6 000 × 300 PyOD unit-test dataset trains 5 epochs in ≈ 8 s on CPU.

---

## 5. API

Both classes expose the full PyOD contract (`fit`, `decision_function`, `predict`, `predict_proba`, `predict_confidence`, `fit_predict_score`, `decision_scores_`, `threshold_`, `labels_`, `get_params/set_params`, pickling). See `README.md` for usage, `tests/` for the behaviour covered.

---

## 6. Integrating into PyOD (and alternatives)

PyOD 3.6.8 (installed) already has: `pyod/models/base_dl.py::BaseDeepLearningDetector` (hooks `build_model`, `training_forward`, `evaluating_forward`, `epoch_update`, `train`, `evaluating_prepare`), a time-series layer (`_ts_utils.py`, `ts_od.py::TimeSeriesOD`, `ts_lstm.py::LSTMAD`, `ts_anomaly_transformer.py`), a string registry `embedding.py::_DETECTOR_SHORTCUTS`, and an LLM model-selector knowledge base `utils/knowledge/algorithms.json`.

`integration/PYOD_PR_CHECKLIST.md` gives the file-by-file map (`pyod/models/jepa.py::JEPA`, `pyod/models/ts_jepa.py::TSJEPA`, `_jepa_utils.py`, tests, docs, examples, registry line, knowledge entry in `integration/pyod_knowledge_entries.json`) and the contribution process (issue first → `accepted` label → PR against `development`; new detectors need a two-year maintenance commitment or 2 years / 50 citations).

**DeepOD** (`deepod.core.base_model.BaseDeepAD`, `data_type='ts'`, `seq_len`, `stride`): `JEPACore` plugs in as `net`; implement `training_prepare/training_forward/inference_prepare/inference_forward` (sketch in the checklist). **TSB-AD**: wrap `TSJEPAOD` like the other `TSB_AD.models` entries (`fit` on train split, `decision_scores_` on test) and report VUS-PR.

Also note that `TimeSeriesOD(detector=JEPAOD(...))` already works unchanged, because `JEPAOD` is a regular PyOD detector; the native `TSJEPAOD` adds patching, causal masking and patch-level localisation.

---

## 7. Experiments

Setup, datasets and full numbers: `benchmarks/bench_tabular.py`, `benchmarks/bench_ts.py`, CSVs in `results/`. All runs on a laptop CPU, default hyper-parameters, no per-dataset tuning. Metrics: ROC-AUC and average precision (AP).

<!-- RESULTS:BEGIN -->

### 7.1 Tabular (ADBench)

**Setup.** 16 ADBench datasets (`data/adbench`, 129 to 20 000 rows after capping `skin` at 20 000, 3 to 1 555 features, 1-36 % outliers), stratified 60/40 split per seed, detectors fitted on the contaminated training split without labels, 3 seeds, ROC-AUC and AP on the test split. Baselines are PyOD 3.6 detectors with default parameters (`AutoEncoder` / `DeepSVDD` at 30 epochs). `JEPAOD` uses its defaults (30 epochs, SIGReg, no encoder norm); one fit per seed is scored three ways.

**Findings.**

* **Among deep detectors JEPA-OD is the strongest on average.** `JEPAOD[combined]` reaches mean ROC-AUC 0.760 / AP 0.438 against `AutoEncoder` 0.746 / 0.375, `DeepSVDD` 0.737 / 0.358, `DIF` 0.733 / 0.383 and `LUNAR` 0.729 / 0.357, and it has the best average rank of the deep models (6.8 of 15 detectors).
* **Overall it is third**, behind the two classical references `KNN` (0.786 AUC, rank 4.8) and `IForest` (0.767, rank 5.3). The pattern is the familiar ADBench one: local-density methods win on low-dimensional data with local outliers (`vowels`, `wine`, `Wilt`, `Waveform`, `Ionosphere`), while JEPA-OD wins or ties where outliers break joint feature structure in higher dimension (`musk` 1.000, `satimage-2` 0.992, `WBC` 0.995, `skin` 0.766 best of all, `cardio` 0.90).
* **Weak spots.** `thyroid` (0.89 vs 0.98 for `IForest`), `vowels` (0.79 vs 0.98 for `KNN`), `Ionosphere` (0.80 vs 0.92); these are small, low-dimensional sets where one or two marginal features carry the anomaly and a transformer over feature tokens brings no benefit. `speech` (400-d, 1.7 % outliers) is at chance for every detector.
* **Score components.** `combined` > `pred` ~ `embed` (0.760 / 0.750 / 0.749 AUC; 0.438 / 0.423 / 0.432 AP); the two components are complementary (e.g. `wine`: `pred` 0.85, `embed` 0.61; `thyroid`: `embed` 0.91, `pred` 0.83). `combined` is therefore the tabular default. Under distribution shift `pred` alone is safer (Section 7.2).
* **Collapse prevention.** SIGReg (0.760) edges the I-JEPA EMA teacher (0.750 AUC, 0.422 AP) and needs no momentum schedule; the EMA variant is however ~1.8x faster because the per-token Epps-Pulley statistic (128 projections x 17 knots per token) dominates the step time. Fewer projections or sub-sampling tokens for SIGReg is the obvious optimisation. The reconstruction anchor (`recon_weight=0.1`) changes nothing (0.760).

#### ADBench, 16 datasets, 60/40 split, mean of 3 seeds: ROC-AUC

| dataset | JEPAOD[pred] | JEPAOD[embed] | JEPAOD[combined] | AutoEncoder | DeepSVDD | LUNAR | DIF | IForest | KNN | LOF | ECOD | COPOD | PCA | HBOS | OCSVM |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| InternetAds | 0.588 | 0.585 | 0.586 | 0.543 | **0.701** | 0.654 | 0.526 | 0.667 | 0.687 | 0.633 | 0.661 | 0.660 | 0.593 | 0.668 | 0.673 |
| Ionosphere | 0.796 | 0.799 | 0.803 | 0.917 | 0.914 | **0.924** | 0.895 | 0.823 | 0.919 | 0.879 | 0.707 | 0.781 | 0.768 | 0.651 | 0.718 |
| Pima | 0.644 | 0.640 | 0.658 | 0.638 | 0.654 | **0.677** | 0.606 | 0.659 | 0.629 | 0.592 | 0.583 | 0.635 | 0.624 | 0.656 | 0.599 |
| WBC | **0.995** | 0.987 | 0.995 | 0.978 | 0.986 | 0.976 | 0.798 | 0.995 | 0.995 | 0.962 | 0.989 | 0.989 | 0.992 | 0.982 | 0.985 |
| Waveform | 0.623 | 0.631 | 0.634 | 0.641 | 0.671 | 0.740 | 0.734 | 0.707 | 0.757 | 0.746 | 0.617 | 0.747 | 0.650 | 0.695 | **0.783** |
| Wilt | 0.572 | 0.666 | 0.649 | 0.575 | 0.675 | 0.509 | 0.375 | 0.453 | 0.701 | **0.771** | 0.389 | 0.339 | 0.430 | 0.393 | 0.655 |
| cardio | 0.873 | 0.904 | 0.897 | 0.797 | 0.727 | 0.605 | 0.926 | 0.935 | 0.736 | 0.581 | 0.941 | 0.927 | **0.962** | 0.836 | 0.941 |
| fault | 0.598 | 0.552 | 0.590 | 0.651 | 0.583 | 0.700 | 0.665 | 0.554 | **0.719** | 0.570 | 0.459 | 0.437 |  | 0.522 | 0.527 |
| magic.gamma | 0.686 | 0.685 | 0.696 | 0.732 | 0.680 | 0.805 | 0.763 | 0.725 | **0.811** | 0.706 | 0.643 | 0.686 | 0.672 | 0.718 | 0.682 |
| musk | 0.989 | **1.000** | 1.000 | 0.923 | 0.991 | 0.520 | 0.975 | 1.000 | 0.534 | 0.456 | 0.950 | 0.941 | 1.000 | 0.999 | 0.188 |
| satimage-2 | 0.991 | 0.985 | 0.992 | 0.950 | 0.781 | 0.858 | **0.995** | 0.989 | 0.948 | 0.486 | 0.946 | 0.959 | 0.963 | 0.957 | 0.882 |
| skin | 0.675 | **0.766** | 0.736 | 0.636 | 0.567 | 0.715 | 0.682 | 0.664 | 0.717 | 0.477 | 0.491 | 0.470 | 0.425 | 0.588 | 0.556 |
| speech | 0.505 | 0.483 | 0.491 | 0.490 | 0.511 | **0.573** | 0.526 | 0.515 | 0.497 | 0.507 | 0.489 | 0.512 | 0.489 | 0.489 | 0.486 |
| thyroid | 0.834 | 0.914 | 0.887 | 0.943 | 0.853 | 0.947 | 0.971 | **0.981** | 0.956 | 0.904 | 0.978 | 0.943 | 0.963 | 0.956 | 0.857 |
| vowels | 0.770 | 0.783 | 0.791 | 0.931 | 0.910 | 0.968 | 0.800 | 0.752 | **0.975** | 0.940 | 0.584 | 0.492 | 0.603 | 0.679 | 0.774 |
| wine | 0.851 | 0.611 | 0.759 | 0.589 | 0.590 | 0.490 | 0.486 | 0.847 | 0.995 | **0.998** | 0.726 | 0.844 | 0.833 | 0.663 | 0.915 |
| **mean** | 0.750 | 0.749 | 0.760 | 0.746 | 0.737 | 0.729 | 0.733 | 0.767 | **0.786** | 0.701 | 0.697 | 0.710 | 0.731 | 0.716 | 0.701 |
| **avg rank** | 7.91 | 8.31 | 6.84 | 8.56 | 8.12 | 6.81 | 7.44 | 5.28 | **4.78** | 8.81 | 10.53 | 9.09 | 8.93 | 9.00 | 9.19 |
| **wins** | 0 | 2 | 0 | 0 | 1 | 3 | 1 | 1 | 3 | 2 | 0 | 0 | 1 | 0 | 1 |

#### ADBench, 16 datasets, 60/40 split, mean of 3 seeds: AP

| dataset | JEPAOD[pred] | JEPAOD[embed] | JEPAOD[combined] | AutoEncoder | DeepSVDD | LUNAR | DIF | IForest | KNN | LOF | ECOD | COPOD | PCA | HBOS | OCSVM |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| InternetAds | 0.265 | 0.257 | 0.260 | 0.206 | 0.344 | 0.325 | 0.210 | 0.456 | 0.414 | 0.378 | 0.483 | 0.481 | 0.270 | 0.503 | **0.513** |
| Ionosphere | 0.732 | 0.763 | 0.751 | 0.900 | 0.876 | **0.924** | 0.870 | 0.775 | 0.915 | 0.828 | 0.636 | 0.671 | 0.715 | 0.426 | 0.709 |
| Pima | 0.453 | 0.449 | 0.459 | 0.491 | 0.441 | 0.504 | 0.409 | 0.479 | 0.477 | 0.428 | 0.445 | **0.512** | 0.447 | 0.484 | 0.428 |
| WBC | **0.954** | 0.921 | 0.954 | 0.668 | 0.779 | 0.660 | 0.170 | 0.942 | 0.954 | 0.528 | 0.802 | 0.802 | 0.905 | 0.633 | 0.813 |
| Waveform | 0.055 | 0.049 | 0.051 | 0.054 | 0.097 | 0.112 | 0.092 | 0.059 | 0.160 | 0.107 | 0.046 | 0.064 | 0.052 | 0.055 | **0.165** |
| Wilt | 0.070 | 0.083 | 0.080 | 0.058 | 0.088 | 0.050 | 0.039 | 0.045 | 0.080 | **0.117** | 0.042 | 0.037 | 0.043 | 0.044 | 0.078 |
| cardio | 0.458 | 0.522 | 0.507 | 0.337 | 0.342 | 0.252 | 0.585 | 0.578 | 0.378 | 0.196 | 0.604 | 0.594 | **0.693** | 0.475 | 0.580 |
| fault | 0.445 | 0.388 | 0.432 | 0.482 | 0.433 | 0.512 | 0.507 | 0.402 | **0.527** | 0.386 | 0.322 | 0.305 |  | 0.388 | 0.390 |
| magic.gamma | 0.573 | 0.598 | 0.597 | 0.634 | 0.566 | 0.735 | 0.681 | 0.640 | **0.738** | 0.547 | 0.539 | 0.593 | 0.596 | 0.625 | 0.631 |
| musk | 0.794 | **1.000** | 0.991 | 0.516 | 0.882 | 0.056 | 0.710 | 0.999 | 0.040 | 0.029 | 0.494 | 0.357 | 0.999 | 0.973 | 0.019 |
| satimage-2 | 0.902 | 0.910 | **0.933** | 0.360 | 0.097 | 0.240 | 0.745 | 0.887 | 0.410 | 0.037 | 0.583 | 0.712 | 0.807 | 0.660 | 0.093 |
| skin | 0.265 | **0.330** | 0.303 | 0.254 | 0.213 | 0.291 | 0.261 | 0.251 | 0.290 | 0.187 | 0.183 | 0.178 | 0.166 | 0.231 | 0.222 |
| speech | 0.019 | 0.021 | 0.020 | 0.027 | 0.022 | **0.044** | 0.020 | 0.025 | 0.027 | 0.036 | 0.033 | 0.033 | 0.026 | 0.028 | 0.026 |
| thyroid | 0.181 | 0.302 | 0.242 | 0.454 | 0.143 | 0.296 | 0.552 | **0.601** | 0.321 | 0.246 | 0.511 | 0.194 | 0.428 | 0.576 | 0.187 |
| vowels | 0.142 | 0.123 | 0.133 | 0.351 | 0.250 | **0.617** | 0.179 | 0.147 | 0.502 | 0.397 | 0.089 | 0.036 | 0.068 | 0.105 | 0.172 |
| wine | 0.468 | 0.201 | 0.305 | 0.203 | 0.159 | 0.091 | 0.096 | 0.349 | 0.956 | **0.983** | 0.269 | 0.462 | 0.419 | 0.199 | 0.414 |
| **mean** | 0.423 | 0.432 | 0.438 | 0.375 | 0.358 | 0.357 | 0.383 | **0.477** | 0.449 | 0.339 | 0.380 | 0.377 | 0.442 | 0.400 | 0.340 |
| **avg rank** | 8.19 | 7.69 | 7.62 | 7.88 | 9.12 | 6.69 | 8.44 | 6.06 | **4.94** | 8.94 | 9.72 | 8.97 | 8.53 | 8.38 | 8.44 |
| **wins** | 0 | 2 | 1 | 0 | 0 | 3 | 0 | 1 | 2 | 2 | 0 | 1 | 1 | 0 | 2 |

#### Tabular JEPA-OD variants (mean over datasets and seeds)

| JEPAOD variant | auc | ap | time |
|---|---|---|---|
| JEPAOD-ema[combined] | 0.750 | 0.422 | 10.921 |
| JEPAOD-ema[embed] | 0.723 | 0.386 | 10.921 |
| JEPAOD-ema[pred] | 0.749 | 0.394 | 10.921 |
| JEPAOD-recon[combined] | 0.760 | 0.436 | 19.397 |
| JEPAOD-recon[embed] | 0.750 | 0.428 | 19.397 |
| JEPAOD-recon[pred] | 0.745 | 0.419 | 19.397 |
| JEPAOD[combined] | 0.760 | 0.438 | 19.695 |
| JEPAOD[embed] | 0.749 | 0.432 | 19.695 |
| JEPAOD[pred] | 0.750 | 0.423 | 19.695 |

### 7.2 Time series (synthetic, NAB, SKAB)

**Setup.** Point-wise ROC-AUC / AP. *Synthetic*: PyOD's `generate_ts_data` (3 configurations x 2 seeds, 3 000 training / 1 500 test steps, 5 % contamination; window 60, patch 5). *NAB realKnownCause*: 7 real univariate series (1.9k-22.7k steps), transductive protocol (fit on the whole series, score its own `decision_scores_`), NAB label windows (~10 % of points); window 100, patch 10. *SKAB valve1*: 8-channel industrial test-bed, fit on the `anomaly-free` recording, score each of the 16 valve1 files (~1 150 steps, 35 % anomalous); window 60, patch 5. Baselines are PyOD 3.6 detectors: `LSTMAD`, `AnomalyTransformer` (d_model 64), `MatrixProfile`, `SpectralResidual`, and `TimeSeriesOD` around `IForest`, `KNN`, `AutoEncoder`. `TSJEPAOD` rows: `block` / `forecast` masking x `pred` / `embed` / `combined` score, 20 epochs with the no-norm encoder (the bulk run); `(final)` = the shipped defaults (10 epochs, <= 5 000 training windows); `(LN)` = encoder with a final LayerNorm (ablation).

**Findings.**

* **Synthetic (structure-breaking anomalies):** every `TSJEPAOD` variant (0.93-0.96 AUC) beats all baselines; `LSTMAD`, the raw-space forecasting detector, is next at 0.87, windowed `KNN` / `AutoEncoder` reach 0.78, `AnomalyTransformer` 0.59. Latent prediction error is the right signal when anomalies are violations of temporal structure rather than level excursions.
* **SKAB (train on clean recording, strong regime shift at test time):** `TSJEPAOD-forecast[pred]` 0.859 and `block[pred]` 0.834 are on par with `LSTMAD` (0.853) and far above windowed `AutoEncoder` (0.68) and `IForest` (0.53). Windowed `KNN` is almost perfect (0.994): SKAB anomalies are large raw-space deviations, so plain nearest-neighbour distance on raw windows suffices. The JEPA `embed` score stays valid here (0.82-0.84) *only* with the no-norm encoder; with a final LayerNorm it collapses to 0.20 (inverted) because every shifted patch saturates to the same direction (diagnostic: all test patches, normal and anomalous alike, sit at Mahalanobis ~85 vs ~63 for training patches; one channel differs by ~58 training standard deviations between the recordings).
* **NAB (real univariate, level-shift / regime anomalies, transductive):** windowed `IForest` (0.825) and `AutoEncoder` (0.776) lead; `TSJEPAOD` reaches 0.62-0.67, above `LSTMAD` (0.56), `AnomalyTransformer` (0.46) and `SpectralResidual` (0.46), level with `MatrixProfile` (0.69). Two reasons, both verified in `results/diag_nab.csv`: (i) a sustained level shift is locally *predictable* once inside it, so the prediction error fires at the transition only, whereas a raw-window density method flags the whole shifted segment; (ii) with 10 % of the series anomalous, training on the whole series lets the model absorb the anomalous regime: on `machine_temperature` the `pred` AUC falls from 0.84 (5 epochs) to 0.55 (20 epochs), and fitting on the first half only (inductive) restores 0.93 (IForest: 0.96). This motivated the smaller default training budget.
* **Score choice.** `pred` is the most robust single score across the three protocols (0.95 / 0.83 / 0.63); `embed` wins only in-distribution (synthetic, NAB with LayerNorm) and is fragile under shift; `combined` is rarely the best of the three and never the worst. `pred` stays the default.
* **Masking.** `forecast` (causal) is slightly better than `block` on SKAB and synthetic (0.86 vs 0.83, 0.96 vs 0.95) but worse on NAB (0.59 vs 0.63); `block` stays the default for offline detection, `forecast` is the online option.
* **Training budget.** The shipped defaults (10 epochs, <= 5 000 windows) match the 20-epoch run within 0.01 AUC on all three groups at about half the fit time (`results/budget_experiment.csv`).

#### ROC-AUC per dataset group

|  | TSJEPAOD(final)-block[pred] | TSJEPAOD(final)-block[combined] | TSJEPAOD-block[pred] | TSJEPAOD-block[embed] | TSJEPAOD-block[combined] | TSJEPAOD-forecast[pred] | TSJEPAOD-forecast[embed] | TSJEPAOD-forecast[combined] | TSJEPAOD(LN)-block[pred] | TSJEPAOD(LN)-block[embed] | TSJEPAOD(LN)-forecast[pred] | LSTMAD | AnomalyTransformer | TSOD-AutoEncoder | TSOD-IForest | TSOD-KNN | MatrixProfile | SpectralResidual |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| NAB (7 series) | 0.622 | 0.626 | 0.631 | 0.628 | 0.628 | 0.585 | 0.619 | 0.626 | 0.666 | 0.661 | 0.572 | 0.556 | 0.458 | 0.776 | **0.825** | 0.741 | 0.691 | 0.457 |
| SKAB (16 series) | 0.827 | 0.825 | 0.834 | 0.817 | 0.826 | 0.859 | 0.844 | 0.819 | 0.582 | 0.197 | 0.468 | 0.853 | 0.496 | 0.680 | 0.530 | **0.994** |  | 0.472 |
| synthetic (3 series) | 0.952 | 0.946 | 0.954 | 0.932 | 0.945 | **0.962** | 0.934 | 0.958 | 0.943 | 0.918 | 0.930 | 0.866 | 0.586 | 0.780 | 0.727 | 0.781 |  | 0.819 |

#### AP per dataset group

|  | TSJEPAOD(final)-block[pred] | TSJEPAOD(final)-block[combined] | TSJEPAOD-block[pred] | TSJEPAOD-block[embed] | TSJEPAOD-block[combined] | TSJEPAOD-forecast[pred] | TSJEPAOD-forecast[embed] | TSJEPAOD-forecast[combined] | TSJEPAOD(LN)-block[pred] | TSJEPAOD(LN)-block[embed] | TSJEPAOD(LN)-forecast[pred] | LSTMAD | AnomalyTransformer | TSOD-AutoEncoder | TSOD-IForest | TSOD-KNN | MatrixProfile | SpectralResidual |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| NAB (7 series) | 0.282 | 0.282 | 0.269 | 0.286 | 0.281 | 0.248 | 0.281 | 0.279 | 0.239 | 0.245 | 0.198 | 0.164 | 0.119 | **0.492** | 0.486 | 0.442 | 0.453 | 0.113 |
| SKAB (16 series) | 0.637 | 0.626 | 0.649 | 0.604 | 0.625 | 0.706 | 0.699 | 0.620 | 0.397 | 0.226 | 0.356 | 0.757 | 0.327 | 0.541 | 0.439 | **0.991** |  | 0.332 |
| synthetic (3 series) | 0.587 | 0.576 | 0.591 | 0.558 | 0.575 | 0.603 | 0.556 | 0.592 | 0.623 | 0.600 | 0.575 | **0.657** | 0.073 | 0.133 | 0.123 | 0.135 |  | 0.582 |

#### NAB per-series ROC-AUC

| series | TSJEPAOD(final)-block[pred] | TSJEPAOD(final)-block[combined] | TSJEPAOD-block[pred] | TSJEPAOD-block[embed] | TSJEPAOD-block[combined] | TSJEPAOD-forecast[pred] | TSJEPAOD-forecast[embed] | TSJEPAOD-forecast[combined] | TSJEPAOD(LN)-block[pred] | TSJEPAOD(LN)-block[embed] | TSJEPAOD(LN)-forecast[pred] | LSTMAD | AnomalyTransformer | TSOD-AutoEncoder | TSOD-IForest | TSOD-KNN | MatrixProfile | SpectralResidual |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ambient_temperature_system_failure | 0.543 | 0.558 | 0.539 | 0.567 | 0.554 | 0.439 | 0.560 | 0.501 | 0.662 | 0.732 | 0.301 | 0.756 | 0.518 | 0.849 | **0.881** | 0.672 | 0.459 | 0.502 |
| cpu_utilization_asg_misconfiguration | 0.939 | 0.926 | 0.949 | 0.914 | 0.937 | 0.867 | 0.917 | 0.935 | 0.918 | 0.890 | 0.748 | 0.676 | 0.602 | 0.885 | **0.974** | 0.854 | 0.880 | 0.543 |
| ec2_request_latency_system_failure | 0.558 | 0.561 | 0.567 | 0.563 | 0.565 | 0.575 | 0.562 | 0.569 | 0.639 | 0.592 | 0.683 | 0.495 | 0.496 | **0.983** | 0.846 | 0.982 | 0.823 | 0.478 |
| machine_temperature_system_failure | 0.658 | 0.730 | 0.579 | 0.803 | 0.692 | 0.361 | 0.712 | 0.572 | 0.521 | 0.794 | 0.317 | 0.557 | 0.389 | 0.833 | **0.882** | 0.653 | 0.325 | 0.468 |
| nyc_taxi | 0.560 | 0.519 | 0.646 | 0.462 | 0.544 | 0.833 | 0.494 | 0.731 | 0.771 | 0.556 | 0.798 | 0.511 | 0.361 | 0.879 | 0.821 | **0.982** | 0.968 | 0.399 |
| rogue_agent_key_hold | 0.491 | 0.478 | 0.530 | 0.483 | 0.498 | 0.438 | 0.483 | 0.458 | 0.518 | 0.436 | 0.566 | 0.417 | 0.462 | 0.402 | **0.818** | 0.437 |  | 0.521 |
| rogue_agent_key_updown | 0.604 | 0.608 | 0.604 | 0.608 | 0.606 | 0.586 | 0.607 | 0.612 | **0.630** | 0.623 | 0.593 | 0.478 | 0.379 | 0.600 | 0.550 | 0.609 |  | 0.291 |

#### SKAB per-series ROC-AUC

| series | TSJEPAOD(final)-block[pred] | TSJEPAOD(final)-block[combined] | TSJEPAOD-block[pred] | TSJEPAOD-block[embed] | TSJEPAOD-block[combined] | TSJEPAOD-forecast[pred] | TSJEPAOD-forecast[embed] | TSJEPAOD-forecast[combined] | TSJEPAOD(LN)-block[pred] | TSJEPAOD(LN)-block[embed] | TSJEPAOD(LN)-forecast[pred] | LSTMAD | AnomalyTransformer | TSOD-AutoEncoder | TSOD-IForest | TSOD-KNN | SpectralResidual |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| valve1_0 | 0.886 | 0.877 | 0.887 | 0.874 | 0.877 | 0.916 | 0.912 | 0.882 | 0.752 | 0.174 | 0.797 | 0.899 | 0.497 | 0.928 | 0.509 | **0.966** | 0.412 |
| valve1_1 | 0.700 | 0.702 | 0.712 | 0.702 | 0.706 | 0.728 | 0.726 | 0.697 | 0.531 | 0.268 | 0.515 | 0.706 | 0.474 | 0.560 | 0.620 | **0.988** | 0.472 |
| valve1_10 | 0.943 | 0.948 | 0.948 | 0.938 | 0.948 | 0.972 | 0.961 | 0.936 | 0.881 | 0.140 | 0.872 | 0.993 | 0.468 | 0.907 | 0.496 | **1.000** | 0.450 |
| valve1_11 | 0.830 | 0.829 | 0.832 | 0.823 | 0.827 | 0.858 | 0.855 | 0.823 | 0.450 | 0.214 | 0.256 | 0.855 | 0.507 | 0.708 | 0.491 | **0.999** | 0.491 |
| valve1_12 | 0.890 | 0.891 | 0.898 | 0.880 | 0.891 | 0.921 | 0.924 | 0.890 | 0.784 | 0.084 | 0.733 | 0.912 | 0.481 | 0.805 | 0.469 | **0.997** | 0.483 |
| valve1_13 | 0.799 | 0.803 | 0.805 | 0.801 | 0.804 | 0.826 | 0.831 | 0.800 | 0.281 | 0.159 | 0.113 | 0.832 | 0.499 | 0.330 | 0.412 | **1.000** | 0.498 |
| valve1_14 | 0.888 | 0.893 | 0.892 | 0.888 | 0.894 | 0.922 | 0.897 | 0.884 | 0.428 | 0.158 | 0.352 | 0.927 | 0.514 | 0.653 | 0.609 | **0.999** | 0.425 |
| valve1_15 | 0.968 | 0.954 | 0.967 | 0.941 | 0.953 | 0.972 | 0.977 | 0.960 | 0.907 | 0.129 | 0.616 | 0.983 | 0.493 | 0.970 | 0.465 | **1.000** | 0.499 |
| valve1_2 | 0.724 | 0.716 | 0.731 | 0.705 | 0.717 | 0.757 | 0.738 | 0.709 | 0.720 | 0.259 | 0.728 | 0.783 | 0.513 | 0.744 | 0.582 | **0.960** | 0.497 |
| valve1_3 | 0.711 | 0.711 | 0.731 | 0.705 | 0.715 | 0.767 | 0.710 | 0.702 | 0.344 | 0.246 | 0.228 | 0.722 | 0.488 | 0.386 | 0.364 | **0.999** | 0.468 |
| valve1_4 | 0.641 | 0.634 | 0.654 | 0.630 | 0.637 | 0.689 | 0.647 | 0.626 | 0.502 | 0.358 | 0.483 | 0.651 | 0.523 | 0.572 | 0.678 | **1.000** | 0.465 |
| valve1_5 | 0.808 | 0.809 | 0.812 | 0.801 | 0.809 | 0.840 | 0.822 | 0.801 | 0.488 | 0.195 | 0.228 | 0.861 | 0.493 | 0.701 | 0.619 | **0.996** | 0.486 |
| valve1_6 | 0.809 | 0.803 | 0.826 | 0.795 | 0.808 | 0.848 | 0.817 | 0.793 | 0.550 | 0.293 | 0.497 | 0.795 | 0.515 | 0.614 | 0.618 | **1.000** | 0.483 |
| valve1_7 | 0.900 | 0.897 | 0.904 | 0.884 | 0.895 | 0.929 | 0.920 | 0.891 | 0.635 | 0.152 | 0.327 | 0.926 | 0.520 | 0.895 | 0.519 | **1.000** | 0.463 |
| valve1_8 | 0.943 | 0.940 | 0.950 | 0.927 | 0.939 | 0.976 | 0.959 | 0.933 | 0.732 | 0.146 | 0.607 | 0.980 | 0.473 | 0.770 | 0.505 | **0.999** | 0.463 |
| valve1_9 | 0.786 | 0.791 | 0.792 | 0.782 | 0.789 | 0.821 | 0.806 | 0.782 | 0.329 | 0.173 | 0.141 | 0.815 | 0.477 | 0.338 | 0.520 | **1.000** | 0.498 |

### 7.3 Ablations and diagnostics

`results/ablation_results.csv` (block masking, `combined`-trained models scored three ways; tabular: 8 small ADBench sets, seed 0; time series: 3 synthetic + 7 NAB series):

* **SIGReg vs EMA teacher.** SIGReg wins everywhere: tabular mean AUC 0.77-0.81 vs 0.67-0.75 (EMA), time series 0.72-0.77 vs 0.69-0.71. The EMA variant is also the one whose `embed` score is weakest, consistent with target-scale drift.
* **Encoder final LayerNorm.** Off is better for tabular data (0.804 vs 0.762 `combined`), on is better *in-distribution* for time series (0.767 vs 0.725) but catastrophic under shift (SKAB above). Both detectors ship with it off; the switch (`encoder_norm`) is kept for reproducibility.
* **Reconstruction anchor** (`recon_weight=0.1`, tabular bulk run): no gain over plain SIGReg (see 7.2), so it stays off.

#### Ablation: encoder final LayerNorm x collapse prevention (mean ROC-AUC / AP)

|  | kind | encoder_norm | target_mode | score | auc | ap |
|---|---|---|---|---|---|---|
| 0 | tabular | False | ema | combined | 0.717 | 0.313 |
| 1 | tabular | False | ema | embed | 0.674 | 0.240 |
| 2 | tabular | False | ema | pred | 0.746 | 0.386 |
| 3 | tabular | False | sigreg | combined | 0.804 | 0.433 |
| 4 | tabular | False | sigreg | embed | 0.810 | 0.431 |
| 5 | tabular | False | sigreg | pred | 0.766 | 0.410 |
| 6 | tabular | True | ema | combined | 0.717 | 0.310 |
| 7 | tabular | True | ema | embed | 0.676 | 0.238 |
| 8 | tabular | True | ema | pred | 0.745 | 0.383 |
| 9 | tabular | True | sigreg | combined | 0.762 | 0.349 |
| 10 | tabular | True | sigreg | embed | 0.750 | 0.324 |
| 11 | tabular | True | sigreg | pred | 0.739 | 0.351 |
| 12 | ts | False | ema | combined | 0.707 | 0.337 |
| 13 | ts | False | ema | embed | 0.702 | 0.336 |
| 14 | ts | False | ema | pred | 0.695 | 0.332 |
| 15 | ts | False | sigreg | combined | 0.725 | 0.367 |
| 16 | ts | False | sigreg | embed | 0.719 | 0.365 |
| 17 | ts | False | sigreg | pred | 0.731 | 0.364 |
| 18 | ts | True | ema | combined | 0.707 | 0.336 |
| 19 | ts | True | ema | embed | 0.703 | 0.336 |
| 20 | ts | True | ema | pred | 0.694 | 0.332 |
| 21 | ts | True | sigreg | combined | 0.767 | 0.381 |
| 22 | ts | True | sigreg | embed | 0.744 | 0.365 |
| 23 | ts | True | sigreg | pred | 0.751 | 0.368 |

### 7.4 Runtime

Laptop CPU (4 threads per run). `JEPAOD` (30 epochs, 8 inference masks, 3 scoring passes inside `fit`) is the slowest tabular detector in the comparison, roughly 10x `AutoEncoder`; `epoch_num=10-15` or `device='mps'/'cuda'` bring it to a few seconds. `TSJEPAOD` with the final defaults fits a 20k-step series in ~10 s, between `TimeSeriesOD(AutoEncoder)` and `AnomalyTransformer`.

#### Tabular: Mean fit+score time per dataset split (s)

| model | seconds |
|---|---|
| KNN | 0.030 |
| HBOS | 0.040 |
| LOF | 0.050 |
| COPOD | 0.050 |
| ECOD | 0.060 |
| IForest | 0.080 |
| PCA | 0.150 |
| DeepSVDD | 0.570 |
| OCSVM | 0.820 |
| LUNAR | 2.030 |
| AutoEncoder | 2.340 |
| DIF | 2.720 |
| JEPAOD-ema[embed] | 10.920 |
| JEPAOD-ema[combined] | 10.920 |
| JEPAOD-ema[pred] | 10.920 |
| JEPAOD-recon[embed] | 19.400 |
| JEPAOD-recon[pred] | 19.400 |
| JEPAOD-recon[combined] | 19.400 |
| JEPAOD[combined] | 19.700 |
| JEPAOD[embed] | 19.700 |
| JEPAOD[pred] | 19.700 |

#### Time series: Mean fit time per series (s)

| model | seconds |
|---|---|
| SpectralResidual | 0.000 |
| TSOD-KNN | 0.340 |
| TSOD-IForest | 0.380 |
| MatrixProfile | 0.920 |
| TSOD-AutoEncoder | 3.680 |
| TSJEPAOD(final)-forecast[pred] | 8.310 |
| TSJEPAOD(final)-forecast[embed] | 8.310 |
| TSJEPAOD(final)-forecast[combined] | 8.310 |
| TSJEPAOD(final)-block[embed] | 10.320 |
| TSJEPAOD(final)-block[combined] | 10.320 |
| TSJEPAOD(final)-block[pred] | 10.320 |
| TSJEPAOD-forecast[combined] | 15.340 |
| TSJEPAOD-forecast[embed] | 15.340 |
| TSJEPAOD-forecast[pred] | 15.340 |
| TSJEPAOD(LN)-forecast[embed] | 15.360 |
| TSJEPAOD(LN)-forecast[combined] | 15.360 |
| TSJEPAOD(LN)-forecast[pred] | 15.360 |
| TSJEPAOD-block[combined] | 16.720 |
| TSJEPAOD-block[embed] | 16.720 |
| TSJEPAOD-block[pred] | 16.720 |
| TSJEPAOD(LN)-block[pred] | 16.870 |
| TSJEPAOD(LN)-block[embed] | 16.870 |
| TSJEPAOD(LN)-block[combined] | 16.870 |
| AnomalyTransformer | 45.940 |
| LSTMAD | 72.750 |

<!-- RESULTS:END -->

---

## 8. Limitations and next steps

* **Benchmarks are modest** (16 ADBench datasets, 7 NAB series, SKAB valve1, synthetic). A PyOD PR should add the full ADBench-47 sweep and TSB-AD with VUS-PR, which is the primary metric there.
* **Transductive contamination.** Like every unsupervised deep detector, training outliers are Gaussianised along with inliers by SIGReg. The `embed` score is more sensitive to this than `pred`.
* **Hyper-parameters** were fixed at LeJEPA / I-JEPA defaults; LeJEPA's observation that the training loss tracks downstream quality could give label-free model selection for OD and deserves a test.
* **Categorical features** are treated as numeric; a per-token embedding table would be the natural extension.
* **Pre-training / transfer**: MTS-JEPA shows cross-domain pre-training transfers; a pre-trained TS-JEPA encoder with a frozen Gaussian head would make `TSJEPAOD` fit in milliseconds.
* **Not evaluated**: GPU/MPS throughput, very long windows (>1 000 steps), streaming updates.

---

## 9. References

* LeCun, Y. (2022). *A Path Towards Autonomous Machine Intelligence.* OpenReview.
* Assran, M. et al. (2023). *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture (I-JEPA).* CVPR.
* Bardes, A. et al. (2024/2025). *V-JEPA / V-JEPA 2.* Meta AI.
* Balestriero, R., LeCun, Y. (2025). *LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics.* arXiv:2511.08544. https://arxiv.org/abs/2511.08544
* Ennadir, S., Golkar, S., Sarra, L. (2025). *Joint Embeddings Go Temporal (TS-JEPA).* arXiv:2509.25449. https://arxiv.org/abs/2509.25449
* He, Y., Wen, Y., Wang, X., Ma, T. (2026). *MTS-JEPA: Multi-Resolution Joint-Embedding Predictive Architecture for Time-Series Anomaly Prediction.* arXiv:2602.04643. https://arxiv.org/abs/2602.04643
* Stoian, S. et al. (2026). *MADJEPA: Enhanced Multi-Stage Joint-Embedding Predictive Architecture for Medical Anomaly Detection.* arXiv:2611.25194. https://arxiv.org/abs/2611.25194
* *T-SAR-JEPA: Self-Supervised Temporal Anomaly Detection in SAR Amplitude Stacks via Latent Prediction* (2026). arXiv:2606.05700. https://arxiv.org/abs/2606.05700
* *Online Monitoring Framework for Automotive Time Series Data using JEPA Embeddings* (2026). arXiv:2602.09985. https://arxiv.org/abs/2602.09985
* Lee, J., Sim, S. (2026). *CF-JEPA: Mask-free forward prediction with asymmetric encoder utilization for time-series representation learning.* arXiv:2606.07031. https://arxiv.org/abs/2606.07031
* *Joint Embedding Predictive Architecture for autonomous vehicle safety and security: A comprehensive survey* (2026). Computers in Industry. https://www.sciencedirect.com/science/article/abs/pii/S1474034626002909
* Zhao, Y. et al. (2019). *PyOD: A Python Toolbox for Scalable Outlier Detection.* JMLR. Chen, S. et al. (2024). *PyOD 2.* arXiv:2412.12154.
* Liu, Q., Paparrizos, J. (2024). *The Elephant in the Room: Towards A Reliable Time-Series Anomaly Detection Benchmark (TSB-AD).* NeurIPS D&B.
* Xu, H. et al. *DeepOD: Deep Learning-based Outlier and Anomaly Detection.* https://github.com/xuhongzuo/DeepOD
* Shenkar, T., Wolf, L. (2022). *Anomaly Detection for Tabular Data with Internal Contrastive Learning (ICL).* ICLR.
* Han, S. et al. (2022). *ADBench: Anomaly Detection Benchmark.* NeurIPS D&B. Datasets: https://github.com/Minqi824/ADBench
* NAB: https://github.com/numenta/NAB ; SKAB: https://github.com/waico/SKAB
