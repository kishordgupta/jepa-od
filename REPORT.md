# JEPA-OD: Can Joint-Embedding Predictive Architectures Detect Outliers?

*Research study, design and prototype for a PyOD-compatible JEPA outlier detector (tabular and time series).*
*Date: 2026-11-20; updated 2026-11-21 with the T-JEPA family and an untrained control (Section 7.5), and with the full ADBench suite and the time-series control (Section 7.6), which revise the headline results, and with selectable JEPA versions for the time-series detector (Sections 4.8 and 7.7). Code: `jepa_od/`, benchmarks: `benchmarks/`, raw results: `results/`.*

---

## 1. Executive summary

**Question.** JEPA (Joint-Embedding Predictive Architecture) models learn by predicting the *representation* of a hidden part of the input from the visible part. Can that mechanism be turned into an unsupervised outlier detector that fits the PyOD API, for tabular rows and for time series?

**Answer.** It can be built and it works, but the full benchmark does not show that it is better than what PyOD already has, or that JEPA training is what makes it work (see the last bullets and the verdict). The natural score is the **latent prediction error**: a normal sample has internal structure (feature dependencies, temporal continuity) that the predictor has learned, so its hidden parts are predictable in representation space; an outlier breaks that structure. Four 2026 papers already use exactly this signal in narrow domains (medical images, SAR satellite stacks, automotive telemetry, multivariate telemetry), but none provides a generic, library-ready detector. We built one:

* **`JEPAOD`** (tabular) subclasses PyOD's `BaseDeepLearningDetector`; **`TSJEPAOD`** (time series) follows PyOD 3.6's time-series detector pattern (`LSTMAD`, `TimeSeriesOD`). A third class, **`TJEPAOD`**, trains the recipe of T-JEPA (Thimonier et al., ICLR 2025), the closest prior tabular JEPA, and scores rows with the same scores (Section 7.5). All three pass PyOD-style unit tests (65 tests), are deterministic, picklable and `sklearn.clone`-able.
* The architecture is a small *token transformer* (features or time patches are tokens) with an I-JEPA style predictor. Representation collapse is prevented with **SIGReg** from LeJEPA (Balestriero & LeCun, 2025) instead of an EMA teacher: one loss weight, no stop-gradient tricks, and, crucially for outlier detection, the embedding distribution of normal data becomes a *known isotropic Gaussian*. That gives a second score for free, the **Mahalanobis distance of each token embedding** to the training Gaussian, which can be fused with the prediction error. Section 7.5 qualifies this last point: on the tabular benchmark that score is just as good on an untrained network, so the Gaussian shaping is not what makes it work there.
* Two design lessons were found empirically and are, to our knowledge, not reported in the JEPA anomaly literature: (i) the SIGReg-trained encoder must **not** end in a LayerNorm, otherwise extreme inputs saturate to a direction that is *trivially predictable* and outliers score as *more normal* than inliers (on a 2-feature toy problem with the final defaults: mean ROC-AUC 0.29 with the LayerNorm, 0.87 without it; KNN, IForest and AutoEncoder reach 0.98 to 1.00 there); (ii) a pooled (mean) row embedding is a poor density score because an outlier's token embeddings are incoherent and *average out to the origin*; per-token Mahalanobis fixes this. Lesson (i) is specific to this network: the T-JEPA recipe normalises its encoder output and does not show the failure (Section 7.5).
* **Results (Section 7).** Tabular, all 47 ADBench datasets, 5 seeds (Section 7.6): `JEPAOD` is mid-table at ROC-AUC 0.736 / AP 0.353, significantly behind IForest (0.763 / 0.384) and not ahead of AutoEncoder (0.748 / 0.333). The first sweep, on 16 of those datasets with 3 seeds (Section 7.1), had `JEPAOD` as the best deep detector (0.760 / 0.438 vs AutoEncoder 0.746 / 0.375) behind KNN (0.786) and IForest (0.767); that ranking did not carry over to the other 31 datasets. Time series: best of all detectors on PyOD's synthetic series (0.95 vs LSTMAD 0.87), level with LSTMAD on SKAB under a strong train/test regime shift (0.83-0.86), but behind windowed IForest / AutoEncoder on NAB's sustained level shifts (0.63 vs 0.83) where a prediction error only fires at the transition and transductive training absorbs the anomalous regime. Inside the `JEPAOD` network SIGReg beats the EMA teacher in every setting.
* **T-JEPA comparison and untrained control (Section 7.5, added 2026-11-21).** T-JEPA publishes no outlier scores, so its recipe was re-implemented and scored like `JEPAOD`. On the 16 datasets of Section 7.1 `TJEPAOD` reaches ROC-AUC 0.782 / AP 0.412: a higher mean ROC-AUC than `JEPAOD` (ahead on 10 of 16 datasets), a lower AP, about 2.5 times the fit time, and neither difference is statistically significant (p = 0.30 and 0.94). A zero-epoch control changes how all the tabular numbers should be read: with **no training** the `embed` score reaches 0.746 on the `JEPAOD` network and 0.770 on the T-JEPA network, level with the trained models. Training is needed only for the prediction-error score (T-JEPA network: 0.466 untrained, 0.762 trained), and on 16 datasets the trained `combined` scores cannot be told apart from the untrained controls. τ-JEPA, a working paper on fraud detection, could not be read (login required); it is cited and not implemented.
* **Full suite and time-series control (Section 7.6, added 2026-11-21).** On all 47 ADBench datasets with 5 seeds the zero-epoch controls score at least as high as the trained models: `JEPAOD` 0.751 untrained against 0.736 trained (the control is better on 32 of 47 datasets; p = 0.055 before and 0.66 after correction for the 17 planned comparisons), and the untrained `embed` score of the T-JEPA network reaches 0.768 against 0.732 for trained `TJEPAOD`, which cannot be separated from `JEPAOD`. That untrained score is level with IForest (0.763) and more than a hundred times slower. For time series, training is what makes the default prediction-error score work on synthetic series (0.941 trained against 0.798 untrained, better on all 15 series), and it adds nothing measurable on SKAB (0.825 against 0.831) or NAB (0.628 against 0.671).
* **JEPA versions of the time-series detector (Sections 4.8 and 7.7, added 2026-11-21).** `TSJEPAOD(variant=...)` selects which JEPA trains the network: `'lejepa'` (SIGReg, the default and the detector described above), `'ijepa'` (EMA target), `'tsjepa'` (the recipe of TS-JEPA, Ennadir et al.) or `'tjepa'` (the T-JEPA recipe on patch tokens). No version beats the default: on the synthetic series the `pred` score is 0.941 for `lejepa`, 0.931 for `tsjepa`, 0.928 for `tjepa` and 0.885 for `ijepa`, and on SKAB, where the test data are in another regime, the three EMA versions fall to 0.43 to 0.67 against 0.825. `tjepa` is numerically ahead on NAB (0.684 against 0.628, not significant).

**Verdict for the library goal (revised after Section 7.6).** The evidence does not support adding the tabular detector to PyOD on performance grounds: on the full ADBench suite it is a mid-table detector, significantly behind IForest, not ahead of AutoEncoder and about 40 times slower than it, and a network with no training scores at least as well. The time-series detector has one supported strength, anomalies that break temporal structure, where training is what makes its default score work. On the two real benchmarks training adds nothing, and the known weak spots remain (sustained level shifts, large raw-space deviations that a windowed KNN already finds). The earlier verdict, "the highest mean among the deep models on the tabular sweep", described the 16-dataset subset and is withdrawn. The code in this repo is structured to drop into `pyod/models/jepa.py` and `pyod/models/ts_jepa.py` (Section 6). PyOD's inclusion policy requires an accepted issue first and, for a new method, a two-year maintenance commitment.

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
| **T-JEPA** (Thimonier et al., ICLR 2025; arXiv 2410.05016) | tabular rows, one token per feature | a small feature subset as context (20-40 % in the code defaults), 4 target subsets drawn among the other features | EMA teacher (0.996 → 1) plus a learned **[REG] token**, without which the authors report an early collapse | post-norm Transformer whose output passes LayerNorm → Linear → LayerNorm; squared-L2 latent loss; evaluated on classification / regression (7 datasets) with supervised models on the embeddings; anomaly detection is named as future work |
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
| **T-JEPA** (arXiv 2410.05016, ICLR 2025) | tabular (Adult, Higgs, Helena, Jannis, ALOI, California housing, MNIST) | none: representation learning, evaluated with supervised classifiers / regressors on the embeddings | downstream models improve; ResNet + T-JEPA has the best average rank in its Table 2 | the closest prior architecture to `JEPAOD` (feature tokens, masked latent prediction). It has no outlier score and no label-free evaluation. Section 7.5 runs its recipe as a detector |
| **τ-JEPA** (working paper on the Institut Louis Bachelier call-for-papers site) | tabular fraud detection | supervised classifiers on the pre-trained embeddings, according to its indexed abstract | abstract: embeddings help downstream fraud classifiers, most when labels are scarce | **not read in full** (the PDF is behind a login), so it is listed from its indexed title and abstract only and is not re-implemented here |

**Gap.** JEPA has reached tabular data with T-JEPA, but as representation learning for supervised tasks: neither T-JEPA nor τ-JEPA (as far as its abstract shows) scores rows without labels. We found no label-free tabular JEPA outlier detector, no PyOD/DeepOD-style implementation, and no time-series work that scores *timestamps* through the standard sliding-window protocols of PyOD / TSB-AD. All anomaly papers above except MTS-JEPA use the latent prediction error or embedding density, which is what JEPA-OD generalises. An earlier version of this report did not cite T-JEPA; that omission is corrected here (2026-11-21).

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
2. **`embed`** – Mahalanobis distance of each contextualised token embedding to a Gaussian fitted on training embeddings (per-token mean, covariance shared across tokens), averaged over tokens. With SIGReg the training embeddings are close to a known Gaussian, which was the motivation for this score (Section 7.5 shows that on the tabular benchmark it is as good on an untrained network, so that shaping is not what makes it work there); for time series it is computed per patch and localises anomalies **without the temporal smearing** that prediction-based scores have.
3. **`combined`** (tabular default) – both components z-normalised with their training statistics and averaged (MADJEPA-style fusion, but label-free). On the ADBench sweep it beats either component alone; under distribution shift the `embed` term is fragile, so the time-series detector defaults to `pred`.

**Time-series mapping.** Each (window, mask, masked patch) error is assigned to the `patch_len` timestamps that patch covers and aggregated (`mean` default, `max` optional) over all windows with stride `step`. Timestamps never covered (e.g. the first `window_size − m·patch_len` in `forecast` mode) are filled with `threshold_`, exactly as PyOD's `TimeSeriesOD` / `LSTMAD` do.

### 4.6 Two failure modes discovered while building it

1. **Final LayerNorm saturation ⇒ inverted scores.** With a final LayerNorm on the encoder, an extreme standardised value (|x| ≫ 1) produces an embedding ≈ `normalize(W_t)`: constant for all large values. The predictor learns "if the context is saturated, predict the saturated direction", so outliers get *lower* latent error than inliers (mean ROC-AUC 0.29 over three seeds of PyOD's 2-feature `generate_data`, 600 training rows, default 30 epochs). Removing the final norm lets the residual stream keep input magnitude; SIGReg bounds the *normal* embeddings instead. With the final code the same setting gives a mean ROC-AUC of 0.87 without the norm (0.94 with 5 epochs), and the 20- and 300-feature toy problems reach 1.0. The first version of this fix measured 0.19 before and 0.98 after on an earlier iteration of the code; the figures here are the ones the committed code reproduces. Two features remain a weak case: KNN, IForest and AutoEncoder score 0.98 to 1.00 on the same data, and longer training lowers the JEPA score (0.94 at 5 epochs, 0.87 at 30). The same saturation appears for *shifted* time series: on SKAB, whose test files run in a different operating regime from the training recording, a final LayerNorm turns the `embed` score upside down (AUC 0.20) and halves the `pred` score (0.58 vs 0.83), see Section 7. This is a caveat for the SIGReg-trained network of `JEPAOD` / `TSJEPAOD` and is worth stating in the PyOD docstring. It is not general to latent-prediction scores: `TJEPAOD`, whose encoder output is normalised, does not show it on the same toy problem (Section 7.5).
2. **Pooled embeddings cancel.** The mean over tokens of an outlier's embeddings is *closer* to the origin than an inlier's (incoherent directions average out), so Mahalanobis on the pooled vector was anti-correlated with anomaly (AUC 0.17). Per-token Mahalanobis with a shared covariance solves it.

### 4.7 Complexity

Training `O(epochs · n · T² · d_model)`; inference `O(n · K · T² · d_model)` with `K` masks. With `max_tokens = 64` the attention cost is bounded regardless of `d`; a 6 000 × 300 PyOD unit-test dataset trains 5 epochs in ≈ 8 s on CPU.

### 4.8 JEPA versions of the time-series detector (`variant`)

*Added 2026-11-21.* `TSJEPAOD` is a family: `variant` selects the JEPA that trains the network. Masking, scores, model sizes, optimiser, training budget and the PyOD interface are the same for every version. Results: Section 7.7.

| `variant` | network | target and collapse prevention | training loss | source |
|---|---|---|---|---|
| `'lejepa'` (default) | per-patch linear tokenizer, learned positions, pre-norm encoder without a final norm, predictor half as wide as the encoder | one shared encoder, SIGReg | squared error | LeJEPA; this is the detector of Sections 4.1 to 4.5 |
| `'ijepa'` | the same network | EMA copy of the encoder (momentum 0.996 → 1), layer-normalised targets | squared error | I-JEPA; formerly `target_mode='ema'` |
| `'tsjepa'` | convolutional patch tokenizer shared by all positions, sinusoidal positions, pre-norm encoder with a final LayerNorm, predictor as wide as the encoder | EMA copy of the whole encoder, tokenizer included (0.998 → 1), layer-normalised targets | absolute error | TS-JEPA (Ennadir et al.), re-implemented from the paper and the defaults of its released code |
| `'tjepa'` | per-patch linear tokenizer, sinusoidal positions, one [REG] token, post-norm ReLU encoder whose output passes LayerNorm → Linear → LayerNorm | EMA copy of the whole encoder (0.996 → 1) | squared error | T-JEPA (Thimonier et al.), a tabular method, applied here to patch tokens |

What the versions do **not** take from their papers:

* **Masking.** All versions use the detector's `mask_mode` and `mask_ratio` (default: one block of 30 % of the patches), because the scores map masked patches back to timestamps. TS-JEPA masks 70 % of the patches at random, which is available as `mask_mode='random', mask_ratio=0.7` and is benchmarked in Section 7.7. T-JEPA's small context with four target subsets is not implemented for time series.
* **Sizes, optimiser, budget.** Width 64, 4 heads, 2 encoder layers, AdamW at 1e-3 with cosine decay, 10 epochs over at most 5 000 windows. TS-JEPA's paper uses width 128, 2 heads and batch size 32, and does not state the depth.
* **Scores.** None of the four papers defines an anomaly score. `pred`, `embed` and `combined` are computed as in Section 4.5 for every version, with the squared error for `pred` whatever the training loss.
* **Multivariate input.** TS-JEPA's reference tokenizer is univariate. Here the convolution takes the channels of a patch as its input channels, which is the same operation for one channel.

`encoder_norm=None` and `ema_momentum=None` resolve to each version's published value (no normalised output for `'lejepa'` and `'ijepa'`, as found in Section 4.6). `variant` replaces the former `target_mode` of `TSJEPAOD` (`'sigreg'` is `'lejepa'`, `'ema'` is `'ijepa'`); the scores of both are bit-identical to the previous code on the series we compared.

---

## 5. API

All three classes (`JEPAOD`, `TJEPAOD`, `TSJEPAOD`) expose the full PyOD contract (`fit`, `decision_function`, `predict`, `predict_proba`, `predict_confidence`, `fit_predict_score`, `decision_scores_`, `threshold_`, `labels_`, `get_params/set_params`, pickling). The two tabular classes also have `token_embeddings(X)`, which returns the per-token embeddings for use with another detector. `TSJEPAOD(variant=...)` selects the JEPA version (Section 4.8). See `README.md` for usage, `tests/` for the behaviour covered.

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

* **Among deep detectors JEPA-OD is the strongest on average.** `JEPAOD[combined]` reaches mean ROC-AUC 0.760 / AP 0.438 against `AutoEncoder` 0.746 / 0.375, `DeepSVDD` 0.737 / 0.358, `DIF` 0.733 / 0.383 and `LUNAR` 0.729 / 0.357, and its average rank (6.84 of 15 columns) is second among the deep models, just behind `LUNAR` (6.81).
* **Overall it is third**, behind the two classical references `KNN` (0.786 AUC, rank 4.8) and `IForest` (0.767, rank 5.3). The pattern is the familiar ADBench one: local-density methods win on low-dimensional data with local outliers (`vowels`, `wine`, `Wilt`, `Waveform`, `Ionosphere`), while JEPA-OD wins or ties where outliers break joint feature structure in higher dimension (`musk` 1.000, `satimage-2` 0.992, `WBC` 0.995, `skin` 0.766 best of all, `cardio` 0.90).
* **Weak spots.** `thyroid` (0.89 vs 0.98 for `IForest`), `vowels` (0.79 vs 0.98 for `KNN`), `Ionosphere` (0.80 vs 0.92); these are small, low-dimensional sets where one or two marginal features carry the anomaly and a transformer over feature tokens brings no benefit. `speech` (400-d, 1.7 % outliers) is at chance for every detector.
* **Score components.** `combined` > `pred` ~ `embed` (0.760 / 0.750 / 0.749 AUC; 0.438 / 0.423 / 0.432 AP); the two components are complementary (e.g. `wine`: `pred` 0.85, `embed` 0.61; `thyroid`: `embed` 0.91, `pred` 0.83). `combined` is therefore the tabular default. Under distribution shift `pred` alone is safer (Section 7.2).
* **Collapse prevention.** SIGReg (0.760) edges the I-JEPA EMA teacher (0.750 AUC, 0.422 AP) and needs no momentum schedule; the EMA variant is however ~1.8x faster because the per-token Epps-Pulley statistic (128 projections x 17 knots per token) dominates the step time. Fewer projections or sub-sampling tokens for SIGReg is the obvious optimisation. The reconstruction anchor (`recon_weight=0.1`) changes nothing (0.760).
* **Read with Sections 7.5 and 7.6.** A zero-epoch control added later shows that the `embed` score is as good on an untrained network (0.746 against 0.750), and `combined` 0.750 against 0.760. The lead over the other deep detectors should therefore not be attributed to JEPA training. Section 7.6 repeats the sweep on all 47 ADBench datasets with 5 seeds: there `JEPAOD[combined]` (0.736) is below `AutoEncoder` (0.748) and significantly below `IForest` (0.763), so the ranking of this section holds for these 16 datasets only.

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
* **Read with Section 7.6.** A zero-epoch control with 5 seeds shows what training contributes: it is what makes `pred` work on the synthetic series (0.941 trained against 0.798 untrained), and it adds nothing measurable on SKAB or NAB. The parity with `LSTMAD` on SKAB is reached by an untrained network as well (0.831 and 0.859), and the untrained `embed` score already reaches 0.932 on the synthetic series.

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

* **SIGReg vs EMA teacher.** Inside the `JEPAOD` network SIGReg wins everywhere (the EMA-based T-JEPA recipe of Section 7.5 is a different network and is not worse than SIGReg): tabular mean AUC 0.77-0.81 vs 0.67-0.75 (EMA), time series 0.72-0.77 vs 0.69-0.71. The EMA variant is also the one whose `embed` score is weakest, consistent with target-scale drift.
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

<!-- TJEPA:BEGIN -->

### 7.5 T-JEPA family: the T-JEPA recipe as a detector, and an untrained control

*Added 2026-11-21. Code: `jepa_od/tjepa.py`, `jepa_od/_tjepa_nets.py`. Benchmarks: `benchmarks/bench_tjepa.py`, `benchmarks/toy_two_features.py`, tables by `benchmarks/make_tjepa_tables.py`. Raw results: `results/tjepa_results.csv`, `results/toy_two_features.csv`.*

**What can be compared.** T-JEPA publishes no outlier-detection numbers. Its paper reports the accuracy and regression error of supervised models trained on the embeddings, and none of those figures can be set against a ROC-AUC. τ-JEPA could not be read (Section 3). The comparison that is possible is to train the T-JEPA recipe on the same data and score rows with the same scores as `JEPAOD`. `jepa_od.TJEPAOD` does that. It is a subclass of `JEPAOD` that replaces the network and the training step and inherits the scores and the PyOD interface.

**How close `TJEPAOD` is to T-JEPA.** It is a re-implementation written from the paper and the defaults of the released code, not the authors' code.

| | T-JEPA (paper and released code) | `TJEPAOD` |
|---|---|---|
| tokens | one per feature, linear tokenizer, fixed sinusoidal positions, a learned [REG] token | the same; tables wider than 64 features are grouped into 64 tokens (`max_tokens`) and categorical features are treated as numeric |
| encoder | post-norm ReLU Transformer, output passed through LayerNorm → Linear → LayerNorm | the same, at the code defaults (4 layers, 8 heads, width 64, feed-forward 256, dropout 0.1), not the sizes tuned per dataset (16 layers in the released launch script) |
| target | EMA copy of the encoder, momentum 0.996 → 1 | the same |
| masking | context share and target share drawn per batch, 4 target subsets among the features outside the context | the same, at the code-default ranges (20-40 % each) |
| predictor | Transformer with a mask token and fixed positions | the same, at the code defaults (2 layers, width 64, 4 heads) |
| loss, optimiser | squared error; AdamW, warm-up and cosine decay, gradients clipped by value | squared error; Adam with zero weight decay (the launch script also sets it to zero), warm-up and cosine decay, gradients clipped by norm |
| training budget | batch 512, learning rate 1e-4, 100 epochs by default; 300 epochs in the launch script; hyper-parameters tuned per dataset with labels | the budget of `JEPAOD`: batch 64, learning rate 1e-3, 30 epochs, 1 warm-up epoch |
| use of the model | embeddings are the input of supervised models | **not part of T-JEPA:** latent prediction error over 8 fixed masks (`pred`), per-token Mahalanobis distance of the target-encoder embeddings (`embed`), and both (`combined`) |

**Variants and controls.** `TJEPAOD-nonorm` (`encoder_norm=False`: pre-norm blocks, no output LayerNorms), `TJEPAOD-noreg` (`n_reg_tokens=0`), `TJEPAOD-repoopt` (T-JEPA's optimiser defaults: batch 512, learning rate 1e-4, 100 epochs, 5 warm-up epochs), and two zero-epoch controls, `JEPAOD-untrained` and `TJEPAOD-untrained` (`epoch_num=0`: the network keeps its random initial weights, and only the score statistics are fitted on the training split). Every fit is scored four ways: `pred`, `embed`, `combined`, and `emb+KNN`, which is PyOD's `KNN` on the flattened token embeddings, the label-free analogue of T-JEPA's protocol of feeding embeddings to a downstream model.

**Setup.** As in Section 7.1: the same 16 datasets, splits and 3 seeds, no tuning. `JEPAOD` was re-run alongside as a reference. Of its 144 rows, 106 are identical to the stored results and 125 agree to 0.001; on the two largest datasets (`magic.gamma`, `skin`) single rows differ by up to 0.021 ROC-AUC, and the means over datasets agree to 0.001. The cause is the thread count: this benchmark ran with 3 threads per job and Section 7.1 with 4, and re-running `magic.gamma` with 4 threads reproduces the stored rows exactly. Results are deterministic for a fixed seed and thread count, not across thread counts. The five benchmark jobs ran in parallel, so fit times are comparable within this section only (`JEPAOD` takes 28 s here and 20 s in Section 7.4). The p-values are two-sided Wilcoxon signed-rank tests on the 16 per-dataset means. They are not corrected for the 18 comparisons made; only the first comparison of finding 5 would survive a Bonferroni correction.

**Findings.**

1. **`TJEPAOD` against `JEPAOD`: higher mean ROC-AUC, lower mean AP, neither difference significant.** `TJEPAOD[combined]` reaches 0.782 / 0.412 and `JEPAOD[combined]` 0.760 / 0.438. `TJEPAOD` is ahead on 10 of 16 datasets and in each of the three seeds, but the paired test gives p = 0.30 for ROC-AUC and p = 0.94 for AP, and the per-seed means of a single detector vary by 0.04 to 0.05, more than the gap between the two. The differences are large in both directions and specific to datasets: `Waveform` 0.870 against 0.634, `skin` 0.858 against 0.748 and `vowels` 0.883 against 0.791 in favour of `TJEPAOD`; `fault` 0.451 against 0.590 and `musk` 0.874 against 1.000 in favour of `JEPAOD`. `TJEPAOD` takes about 2.5 times as long to fit. Its mean ROC-AUC is level with `KNN` (0.786) and above `IForest` (0.767), but `IForest` is the better of the two on 11 of the 16 datasets and both classical detectors have a higher AP (0.449 and 0.477).
2. **T-JEPA's two stabilisers do not change the scores at this budget.** Without the [REG] token `combined` is 0.782, without the output normalisation 0.777 (p = 0.82 and 0.86 against the full recipe). We did not measure the rank of the representations, so this does not contradict the collapse that the T-JEPA authors report without the [REG] token over much longer training. It says that the outlier scores do not depend on the token here.
3. **T-JEPA's own optimiser settings are not better for detection.** With batch 512, learning rate 1e-4 and 100 epochs, `combined` is 0.752 and `pred` 0.725, at twice the fit time (p = 0.35 against the 30-epoch budget).
4. **Untrained control: the `embed` score does not need training.** With zero epochs `embed` reaches 0.770 on the T-JEPA network and 0.746 on the `JEPAOD` network. The trained networks reach 0.763 and 0.750 (8 / 8 and 7 / 9 datasets, p = 0.86 and 0.94). The per-token Mahalanobis distance on the embeddings of a randomly initialised token transformer is therefore a competitive detector by itself, at 2 to 4 s per fit: 0.770 is above every trained deep baseline of Section 7.1, and on the 15 datasets where `PCA` ran it is above `PCA` (0.790 against 0.731, 12 / 3 datasets, p = 0.008 uncorrected), so it is more than a linear Mahalanobis score. This contradicts the reading in earlier versions of Sections 1 and 4.5 that SIGReg's Gaussian shaping is what makes `embed` a good score. On this benchmark the score is equally good with no shaping at all.
5. **Training matters for the prediction error only.** `pred` needs a trained predictor on the T-JEPA network: 0.466 untrained, 0.762 trained, better on all 16 datasets (p < 0.001). On the `JEPAOD` network the gain is smaller and not significant: 0.696 untrained, 0.749 trained (10 / 5 datasets, p = 0.09). We did not investigate why an untrained `JEPAOD` predictor is this far above chance. The trained `combined` scores cannot be told apart from the untrained controls: `TJEPAOD[combined]` 0.782 against `TJEPAOD-untrained[embed]` 0.770 (7 / 9 datasets, p = 1.0), and `JEPAOD[combined]` 0.760 against `JEPAOD-untrained[combined]` 0.750 (9 / 7, p = 0.86).
6. **Embeddings plus a classical detector is the weaker route, and SIGReg training makes it worse.** `emb+KNN` scores 0.68 to 0.70 for the trained T-JEPA variants with a normalised output, below their own `embed` score and below `KNN` on the raw features (0.786); without the output normalisation it reaches 0.761. For `JEPAOD`, `KNN` on the trained embeddings (0.675) is worse than on the untrained ones (0.757; 13 / 2 datasets, p = 0.005 uncorrected). This is consistent with the transductive contamination of Section 8: SIGReg pushes the embeddings of the unlabelled outliers in the training split towards the same Gaussian as the inliers.
7. **Two-feature toy problem (Section 4.6; last table below).** The LayerNorm failure reproduces for `JEPAOD`: `combined` is 0.33 with a final LayerNorm (an earlier run of this problem, quoted in Sections 1 and 4.6, gave 0.29) and 0.87 without. It does **not** occur for `TJEPAOD`, whose encoder output is normalised by default (0.96, and 0.99 without the normalisation). The rule "no final LayerNorm" therefore holds for the SIGReg-trained network and is not a general property of latent-prediction scores. We did not isolate which difference (EMA target, tokenizer, masking) removes the failure. The untrained `JEPAOD` control scores 0.97 here against 0.87 after 30 epochs, in line with the observation in Section 4.6 that more training lowers the score on this problem.

**What this changes.**

* `TJEPAOD` is a usable second tabular member of the family, with the numerically highest mean ROC-AUC of the deep detectors here. It is not a demonstrated improvement over `JEPAOD`, which is about 2.5 times faster and has the higher AP. The benchmark does not separate the two.
* The benchmark does not support the claim that JEPA *training* is why these detectors work on tabular data. What it supports is narrower: (a) a per-token Mahalanobis score on the features of a random token transformer is a strong and cheap detector; (b) training adds a prediction-error score that is clearly better than its untrained counterpart on the T-JEPA network, and adding it raises the mean by about 0.01, which is within the noise of 16 datasets.
* The untrained control was not run for `TSJEPAOD` in this section. Section 7.6 runs it.
* A mean difference of 0.02 ROC-AUC is not detectable with 16 datasets and 3 seeds. Separating the family members, or showing a benefit of training, needs the full ADBench sweep and more seeds. Section 7.6 is that sweep: on 47 datasets the two family members are level, both are mid-table, and the controls score at least as high as the trained models. Where this section and Section 7.6 disagree, Section 7.6 is the better evidence.

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

#### Two-feature toy problem: mean ROC-AUC over 3 seeds (per-seed values in brackets)

| detector | pred | embed | combined |
|---|---|---|---|
| JEPAOD | 0.917 (0.81 / 0.96 / 0.97) | 0.722 (0.56 / 0.65 / 0.95) | 0.866 (0.72 / 0.89 / 0.99) |
| JEPAOD(encoder_norm=True) | 0.265 (0.30 / 0.32 / 0.17) | 0.443 (0.31 / 0.30 / 0.72) | 0.330 (0.31 / 0.31 / 0.38) |
| JEPAOD-untrained | 0.947 (0.95 / 0.99 / 0.90) | 0.864 (0.77 / 0.82 / 1.00) | 0.966 (0.93 / 0.97 / 1.00) |
| TJEPAOD | 0.900 (0.84 / 0.86 / 1.00) | 0.985 (0.97 / 0.99 / 1.00) | 0.960 (0.93 / 0.95 / 1.00) |
| TJEPAOD-nonorm | 0.940 (0.86 / 0.96 / 1.00) | 0.991 (0.98 / 1.00 / 1.00) | 0.988 (0.97 / 0.99 / 1.00) |
| TJEPAOD-untrained | 0.560 (0.96 / 0.68 / 0.04) | 0.985 (0.97 / 0.99 / 1.00) | 0.767 (0.97 / 0.96 / 0.37) |
| KNN | | | 0.989 (0.97 / 1.00 / 1.00) |
| IForest | | | 0.988 (0.97 / 1.00 / 1.00) |

<!-- TJEPA:END -->

<!-- EXTENDED:BEGIN -->

### 7.6 Extended benchmark: all 47 ADBench datasets with 5 seeds, and the untrained control for time series

*Added 2026-11-21. Code: `benchmarks/bench_extended.py`, the `--epochs 0` option of `benchmarks/bench_ts_jepa_final.py`, tables by `benchmarks/make_extended_tables.py`. Raw results: `results/extended_results.csv`, `results/ts_control_trained.csv`, `results/ts_control_untrained.csv`.*

**Why.** Section 7.5 left two questions open: whether 16 datasets and 3 seeds were too few to see a benefit of training, and whether the untrained control changes the time-series results. This section answers both. On tabular data more data does not reveal a benefit of training, and it overturns the ranking of Section 7.1. On time series training helps in one setting out of three.

**Setup, tabular.** All 47 datasets of the ADBench classical suite: the 16 of Section 7.1 and the other 31, fetched by `download_data.py --all-adbench`. 5 seeds. Otherwise the protocol of Section 7.1: at most 20 000 rows (a seeded subsample), a stratified 60/40 split per seed, detectors fitted without labels on the contaminated training split, ROC-AUC and AP on the test split, default parameters, no tuning. Columns: the 12 PyOD baselines of Section 7.1, `JEPAOD`, `TJEPAOD` and their zero-epoch controls, each fit scored as `pred`, `embed`, `combined` and `emb+KNN`. Every fit ran on one thread, with 13 dataset-seed tasks in parallel, so the times include contention and are comparable within this section only. All 235 dataset-seed pairs produced results. Eight of them (seeds 0 and 1 of `wine`, `Hepatitis`, `smtp` and `cardio`) come from a first test of the script with the same settings and 8 parallel tasks; `results/bench_extended.log` covers the other 227. PyOD's `PCA` raised an error ("Input contains infinity or a value too large") in 33 of them: on every seed of `backdoor`, `campaign`, `census`, `mnist` and `optdigits`, and on some seeds of `Cardiotocography`, `fault` and `cardio`. Its mean is over 42 datasets and is not comparable with the other columns. We did not investigate the cause. No other fit failed.

**Statistics.** The 17 comparisons of the third table were written into `make_extended_tables.py` before the results were looked at. Each is a two-sided Wilcoxon signed-rank test on the per-dataset means, with a Holm correction over the 17 tests and a bootstrap 95 % interval for the mean difference. The size analysis in the fifth table was added after seeing the results and is labelled post hoc.

**Consistency with the earlier runs.** On the 16 datasets and 3 seeds shared with Sections 7.1 and 7.5, every row of the two zero-epoch controls and of `KNN`, `IForest`, `ECOD`, `DeepSVDD` and `DIF` is identical to the stored one. The trained rows differ, because this run uses one thread and the earlier ones three or four: `JEPAOD[combined]` on 12 of 48 rows (largest difference 0.015) and `TJEPAOD[combined]` on 35 of 48 (largest 0.076). The means agree: 0.760 against 0.760 and 0.780 against 0.782.

**Findings, tabular.**

1. **On the full suite the trained JEPA detectors are mid-table.** `JEPAOD[combined]` reaches 0.736 ROC-AUC / 0.353 AP and `TJEPAOD[combined]` 0.732 / 0.318. Of the 16 columns of the first table they are 9th and 11th by mean ROC-AUC and 13th and 10th by average rank. `JEPAOD[combined]` is not the best column on any of the 47 datasets.
2. **`JEPAOD` is significantly behind `IForest` and is not ahead of the other deep detectors.** `IForest` reaches 0.763 and is the better of the two on 35 of 47 datasets (difference -0.027, interval -0.045 to -0.009, corrected p = 0.010). This is the only comparison with a baseline that survives the correction. `AutoEncoder` (0.748 / 0.333) has the higher mean ROC-AUC and `JEPAOD` the higher AP, neither significantly (p = 0.33 and 0.70). `KNN` has the same mean (0.732) and is the better of the two on 31 datasets; its mean is pulled down by a few datasets where it fails (`http` 0.036, `optdigits` 0.432, `musk` 0.504). The statement of Sections 1 and 7.1 that `JEPAOD` has the highest mean of the deep detectors does not hold on the full suite: `AutoEncoder` 0.748, `JEPAOD` 0.736, `DIF` 0.734, `DeepSVDD` 0.729, `LUNAR` 0.696.
3. **The 16 datasets of Section 7.1 flattered the JEPA detectors.** On those 16, now with 5 seeds, `JEPAOD` (0.753) and `TJEPAOD` (0.773) are above `AutoEncoder` (0.736). On the 31 added datasets they are below it (0.728 and 0.711 against 0.754), while `IForest` is the same on both parts (0.764 and 0.763). The 16 were a convenience subset and not a random sample of the 47, and the rankings drawn from them do not generalise.
4. **The zero-epoch controls score at least as high as the trained detectors.** `JEPAOD-untrained[combined]` reaches 0.751 / 0.379 against 0.736 / 0.353 for the trained model. The control is better on 32 of 47 datasets and in each of the five seeds (difference -0.015, interval -0.033 to +0.003, p = 0.055 before and 0.66 after correction). `TJEPAOD-untrained[embed]` reaches 0.768 / 0.371 against 0.732 / 0.318 for `TJEPAOD[combined]` (control better on 29 datasets, p = 0.057 before correction) and against 0.726 for `TJEPAOD[embed]` (34 datasets, difference -0.041, interval -0.073 to -0.012, p = 0.004 before and 0.062 after correction). With three times the datasets of Section 7.5 the sign is against training in all three comparisons, and none is significant after correction. What the data support: at this budget training gives no benefit on the default scores, and probably costs a little.
5. **Training makes the embeddings worse for a nearest-neighbour detector.** `JEPAOD[emb+KNN]` is 0.664 trained and 0.726 untrained; the control is better on 40 of 47 datasets (difference -0.062, interval -0.093 to -0.035, corrected p < 0.001 for ROC-AUC and for AP). This is the clearest effect of training in the table, and it is negative.
6. **Only the T-JEPA prediction error needs training.** `TJEPAOD[pred]` is 0.702 trained and 0.479 untrained (41 of 47 datasets, corrected p < 0.001). For `JEPAOD[pred]` training changes nothing: 0.719 against 0.722 (p = 0.61), with a lower AP after training (0.338 against 0.374, p = 0.031 before and 0.44 after correction). The gain reported for `JEPAOD[pred]` in Section 7.5 (0.696 to 0.749 on 16 datasets, p = 0.09) does not hold on 47.
7. **`TJEPAOD` and `JEPAOD` cannot be separated.** 0.732 against 0.736 ROC-AUC (`TJEPAOD` ahead on 29 of 47 datasets, interval of the difference -0.033 to +0.024, p = 0.32) and 0.318 against 0.353 AP (p = 0.34), at twice the fit time. The ROC-AUC lead of Section 7.5 belonged to the 16-dataset subset.
8. **The untrained per-token Mahalanobis score is level with `IForest`, not better.** `TJEPAOD-untrained[embed]` 0.768 / 0.371 against `IForest` 0.763 / 0.384: better on 20 datasets and worse on 27, p = 0.95. Against `PCA` on its 42 datasets it is 0.770 against 0.738 (p = 0.008 before and 0.10 after correction). It takes 15.5 s per fit here against 0.1 s for `IForest`, so it is an observation about random token transformers and not a detector to recommend. Section 7.5 called it "strong and cheap"; it is as strong as `IForest` and more than a hundred times slower.
9. **Post hoc: for `TJEPAOD` the loss from training grows with the size of the training split** (fifth table). The difference between `TJEPAOD[embed]` and its control correlates with the number of training rows at rho = -0.56 (p < 0.001). In the larger half of the datasets `TJEPAOD[combined]` is 0.074 below the control, in the smaller half level with it. The largest losses are on `cover` (-0.37), `mammography` (-0.28), `shuttle` (-0.25) and `celeba` (-0.24). `JEPAOD` shows no such dependence (rho = -0.11). With a fixed number of epochs a larger split means more gradient steps, so this is consistent with the T-JEPA network losing outlier signal as training goes on, the effect seen for time series in Section 7.2. Size is confounded with everything else that separates large datasets from small ones, and the analysis was not planned; a sweep over epochs on the large datasets would test it.

**Setup, time series.** `TSJEPAOD` at its class defaults (10 epochs, at most 5 000 training windows) against the same class with `epochs=0`, 5 seeds, the three protocols of Section 7.2, six variants (two maskings, three scores). Synthetic: 3 configurations and 5 seeds give 15 different series. NAB: 7 series, 5 initialisations each. SKAB: 16 files, 5 initialisations each. Tests are Wilcoxon signed-rank over series with a Holm correction over the 18 rows of the last table. The seed-0 trained rows agree with `results/ts_results_jepa_final.csv` to within 0.002 ROC-AUC (that run used three threads and this one two).

**Findings, time series.**

10. **Synthetic series: training is what makes the prediction error work.** The default `block[pred]` reaches 0.941 trained and 0.798 untrained, better on all 15 series (difference +0.144, interval +0.088 to +0.199, corrected p < 0.001); `forecast[pred]` 0.946 against 0.786. An untrained `pred` would be below `LSTMAD` (0.866 in Section 7.2). This is the one result in this report that the JEPA training objective demonstrably produces. It comes with a qualification: on the same series the untrained network's `embed` score reaches 0.932, above every baseline mean of Section 7.2, and the trained default is ahead of it by 0.010 (13 of 15 series, p = 0.015; post hoc, uncorrected). The lead over the baselines does not depend on training. The last 0.01 does.
11. **SKAB: training adds nothing to the default.** `block[pred]` 0.825 trained against 0.831 untrained, `forecast[pred]` 0.857 against 0.859. The parity with `LSTMAD` (0.853) reported in Section 7.2 is therefore not a result of training. Only `forecast[combined]` gains (0.819 against 0.795, on all 16 files). The 16 files are scored by the same five fitted models per masking, so they are not independent, and the p-values of the SKAB rows overstate the evidence; the "significant" difference of -0.006 for `block[pred]` has no practical meaning.
12. **NAB: no detectable effect.** `block[pred]` 0.628 against 0.671 and `forecast[pred]` 0.585 against 0.664, with 7 series, wide intervals and every p above 0.29 before correction. The untrained `pred` is numerically higher, which fits the diagnosis of Section 7.2 that transductive training absorbs the anomalous regime.
13. **`embed` needs no training on any group**: 0.928 against 0.932 (synthetic), 0.825 against 0.828 (SKAB), 0.627 against 0.619 (NAB) for `block`.

**What this changes.**

* The rankings of Sections 1, 7.1 and 7.5 were measured on 16 datasets and do not survive the full suite. On 47 datasets `JEPAOD` and `TJEPAOD` are ordinary mid-table detectors. `JEPAOD` is significantly worse than `IForest`, neither is ahead of `AutoEncoder`, and they take about 40 and 80 times as long to fit as `AutoEncoder` and one to two thousand times as long as `IForest`.
* On tabular data there is no evidence that JEPA training helps outlier detection, and consistent evidence, not significant after correction, that it costs a little at this budget. The one score that needs training, the T-JEPA prediction error, is no better than the untrained `embed` score.
* On time series the training objective earns the default score on synthetic anomalies that break temporal structure, and nothing measurable on SKAB and NAB.
* What remains as positive results: the prediction-error score for time series with structure-breaking anomalies; the two failure modes of Section 4.6; and the observation that per-token Mahalanobis distances on random token-transformer features match `IForest`.
* Not tested, so not ruled out: other training budgets (finding 9 suggests fewer steps on large tables), early stopping, tuned hyper-parameters, training on clean data only, and the TSB-AD benchmark. A claim that training helps under any of these needs the same zero-epoch control.

Coverage: 47 datasets, 5 seeds, 235 dataset-seed pairs with results. Fits that raised an error: PCA 33.

#### Extended benchmark: 47 ADBench datasets, mean of 5 seeds, ranked by mean ROC-AUC (16 columns; time is fit plus scoring with one thread)

| detector | mean ROC-AUC | mean AP | avg rank | wins | datasets | mean time (s) |
|---|---|---|---|---|---|---|
| TJEPAOD-untrained[embed] | 0.768 | 0.371 | 6.46 | 1 | 47 | 15.5 |
| IForest | 0.763 | 0.384 | 6.46 | 5 | 47 | 0.1 |
| JEPAOD-untrained[combined] | 0.751 | 0.379 | 7.73 | 2 | 47 | 10.7 |
| AutoEncoder | 0.748 | 0.333 | 8.48 | 0 | 47 | 2.9 |
| COPOD | 0.746 | 0.346 | 7.89 | 4 | 47 | 0.1 |
| ECOD | 0.742 | 0.359 | 8.15 | 3 | 47 | 0.1 |
| PCA | 0.738 | 0.393 | 8.40 | 2 | 42 | 0.2 |
| HBOS | 0.737 | 0.341 | 8.39 | 4 | 47 | 0.1 |
| JEPAOD[combined] | 0.736 | 0.353 | 9.39 | 0 | 47 | 118.6 |
| DIF | 0.734 | 0.317 | 8.64 | 5 | 47 | 7.2 |
| TJEPAOD[combined] | 0.732 | 0.318 | 8.55 | 3 | 47 | 239.7 |
| KNN | 0.732 | 0.360 | 6.98 | 6 | 47 | 0.2 |
| DeepSVDD | 0.729 | 0.317 | 9.19 | 3 | 47 | 1.0 |
| LUNAR | 0.696 | 0.289 | 9.47 | 3 | 47 | 9.7 |
| OCSVM | 0.685 | 0.293 | 10.39 | 1 | 47 | 4.7 |
| LOF | 0.666 | 0.281 | 10.61 | 3 | 47 | 0.2 |

#### JEPA family and untrained controls: mean ROC-AUC / AP per score (47 datasets)

| variant | pred | embed | combined | emb+KNN | combined ROC-AUC by seed | fit time (s) |
|---|---|---|---|---|---|---|
| JEPAOD | 0.719 / 0.338 | 0.737 / 0.360 | 0.736 / 0.353 | 0.664 / 0.283 | 0.739 / 0.744 / 0.743 / 0.735 / 0.721 | 118.6 |
| JEPAOD-untrained | 0.722 / 0.374 | 0.741 / 0.357 | 0.751 / 0.379 | 0.726 / 0.330 | 0.748 / 0.757 / 0.757 / 0.746 / 0.746 | 10.7 |
| TJEPAOD | 0.702 / 0.308 | 0.726 / 0.317 | 0.732 / 0.318 | 0.640 / 0.274 | 0.740 / 0.745 / 0.723 / 0.740 / 0.715 | 239.7 |
| TJEPAOD-untrained | 0.479 / 0.183 | 0.768 / 0.371 | 0.645 / 0.282 | 0.660 / 0.284 | 0.626 / 0.629 / 0.653 / 0.701 / 0.618 | 15.5 |

#### Paired comparisons over the datasets (two-sided Wilcoxon signed-rank on per-dataset means; Holm correction over the 17 comparisons, which were fixed in advance; CI: bootstrap over datasets)

|  | A | B | datasets | ROC-AUC A / B | A − B [95 % CI] | A / B better | p | p (Holm) | AP A / B | p AP | p AP (Holm) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | JEPAOD[combined] | JEPAOD-untrained[combined] | 47 | 0.736 / 0.751 | -0.015 [-0.033, +0.003] | 14 / 32 | 0.055 | 0.662 | 0.353 / 0.379 | 0.050 | 0.603 |
| 2 | JEPAOD[embed] | JEPAOD-untrained[embed] | 47 | 0.737 / 0.741 | -0.004 [-0.018, +0.010] | 17 / 29 | 0.249 | 1.000 | 0.360 / 0.357 | 0.516 | 1.000 |
| 3 | JEPAOD[pred] | JEPAOD-untrained[pred] | 47 | 0.719 / 0.722 | -0.004 [-0.030, +0.024] | 20 / 27 | 0.611 | 1.000 | 0.338 / 0.374 | 0.031 | 0.437 |
| 4 | JEPAOD[emb+KNN] | JEPAOD-untrained[emb+KNN] | 47 | 0.664 / 0.726 | -0.062 [-0.093, -0.035] | 6 / 40 | <0.001 | <0.001 | 0.283 / 0.330 | <0.001 | <0.001 |
| 5 | TJEPAOD[combined] | TJEPAOD-untrained[embed] | 47 | 0.732 / 0.768 | -0.035 [-0.066, -0.006] | 18 / 29 | 0.057 | 0.662 | 0.318 / 0.371 | 0.074 | 0.670 |
| 6 | TJEPAOD[embed] | TJEPAOD-untrained[embed] | 47 | 0.726 / 0.768 | -0.041 [-0.073, -0.012] | 12 / 34 | 0.004 | 0.062 | 0.317 / 0.371 | 0.011 | 0.171 |
| 7 | TJEPAOD[pred] | TJEPAOD-untrained[pred] | 47 | 0.702 / 0.479 | +0.223 [+0.168, +0.282] | 41 / 6 | <0.001 | <0.001 | 0.308 / 0.183 | <0.001 | 0.001 |
| 8 | TJEPAOD[combined] | JEPAOD[combined] | 47 | 0.732 / 0.736 | -0.004 [-0.033, +0.024] | 29 / 18 | 0.320 | 1.000 | 0.318 / 0.353 | 0.341 | 1.000 |
| 9 | JEPAOD[combined] | AutoEncoder | 47 | 0.736 / 0.748 | -0.011 [-0.035, +0.011] | 21 / 26 | 0.326 | 1.000 | 0.353 / 0.333 | 0.702 | 1.000 |
| 10 | JEPAOD[combined] | KNN | 47 | 0.736 / 0.732 | +0.005 [-0.046, +0.064] | 16 / 31 | 0.166 | 1.000 | 0.353 / 0.360 | 0.058 | 0.627 |
| 11 | JEPAOD[combined] | IForest | 47 | 0.736 / 0.763 | -0.027 [-0.045, -0.009] | 12 / 35 | <0.001 | 0.010 | 0.353 / 0.384 | 0.035 | 0.453 |
| 12 | TJEPAOD[combined] | KNN | 47 | 0.732 / 0.732 | +0.001 [-0.052, +0.059] | 17 / 30 | 0.326 | 1.000 | 0.318 / 0.360 | 0.057 | 0.627 |
| 13 | TJEPAOD[combined] | IForest | 47 | 0.732 / 0.763 | -0.031 [-0.062, -0.000] | 15 / 32 | 0.061 | 0.662 | 0.318 / 0.384 | 0.076 | 0.670 |
| 14 | TJEPAOD-untrained[embed] | KNN | 47 | 0.768 / 0.732 | +0.036 [-0.017, +0.096] | 27 / 20 | 0.434 | 1.000 | 0.371 / 0.360 | 0.912 | 1.000 |
| 15 | TJEPAOD-untrained[embed] | IForest | 47 | 0.768 / 0.763 | +0.004 [-0.011, +0.020] | 20 / 27 | 0.946 | 1.000 | 0.371 / 0.384 | 0.197 | 1.000 |
| 16 | TJEPAOD-untrained[embed] | PCA | 42 | 0.770 / 0.738 | +0.032 [+0.011, +0.057] | 27 / 15 | 0.008 | 0.100 | 0.382 / 0.393 | 0.872 | 1.000 |
| 17 | JEPAOD-untrained[combined] | AutoEncoder | 47 | 0.751 / 0.748 | +0.003 [-0.021, +0.026] | 28 / 19 | 0.453 | 1.000 | 0.379 / 0.333 | 0.088 | 0.670 |

#### Mean ROC-AUC on the 16 datasets of Section 7.1 and on the datasets added here

| detector | original 16 (n=16) | added (n=31) | all (n=47) |
|---|---|---|---|
| KNN | 0.783 | 0.705 | 0.732 |
| IForest | 0.764 | 0.763 | 0.763 |
| AutoEncoder | 0.736 | 0.754 | 0.748 |
| JEPAOD[combined] | 0.753 | 0.728 | 0.736 |
| JEPAOD-untrained[combined] | 0.748 | 0.752 | 0.751 |
| TJEPAOD[combined] | 0.773 | 0.711 | 0.732 |
| TJEPAOD-untrained[embed] | 0.764 | 0.769 | 0.768 |

#### Post hoc: trained minus control ROC-AUC against the size of the training split (mean difference, with datasets where the trained model / the control is better in brackets; not corrected, not fixed in advance)

| trained | control | Spearman rho with n_train | smaller half (n_train <= 2524, 24 datasets) | larger half (23 datasets) |
|---|---|---|---|---|
| JEPAOD[combined] | JEPAOD-untrained[combined] | -0.11 (p = 0.476) | -0.010 (4 / 19) | -0.019 (10 / 13) |
| JEPAOD[embed] | JEPAOD-untrained[embed] | +0.01 (p = 0.928) | -0.006 (8 / 15) | -0.002 (9 / 14) |
| JEPAOD[pred] | JEPAOD-untrained[pred] | -0.25 (p = 0.096) | +0.008 (10 / 14) | -0.016 (10 / 13) |
| TJEPAOD[combined] | TJEPAOD-untrained[embed] | -0.36 (p = 0.013) | +0.003 (11 / 13) | -0.074 (7 / 16) |
| TJEPAOD[embed] | TJEPAOD-untrained[embed] | -0.56 (p < 0.001) | +0.004 (7 / 16) | -0.088 (5 / 18) |
| TJEPAOD[pred] | TJEPAOD-untrained[pred] | -0.17 (p = 0.241) | +0.254 (19 / 5) | +0.190 (22 / 1) |

#### Per-dataset ROC-AUC (mean of seeds; best of the row in bold)

| dataset | KNN | IForest | AutoEncoder | JEPAOD[combined] | JEPAOD-untrained[combined] | TJEPAOD[combined] | TJEPAOD-untrained[embed] |
|---|---|---|---|---|---|---|---|
| ALOI | **0.636** | 0.548 | 0.566 | 0.561 | 0.549 | 0.585 | 0.550 |
| annthyroid | 0.732 | 0.816 | 0.734 | 0.637 | **0.827** | 0.681 | 0.812 |
| backdoor | 0.754 | 0.727 | 0.870 | 0.843 | 0.818 | 0.878 | **0.898** |
| breastw | 0.975 | 0.986 | 0.955 | 0.945 | **0.986** | 0.977 | 0.984 |
| campaign | 0.721 | 0.707 | **0.763** | 0.666 | 0.734 | 0.701 | 0.711 |
| cardio | 0.740 | 0.929 | 0.811 | 0.887 | 0.936 | 0.942 | **0.943** |
| Cardiotocography | 0.619 | 0.681 | 0.559 | 0.610 | 0.714 | **0.757** | 0.675 |
| celeba | 0.617 | 0.681 | 0.707 | 0.657 | 0.731 | 0.539 | **0.779** |
| census | 0.647 | 0.626 | **0.673** | 0.606 | 0.671 | 0.641 | 0.654 |
| cover | 0.828 | 0.867 | 0.906 | 0.911 | 0.911 | 0.570 | **0.940** |
| donors | 0.800 | 0.773 | 0.824 | 0.664 | 0.816 | 0.708 | **0.838** |
| fault | **0.730** | 0.570 | 0.668 | 0.601 | 0.467 | 0.465 | 0.477 |
| fraud | 0.935 | 0.933 | 0.928 | 0.933 | **0.935** | 0.795 | 0.931 |
| glass | **0.868** | 0.777 | 0.790 | 0.749 | 0.764 | 0.764 | 0.825 |
| Hepatitis | 0.613 | 0.745 | 0.750 | 0.739 | 0.757 | 0.785 | **0.794** |
| http | 0.036 | **1.000** | 0.996 | 0.997 | 0.995 | 0.966 | 0.995 |
| InternetAds | **0.685** | 0.664 | 0.545 | 0.592 | 0.597 | 0.639 | 0.642 |
| Ionosphere | 0.914 | 0.822 | **0.919** | 0.799 | 0.809 | 0.770 | 0.804 |
| landsat | **0.586** | 0.473 | 0.470 | 0.442 | 0.357 | 0.487 | 0.387 |
| letter | **0.875** | 0.640 | 0.840 | 0.608 | 0.577 | 0.554 | 0.588 |
| Lymphography | 0.998 | **1.000** | 0.988 | 0.997 | 0.998 | 0.995 | 0.988 |
| magic.gamma | **0.811** | 0.721 | 0.736 | 0.682 | 0.704 | 0.671 | 0.718 |
| mammography | 0.845 | 0.864 | 0.870 | 0.823 | **0.880** | 0.570 | 0.851 |
| mnist | 0.813 | 0.790 | **0.856** | 0.763 | 0.809 | 0.732 | 0.854 |
| musk | 0.504 | 0.999 | 0.928 | 1.000 | **1.000** | 0.845 | 0.993 |
| optdigits | 0.432 | **0.737** | 0.499 | 0.630 | 0.506 | 0.597 | 0.599 |
| PageBlocks | 0.578 | 0.904 | 0.900 | 0.911 | 0.871 | **0.937** | 0.914 |
| pendigits | 0.721 | **0.953** | 0.794 | 0.791 | 0.922 | 0.793 | 0.924 |
| Pima | 0.632 | 0.676 | 0.648 | 0.661 | 0.660 | 0.670 | **0.680** |
| satellite | 0.687 | **0.714** | 0.633 | 0.642 | 0.600 | 0.644 | 0.624 |
| satimage-2 | 0.946 | 0.988 | 0.968 | **0.994** | 0.976 | 0.979 | 0.968 |
| shuttle | 0.749 | **0.998** | 0.993 | 0.873 | 0.996 | 0.749 | 0.995 |
| skin | 0.719 | 0.669 | 0.607 | 0.760 | 0.620 | **0.827** | 0.788 |
| smtp | 0.972 | 0.946 | 0.955 | 0.961 | **0.978** | 0.781 | 0.975 |
| SpamBase | **0.702** | 0.624 | 0.551 | 0.544 | 0.564 | 0.610 | 0.589 |
| speech | 0.485 | **0.494** | 0.476 | 0.470 | 0.471 | 0.483 | 0.486 |
| Stamps | 0.847 | 0.889 | 0.788 | 0.845 | 0.846 | **0.924** | 0.912 |
| thyroid | 0.955 | **0.979** | 0.930 | 0.885 | 0.935 | 0.933 | 0.970 |
| vertebral | 0.305 | 0.338 | 0.422 | 0.318 | 0.323 | **0.426** | 0.331 |
| vowels | **0.967** | 0.762 | 0.920 | 0.784 | 0.795 | 0.834 | 0.822 |
| Waveform | 0.758 | 0.725 | 0.638 | 0.614 | 0.687 | **0.885** | 0.701 |
| WBC | 0.994 | **0.994** | 0.965 | 0.992 | 0.992 | 0.981 | 0.973 |
| WDBC | **0.998** | 0.991 | 0.901 | 0.984 | 0.985 | 0.979 | 0.978 |
| Wilt | **0.699** | 0.463 | 0.529 | 0.646 | 0.624 | 0.695 | 0.533 |
| wine | **0.995** | 0.773 | 0.481 | 0.688 | 0.696 | 0.753 | 0.720 |
| WPBC | **0.568** | 0.516 | 0.484 | 0.493 | 0.508 | 0.508 | 0.544 |
| yeast | 0.406 | 0.401 | **0.421** | 0.415 | 0.397 | 0.419 | 0.415 |
| **mean** | 0.732 | 0.763 | 0.748 | 0.736 | 0.751 | 0.732 | 0.768 |

#### Time series: trained `TSJEPAOD` (class defaults) against the zero-epoch control, 5 seeds (Wilcoxon signed-rank over series; Holm correction over the 18 rows; default variant: `block[pred]`)

|  | group | variant | series | ROC-AUC trained / untrained | difference [95 % CI] | trained / untrained better | p | p (Holm) | AP trained / untrained |
|---|---|---|---|---|---|---|---|---|---|
| 1 | synthetic | block[pred] | 15 | 0.941 / 0.798 | +0.144 [+0.088, +0.199] | 15 / 0 | <0.001 | <0.001 | 0.560 / 0.345 |
| 2 | synthetic | block[embed] | 15 | 0.928 / 0.932 | -0.003 [-0.007, -0.001] | 2 / 13 | 0.018 | 0.181 | 0.540 / 0.547 |
| 3 | synthetic | block[combined] | 15 | 0.936 / 0.877 | +0.059 [+0.031, +0.091] | 15 / 0 | <0.001 | <0.001 | 0.551 / 0.474 |
| 4 | synthetic | forecast[pred] | 15 | 0.946 / 0.786 | +0.159 [+0.095, +0.225] | 15 / 0 | <0.001 | <0.001 | 0.572 / 0.339 |
| 5 | synthetic | forecast[embed] | 15 | 0.928 / 0.932 | -0.004 [-0.007, -0.002] | 1 / 14 | <0.001 | 0.007 | 0.538 / 0.547 |
| 6 | synthetic | forecast[combined] | 15 | 0.945 / 0.876 | +0.069 [+0.033, +0.110] | 15 / 0 | <0.001 | <0.001 | 0.564 / 0.472 |
| 7 | SKAB | block[pred] | 16 | 0.825 / 0.831 | -0.006 [-0.008, -0.004] | 0 / 16 | <0.001 | <0.001 | 0.626 / 0.637 |
| 8 | SKAB | block[embed] | 16 | 0.825 / 0.828 | -0.003 [-0.007, +0.001] | 4 / 12 | 0.065 | 0.589 | 0.618 / 0.643 |
| 9 | SKAB | block[combined] | 16 | 0.825 / 0.829 | -0.004 [-0.006, -0.002] | 1 / 15 | 0.004 | 0.046 | 0.621 / 0.642 |
| 10 | SKAB | forecast[pred] | 16 | 0.857 / 0.859 | -0.002 [-0.006, +0.000] | 7 / 9 | 0.298 | 1.000 | 0.732 / 0.730 |
| 11 | SKAB | forecast[embed] | 16 | 0.829 / 0.828 | +0.001 [-0.001, +0.003] | 10 / 6 | 0.495 | 1.000 | 0.650 / 0.643 |
| 12 | SKAB | forecast[combined] | 16 | 0.819 / 0.795 | +0.024 [+0.021, +0.028] | 16 / 0 | <0.001 | <0.001 | 0.634 / 0.570 |
| 13 | NAB | block[pred] | 7 | 0.628 / 0.671 | -0.043 [-0.144, +0.041] | 4 / 3 | 0.938 | 1.000 | 0.283 / 0.322 |
| 14 | NAB | block[embed] | 7 | 0.627 / 0.619 | +0.009 [-0.014, +0.035] | 5 / 2 | 0.375 | 1.000 | 0.281 / 0.250 |
| 15 | NAB | block[combined] | 7 | 0.629 / 0.658 | -0.029 [-0.085, +0.011] | 5 / 2 | 0.938 | 1.000 | 0.284 / 0.309 |
| 16 | NAB | forecast[pred] | 7 | 0.585 / 0.664 | -0.080 [-0.239, +0.071] | 4 / 3 | 0.688 | 1.000 | 0.243 / 0.315 |
| 17 | NAB | forecast[embed] | 7 | 0.625 / 0.619 | +0.006 [-0.009, +0.020] | 5 / 2 | 0.297 | 1.000 | 0.281 / 0.250 |
| 18 | NAB | forecast[combined] | 7 | 0.607 / 0.654 | -0.047 [-0.143, +0.042] | 4 / 3 | 0.688 | 1.000 | 0.273 / 0.306 |

<!-- EXTENDED:END -->

<!-- TSVAR:BEGIN -->

### 7.7 JEPA versions of the time-series detector

*Added 2026-11-21. Code: `jepa_od/ts_jepa.py` (`variant`), `jepa_od/_tsjepa_nets.py`. Benchmark: `benchmarks/bench_ts_jepa_final.py --variant`, tables by `benchmarks/make_ts_variant_tables.py`, diagnostic `benchmarks/diag_ts_variant_norm.py`. Raw results: `results/ts_variants/`, and `results/ts_control_*.csv` for the default version.*

**Setup.** The four versions of Section 4.8 at the detector's defaults, each trained and with zero epochs, on the protocols and the 5 seeds of Section 7.6 (15 synthetic series, 16 SKAB files, 7 NAB series). The `lejepa` rows are the runs of Section 7.6. One more pair of runs gives `tsjepa` the masking of its paper (70 % of the patches, at random). The 18 tests of the second table were fixed before the results were looked at: on the default score `block[pred]`, each new version against its own zero-epoch control and against the trained default, in each group, with a Holm correction over the 18. As in Section 7.6 the 16 SKAB files are scored by the same five fitted models and are not independent. The fit times are indicative only: the `lejepa` runs shared the machine with one other job and the others with seven.

**Findings.**

1. **No version beats the default.** On the synthetic series `lejepa` has the highest `pred` score (0.941). `tsjepa` (0.931) and `tjepa` (0.928) are close and not significantly different from it, `ijepa` is lower (0.885). All four are above `LSTMAD` (0.866 in Section 7.2).
2. **Under a regime shift only `lejepa` holds up.** On SKAB it reaches 0.825, against 0.671 for `tjepa`, 0.442 for `tsjepa` and 0.431 for `ijepa`. All three are below it on all 16 files (corrected p < 0.001), and two of them are below chance. These are the three versions that normalise their targets or their encoder output, which is the failure of Section 4.6. A diagnostic run after the results (not one of the 18 tests, SKAB only) supports that reading: `tjepa` with `encoder_norm=False` reaches 0.836 (`pred`), 0.828 (`embed`) and 0.859 (`combined`), level with `lejepa`, while `tsjepa` with `encoder_norm=False`, whose targets stay layer-normalised, remains at 0.495.
3. **NAB.** `tjepa` has the numerically highest `pred` score (0.684 against 0.628 for `lejepa`), but it is the better of the two on only 3 of the 7 series (p = 0.58). The highest score of any version on NAB is the `embed` score of the `tsjepa` network, 0.711 trained and 0.720 untrained. Every version stays below the windowed `IForest` of Section 7.2 (0.825).
4. **The EMA versions depend on training more than the default does.** Their untrained `pred` score is at chance on the synthetic series (0.49 to 0.52, against 0.798 for `lejepa`), and training lifts it to 0.89 to 0.93 on all 15 series (corrected p < 0.001 for each). For `tjepa` training also helps on SKAB (0.671 against 0.426, all 16 files) and numerically on NAB (+0.15, 6 of 7 series; p = 0.047 before and 0.42 after correction). The `embed` score needs no training on the synthetic series in any version (0.89 to 0.94 untrained).
5. **TS-JEPA's own masking is worse for detection.** With 70 % of the patches hidden at random the `pred` score of `tsjepa` falls from 0.931 to 0.804 on the synthetic series (descriptive, not one of the tests). A patch predicted from a third of its window is a weak reference for what is normal.
6. **The EMA versions are cheaper.** They fit in 3.6 to 7.5 s per synthetic series against 14.9 s for `lejepa`, whose SIGReg statistic dominates the training step.

**What this changes.** The default stays `lejepa`. `variant` makes the JEPA version a choice and the comparison reproducible, and the benchmark gives no accuracy reason to choose another one: `tjepa` is the only alternative that is competitive on two of the three groups, and it needs `encoder_norm=False` when the test data may be in another regime (checked on SKAB only). The conclusion of Section 7.6 holds for every version: training earns the prediction-error score on the synthetic series, where an untrained `embed` score is at most 0.05 below the trained default, and it earns nothing demonstrable on NAB.

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

<!-- TSVAR:END -->

---

## 8. Limitations and next steps

* **Training has no demonstrated benefit on tabular data.** On all 47 ADBench datasets the zero-epoch controls score at least as high as the trained detectors (Section 7.6). For time series, training helps the default score on synthetic series only. One training budget was tested; a claim that JEPA training helps outlier detection under another budget needs the same control.
* **The JEPA versions of `TSJEPAOD` share the detector's masking, sizes and budget.** They are re-implementations of each paper's network, target mechanism and loss, not of its full training setup (Section 4.8), so Section 7.7 compares recipes inside this detector and says nothing about the papers' own results. The repair of `tjepa` by `encoder_norm=False` was checked on SKAB only.
* **The T-JEPA comparison is a re-implementation at a small budget.** `TJEPAOD` uses the released code's default architecture and `JEPAOD`'s training budget, not the authors' code or their per-dataset tuned configurations, and T-JEPA itself has no outlier score. τ-JEPA was not read and is not implemented.
* **The time-series benchmarks are modest** (7 NAB series, SKAB valve1, synthetic series). The tabular side now covers the 47 ADBench classical datasets (Section 7.6). TSB-AD with VUS-PR, the primary metric there, has not been run. The ablations of Section 7.3, including the SIGReg-against-EMA comparison, were run on small subsets only.
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
* Thimonier, H., De Melo Costa, J. L., Popineau, F., Rimmel, A., Doan, B.-L. (2025). *T-JEPA: Augmentation-Free Self-Supervised Learning for Tabular Data.* ICLR 2025. arXiv:2410.05016. https://arxiv.org/abs/2410.05016. Code: https://github.com/jose-melo/t-jepa
* *τ-JEPA* (working paper, indexed title "self-supervised learning for fraud detection in tabular data"; authors not visible to us). https://callforpapers.institutlouisbachelier.org/Papers/45d23018-3e36-4042-9c53-c905c84b8f5f.pdf (login required; not read in full)
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
