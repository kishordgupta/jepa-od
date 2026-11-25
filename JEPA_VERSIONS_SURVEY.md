# JEPA versions: what else could join the `variant` family

Survey of 2026-11-21. Question: which published JEPA training recipes, beyond the four that
`TSJEPAOD(variant=...)` already has (`'lejepa'`, `'ijepa'`, `'tsjepa'`, `'tjepa'`), could be added as
further versions. The survey itself is a reading of the papers against the detector's code.

**Status, 2026-11-22.** Twelve of the recipes below are now implemented as `variant` values (version
0.4.0): the four of group A, the four of group B, the three of group C (as versions, not as a
regulariser option), and LeWorldModel without its action input, which group B first folded into
LeNEPA (`'lewm'`; it differs from `'lenepa'` in where SIGReg is applied and in what is predicted).
`REPORT.md` section 4.9 describes each implementation and every choice the papers leave open,
section 7.8 has the benchmark. The recommendation at the end of this file was written before that.

**Outcome.** `kerjepa` is level with the default in all three benchmark groups at about half the
fitting time. `vicreg`, `lambdajepa` and `lpjepa` match the default under a regime shift, which
answers the question put to `vicreg` below: the default's robustness comes from its un-normalised
shared-encoder target, not from SIGReg. `cjepa` does not repair the EMA version. `salt` and `cfjepa`
are numerically ahead on NAB only. `lenepa` and `cfjepa` have no usable prediction-error score in
this detector. Numbers and tests: `REPORT.md` section 7.8.

**Status, 2026-11-25 (version 0.5.0).** A second pass over the larger list (404 entries by then) read 24
abstracts and 11 method sections of papers the first pass had judged from their titles. Ten more recipes are
implemented as `variant` values, in the prototype only: `ststjepa` (STST-JEPA, arXiv 2607.06629), `hepa` (HEPA,
2605.11130), `subjepa` (Sub-JEPA, 2605.09241), `straight` (temporal straightening, 2603.12231), `motionjepa`
(MotionJEPA, 2609.23881), `njepa` (N-JEPA, 2507.15216), `gmmjepa` (GMM-anchored JEPA, 2602.09040), `variational`
(VJEPA, 2601.14354), `phijepa` (PhiJEPA, 2611.27516) and `harjepa` (HAR-JEPA, 2607.16350). `REPORT.md` section 4.10
says for each what is the paper's and what is ours; section 7.11 has the benchmark. Of the papers in "Not read"
below, STST-JEPA is now implemented and W-JEPA was read (wavelet tokens that cover different time spans: not a
version). SC-JEPA (MTS-JEPA) was read again and is still in group D: besides what the table says, the text read
gives none of its nine loss weights, its codebook size or its temperature. Var-JEPA (2603.20111) and VJEPA
(2601.14354, implemented as `variational`) are two different papers.

## How the candidates were found and judged

* Two curated lists were read in full: [AbdelStark/awesome-jepa](https://github.com/AbdelStark/awesome-jepa)
  (about 80 entries) and [AI-in-Transportation-Lab/awesome-jepa](https://github.com/AI-in-Transportation-Lab/awesome-jepa)
  (about 250 titles, mostly from 2026). Together they name roughly 320 JEPA papers, about 35 of them on
  1-D signals (time series, EEG, ECG, audio).
* 19 papers were read for this survey (most of the text, not every appendix). The rest were judged from
  their title or abstract.
* A recipe counts as **addable as a version** when it can be written as: patch tokens of a sliding
  window, an encoder, a predictor of hidden-token embeddings, and its own choice of target network,
  anti-collapse term, loss or network. That is the contract of the existing four: the masking, sizes
  (64-d, 4 heads, 2 layers), optimiser, budget (10 epochs, at most 5000 windows) and the three scores stay
  the detector's own. A new network only has to provide `token_embeddings`, `forward` and
  `multi_mask_errors`, as `TSJEPACore` and `TJEPACore` do.
* Every version would be re-implemented from its paper, as `'tsjepa'` and `'tjepa'` were. No released
  code is copied (for several repositories no licence was found, and LeJEPA's own is non-commercial).

## Count

| Group | Versions | Family size if all are added |
|---|---|---|
| In the detector today | 4 | 4 |
| A. Reuse the current networks; only target, loss or regulariser changes | 4 | 8 |
| B. Fit the contract but need a new network path | 4 | 12 |
| C. Better exposed as a regulariser option of the default than as versions | 3 | (not versions) |
| D. Not addable as a version | about 300 papers | |

## A. Reuse the current networks (4)

| Proposed key | Paper | Recipe in the paper | What changes in the detector | What it would test | Size |
|---|---|---|---|---|---|
| `'vjepa'` | V-JEPA, Bardes et al., [arXiv 2404.08471](https://arxiv.org/abs/2404.08471) | EMA target 0.998 → 1, **L1** loss, no regulariser | `'ijepa'` with an L1 loss and momentum 0.998 | little: a near-duplicate of `'ijepa'` (loss and momentum start only); `'tsjepa'` already has the L1 loss | about 10 lines |
| `'cjepa'` | C-JEPA, Mo & Tong, [arXiv 2410.19560](https://arxiv.org/abs/2410.19560) | I-JEPA plus VICReg terms (invariance 25, variance 25, covariance 1) at overall weight 0.001, on projected context embeddings averaged per mask block | `'ijepa'` plus variance / covariance terms on mean-pooled context embeddings through a small projector; the invariance term needs a second mask draw per window | whether a regulariser rescues the EMA versions under SKAB's regime shift | about 40 lines |
| `'vicreg'` | EB-JEPA, Terver et al. (Meta FAIR), [arXiv 2602.03604](https://arxiv.org/abs/2602.03604) | one encoder for both sides, **no EMA**, L2 prediction + VICReg variance and covariance on a projector output (image defaults 1 and 80) | the `'lejepa'` path with SIGReg replaced by variance + covariance terms | whether the default's advantage comes from SIGReg itself or from having no EMA teacher | about 25 lines |
| `'salt'` | SALT, Li et al. (Apple), [arXiv 2509.24317](https://arxiv.org/abs/2509.24317) | stage 1: a teacher trained by masked reconstruction of the input; stage 2: teacher **frozen**, student predicts its latents with L1; no EMA, no regulariser | two training stages inside the same epoch budget (the paper's best split is about 1 : 5 teacher : student) | a fourth way to get targets: neither SIGReg nor a moving teacher | about 70 lines |

Caveats: C-JEPA's weight is sensitive in its own ablation (ten times larger is much worse, a hundred times
larger collapses), and its name is also used for Causal-JEPA. EB-JEPA's examples are two-view image and
video models, not masked prediction, and it uses no stop-gradient, where our `'lejepa'` path keeps one on
the targets (REPORT.md, section 2.3). SALT's teacher is trained in value space, which is the bias latent
prediction is meant to avoid.

## B. Fit the contract but need a new network path (4)

| Proposed key | Paper | Recipe in the paper | What is new in code | Reasons to wait |
|---|---|---|---|---|
| `'lenepa'` | LeNEPA, Chemeris, Jin, Balestriero, MiLeTS'26 workshop, [arXiv 2607.00958](https://arxiv.org/abs/2607.00958) | **causal** Transformer predicts the tokeniser embedding of the next patch; squared L2 in a shared projector's space; SIGReg computed per sample over its own tokens; no EMA, no stop-gradient, no masks | a causal core; `pred` becomes the next-latent error of each patch (the first patch has none); `embed` unchanged. About 150 lines | workshop paper; not evaluated on anomaly detection; slightly behind an ECG-tuned masked JEPA on PTB-XL (AUROC 0.880 against 0.891); its 8-layer encoder becomes 2 layers here |
| `'sjepa'` | stochastic-latent JEPA, Giakoumoglou et al., [arXiv 2611.09302](https://arxiv.org/abs/2611.09302) | I-JEPA plus a Gaussian latent given to the predictor as a prefix token (prior from the context, posterior from context and target) and a KL term; EMA kept | prior and posterior heads, a KL term, a choice of which latent to use when scoring. About 120 lines | posted 6 Nov 2026; default KL weight not found in the text read; name collides with Signal-JEPA |
| `'cfjepa'` | CF-JEPA, Lee & Sim, [arXiv 2606.07031](https://arxiv.org/abs/2606.07031) | no masks: random crops as context, the rest of the series split into three zones, one linear predictor per zone; L1 on unit-normalised vectors with a weight annealed to zero; VICReg variance / covariance (0.081, 0.076) and a multi-scale invariance term across crops (1.101); EMA from 0.983; encoder of 5 multi-scale dilated depthwise-convolution blocks with one embedding per time step | a per-time-step convolutional core and a crop sampler. Only the `embed` score follows the paper (it scores by k-NN or Gaussian distance of the EMA encoder's embeddings to training embeddings). About 250 lines | the largest job; how a crop's embeddings are mapped onto a zone of another length is not given in the main text; the code is announced, not released |
| `'vjepa21'` | V-JEPA 2.1, Meta FAIR, [arXiv 2603.14482](https://arxiv.org/abs/2603.14482) | adds a loss on the visible tokens, weighted by 0.5 / sqrt(distance to the nearest hidden token), and applies the losses at four encoder depths; L1; fixed EMA 0.99925 | the dense loss is about 20 lines on top of `'vjepa'`; deep supervision needs intermediate outputs and a fusion layer | designed for very deep encoders: with 2 layers the deep supervision has two levels (correction of 2026-11-22: a fixed momentum of 0.99925 is not an obstacle, the target still moves about 40 % of the way to the online encoder in the 10-epoch budget) |

LeWorldModel (Maes et al., [arXiv 2603.19312](https://arxiv.org/abs/2603.19312), NeurIPS 2026) is not counted
separately: its two loss terms (next-embedding error and SIGReg per time step, no EMA, no stop-gradient)
are the LeNEPA recipe with a predictor conditioned on robot actions, and the paper reports no run without
actions. The recipe of "Does JEPA Pretraining Help Time Series Forecasting?" ([arXiv 2609.31680](https://arxiv.org/abs/2609.31680):
one encoder on the past and the future window, SIGReg, no EMA) is already available as
`variant='lejepa', mask_mode='forecast'`.

## C. Better as a regulariser option of the default (3)

These keep the LeJEPA recipe and replace SIGReg by another term on the same embeddings. They would fit a
`regularizer=` parameter better than new versions.

| Paper | Term | Notes |
|---|---|---|
| KerJEPA, Zimmermann et al., [arXiv 2512.19605](https://arxiv.org/abs/2512.19605) | kernel discrepancy (MMD or Stein) to a Gaussian prior in closed form, no random slices; SIGReg is shown to be one member of the family | quadratic in batch size; the paper reports collapse without target standardisation; no code found |
| Rectified LpJEPA, Kuang et al., ICML 2026, [arXiv 2602.01456](https://arxiv.org/abs/2602.01456) | sliced Wasserstein match to a rectified generalised Gaussian; needs a ReLU on the output | gives sparse, non-negative embeddings, which changes what the Mahalanobis `embed` score measures; on ImageNet-100 its denser settings are within -0.08 to +0.28 points of LeJEPA and its sparsest ones 2 to 3 points below |
| lambda-JEPA (SACReg), Demirel et al., [arXiv 2609.35288](https://arxiv.org/abs/2609.35288) | closed-form Gaussian term (mean, trace and log-determinant of the covariance) on pooled features, added on the backbone output | multi-view recipe; weights are calibrated by a procedure, not fixed numbers |

Not recommended: UR-JEPA ([arXiv 2606.01443](https://arxiv.org/abs/2606.01443)), a single-author preprint whose
gains over SIGReg are within noise on two of three datasets and which reports trivial minimisers of its own
regulariser. Var-JEPA ([arXiv 2603.20111](https://arxiv.org/abs/2603.20111)) replaces the whole objective by an
ELBO with decoders for both views and is evaluated on tabular data only: a different model, not a version.

## D. Not addable as a version

| Papers | What they need that the detector does not have |
|---|---|
| MTS-JEPA, renamed SC-JEPA in v2 ([arXiv 2602.04643](https://arxiv.org/abs/2602.04643)) | a soft codebook, a coarse one-token view, a reconstruction branch, and a *supervised* head for anomaly prediction; its own ablation says the model collapses without the codebook or the reconstruction |
| CHARM ([arXiv 2605.31580](https://arxiv.org/abs/2605.31580)) | a text description of every channel, embedded by a frozen language model |
| ECG and EEG models: ER-JEPA ([2607.01145](https://arxiv.org/abs/2607.01145)), ECG-JEPA ([2410.08559](https://arxiv.org/abs/2410.08559)), Signal-JEPA ([2403.11772](https://arxiv.org/abs/2403.11772)), Brain-JEPA, EEG-VJEPA; audio models: A-JEPA, Audio-JEPA, WavJEPA | tokens per lead or channel, a second (channel) level, spectrogram or waveform frames, and masking over channels (all but ER-JEPA judged from abstracts). With the detector's own tokens and masking held fixed, what is left of these recipes is an EMA teacher with an L1 or L2 loss, which is `'ijepa'` or `'vjepa'`. A channel-masking mode for multivariate series would be a separate feature, not a version |
| V-JEPA 2-AC, LeWorldModel as published, ACTS-WM, BeliefGraph-JEPA | action inputs |
| LaT-PFN | in-context forecasting with a prior-fitted network |
| Causal-JEPA, D-JEPA, IWM, MC-JEPA, LLM-JEPA, VL-JEPA, Graph-JEPA, Point-JEPA and the rest | object slots, generation, image transformations, optical flow, text, graphs, point clouds |

## What these papers report that bears on our results

* CF-JEPA re-ran ten self-supervised methods on KPI and Yahoo. **TS-JEPA is last of the ten** under both
  scorings (average F1 0.188 with k-NN, 0.041 with Gaussian distance); CF-JEPA has 0.501 with k-NN and
  TimesURL 0.521 with Gaussian distance. That matches our SKAB result for `'tsjepa'` (REPORT.md, section 7.7).
  Their protocol scores against training embeddings and thresholds at mean + 4 standard deviations.
* In CF-JEPA's own harness, masked latent prediction is behind forward prediction (UCR accuracy 0.789
  against 0.819); the authors note the masked variant was not tuned separately.
* "Does JEPA Pretraining Help Time Series Forecasting?" finds the effect depends on the backbone: gains on
  some (FEDformer on Weather, +6 %), losses on others (Autoformer on Weather, -16 %).
* ER-JEPA (ECG), which relies on the EMA teacher alone, reports that about 1 % of 500 repeated runs gave
  an anomalous, poorer pre-training.

## Names that mean two things

C-JEPA (Mo & Tong's VICReg version; also Causal-JEPA), S-JEPA (Signal-JEPA for EEG; also the stochastic-latent
JEPA), T-JEPA (tabular; also trajectory similarity), ER-JEPA (ECG; also an experience-replay paper on
language), D-JEPA (denoising; also "design-recoverable"). Keys for `variant` have to avoid these.

## Not read

About a dozen 2026 papers on 1-D signals are known from their titles only: W-JEPA (wearables), DES-JEPA
(event sequences), Phys-JEPA, S2S-JEPA (climate), Distributed JEPA (energy load), a cross-domain
pre-training paper for industrial series (arXiv 2611.02489), CardioState-JEPA, STST-JEPA, SPERA and Laya
(EEG), Phenomenon-Graph JEPA. Some may hold a further recipe. Parts of the papers that were read came through
summaries of long pages (appendices mostly); the numbers above are from the main text or tables.

## Recommendation

Add four to the prototype and benchmark them with the protocol of section 7.7 (5 seeds, trained and
untrained, synthetic / SKAB / NAB) before anything goes to the PyOD branch:

1. `'vicreg'` and `'salt'`: the two ways of getting targets that the family lacks.
2. `'cjepa'`: tests whether a regulariser fixes the EMA versions' failure under regime shift.
3. `'lenepa'`: the one time-series recipe without an EMA teacher, and the only next-latent one.

`'vjepa'` costs almost nothing but is a near-duplicate of `'ijepa'`; add it only if the name is wanted in
the list. Leave `'cfjepa'` until its code is released, `'sjepa'` until it is older than two weeks, and
`'vjepa21'` because it is built for deep encoders. Decide on a `regularizer` option after `'vicreg'` shows whether the
choice of term matters at all.

The comment on yzhao062/pyod#795 describes four versions. More versions make a larger pull request to
review, and so far no version beats the default.

**2026-11-22, later.** All twelve implemented versions were ported to the PyOD branch `feature/ts-jepa` as a
second commit (`integration/PYOD_PR_CHECKLIST.md`). That commit is about as large as the first one.
