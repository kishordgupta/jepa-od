# -*- coding: utf-8 -*-
"""TS-JEPA-OD: time-series anomaly detection with a Joint-Embedding
Predictive Architecture (PyOD time-series ``BaseDetector`` API).

A sliding window of ``window_size`` timestamps is cut into
``window_size // patch_len`` patches (all channels of a patch form one
token). The JEPA hides some patches from the context encoder and predicts
their *latent* embeddings; timestamps whose patch embedding cannot be
predicted from the surrounding (or preceding, in ``'forecast'`` mode)
context are anomalous. Per-patch errors are mapped back to the timestamps
they cover and averaged over all windows / masks, following PyOD's
masked-score workflow (timestamps never scored get ``threshold_``).

Compared with raw-space forecasting (LSTMAD) or reconstruction, the error
lives in representation space: unpredictable high-frequency noise is not
penalised and the detector focuses on breaks of temporal structure. A
second score is the Mahalanobis distance of every patch embedding to the
training embeddings (``score='embed'``).

The detector is a family: ``variant`` selects which JEPA trains the
network (LeJEPA, I-JEPA, TS-JEPA, T-JEPA and twenty-two further versions,
see :class:`TSJEPAOD`). Scores, training budget and the PyOD interface are
the same for all of them.
"""
# License: BSD 2 clause

import math

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.utils.validation import check_is_fitted

from pyod.models.base import BaseDetector
from pyod.models._ts_utils import validate_ts_input

from ._nets import JEPACore, sample_masks, eval_masks, _MASK_MODES
from ._recipes import RECIPES, momentum_at
from ._sigreg import sigreg_loss, ema_update
from ._tjepa_nets import TJEPACore
from ._tsjepa_nets import TSJEPACore

_SCORES = ('pred', 'embed', 'combined')
# the first four versions are implemented in this module, the others are
# recipes (``_recipes.py``)
_VARIANTS = ('lejepa', 'ijepa', 'tsjepa', 'tjepa') + tuple(RECIPES)
# what ``None`` means for the two parameters whose published value differs
# between the versions: (encoder_norm, ema_momentum)
_VARIANT_DEFAULTS = {'lejepa': (False, None), 'ijepa': (False, 0.996),
                     'tsjepa': (True, 0.998), 'tjepa': (True, 0.996)}
_VARIANT_DEFAULTS.update({name: (r.encoder_norm, r.ema_momentum)
                          for name, r in RECIPES.items()})


class TSJEPAOD(BaseDetector):
    """Joint-Embedding Predictive Architecture for time-series anomaly
    detection.

    ``variant`` selects the JEPA that trains the network:

    ============  ===========================  =========================
    ``variant``   network                      collapse prevention, loss
    ============  ===========================  =========================
    ``'lejepa'``  per-patch linear tokenizer,  SIGReg on one shared
                  learned positions, pre-norm  encoder (LeJEPA), squared
                  encoder, narrow predictor    error
    ``'ijepa'``   the same network             EMA target encoder with
                                               normalised targets
                                               (I-JEPA), squared error
    ``'tsjepa'``  convolutional patch          EMA copy of the whole
                  tokenizer, sinusoidal        encoder with normalised
                  positions, encoder with a    targets (TS-JEPA),
                  final LayerNorm, predictor   absolute error
                  as wide as the encoder
    ``'tjepa'``   per-patch linear tokenizer,  EMA copy of the whole
                  sinusoidal positions, a      encoder (T-JEPA), squared
                  [REG] token, post-norm       error
                  encoder with a normalised
                  output
    ============  ===========================  =========================

    TS-JEPA and T-JEPA are re-implemented from their papers and the
    defaults of their released code; T-JEPA is a tabular method and is
    applied to patch tokens here. From each version this class takes the
    network, the target mechanism and the loss. The masking
    (``mask_mode``, ``mask_ratio``), the model sizes, the optimiser and
    the training budget are this detector's own and are the same for all
    versions, and the anomaly scores are not part of any of the
    papers. The masking of TS-JEPA's paper is ``mask_mode='random',
    mask_ratio=0.7``.

    Twelve further versions are re-implemented from their papers
    (constants and what was adapted: ``jepa_od/_recipes.py``; benchmark:
    REPORT.md, section 7.8):

    ================  ====================================================
    ``variant``       what trains the network
    ================  ====================================================
    ``'vjepa'``       V-JEPA: ``'ijepa'`` with an absolute-error loss
    ``'vjepa21'``     V-JEPA 2.1: also predicts the visible patches, at
                      several encoder depths
    ``'cjepa'``       C-JEPA: ``'ijepa'`` plus VICReg terms on the context
                      embeddings
    ``'sjepa'``       stochastic-latent JEPA: ``'ijepa'`` with a Gaussian
                      latent for the predictor and a KL term
    ``'salt'``        SALT: a teacher trained by masked reconstruction and
                      then frozen provides the targets
    ``'vicreg'``      ``'lejepa'`` with VICReg variance and covariance
                      terms in place of SIGReg (EB-JEPA)
    ``'kerjepa'``     KerJEPA: a closed-form kernel discrepancy in place
                      of SIGReg, standardised targets
    ``'lpjepa'``      Rectified LpJEPA: non-negative embeddings matched to
                      a rectified Laplace distribution
    ``'lambdajepa'``  lambda-JEPA: a Gaussian divergence (SACReg) in place
                      of SIGReg
    ``'lenepa'``      LeNEPA: a causal Transformer predicts the embedding
                      of the next patch; SIGReg per window; no masks
    ``'lewm'``        LeWorldModel without actions: next-patch embedding
                      prediction; SIGReg per patch position; no masks
    ``'cfjepa'``      CF-JEPA: crops of the window as context, forward
                      prediction of three zones, convolutional encoder
                      with one embedding per time step; no masks
    ================  ====================================================

    ``'lenepa'``, ``'lewm'`` and ``'cfjepa'`` do not hide patches:
    ``mask_mode`` and ``mask_ratio`` are ignored, their ``pred`` score is
    the error of predicting a patch from the patches before it, and the
    first patch of a window (for ``'cfjepa'`` the leading crop) gets none.

    Ten more versions were added in version 0.5.0 (constants and what was
    adapted: ``jepa_od/_recipes2.py``; benchmark: REPORT.md, section
    7.11):

    =================  ===================================================
    ``variant``        what trains the network
    =================  ===================================================
    ``'ststjepa'``     STST-JEPA: the targets come from an EMA copy of the
                       tokenizer, not of the encoder; a reconstruction
                       term on the predictions
    ``'hepa'``         HEPA: a causal encoder and a predictor conditioned
                       on the horizon predict the embedding of the next
                       ``dt`` patches; SIGReg on the predictions; no masks
    ``'subjepa'``      Sub-JEPA: ``'lewm'`` with SIGReg in frozen random
                       subspaces; no masks
    ``'straight'``     temporal straightening: next-patch prediction with
                       a stop-gradient and a curvature term; no masks
    ``'motionjepa'``   MotionJEPA: ``'lewm'`` plus an embedding of patch
                       differences (DISReg); no masks
    ``'njepa'``        N-JEPA: ``'ijepa'`` with a second predictor whose
                       mask tokens carry Gaussian noise
    ``'gmmjepa'``      GMM-anchored JEPA: ``'ijepa'`` plus a cluster head
                       trained on the soft posteriors of a frozen Gaussian
                       mixture of the patches
    ``'variational'``  variational JEPA (VJEPA): the predictor returns a
                       Gaussian; ``pred`` is a negative log-likelihood
    ``'phijepa'``      PhiJEPA: two prediction stages, towards a corrupted
                       and then the clean view; cosine distance
    ``'harjepa'``      HAR-JEPA: an EMA target that is not normalised, and
                       VICReg terms with a norm term on the predictions
    =================  ===================================================

    ``'hepa'``, ``'subjepa'``, ``'straight'`` and ``'motionjepa'`` do not
    hide patches either. The untrained networks of the last three are
    that of ``'lewm'``.

    Parameters
    ----------
    window_size : int, optional (default=100)
        Sliding window length (must be a multiple of ``patch_len``).

    patch_len : int, optional (default=10)
        Timestamps per token; a window has ``window_size // patch_len``
        tokens.

    step : int, optional (default=1)
        Stride between scoring windows.

    train_step : int or None, optional (default=None)
        Stride between training windows (``None`` -> ``patch_len // 2``).

    max_train_windows : int, optional (default=5000)
        Random cap on the number of training windows per fit. Together
        with ``epochs`` this bounds the number of gradient steps; long
        contaminated series are otherwise over-fitted (the anomalous
        regime gets absorbed into the model, see REPORT.md).

    d_model, n_heads, n_layers, pred_layers, dropout
        Transformer sizes, see :class:`jepa_od.JEPAOD`.

    mask_mode : {'block', 'random', 'forecast', 'mixed'}, default 'block'
        'block' hides one contiguous run of patches anywhere in the window
        (bidirectional context); 'forecast' always hides the last patches
        (causal, online-friendly); 'mixed' trains on both.

    mask_ratio : float, optional (default=0.3)
        Fraction of patches hidden.

    n_mask_samples : int or None, optional (default=None)
        Masks per window at inference. ``None`` enumerates every block
        position (deterministic, complete coverage).

    variant : str, optional (default='lejepa')
        The JEPA version, one of the twenty-six names in the tables above.

    variant_params : dict or None, optional (default=None)
        Overrides of the constants of a version of the second or third
        table, e.g. ``{'var_weight': 1.0, 'cov_weight': 40.0}`` for
        ``'vicreg'``. The names and published values are in
        ``jepa_od/_recipes.py`` and ``jepa_od/_recipes2.py``; the resolved
        values are in ``variant_params_`` after ``fit``.

    sigreg_weight, sigreg_slices
        SIGReg settings of ``variant='lejepa'``, see
        :class:`jepa_od.JEPAOD`. ``sigreg_slices`` is also the number of
        random projections of ``'lenepa'``, ``'lewm'``, ``'lpjepa'``,
        ``'hepa'``, ``'subjepa'`` and ``'motionjepa'``; the other versions
        ignore both.

    ema_momentum : float or None, optional (default=None)
        Start value of the target-encoder momentum of the EMA versions; it
        rises linearly to 1 during training. ``None`` is the published
        value: 0.996 for ``'ijepa'``, ``'tjepa'``, ``'cjepa'`` and
        ``'sjepa'``, 0.998 for ``'tsjepa'`` and ``'vjepa'``, 0.99925 (held
        fixed) for ``'vjepa21'``, 0.983 (cosine schedule) for
        ``'cfjepa'`` and 0.9996 (rising to 0.9999) for ``'ststjepa'``;
        ``'njepa'``, ``'gmmjepa'``, ``'variational'``, ``'phijepa'`` and
        ``'harjepa'`` use 0.996. Ignored by the versions without an EMA
        target.

    recon_weight : float, optional (default=0.0)
        Weight of the reconstruction anchor, see :class:`jepa_od.JEPAOD`.
        Only for ``'lejepa'`` and ``'ijepa'``.

    score : {'pred', 'embed', 'combined'}, optional (default='pred')

    score_aggregation : {'mean', 'max'}, optional (default='mean')
        How the (window, mask, patch) scores covering a timestamp are
        aggregated.

    epochs : int, optional (default=10)

    lr : float, optional (default=1e-3)

    weight_decay : float, optional (default=1e-4)

    batch_size : int, optional (default=64)

    grad_clip : float or None, optional (default=1.0)

    encoder_norm : bool or None, optional (default=None)
        Normalised encoder output. ``None`` is True, the published form,
        for ``'tsjepa'`` and ``'tjepa'`` and False for all other versions
        (``'cfjepa'`` has no such layer). For ``'lejepa'`` keep it off: a final LayerNorm gains
        ~0.04 ROC-AUC when test data follow the training distribution
        (synthetic, NAB) but saturates shifted inputs and loses ~0.3
        ROC-AUC when the test series has a different operating regime
        (SKAB), where it also inverts the ``'embed'`` score.

    device : str, optional (default='auto')
        'auto' -> CUDA if available else CPU; 'mps' for Apple silicon.

    random_state : int, optional (default=42)

    verbose : int, optional (default=0)

    contamination : float, optional (default=0.1)

    Attributes
    ----------
    decision_scores_ : numpy array of shape (n_timestamps,)
    threshold_ : float
    labels_ : numpy array of shape (n_timestamps,)
    model_ : torch.nn.Module
    """

    def __init__(self, window_size=100, patch_len=10, step=1,
                 train_step=None, max_train_windows=5000,
                 d_model=64, n_heads=4, n_layers=2, pred_layers=1,
                 dropout=0.0,
                 mask_mode='block', mask_ratio=0.3, n_mask_samples=None,
                 variant='lejepa', sigreg_weight=0.05,
                 sigreg_slices=128, ema_momentum=None, recon_weight=0.0,
                 score='pred', score_aggregation='mean',
                 epochs=10, lr=1e-3, weight_decay=1e-4, batch_size=64,
                 grad_clip=1.0, encoder_norm=None, variant_params=None,
                 device='auto', random_state=42, verbose=0,
                 contamination=0.1):
        super(TSJEPAOD, self).__init__(contamination=contamination)
        self.window_size = window_size
        self.patch_len = patch_len
        self.step = step
        self.train_step = train_step
        self.max_train_windows = max_train_windows
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.pred_layers = pred_layers
        self.dropout = dropout
        self.mask_mode = mask_mode
        self.mask_ratio = mask_ratio
        self.n_mask_samples = n_mask_samples
        self.variant = variant
        self.sigreg_weight = sigreg_weight
        self.sigreg_slices = sigreg_slices
        self.ema_momentum = ema_momentum
        self.recon_weight = recon_weight
        self.score = score
        self.score_aggregation = score_aggregation
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.grad_clip = grad_clip
        self.encoder_norm = encoder_norm
        self.variant_params = variant_params
        self.device = device
        self.random_state = random_state
        self.verbose = verbose

    # ------------------------------------------------------------------
    def _get_min_length(self):
        return self.window_size + self.patch_len

    def _resolve_device(self):
        if self.device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(self.device)

    def _validate_params(self):
        if self.window_size % self.patch_len != 0:
            raise ValueError("window_size (%d) must be a multiple of "
                             "patch_len (%d)" % (self.window_size,
                                                  self.patch_len))
        if self.window_size // self.patch_len < 2:
            raise ValueError("window_size must hold at least 2 patches")
        if self.variant not in _VARIANTS:
            raise ValueError("variant must be one of %s" % (_VARIANTS,))
        if self.recon_weight > 0 and self.variant not in ('lejepa', 'ijepa'):
            raise ValueError("recon_weight is only available for the "
                             "'lejepa' and 'ijepa' variants")
        if self.variant_params:
            known = RECIPES[self.variant].params \
                if self.variant in RECIPES else {}
            unknown = sorted(set(self.variant_params) - set(known))
            if not isinstance(self.variant_params, dict) or unknown:
                raise ValueError(
                    "variant_params of %r must be a dict with keys from %s; "
                    "got %r" % (self.variant, sorted(known),
                                self.variant_params))
        if self.mask_mode not in _MASK_MODES:
            raise ValueError("mask_mode must be one of %s" % (_MASK_MODES,))
        if self.score not in _SCORES:
            raise ValueError("score must be one of %s" % (_SCORES,))
        if self.score_aggregation not in ('mean', 'max'):
            raise ValueError("score_aggregation must be 'mean' or 'max'")
        if not 0.0 < self.mask_ratio < 1.0:
            raise ValueError("mask_ratio must be in (0, 1)")

    @staticmethod
    def _windows(X, window_size, step):
        """(n, C) -> (n_windows, window_size, C) float32 view-copy."""
        n = X.shape[0]
        n_w = (n - window_size) // step + 1
        idx = np.arange(window_size)[None, :] + step * np.arange(n_w)[:, None]
        return np.ascontiguousarray(X[idx], dtype=np.float32)

    def _to_patches(self, W):
        """(n_w, window_size, C) -> (n_w, P, patch_len * C)."""
        n_w, L, C = W.shape
        P = L // self.patch_len
        return W.reshape(n_w, P, self.patch_len * C)

    def _build_model(self, n_tokens, n_channels):
        """The network of ``variant`` (all share one scoring interface)."""
        token_dim = self.patch_len * n_channels
        size = dict(d_model=self.d_model, n_heads=self.n_heads,
                    n_layers=self.n_layers, pred_layers=self.pred_layers,
                    dropout=self.dropout, encoder_norm=self.encoder_norm_)
        if self.variant in RECIPES:
            return RECIPES[self.variant].build(self, n_tokens, n_channels,
                                               size, self.variant_params_)
        if self.variant == 'tsjepa':
            return TSJEPACore(n_tokens=n_tokens, patch_len=self.patch_len,
                              n_channels=n_channels, **size)
        if self.variant == 'tjepa':
            return TJEPACore(n_tokens=n_tokens, in_dim=token_dim,
                             dim_feedforward=4 * self.d_model,
                             pred_dim=self.d_model, pred_heads=self.n_heads,
                             pred_feedforward=4 * self.d_model,
                             n_reg_tokens=1, **size)
        return JEPACore(
            n_tokens=n_tokens, in_dim=token_dim,
            target_mode='sigreg' if self.variant == 'lejepa' else 'ema',
            recon_dim=token_dim if self.recon_weight > 0 else None, **size)

    # ------------------------------------------------------------------
    def fit(self, X, y=None):
        """Fit detector on a (mostly normal) time series.

        Parameters
        ----------
        X : array-like of shape (n_timestamps,) or (n_timestamps, n_channels)

        y : Ignored

        Returns
        -------
        self : object
        """
        self._validate_params()
        X = validate_ts_input(X)
        n, C = X.shape
        if n < self._get_min_length():
            raise ValueError("Time series length %d is shorter than minimum "
                             "required length %d (window_size=%d)"
                             % (n, self._get_min_length(), self.window_size))
        self._set_n_classes(y)

        torch.manual_seed(self.random_state)
        rng = np.random.RandomState(self.random_state)
        device = self._resolve_device()

        self.n_channels_ = C
        self.X_mean_ = X.mean(0)
        self.X_std_ = X.std(0) + 1e-8
        Xs = (X - self.X_mean_) / self.X_std_

        P = self.window_size // self.patch_len
        self.n_tokens_ = P
        self.n_mask_ = int(np.clip(round(self.mask_ratio * P), 1, P - 1))
        norm, momentum = _VARIANT_DEFAULTS[self.variant]
        self.encoder_norm_ = (norm if self.encoder_norm is None
                              else bool(self.encoder_norm))
        self.ema_momentum_ = (momentum if self.ema_momentum is None
                              else self.ema_momentum)
        self.variant_params_ = dict(
            RECIPES[self.variant].params if self.variant in RECIPES else {},
            **(self.variant_params or {}))

        tstep = self.train_step or max(1, self.patch_len // 2)
        W = self._windows(Xs, self.window_size, tstep)
        if len(W) > self.max_train_windows:
            keep = rng.choice(len(W), self.max_train_windows, replace=False)
            W = W[np.sort(keep)]
        train_patches = torch.from_numpy(self._to_patches(W))

        self.model_ = self._build_model(P, C).to(device)
        self._train(train_patches, device)
        self._calibrate(train_patches, device)

        scores, valid = self._score_series(Xs, device)
        self.decision_scores_ = scores[valid]
        self._process_decision_scores()
        full = scores.copy()
        full[~valid] = self.threshold_
        self.decision_scores_ = full
        self.labels_ = (full > self.threshold_).astype(int)
        return self

    def decision_function(self, X):
        """Raw anomaly scores per timestamp (higher = more abnormal)."""
        check_is_fitted(self, ['decision_scores_', 'threshold_', 'labels_'])
        X = validate_ts_input(X)
        if X.shape[1] != self.n_channels_:
            raise ValueError("Expected %d channels, got %d"
                             % (self.n_channels_, X.shape[1]))
        if X.shape[0] < self.window_size:
            raise ValueError("Time series shorter than window_size")
        Xs = (X - self.X_mean_) / self.X_std_
        scores, valid = self._score_series(Xs, self._resolve_device())
        scores[~valid] = self.threshold_
        return scores

    # ------------------------------------------------------------------
    def _loss(self, model, xb, ctx, tgt, gen):
        """Loss of one batch for the four versions implemented here."""
        if self.variant == 'tjepa':   # one target set per window
            pred, h_tgt, h_all = model(xb, ctx, tgt.unsqueeze(1))
        else:
            pred, h_tgt, h_all = model(xb, ctx, tgt)
        if self.variant == 'tsjepa':
            loss_pred = F.l1_loss(pred, h_tgt.detach())
        else:
            loss_pred = F.mse_loss(pred, h_tgt.detach())
        loss = loss_pred
        if self.variant == 'lejepa':
            reg = sigreg_loss(h_all.transpose(0, 1),
                              num_slices=self.sigreg_slices,
                              generator=gen)
            loss = (1 - self.sigreg_weight) * loss_pred \
                + self.sigreg_weight * reg
        if self.recon_weight > 0:
            loss = loss + self.recon_weight * F.mse_loss(
                model.recon_head(h_all), xb)
        return loss, loss_pred

    def _train(self, patches, device):
        model = self.model_
        recipe = RECIPES.get(self.variant)
        hp = self.variant_params_
        gen = torch.Generator().manual_seed(self.random_state)

        def make_loader():
            return torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(patches),
                batch_size=self.batch_size, shuffle=True,
                drop_last=len(patches) >= self.batch_size, generator=gen)

        if recipe is None:
            n_epochs = self.epochs
            schedule = None if self.variant == 'lejepa' else 'linear'
            masked = True
        else:
            n_epochs = recipe.train_epochs(self, hp)
            schedule, masked = recipe.ema_schedule, recipe.masked
            recipe.pretrain(self, model, make_loader, device, gen, hp)
        ema_pairs = [] if schedule is None else (
            [(model.target_encoder, model.encoder)] if recipe is None
            else recipe.ema_pairs(model))

        loader = make_loader()
        opt = torch.optim.AdamW(model.parameters(), lr=self.lr,
                                weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(1, n_epochs), eta_min=self.lr * 0.05)
        total_steps = max(1, n_epochs * len(loader))
        step = 0
        model.train()
        for epoch in range(n_epochs):
            run = []
            for (xb,) in loader:
                xb = xb.to(device)
                ctx = tgt = None
                if masked:
                    ctx, tgt = sample_masks(xb.size(0), self.n_tokens_,
                                            self.n_mask_, self.mask_mode, gen)
                    ctx, tgt = ctx.to(device), tgt.to(device)
                opt.zero_grad()
                if recipe is None:
                    loss, loss_pred = self._loss(model, xb, ctx, tgt, gen)
                else:
                    state = dict(frac=min(1.0, step / total_steps),
                                 epoch=epoch, n_epochs=n_epochs)
                    loss, loss_pred = recipe.loss(self, model, xb, ctx, tgt,
                                                  gen, state, hp)
                loss.backward()
                if self.grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                   self.grad_clip)
                opt.step()
                if ema_pairs:
                    m = momentum_at(self.ema_momentum_,
                                    min(1.0, step / total_steps), schedule,
                                    1.0 if recipe is None else recipe.ema_end)
                    for target, online in ema_pairs:
                        ema_update(target, online, m)
                step += 1
                run.append(loss_pred.item())
            sched.step()
            if self.verbose:
                print('Epoch %d/%d, latent pred loss=%.4f'
                      % (epoch + 1, n_epochs, float(np.mean(run))))
        model.eval()

    def _eval_mask_set(self):
        if hasattr(self.model_, 'eval_mask_set'):   # versions without masks
            return self.model_.eval_mask_set()
        gen = torch.Generator().manual_seed(self.random_state + 1)
        return eval_masks(self.n_tokens_, self.n_mask_, self.mask_mode,
                          self.n_mask_samples, gen)

    @torch.no_grad()
    def _raw(self, patches, device, ctx, tgt):
        """Yield (err (B,K,m), h_all (B,P,D)) batch by batch, on CPU."""
        self.model_.eval()
        bs = max(8, self.batch_size)
        for s in range(0, len(patches), bs):
            xb = patches[s:s + bs].to(device)
            err, h_all = self.model_.multi_mask_errors(xb, ctx, tgt)
            yield err.cpu().numpy(), h_all.cpu().numpy()

    def _calibrate(self, patches, device):
        ctx, tgt = self._eval_mask_set()
        errs, embs = [], []
        for err, h_all in self._raw(patches, device, ctx, tgt):
            errs.append(err.reshape(-1))
            embs.append(h_all.reshape(-1, h_all.shape[-1]))
        errs = np.concatenate(errs)
        self.err_mean_ = float(errs.mean())
        self.err_std_ = float(max(errs.std(), 1e-8))
        Z = np.concatenate(embs, 0)
        self.emb_mean_ = Z.mean(0)
        cov = np.atleast_2d(np.cov(Z, rowvar=False))
        ridge = 1e-3 * max(np.trace(cov) / cov.shape[0], 1e-6)
        self.emb_prec_ = np.linalg.inv(cov + ridge * np.eye(cov.shape[0]))
        # component statistics for 'combined' (patch-level, on training)
        sp, se = [], []
        for err, h_all in self._raw(patches, device, ctx, tgt):
            sp.append(((err - self.err_mean_) / self.err_std_).reshape(-1))
            se.append(self._mahal(h_all).reshape(-1))
        sp, se = np.concatenate(sp), np.concatenate(se)
        self.comp_mean_ = np.array([sp.mean(), se.mean()])
        self.comp_std_ = np.maximum(np.array([sp.std(), se.std()]), 1e-8)

    def _mahal(self, h_all):
        d = h_all - self.emb_mean_
        return np.einsum('...i,ij,...j->...', d, self.emb_prec_, d)

    def _score_series(self, Xs, device):
        """Standardised series (n, C) -> (scores (n,), valid_mask (n,))."""
        n = Xs.shape[0]
        W = self._windows(Xs, self.window_size, self.step)
        patches = torch.from_numpy(self._to_patches(W))
        n_w, L = len(W), self.patch_len
        ctx, tgt = self._eval_mask_set()
        tgt_np = tgt.numpy()  # (K, m)
        use_max = self.score_aggregation == 'max'
        init = -np.inf if use_max else 0.0
        acc_p = np.full(n, init)
        cnt_p = np.zeros(n)
        acc_e = np.full(n, init)
        cnt_e = np.zeros(n)
        pos_in_patch = np.arange(L)
        offs_p = np.arange(self.n_tokens_) * L   # (P,)
        start = 0
        for err, h_all in self._raw(patches, device, ctx, tgt):
            B = err.shape[0]
            w0 = (np.arange(start, start + B) * self.step)  # (B,)
            start += B
            if self.score in ('pred', 'combined'):
                e = (err - self.err_mean_) / self.err_std_        # (B,K,m)
                base = w0[:, None, None] + tgt_np[None] * L        # (B,K,m)
                idx = (base[..., None] + pos_in_patch).reshape(-1)
                val = np.broadcast_to(e[..., None],
                                      e.shape + (L,)).reshape(-1)
                self._accumulate(acc_p, cnt_p, idx, val, use_max)
            if self.score in ('embed', 'combined'):
                me = self._mahal(h_all)                           # (B,P)
                base = w0[:, None] + offs_p[None]                  # (B,P)
                idx = (base[..., None] + pos_in_patch).reshape(-1)
                val = np.broadcast_to(me[..., None],
                                      me.shape + (L,)).reshape(-1)
                self._accumulate(acc_e, cnt_e, idx, val, use_max)

        def _finish(acc, cnt):
            out = np.full(n, np.nan)
            ok = cnt > 0
            out[ok] = acc[ok] if use_max else acc[ok] / cnt[ok]
            return out, ok

        if self.score == 'pred':
            return _finish(acc_p, cnt_p)
        if self.score == 'embed':
            return _finish(acc_e, cnt_e)
        sp, okp = _finish(acc_p, cnt_p)
        se, oke = _finish(acc_e, cnt_e)
        zp = (sp - self.comp_mean_[0]) / self.comp_std_[0]
        ze = (se - self.comp_mean_[1]) / self.comp_std_[1]
        out = np.where(okp & oke, 0.5 * (zp + ze),
                       np.where(okp, zp, ze))
        return out, okp | oke

    @staticmethod
    def _accumulate(acc, cnt, idx, val, use_max):
        if use_max:
            np.maximum.at(acc, idx, val)
        else:
            np.add.at(acc, idx, val)
        np.add.at(cnt, idx, 1.0)
