# -*- coding: utf-8 -*-
"""JEPA-OD: outlier detection with a Joint-Embedding Predictive
Architecture (tabular data, PyOD ``BaseDeepLearningDetector`` API).

A JEPA learns to *predict the latent representation* of hidden parts of
an input from the visible parts (LeCun, 2022; I-JEPA, Assran et al. 2023).
Normal rows have feature dependencies the predictor can model; a row whose
masked features cannot be predicted from the others *in representation
space* is an outlier. Compared with reconstruction (AutoEncoder) the error
is measured in a learned latent space, so unpredictable noise in the raw
features is not penalised and the score focuses on broken dependencies.

Scoring components (``score``):

* ``'pred'``     -- calibrated latent prediction error averaged over
  several masks;
* ``'embed'``    -- mean over tokens of the Mahalanobis distance of each
  (contextualised) token embedding to its training Gaussian;
* ``'combined'`` -- z-normalised sum of both (default; best mean ROC-AUC
  and AP of the three on 16 ADBench datasets).

Collapse prevention (``target_mode``): ``'sigreg'`` (LeJEPA's Sketched
Isotropic Gaussian Regularization, no teacher network) or ``'ema'``
(I-JEPA style EMA target encoder).
"""
# License: BSD 2 clause

import math

import numpy as np
import torch
import torch.nn.functional as F

from sklearn.utils import check_array

from pyod.models.base_dl import BaseDeepLearningDetector
from pyod.utils.torch_utility import TorchDataset

from ._nets import JEPACore, sample_masks, eval_masks, _MASK_MODES
from ._sigreg import sigreg_loss, ema_update

_SCORES = ('pred', 'embed', 'combined')


class JEPAOD(BaseDeepLearningDetector):
    """Joint-Embedding Predictive Architecture Outlier Detector.

    Each feature (or group of ``ceil(n_features / max_tokens)``
    consecutive features) is a token. During training a random subset of
    tokens is masked; the encoder embeds the visible tokens, a light
    predictor forecasts the embeddings of the masked tokens, and the loss
    is the squared error against the (stop-gradient) target embeddings
    produced by the same encoder on the full row. Embeddings are kept
    non-degenerate with SIGReg (or an EMA teacher).

    Parameters
    ----------
    contamination : float in (0., 0.5), optional (default=0.1)
        The proportion of outliers in the data set, used to set the
        threshold on the decision function.

    preprocessing : bool, optional (default=True)
        Standardise features with training mean / std.

    lr : float, optional (default=1e-3)

    epoch_num : int, optional (default=30)

    batch_size : int, optional (default=64)

    optimizer_name : str, optional (default='adam')

    criterion_name : str, optional (default='mse')
        Latent prediction loss, 'mse' or 'mae' (smooth alternatives can be
        passed through ``criterion``).

    device : str, optional (default=None)
        ``None`` picks CUDA if available else CPU; pass ``'mps'`` for
        Apple silicon.

    random_state : int, optional (default=42)

    use_compile, compile_mode, verbose, optimizer_params, criterion_params
        See :class:`pyod.models.base_dl.BaseDeepLearningDetector`.

    d_model : int, optional (default=64)
        Token embedding width.

    n_heads : int, optional (default=4)

    n_layers : int, optional (default=2)
        Encoder depth.

    pred_layers : int, optional (default=1)
        Predictor depth (width ``d_model // 2``).

    dropout : float, optional (default=0.0)

    max_tokens : int, optional (default=64)
        Features are grouped into at most this many tokens.

    mask_ratio : float in (0, 1), optional (default=0.3)
        Fraction of tokens hidden from the context encoder.

    mask_mode : {'random', 'block', 'forecast', 'mixed'}, default 'random'
        Masking pattern; 'random' is the natural choice for tabular data.

    n_mask_samples : int, optional (default=8)
        Number of masks averaged at inference (deterministic, seeded).

    target_mode : {'sigreg', 'ema'}, optional (default='sigreg')

    sigreg_weight : float, optional (default=0.05)
        ``lambda`` of LeJEPA: loss = (1-lambda) * pred + lambda * SIGReg.

    sigreg_slices : int, optional (default=128)
        Random projections per SIGReg evaluation.

    ema_momentum : float, optional (default=0.996)
        Initial EMA momentum (annealed to 1.0) for ``target_mode='ema'``.

    recon_weight : float, optional (default=0.0)
        Weight of an auxiliary linear reconstruction of each raw token from
        its target embedding (anchors embeddings to token content).

    score : {'pred', 'embed', 'combined'}, optional (default='combined')
        Use ``'pred'`` alone when test data may be shifted relative to the
        training distribution (the embedding density term is shift-sensitive).

    error_norm : bool, optional (default=True)
        Z-normalise the per-token latent errors with training statistics
        so that intrinsically unpredictable features do not dominate.

    lr_schedule : bool, optional (default=True)
        Cosine learning-rate decay over ``epoch_num``.

    grad_clip : float or None, optional (default=1.0)

    encoder_norm : bool, optional (default=False)
        Final LayerNorm on the encoder (ablation switch; True reproduces
        the saturation failure described in the module docstring).

    Attributes
    ----------
    model : torch.nn.Module
        The underlying :class:`JEPACore`.

    decision_scores_ : numpy array of shape (n_samples,)
        Outlier scores of the training data (higher = more abnormal).

    threshold_ : float

    labels_ : numpy array of shape (n_samples,)

    err_mean_, err_std_ : numpy arrays of shape (n_tokens,)
        Training latent-error statistics per token.

    emb_mean_ : numpy array of shape (n_tokens, d_model)
        Per-token mean of training embeddings.

    emb_prec_ : numpy array of shape (d_model, d_model)
        Inverse of the shared (pooled over tokens) covariance.
    """

    def __init__(self,
                 contamination=0.1, preprocessing=True,
                 lr=1e-3, epoch_num=30, batch_size=64,
                 optimizer_name='adam', criterion_name='mse',
                 device=None, random_state=42,
                 use_compile=False, compile_mode='default',
                 verbose=1,
                 optimizer_params: dict = {'weight_decay': 1e-4},
                 criterion_params: dict = {},
                 d_model=64, n_heads=4, n_layers=2, pred_layers=1,
                 dropout=0.0, max_tokens=64,
                 mask_ratio=0.3, mask_mode='random', n_mask_samples=8,
                 target_mode='sigreg', sigreg_weight=0.05,
                 sigreg_slices=128, ema_momentum=0.996,
                 recon_weight=0.0,
                 score='combined', error_norm=True,
                 lr_schedule=True, grad_clip=1.0, encoder_norm=False):
        super(JEPAOD, self).__init__(contamination=contamination,
                                     preprocessing=preprocessing,
                                     lr=lr, epoch_num=epoch_num,
                                     batch_size=batch_size,
                                     optimizer_name=optimizer_name,
                                     criterion_name=criterion_name,
                                     device=device,
                                     random_state=random_state,
                                     use_compile=use_compile,
                                     compile_mode=compile_mode,
                                     verbose=verbose,
                                     optimizer_params=optimizer_params,
                                     criterion_params=criterion_params)
        self.criterion_name = criterion_name
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.pred_layers = pred_layers
        self.dropout = dropout
        self.max_tokens = max_tokens
        self.mask_ratio = mask_ratio
        self.mask_mode = mask_mode
        self.n_mask_samples = n_mask_samples
        self.target_mode = target_mode
        self.sigreg_weight = sigreg_weight
        self.sigreg_slices = sigreg_slices
        self.ema_momentum = ema_momentum
        self.recon_weight = recon_weight
        self.score = score
        self.error_norm = error_norm
        self.lr_schedule = lr_schedule
        self.grad_clip = grad_clip
        self.encoder_norm = encoder_norm

    # ------------------------------------------------------------------
    # PyOD BaseDeepLearningDetector hooks
    # ------------------------------------------------------------------
    def build_model(self):
        if self.feature_size < 2:
            raise ValueError("JEPAOD needs at least 2 features to form a "
                             "context / target split.")
        if self.score not in _SCORES:
            raise ValueError("score must be one of %s" % (_SCORES,))
        if self.mask_mode not in _MASK_MODES:
            raise ValueError("mask_mode must be one of %s" % (_MASK_MODES,))
        if not 0.0 < self.mask_ratio < 1.0:
            raise ValueError("mask_ratio must be in (0, 1)")
        d = self.feature_size
        n_tokens = min(d, self.max_tokens)
        token_dim = int(math.ceil(d / n_tokens))
        n_tokens = int(math.ceil(d / token_dim))
        self.n_tokens_ = n_tokens
        self.token_dim_ = token_dim
        self.pad_ = n_tokens * token_dim - d
        self.n_mask_ = int(np.clip(round(self.mask_ratio * n_tokens),
                                   1, n_tokens - 1))
        self.model = JEPACore(
            n_tokens=n_tokens, in_dim=token_dim, d_model=self.d_model,
            n_heads=self.n_heads, n_layers=self.n_layers,
            pred_layers=self.pred_layers, dropout=self.dropout,
            target_mode=self.target_mode,
            recon_dim=token_dim if self.recon_weight > 0 else None,
            encoder_norm=self.encoder_norm)

    def training_prepare(self):
        super(JEPAOD, self).training_prepare()
        self._train_gen = torch.Generator().manual_seed(self.random_state)
        self.scheduler = None
        if self.lr_schedule and self.epoch_num > 1:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.epoch_num, eta_min=self.lr * 0.05)
        steps_per_epoch = max(1, self.data_num // self.batch_size)
        self._total_steps = max(1, self.epoch_num * steps_per_epoch)
        self._step = 0

    def _to_tokens(self, x):
        """(B, n_features) -> (B, n_tokens, token_dim) with zero padding."""
        if self.pad_ > 0:
            x = F.pad(x, (0, self.pad_))
        return x.view(x.size(0), self.n_tokens_, self.token_dim_)

    def training_forward(self, batch_data):
        x = batch_data.to(self.device)
        xt = self._to_tokens(x)
        B = xt.size(0)
        ctx, tgt = sample_masks(B, self.n_tokens_, self.n_mask_,
                                self.mask_mode, self._train_gen)
        ctx, tgt = ctx.to(self.device), tgt.to(self.device)

        self.optimizer.zero_grad()
        pred, h_tgt, h_all = self.model(xt, ctx, tgt)
        # stop-gradient on targets: the encoder must not make its own
        # targets easier to predict (the "too invariant" failure mode)
        loss_pred = self.criterion(pred, h_tgt.detach())
        loss = loss_pred
        parts = [loss_pred.item()]
        if self.target_mode == 'sigreg':
            reg = sigreg_loss(h_all.transpose(0, 1),
                              num_slices=self.sigreg_slices,
                              generator=self._train_gen)
            loss = (1.0 - self.sigreg_weight) * loss_pred \
                + self.sigreg_weight * reg
            parts.append(reg.item())
        if self.recon_weight > 0:
            loss_rec = F.mse_loss(self.model.recon_head(h_all), xt)
            loss = loss + self.recon_weight * loss_rec
            parts.append(loss_rec.item())
        loss.backward()
        if self.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                           self.grad_clip)
        self.optimizer.step()

        if self.target_mode == 'ema':
            frac = min(1.0, self._step / self._total_steps)
            m = self.ema_momentum + (1.0 - self.ema_momentum) * frac
            ema_update(self.model.target_encoder, self.model.encoder, m)
        self._step += 1
        return tuple(parts) if len(parts) > 1 else parts[0]

    def epoch_update(self):
        if self.scheduler is not None:
            self.scheduler.step()

    def train(self, train_loader):
        super(JEPAOD, self).train(train_loader)
        self._calibrate(train_loader)

    def evaluating_prepare(self):
        self.model.eval()
        gen = torch.Generator().manual_seed(self.random_state + 1)
        self._eval_ctx, self._eval_tgt = eval_masks(
            self.n_tokens_, self.n_mask_, self.mask_mode,
            self.n_mask_samples, gen)

    def evaluating_forward(self, batch_data):
        x = batch_data.to(self.device)
        s_pred, s_emb = self._component_scores(x)
        return self._combine(s_pred, s_emb)

    @torch.no_grad()
    def token_embeddings(self, X, batch_size=None):
        """Embeddings of the fitted encoder, for use by another detector.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)

        batch_size : int, optional (default=None)

        Returns
        -------
        embeddings : numpy array of shape (n_samples, n_tokens, d_model)
            The target embeddings of every token of the full row, i.e. the
            quantities the ``'embed'`` score is computed from.
        """
        X = check_array(X)
        if self.preprocessing:
            dataset = TorchDataset(X=X, y=None, mean=self.X_mean,
                                   std=self.X_std)
        else:
            dataset = TorchDataset(X=X, y=None)
        loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size if batch_size is None else batch_size,
            shuffle=False, drop_last=False)
        self.model.eval()
        out = [self.model.token_embeddings(
            self._to_tokens(batch.to(self.device))).cpu().numpy()
            for batch in loader]
        return np.concatenate(out)

    # ------------------------------------------------------------------
    # scoring internals
    # ------------------------------------------------------------------
    def _raw(self, x):
        """Per-token latent errors (B, K, m), masked ids (K, m), token
        embeddings (B, T, D) -- all on CPU."""
        xt = self._to_tokens(x)
        err, h_all = self.model.multi_mask_errors(xt, self._eval_ctx,
                                                  self._eval_tgt)
        return err.cpu(), self._eval_tgt, h_all.cpu()

    def _component_scores(self, x):
        err, tgt, h_all = self._raw(x)
        if self.error_norm:
            mu = torch.as_tensor(self.err_mean_, dtype=err.dtype)[tgt]
            sd = torch.as_tensor(self.err_std_, dtype=err.dtype)[tgt]
            err = (err - mu) / sd
        s_pred = err.mean(dim=(1, 2)).numpy()
        diff = h_all.numpy() - self.emb_mean_                 # (B, T, D)
        s_emb = np.einsum('bti,ij,btj->bt', diff, self.emb_prec_,
                          diff).mean(1)
        return s_pred, s_emb

    def _combine(self, s_pred, s_emb):
        if self.score == 'pred':
            return s_pred
        if self.score == 'embed':
            return s_emb
        zp = (s_pred - self.comp_mean_[0]) / self.comp_std_[0]
        ze = (s_emb - self.comp_mean_[1]) / self.comp_std_[1]
        return 0.5 * (zp + ze)

    @torch.no_grad()
    def _calibrate(self, train_loader):
        """Fit per-token error statistics, the embedding Gaussian and the
        component-score statistics on the training data."""
        self.evaluating_prepare()
        T = self.n_tokens_
        s1 = torch.zeros(T, dtype=torch.float64)
        s2 = torch.zeros(T, dtype=torch.float64)
        cnt = torch.zeros(T, dtype=torch.float64)
        D = self.d_model
        emb_sum = torch.zeros(T, D, dtype=torch.float64)
        emb_sq = torch.zeros(D, D, dtype=torch.float64)
        n_rows = 0
        for batch in train_loader:
            err, tgt, h_all = self._raw(batch.to(self.device))
            h = h_all.double()
            emb_sum += h.sum(0)
            emb_sq += torch.einsum('bti,btj->ij', h, h)
            n_rows += h.size(0)
            e = err.double().reshape(err.size(0), -1)      # (B, K*m)
            idx = tgt.reshape(-1).expand(e.size(0), -1)     # (B, K*m)
            s1.index_add_(0, idx.reshape(-1), e.reshape(-1))
            s2.index_add_(0, idx.reshape(-1), e.reshape(-1) ** 2)
            cnt.index_add_(0, idx.reshape(-1),
                           torch.ones_like(e.reshape(-1)))
        seen = cnt > 0
        g_mean = (s1.sum() / cnt.sum()).item()
        g_var = max((s2.sum() / cnt.sum()).item() - g_mean ** 2, 1e-12)
        mean = torch.full((T,), g_mean, dtype=torch.float64)
        var = torch.full((T,), g_var, dtype=torch.float64)
        mean[seen] = s1[seen] / cnt[seen]
        var[seen] = (s2[seen] / cnt[seen] - mean[seen] ** 2).clamp_min(0)
        std = var.sqrt()
        std = torch.where(std > 1e-8, std, torch.full_like(std, g_var ** .5))
        self.err_mean_ = mean.numpy()
        self.err_std_ = np.maximum(std.numpy(), 1e-8)

        mu_tok = emb_sum / max(n_rows, 1)                      # (T, D)
        n_tok = n_rows * T
        # shared covariance of token embeddings around their token means
        cov = (emb_sq - n_rows * mu_tok.t() @ mu_tok) / max(n_tok - 1, 1)
        cov = cov.numpy()
        ridge = 1e-3 * max(np.trace(cov) / D, 1e-6)
        self.emb_mean_ = mu_tok.numpy()
        self.emb_prec_ = np.linalg.inv(cov + ridge * np.eye(D))

        sp, se = [], []
        for batch in train_loader:
            a, b = self._component_scores(batch.to(self.device))
            sp.append(a)
            se.append(b)
        sp, se = np.concatenate(sp), np.concatenate(se)
        self.comp_mean_ = np.array([sp.mean(), se.mean()])
        self.comp_std_ = np.maximum(np.array([sp.std(), se.std()]), 1e-8)
