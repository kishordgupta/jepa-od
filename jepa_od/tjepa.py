# -*- coding: utf-8 -*-
"""T-JEPA as an outlier detector (tabular data, PyOD
``BaseDeepLearningDetector`` API).

T-JEPA (Thimonier et al., ICLR 2025) pre-trains a tabular encoder by
predicting the latent representation of one subset of features from
another subset, with an EMA target encoder and a regularisation token.
The paper evaluates the representation on classification and regression
and leaves outlier detection as future work. :class:`TJEPAOD` trains the
T-JEPA recipe and scores rows with the JEPA-OD scores of
:class:`jepa_od.JEPAOD`, so the two members of the family differ only in
how the network is built and trained:

=====================  ==========================  ======================
                       ``JEPAOD``                  ``TJEPAOD``
=====================  ==========================  ======================
collapse prevention    SIGReg (LeJEPA), one        EMA target encoder and
                       shared encoder              a [REG] token
context                most features (70 %)        few features (20-40 %)
targets per row        one mask                    four masks
encoder output         not normalised              LayerNorm'ed
=====================  ==========================  ======================

This is a re-implementation written from the paper and the defaults of
the released code, not the authors' code, and the outlier scores are not
part of T-JEPA.
"""
# License: BSD 2 clause

import math

import torch

from ._sigreg import ema_update
from ._tjepa_nets import (TJEPACore, tjepa_size_ranges, sample_tjepa_sizes,
                          sample_tjepa_masks, eval_tjepa_masks)
from .jepa import JEPAOD, _SCORES


class TJEPAOD(JEPAOD):
    """T-JEPA Outlier Detector.

    Each feature (or group of ``ceil(n_features / max_tokens)``
    consecutive features) is a token. For every row a small random subset
    of tokens is the context and ``n_pred_masks`` random subsets of the
    other tokens are the targets. The context encoder embeds the context
    together with the [REG] tokens, a predictor forecasts the embeddings
    of the target tokens, and the loss is the squared error against the
    embeddings an EMA copy of the encoder produces for the full row.

    The architecture and masking defaults are the defaults of the released
    T-JEPA code. The training budget (``lr``, ``epoch_num``,
    ``batch_size``) is the one of :class:`JEPAOD`, because T-JEPA's own
    budget (batch size 512, 100 or more epochs, tuned per data set with
    labels) is meant for much larger tables and is not available without
    labels.

    Parameters
    ----------
    contamination : float in (0., 0.5), optional (default=0.1)
        The proportion of outliers in the data set, used to set the
        threshold on the decision function.

    preprocessing : bool, optional (default=True)
        Standardise features with training mean / std.

    lr : float, optional (default=1e-3)
        Peak learning rate.

    epoch_num : int, optional (default=30)

    batch_size : int, optional (default=64)

    optimizer_name : str, optional (default='adam')
        With the default ``weight_decay`` of 0 this equals the AdamW of
        the released T-JEPA launch script.

    criterion_name : str, optional (default='mse')

    device, random_state, use_compile, compile_mode, verbose,
    optimizer_params, criterion_params
        See :class:`pyod.models.base_dl.BaseDeepLearningDetector`.

    d_model : int, optional (default=64)
        Token embedding width.

    n_heads : int, optional (default=8)

    n_layers : int, optional (default=4)
        Encoder depth.

    dim_feedforward : int, optional (default=256)

    dropout : float, optional (default=0.1)

    pred_dim : int, optional (default=64)
        Predictor width.

    pred_heads : int, optional (default=4)

    pred_layers : int, optional (default=2)

    pred_feedforward : int, optional (default=256)

    max_tokens : int, optional (default=64)
        Features are grouped into at most this many tokens.

    min_ctx_share, max_ctx_share : float, optional (default=0.2, 0.4)
        Range of the share of tokens visible to the context encoder. One
        share is sampled per batch.

    min_tgt_share, max_tgt_share : float, optional (default=0.2, 0.4)
        Range of the share of tokens in each target mask.

    n_pred_masks : int, optional (default=4)
        Target masks per row during training.

    n_reg_tokens : int, optional (default=1)
        Number of [REG] tokens. The T-JEPA paper reports that training
        collapses with 0.

    ema_momentum : float, optional (default=0.996)
        Initial EMA momentum of the target encoder, increased linearly to
        1.0 over training.

    warmup_epochs : int, optional (default=1)
        Linear learning-rate warm-up; the released launch script warms up
        for 10 of 300 epochs.

    n_mask_samples : int, optional (default=8)
        Number of masks averaged at inference (deterministic, seeded).

    score : {'pred', 'embed', 'combined'}, optional (default='combined')
        See :class:`JEPAOD`.

    error_norm : bool, optional (default=True)
        Z-normalise the per-token latent errors with training statistics.

    lr_schedule : bool, optional (default=True)
        Cosine learning-rate decay to zero after the warm-up.

    grad_clip : float or None, optional (default=1.0)

    encoder_norm : bool, optional (default=True)
        True is T-JEPA as published: post-norm blocks and a
        ``LayerNorm -> Linear -> LayerNorm`` output head. False uses
        pre-norm blocks and no output normalisation, which is the JEPA-OD
        design applied to T-JEPA (ablation switch).

    Attributes
    ----------
    model : torch.nn.Module
        The underlying :class:`jepa_od._tjepa_nets.TJEPACore`.

    decision_scores_ : numpy array of shape (n_samples,)
        Outlier scores of the training data (higher = more abnormal).

    threshold_ : float

    labels_ : numpy array of shape (n_samples,)

    n_ctx_, n_tgt_ : int
        Context and target sizes (in tokens) of the inference masks, the
        midpoints of the training ranges ``ctx_range_`` / ``tgt_range_``.

    err_mean_, err_std_, emb_mean_, emb_prec_
        See :class:`JEPAOD`.
    """

    def __init__(self,
                 contamination=0.1, preprocessing=True,
                 lr=1e-3, epoch_num=30, batch_size=64,
                 optimizer_name='adam', criterion_name='mse',
                 device=None, random_state=42,
                 use_compile=False, compile_mode='default',
                 verbose=1,
                 optimizer_params: dict = {'weight_decay': 0.0},
                 criterion_params: dict = {},
                 d_model=64, n_heads=8, n_layers=4, dim_feedforward=256,
                 dropout=0.1, pred_dim=64, pred_heads=4, pred_layers=2,
                 pred_feedforward=256, max_tokens=64,
                 min_ctx_share=0.2, max_ctx_share=0.4,
                 min_tgt_share=0.2, max_tgt_share=0.4,
                 n_pred_masks=4, n_reg_tokens=1, ema_momentum=0.996,
                 warmup_epochs=1, n_mask_samples=8,
                 score='combined', error_norm=True,
                 lr_schedule=True, grad_clip=1.0, encoder_norm=True):
        super(TJEPAOD, self).__init__(
            contamination=contamination, preprocessing=preprocessing,
            lr=lr, epoch_num=epoch_num, batch_size=batch_size,
            optimizer_name=optimizer_name, criterion_name=criterion_name,
            device=device, random_state=random_state,
            use_compile=use_compile, compile_mode=compile_mode,
            verbose=verbose, optimizer_params=optimizer_params,
            criterion_params=criterion_params,
            d_model=d_model, n_heads=n_heads, n_layers=n_layers,
            pred_layers=pred_layers, dropout=dropout, max_tokens=max_tokens,
            n_mask_samples=n_mask_samples, target_mode='ema',
            ema_momentum=ema_momentum, score=score, error_norm=error_norm,
            lr_schedule=lr_schedule, grad_clip=grad_clip,
            encoder_norm=encoder_norm)
        self.dim_feedforward = dim_feedforward
        self.pred_dim = pred_dim
        self.pred_heads = pred_heads
        self.pred_feedforward = pred_feedforward
        self.min_ctx_share = min_ctx_share
        self.max_ctx_share = max_ctx_share
        self.min_tgt_share = min_tgt_share
        self.max_tgt_share = max_tgt_share
        self.n_pred_masks = n_pred_masks
        self.n_reg_tokens = n_reg_tokens
        self.warmup_epochs = warmup_epochs

    # ------------------------------------------------------------------
    # PyOD BaseDeepLearningDetector hooks
    # ------------------------------------------------------------------
    def build_model(self):
        if self.feature_size < 2:
            raise ValueError("TJEPAOD needs at least 2 features to form a "
                             "context / target split.")
        if self.score not in _SCORES:
            raise ValueError("score must be one of %s" % (_SCORES,))
        shares = (self.min_ctx_share, self.max_ctx_share,
                  self.min_tgt_share, self.max_tgt_share)
        if not all(0.0 < s < 1.0 for s in shares):
            raise ValueError("context and target shares must be in (0, 1)")
        if self.min_ctx_share > self.max_ctx_share \
                or self.min_tgt_share > self.max_tgt_share:
            raise ValueError("a minimum share exceeds its maximum share")
        if self.n_pred_masks < 1 or self.n_reg_tokens < 0:
            raise ValueError("n_pred_masks must be >= 1 and n_reg_tokens "
                             ">= 0")
        d = self.feature_size
        n_tokens = min(d, self.max_tokens)
        token_dim = int(math.ceil(d / n_tokens))
        n_tokens = int(math.ceil(d / token_dim))
        self.n_tokens_ = n_tokens
        self.token_dim_ = token_dim
        self.pad_ = n_tokens * token_dim - d
        self.ctx_range_, self.tgt_range_ = tjepa_size_ranges(n_tokens, *shares)
        # inference masks use the midpoints of the training ranges
        self.n_ctx_ = int(round(sum(self.ctx_range_) / 2.0))
        self.n_tgt_ = min(int(round(sum(self.tgt_range_) / 2.0)),
                          n_tokens - self.n_ctx_)
        self.model = TJEPACore(
            n_tokens=n_tokens, in_dim=token_dim, d_model=self.d_model,
            n_heads=self.n_heads, n_layers=self.n_layers,
            dim_feedforward=self.dim_feedforward, dropout=self.dropout,
            pred_dim=self.pred_dim, pred_heads=self.pred_heads,
            pred_layers=self.pred_layers,
            pred_feedforward=self.pred_feedforward,
            n_reg_tokens=self.n_reg_tokens, encoder_norm=self.encoder_norm)

    def training_prepare(self):
        super(TJEPAOD, self).training_prepare()
        # per-step warm-up + cosine schedule replaces the per-epoch one
        self.scheduler = None
        self._steps_per_epoch = max(1, self.data_num // self.batch_size)

    def _lr_at(self, step):
        warm = self.warmup_epochs * self._steps_per_epoch
        if step < warm:
            return self.lr * (step + 1) / warm
        if not self.lr_schedule:
            return self.lr
        frac = (step - warm) / max(1, self._total_steps - warm)
        return self.lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, frac)))

    def training_forward(self, batch_data):
        x = batch_data.to(self.device)
        xt = self._to_tokens(x)
        n_ctx, n_tgt = sample_tjepa_sizes(self.n_tokens_, self.ctx_range_,
                                          self.tgt_range_, self._train_gen)
        ctx, tgt = sample_tjepa_masks(xt.size(0), self.n_tokens_, n_ctx,
                                      n_tgt, self.n_pred_masks,
                                      self._train_gen)
        ctx, tgt = ctx.to(self.device), tgt.to(self.device)

        for group in self.optimizer.param_groups:
            group['lr'] = self._lr_at(self._step)
        self.optimizer.zero_grad()
        pred, h_tgt, _ = self.model(xt, ctx, tgt)
        loss = self.criterion(pred, h_tgt)
        loss.backward()
        if self.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                           self.grad_clip)
        self.optimizer.step()

        frac = min(1.0, self._step / self._total_steps)
        m = self.ema_momentum + (1.0 - self.ema_momentum) * frac
        ema_update(self.model.target_encoder, self.model.encoder, m)
        self._step += 1
        return loss.item()

    def evaluating_prepare(self):
        self.model.eval()
        gen = torch.Generator().manual_seed(self.random_state + 1)
        self._eval_ctx, self._eval_tgt = eval_tjepa_masks(
            self.n_tokens_, self.n_ctx_, self.n_tgt_, self.n_mask_samples,
            gen)
