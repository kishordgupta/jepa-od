# -*- coding: utf-8 -*-
"""Networks of the further JEPA versions of :class:`jepa_od.TSJEPAOD`.

Each class is the network of one or two versions and offers the scoring
interface of :class:`jepa_od._nets.JEPACore`: ``token_embeddings(x)`` and
``multi_mask_errors(x, ctx_idx, tgt_idx)``. Networks that do not mask
patches also define ``eval_mask_set()``, the single (context, target)
split they are scored with. All of them are written from the papers named
in the docstrings; sizes follow the detector, not the papers.
"""
# License: BSD 2 clause

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._nets import JEPACore, Tokenizer, TransformerStack, _gather_tokens


def _frozen_copy(module):
    target = copy.deepcopy(module)
    for p in target.parameters():
        p.requires_grad_(False)
    return target


def _repeat_masks(x, ctx_idx, tgt_idx):
    """Rows of ``x`` repeated once per mask of a shared mask set."""
    B, K = x.size(0), tgt_idx.size(0)
    ctx = ctx_idx.to(x.device).repeat(B, 1)   # row b -> masks 0..K-1
    tgt = tgt_idx.to(x.device).repeat(B, 1)
    return x.repeat_interleave(K, dim=0), ctx, tgt


# ----------------------------------------------------------------------
# C-JEPA: I-JEPA plus VICReg on the context embeddings
# ----------------------------------------------------------------------
class CJEPACore(JEPACore):
    """I-JEPA network with the projector on which C-JEPA (Mo and Tong,
    arXiv:2410.19560) computes its VICReg terms."""

    def __init__(self, n_tokens, in_dim, **kwargs):
        super().__init__(n_tokens, in_dim, target_mode='ema', **kwargs)
        d = self.d_model
        self.projector = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(),
                                       nn.Linear(2 * d, d))

    def forward_views(self, x, ctx_idx, tgt_idx):
        """:meth:`forward` plus the projected mean context embedding (B, D)."""
        tokens = self.embed(x)
        h_ctx = self.encoder(_gather_tokens(tokens, ctx_idx))
        h_all = self.encode_target(tokens)
        pred = self.predict(h_ctx, ctx_idx, tgt_idx)
        return (pred, _gather_tokens(h_all, tgt_idx), h_all,
                self.projector(h_ctx.mean(1)))


# ----------------------------------------------------------------------
# stochastic-latent JEPA
# ----------------------------------------------------------------------
class StochasticJEPACore(JEPACore):
    """I-JEPA network whose predictor also receives a Gaussian latent.

    After "A Stochastic Latent Variable Generalization of Joint-Embedding
    Predictive Architectures" (Giakoumoglou et al., arXiv:2611.09302): a
    prior ``p(z | context)`` and a posterior ``q(z | context, target)``,
    the latent enters the predictor as a prefix token, and the loss adds
    ``KL(q || p)``. Scoring uses the prior mean, i.e. only what the context
    says.
    """

    def __init__(self, n_tokens, in_dim, latent_dim=None, **kwargs):
        super().__init__(n_tokens, in_dim, target_mode='ema', **kwargs)
        d = self.d_model
        self.latent_dim = int(latent_dim or max(8, d // 4))
        self.prior_net = nn.Sequential(
            nn.Linear(d, d), nn.GELU(), nn.Linear(d, 2 * self.latent_dim))
        self.post_net = nn.Sequential(
            nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 2 * self.latent_dim))
        self.latent_in = nn.Linear(self.latent_dim, self.pred_in.out_features)

    @staticmethod
    def _gaussian(out):
        mu, logvar = out.chunk(2, dim=-1)
        return mu, logvar.clamp(-8.0, 8.0)

    def prior(self, h_ctx):
        return self._gaussian(self.prior_net(h_ctx.mean(1)))

    def posterior(self, h_ctx, h_tgt):
        return self._gaussian(self.post_net(
            torch.cat([h_ctx.mean(1), h_tgt.mean(1)], dim=-1)))

    def predict(self, h_ctx, ctx_idx, tgt_idx, z=None):
        if z is None:
            z = self.prior(h_ctx)[0]
        c = self.pred_in(h_ctx) + self.pred_pos[ctx_idx]
        q = self.mask_token + self.pred_pos[tgt_idx]
        out = self.predictor(
            torch.cat([self.latent_in(z).unsqueeze(1), c, q], dim=1))
        return self.pred_out(out[:, 1 + c.size(1):])

    def forward_latent(self, x, ctx_idx, tgt_idx, noise):
        """Training pass with a posterior sample.

        noise : (B, latent_dim) standard normal draws.

        Returns ``pred, h_tgt, h_all, kl`` with ``kl`` the mean over the
        batch of ``KL(q || p)`` summed over the latent dimensions.
        """
        tokens = self.embed(x)
        h_ctx = self.encoder(_gather_tokens(tokens, ctx_idx))
        h_all = self.encode_target(tokens)
        h_tgt = _gather_tokens(h_all, tgt_idx)
        mu_p, lv_p = self.prior(h_ctx)
        mu_q, lv_q = self.posterior(h_ctx, h_tgt)
        z = mu_q + noise * torch.exp(0.5 * lv_q)
        pred = self.predict(h_ctx, ctx_idx, tgt_idx, z)
        kl = 0.5 * (lv_p - lv_q + (lv_q.exp() + (mu_q - mu_p).pow(2))
                    / lv_p.exp() - 1.0).sum(-1).mean()
        return pred, h_tgt, h_all, kl


# ----------------------------------------------------------------------
# V-JEPA 2.1: dense loss and deep self-supervision
# ----------------------------------------------------------------------
class DenseJEPACore(nn.Module):
    """Network of the ``'vjepa21'`` version.

    After V-JEPA 2.1 (arXiv:2603.14482): the outputs of several encoder
    depths are fused before the predictor, the predictor returns one
    prediction per depth for *every* token (visible ones included), and
    the targets are the EMA encoder's outputs at the same depths.

    Parameters
    ----------
    max_levels : int, optional (default=4)
        Depths used, equally spaced and ending at the last layer; never
        more than ``n_layers``.
    """

    def __init__(self, n_tokens, in_dim, d_model=64, n_heads=4, n_layers=2,
                 pred_layers=1, dropout=0.0, encoder_norm=False,
                 max_levels=4):
        super().__init__()
        self.n_tokens = n_tokens
        self.d_model = d_model
        self.n_layers = n_layers
        n_levels = max(1, min(int(max_levels), n_layers))
        self.levels = sorted({int(round(n_layers * (i + 1) / n_levels))
                              for i in range(n_levels)})
        pred_dim = max(16, d_model // 2)
        pred_heads = max(1, min(n_heads, pred_dim // 16))

        self.tokenizer = Tokenizer(n_tokens, in_dim, d_model)
        self.pos = nn.Parameter(torch.randn(n_tokens, d_model) * 0.02)
        self.encoder = TransformerStack(d_model, n_heads, n_layers, dropout,
                                        final_norm=encoder_norm)
        self.target_encoder = _frozen_copy(self.encoder)
        self.fuse = nn.Sequential(
            nn.Linear(len(self.levels) * d_model, d_model), nn.GELU(),
            nn.Linear(d_model, pred_dim))
        self.mask_token = nn.Parameter(torch.randn(pred_dim) * 0.02)
        self.pred_pos = nn.Parameter(torch.randn(n_tokens, pred_dim) * 0.02)
        self.predictor = TransformerStack(pred_dim, pred_heads, pred_layers,
                                          dropout)
        self.pred_out = nn.ModuleList(
            [nn.Linear(pred_dim, d_model) for _ in self.levels])

    def embed(self, x):
        return self.tokenizer(x) + self.pos

    def _level_outputs(self, stack, tokens):
        """Layer-normalised outputs of ``stack`` after the chosen depths."""
        outs, h = [], tokens
        for i, layer in enumerate(stack.layers.layers, 1):
            h = layer(h)
            if i in self.levels:
                o = stack.norm(h) if i == self.n_layers else h
                outs.append(F.layer_norm(o, (o.size(-1),)))
        return outs

    @torch.no_grad()
    def target_levels(self, tokens):
        return self._level_outputs(self.target_encoder, tokens)

    @torch.no_grad()
    def token_embeddings(self, x):
        """Target embeddings of all tokens at the last depth, (B, T, D)."""
        return self.target_levels(self.embed(x))[-1]

    def predict_levels(self, ctx_levels, ctx_idx, tgt_idx):
        """One (B, c + m, D) prediction per depth: context tokens first."""
        c = self.fuse(torch.cat(ctx_levels, dim=-1)) + self.pred_pos[ctx_idx]
        q = self.mask_token + self.pred_pos[tgt_idx]
        out = self.predictor(torch.cat([c, q], dim=1))
        return [head(out) for head in self.pred_out]

    def forward(self, x, ctx_idx, tgt_idx):
        """Returns the per-depth predictions and the per-depth targets of
        all tokens (lists of (B, c + m, D) and (B, T, D))."""
        tokens = self.embed(x)
        ctx_levels = self._level_outputs(self.encoder,
                                         _gather_tokens(tokens, ctx_idx))
        return (self.predict_levels(ctx_levels, ctx_idx, tgt_idx),
                self.target_levels(tokens))

    @torch.no_grad()
    def multi_mask_errors(self, x, ctx_idx, tgt_idx):
        """Squared error of the hidden tokens at the last depth, (B, K, m),
        and the target embeddings (B, T, D)."""
        B = x.size(0)
        K, m = tgt_idx.shape
        tokens = self.embed(x)
        h_all = self.target_levels(tokens)[-1]
        tokens_rep, ctx, tgt = _repeat_masks(tokens, ctx_idx, tgt_idx)
        ctx_levels = self._level_outputs(self.encoder,
                                         _gather_tokens(tokens_rep, ctx))
        pred = self.predict_levels(ctx_levels, ctx, tgt)[-1][:, ctx.size(1):]
        h_tgt = _gather_tokens(h_all.repeat_interleave(K, dim=0), tgt)
        return (pred - h_tgt).pow(2).mean(-1).view(B, K, m), h_all


# ----------------------------------------------------------------------
# SALT: frozen teacher trained by masked reconstruction
# ----------------------------------------------------------------------
class _TokenEncoder(nn.Module):
    """Tokenizer, learned positions and Transformer in one module."""

    def __init__(self, n_tokens, in_dim, d_model, n_heads, n_layers, dropout,
                 final_norm):
        super().__init__()
        self.tokenizer = Tokenizer(n_tokens, in_dim, d_model)
        self.pos = nn.Parameter(torch.randn(n_tokens, d_model) * 0.02)
        self.body = TransformerStack(d_model, n_heads, n_layers, dropout,
                                     final_norm=final_norm)

    def forward(self, x, idx=None):
        tok = self.tokenizer(x) + self.pos
        if idx is not None:
            tok = _gather_tokens(tok, idx)
        return self.body(tok)


class SALTCore(JEPACore):
    """Student, frozen teacher and the teacher's reconstruction decoder.

    After SALT (Li et al., arXiv:2509.24317). Stage 1 trains ``teacher``
    with :meth:`reconstruct` (the values of the hidden patches from the
    visible ones); stage 2 freezes it and trains the student (the
    :class:`JEPACore` parts) to predict the teacher's layer-normalised
    embeddings of the hidden patches. ``token_embeddings`` are the
    student's.
    """

    def __init__(self, n_tokens, in_dim, d_model=64, n_heads=4, n_layers=2,
                 pred_layers=1, dropout=0.0, encoder_norm=False):
        super().__init__(n_tokens, in_dim, d_model=d_model, n_heads=n_heads,
                         n_layers=n_layers, pred_layers=pred_layers,
                         dropout=dropout, target_mode='sigreg',
                         encoder_norm=encoder_norm)
        pred_dim = self.pred_in.out_features
        self.teacher = _TokenEncoder(n_tokens, in_dim, d_model, n_heads,
                                     n_layers, dropout, encoder_norm)
        self.dec_in = nn.Linear(d_model, pred_dim)
        self.dec_mask = nn.Parameter(torch.randn(pred_dim) * 0.02)
        self.dec_pos = nn.Parameter(torch.randn(n_tokens, pred_dim) * 0.02)
        self.decoder = TransformerStack(pred_dim, max(1, min(n_heads,
                                                             pred_dim // 16)),
                                        pred_layers, dropout)
        self.dec_out = nn.Linear(pred_dim, in_dim)

    def teacher_parameters(self):
        mods = (self.teacher, self.dec_in, self.decoder, self.dec_out)
        return [p for m in mods for p in m.parameters()] \
            + [self.dec_mask, self.dec_pos]

    def freeze_teacher(self):
        for p in self.teacher_parameters():
            p.requires_grad_(False)

    def reconstruct(self, x, ctx_idx, tgt_idx):
        """Stage 1: raw values of the hidden patches, (B, m, in_dim)."""
        h = self.teacher(x, ctx_idx)
        c = self.dec_in(h) + self.dec_pos[ctx_idx]
        q = self.dec_mask + self.dec_pos[tgt_idx]
        out = self.decoder(torch.cat([c, q], dim=1))
        return self.dec_out(out[:, c.size(1):])

    @torch.no_grad()
    def teacher_embeddings(self, x):
        h = self.teacher(x)
        return F.layer_norm(h, (h.size(-1),))

    def forward(self, x, ctx_idx, tgt_idx):
        """Returns ``pred, h_tgt, h_all`` with teacher targets."""
        tokens = self.embed(x)
        h_ctx = self.encoder(_gather_tokens(tokens, ctx_idx))
        pred = self.predict(h_ctx, ctx_idx, tgt_idx)
        t_all = self.teacher_embeddings(x)
        return pred, _gather_tokens(t_all, tgt_idx), t_all

    @torch.no_grad()
    def multi_mask_errors(self, x, ctx_idx, tgt_idx):
        B = x.size(0)
        K, m = tgt_idx.shape
        tokens = self.embed(x)
        t_all = self.teacher_embeddings(x)
        tokens_rep, ctx, tgt = _repeat_masks(tokens, ctx_idx, tgt_idx)
        h_ctx = self.encoder(_gather_tokens(tokens_rep, ctx))
        pred = self.predict(h_ctx, ctx, tgt)
        h_tgt = _gather_tokens(t_all.repeat_interleave(K, dim=0), tgt)
        return (pred - h_tgt).pow(2).mean(-1).view(B, K, m), \
            self.encoder(tokens)


# ----------------------------------------------------------------------
# KerJEPA: standardised targets
# ----------------------------------------------------------------------
class KerJEPACore(JEPACore):
    """LeJEPA-style network whose prediction targets are standardised by
    running estimates of the embedding mean and variance, which KerJEPA
    (Zimmermann et al., arXiv:2512.19605) reports as necessary."""

    def __init__(self, n_tokens, in_dim, stat_momentum=0.99, **kwargs):
        super().__init__(n_tokens, in_dim, target_mode='sigreg', **kwargs)
        self.stat_momentum = stat_momentum
        self.register_buffer('tgt_mean', torch.zeros(self.d_model))
        self.register_buffer('tgt_var', torch.ones(()))

    def standardise(self, h):
        return (h - self.tgt_mean) / self.tgt_var.clamp_min(1e-8).sqrt()

    @torch.no_grad()
    def _update_stats(self, h_all):
        flat = h_all.detach().reshape(-1, h_all.size(-1))
        m = self.stat_momentum
        self.tgt_mean.mul_(m).add_(flat.mean(0), alpha=1.0 - m)
        self.tgt_var.mul_(m).add_(flat.var(0).mean(), alpha=1.0 - m)

    def forward(self, x, ctx_idx, tgt_idx):
        pred, h_tgt, h_all = super().forward(x, ctx_idx, tgt_idx)
        if self.training:
            self._update_stats(h_all)
        return pred, self.standardise(h_tgt), h_all

    @torch.no_grad()
    def multi_mask_errors(self, x, ctx_idx, tgt_idx):
        B = x.size(0)
        K, m = tgt_idx.shape
        tokens = self.embed(x)
        h_all = self.encode_target(tokens)
        tokens_rep, ctx, tgt = _repeat_masks(tokens, ctx_idx, tgt_idx)
        pred = self.predict(self.encoder(_gather_tokens(tokens_rep, ctx)),
                            ctx, tgt)
        h_tgt = self.standardise(
            _gather_tokens(h_all.repeat_interleave(K, dim=0), tgt))
        return (pred - h_tgt).pow(2).mean(-1).view(B, K, m), h_all


# ----------------------------------------------------------------------
# Rectified LpJEPA: non-negative embeddings
# ----------------------------------------------------------------------
class _Rectified(nn.Module):
    def __init__(self, body):
        super().__init__()
        self.body = body

    def forward(self, x):
        return F.relu(self.body(x))


class RectifiedJEPACore(JEPACore):
    """LeJEPA-style network with a ReLU on the encoder output, as
    Rectified LpJEPA (Kuang et al., arXiv:2602.01456) requires."""

    def __init__(self, n_tokens, in_dim, **kwargs):
        super().__init__(n_tokens, in_dim, target_mode='sigreg', **kwargs)
        self.encoder = _Rectified(self.encoder)


# ----------------------------------------------------------------------
# next-latent prediction: LeNEPA and LeWorldModel
# ----------------------------------------------------------------------
class NextLatentCore(nn.Module):
    """Causal next-patch prediction in latent space, without masks.

    ``style='lenepa'`` follows LeNEPA (Chemeris, Jin and Balestriero,
    arXiv:2607.00958): one linear patch embedding ``z``, a causal
    Transformer over it, and a shared projector; position ``t`` of the
    Transformer predicts ``z`` of patch ``t + 1`` in the projector's
    space. ``style='lewm'`` follows the two-term objective of
    LeWorldModel (Maes et al., arXiv:2603.19312) with its action input
    removed: a per-patch encoder gives ``z`` and a causal Transformer
    predicts ``z`` of the next patch directly.

    ``token_embeddings`` are the Transformer outputs for ``'lenepa'`` (what
    that paper reads out) and the per-patch embeddings ``z`` for ``'lewm'``.
    """

    def __init__(self, n_tokens, in_dim, d_model=64, n_heads=4, n_layers=2,
                 dropout=0.0, encoder_norm=False, style='lenepa',
                 proj_dim=64):
        super().__init__()
        if style not in ('lenepa', 'lewm'):
            raise ValueError("style must be 'lenepa' or 'lewm'")
        self.n_tokens = n_tokens
        self.d_model = d_model
        self.style = style
        if style == 'lenepa':
            self.tokenizer = nn.Linear(in_dim, d_model)
            self.projector = nn.Sequential(
                nn.Linear(d_model, 2 * d_model), nn.BatchNorm1d(2 * d_model),
                nn.ReLU(), nn.Linear(2 * d_model, proj_dim))
        else:
            self.tokenizer = nn.Sequential(
                nn.Linear(in_dim, d_model), nn.GELU(),
                nn.Linear(d_model, d_model))
            self.head = nn.Linear(d_model, d_model)
        self.pos = nn.Parameter(torch.randn(n_tokens, d_model) * 0.02)
        self.backbone = TransformerStack(d_model, n_heads, n_layers, dropout,
                                         final_norm=encoder_norm)
        self.register_buffer('causal', torch.triu(
            torch.full((n_tokens, n_tokens), float('-inf')), diagonal=1))

    def eval_mask_set(self):
        """Every patch but the first is predicted from the ones before it."""
        return (torch.zeros(1, 1, dtype=torch.long),
                torch.arange(1, self.n_tokens).unsqueeze(0))

    def _causal(self, z):
        h = self.backbone.layers(z + self.pos, mask=self.causal)
        return self.backbone.norm(h)

    def _project(self, h):
        B, T, D = h.shape
        return self.projector(h.reshape(B * T, D)).view(B, T, -1)

    def forward(self, x):
        """Returns ``pred`` (B, T-1, E), ``tgt`` (B, T-1, E), the list of
        (B, T, E) embeddings to regularise, and the Transformer output."""
        z = self.tokenizer(x)
        y = self._causal(z)
        if self.style == 'lenepa':
            u_in, u_out = self._project(z), self._project(y)
            return u_out[:, :-1], u_in[:, 1:], [u_in, u_out], y
        return self.head(y)[:, :-1], z[:, 1:], [z], y

    @torch.no_grad()
    def token_embeddings(self, x):
        z = self.tokenizer(x)
        return self._causal(z) if self.style == 'lenepa' else z

    @torch.no_grad()
    def multi_mask_errors(self, x, ctx_idx, tgt_idx):
        """Next-latent error of the patches in ``tgt_idx`` (>= 1), (B, K, m),
        and the token embeddings. ``ctx_idx`` is not used."""
        pred, tgt, reg, y = self.forward(x)
        err = (pred - tgt).pow(2).mean(-1)                 # (B, T-1)
        h_all = y if self.style == 'lenepa' else reg[0]
        return err[:, tgt_idx.to(x.device) - 1], h_all


# ----------------------------------------------------------------------
# CF-JEPA: crops, forward prediction, convolutional encoder
# ----------------------------------------------------------------------
class _DilatedBlock(nn.Module):
    """Parallel depthwise convolutions (kernels 3, 9, 15), batch norm,
    GELU, a pointwise convolution and a residual connection.

    Input and output are (B, L, C). The convolutions are computed as sums
    over the filter taps: the grouped dilated ``Conv1d`` of PyTorch is
    more than ten times slower on CPU. The ``Conv1d`` modules only hold
    the parameters.
    """

    def __init__(self, width, dilation, kernels=(3, 9, 15)):
        super().__init__()
        self.dilation = dilation
        self.convs = nn.ModuleList([
            nn.Conv1d(width, width, k, groups=width) for k in kernels])
        self.bn = nn.BatchNorm1d(width)
        self.point = nn.Conv1d(width, width, 1)

    def forward(self, x):
        B, L, C = x.shape
        d, y = self.dilation, 0.0
        for conv in self.convs:
            k = conv.kernel_size[0]
            pad = d * (k - 1) // 2            # zero 'same' padding
            xp = F.pad(x, (0, 0, pad, pad))
            w = conv.weight[:, 0]             # (C, k)
            y = y + conv.bias
            for j in range(k):
                if abs(j * d - pad) < L:      # other taps only see padding
                    y = y + xp[:, j * d:j * d + L] * w[:, j]
        y = F.gelu(self.bn(y.reshape(B * L, C))).view(B, L, C)
        return x + F.linear(y, self.point.weight[:, :, 0], self.point.bias)


class _ConvEncoder(nn.Module):
    """(B, L, C) series -> (B, L, D): one embedding per time step."""

    def __init__(self, n_channels, width, out_dim, n_blocks):
        super().__init__()
        self.inp = nn.Linear(n_channels, width)
        self.blocks = nn.Sequential(
            *[_DilatedBlock(width, 2 ** i) for i in range(n_blocks)])
        self.out = nn.Linear(width, out_dim)

    def forward(self, x):
        return self.out(self.blocks(self.inp(x)))


class CFJEPACore(nn.Module):
    """Network and training terms of the ``'cfjepa'`` version.

    After CF-JEPA (Lee and Sim, arXiv:2606.07031). There are no masks: a
    random crop of the window is the context, the part of the window after
    the crop is cut into three zones, and one linear predictor per zone
    maps the crop's per-time-step embeddings to the EMA encoder's
    embeddings of that zone. The paper does not say how a prediction of the
    crop's length is laid over a zone of another length; here it is
    average-pooled along time to the zone's length. One crop position is
    drawn per crop and step and shared by the batch.

    For scoring, the context is the leading ``n_ctx_tokens`` patches and
    every later patch gets the mean absolute error of its time steps.
    ``token_embeddings`` are the EMA encoder's embeddings averaged within
    each patch (the paper scores with the EMA encoder).
    """

    def __init__(self, n_tokens, patch_len, n_channels, d_model=64,
                 n_blocks=5, n_crops=4, crop_ratio=None):
        super().__init__()
        self.n_tokens = n_tokens
        self.patch_len = patch_len
        self.window = n_tokens * patch_len
        self.d_model = d_model
        self.n_crops = n_crops
        if crop_ratio is None:   # the published ranges
            crop_ratio = (0.367, 0.418) if self.window >= 50 else (0.6, 0.8)
        self.crop_ratio = tuple(crop_ratio)
        mid = 0.5 * (self.crop_ratio[0] + self.crop_ratio[1])
        self.n_ctx_tokens = int(min(n_tokens - 1,
                                    max(1, round(mid * n_tokens))))
        self.encoder = _ConvEncoder(n_channels, d_model, d_model, n_blocks)
        self.predictors = nn.ModuleList(
            [nn.Linear(d_model, d_model) for _ in range(3)])
        for p in self.predictors:   # near the identity
            with torch.no_grad():
                p.weight.copy_(torch.eye(d_model)
                               + 1e-3 * torch.randn(d_model, d_model))
                p.bias.zero_()
        self.target_encoder = _frozen_copy(self.encoder)

    def eval_mask_set(self):
        c = self.n_ctx_tokens
        return (torch.arange(c).unsqueeze(0),
                torch.arange(c, self.n_tokens).unsqueeze(0))

    def series(self, x):
        """Patches (B, T, patch_len * C) back to the window (B, L, C)."""
        return x.reshape(x.size(0), self.window, -1)

    def _zone_errors(self, z_crop, z_future):
        """Per-time-step L1 distance between the unit-normalised zone
        predictions and targets, (B, R); None if nothing follows the crop.
        A future shorter than three steps has one zone per step."""
        R = z_future.size(1)
        if R < 1:
            return None
        out = []
        for pred, idx in zip(self.predictors,
                             torch.tensor_split(torch.arange(R), min(3, R))):
            p = F.adaptive_avg_pool1d(pred(z_crop).transpose(1, 2),
                                      len(idx)).transpose(1, 2)
            t = z_future[:, idx[0]:idx[-1] + 1]
            out.append((F.normalize(p, dim=-1)
                        - F.normalize(t, dim=-1)).abs().mean(-1))
        return torch.cat(out, dim=1)

    def training_terms(self, x, generator=None):
        """Returns the horizon loss and the list of crop embeddings
        (each (B, L_i, D)) for the regularisers."""
        xs = self.series(x)
        W = self.window
        with torch.no_grad():
            z_full = self.target_encoder(xs)
        lo, hi = self.crop_ratio
        crops, horizon = [], []
        for _ in range(self.n_crops):
            r = lo + (hi - lo) * torch.rand((), generator=generator).item()
            L = int(min(W - 1, max(2, int(r * W))))
            s = int(torch.randint(0, W - L + 1, (), generator=generator))
            z_crop = self.encoder(xs[:, s:s + L])
            crops.append(z_crop)
            e = self._zone_errors(z_crop, z_full[:, s + L:])
            if e is not None:
                horizon.append(e.mean())
        loss = torch.stack(horizon).mean() if horizon else xs.sum() * 0.0
        return loss, crops

    @torch.no_grad()
    def token_embeddings(self, x):
        h = self.target_encoder(self.series(x))
        return h.view(x.size(0), self.n_tokens, self.patch_len, -1).mean(2)

    @torch.no_grad()
    def multi_mask_errors(self, x, ctx_idx, tgt_idx):
        """Error of the patches after the leading crop, (B, 1, m), and the
        token embeddings. The split is fixed; the arguments are unused."""
        xs = self.series(x)
        L = self.n_ctx_tokens * self.patch_len
        z_full = self.target_encoder(xs)
        e = self._zone_errors(self.encoder(xs[:, :L]), z_full[:, L:])
        err = e.view(x.size(0), self.n_tokens - self.n_ctx_tokens,
                     self.patch_len).mean(-1)
        h_all = z_full.view(x.size(0), self.n_tokens, self.patch_len,
                            -1).mean(2)
        return err.unsqueeze(1), h_all
