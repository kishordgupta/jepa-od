# -*- coding: utf-8 -*-
"""Networks of the ten JEPA versions added to :class:`jepa_od.TSJEPAOD`
in version 0.5.0.

As in ``_variant_nets.py``, each class offers the scoring interface of
:class:`jepa_od._nets.JEPACore` (``token_embeddings`` and
``multi_mask_errors``; networks that do not mask patches also define
``eval_mask_set``), is written from the paper named in its docstring, and
takes its sizes from the detector, not from the paper.
"""
# License: BSD 2 clause

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._nets import JEPACore, TransformerStack, _gather_tokens
from ._variant_nets import NextLatentCore, _frozen_copy, _repeat_masks


def cosine_distance(u, v):
    """``2 - 2 cos(u, v)`` along the last dimension."""
    return 2.0 - 2.0 * F.cosine_similarity(u, v, dim=-1)


# ----------------------------------------------------------------------
# STST-JEPA: the target is an EMA copy of the tokenizer
# ----------------------------------------------------------------------
class ShallowTargetCore(JEPACore):
    """Network of the ``'ststjepa'`` version.

    After STST-JEPA (arXiv:2607.06629): the targets of the hidden patches
    are the outputs of an EMA copy of the *tokenizer*, not of the encoder,
    which has no EMA twin. The predictor output goes through a projection
    head ``W2 GELU(W1 LayerNorm(.))`` into the target space, and a second
    head maps each projected prediction back to the values of its patch.
    ``token_embeddings`` are the encoder's.
    """

    def __init__(self, n_tokens, in_dim, **kwargs):
        super().__init__(n_tokens, in_dim, target_mode='sigreg', **kwargs)
        d = self.d_model
        self.target_tokenizer = _frozen_copy(self.tokenizer)
        self.proj = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d),
                                  nn.GELU(), nn.Linear(d, d))
        self.rec_head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2 * d),
                                      nn.GELU(), nn.Linear(2 * d, in_dim))

    def predict(self, h_ctx, ctx_idx, tgt_idx):
        return self.proj(super().predict(h_ctx, ctx_idx, tgt_idx))

    def forward(self, x, ctx_idx, tgt_idx):
        """Returns the projected predictions (B, m, D), the EMA tokenizer's
        targets (B, m, D) and the reconstructed patches (B, m, in_dim)."""
        h_ctx = self.encoder(_gather_tokens(self.embed(x), ctx_idx))
        pred = self.predict(h_ctx, ctx_idx, tgt_idx)
        with torch.no_grad():
            target = _gather_tokens(self.target_tokenizer(x), tgt_idx)
        return pred, target, self.rec_head(pred)

    @torch.no_grad()
    def multi_mask_errors(self, x, ctx_idx, tgt_idx):
        B = x.size(0)
        K, m = tgt_idx.shape
        tokens = self.embed(x)
        tokens_rep, ctx, tgt = _repeat_masks(tokens, ctx_idx, tgt_idx)
        pred = self.predict(self.encoder(_gather_tokens(tokens_rep, ctx)),
                            ctx, tgt)
        target = _gather_tokens(
            self.target_tokenizer(x).repeat_interleave(K, dim=0), tgt)
        return (pred - target).pow(2).mean(-1).view(B, K, m), \
            self.encoder(tokens)


# ----------------------------------------------------------------------
# HEPA: horizon-conditioned prediction of a future interval
# ----------------------------------------------------------------------
def _sinusoid(n, d):
    pos = torch.arange(n, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, d, 2, dtype=torch.float32)
                    * (-math.log(10000.0) / d))
    pe = torch.zeros(n, d)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)[:, :d // 2]
    return pe


class HEPACore(nn.Module):
    """Network of the ``'hepa'`` version.

    After HEPA (arXiv:2605.11130): a causal Transformer summarises the
    patches up to a point ``t``; a two-layer MLP maps that summary and a
    horizon ``dt`` to the embedding of the interval of the next ``dt``
    patches; the target is the *same* encoder applied without the causal
    mask to that interval alone and attention-pooled. No EMA and no
    stop-gradient.

    The paper feeds the horizon to the MLP without saying how; here a
    learned embedding of ``dt`` is added to the summary. The interval is
    encoded with positions that restart at zero.

    For scoring, every interval of ``score_horizons`` patches that starts
    after the first patch is predicted, and a patch gets the mean absolute
    error of the intervals that contain it. ``token_embeddings`` are the
    causal summaries.
    """

    def __init__(self, n_tokens, in_dim, d_model=64, n_heads=4, n_layers=2,
                 dropout=0.0, encoder_norm=False):
        super().__init__()
        self.n_tokens = n_tokens
        self.d_model = d_model
        self.tokenizer = nn.Linear(in_dim, d_model)
        self.register_buffer('pos', _sinusoid(n_tokens, d_model))
        self.encoder = TransformerStack(d_model, n_heads, n_layers, dropout,
                                        final_norm=encoder_norm)
        self.register_buffer('causal', torch.triu(
            torch.full((n_tokens, n_tokens), float('-inf')), diagonal=1))
        self.pool = nn.Linear(d_model, 1)
        self.horizon = nn.Embedding(n_tokens, d_model)
        self.predictor = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, d_model))
        self.score_horizons = tuple(
            h for h in (1, 2, 4, 8, 16, 32) if h <= n_tokens - 1)

    def eval_mask_set(self):
        """Every patch but the first lies in a predicted interval."""
        return (torch.zeros(1, 1, dtype=torch.long),
                torch.arange(1, self.n_tokens).unsqueeze(0))

    def summaries(self, x):
        """(B, T, D): entry ``t`` summarises the patches ``0 .. t``."""
        h = self.encoder.layers(self.tokenizer(x) + self.pos,
                                mask=self.causal)
        return self.encoder.norm(h)

    def interval(self, x):
        """Embedding (N, D) of intervals given as (N, dt, in_dim)."""
        h = self.encoder(self.tokenizer(x) + self.pos[:x.size(1)])
        w = torch.softmax(self.pool(h), dim=1)
        return (w * h).sum(1)

    def predict(self, summary, dt):
        return self.predictor(summary + self.horizon.weight[dt - 1])

    @torch.no_grad()
    def token_embeddings(self, x):
        return self.summaries(x)

    @torch.no_grad()
    def multi_mask_errors(self, x, ctx_idx, tgt_idx):
        """Error of the patches in ``tgt_idx`` (>= 1), (B, 1, m), and the
        token embeddings. ``ctx_idx`` is not used."""
        B, T = x.size(0), self.n_tokens
        h = self.summaries(x)
        acc = x.new_zeros(B, T)
        cnt = x.new_zeros(T)
        for dt in self.score_horizons:
            # the intervals of dt patches that start at patch 1 .. T - dt
            spans = x.unfold(1, dt, 1)[:, 1:].permute(0, 1, 3, 2)
            n = spans.size(1)
            target = self.interval(spans.reshape(B * n, dt, -1)).view(B, n, -1)
            err = (self.predict(h[:, :n], dt) - target).abs().mean(-1)
            for s in range(n):
                acc[:, s + 1:s + 1 + dt] += err[:, s:s + 1]
                cnt[s + 1:s + 1 + dt] += 1
        err = acc[:, 1:] / cnt[1:]
        return err[:, tgt_idx.to(x.device) - 1], h


# ----------------------------------------------------------------------
# Sub-JEPA: SIGReg in frozen random subspaces
# ----------------------------------------------------------------------
class SubspaceLatentCore(NextLatentCore):
    """LeWorldModel-style network with the frozen row-orthonormal random
    projections of Sub-JEPA (arXiv:2605.09241), each from the QR
    decomposition of a Gaussian matrix. The paper uses ``K = 32``
    subspaces at embedding dimension 192, i.e. six dimensions each; here
    the subspace dimension is kept and ``K = round(d_model / 6)``."""

    def __init__(self, n_tokens, in_dim, subspace_dim=6, seed=0, **kwargs):
        super().__init__(n_tokens, in_dim, style='lewm', **kwargs)
        D = self.d_model
        d_s = int(min(D, max(1, subspace_dim)))
        K = max(1, int(round(D / d_s)))
        gen = torch.Generator().manual_seed(int(seed))
        self.register_buffer('subspaces', torch.stack([
            torch.linalg.qr(torch.randn(D, d_s, generator=gen))[0].T
            for _ in range(K)]))                           # (K, d_s, D)

    def project(self, z):
        """(N, D) embeddings -> (K, N, d_s), one group per subspace."""
        return torch.einsum('nd,ksd->kns', z, self.subspaces)


# ----------------------------------------------------------------------
# MotionJEPA: embeddings of patch differences
# ----------------------------------------------------------------------
class MotionCore(NextLatentCore):
    """LeWorldModel-style network with the two modules of DISReg
    (MotionJEPA, arXiv:2609.23881): an encoder of the difference of two
    consecutive observations (here: patches) and a predictor of that
    embedding from the embeddings of the two observations."""

    def __init__(self, n_tokens, in_dim, **kwargs):
        super().__init__(n_tokens, in_dim, style='lewm', **kwargs)
        d = self.d_model
        self.diff_enc = nn.Sequential(nn.Linear(in_dim, d), nn.GELU(),
                                      nn.Linear(d, d))
        self.diff_pred = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(),
                                       nn.Linear(d, d))

    def difference_terms(self, x, z):
        """Embedding of every patch difference and its prediction from
        the two patch embeddings, both (B, T - 1, D)."""
        d = self.diff_enc(x[:, 1:] - x[:, :-1])
        return d, self.diff_pred(torch.cat([z[:, :-1], z[:, 1:]], dim=-1))


# ----------------------------------------------------------------------
# masked versions with an EMA target
# ----------------------------------------------------------------------
class _HiddenOutCore(JEPACore):
    """I-JEPA network that also exposes the predictor's output before the
    last linear layer."""

    def __init__(self, n_tokens, in_dim, **kwargs):
        super().__init__(n_tokens, in_dim, target_mode='ema', **kwargs)

    def predict_hidden(self, h_ctx, ctx_idx, tgt_idx):
        c = self.pred_in(h_ctx) + self.pred_pos[ctx_idx]
        q = self.mask_token + self.pred_pos[tgt_idx]
        return self.predictor(torch.cat([c, q], dim=1))[:, c.size(1):]

    def context(self, x, ctx_idx):
        """Returns the token embeddings and the encoded visible tokens."""
        tokens = self.embed(x)
        return tokens, self.encoder(_gather_tokens(tokens, ctx_idx))


class NoiseJEPACore(JEPACore):
    """Network of the ``'njepa'`` version.

    After N-JEPA (arXiv:2507.15216): a second predictor with its own
    parameters and mask token, whose mask-token inputs are perturbed by
    Gaussian noise. Scoring uses the first predictor, without noise.
    """

    def __init__(self, n_tokens, in_dim, d_model=64, n_heads=4, n_layers=2,
                 pred_layers=1, dropout=0.0, encoder_norm=False):
        super().__init__(n_tokens, in_dim, d_model=d_model, n_heads=n_heads,
                         n_layers=n_layers, pred_layers=pred_layers,
                         dropout=dropout, target_mode='ema',
                         encoder_norm=encoder_norm)
        pred_dim = self.pred_in.out_features
        self.pred_in2 = nn.Linear(d_model, pred_dim)
        self.mask_token2 = nn.Parameter(torch.randn(pred_dim) * 0.02)
        self.pred_pos2 = nn.Parameter(torch.randn(n_tokens, pred_dim) * 0.02)
        self.predictor2 = TransformerStack(
            pred_dim, max(1, min(n_heads, pred_dim // 16)), pred_layers,
            dropout)
        self.pred_out2 = nn.Linear(pred_dim, d_model)

    def predict_noisy(self, h_ctx, ctx_idx, tgt_idx, noise):
        """Second predictor; ``noise`` (B, m, pred_dim) is added to the
        mask tokens and their position embeddings."""
        c = self.pred_in2(h_ctx) + self.pred_pos2[ctx_idx]
        q = self.mask_token2 + self.pred_pos2[tgt_idx] + noise
        out = self.predictor2(torch.cat([c, q], dim=1))
        return self.pred_out2(out[:, c.size(1):])


class GMMJEPACore(_HiddenOutCore):
    """Network of the ``'gmmjepa'`` version.

    After GMM-Anchored JEPA (arXiv:2602.09040): a frozen diagonal Gaussian
    mixture of the input features (here: the patches) gives soft cluster
    posteriors, and a linear cluster head is trained to match them at the
    hidden positions. The paper's head reads the student encoder, which
    sees mask embeddings at those positions; this encoder does not see the
    hidden patches at all, so the head reads the predictor's output.
    """

    def __init__(self, n_tokens, in_dim, n_components=32, **kwargs):
        super().__init__(n_tokens, in_dim, **kwargs)
        k = int(n_components)
        self.cluster_head = nn.Linear(self.pred_in.out_features, k)
        self.register_buffer('gmm_mean', torch.zeros(k, in_dim))
        self.register_buffer('gmm_prec', torch.ones(k, in_dim))
        self.register_buffer('gmm_const', torch.zeros(k))

    def set_gmm(self, weights, means, variances):
        """Store a fitted mixture; components beyond those given get zero
        weight."""
        k = len(weights)
        with torch.no_grad():
            var = torch.as_tensor(variances, dtype=torch.float32)
            self.gmm_mean[:k] = torch.as_tensor(means, dtype=torch.float32)
            self.gmm_prec[:k] = 1.0 / var
            self.gmm_const[:k] = (
                torch.as_tensor(weights, dtype=torch.float32).clamp_min(
                    1e-30).log() - 0.5 * var.log().sum(1))
            self.gmm_const[k:] = -1e30

    @torch.no_grad()
    def posteriors(self, x):
        """Mixture posteriors (..., K) of raw tokens (..., in_dim)."""
        mp = self.gmm_mean * self.gmm_prec
        d2 = (x.pow(2) @ self.gmm_prec.T - 2.0 * x @ mp.T
              + (self.gmm_mean * mp).sum(1))
        return torch.softmax(self.gmm_const - 0.5 * d2, dim=-1)

    def forward_cluster(self, x, ctx_idx, tgt_idx):
        """Returns ``pred, h_tgt`` and the cluster logits (B, m, K)."""
        tokens, h_ctx = self.context(x, ctx_idx)
        hidden = self.predict_hidden(h_ctx, ctx_idx, tgt_idx)
        h_tgt = _gather_tokens(self.encode_target(tokens), tgt_idx)
        return self.pred_out(hidden), h_tgt, self.cluster_head(hidden)


class VariationalJEPACore(_HiddenOutCore):
    """Network of the ``'variational'`` version.

    After VJEPA (arXiv:2601.14354): the predictor returns a diagonal
    Gaussian over the embedding of each hidden patch instead of a point.
    The error of a hidden patch is the negative log-likelihood of its EMA
    target under that Gaussian (up to a constant), so a patch the model
    expects to be hard to predict is penalised less.
    """

    def __init__(self, n_tokens, in_dim, **kwargs):
        super().__init__(n_tokens, in_dim, **kwargs)
        self.pred_logvar = nn.Linear(self.pred_in.out_features, self.d_model)

    def predict_dist(self, h_ctx, ctx_idx, tgt_idx):
        hidden = self.predict_hidden(h_ctx, ctx_idx, tgt_idx)
        return self.pred_out(hidden), self.pred_logvar(hidden).clamp(-8., 8.)

    @staticmethod
    def nll(target, mean, logvar):
        """Per-element Gaussian negative log-likelihood without its
        constant."""
        return 0.5 * (logvar + (target - mean).pow(2) / logvar.exp())

    def forward_dist(self, x, ctx_idx, tgt_idx):
        """Returns the predictive mean and log-variance (B, m, D), the EMA
        targets (B, m, D) and the online encoder's embeddings of all
        tokens (B, T, D)."""
        tokens, h_ctx = self.context(x, ctx_idx)
        mean, logvar = self.predict_dist(h_ctx, ctx_idx, tgt_idx)
        h_tgt = _gather_tokens(self.encode_target(tokens), tgt_idx)
        return mean, logvar, h_tgt, self.encoder(tokens)

    @torch.no_grad()
    def multi_mask_errors(self, x, ctx_idx, tgt_idx):
        B = x.size(0)
        K, m = tgt_idx.shape
        tokens = self.embed(x)
        h_all = self.encode_target(tokens)
        tokens_rep, ctx, tgt = _repeat_masks(tokens, ctx_idx, tgt_idx)
        mean, logvar = self.predict_dist(
            self.encoder(_gather_tokens(tokens_rep, ctx)), ctx, tgt)
        h_tgt = _gather_tokens(h_all.repeat_interleave(K, dim=0), tgt)
        return self.nll(h_tgt, mean, logvar).mean(-1).view(B, K, m), h_all


class PhiJEPACore(JEPACore):
    """Network of the ``'phijepa'`` version.

    After PhiJEPA (arXiv:2611.27516): the predictor's output for a hidden
    patch (first stage) is trained towards the EMA encoder's token of an
    independently corrupted view, and a token-wise two-layer MLP maps it
    on to the token of the clean view (second stage). The error of a
    hidden patch is the cosine distance of the second stage to the clean
    token, computed from uncorrupted input.

    The paper corrupts images photometrically. :meth:`corrupt` is this
    implementation's counterpart for a series: a gain and an offset per
    channel, a three-point moving average along time, and additive
    Gaussian noise, each applied with its own probability.
    """

    def __init__(self, n_tokens, in_dim, n_channels=1, **kwargs):
        super().__init__(n_tokens, in_dim, target_mode='ema', **kwargs)
        d = self.d_model
        self.n_channels = n_channels
        hidden = int(round(8 * d / 3))       # the paper: 768 -> 2048 -> 768
        self.head = nn.Sequential(nn.Linear(d, hidden), nn.GELU(),
                                  nn.Linear(hidden, d))

    def corrupt(self, x, generator=None, p=1.0, gain_std=0.1, noise_std=0.1,
                p_gain=0.8, p_smooth=0.5, p_noise=0.5):
        """A corrupted copy of the raw tokens (B, T, in_dim). The draws
        are made on the CPU, so they do not depend on the device."""
        B, C = x.size(0), self.n_channels

        def rand(*shape):
            return torch.rand(*shape, generator=generator)

        def randn(*shape):
            return torch.randn(*shape, generator=generator)

        on = rand(B, 1, 1) < p
        gain = (on & (rand(B, 1, 1) < p_gain)).to(x)
        smooth = (on & (rand(B, 1, 1) < p_smooth)).to(x)
        noise = (on & (rand(B, 1, 1) < p_noise)).to(x)
        a, b = randn(B, 1, C).to(x), randn(B, 1, C).to(x)
        xs = x.reshape(B, -1, C)                       # (B, window, C)
        eps = randn(*xs.shape).to(x)
        xs = xs * (1.0 + gain * gain_std * a) + gain * gain_std * b
        avg = F.avg_pool1d(xs.transpose(1, 2), 3, stride=1, padding=1,
                           count_include_pad=False).transpose(1, 2)
        xs = smooth * avg + (1.0 - smooth) * xs
        return (xs + noise * noise_std * eps).reshape(x.shape)

    @torch.no_grad()
    def multi_mask_errors(self, x, ctx_idx, tgt_idx):
        B = x.size(0)
        K, m = tgt_idx.shape
        tokens = self.embed(x)
        h_all = self.encode_target(tokens)
        tokens_rep, ctx, tgt = _repeat_masks(tokens, ctx_idx, tgt_idx)
        pred = self.head(self.predict(
            self.encoder(_gather_tokens(tokens_rep, ctx)), ctx, tgt))
        h_tgt = _gather_tokens(h_all.repeat_interleave(K, dim=0), tgt)
        return cosine_distance(pred, h_tgt).view(B, K, m), h_all


class PlainTargetCore(JEPACore):
    """I-JEPA network whose EMA targets are *not* layer-normalised, as in
    HAR-JEPA (arXiv:2607.16350), whose norm term asks the predictions for
    a set vector norm."""

    def __init__(self, n_tokens, in_dim, **kwargs):
        super().__init__(n_tokens, in_dim, target_mode='ema', **kwargs)

    def encode_target(self, tokens):
        with torch.no_grad():
            return self.target_encoder(tokens)


# ----------------------------------------------------------------------
# SC-JEPA (MTS-JEPA): soft codes, two resolutions, one variable at a time
# ----------------------------------------------------------------------
class _CodeEncoder(nn.Module):
    """An MLP per patch of one variable, sinusoidal positions and a
    Transformer; takes any number of patches up to ``max_tokens``."""

    def __init__(self, patch_len, d_model, n_heads, n_layers, dropout,
                 final_norm, max_tokens):
        super().__init__()
        self.tokenizer = nn.Sequential(nn.Linear(patch_len, d_model),
                                       nn.GELU(), nn.Linear(d_model, d_model))
        self.register_buffer('pos', _sinusoid(max_tokens, d_model))
        self.body = TransformerStack(d_model, n_heads, n_layers, dropout,
                                     final_norm=final_norm)

    def forward(self, x):
        return self.body(self.tokenizer(x) + self.pos[:x.size(1)])


class _Codebook(nn.Module):
    """Soft codebook: the distribution over the prototypes is a softmax of
    the cosine similarities divided by the temperature."""

    def __init__(self, n_codes, d_model, temperature):
        super().__init__()
        self.codes = nn.Parameter(torch.randn(n_codes, d_model)
                                  / math.sqrt(d_model))
        self.temperature = temperature

    def forward(self, h):
        """Returns the log code probabilities (..., K) and the expected
        code embedding (..., D)."""
        logits = F.normalize(h, dim=-1) @ F.normalize(self.codes, dim=-1).T
        logp = F.log_softmax(logits / self.temperature, dim=-1)
        return logp, logp.exp() @ self.codes


def _kl(target_logp, logp):
    """KL(target || prediction) of log-probabilities, over the last axis."""
    return (target_logp.exp() * (target_logp - logp)).sum(-1)


class SCJEPACore(nn.Module):
    """Network and training terms of the ``'scjepa'`` version.

    After SC-JEPA (He et al., arXiv:2602.04643; its first version is named
    MTS-JEPA), which predicts the next window from the current one. Here
    the window of the detector is cut in two: the first half is the
    paper's current window and the second half the next one.

    * Every variable is processed on its own (the paper's
      channel-independent input); variables that are constant in the
      training data are dropped (:meth:`set_channels`).
    * Each half is normalised by its own mean and standard deviation per
      variable (the paper applies RevIN; no affine map is learned here).
      A level or scale that differs between the halves, or from the
      training data, is therefore invisible to the network;
      ``instance_norm=False`` switches the normalisation off.
    * A shared encoder embeds the patches of the first half; a soft
      codebook turns every embedding into a distribution over
      ``n_codes`` prototypes and into the expected prototype.
    * A fine predictor maps the code distributions of the first half to
      those of the patches of the second half, a coarse predictor (a
      learned query attending to the history) to the code distribution of
      one coarse view of the second half: its time-average over every
      ``n_fut`` steps, which has the length of one patch.
    * EMA copies of the encoder and of the codebook give the targets.
    * A linear decoder reconstructs the patches of the first half from
      their expected prototypes, in the units before normalisation.

    For scoring, a patch of the second half gets, per variable, the
    training's predictive terms (the KL divergence of its code
    distribution, ``mse_weight`` times the squared error of its expected
    prototype, and ``coarse_weight`` times the coarse KL divergence of its
    window), and the largest value over the variables. The embedding of a
    patch is the EMA branch's code distribution, with the largest
    probability over the variables for every code: the features the paper
    passes to its downstream classifier.
    """

    def __init__(self, n_tokens, patch_len, n_channels, d_model=64, n_heads=4,
                 n_layers=2, pred_layers=1, dropout=0.0, encoder_norm=False,
                 n_codes=128, temperature=0.1, mse_weight=0.1,
                 coarse_weight=0.5, instance_norm=True, max_sequences=32768):
        super().__init__()
        self.n_tokens = n_tokens
        self.patch_len = patch_len
        self.n_channels = n_channels
        self.instance_norm = bool(instance_norm)
        self.n_fut = n_tokens // 2
        self.n_ctx = n_tokens - self.n_fut
        self.d_model = d_model
        self.mse_weight = mse_weight
        self.coarse_weight = coarse_weight
        self.max_sequences = max_sequences
        pred_dim = max(16, d_model // 2)
        pred_heads = max(1, min(n_heads, pred_dim // 16))

        self.encoder = _CodeEncoder(patch_len, d_model, n_heads, n_layers,
                                    dropout, encoder_norm, self.n_ctx)
        self.codebook = _Codebook(n_codes, d_model, temperature)
        self.target_encoder = _frozen_copy(self.encoder)
        self.target_codebook = _frozen_copy(self.codebook)

        self.fine_in = nn.Linear(n_codes, pred_dim)
        self.fine_pos = nn.Parameter(torch.randn(self.n_ctx, pred_dim) * 0.02)
        self.fine_body = TransformerStack(pred_dim, pred_heads, pred_layers,
                                          dropout)
        self.fine_out = nn.Linear(pred_dim, n_codes)
        self.coarse_in = nn.Linear(n_codes, pred_dim)
        self.coarse_pos = nn.Parameter(
            torch.randn(self.n_ctx, pred_dim) * 0.02)
        self.coarse_body = TransformerStack(pred_dim, pred_heads, pred_layers,
                                            dropout)
        self.coarse_query = nn.Parameter(torch.randn(pred_dim) * 0.02)
        self.coarse_attn = nn.MultiheadAttention(
            pred_dim, pred_heads, dropout=dropout, batch_first=True)
        self.coarse_out = nn.Linear(pred_dim, n_codes)
        self.decoder = nn.Linear(d_model, patch_len)
        self.register_buffer('keep', torch.ones(n_channels, dtype=torch.bool))

    def set_channels(self, keep):
        """Use only the variables marked in ``keep`` (all, if none is)."""
        keep = torch.as_tensor(keep, dtype=torch.bool)
        if not keep.any():
            keep = torch.ones_like(keep)
        self.keep.copy_(keep.to(self.keep.device))

    def eval_mask_set(self):
        """The first half is the context, the second half is predicted."""
        return (torch.arange(self.n_ctx).unsqueeze(0),
                torch.arange(self.n_ctx, self.n_tokens).unsqueeze(0))

    def views(self, x, eps=1e-5):
        """Raw tokens (B, T, patch_len * C) -> one sequence per kept
        variable, N = B * V in all: the normalised patches of the first
        half (N, n_ctx, L) and of the second half (N, n_fut, L), the first
        half before normalisation, and its mean and standard deviation
        (N, 1, 1)."""
        L = self.patch_len
        xs = x.reshape(x.size(0), self.n_tokens * L, self.n_channels)
        xs = xs[:, :, self.keep].transpose(1, 2).reshape(-1, self.n_tokens * L)

        def normalise(u):
            if not self.instance_norm:     # an ablation, not the paper
                return u, torch.zeros_like(u[:, :1]), torch.ones_like(u[:, :1])
            mean = u.mean(1, keepdim=True)
            std = (u.var(1, unbiased=False, keepdim=True) + eps).sqrt()
            return (u - mean) / std, mean, std

        raw, fut = xs[:, :self.n_ctx * L], xs[:, self.n_ctx * L:]
        ctx, mean, std = normalise(raw)
        fut = normalise(fut)[0]
        N = xs.size(0)
        return (ctx.view(N, self.n_ctx, L), fut.view(N, self.n_fut, L),
                raw.view(N, self.n_ctx, L), mean.unsqueeze(-1),
                std.unsqueeze(-1))

    def coarse_view(self, fut):
        """The second half (N, n_fut, L) averaged over every ``n_fut``
        time steps: one patch (N, 1, L)."""
        N = fut.size(0)
        return fut.reshape(N, self.patch_len, self.n_fut).mean(-1).unsqueeze(1)

    def predict(self, p):
        """Code distributions of the first half (N, n_ctx, K) -> log code
        probabilities of the patches of the second half (N, n_fut, K) and
        of its coarse view (N, K). Position ``i`` of the last ``n_fut``
        patches of the first half predicts patch ``i`` of the second."""
        fine = self.fine_out(self.fine_body(self.fine_in(p) + self.fine_pos))
        memory = self.coarse_body(self.coarse_in(p) + self.coarse_pos)
        query = self.coarse_query.expand(p.size(0), 1, -1)
        coarse = self.coarse_attn(query, memory, memory,
                                  need_weights=False)[0][:, 0]
        return (F.log_softmax(fine[:, self.n_ctx - self.n_fut:], dim=-1),
                F.log_softmax(self.coarse_out(coarse), dim=-1))

    @torch.no_grad()
    def targets(self, fut):
        """EMA targets: log code probabilities (N, n_fut, K) and expected
        prototypes (N, n_fut, D) of the second half, and the log code
        probabilities of its coarse view (N, K)."""
        logp, z = self.target_codebook(self.target_encoder(fut))
        coarse = self.target_codebook(
            self.target_encoder(self.coarse_view(fut)))[0][:, 0]
        return logp, z, coarse

    def training_terms(self, x):
        """The eight terms of the paper's objective, each summed over the
        patches (and embedding dimensions or patch values) and averaged
        over the sequences."""
        ctx, fut, raw, mean, std = self.views(x)
        h = self.encoder(ctx)
        logp, z = self.codebook(h)
        p = logp.exp()
        fine, coarse = self.predict(p)
        t_logp, t_z, t_coarse = self.targets(fut)
        marginal = p.mean(dim=(0, 1))
        return dict(
            kl_fine=_kl(t_logp, fine).sum(1).mean(),
            mse_fine=(t_z - fine.exp() @ self.codebook.codes).pow(2).sum(
                (1, 2)).mean(),
            kl_coarse=_kl(t_coarse, coarse).mean(),
            emb=(z.detach() - h).pow(2).sum((1, 2)).mean(),
            com=(z - h.detach()).pow(2).sum((1, 2)).mean(),
            ent_sample=-(p * logp).sum(-1).mean(),
            ent_batch=-(marginal * (marginal + 1e-12).log()).sum(),
            rec=(self.decoder(z) * std + mean - raw).pow(2).sum(
                (1, 2)).mean())

    def _score(self, x):
        B = x.size(0)
        ctx, fut, _, _, _ = self.views(x)
        fine, coarse = self.predict(self.codebook(self.encoder(ctx))[0].exp())
        t_logp, t_z, t_coarse = self.targets(fut)
        err = (_kl(t_logp, fine) + self.mse_weight
               * (t_z - fine.exp() @ self.codebook.codes).pow(2).sum(-1)
               + self.coarse_weight * _kl(t_coarse, coarse).unsqueeze(1))
        codes = torch.cat([self.target_codebook(self.target_encoder(ctx))[0],
                           t_logp], dim=1).exp()
        return (err.view(B, -1, self.n_fut).max(1).values,
                codes.view(B, -1, self.n_tokens, codes.size(-1)).max(1).values)

    @torch.no_grad()
    def multi_mask_errors(self, x, ctx_idx, tgt_idx):
        """Error of the patches of the second half, (B, 1, n_fut), and the
        code distributions of all patches, (B, T, K). The split is fixed;
        the arguments are unused. Windows are processed a few at a time so
        that at most ``max_sequences`` sequences are in one pass."""
        step = max(1, self.max_sequences // int(self.keep.sum()))
        out = [self._score(x[i:i + step]) for i in range(0, x.size(0), step)]
        return (torch.cat([o[0] for o in out]).unsqueeze(1),
                torch.cat([o[1] for o in out]))

    @torch.no_grad()
    def token_embeddings(self, x):
        return self.multi_mask_errors(x, None, None)[1]
