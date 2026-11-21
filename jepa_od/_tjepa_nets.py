# -*- coding: utf-8 -*-
"""Networks and mask sampling for the T-JEPA member of the JEPA-OD family.

T-JEPA (Thimonier et al., ICLR 2025, arXiv:2410.05016) is a JEPA for
tabular *representation learning*. This module re-implements its training
recipe from the paper and from the defaults of the released code
(https://github.com/jose-melo/t-jepa, MIT licence) so that it can be
scored as an outlier detector next to :class:`jepa_od._nets.JEPACore`::

    context tokens + [REG] --encoder--> h_ctx --predictor(+mask tokens)--> pred
    all tokens     + [REG] --EMA target encoder--> h_all --gather--> h_tgt

What differs from :class:`JEPACore`:

* the target encoder is an EMA copy of the *whole* encoder, tokenizer
  included, and there is no SIGReg term;
* ``n_reg_tokens`` learned [REG] tokens are appended to the input of both
  encoders, are never masked and are dropped from the output (the paper
  reports that training collapses without them);
* the context is a *small* subset of the features and ``n_pred``
  target subsets are drawn among the remaining ones;
* the encoder is a post-norm ReLU Transformer followed by
  ``LayerNorm -> Linear -> LayerNorm``, so its output is normalised;
* positions are fixed sinusoidal tables.
"""
# License: BSD 2 clause

import copy
import math

import torch
import torch.nn as nn

from ._nets import Tokenizer, _gather_tokens


def sincos_positions(n_positions, dim, base=10000.0):
    """Fixed sinusoidal position table of shape (n_positions, dim)."""
    half = (dim + 1) // 2
    pos = torch.arange(n_positions, dtype=torch.float32).unsqueeze(1)
    freq = torch.exp(torch.arange(half, dtype=torch.float32)
                     * (-math.log(base) / half))
    ang = pos * freq
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=1)[:, :dim]


def _init_weights(m):
    """Truncated-normal initialisation used by T-JEPA (``trunc_normal``)."""
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


class NormHeadStack(nn.Module):
    """T-JEPA's ``TabularEncoder`` block.

    ``output_norm=True`` is the published form: a post-norm ReLU
    Transformer, then ``LayerNorm -> Linear -> LayerNorm``. With
    ``output_norm=False`` the blocks are pre-norm and both LayerNorms of
    the head are removed, so the output keeps the magnitude of the input
    (the JEPA-OD finding of REPORT.md section 4.6 applied to T-JEPA).
    """

    def __init__(self, dim, n_heads, n_layers, dim_feedforward, dropout=0.0,
                 output_norm=True):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=n_heads, dim_feedforward=dim_feedforward,
            dropout=dropout, activation='relu', batch_first=True,
            norm_first=not output_norm)
        self.layers = nn.TransformerEncoder(layer, num_layers=n_layers,
                                            enable_nested_tensor=False)
        self.drop = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(dim) if output_norm else nn.Identity()
        self.fc = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim) if output_norm else nn.Identity()

    def forward(self, x):
        x = self.norm1(self.drop(self.layers(x)))
        return self.norm2(self.drop(self.fc(x)))


class TJEPAEncoder(nn.Module):
    """Tokenizer, [REG] tokens and Transformer of T-JEPA.

    Parameters
    ----------
    n_tokens, in_dim, d_model, n_heads, n_layers, dim_feedforward, dropout
        See :class:`TJEPACore`.
    n_reg_tokens : int
        Number of learned [REG] tokens (0 disables them).
    output_norm : bool
        See :class:`NormHeadStack`.
    """

    def __init__(self, n_tokens, in_dim, d_model, n_heads, n_layers,
                 dim_feedforward, dropout, n_reg_tokens, output_norm):
        super().__init__()
        self.tokenizer = Tokenizer(n_tokens, in_dim, d_model)
        self.register_buffer('pos', sincos_positions(n_tokens, d_model, 100.))
        self.reg = (nn.Parameter(torch.randn(n_reg_tokens, d_model) * 0.02)
                    if n_reg_tokens > 0 else None)
        self.body = NormHeadStack(d_model, n_heads, n_layers, dim_feedforward,
                                  dropout, output_norm)

    def forward(self, x, idx=None):
        """Encode the tokens ``idx`` (all tokens when None) of ``x``.

        x : (B, T, in_dim); idx : LongTensor (B, n) or None.
        Returns (B, n, d_model): the [REG] outputs are dropped.
        """
        tok = self.tokenizer(x) + self.pos
        if idx is not None:
            tok = _gather_tokens(tok, idx)
        n = tok.size(1)
        if self.reg is not None:
            reg = self.reg.unsqueeze(0).expand(tok.size(0), -1, -1)
            tok = torch.cat([tok, reg], dim=1)
        return self.body(tok)[:, :n]


# ----------------------------------------------------------------------
# masking
# ----------------------------------------------------------------------
def tjepa_size_ranges(n_tokens, min_ctx_share, max_ctx_share,
                      min_tgt_share, max_tgt_share):
    """Token-count ranges ``(c_lo, c_hi), (t_lo, t_hi)`` for the shares.

    Counts are ``round(share * n_tokens)`` as in T-JEPA's ``MaskCollator``,
    clamped so that at least one context and one target token exist and
    ``c_lo + t_lo <= n_tokens``.
    """
    T = n_tokens
    c_lo = min(max(1, round(T * min_ctx_share)), T - 1)
    t_lo = min(max(1, round(T * min_tgt_share)), T - c_lo)
    c_hi = min(max(c_lo, round(T * max_ctx_share)), T - t_lo)
    t_hi = min(max(t_lo, round(T * max_tgt_share)), T - c_lo)
    return (c_lo, c_hi), (t_lo, t_hi)


def sample_tjepa_sizes(n_tokens, ctx_range, tgt_range, generator=None):
    """Sample one (n_ctx, n_tgt) pair, shared by a batch.

    ``n = lo + round((hi - lo) * u)`` with ``u ~ U(0, 1)``, resampled
    until ``n_ctx + n_tgt <= n_tokens``.
    """
    (c_lo, c_hi), (t_lo, t_hi) = ctx_range, tgt_range
    for _ in range(100):
        u = torch.rand(2, generator=generator)
        n_ctx = c_lo + int(round((c_hi - c_lo) * u[0].item()))
        n_tgt = t_lo + int(round((t_hi - t_lo) * u[1].item()))
        if n_ctx + n_tgt <= n_tokens:
            return n_ctx, n_tgt
    return c_lo, t_lo


def sample_tjepa_masks(batch_size, n_tokens, n_ctx, n_tgt, n_pred,
                       generator=None):
    """Sample T-JEPA masks for training, one independent draw per row.

    The context is a random subset of ``n_ctx`` tokens. Each of the
    ``n_pred`` target masks is a random subset of ``n_tgt`` tokens among
    the *other* tokens: targets never overlap the context, but may overlap
    each other.

    Returns
    -------
    ctx_idx : LongTensor (batch_size, n_ctx)
    tgt_idx : LongTensor (batch_size, n_pred, n_tgt)
    """
    B, T = batch_size, n_tokens
    perm = torch.rand(B, T, generator=generator).argsort(dim=1)
    ctx = perm[:, :n_ctx].sort(dim=1).values
    rest = perm[:, n_ctx:].unsqueeze(1).expand(-1, n_pred, -1)
    pick = torch.rand(B, n_pred, T - n_ctx,
                      generator=generator).argsort(dim=2)[:, :, :n_tgt]
    tgt = torch.gather(rest, 2, pick).sort(dim=2).values
    return ctx, tgt


def eval_tjepa_masks(n_tokens, n_ctx, n_tgt, n_samples=8, generator=None):
    """Mask set shared by all rows at inference.

    Targets walk through chained random permutations in chunks of
    ``n_tgt`` tokens, so every token is a target about equally often; the
    context of each mask is a random subset of the remaining tokens.

    Returns
    -------
    ctx_idx : LongTensor (n_samples, n_ctx)
    tgt_idx : LongTensor (n_samples, n_tgt)
    """
    T = n_tokens
    ctx_rows, tgt_rows = [], []
    while len(tgt_rows) < n_samples:
        perm = torch.randperm(T, generator=generator)
        for s in range(0, T - n_tgt + 1, n_tgt):
            tgt = perm[s:s + n_tgt]
            is_tgt = torch.zeros(T, dtype=torch.bool)
            is_tgt[tgt] = True
            rest = torch.arange(T)[~is_tgt]
            keep = torch.randperm(len(rest), generator=generator)[:n_ctx]
            ctx_rows.append(rest[keep].sort().values)
            tgt_rows.append(tgt.sort().values)
            if len(tgt_rows) == n_samples:
                break
    return torch.stack(ctx_rows, 0), torch.stack(tgt_rows, 0)


# ----------------------------------------------------------------------
# T-JEPA
# ----------------------------------------------------------------------
class TJEPACore(nn.Module):
    """T-JEPA over a token sequence (one token per feature or group).

    Parameters
    ----------
    n_tokens : int
    in_dim : int
        Raw values per token.
    d_model : int, optional (default=64)
    n_heads : int, optional (default=8)
    n_layers : int, optional (default=4)
    dim_feedforward : int, optional (default=256)
    dropout : float, optional (default=0.1)
    pred_dim : int, optional (default=64)
        Width of the predictor.
    pred_heads : int, optional (default=4)
    pred_layers : int, optional (default=2)
    pred_feedforward : int, optional (default=256)
    n_reg_tokens : int, optional (default=1)
    encoder_norm : bool, optional (default=True)
        Normalised encoder output, as published. See
        :class:`NormHeadStack` for what False changes.
    """

    def __init__(self, n_tokens, in_dim, d_model=64, n_heads=8, n_layers=4,
                 dim_feedforward=256, dropout=0.1, pred_dim=64, pred_heads=4,
                 pred_layers=2, pred_feedforward=256, n_reg_tokens=1,
                 encoder_norm=True):
        super().__init__()
        self.n_tokens = n_tokens
        self.d_model = d_model
        self.encoder = TJEPAEncoder(n_tokens, in_dim, d_model, n_heads,
                                    n_layers, dim_feedforward, dropout,
                                    n_reg_tokens, encoder_norm)
        self.pred_in = nn.Linear(d_model, pred_dim)
        self.register_buffer('pred_pos', sincos_positions(n_tokens, pred_dim))
        self.mask_token = nn.Parameter(torch.zeros(pred_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.predictor = NormHeadStack(pred_dim, pred_heads, pred_layers,
                                       pred_feedforward, dropout)
        self.pred_norm = nn.LayerNorm(pred_dim)
        self.pred_out = nn.Linear(pred_dim, d_model)
        self.apply(_init_weights)

        # EMA copy of the whole encoder (tokenizer and [REG] included)
        self.target_encoder = copy.deepcopy(self.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def token_embeddings(self, x):
        """Target-encoder embeddings of all tokens, (B, T, D)."""
        return self.target_encoder(x)

    def predict(self, h_ctx, ctx_idx, tgt_idx):
        """Predict target-token representations from encoded context."""
        c = self.pred_in(h_ctx) + self.pred_pos[ctx_idx]
        q = self.mask_token + self.pred_pos[tgt_idx]
        out = self.pred_norm(self.predictor(torch.cat([c, q], dim=1)))
        return self.pred_out(out[:, c.size(1):])

    def forward(self, x, ctx_idx, tgt_idx):
        """T-JEPA training pass.

        x : (B, T, in_dim); ctx_idx : (B, c); tgt_idx : (B, P, m).

        Returns
        -------
        pred : (B * P, m, D) predicted target embeddings
        h_tgt : (B * P, m, D) EMA target embeddings (no gradient)
        h_all : (B, T, D) EMA embeddings of all tokens
        """
        B, P, m = tgt_idx.shape
        h_all = self.token_embeddings(x)
        h_ctx = self.encoder(x, ctx_idx).repeat_interleave(P, dim=0)
        ctx = ctx_idx.repeat_interleave(P, dim=0)
        tgt = tgt_idx.reshape(B * P, m)
        pred = self.predict(h_ctx, ctx, tgt)
        h_tgt = _gather_tokens(h_all.repeat_interleave(P, dim=0), tgt)
        return pred, h_tgt, h_all

    @torch.no_grad()
    def multi_mask_errors(self, x, ctx_idx, tgt_idx):
        """Latent prediction error of every row under a shared mask set.

        x : (B, T, in_dim); ctx_idx : (K, c); tgt_idx : (K, m).

        Returns
        -------
        err : (B, K, m) per-target-token squared error (mean over dims)
        h_all : (B, T, D) target embeddings (mask independent)
        """
        B = x.size(0)
        K, m = tgt_idx.shape
        h_all = self.token_embeddings(x)
        ctx = ctx_idx.to(x.device).repeat(B, 1)   # row b -> masks 0..K-1
        tgt = tgt_idx.to(x.device).repeat(B, 1)
        h_ctx = self.encoder(x.repeat_interleave(K, dim=0), ctx)
        pred = self.predict(h_ctx, ctx, tgt)
        h_tgt = _gather_tokens(h_all.repeat_interleave(K, dim=0), tgt)
        err = (pred - h_tgt).pow(2).mean(-1)  # (B*K, m)
        return err.view(B, K, m), h_all
