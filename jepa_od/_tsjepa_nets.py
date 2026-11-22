# -*- coding: utf-8 -*-
"""Network of the ``'tsjepa'`` version of :class:`jepa_od.TSJEPAOD`.

TS-JEPA ("Joint Embeddings Go Temporal", Ennadir, Golkar and Sarra,
arXiv:2509.25449) is a JEPA for time-series *representation learning*.
This module re-implements its network from the paper and from the
defaults of the released code (https://github.com/Sennadir/TS_JEPA) so
that it can be scored as an anomaly detector next to
:class:`jepa_od._nets.JEPACore`::

    context patches --encoder--> h_ctx --predictor(+mask tokens)--> pred
    all patches --EMA target encoder--> LayerNorm --> h_all --gather--> h_tgt

What differs from :class:`JEPACore`:

* patches are embedded by a 1-D convolution over the time axis of the
  patch (kernel = stride) followed by a linear layer, shared by all
  positions;
* positions are fixed sinusoidal tables, in the encoder and the predictor;
* the encoder ends in a LayerNorm, its feed-forward layers are four times
  as wide as the model, and the predictor is as wide as the encoder;
* the target encoder is an EMA copy of the *whole* encoder, tokenizer
  included, and there is no SIGReg term.

The reference tokenizer is univariate. Here the convolution takes the
channels of a multivariate patch as its input channels, which is the same
operation when there is one channel.
"""
# License: BSD 2 clause

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._nets import TransformerStack, _gather_tokens
from ._tjepa_nets import sincos_positions, _init_weights


class ConvPatchTokenizer(nn.Module):
    """TS-JEPA's patch embedding: ``Conv1d`` then ``Linear``.

    Parameters
    ----------
    patch_len : int
        Timestamps per patch.
    n_channels : int
        Channels of the series (input channels of the convolution).
    out_dim : int
    kernel_size : int, optional (default=3)
        Kernel and stride of the convolution, capped at ``patch_len``.
    """

    def __init__(self, patch_len, n_channels, out_dim, kernel_size=3):
        super().__init__()
        self.patch_len = patch_len
        self.n_channels = n_channels
        k = max(1, min(kernel_size, patch_len))
        self.conv = nn.Conv1d(n_channels, out_dim, kernel_size=k, stride=k)
        self.fc = nn.Linear(out_dim * ((patch_len - k) // k + 1), out_dim)

    def forward(self, x):
        # x: (B, T, patch_len * n_channels), time-major inside a patch
        B, T, _ = x.shape
        x = x.reshape(B * T, self.patch_len, self.n_channels).transpose(1, 2)
        return self.fc(self.conv(x).flatten(1)).view(B, T, -1)


class TSJEPAEncoder(nn.Module):
    """Tokenizer, positions and Transformer of TS-JEPA."""

    def __init__(self, n_tokens, patch_len, n_channels, d_model, n_heads,
                 n_layers, dropout, kernel_size, final_norm):
        super().__init__()
        self.tokenizer = ConvPatchTokenizer(patch_len, n_channels, d_model,
                                            kernel_size)
        self.register_buffer('pos', sincos_positions(n_tokens, d_model))
        self.body = TransformerStack(d_model, n_heads, n_layers, dropout,
                                     ff_mult=4, final_norm=final_norm)

    def forward(self, x, idx=None):
        """Encode the patches ``idx`` (all patches when None) of ``x``.

        x : (B, T, patch_len * n_channels); idx : LongTensor (B, n) or None.
        """
        tok = self.tokenizer(x) + self.pos
        if idx is not None:
            tok = _gather_tokens(tok, idx)
        return self.body(tok)


class TSJEPACore(nn.Module):
    """TS-JEPA over the patches of a window.

    Parameters
    ----------
    n_tokens : int
        Patches per window.
    patch_len : int
    n_channels : int
    d_model : int, optional (default=64)
    n_heads : int, optional (default=4)
    n_layers : int, optional (default=2)
    pred_layers : int, optional (default=1)
    dropout : float, optional (default=0.0)
    kernel_size : int, optional (default=3)
        Kernel of the patch convolution (the default of the released code).
    encoder_norm : bool, optional (default=True)
        Final LayerNorm on the encoder, as published. The targets are
        layer-normalised in both cases.
    """

    def __init__(self, n_tokens, patch_len, n_channels, d_model=64,
                 n_heads=4, n_layers=2, pred_layers=1, dropout=0.0,
                 kernel_size=3, encoder_norm=True):
        super().__init__()
        self.n_tokens = n_tokens
        self.d_model = d_model
        self.encoder = TSJEPAEncoder(n_tokens, patch_len, n_channels,
                                     d_model, n_heads, n_layers, dropout,
                                     kernel_size, encoder_norm)
        self.pred_in = nn.Linear(d_model, d_model)
        self.register_buffer('pred_pos', sincos_positions(n_tokens, d_model))
        self.mask_token = nn.Parameter(torch.zeros(d_model))
        self.predictor = TransformerStack(d_model, n_heads, pred_layers,
                                          dropout, ff_mult=4, final_norm=True)
        self.pred_out = nn.Linear(d_model, d_model)
        self.apply(_init_weights)

        # EMA copy of the whole encoder (tokenizer included)
        self.target_encoder = copy.deepcopy(self.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def token_embeddings(self, x):
        """Layer-normalised target embeddings of all patches, (B, T, D)."""
        h = self.target_encoder(x)
        return F.layer_norm(h, (h.size(-1),))

    def predict(self, h_ctx, ctx_idx, tgt_idx):
        """Predict target-patch representations from encoded context."""
        c = self.pred_in(h_ctx) + self.pred_pos[ctx_idx]
        q = self.mask_token + self.pred_pos[tgt_idx]
        out = self.predictor(torch.cat([c, q], dim=1))
        return self.pred_out(out[:, c.size(1):])

    def forward(self, x, ctx_idx, tgt_idx):
        """TS-JEPA training pass.

        x : (B, T, patch_len * n_channels); ctx_idx : (B, c); tgt_idx : (B, m).

        Returns
        -------
        pred : (B, m, D) predicted target embeddings
        h_tgt : (B, m, D) EMA target embeddings (no gradient)
        h_all : (B, T, D) EMA embeddings of all patches
        """
        h_all = self.token_embeddings(x)
        pred = self.predict(self.encoder(x, ctx_idx), ctx_idx, tgt_idx)
        return pred, _gather_tokens(h_all, tgt_idx), h_all

    @torch.no_grad()
    def multi_mask_errors(self, x, ctx_idx, tgt_idx):
        """Latent prediction error of every window under a shared mask set.

        x : (B, T, patch_len * n_channels); ctx_idx : (K, c); tgt_idx : (K, m).

        Returns
        -------
        err : (B, K, m) per-target-patch squared error (mean over dims)
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
