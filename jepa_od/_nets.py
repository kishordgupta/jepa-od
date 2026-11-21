# -*- coding: utf-8 -*-
"""Neural building blocks shared by the tabular and time-series JEPA-OD
detectors.

The data is always presented to the network as a sequence of *tokens*:

* tabular rows  -> one token per feature (or per small group of
  consecutive features when ``n_features > max_tokens``);
* time series   -> one token per patch of ``patch_len`` consecutive
  timestamps (all channels of the patch are flattened into the token).

:class:`JEPACore` implements the Joint-Embedding Predictive Architecture
on top of these tokens::

    context tokens --encoder--> h_ctx --predictor(+mask tokens)--> pred
    all tokens     --target encoder--> h_all --gather--> h_tgt

and the training signal is ``||pred - h_tgt||`` in *latent* space.
"""
# License: BSD 2 clause

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Tokenizer(nn.Module):
    """Per-token affine map ``x[:, t] @ W[t] + b[t]`` (FT-Transformer style).

    Parameters
    ----------
    n_tokens : int
    in_dim : int
        Raw values per token (1 for one-feature-per-token tabular data,
        ``patch_len * n_channels`` for time-series patches).
    out_dim : int
    """

    def __init__(self, n_tokens, in_dim, out_dim):
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(n_tokens, in_dim, out_dim) / math.sqrt(in_dim))
        self.bias = nn.Parameter(torch.zeros(n_tokens, out_dim))

    def forward(self, x):
        # x: (B, T, in_dim) -> (B, T, out_dim)
        return torch.einsum('bti,tio->bto', x, self.weight) + self.bias


class TransformerStack(nn.Module):
    """Pre-norm Transformer encoder with a final LayerNorm."""

    def __init__(self, d_model, n_heads, n_layers, dropout=0.0, ff_mult=2,
                 final_norm=True):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=ff_mult * d_model, dropout=dropout,
            activation='gelu', batch_first=True, norm_first=True)
        self.layers = nn.TransformerEncoder(layer, num_layers=n_layers,
                                            enable_nested_tensor=False)
        # The encoder deliberately has *no* final LayerNorm: the residual
        # stream keeps the magnitude of extreme inputs, so an out-of-range
        # token maps to an out-of-range embedding instead of saturating to
        # a direction that is trivially predictable. SIGReg bounds the
        # embeddings of normal data instead.
        self.norm = nn.LayerNorm(d_model) if final_norm else nn.Identity()

    def forward(self, x):
        return self.norm(self.layers(x))


# ----------------------------------------------------------------------
# masking
# ----------------------------------------------------------------------
_MASK_MODES = ('random', 'block', 'forecast', 'mixed')


def _complement(tgt, n_tokens):
    """Context indices (sorted) complementary to ``tgt`` (rows of size m)."""
    B, m = tgt.shape
    is_tgt = torch.zeros(B, n_tokens, dtype=torch.bool)
    is_tgt.scatter_(1, tgt, True)
    ctx = torch.arange(n_tokens).unsqueeze(0).expand(B, n_tokens)
    return ctx[~is_tgt].view(B, n_tokens - m)


def sample_masks(batch_size, n_tokens, n_mask, mode='random',
                 generator=None):
    """Sample one (context, target) split per row -- used for training.

    Parameters
    ----------
    batch_size, n_tokens, n_mask : int
        ``1 <= n_mask <= n_tokens - 1``.
    mode : {'random', 'block', 'forecast', 'mixed'}
        * ``'random'``   -- uniformly random token subset (tabular default)
        * ``'block'``    -- one contiguous block at a random position
        * ``'forecast'`` -- always the last ``n_mask`` tokens (causal)
        * ``'mixed'``    -- per-row coin flip between block / forecast
    generator : torch.Generator or None (CPU)

    Returns
    -------
    ctx_idx : LongTensor (batch_size, n_tokens - n_mask)
    tgt_idx : LongTensor (batch_size, n_mask)
    """
    B, T, m = batch_size, n_tokens, n_mask
    if mode == 'random':
        tgt = torch.rand(B, T, generator=generator).argsort(dim=1)[:, :m]
        tgt, _ = tgt.sort(dim=1)
    elif mode in ('block', 'forecast', 'mixed'):
        start = torch.randint(0, T - m + 1, (B,), generator=generator)
        if mode == 'forecast':
            start = torch.full((B,), T - m, dtype=torch.long)
        elif mode == 'mixed':
            flip = torch.rand(B, generator=generator) < 0.5
            start = torch.where(flip, torch.full_like(start, T - m), start)
        tgt = start.unsqueeze(1) + torch.arange(m).unsqueeze(0)
    else:
        raise ValueError("mask_mode must be one of %s; got %r"
                         % (_MASK_MODES, mode))
    return _complement(tgt, T), tgt


def eval_masks(n_tokens, n_mask, mode='random', n_samples=8,
               generator=None):
    """Deterministic-coverage mask set shared by all rows at inference.

    * block-type modes enumerate *all* distinct block positions when
      ``n_samples`` is None or large enough (complete, deterministic
      coverage of every token); otherwise a random subset of positions.
    * ``'forecast'`` is the single last block.
    * ``'random'`` chains random permutations chopped into groups of
      ``n_mask`` tokens, so every token is masked about equally often.

    Returns
    -------
    ctx_idx : LongTensor (K, n_tokens - n_mask)
    tgt_idx : LongTensor (K, n_mask)
    """
    T, m = n_tokens, n_mask
    if mode == 'forecast':
        tgt = torch.arange(T - m, T).unsqueeze(0)
    elif mode in ('block', 'mixed'):
        starts = torch.arange(0, T - m + 1)
        if n_samples is not None and n_samples < len(starts):
            keep = torch.randperm(len(starts), generator=generator)[:n_samples]
            starts = starts[keep.sort().values]
        tgt = starts.unsqueeze(1) + torch.arange(m).unsqueeze(0)
    elif mode == 'random':
        K = 8 if n_samples is None else int(n_samples)
        rows = []
        while len(rows) < K:
            perm = torch.randperm(T, generator=generator)
            for s in range(0, T - m + 1, m):
                rows.append(perm[s:s + m].sort().values)
                if len(rows) == K:
                    break
        tgt = torch.stack(rows, 0)
    else:
        raise ValueError("mask_mode must be one of %s; got %r"
                         % (_MASK_MODES, mode))
    return _complement(tgt, T), tgt


def _gather_tokens(x, idx):
    """Gather tokens ``x[b, idx[b, j]]`` -> (B, len(idx), D)."""
    return torch.gather(x, 1, idx.unsqueeze(-1).expand(-1, -1, x.size(-1)))


# ----------------------------------------------------------------------
# JEPA
# ----------------------------------------------------------------------
class JEPACore(nn.Module):
    """Joint-Embedding Predictive Architecture over a token sequence.

    Parameters
    ----------
    n_tokens : int
    in_dim : int
    d_model : int, optional (default=64)
    n_heads : int, optional (default=4)
    n_layers : int, optional (default=2)
        Depth of the (shared / online) encoder.
    pred_layers : int, optional (default=1)
        Depth of the narrow predictor (``pred_dim = d_model // 2``).
    dropout : float, optional (default=0.0)
    target_mode : {'sigreg', 'ema'}
        ``'sigreg'`` -- a single shared encoder produces both context and
        target embeddings; collapse is prevented by SIGReg in the loss
        (LeJEPA). ``'ema'`` -- I-JEPA style frozen EMA target encoder with
        stop-gradient and LayerNorm'ed targets.
    recon_dim : int or None
        If given, a per-token linear head decodes each *target* embedding
        back to its raw token (an MTS-JEPA style anchor that keeps the
        target embeddings informative about the token's own content).
    encoder_norm : bool, optional (default=False)
        Final LayerNorm on the encoder output. Keep False for outlier
        detection: with it, extreme inputs saturate to a fixed direction
        that is trivially predictable and outliers score as *normal*.
    """

    def __init__(self, n_tokens, in_dim, d_model=64, n_heads=4, n_layers=2,
                 pred_layers=1, dropout=0.0, target_mode='sigreg',
                 recon_dim=None, encoder_norm=False):
        super().__init__()
        if target_mode not in ('sigreg', 'ema'):
            raise ValueError("target_mode must be 'sigreg' or 'ema'")
        self.n_tokens = n_tokens
        self.d_model = d_model
        self.target_mode = target_mode
        pred_dim = max(16, d_model // 2)
        pred_heads = max(1, min(n_heads, pred_dim // 16))

        self.tokenizer = Tokenizer(n_tokens, in_dim, d_model)
        self.pos = nn.Parameter(torch.randn(n_tokens, d_model) * 0.02)
        self.encoder = TransformerStack(d_model, n_heads, n_layers, dropout,
                                        final_norm=encoder_norm)
        if target_mode == 'ema':
            self.target_encoder = copy.deepcopy(self.encoder)
            for p in self.target_encoder.parameters():
                p.requires_grad_(False)
        else:
            self.target_encoder = None

        self.pred_in = nn.Linear(d_model, pred_dim)
        self.mask_token = nn.Parameter(torch.randn(pred_dim) * 0.02)
        self.pred_pos = nn.Parameter(torch.randn(n_tokens, pred_dim) * 0.02)
        self.predictor = TransformerStack(pred_dim, pred_heads, pred_layers,
                                          dropout)
        self.pred_out = nn.Linear(pred_dim, d_model)
        self.recon_head = (Tokenizer(n_tokens, d_model, recon_dim)
                           if recon_dim else None)

    # ------------------------------------------------------------------
    def embed(self, x):
        """Raw tokens (B, T, in_dim) -> positional token embeddings."""
        return self.tokenizer(x) + self.pos

    def encode_target(self, tokens):
        """Target representation of *all* tokens, (B, T, D)."""
        if self.target_mode == 'ema':
            with torch.no_grad():
                h = self.target_encoder(tokens)
                return F.layer_norm(h, (h.size(-1),))
        return self.encoder(tokens)

    @torch.no_grad()
    def token_embeddings(self, x):
        """Target embeddings of all tokens of raw tokens ``x``, (B, T, D)."""
        return self.encode_target(self.embed(x))

    def predict(self, h_ctx, ctx_idx, tgt_idx):
        """Predict target-token representations from encoded context."""
        c = self.pred_in(h_ctx) + self.pred_pos[ctx_idx]
        q = self.mask_token + self.pred_pos[tgt_idx]
        out = self.predictor(torch.cat([c, q], dim=1))
        return self.pred_out(out[:, c.size(1):])

    def forward(self, x, ctx_idx, tgt_idx):
        """Full JEPA training pass.

        Returns
        -------
        pred : (B, m, D) predicted target embeddings
        h_tgt : (B, m, D) actual target embeddings (not detached)
        h_all : (B, T, D) target-encoder embeddings of all tokens
        """
        tokens = self.embed(x)
        h_ctx = self.encoder(_gather_tokens(tokens, ctx_idx))
        h_all = self.encode_target(tokens)
        h_tgt = _gather_tokens(h_all, tgt_idx)
        pred = self.predict(h_ctx, ctx_idx, tgt_idx)
        return pred, h_tgt, h_all

    @torch.no_grad()
    def multi_mask_errors(self, x, ctx_idx, tgt_idx):
        """Latent prediction error of every row under a shared mask set.

        Parameters
        ----------
        x : (B, T, in_dim)
        ctx_idx : (K, T - m), tgt_idx : (K, m)  -- from :func:`eval_masks`

        Returns
        -------
        err : (B, K, m) per-target-token squared error (mean over dims)
        h_all : (B, T, D) target embeddings (mask independent)
        """
        B, T = x.shape[0], x.shape[1]
        K, m = tgt_idx.shape
        tokens = self.embed(x)
        h_all = self.encode_target(tokens)
        ctx = ctx_idx.to(x.device).repeat(B, 1)   # row b -> masks 0..K-1
        tgt = tgt_idx.to(x.device).repeat(B, 1)
        tokens_rep = tokens.repeat_interleave(K, dim=0)
        h_all_rep = h_all.repeat_interleave(K, dim=0)
        h_ctx = self.encoder(_gather_tokens(tokens_rep, ctx))
        pred = self.predict(h_ctx, ctx, tgt)
        h_tgt = _gather_tokens(h_all_rep, tgt)
        err = (pred - h_tgt).pow(2).mean(-1)  # (B*K, m)
        return err.view(B, K, m), h_all
