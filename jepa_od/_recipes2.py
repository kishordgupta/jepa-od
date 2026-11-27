# -*- coding: utf-8 -*-
"""Training recipes of the ten JEPA versions added to
:class:`jepa_od.TSJEPAOD` in version 0.5.0.

The rules are those of ``_recipes.py``: a recipe takes its network, loss
and target from the paper; masking, sizes, optimiser, budget and anomaly
scores stay the detector's own; ``params`` holds the published constants,
and a value the paper does not give is fixed here and marked "ours".
"""
# License: BSD 2 clause

import math

import numpy as np
import torch
import torch.nn.functional as F

from ._nets import _gather_tokens
from ._recipes import Recipe, _NextLatent, _token_dim
from ._sigreg import sigreg_loss
from ._variant_nets2 import (ShallowTargetCore, HEPACore, SubspaceLatentCore,
                             MotionCore, NoiseJEPACore, GMMJEPACore,
                             VariationalJEPACore, PhiJEPACore,
                             PlainTargetCore, SCJEPACore, cosine_distance)


def _sizes(size, *names):
    return {k: size[k] for k in names}


_CAUSAL = ('d_model', 'n_heads', 'n_layers', 'dropout', 'encoder_norm')


# ----------------------------------------------------------------------
class STSTJEPA(Recipe):
    """STST-JEPA (arXiv:2607.06629): the hidden patches are predicted in
    the output space of an EMA copy of the tokenizer (squared error, weight
    1), and a small head reconstructs their values from the predictions
    (smooth L1 with ``beta = 1``, weight 0.35). The momentum goes from
    0.9996 to 0.9999; the paper reaches 0.9999 after 12,000 steps and holds
    it, here it rises over the whole (much shorter) training."""
    ema_momentum, ema_schedule, ema_end = 0.9996, 'linear', 0.9999
    params = dict(rec_weight=0.35)

    def build(self, det, n_tokens, n_channels, size, hp):
        return ShallowTargetCore(n_tokens, _token_dim(det, n_channels),
                                 **size)

    def ema_pairs(self, model):
        return [(model.target_tokenizer, model.tokenizer)]

    def loss(self, det, model, xb, ctx, tgt, gen, state, hp):
        pred, target, rec = model(xb, ctx, tgt)
        l_pred = F.mse_loss(pred, target)
        l_rec = F.smooth_l1_loss(rec, _gather_tokens(xb, tgt), beta=1.0)
        return l_pred + hp['rec_weight'] * l_rec, l_pred


class HEPA(Recipe):
    """HEPA (arXiv:2605.11130): a causal encoder, a predictor conditioned
    on the horizon, and a weight-shared bidirectional target; the loss is
    ``(1 - a) * L1 + a * SIGReg`` on the predictions with ``a = 0.1``. No
    EMA, no stop-gradient, no masks. Every step draws ``n_pairs`` split
    points and horizons for the batch (4, ours), the horizon log-uniformly
    up to the end of the window. The per-context instance normalisation of
    the paper is not applied: the detector standardises the series once."""
    masked = False
    params = dict(sigreg_weight=0.1, n_pairs=4)

    def build(self, det, n_tokens, n_channels, size, hp):
        return HEPACore(n_tokens, _token_dim(det, n_channels),
                        **_sizes(size, *_CAUSAL))

    def ema_pairs(self, model):
        return []

    def loss(self, det, model, xb, ctx, tgt, gen, state, hp):
        T = model.n_tokens
        h = model.summaries(xb)
        preds, targets = [], []
        for _ in range(int(hp['n_pairs'])):
            t = int(torch.randint(1, T, (), generator=gen))
            u = torch.rand((), generator=gen).item()
            dt = int(min(T - t, max(1, round(math.exp(u * math.log(T - t))))))
            preds.append(model.predict(h[:, t - 1], dt))
            targets.append(model.interval(xb[:, t:t + dt]))
        pred = torch.stack(preds)                          # (n_pairs, B, D)
        l_pred = F.l1_loss(pred, torch.stack(targets))
        reg = sigreg_loss(pred, num_slices=det.sigreg_slices, generator=gen)
        a = hp['sigreg_weight']
        return (1.0 - a) * l_pred + a * reg, l_pred


class SubJEPA(_NextLatent):
    """Sub-JEPA (arXiv:2605.09241): LeWorldModel with its Gaussian
    regulariser applied in frozen random subspaces instead of the whole
    embedding space. In every subspace the Epps-Pulley statistic is taken
    over the embeddings of all patches of the batch, as in the paper;
    the weight is that of ``'lewm'``, whose protocol the paper follows."""
    params = dict(sigreg_weight=0.1, subspace_dim=6)

    def build(self, det, n_tokens, n_channels, size, hp):
        return SubspaceLatentCore(
            n_tokens, _token_dim(det, n_channels),
            subspace_dim=hp['subspace_dim'], seed=det.random_state,
            **_sizes(size, *_CAUSAL))

    def loss(self, det, model, xb, ctx, tgt, gen, state, hp):
        pred, target, reg, _ = model(xb)
        l_pred = F.mse_loss(pred, target)
        z = reg[0].reshape(-1, reg[0].size(-1))
        l_reg = sigreg_loss(model.project(z), num_slices=det.sigreg_slices,
                            generator=gen)
        return l_pred + hp['sigreg_weight'] * l_reg, l_pred


class Straightening(_NextLatent):
    """Temporal straightening (arXiv:2603.12231): next-embedding prediction
    with a stop-gradient on the target as the only collapse prevention,
    plus a curvature term ``1 - cos(v_t, v_{t+1})`` on the differences
    ``v_t = z_{t+1} - z_t`` of consecutive patch embeddings, at weight
    0.1. No EMA, no masks. The network is that of ``'lewm'``."""
    style = 'lewm'
    params = dict(curv_weight=0.1)

    def loss(self, det, model, xb, ctx, tgt, gen, state, hp):
        pred, target, reg, _ = model(xb)
        l_pred = F.mse_loss(pred, target.detach())
        v = reg[0][:, 1:] - reg[0][:, :-1]
        if v.size(1) < 2:      # two patches: no curvature
            return l_pred, l_pred
        curv = 1.0 - F.cosine_similarity(v[:, :-1], v[:, 1:], dim=-1).mean()
        return l_pred + hp['curv_weight'] * curv, l_pred


class MotionJEPA(_NextLatent):
    """MotionJEPA (arXiv:2609.23881): next-embedding prediction plus
    DISReg, ``0.25 SIGReg(z) + 2 SIGReg(d) + 0.5 MSE(d, d_pred)``, where
    ``d`` embeds the difference of two consecutive patches and ``d_pred``
    is predicted from their two embeddings. No EMA, no stop-gradient, no
    masks. SIGReg is taken per patch position over the batch, as for
    ``'lewm'``."""
    params = dict(z_weight=0.25, d_weight=2.0, diff_weight=0.5)

    def build(self, det, n_tokens, n_channels, size, hp):
        return MotionCore(n_tokens, _token_dim(det, n_channels),
                          **_sizes(size, *_CAUSAL))

    def loss(self, det, model, xb, ctx, tgt, gen, state, hp):
        pred, target, reg, _ = model(xb)
        z = reg[0]
        d, d_pred = model.difference_terms(xb, z)
        l_pred = F.mse_loss(pred, target)

        def sig(u):
            return sigreg_loss(u.transpose(0, 1),
                               num_slices=det.sigreg_slices, generator=gen)

        return (l_pred + hp['z_weight'] * sig(z) + hp['d_weight'] * sig(d)
                + hp['diff_weight'] * F.mse_loss(d_pred, d)), l_pred


# ----------------------------------------------------------------------
class NJEPA(Recipe):
    """N-JEPA (arXiv:2507.15216): I-JEPA with two predictors. The second
    one gets Gaussian noise on its mask tokens, with a standard deviation
    drawn per window as in EDM (``ln sigma ~ N(-1.2, 1.2^2)``; the paper
    draws one level per mask block and has four blocks per image). Both
    predictions are trained towards the EMA targets with a smooth L1 loss
    and towards each other with a squared error; the three terms are
    summed (the text read gives no weights)."""
    ema_momentum, ema_schedule = 0.996, 'linear'
    params = dict(p_mean=-1.2, p_std=1.2, denoise_weight=1.0)

    def build(self, det, n_tokens, n_channels, size, hp):
        return NoiseJEPACore(n_tokens, _token_dim(det, n_channels), **size)

    def loss(self, det, model, xb, ctx, tgt, gen, state, hp):
        tokens = model.embed(xb)
        h_ctx = model.encoder(_gather_tokens(tokens, ctx))
        h_tgt = _gather_tokens(model.encode_target(tokens), tgt)
        B, m = tgt.shape
        sigma = torch.exp(hp['p_mean'] + hp['p_std']
                          * torch.randn(B, 1, 1, generator=gen))
        noise = (sigma * torch.randn(B, m, model.mask_token2.numel(),
                                     generator=gen)).to(xb)
        clean = model.predict(h_ctx, ctx, tgt)
        noisy = model.predict_noisy(h_ctx, ctx, tgt, noise)
        l_pred = F.smooth_l1_loss(clean, h_tgt)
        return (l_pred + F.smooth_l1_loss(noisy, h_tgt)
                + hp['denoise_weight'] * F.mse_loss(noisy, clean)), l_pred


class GMMJEPA(Recipe):
    """GMM-Anchored JEPA (arXiv:2602.09040): a diagonal Gaussian mixture
    is fitted once on the input features, and the KL divergence from its
    frozen soft posteriors to a cluster head is added at the hidden
    positions with a weight that falls linearly from 1 to 0.01. The paper
    fits 1024 components on log-mel frames of speech; here 32 components
    (ours) are fitted on at most ``max_fit`` patches of the training
    windows. Its noise and mixing augmentations of speech are not used.
    The EMA momentum is that of ``'ijepa'`` (ours)."""
    ema_momentum, ema_schedule = 0.996, 'linear'
    params = dict(n_components=32, weight_start=1.0, weight_end=0.01,
                  max_fit=10000)

    def build(self, det, n_tokens, n_channels, size, hp):
        return GMMJEPACore(n_tokens, _token_dim(det, n_channels),
                           n_components=hp['n_components'], **size)

    def pretrain(self, det, model, make_loader, device, gen, hp):
        if det.epochs <= 0:      # the cluster head is used in training only
            return
        from sklearn.mixture import GaussianMixture
        patches = make_loader().dataset.tensors[0]
        flat = patches.reshape(-1, patches.size(-1))
        if len(flat) > hp['max_fit']:
            keep = torch.randperm(len(flat), generator=gen)[:int(hp['max_fit'])]
            flat = flat[keep.sort().values]
        k = int(min(hp['n_components'], max(1, len(flat) // 2)))
        gmm = GaussianMixture(k, covariance_type='diag', reg_covar=1e-3,
                              max_iter=50, random_state=det.random_state)
        gmm.fit(flat.numpy().astype(np.float64))
        model.set_gmm(gmm.weights_, gmm.means_, gmm.covariances_)

    def loss(self, det, model, xb, ctx, tgt, gen, state, hp):
        pred, h_tgt, logits = model.forward_cluster(xb, ctx, tgt)
        l_pred = F.mse_loss(pred, h_tgt)
        q = model.posteriors(_gather_tokens(xb, tgt))
        l_cluster = F.kl_div(F.log_softmax(logits, dim=-1), q,
                             reduction='sum') / q[..., 0].numel()
        w = hp['weight_start'] + (hp['weight_end'] - hp['weight_start']) \
            * state['frac']
        return l_pred + w * l_cluster, l_pred


class VariationalJEPA(Recipe):
    """Variational JEPA (VJEPA, arXiv:2601.14354): the predictor outputs a
    diagonal Gaussian, trained by the negative log-likelihood of samples
    of the target distribution, plus ``beta`` times the KL divergence of
    the target distribution to a standard normal. The paper leaves the
    target distribution, ``beta`` and the networks open. Here (all ours)
    the target distribution is a Gaussian of fixed standard deviation
    ``target_std`` around the EMA target; with a stop-gradient on the EMA
    branch the KL term can only act on the online encoder, where it is
    applied to the embeddings of all patches; ``kl_weight`` is ``beta``."""
    ema_momentum, ema_schedule = 0.996, 'linear'
    params = dict(kl_weight=0.01, target_std=0.1)

    def build(self, det, n_tokens, n_channels, size, hp):
        return VariationalJEPACore(n_tokens, _token_dim(det, n_channels),
                                   **size)

    def loss(self, det, model, xb, ctx, tgt, gen, state, hp):
        mean, logvar, h_tgt, h_on = model.forward_dist(xb, ctx, tgt)
        s = float(hp['target_std'])
        sample = h_tgt + s * torch.randn(h_tgt.shape, generator=gen).to(xb)
        nll = model.nll(sample, mean, logvar).mean()
        kl = 0.5 * (h_on.pow(2).mean() + s * s - math.log(s * s) - 1.0)
        return nll + hp['kl_weight'] * kl, F.mse_loss(mean, h_tgt)


class PhiJEPA(Recipe):
    """PhiJEPA (arXiv:2611.27516): two prediction stages with the cosine
    distance ``2 - 2 cos``. The student sees the visible patches of a
    corrupted view; the predictor is trained towards the EMA encoder's
    tokens of a second, independently corrupted view, and a token-wise MLP
    on top of it towards the tokens of the clean view, at weight 0.5.
    Every window is corrupted (``p = 1``). The corruptions of a series and
    their strengths are ours (:meth:`PhiJEPACore.corrupt`), and so is the
    EMA momentum, that of ``'ijepa'``."""
    ema_momentum, ema_schedule = 0.996, 'linear'
    params = dict(clean_weight=0.5, p_corrupt=1.0, gain_std=0.1,
                  noise_std=0.1)

    def build(self, det, n_tokens, n_channels, size, hp):
        return PhiJEPACore(n_tokens, _token_dim(det, n_channels),
                           n_channels=n_channels, **size)

    def loss(self, det, model, xb, ctx, tgt, gen, state, hp):
        kw = dict(p=hp['p_corrupt'], gain_std=hp['gain_std'],
                  noise_std=hp['noise_std'])
        x1 = model.corrupt(xb, gen, **kw)
        x2 = model.corrupt(xb, gen, **kw)
        h_ctx = model.encoder(_gather_tokens(model.embed(x1), ctx))
        first = model.predict(h_ctx, ctx, tgt)
        y_view = _gather_tokens(model.encode_target(model.embed(x2)), tgt)
        y_clean = _gather_tokens(model.encode_target(model.embed(xb)), tgt)
        l_pred = cosine_distance(first, y_view).mean()
        l_clean = cosine_distance(model.head(first), y_clean).mean()
        return l_pred + hp['clean_weight'] * l_clean, l_pred


class HARJEPA(Recipe):
    """HAR-JEPA (arXiv:2607.16350): an EMA target and a four-term loss on
    the predictions ``Z``: smooth L1 to the targets (weight 1), the VICReg
    variance hinge with threshold 0.5 (weight 5), the squared off-diagonal
    correlations (weight 0.2), and a norm term ``(||z|| - tau)^2`` with
    ``tau = 1 + s (sqrt(d) - 1)`` (weight 0.1; ``s = 0.3``, the value at
    which the paper's Transformer encoder did best). The targets are not
    normalised. The paper's variance and covariance terms on the targets
    have no gradient under an EMA target and are left out; the momentum
    starts at 0.996 (the paper's value begins with 0.99 in the text
    read)."""
    ema_momentum, ema_schedule = 0.996, 'linear'
    params = dict(sim_weight=1.0, var_weight=5.0, cov_weight=0.2,
                  norm_weight=0.1, gamma=0.5, dispersion=0.3)

    def build(self, det, n_tokens, n_channels, size, hp):
        return PlainTargetCore(n_tokens, _token_dim(det, n_channels), **size)

    def loss(self, det, model, xb, ctx, tgt, gen, state, hp):
        pred, h_tgt, _ = model(xb, ctx, tgt)
        l_pred = F.smooth_l1_loss(pred, h_tgt)
        z = pred.reshape(-1, pred.size(-1))
        N, D = z.shape
        std = torch.sqrt(z.var(dim=0) + 1e-4)
        l_var = F.relu(hp['gamma'] - std).mean()
        zn = (z - z.mean(dim=0)) / (std + 1e-4)
        corr = zn.T @ zn / max(1, N - 1)
        l_cov = (corr.pow(2).sum() - corr.diagonal().pow(2).sum()) / D
        tau = 1.0 + hp['dispersion'] * (math.sqrt(D) - 1.0)
        l_norm = (z.norm(dim=-1) - tau).pow(2).mean()
        return (hp['sim_weight'] * l_pred + hp['var_weight'] * l_var
                + hp['cov_weight'] * l_cov + hp['norm_weight'] * l_norm), \
            l_pred


class SCJEPA(Recipe):
    """SC-JEPA (He et al., arXiv:2602.04643; first version: MTS-JEPA), the
    JEPA its authors built for anomalies in multivariate series. The first
    half of the window predicts the second half, one variable at a time,
    in the space of a soft codebook and at two resolutions; EMA copies of
    the encoder and the codebook give the targets (decay 0.996, constant);
    a decoder reconstructs the first half. The objective and all its
    constants are the paper's (its Appendix B.1): 128 codes, temperature
    0.1, weights 1 (fine), 0.5 (coarse), 0.1 (squared error of the
    expected prototype), 1 and 0.25 (codebook alignment), 0.005 and 0.01
    (the two entropy terms), and a reconstruction weight that falls
    linearly from 0.5 to 0.1.

    Ours: the two halves of the detector's window stand for the paper's
    two consecutive windows (five patches each at the defaults, as in the
    paper); the tokenizer is an MLP per patch (the paper: a residual CNN);
    positions are sinusoidal; the decoder is linear; the instance
    normalisation has no learned affine map; and the scores
    (:class:`SCJEPACore`), since the paper trains a classifier on labels
    instead."""
    ema_momentum, ema_schedule = 0.996, 'fixed'
    masked = False
    params = dict(n_codes=128, temperature=0.1, fine_weight=1.0,
                  coarse_weight=0.5, mse_weight=0.1, emb_weight=1.0,
                  com_weight=0.25, ent_sample_weight=0.005,
                  ent_batch_weight=0.01, rec_weight_start=0.5,
                  rec_weight_end=0.1)

    def build(self, det, n_tokens, n_channels, size, hp):
        return SCJEPACore(
            n_tokens, det.patch_len, n_channels, n_codes=int(hp['n_codes']),
            temperature=hp['temperature'], mse_weight=hp['mse_weight'],
            coarse_weight=hp['coarse_weight'] / hp['fine_weight'], **size)

    def ema_pairs(self, model):
        return [(model.target_encoder, model.encoder),
                (model.target_codebook, model.codebook)]

    def pretrain(self, det, model, make_loader, device, gen, hp):
        # the paper removes the variables with zero variance in training
        patches = make_loader().dataset.tensors[0]
        model.set_channels(patches.reshape(-1, model.n_channels).std(0) > 1e-6)

    def loss(self, det, model, xb, ctx, tgt, gen, state, hp):
        t = model.training_terms(xb)
        rec_weight = hp['rec_weight_start'] + (
            hp['rec_weight_end'] - hp['rec_weight_start']) * state['frac']
        loss = (hp['fine_weight'] * (t['kl_fine']
                                     + hp['mse_weight'] * t['mse_fine'])
                + hp['coarse_weight'] * t['kl_coarse']
                + hp['emb_weight'] * t['emb'] + hp['com_weight'] * t['com']
                + hp['ent_sample_weight'] * t['ent_sample']
                - hp['ent_batch_weight'] * t['ent_batch']
                + rec_weight * t['rec'])
        return loss, t['kl_fine']


MORE_RECIPES = {
    'ststjepa': STSTJEPA(), 'hepa': HEPA(), 'subjepa': SubJEPA(),
    'straight': Straightening(), 'motionjepa': MotionJEPA(),
    'njepa': NJEPA(), 'gmmjepa': GMMJEPA(), 'variational': VariationalJEPA(),
    'phijepa': PhiJEPA(), 'harjepa': HARJEPA(),
}
