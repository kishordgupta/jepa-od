# -*- coding: utf-8 -*-
"""Training recipes of the further JEPA versions of
:class:`jepa_od.TSJEPAOD`.

A recipe names the network of a version, its loss and how its target is
obtained. The masking, the model sizes, the optimiser, the budget and the
anomaly scores stay the detector's own, as for the first four versions
(``'lejepa'``, ``'ijepa'``, ``'tsjepa'``, ``'tjepa'``), which live in
``ts_jepa.py``.

``params`` holds the constants of a version: the published value where the
paper gives one, otherwise a value fixed here before any benchmark was run
(marked "ours" below). ``TSJEPAOD(variant_params={...})`` overrides them.
"""
# License: BSD 2 clause

import math

import torch
import torch.nn.functional as F

from ._nets import JEPACore, sample_masks, _gather_tokens
from ._regs import (variance_loss, covariance_loss, cw_mmd, rdm_reg,
                    sac_reg)
from ._sigreg import sigreg_loss
from ._variant_nets import (CJEPACore, StochasticJEPACore, DenseJEPACore,
                            SALTCore, KerJEPACore, RectifiedJEPACore,
                            NextLatentCore, CFJEPACore)


class Recipe(object):
    """Base class of a JEPA version.

    Attributes
    ----------
    encoder_norm : bool
        What ``encoder_norm=None`` means for this version.
    ema_momentum : float or None
        What ``ema_momentum=None`` means; None for versions without an
        EMA target.
    ema_schedule : {None, 'linear', 'cosine', 'fixed'}
        How the momentum goes from its start value to ``ema_end`` during
        training.
    ema_end : float
        The momentum at the end of training (1 for all versions but
        ``'ststjepa'``).
    masked : bool
        False for versions that do not hide patches (``mask_mode`` and
        ``mask_ratio`` are then ignored).
    params : dict
        Constants of the version.
    """
    encoder_norm = False
    ema_momentum = None
    ema_schedule = None
    ema_end = 1.0
    masked = True
    params = {}

    def build(self, det, n_tokens, n_channels, size, hp):
        raise NotImplementedError

    def loss(self, det, model, xb, ctx, tgt, gen, state, hp):
        """Returns ``(loss, latent prediction loss)`` of one batch.

        ``state`` has ``frac`` (share of the training steps done) and
        ``epoch`` / ``n_epochs``.
        """
        raise NotImplementedError

    def train_epochs(self, det, hp):
        """Epochs of the main training loop."""
        return det.epochs

    def pretrain(self, det, model, make_loader, device, gen, hp):
        """Anything that has to happen before the main loop."""

    def ema_pairs(self, model):
        return [(model.target_encoder, model.encoder)]


def momentum_at(start, frac, schedule, end=1.0):
    """Target momentum after a share ``frac`` of the training steps."""
    if schedule == 'fixed':
        return start
    if schedule == 'cosine':
        return end - (end - start) * 0.5 * (1.0 + math.cos(math.pi * frac))
    return start + (end - start) * frac


def _token_dim(det, n_channels):
    return det.patch_len * n_channels


# ----------------------------------------------------------------------
class VJEPA(Recipe):
    """V-JEPA (Bardes et al., arXiv:2404.08471): EMA target with momentum
    0.998 -> 1 and an L1 loss. Network and target normalisation are those
    of ``'ijepa'``."""
    ema_momentum, ema_schedule = 0.998, 'linear'

    def build(self, det, n_tokens, n_channels, size, hp):
        return JEPACore(n_tokens, _token_dim(det, n_channels),
                        target_mode='ema', **size)

    def loss(self, det, model, xb, ctx, tgt, gen, state, hp):
        pred, h_tgt, _ = model(xb, ctx, tgt)
        loss = F.l1_loss(pred, h_tgt.detach())
        return loss, loss


class VJEPA21(Recipe):
    """V-JEPA 2.1 (arXiv:2603.14482): L1 loss on the hidden tokens plus a
    loss on the visible tokens weighted by ``context_weight / sqrt(distance
    to the nearest hidden token)``, both at several encoder depths; fixed
    EMA momentum 0.99925. The context weight is ramped from 0 over the
    first ``warmup`` share of training (the paper ramps it during its
    schedule without giving the shape; ours)."""
    ema_momentum, ema_schedule = 0.99925, 'fixed'
    params = dict(context_weight=0.5, warmup=0.5, max_levels=4)

    def build(self, det, n_tokens, n_channels, size, hp):
        return DenseJEPACore(n_tokens, _token_dim(det, n_channels),
                             max_levels=hp['max_levels'], **size)

    def loss(self, det, model, xb, ctx, tgt, gen, state, hp):
        preds, targets = model(xb, ctx, tgt)
        c = ctx.size(1)
        ramp = 1.0 if hp['warmup'] <= 0 else min(1.0,
                                                 state['frac'] / hp['warmup'])
        dist = (ctx.unsqueeze(2) - tgt.unsqueeze(1)).abs().min(dim=2).values
        w = (hp['context_weight'] * ramp
             / dist.clamp_min(1).to(xb.dtype).sqrt()).unsqueeze(-1)  # (B,c,1)
        l_pred = l_ctx = 0.0
        for pred, h_all in zip(preds, targets):
            l_pred = l_pred + F.l1_loss(pred[:, c:],
                                        _gather_tokens(h_all, tgt))
            l_ctx = l_ctx + (w * (pred[:, :c]
                                  - _gather_tokens(h_all, ctx)).abs()).mean()
        l_pred, l_ctx = l_pred / len(preds), l_ctx / len(preds)
        return l_pred + l_ctx, l_pred


class CJEPA(Recipe):
    """C-JEPA (Mo and Tong, arXiv:2410.19560): I-JEPA plus VICReg on the
    projected mean context embeddings of two mask draws of the same window
    (the paper: of the mask blocks of the same image), at overall weight
    0.001 with VICReg's own weights 25 / 25 / 1."""
    ema_momentum, ema_schedule = 0.996, 'linear'
    params = dict(vicreg_weight=0.001, sim_weight=25.0, var_weight=25.0,
                  cov_weight=1.0)

    def build(self, det, n_tokens, n_channels, size, hp):
        return CJEPACore(n_tokens, _token_dim(det, n_channels), **size)

    def loss(self, det, model, xb, ctx, tgt, gen, state, hp):
        ctx2, tgt2 = sample_masks(xb.size(0), det.n_tokens_, det.n_mask_,
                                  det.mask_mode, gen)
        ctx2, tgt2 = ctx2.to(xb.device), tgt2.to(xb.device)
        p1, t1, _, z1 = model.forward_views(xb, ctx, tgt)
        p2, t2, _, z2 = model.forward_views(xb, ctx2, tgt2)
        l_pred = 0.5 * (F.mse_loss(p1, t1.detach())
                        + F.mse_loss(p2, t2.detach()))
        reg = (hp['sim_weight'] * F.mse_loss(z1, z2)
               + hp['var_weight'] * 0.5 * (variance_loss(z1)
                                           + variance_loss(z2))
               + hp['cov_weight'] * (covariance_loss(z1)
                                     + covariance_loss(z2)))
        return l_pred + hp['vicreg_weight'] * reg, l_pred


class SJEPA(Recipe):
    """Stochastic-latent JEPA (Giakoumoglou et al., arXiv:2611.09302):
    I-JEPA with a Gaussian latent for the predictor and a KL term. The
    paper's objective is the summed squared error plus ``beta * KL``;
    ``kl_weight`` is that ``beta`` (1, ours: the text read gives no
    default). ``latent_dim`` None is ``d_model // 4`` (ours)."""
    ema_momentum, ema_schedule = 0.996, 'linear'
    params = dict(kl_weight=1.0, latent_dim=None)

    def build(self, det, n_tokens, n_channels, size, hp):
        return StochasticJEPACore(n_tokens, _token_dim(det, n_channels),
                                  latent_dim=hp['latent_dim'], **size)

    def loss(self, det, model, xb, ctx, tgt, gen, state, hp):
        noise = torch.randn(xb.size(0), model.latent_dim,
                            generator=gen).to(xb)
        pred, h_tgt, _, kl = model.forward_latent(xb, ctx, tgt, noise)
        l_pred = F.mse_loss(pred, h_tgt.detach())
        # the squared error is averaged over m * D elements here
        scale = pred.size(1) * pred.size(2)
        return l_pred + hp['kl_weight'] * kl / scale, l_pred


class SALT(Recipe):
    """SALT (Li et al., arXiv:2509.24317): a teacher trained by masked
    reconstruction, then frozen; the student predicts its latents with an
    L1 loss. No EMA and no regulariser. The teacher gets ``teacher_share``
    of the epochs (1/6, the 1 : 5 split the paper found best), at least
    one."""
    params = dict(teacher_share=1.0 / 6.0)

    def build(self, det, n_tokens, n_channels, size, hp):
        return SALTCore(n_tokens, _token_dim(det, n_channels), **size)

    @staticmethod
    def _teacher_epochs(det, hp):
        if det.epochs <= 0:
            return 0
        return max(1, int(round(det.epochs * hp['teacher_share'])))

    def train_epochs(self, det, hp):
        if det.epochs <= 0:
            return 0
        return max(1, det.epochs - self._teacher_epochs(det, hp))

    def pretrain(self, det, model, make_loader, device, gen, hp):
        n_epochs = self._teacher_epochs(det, hp)
        if n_epochs > 0:
            loader = make_loader()
            opt = torch.optim.AdamW(model.teacher_parameters(), lr=det.lr,
                                    weight_decay=det.weight_decay)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=n_epochs, eta_min=det.lr * 0.05)
            model.train()
            for epoch in range(n_epochs):
                run = []
                for (xb,) in loader:
                    xb = xb.to(device)
                    ctx, tgt = sample_masks(xb.size(0), det.n_tokens_,
                                            det.n_mask_, det.mask_mode, gen)
                    ctx, tgt = ctx.to(device), tgt.to(device)
                    opt.zero_grad()
                    loss = F.mse_loss(model.reconstruct(xb, ctx, tgt),
                                      _gather_tokens(xb, tgt))
                    loss.backward()
                    if det.grad_clip:
                        torch.nn.utils.clip_grad_norm_(
                            model.teacher_parameters(), det.grad_clip)
                    opt.step()
                    run.append(loss.item())
                sched.step()
                if det.verbose:
                    print('Teacher epoch %d/%d, reconstruction loss=%.4f'
                          % (epoch + 1, n_epochs,
                             sum(run) / max(1, len(run))))
        model.freeze_teacher()

    def loss(self, det, model, xb, ctx, tgt, gen, state, hp):
        pred, h_tgt, _ = model(xb, ctx, tgt)
        loss = F.l1_loss(pred, h_tgt.detach())
        return loss, loss


class _Regularised(Recipe):
    """Shared-encoder versions: squared error against stop-gradient
    targets, as ``'lejepa'``, plus the version's own regulariser."""
    core = JEPACore

    def build(self, det, n_tokens, n_channels, size, hp):
        if self.core is JEPACore:
            return JEPACore(n_tokens, _token_dim(det, n_channels),
                            target_mode='sigreg', **size)
        return self.core(n_tokens, _token_dim(det, n_channels), **size)

    def regulariser(self, det, h_all, gen, hp):
        raise NotImplementedError

    def loss(self, det, model, xb, ctx, tgt, gen, state, hp):
        pred, h_tgt, h_all = model(xb, ctx, tgt)
        l_pred = F.mse_loss(pred, h_tgt.detach())
        return l_pred + self.regulariser(det, h_all, gen, hp), l_pred


class VICRegJEPA(_Regularised):
    """JEPA regularised by the variance and covariance terms of VICReg,
    with the image defaults of EB-JEPA (Terver et al., arXiv:2602.03604):
    weights 1 and 80. Computed per patch position over the batch."""
    params = dict(var_weight=1.0, cov_weight=80.0)

    def regulariser(self, det, h_all, gen, hp):
        z = h_all.transpose(0, 1)
        return (hp['var_weight'] * variance_loss(z)
                + hp['cov_weight'] * covariance_loss(z, pairwise=True))


class KerJEPA(_Regularised):
    """KerJEPA (Zimmermann et al., arXiv:2512.19605): the closed-form
    Cramer-Wold MMD to a Gaussian prior at weight 7.5, with standardised
    prediction targets. Computed per patch position over the batch."""
    core = KerJEPACore
    params = dict(reg_weight=7.5, prior_scale=1.0)

    def regulariser(self, det, h_all, gen, hp):
        return hp['reg_weight'] * cw_mmd(h_all.transpose(0, 1),
                                         sigma=hp['prior_scale'])


class LpJEPA(_Regularised):
    """Rectified LpJEPA (Kuang et al., arXiv:2602.01456): non-negative
    embeddings matched to a rectified Laplace (``p=1``) by RDMReg. The
    weight is 5, the ratio of the released code's 125 to its 25 on the
    invariance term; the projections are ``sigreg_slices`` many."""
    core = RectifiedJEPACore
    params = dict(reg_weight=5.0, p=1.0, mean_shift=0.0)

    def regulariser(self, det, h_all, gen, hp):
        return hp['reg_weight'] * rdm_reg(
            h_all.transpose(0, 1), p=hp['p'], mean_shift=hp['mean_shift'],
            num_slices=det.sigreg_slices, generator=gen)


class LambdaJEPA(_Regularised):
    """lambda-JEPA (Demirel et al., arXiv:2609.35288): SACReg on the
    encoder output, over all patch embeddings of the batch. The paper
    calibrates its weights per model; the weight here is 1 on the
    per-dimension value (ours)."""
    params = dict(reg_weight=1.0, eps=1e-4)

    def regulariser(self, det, h_all, gen, hp):
        return hp['reg_weight'] * sac_reg(
            h_all.reshape(-1, h_all.size(-1)), eps=hp['eps'], generator=gen)


class _NextLatent(Recipe):
    masked = False
    style = None

    def build(self, det, n_tokens, n_channels, size, hp):
        return NextLatentCore(
            n_tokens, _token_dim(det, n_channels), d_model=size['d_model'],
            n_heads=size['n_heads'], n_layers=size['n_layers'],
            dropout=size['dropout'], encoder_norm=size['encoder_norm'],
            style=self.style)

    def ema_pairs(self, model):
        return []


class LeNEPA(_NextLatent):
    """LeNEPA (Chemeris, Jin and Balestriero, arXiv:2607.00958): causal
    next-latent prediction with the summed squared error in the projector's
    space, plus SIGReg computed for every window over its own patches, on
    the projected input and output of the Transformer, at weight 20. No
    EMA, no stop-gradient, no masks."""
    style = 'lenepa'
    params = dict(sigreg_weight=20.0)

    def loss(self, det, model, xb, ctx, tgt, gen, state, hp):
        pred, target, reg, _ = model(xb)
        l_pred = (pred - target).pow(2).sum(-1).mean()
        l_reg = sum(sigreg_loss(u, num_slices=det.sigreg_slices,
                                generator=gen) for u in reg) / len(reg)
        return l_pred + hp['sigreg_weight'] * l_reg, l_pred


class LeWM(_NextLatent):
    """LeWorldModel (Maes et al., arXiv:2603.19312) without its action
    input: next-embedding squared error plus SIGReg over the batch at every
    patch position, at weight 0.1. No EMA, no stop-gradient, no masks."""
    style = 'lewm'
    params = dict(sigreg_weight=0.1)

    def loss(self, det, model, xb, ctx, tgt, gen, state, hp):
        pred, target, reg, _ = model(xb)
        l_pred = F.mse_loss(pred, target)
        l_reg = sigreg_loss(reg[0].transpose(0, 1),
                            num_slices=det.sigreg_slices, generator=gen)
        return l_pred + hp['sigreg_weight'] * l_reg, l_pred


class CFJEPA(Recipe):
    """CF-JEPA (Lee and Sim, arXiv:2606.07031): crops as context, three
    forward zones, linear predictors, L1 on unit-normalised embeddings
    with a weight that falls to zero over the epochs, VICReg variance and
    covariance on the mean crop embeddings, a multi-scale invariance term
    across the crops of a window, and an EMA target with a cosine
    momentum schedule from 0.983. All constants are the published ones."""
    ema_momentum, ema_schedule = 0.983, 'cosine'
    masked = False
    params = dict(n_crops=4, var_weight=0.081, cov_weight=0.076,
                  inv_weight=1.101, n_blocks=5, crop_ratio=None)

    def build(self, det, n_tokens, n_channels, size, hp):
        return CFJEPACore(n_tokens, det.patch_len, n_channels,
                          d_model=size['d_model'], n_blocks=hp['n_blocks'],
                          n_crops=hp['n_crops'], crop_ratio=hp['crop_ratio'])

    def loss(self, det, model, xb, ctx, tgt, gen, state, hp):
        horizon, crops = model.training_terms(xb, gen)
        pooled = torch.cat([z.mean(1) for z in crops], dim=0)
        inv = 0.0
        for p in (2, 4, 8):
            views = torch.stack([F.adaptive_avg_pool1d(z.transpose(1, 2), p)
                                 for z in crops], dim=0)
            inv = inv + views.var(dim=0, unbiased=False).mean() / 3.0
        w = 1.0 - (state['epoch'] + 1) / max(1, state['n_epochs'])
        loss = (w * horizon + hp['var_weight'] * variance_loss(pooled)
                + hp['cov_weight'] * covariance_loss(pooled)
                + hp['inv_weight'] * inv)
        return loss, horizon


RECIPES = {
    'vjepa': VJEPA(), 'vjepa21': VJEPA21(), 'cjepa': CJEPA(),
    'sjepa': SJEPA(), 'salt': SALT(), 'vicreg': VICRegJEPA(),
    'kerjepa': KerJEPA(), 'lpjepa': LpJEPA(), 'lambdajepa': LambdaJEPA(),
    'lenepa': LeNEPA(), 'lewm': LeWM(), 'cfjepa': CFJEPA(),
}

# the ten versions of version 0.5.0 (the module needs the classes above)
from ._recipes2 import MORE_RECIPES, SCJEPA  # noqa: E402

RECIPES.update(MORE_RECIPES)
RECIPES['scjepa'] = SCJEPA()   # added in version 0.5.1
