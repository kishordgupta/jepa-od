# -*- coding: utf-8 -*-
"""Collapse-prevention utilities for JEPA-OD.

Two mechanisms are provided:

* :func:`sigreg_loss` -- Sketched Isotropic Gaussian Regularization
  (SIGReg) from LeJEPA (Balestriero & LeCun, 2025, arXiv:2511.08544).
  Random 1-D projections of the embeddings are pushed towards N(0, 1)
  with the Epps-Pulley characteristic-function statistic. This removes
  the need for an EMA teacher / stop-gradient and, importantly for
  outlier detection, pins the embedding distribution of *normal* data to
  a known isotropic Gaussian so that latent prediction errors and
  embedding-space distances have a fixed, interpretable scale.

* :func:`ema_update` -- the classic I-JEPA / BYOL exponential moving
  average teacher, kept as an alternative (``target_mode='ema'``).
"""
# License: BSD 2 clause

import torch


def sigreg_loss(z, num_slices=128, n_points=17, t_max=5.0, generator=None):
    """SIGReg / Epps-Pulley statistic on random 1-D projections.

    Parameters
    ----------
    z : torch.Tensor of shape (N, K) or (G, N, K)
        Embeddings. With a leading group dimension ``G`` (e.g. one group
        per token position) the statistic is computed per group and
        averaged, so that every token position is individually pushed
        towards an isotropic Gaussian.
    num_slices : int, optional (default=128)
        Number of random unit directions ``a`` (resampled every call).
    n_points : int, optional (default=17)
        Trapezoidal quadrature knots on ``[-t_max, t_max]``.
    t_max : float, optional (default=5.0)
        Integration domain half-width.
    generator : torch.Generator or None
        CPU generator for the projection directions (reproducibility).

    Returns
    -------
    loss : torch.Tensor (scalar)
        ``N * \\int |ecf(t) - exp(-t^2/2)|^2 exp(-t^2/2) dt`` averaged
        over directions (and groups).
    """
    squeeze = False
    if z.dim() == 2:
        z = z.unsqueeze(0)
        squeeze = True
    G, N, K = z.shape
    # directions sampled on CPU for device-independent reproducibility
    A = torch.randn(K, num_slices, generator=generator)
    A = (A / A.norm(dim=0, keepdim=True).clamp_min(1e-12)).to(
        device=z.device, dtype=z.dtype)
    t = torch.linspace(-t_max, t_max, n_points, device=z.device,
                       dtype=z.dtype)
    exp_f = torch.exp(-0.5 * t ** 2)  # CF of N(0,1) == Gaussian window
    proj = z @ A  # (G, N, M)
    x_t = proj.unsqueeze(-1) * t  # (G, N, M, T)
    ecf_re = torch.cos(x_t).mean(1)  # (G, M, T)
    ecf_im = torch.sin(x_t).mean(1)
    err = ((ecf_re - exp_f) ** 2 + ecf_im ** 2) * exp_f
    stat = torch.trapezoid(err, t, dim=-1) * N  # (G, M)
    loss = stat.mean()
    return loss if not squeeze else loss


@torch.no_grad()
def ema_update(target, online, momentum):
    """In-place EMA update ``target <- m * target + (1 - m) * online``."""
    for p_t, p_o in zip(target.parameters(), online.parameters()):
        p_t.mul_(momentum).add_(p_o.detach(), alpha=1.0 - momentum)
    for b_t, b_o in zip(target.buffers(), online.buffers()):
        b_t.copy_(b_o)
