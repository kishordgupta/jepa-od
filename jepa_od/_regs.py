# -*- coding: utf-8 -*-
"""Anti-collapse regularisers of the JEPA versions that do not use SIGReg.

All functions take embeddings ``z`` of shape ``(N, D)`` or ``(G, N, D)``;
with a leading group dimension (one group per token position) the term
is computed per group and averaged, as :func:`jepa_od._sigreg.sigreg_loss`
does. They are written from the formulas of the papers named in each
docstring, not from released code.
"""
# License: BSD 2 clause

import math

import torch
import torch.nn.functional as F


def _groups(z):
    return z.unsqueeze(0) if z.dim() == 2 else z


def _covariance(z):
    """(G, N, D) -> (G, D, D) unbiased covariance over the N samples."""
    zc = z - z.mean(dim=1, keepdim=True)
    return zc.transpose(1, 2) @ zc / max(1, z.size(1) - 1)


def variance_loss(z, gamma=1.0, eps=1e-4):
    """VICReg variance term: hinge on the per-dimension standard deviation.

    ``mean_j max(0, gamma - sqrt(Var(z_j) + eps))`` (Bardes, Ponce and
    LeCun, 2022).
    """
    z = _groups(z)
    std = torch.sqrt(z.var(dim=1) + eps)
    return F.relu(gamma - std).mean()


def covariance_loss(z, pairwise=False):
    """VICReg covariance term: squared off-diagonal covariances.

    Divided by ``D`` as in VICReg, C-JEPA and CF-JEPA, or by
    ``D * (D - 1)`` when ``pairwise`` is True, as in EB-JEPA.
    """
    z = _groups(z)
    D = z.size(-1)
    cov = _covariance(z)
    off = cov.pow(2).sum(dim=(1, 2)) - cov.diagonal(dim1=1, dim2=2).pow(2).sum(1)
    return (off / (D * max(1, D - 1) if pairwise else D)).mean()


def cw_mmd(z, sigma=1.0, gamma=None):
    """Cramer-Wold MMD between the embeddings and ``N(0, sigma^2 I)``.

    The closed form of KerJEPA (Zimmermann et al., arXiv:2512.19605,
    Theorem 4): the Gaussian-kernel MMD of every 1-D projection, averaged
    over all directions, with the average over directions replaced by its
    large-dimension inverse-multiquadric approximation. No random slices
    are drawn. The bandwidth follows Silverman's rule for the group size
    when ``gamma`` is None. The estimate of the first term leaves out the
    diagonal, so the value is about zero for Gaussian embeddings and can be
    slightly negative.
    """
    z = _groups(z)
    G, N, D = z.shape
    if N < 2:
        return z.sum() * 0.0
    if gamma is None:
        h = (4.0 / (3.0 * N)) ** 0.2 * sigma
        gamma = 1.0 / (2.0 * h * h)
    a = 4.0 * gamma / max(1, 2 * D - 3)
    s = 1.0 + 2.0 * gamma * sigma ** 2
    sq = z.pow(2).sum(-1)                                   # (G, N)
    d2 = (sq.unsqueeze(2) + sq.unsqueeze(1)
          - 2.0 * z @ z.transpose(1, 2)).clamp_min(0.0)     # (G, N, N)
    k = (1.0 + a * d2).rsqrt()
    within = (k.sum(dim=(1, 2)) - k.diagonal(dim1=1, dim2=2).sum(1)) \
        / (N * (N - 1))
    cross = (1.0 + (a / s) * sq).rsqrt().mean(1) / math.sqrt(s)
    const = 1.0 / math.sqrt(1.0 + 4.0 * gamma * sigma ** 2)
    return (within - 2.0 * cross + const).mean()


def rectified_gn_sample(shape, p=1.0, mean_shift=0.0, generator=None):
    """Samples of the rectified generalised Gaussian target of Rectified
    LpJEPA: ``relu(mean_shift + sigma_p * e)`` with ``e`` of unit variance,
    Laplace for ``p = 1`` and Gaussian for ``p = 2`` (drawn on CPU)."""
    if p == 2.0:
        e = torch.randn(shape, generator=generator)
    elif p == 1.0:
        u = torch.rand(shape, generator=generator).clamp(1e-7, 1 - 1e-7) - 0.5
        e = -torch.sign(u) * torch.log1p(-2.0 * u.abs()) / math.sqrt(2.0)
    else:
        raise ValueError("p must be 1.0 or 2.0")
    return F.relu(mean_shift + e)


def rdm_reg(z, p=1.0, mean_shift=0.0, num_slices=128, generator=None):
    """RDMReg of Rectified LpJEPA (Kuang et al., arXiv:2602.01456).

    Sliced two-sample match between the (rectified) embeddings and samples
    of a rectified generalised Gaussian: both are projected on random unit
    directions, each projection is sorted over the samples, and the squared
    differences are averaged.
    """
    z = _groups(z)
    G, N, D = z.shape
    A = torch.randn(D, num_slices, generator=generator)
    A = (A / A.norm(dim=0, keepdim=True).clamp_min(1e-12)).to(z)
    y = rectified_gn_sample((G, N, D), p, mean_shift, generator).to(z)
    zs = (z @ A).sort(dim=1).values
    ys = (y @ A).sort(dim=1).values
    return (zs - ys).pow(2).mean()


def sac_reg(z, eps=1e-4, generator=None):
    """SACReg of lambda-JEPA (Demirel et al., arXiv:2609.35288).

    ``0.5 * (|mu|^2 + tr(S) - logdet(S + eps I) - d) / d``: the divergence
    between the Gaussian fitted to the embeddings and ``N(0, I)``, per
    dimension. ``z`` is ``(N, D)``. With fewer than ``2 D`` samples the
    embeddings are first projected on ``N // 2`` random orthonormal
    directions so that the covariance has full rank.
    """
    N, D = z.shape
    if N < 2:
        return z.sum() * 0.0
    if N < 2 * D:
        k = max(1, N // 2)
        q, _ = torch.linalg.qr(torch.randn(D, k, generator=generator))
        z = z @ q.to(z)
        D = k
    mu = z.mean(0)
    cov = _covariance(z.unsqueeze(0))[0]
    cov = cov + eps * torch.eye(D, device=z.device, dtype=z.dtype)
    logdet = torch.linalg.eigvalsh(cov).clamp_min(1e-12).log().sum()
    return 0.5 * (mu.pow(2).sum() + cov.diagonal().sum() - logdet - D) / D
