# -*- coding: utf-8 -*-
"""Unit tests for the regularisers of the further JEPA versions."""
import math
import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jepa_od._regs import (variance_loss, covariance_loss, cw_mmd,
                           rectified_gn_sample, rdm_reg, sac_reg)


class TestRegularisers(unittest.TestCase):
    def setUp(self):
        self.gen = torch.Generator().manual_seed(0)
        self.z = torch.randn(4, 2000, 16, generator=self.gen)

    def test_variance_loss(self):
        self.assertLess(variance_loss(self.z).item(), 0.03)
        const = torch.ones(50, 8)
        self.assertAlmostEqual(variance_loss(const).item(), 1 - 1e-2, places=5)
        half = 0.5 * self.z
        self.assertAlmostEqual(variance_loss(half).item(), 0.5, delta=0.03)
        # a 2-D input is one group
        self.assertAlmostEqual(variance_loss(self.z[0]).item(),
                               variance_loss(self.z[:1]).item(), places=6)

    def test_covariance_loss(self):
        self.assertLess(covariance_loss(self.z).item(), 0.02)
        # one direction repeated in every dimension: all covariances are 1
        dup = self.z[..., :1].expand(-1, -1, 16)
        D = 16
        self.assertAlmostEqual(covariance_loss(dup).item(), D - 1, delta=0.5)
        self.assertAlmostEqual(covariance_loss(dup, pairwise=True).item(),
                               covariance_loss(dup).item() / (D - 1),
                               places=5)

    def test_cw_mmd_is_zero_for_the_prior_only(self):
        z = torch.randn(2, 1500, 32, generator=self.gen)
        self.assertLess(abs(cw_mmd(z).item()), 5e-3)
        self.assertGreater(cw_mmd(z + 1.5).item(), 0.05)
        self.assertGreater(cw_mmd(0.1 * z).item(), 0.05)
        self.assertGreater(cw_mmd(3.0 * z).item(), 0.05)
        self.assertLess(abs(cw_mmd(2.0 * z, sigma=2.0).item()), 5e-3)

    def test_cw_mmd_matches_sliced_gaussian_mmd(self):
        # brute force: exact 1-D Gaussian-kernel MMD to N(0, 1), averaged
        # over many random directions
        N, D, gamma = 300, 48, 0.7
        z = 0.6 * torch.randn(N, D, generator=self.gen) + 0.3
        A = torch.randn(D, 4000, generator=self.gen)
        A = A / A.norm(dim=0, keepdim=True)
        p = z @ A                                              # (N, M)
        diff = p.unsqueeze(0) - p.unsqueeze(1)                 # (N, N, M)
        k = torch.exp(-gamma * diff.pow(2))
        within = (k.sum((0, 1)) - N) / (N * (N - 1))
        s = 1 + 2 * gamma
        cross = torch.exp(-gamma * p.pow(2) / s).mean(0) / math.sqrt(s)
        ref = (within - 2 * cross + 1 / math.sqrt(1 + 4 * gamma)).mean()
        self.assertAlmostEqual(cw_mmd(z, gamma=gamma).item(), ref.item(),
                               delta=0.01)

    def test_rectified_target(self):
        y = rectified_gn_sample((200000,), 1.0, 0.0, self.gen)
        self.assertGreaterEqual(y.min().item(), 0.0)
        self.assertAlmostEqual((y == 0).float().mean().item(), 0.5, delta=0.01)
        # E relu(Laplace(0, b)) = b / 2 with b = 1 / sqrt(2)
        self.assertAlmostEqual(y.mean().item(), 0.5 / math.sqrt(2), delta=0.01)
        g = rectified_gn_sample((200000,), 2.0, 0.0, self.gen)
        self.assertAlmostEqual(g.mean().item(), 1 / math.sqrt(2 * math.pi),
                               delta=0.01)
        shifted = rectified_gn_sample((50000,), 1.0, 5.0, self.gen)
        self.assertAlmostEqual(shifted.var().item(), 1.0, delta=0.05)
        with self.assertRaises(ValueError):
            rectified_gn_sample((3,), 1.5)

    def test_rdm_reg(self):
        match = rectified_gn_sample((3, 400, 16), 1.0, 0.0, self.gen)
        g = lambda: torch.Generator().manual_seed(5)
        a = rdm_reg(match, generator=g()).item()
        self.assertLess(a, rdm_reg(torch.zeros(3, 400, 16),
                                   generator=g()).item() / 5)
        self.assertLess(a, rdm_reg(match + 1.0, generator=g()).item() / 5)
        self.assertEqual(a, rdm_reg(match, generator=g()).item())

    def test_sac_reg(self):
        z = torch.randn(6000, 16, generator=self.gen)
        self.assertLess(sac_reg(z).item(), 0.01)
        for a in (0.3, 2.0):
            expected = 0.5 * (a * a - math.log(a * a) - 1)
            self.assertAlmostEqual(sac_reg(a * z).item(), expected, delta=0.02)
        self.assertAlmostEqual(sac_reg(z + 1.0).item(), 0.5, delta=0.03)
        # fewer samples than twice the dimension: random projection first
        small = sac_reg(torch.randn(20, 16, generator=self.gen),
                        generator=self.gen)
        self.assertTrue(torch.isfinite(small))

    def test_gradients(self):
        w = torch.randn(3, 64, 16, generator=self.gen, requires_grad=True)
        loss = (variance_loss(w) + covariance_loss(w) + cw_mmd(w)
                + rdm_reg(torch.relu(w), generator=self.gen)
                + sac_reg(w.reshape(-1, 16)))
        loss.backward()
        self.assertTrue(torch.isfinite(w.grad).all())
        self.assertGreater(w.grad.abs().sum().item(), 0)


if __name__ == '__main__':
    unittest.main()
