# -*- coding: utf-8 -*-
"""Unit tests for the twelve further JEPA versions of TSJEPAOD."""
import os
import pickle
import sys
import unittest

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.base import clone
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyod.utils.data import generate_ts_data
from jepa_od import TSJEPAOD
from jepa_od._recipes import RECIPES, momentum_at
from jepa_od._variant_nets import (DenseJEPACore, NextLatentCore,
                                   CFJEPACore, _DilatedBlock)
from jepa_od.ts_jepa import _VARIANTS

NEW = tuple(RECIPES)
EMA = ('vjepa', 'vjepa21', 'cjepa', 'sjepa', 'cfjepa')
MASK_FREE = ('lenepa', 'lewm', 'cfjepa')
# (encoder_norm_, ema_momentum_) after fit with both left at None
DEFAULTS = {'vjepa': (False, 0.998), 'vjepa21': (False, 0.99925),
            'cjepa': (False, 0.996), 'sjepa': (False, 0.996),
            'cfjepa': (False, 0.983)}


class TestTSJEPAODMoreVersions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = generate_ts_data(
            n_train=1500, n_test=600, n_channels=3, contamination=0.05,
            anomaly_type='both', random_state=1)
        cls.kw = dict(window_size=40, patch_len=5, epochs=3,
                      score='combined', random_state=3, verbose=0)
        cls.fitted = {v: TSJEPAOD(variant=v, **cls.kw).fit(cls.X_train)
                      for v in NEW}

    def test_registry(self):
        self.assertEqual(_VARIANTS[:4], ('lejepa', 'ijepa', 'tsjepa', 'tjepa'))
        self.assertEqual(_VARIANTS[4:], NEW)
        self.assertEqual(len(set(_VARIANTS)), 16)
        for v, recipe in RECIPES.items():
            self.assertTrue(recipe.__doc__, v)
            self.assertIsInstance(recipe.params, dict)
            self.assertEqual(recipe.masked, v not in MASK_FREE)
            self.assertEqual(recipe.ema_schedule is not None, v in EMA)
        with self.assertRaises(ValueError):
            TSJEPAOD(variant='xjepa', **self.kw).fit(self.X_train)

    def test_scores(self):
        for v, clf in self.fitted.items():
            self.assertEqual(clf.decision_scores_.shape, (len(self.X_train),))
            self.assertTrue(np.isfinite(clf.decision_scores_).all(), v)
            for score in ('pred', 'embed', 'combined'):
                clf.score = score
                s = clf.decision_function(self.X_test)
                self.assertEqual(s.shape, (len(self.X_test),))
                self.assertTrue(np.isfinite(s).all(), (v, score))
            clf.score = 'combined'
            auc = roc_auc_score(self.y_test,
                                clf.decision_function(self.X_test))
            self.assertGreaterEqual(auc, 0.75, v)

    def test_versions_differ(self):
        ref = TSJEPAOD(**self.kw).fit(self.X_train).decision_scores_
        seen = [ref]
        for v in NEW:
            s = self.fitted[v].decision_scores_
            for other in seen:
                self.assertFalse(np.allclose(s, other), v)
            seen.append(s)

    def test_deterministic(self):
        for v in NEW:
            again = TSJEPAOD(variant=v, **self.kw).fit(self.X_train)
            np.testing.assert_array_equal(again.decision_scores_,
                                          self.fitted[v].decision_scores_)
            np.testing.assert_array_equal(
                again.decision_function(self.X_test),
                self.fitted[v].decision_function(self.X_test))

    def test_resolved_defaults(self):
        for v, clf in self.fitted.items():
            norm, momentum = DEFAULTS.get(v, (False, None))
            self.assertEqual(clf.encoder_norm_, norm, v)
            self.assertEqual(clf.ema_momentum_, momentum, v)
            self.assertEqual(clf.variant_params_, RECIPES[v].params, v)
            self.assertIsNone(clf.variant_params)   # stored as given
        clf = TSJEPAOD(variant='vjepa', ema_momentum=0.9, encoder_norm=True,
                       **self.kw).fit(self.X_train)
        self.assertEqual((clf.encoder_norm_, clf.ema_momentum_), (True, 0.9))
        self.assertTrue(np.isfinite(clf.decision_scores_).all())

    def test_variant_params(self):
        given = {'cov_weight': 5.0}
        before = dict(RECIPES['vicreg'].params)
        clf = TSJEPAOD(variant='vicreg', variant_params=given,
                       **self.kw).fit(self.X_train)
        self.assertEqual(clf.variant_params_,
                         {'var_weight': 1.0, 'cov_weight': 5.0})
        self.assertEqual(given, {'cov_weight': 5.0})
        self.assertEqual(RECIPES['vicreg'].params, before)
        self.assertFalse(np.allclose(clf.decision_scores_,
                                     self.fitted['vicreg'].decision_scores_))
        # constants that change the network
        wide = TSJEPAOD(variant='sjepa', variant_params={'latent_dim': 5},
                        **self.kw).fit(self.X_train)
        self.assertEqual(wide.model_.latent_dim, 5)
        few = TSJEPAOD(variant='cfjepa', variant_params={
            'n_blocks': 2, 'n_crops': 2}, **dict(self.kw, epochs=1)).fit(
            self.X_train)
        self.assertEqual(len(few.model_.encoder.blocks), 2)
        for v, bad in (('vicreg', {'no_such': 1}), ('lejepa', {'x': 1}),
                       ('tsjepa', {'var_weight': 1.0})):
            with self.assertRaises(ValueError):
                TSJEPAOD(variant=v, variant_params=bad,
                         **self.kw).fit(self.X_train)
        # an empty dict is the same as None, for every version
        clf = TSJEPAOD(variant_params={}, **self.kw).fit(self.X_train)
        self.assertEqual(clf.variant_params_, {})

    def test_ema_target_follows_encoder(self):
        for v in EMA:
            m = self.fitted[v].model_
            init = TSJEPAOD(variant=v, **dict(self.kw, epochs=0)).fit(
                self.X_train).model_
            self.assertFalse(any(p.requires_grad
                                 for p in m.target_encoder.parameters()))
            moved = max((a - b).abs().max().item() for a, b in zip(
                m.target_encoder.parameters(),
                init.target_encoder.parameters()))
            lag = max((a - b).abs().max().item() for a, b in zip(
                m.target_encoder.parameters(), m.encoder.parameters()))
            self.assertGreater(moved, 0, v)   # the target was updated
            self.assertGreater(lag, 0, v)     # and is not the online copy
        for v in NEW:
            if v not in EMA:
                self.assertIsNone(getattr(self.fitted[v].model_,
                                          'target_encoder', None), v)

    def test_momentum_schedules(self):
        self.assertEqual(momentum_at(0.9, 0.5, 'fixed'), 0.9)
        self.assertAlmostEqual(momentum_at(0.9, 0.5, 'linear'), 0.95)
        self.assertAlmostEqual(momentum_at(0.9, 0.0, 'cosine'), 0.9)
        self.assertAlmostEqual(momentum_at(0.9, 0.5, 'cosine'), 0.95)
        self.assertAlmostEqual(momentum_at(0.9, 1.0, 'cosine'), 1.0)

    def test_zero_epochs_keeps_initial_weights(self):
        for v in NEW:
            clf = TSJEPAOD(variant=v, **dict(self.kw, epochs=0)).fit(
                self.X_train)
            torch.manual_seed(self.kw['random_state'])
            fresh = clf._build_model(clf.n_tokens_, clf.n_channels_)
            for a, b in zip(clf.model_.parameters(), fresh.parameters()):
                self.assertTrue(torch.equal(a, b), v)
            self.assertTrue(np.isfinite(clf.decision_scores_).all())
            self.assertFalse(clf.model_.training)

    def test_versions_without_masks(self):
        P, L = 8, 5
        for v in MASK_FREE:
            clf = self.fitted[v]
            other = TSJEPAOD(variant=v, **dict(
                self.kw, mask_mode='forecast', mask_ratio=0.6)).fit(
                self.X_train)
            np.testing.assert_array_equal(other.decision_scores_,
                                          clf.decision_scores_)
            ctx, tgt = clf._eval_mask_set()
            first = 1 if v != 'cfjepa' else clf.model_.n_ctx_tokens
            np.testing.assert_array_equal(tgt.numpy(),
                                          np.arange(first, P)[None])
            # the leading patches of the series have no prediction score
            clf.score = 'pred'
            s = clf.decision_function(self.X_test)
            np.testing.assert_array_equal(s[:first * L],
                                          np.full(first * L, clf.threshold_))
            self.assertFalse(np.any(s[first * L:] == clf.threshold_))
            clf.score = 'combined'
        self.assertEqual(self.fitted['cfjepa'].model_.n_ctx_tokens, 6)

    def test_masked_versions_other_modes(self):
        for v in NEW:
            if v in MASK_FREE:
                continue
            for mode in ('forecast', 'random'):
                clf = TSJEPAOD(variant=v, **dict(self.kw, mask_mode=mode,
                                                 epochs=1)).fit(self.X_train)
                self.assertTrue(np.isfinite(
                    clf.decision_function(self.X_test)).all(), (v, mode))

    def test_univariate_and_short_windows(self):
        Xtr, Xte, _, _ = generate_ts_data(
            n_train=1000, n_test=400, n_channels=1, contamination=0.05,
            anomaly_type='point', random_state=0)
        for v in NEW:
            clf = TSJEPAOD(variant=v, **dict(self.kw, epochs=1)).fit(Xtr)
            self.assertTrue(np.isfinite(clf.decision_function(Xte)).all(), v)
            tiny = TSJEPAOD(variant=v, window_size=4, patch_len=2, epochs=1,
                            score='combined', random_state=0).fit(Xtr[:300])
            self.assertTrue(np.isfinite(tiny.decision_scores_).all(), v)

    def test_clone_and_pickle(self):
        for v, clf in self.fitted.items():
            c = clone(clf)
            self.assertEqual(c.get_params(), clf.get_params())
            self.assertEqual(c.variant, v)
            again = pickle.loads(pickle.dumps(clf))
            np.testing.assert_array_equal(
                again.decision_function(self.X_test),
                clf.decision_function(self.X_test))

    # ------------------------------------------------------------------
    def test_salt_stages(self):
        salt = RECIPES['salt']
        hp = salt.params
        for epochs, teacher, student in ((10, 2, 8), (3, 1, 2), (1, 1, 1),
                                         (0, 0, 0)):
            det = TSJEPAOD(variant='salt', epochs=epochs)
            self.assertEqual(salt._teacher_epochs(det, hp), teacher)
            self.assertEqual(salt.train_epochs(det, hp), student)
        m = self.fitted['salt'].model_
        self.assertFalse(any(p.requires_grad for p in m.teacher_parameters()))
        self.assertTrue(all(p.requires_grad for p in m.encoder.parameters()))
        init = TSJEPAOD(variant='salt', **dict(self.kw, epochs=0)).fit(
            self.X_train).model_
        # stage 1 trained the teacher; a teacher-only run ends at the same
        # teacher, so stage 2 did not touch it
        self.assertFalse(torch.equal(m.teacher.pos, init.teacher.pos))
        det = TSJEPAOD(variant='salt', **self.kw)
        det._validate_params()
        det.n_tokens_, det.n_mask_ = 8, 2
        det.variant_params_ = dict(hp)
        torch.manual_seed(self.kw['random_state'])
        Xs = (self.X_train - self.X_train.mean(0)) / (self.X_train.std(0) + 1e-8)
        patches = torch.from_numpy(det._to_patches(det._windows(Xs, 40, 2)))
        det.encoder_norm_ = False
        only = det._build_model(8, 3)
        gen = torch.Generator().manual_seed(self.kw['random_state'])
        salt.pretrain(det, only, lambda: torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(patches), batch_size=64,
            shuffle=True, drop_last=True, generator=gen), 'cpu', gen, hp)
        for a, b in zip(only.teacher.parameters(), m.teacher.parameters()):
            self.assertTrue(torch.equal(a, b))
        # the student is what is embedded
        x = patches[:4]
        with torch.no_grad():
            self.assertTrue(torch.equal(m.token_embeddings(x),
                                        m.encoder(m.embed(x))))

    def test_next_latent_is_causal(self):
        torch.manual_seed(0)
        for style in ('lenepa', 'lewm'):
            net = NextLatentCore(6, 10, d_model=16, n_heads=2, n_layers=2,
                                 style=style).eval()
            x = torch.randn(3, 6, 10)
            y = x.clone()
            y[:, 4:] += 1.0                       # change patches 4 and 5
            with torch.no_grad():
                pa, ta, _, ha = net(x)
                pb, tb, _, hb = net(y)
            # predictions of patches 1..4 use patches 0..3 only
            self.assertTrue(torch.allclose(pa[:, :4], pb[:, :4], atol=1e-6))
            self.assertFalse(torch.allclose(pa[:, 4], pb[:, 4]))
            self.assertTrue(torch.allclose(ta[:, :3], tb[:, :3], atol=1e-6))
            self.assertTrue(torch.allclose(ha[:, :4], hb[:, :4], atol=1e-6))
            err, h = net.multi_mask_errors(x, *net.eval_mask_set())
            self.assertEqual(err.shape, (3, 1, 5))
            self.assertTrue(torch.allclose(
                err[:, 0], (pa - ta).pow(2).mean(-1), atol=1e-6))
            self.assertEqual(h.shape, (3, 6, 16))
        with self.assertRaises(ValueError):
            NextLatentCore(6, 10, style='other')

    def test_cfjepa_network(self):
        torch.manual_seed(0)

        def reference(blk, x):          # with PyTorch's own convolutions
            h, d = x.transpose(1, 2), blk.dilation
            y = sum(F.conv1d(h, c.weight, c.bias,
                             padding=d * (c.kernel_size[0] - 1) // 2,
                             dilation=d, groups=h.size(1))
                    for c in blk.convs)
            return (h + blk.point(F.gelu(blk.bn(y)))).transpose(1, 2)

        for d, L in ((1, 23), (4, 23), (16, 9), (16, 2)):
            blk = _DilatedBlock(8, d).eval()
            x = torch.randn(3, L, 8)
            self.assertTrue(torch.allclose(reference(blk, x), blk(x),
                                           atol=1e-5), (d, L))
        net = CFJEPACore(n_tokens=12, patch_len=5, n_channels=2, d_model=16)
        self.assertEqual(net.crop_ratio, (0.367, 0.418))
        self.assertEqual(net.n_ctx_tokens, 5)
        self.assertEqual(CFJEPACore(8, 5, 2, d_model=16).crop_ratio,
                         (0.6, 0.8))
        for p in net.predictors:        # near the identity
            self.assertLess((p.weight - torch.eye(16)).abs().max().item(),
                            0.01)
        x = torch.randn(4, 12, 10)
        # patches are time-major: the window is recovered by a reshape
        self.assertTrue(torch.equal(net.series(x)[:, 5:10].reshape(4, 10),
                                    x[:, 1]))
        gen = torch.Generator().manual_seed(1)
        horizon, crops = net.training_terms(x, gen)
        self.assertEqual(len(crops), 4)
        for z in crops:
            self.assertIn(z.size(1), (22, 23, 24, 25))
        self.assertTrue(torch.isfinite(horizon))
        net.eval()
        err, h = net.multi_mask_errors(x, *net.eval_mask_set())
        self.assertEqual(err.shape, (4, 1, 7))
        self.assertEqual(h.shape, (4, 12, 16))
        self.assertGreaterEqual(err.min().item(), 0.0)
        # unit-normalised vectors: the mean absolute difference is bounded
        self.assertLessEqual(err.max().item(), 2.0 / 16 ** 0.5 + 1e-6)

    def test_dense_levels(self):
        for n_layers, levels in ((1, [1]), (2, [1, 2]), (3, [1, 2, 3]),
                                 (8, [2, 4, 6, 8])):
            net = DenseJEPACore(6, 10, d_model=16, n_heads=2,
                                n_layers=n_layers)
            self.assertEqual(net.levels, levels)
        net = self.fitted['vjepa21'].model_
        x = torch.randn(5, 8, 15)
        ctx = torch.arange(0, 6).repeat(5, 1)
        tgt = torch.arange(6, 8).repeat(5, 1)
        with torch.no_grad():
            preds, targets = net(x, ctx, tgt)
        self.assertEqual(len(preds), 2)
        self.assertEqual(preds[0].shape, (5, 8, 64))   # every patch
        self.assertEqual(targets[1].shape, (5, 8, 64))
        self.assertTrue(torch.equal(targets[-1], net.token_embeddings(x)))

    def test_stochastic_latent(self):
        net = self.fitted['sjepa'].model_
        self.assertEqual(net.latent_dim, 16)
        x = torch.randn(5, 8, 15)
        ctx = torch.arange(0, 6).repeat(5, 1)
        tgt = torch.arange(6, 8).repeat(5, 1)
        with torch.no_grad():
            h_ctx = net.encoder(net.embed(x)[:, :6])
            mu, _ = net.prior(h_ctx)
            # scoring uses the prior mean
            self.assertTrue(torch.equal(net.predict(h_ctx, ctx, tgt),
                                        net.predict(h_ctx, ctx, tgt, mu)))
            self.assertFalse(torch.allclose(
                net.predict(h_ctx, ctx, tgt),
                net.predict(h_ctx, ctx, tgt, mu + 1.0)))
            _, _, _, kl = net.forward_latent(x, ctx, tgt, torch.zeros(5, 16))
        self.assertGreaterEqual(kl.item(), 0.0)

    def test_regularised_versions(self):
        patches = torch.randn(32, 8, 15)
        # Rectified LpJEPA: non-negative embeddings, many exact zeros
        emb = self.fitted['lpjepa'].model_.token_embeddings(patches)
        self.assertGreaterEqual(emb.min().item(), 0.0)
        self.assertGreater((emb == 0).float().mean().item(), 0.05)
        # KerJEPA: running target statistics, used when scoring
        ker = self.fitted['kerjepa'].model_
        init = TSJEPAOD(variant='kerjepa', **dict(self.kw, epochs=0)).fit(
            self.X_train).model_
        self.assertTrue(torch.equal(init.tgt_mean, torch.zeros(64)))
        self.assertEqual(init.tgt_var.item(), 1.0)
        self.assertGreater(ker.tgt_mean.abs().max().item(), 0)
        before = ker.tgt_mean.clone()
        ker.multi_mask_errors(patches, *self.fitted['kerjepa']._eval_mask_set())
        self.assertTrue(torch.equal(before, ker.tgt_mean))   # not in eval
        # C-JEPA has the projector for its VICReg terms
        self.assertTrue(hasattr(self.fitted['cjepa'].model_, 'projector'))


if __name__ == '__main__':
    unittest.main()
