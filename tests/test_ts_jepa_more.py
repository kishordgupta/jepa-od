# -*- coding: utf-8 -*-
"""Unit tests for the ten JEPA versions of TSJEPAOD added in 0.5.0."""
import math
import os
import pickle
import sys
import unittest

import numpy as np
import torch
from sklearn.base import clone
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyod.utils.data import generate_ts_data
from jepa_od import TSJEPAOD
from jepa_od._recipes import RECIPES, momentum_at
from jepa_od._recipes2 import MORE_RECIPES
from jepa_od._variant_nets2 import (HEPACore, SubspaceLatentCore,
                                    GMMJEPACore, PhiJEPACore,
                                    VariationalJEPACore, cosine_distance)
from jepa_od.ts_jepa import _VARIANTS

MORE = tuple(MORE_RECIPES)
# versions with an EMA target encoder; 'ststjepa' has an EMA tokenizer
EMA_ENCODER = ('njepa', 'gmmjepa', 'variational', 'phijepa', 'harjepa')
MASK_FREE = ('hepa', 'subjepa', 'straight', 'motionjepa')


class TestTSJEPAODTenVersions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = generate_ts_data(
            n_train=1500, n_test=600, n_channels=3, contamination=0.05,
            anomaly_type='both', random_state=1)
        cls.kw = dict(window_size=40, patch_len=5, epochs=3,
                      score='combined', random_state=3, verbose=0)
        cls.fitted = {v: TSJEPAOD(variant=v, **cls.kw).fit(cls.X_train)
                      for v in MORE}

    def test_registry(self):
        self.assertEqual(_VARIANTS[16:], MORE)
        self.assertEqual(len(_VARIANTS), 26)
        self.assertEqual(len(set(_VARIANTS)), 26)
        for v in MORE:
            recipe = RECIPES[v]
            self.assertTrue(recipe.__doc__, v)
            self.assertIn('arXiv:', recipe.__doc__, v)
            self.assertEqual(recipe.masked, v not in MASK_FREE)
            self.assertEqual(recipe.ema_schedule is not None,
                             v in EMA_ENCODER + ('ststjepa',))

    def test_scores(self):
        for v, clf in self.fitted.items():
            self.assertEqual(clf.decision_scores_.shape, (len(self.X_train),))
            self.assertTrue(np.isfinite(clf.decision_scores_).all(), v)
            for score in ('pred', 'embed', 'combined'):
                clf.score = score
                s = clf.decision_function(self.X_test)
                self.assertEqual(s.shape, (len(self.X_test),))
                self.assertTrue(np.isfinite(s).all(), (v, score))
            clf.score = 'embed'
            auc = roc_auc_score(self.y_test,
                                clf.decision_function(self.X_test))
            self.assertGreaterEqual(auc, 0.9, v)
            clf.score = 'combined'

    def test_versions_differ(self):
        seen = [TSJEPAOD(variant=v, **self.kw).fit(self.X_train)
                .decision_scores_ for v in ('lejepa', 'ijepa', 'lewm')]
        for v in MORE:
            s = self.fitted[v].decision_scores_
            for other in seen:
                self.assertFalse(np.allclose(s, other), v)
            seen.append(s)

    def test_deterministic(self):
        for v in MORE:
            again = TSJEPAOD(variant=v, **self.kw).fit(self.X_train)
            np.testing.assert_array_equal(again.decision_scores_,
                                          self.fitted[v].decision_scores_)
            np.testing.assert_array_equal(
                again.decision_function(self.X_test),
                self.fitted[v].decision_function(self.X_test))

    def test_resolved_defaults(self):
        for v, clf in self.fitted.items():
            self.assertFalse(clf.encoder_norm_, v)
            self.assertEqual(clf.ema_momentum_, RECIPES[v].ema_momentum, v)
            self.assertEqual(clf.variant_params_, RECIPES[v].params, v)
        self.assertEqual(self.fitted['ststjepa'].ema_momentum_, 0.9996)
        for v in MASK_FREE:
            self.assertIsNone(self.fitted[v].ema_momentum_)

    def test_clone_pickle_and_univariate(self):
        X = self.X_train[:, 0]
        for v in MORE:
            clf = clone(self.fitted[v])
            self.assertEqual(clf.get_params()['variant'], v)
            again = pickle.loads(pickle.dumps(self.fitted[v]))
            np.testing.assert_array_equal(
                again.decision_function(self.X_test),
                self.fitted[v].decision_function(self.X_test))
            uni = TSJEPAOD(variant=v, **dict(self.kw, epochs=1)).fit(X)
            self.assertTrue(np.isfinite(uni.decision_scores_).all(), v)

    def test_untrained_and_two_patches(self):
        for v in MORE:
            clf = TSJEPAOD(variant=v, **dict(self.kw, epochs=0)).fit(
                self.X_train)
            self.assertTrue(np.isfinite(clf.decision_scores_).all(), v)
            small = TSJEPAOD(variant=v, window_size=10, patch_len=5,
                             epochs=1, random_state=0).fit(self.X_train[:400])
            self.assertTrue(np.isfinite(small.decision_scores_).all(), v)

    def test_momentum_end(self):
        self.assertAlmostEqual(momentum_at(0.9996, 0.0, 'linear', 0.9999),
                               0.9996)
        self.assertAlmostEqual(momentum_at(0.9996, 1.0, 'linear', 0.9999),
                               0.9999)
        self.assertAlmostEqual(momentum_at(0.9, 1.0, 'cosine', 0.95), 0.95)
        self.assertAlmostEqual(momentum_at(0.9, 0.5, 'linear'), 0.95)
        self.assertEqual(RECIPES['ststjepa'].ema_end, 0.9999)
        self.assertEqual(RECIPES['vjepa'].ema_end, 1.0)

    def test_ema_targets(self):
        for v in EMA_ENCODER:
            m = self.fitted[v].model_
            init = TSJEPAOD(variant=v, **dict(self.kw, epochs=0)).fit(
                self.X_train).model_
            self.assertFalse(any(p.requires_grad
                                 for p in m.target_encoder.parameters()))
            moved = max((a - b).abs().max().item() for a, b in zip(
                m.target_encoder.parameters(),
                init.target_encoder.parameters()))
            self.assertGreater(moved, 0, v)
        # 'ststjepa': the tokenizer has the EMA twin, the encoder has none
        m = self.fitted['ststjepa'].model_
        self.assertIsNone(m.target_encoder)
        self.assertFalse(m.target_tokenizer.weight.requires_grad)
        gap = (m.target_tokenizer.weight - m.tokenizer.weight).abs().max()
        self.assertGreater(gap.item(), 0)
        init = TSJEPAOD(variant='ststjepa', **dict(self.kw, epochs=0)).fit(
            self.X_train).model_
        moved = (m.target_tokenizer.weight
                 - init.target_tokenizer.weight).abs().max().item()
        self.assertGreater(moved, 0)
        # with momentum 0.9996 the target stays near its start
        self.assertLess(moved, gap.item())

    def test_mask_free_versions(self):
        for v in MASK_FREE:
            clf = self.fitted[v]
            ctx, tgt = clf._eval_mask_set()
            self.assertEqual(tgt.tolist(), [list(range(1, clf.n_tokens_))])
            other = TSJEPAOD(variant=v, mask_mode='forecast', mask_ratio=0.6,
                             **self.kw).fit(self.X_train)
            np.testing.assert_array_equal(other.decision_scores_,
                                          clf.decision_scores_)
            # the first patch of the series has no prediction error
            clf.score = 'pred'
            s, valid = clf._score_series(
                (self.X_test - clf.X_mean_) / clf.X_std_,
                clf._resolve_device())
            self.assertFalse(valid[:clf.patch_len].any())
            self.assertTrue(valid[clf.patch_len:].all())
            clf.score = 'combined'

    def test_untrained_controls_share_the_lewm_network(self):
        # these three add modules or buffers after the 'lewm' network, so
        # the zero-epoch controls have the same scores
        kw = dict(self.kw, epochs=0)
        ref = TSJEPAOD(variant='lewm', **kw).fit(self.X_train)
        for v in ('subjepa', 'straight', 'motionjepa'):
            clf = TSJEPAOD(variant=v, **kw).fit(self.X_train)
            np.testing.assert_array_equal(clf.decision_scores_,
                                          ref.decision_scores_)

    def test_hepa_network(self):
        torch.manual_seed(0)
        net = HEPACore(n_tokens=8, in_dim=6, d_model=16, n_heads=2,
                       n_layers=1).eval()
        self.assertEqual(net.score_horizons, (1, 2, 4))
        x = torch.randn(3, 8, 6)
        h = net.summaries(x)
        # causal: a summary does not change when later patches change
        x2 = x.clone()
        x2[:, 5:] += 1.0
        h2 = net.summaries(x2)
        self.assertTrue(torch.allclose(h[:, :5], h2[:, :5], atol=1e-6))
        self.assertFalse(torch.allclose(h[:, 5:], h2[:, 5:]))
        # the target of an interval depends on that interval only
        self.assertTrue(torch.allclose(net.interval(x[:, 2:4]),
                                       net.interval(x2[:, 2:4])))
        # the horizon changes the prediction
        self.assertFalse(torch.allclose(net.predict(h[:, 0], 1),
                                        net.predict(h[:, 0], 2)))
        err, emb = net.multi_mask_errors(x, *net.eval_mask_set())
        self.assertEqual(err.shape, (3, 1, 7))
        self.assertEqual(emb.shape, (3, 8, 16))
        # the error of the last patch by hand: intervals of 1, 2 and 4
        # patches that end at the last patch
        want = 0.0
        for dt in (1, 2, 4):
            s = 8 - dt
            want = want + (net.predict(h[:, s - 1], dt)
                           - net.interval(x[:, s:])).abs().mean(-1)
        self.assertTrue(torch.allclose(err[:, 0, -1], want.detach() / 3,
                                       atol=1e-5))

    def test_subspaces(self):
        net = SubspaceLatentCore(n_tokens=6, in_dim=4, d_model=64, n_heads=4,
                                 n_layers=1, subspace_dim=6, seed=5)
        P = net.subspaces
        self.assertEqual(tuple(P.shape), (11, 6, 64))    # round(64 / 6)
        eye = torch.eye(6).expand(11, 6, 6)
        self.assertTrue(torch.allclose(P @ P.transpose(1, 2), eye, atol=1e-5))
        self.assertFalse(P.requires_grad)
        same = SubspaceLatentCore(n_tokens=6, in_dim=4, d_model=64,
                                  n_heads=4, n_layers=1, seed=5).subspaces
        self.assertTrue(torch.equal(P, same))
        z = torch.randn(10, 64)
        self.assertTrue(torch.allclose(net.project(z)[3], z @ P[3].T,
                                       atol=1e-5))
        clf = TSJEPAOD(variant='subjepa', variant_params={'subspace_dim': 16},
                       **dict(self.kw, epochs=1)).fit(self.X_train)
        self.assertEqual(tuple(clf.model_.subspaces.shape), (4, 16, 64))

    def test_gmm_anchor(self):
        clf = self.fitted['gmmjepa']
        m = clf.model_
        self.assertEqual(m.cluster_head.out_features, 32)
        x = torch.randn(5, 3, m.gmm_mean.size(1))
        q = m.posteriors(x)
        self.assertEqual(tuple(q.shape), (5, 3, 32))
        self.assertTrue(torch.allclose(q.sum(-1), torch.ones(5, 3),
                                       atol=1e-5))
        # the posteriors are those of the mixture
        from sklearn.mixture import GaussianMixture
        rng = np.random.RandomState(0)
        data = np.concatenate([rng.randn(200, 4) + 3, rng.randn(200, 4) - 3])
        gmm = GaussianMixture(2, covariance_type='diag',
                              random_state=0).fit(data)
        net = GMMJEPACore(n_tokens=4, in_dim=4, n_components=3, d_model=16,
                          n_heads=2, n_layers=1)
        net.set_gmm(gmm.weights_, gmm.means_, gmm.covariances_)
        q = net.posteriors(torch.as_tensor(data, dtype=torch.float32))
        np.testing.assert_allclose(q[:, :2].numpy(), gmm.predict_proba(data),
                                   atol=1e-4)
        self.assertEqual(float(q[:, 2].abs().max()), 0.0)   # unused
        # more components than half the patches: the mixture is smaller
        tiny = TSJEPAOD(variant='gmmjepa', window_size=10, patch_len=5,
                        epochs=1, max_train_windows=20, random_state=0,
                        variant_params={'n_components': 64}).fit(
            self.X_train[:300])
        self.assertTrue(np.isfinite(tiny.decision_scores_).all())
        self.assertEqual(int((tiny.model_.gmm_const > -1e29).sum()), 20)

    def test_variational_score(self):
        nll = VariationalJEPACore.nll
        t, m = torch.tensor([1.0]), torch.tensor([0.0])
        self.assertAlmostEqual(nll(t, m, torch.tensor([0.0])).item(), 0.5)
        self.assertAlmostEqual(nll(t, m, torch.tensor([1.0])).item(),
                               0.5 * (1.0 + math.exp(-1.0)), places=6)
        # a larger predicted variance lowers the cost of a large miss
        far = torch.tensor([3.0])
        self.assertLess(nll(far, m, torch.tensor([2.0])).item(),
                        nll(far, m, torch.tensor([0.0])).item())
        clf = self.fitted['variational']
        clf.score = 'pred'
        self.assertTrue(np.isfinite(
            clf.decision_function(self.X_test)).all())
        clf.score = 'combined'

    def test_phijepa_corruption(self):
        net = PhiJEPACore(n_tokens=6, in_dim=10, n_channels=2, d_model=16,
                          n_heads=2, n_layers=1)
        self.assertEqual(net.head[0].out_features, 43)   # round(16 * 8 / 3)
        x = torch.randn(7, 6, 10)
        gen = torch.Generator().manual_seed(0)
        self.assertTrue(torch.equal(net.corrupt(x, gen, p=0.0), x))
        a = net.corrupt(x, torch.Generator().manual_seed(1))
        b = net.corrupt(x, torch.Generator().manual_seed(1))
        c = net.corrupt(x, torch.Generator().manual_seed(2))
        self.assertTrue(torch.equal(a, b))
        self.assertFalse(torch.allclose(a, c))
        self.assertEqual(a.shape, x.shape)
        # gain only: every channel of a window is an affine map of itself
        g = net.corrupt(x, torch.Generator().manual_seed(3), p_gain=1.0,
                        p_smooth=0.0, p_noise=0.0).reshape(7, 30, 2)
        xs = x.reshape(7, 30, 2)
        ratio = (g[:, 1:] - g[:, :-1]) / (xs[:, 1:] - xs[:, :-1])
        self.assertTrue(torch.allclose(ratio, ratio[:, :1].expand_as(ratio),
                                       atol=1e-3))
        u = torch.randn(4, 8)
        self.assertTrue(torch.allclose(cosine_distance(u, 3 * u),
                                       torch.zeros(4), atol=1e-5))
        self.assertTrue(torch.allclose(cosine_distance(u, -u),
                                       4 * torch.ones(4), atol=1e-5))

    def test_harjepa_targets_are_not_normalised(self):
        m = self.fitted['harjepa'].model_
        x = torch.as_tensor(self.fitted['harjepa']._to_patches(
            self.fitted['harjepa']._windows(
                self.X_test.astype(np.float32), 40, 40)))
        h = m.token_embeddings(x)
        # a layer-normalised vector has unit variance over its dimensions
        self.assertFalse(torch.allclose(h.std(-1), torch.ones(h.shape[:2]),
                                        atol=0.05))
        n = self.fitted['njepa'].model_
        self.assertTrue(torch.allclose(n.token_embeddings(x).std(-1),
                                       torch.ones(h.shape[:2]), atol=0.05))

    def test_variant_params(self):
        clf = TSJEPAOD(variant='straight', variant_params={'curv_weight': 5.},
                       **self.kw).fit(self.X_train)
        self.assertEqual(clf.variant_params_, {'curv_weight': 5.0})
        self.assertFalse(np.allclose(clf.decision_scores_,
                                     self.fitted['straight'].decision_scores_))
        clf = TSJEPAOD(variant='hepa', variant_params={'n_pairs': 1},
                       **self.kw).fit(self.X_train)
        self.assertFalse(np.allclose(clf.decision_scores_,
                                     self.fitted['hepa'].decision_scores_))
        with self.assertRaises(ValueError):
            TSJEPAOD(variant='hepa', variant_params={'curv_weight': 1.0},
                     **self.kw).fit(self.X_train)


if __name__ == '__main__':
    unittest.main()
