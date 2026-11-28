# -*- coding: utf-8 -*-
"""Unit tests for the 'scjepa' version of TSJEPAOD (0.5.1)."""
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
from jepa_od._recipes import RECIPES
from jepa_od._variant_nets2 import SCJEPACore, _Codebook, _kl
from jepa_od.ts_jepa import _VARIANTS


class TestSCJEPA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = generate_ts_data(
            n_train=1500, n_test=600, n_channels=3, contamination=0.05,
            anomaly_type='both', random_state=1)
        cls.kw = dict(variant='scjepa', window_size=40, patch_len=5,
                      epochs=3, random_state=3, verbose=0)
        cls.clf = TSJEPAOD(**cls.kw).fit(cls.X_train)

    def test_registry(self):
        self.assertEqual(_VARIANTS[26:], ('scjepa',))
        self.assertEqual(len(set(_VARIANTS)), 27)
        recipe = RECIPES['scjepa']
        self.assertIn('arXiv:2602.04643', recipe.__doc__)
        self.assertFalse(recipe.masked)
        self.assertEqual((recipe.ema_momentum, recipe.ema_schedule),
                         (0.996, 'fixed'))
        # the constants of the paper's Appendix B.1
        self.assertEqual(recipe.params, dict(
            n_codes=128, temperature=0.1, fine_weight=1.0, coarse_weight=0.5,
            mse_weight=0.1, emb_weight=1.0, com_weight=0.25,
            ent_sample_weight=0.005, ent_batch_weight=0.01,
            rec_weight_start=0.5, rec_weight_end=0.1, instance_norm=True))

    def test_scores(self):
        clf = self.clf
        self.assertEqual(clf.decision_scores_.shape, (len(self.X_train),))
        self.assertTrue(np.isfinite(clf.decision_scores_).all())
        for score in ('pred', 'embed', 'combined'):
            clf.score = score
            s = clf.decision_function(self.X_test)
            self.assertEqual(s.shape, (len(self.X_test),))
            self.assertTrue(np.isfinite(s).all(), score)
            if score == 'embed':     # three epochs are too few for 'pred'
                self.assertGreater(roc_auc_score(self.y_test, s), 0.8)
        clf.score = 'pred'
        # the patches of the first half of the first window have no error
        s, valid = clf._score_series((self.X_test - clf.X_mean_) / clf.X_std_,
                                     clf._resolve_device())
        self.assertFalse(valid[:20].any())
        self.assertTrue(valid[20:].all())
        self.assertEqual(clf.emb_mean_.shape, (128,))   # code distributions

    def test_deterministic_clone_pickle(self):
        again = TSJEPAOD(**self.kw).fit(self.X_train)
        np.testing.assert_array_equal(again.decision_scores_,
                                      self.clf.decision_scores_)
        self.assertEqual(clone(self.clf).get_params()['variant'], 'scjepa')
        loaded = pickle.loads(pickle.dumps(self.clf))
        np.testing.assert_array_equal(loaded.decision_function(self.X_test),
                                      self.clf.decision_function(self.X_test))
        other = TSJEPAOD(**dict(self.kw, mask_mode='forecast',
                                mask_ratio=0.6)).fit(self.X_train)
        np.testing.assert_array_equal(other.decision_scores_,
                                      self.clf.decision_scores_)

    def test_untrained_univariate_small(self):
        clf = TSJEPAOD(**dict(self.kw, epochs=0)).fit(self.X_train)
        self.assertTrue(np.isfinite(clf.decision_scores_).all())
        self.assertFalse(np.allclose(clf.decision_scores_,
                                     self.clf.decision_scores_))
        uni = TSJEPAOD(**dict(self.kw, epochs=1)).fit(self.X_train[:, 0])
        self.assertTrue(np.isfinite(uni.decision_scores_).all())
        # two patches: one of context, one predicted; three: two and one
        for window in (10, 15):
            small = TSJEPAOD(variant='scjepa', window_size=window,
                             patch_len=5, epochs=1,
                             random_state=0).fit(self.X_train[:400])
            self.assertTrue(np.isfinite(small.decision_scores_).all())
            self.assertEqual(small.model_.n_fut, 1)
            self.assertEqual(small.model_.n_ctx, window // 5 - 1)

    def test_split_and_views(self):
        torch.manual_seed(0)
        net = SCJEPACore(n_tokens=6, patch_len=4, n_channels=3, d_model=16,
                         n_heads=2, n_layers=1, n_codes=12).eval()
        self.assertEqual((net.n_ctx, net.n_fut), (3, 3))
        ctx_idx, tgt_idx = net.eval_mask_set()
        self.assertEqual(ctx_idx.tolist(), [[0, 1, 2]])
        self.assertEqual(tgt_idx.tolist(), [[3, 4, 5]])
        x = torch.randn(5, 6, 12)
        ctx, fut, raw, mean, std = net.views(x)
        self.assertEqual(tuple(ctx.shape), (15, 3, 4))    # 5 windows x 3
        self.assertEqual(tuple(fut.shape), (15, 3, 4))
        # the second sequence is variable 1 of the first window
        series = x.reshape(5, 24, 3)
        self.assertTrue(torch.allclose(raw[1].reshape(-1), series[0, :12, 1]))
        # each half is normalised on its own
        for half in (ctx, fut):
            flat = half.reshape(15, -1)
            self.assertTrue(torch.allclose(flat.mean(1), torch.zeros(15),
                                           atol=1e-5))
            self.assertTrue(torch.allclose(flat.std(1, unbiased=False),
                                           torch.ones(15), atol=1e-3))
        self.assertTrue(torch.allclose(ctx * std + mean, raw, atol=1e-5))
        # the coarse view averages every n_fut consecutive time steps
        coarse = net.coarse_view(fut)
        self.assertEqual(tuple(coarse.shape), (15, 1, 4))
        self.assertTrue(torch.allclose(
            coarse[0, 0], fut[0].reshape(-1).reshape(4, 3).mean(1)))

    def test_shift_and_scale_of_a_window_do_not_matter(self):
        torch.manual_seed(1)
        net = SCJEPACore(n_tokens=6, patch_len=4, n_channels=2, d_model=16,
                         n_heads=2, n_layers=1, n_codes=12).eval()
        x = torch.randn(4, 6, 8)
        err, codes = net.multi_mask_errors(x, None, None)
        self.assertEqual(tuple(err.shape), (4, 1, 3))
        self.assertEqual(tuple(codes.shape), (4, 6, 12))
        moved, moved_codes = net.multi_mask_errors(3.0 * x + 5.0, None, None)
        self.assertTrue(torch.allclose(err, moved, atol=1e-3))
        self.assertTrue(torch.allclose(codes, moved_codes, atol=1e-3))
        # a few windows at a time gives the same result
        net.max_sequences = 2
        chunked = net.multi_mask_errors(x, None, None)
        self.assertTrue(torch.allclose(err, chunked[0], atol=1e-6))
        self.assertTrue(torch.allclose(codes, chunked[1], atol=1e-6))

    def test_without_instance_norm(self):
        torch.manual_seed(1)
        net = SCJEPACore(n_tokens=6, patch_len=4, n_channels=2, d_model=16,
                         n_heads=2, n_layers=1, n_codes=12,
                         instance_norm=False).eval()
        x = torch.randn(4, 6, 8)
        ctx, fut, raw, mean, std = net.views(x)
        self.assertTrue(torch.equal(ctx, raw))
        self.assertEqual(float(mean.abs().max()), 0.0)
        self.assertEqual(float((std - 1).abs().max()), 0.0)
        err = net.multi_mask_errors(x, None, None)[0]
        moved = net.multi_mask_errors(3.0 * x + 5.0, None, None)[0]
        self.assertFalse(torch.allclose(err, moved, atol=1e-3))
        clf = TSJEPAOD(**dict(self.kw, epochs=1),
                       variant_params={'instance_norm': False}).fit(
            self.X_train)
        self.assertFalse(clf.model_.instance_norm)
        self.assertTrue(np.isfinite(clf.decision_scores_).all())

    def test_codebook(self):
        torch.manual_seed(2)
        book = _Codebook(n_codes=7, d_model=5, temperature=0.1)
        h = torch.randn(4, 3, 5)
        logp, z = book(h)
        self.assertTrue(torch.allclose(logp.exp().sum(-1), torch.ones(4, 3),
                                       atol=1e-5))
        self.assertTrue(torch.allclose(z, logp.exp() @ book.codes, atol=1e-6))
        # the distribution depends on the direction of h only
        self.assertTrue(torch.allclose(book(10.0 * h)[0], logp, atol=1e-4))
        # an embedding that is a prototype is assigned to it
        near, _ = book(book.codes.detach()[2:3])
        self.assertEqual(int(near.argmax()), 2)
        a = torch.log_softmax(torch.randn(6, 7), -1)
        self.assertTrue(torch.allclose(_kl(a, a), torch.zeros(6), atol=1e-6))
        self.assertTrue((_kl(a, torch.log_softmax(torch.randn(6, 7), -1))
                         > 0).all())

    def test_training_terms(self):
        torch.manual_seed(3)
        net = SCJEPACore(n_tokens=6, patch_len=4, n_channels=2, d_model=16,
                         n_heads=2, n_layers=1, n_codes=12)
        terms = net.training_terms(torch.randn(8, 6, 8))
        self.assertEqual(sorted(terms), sorted([
            'kl_fine', 'mse_fine', 'kl_coarse', 'emb', 'com', 'ent_sample',
            'ent_batch', 'rec']))
        for name, value in terms.items():
            self.assertEqual(value.dim(), 0, name)
            self.assertTrue(torch.isfinite(value), name)
            self.assertGreaterEqual(value.item(), 0.0, name)
        self.assertLessEqual(terms['ent_batch'].item(), math.log(12) + 1e-5)
        self.assertLessEqual(terms['ent_sample'].item(),
                             terms['ent_batch'].item() + 1e-5)
        # no gradient reaches the EMA copies
        sum(terms.values()).backward()
        for p in list(net.target_encoder.parameters()) \
                + list(net.target_codebook.parameters()):
            self.assertFalse(p.requires_grad)
            self.assertIsNone(p.grad)
        self.assertIsNotNone(net.codebook.codes.grad)

    def test_ema_targets_and_constant_channels(self):
        m = self.clf.model_
        init = TSJEPAOD(**dict(self.kw, epochs=0)).fit(self.X_train).model_
        for target, online, start in (
                (m.target_encoder, m.encoder, init.target_encoder),
                (m.target_codebook, m.codebook, init.target_codebook)):
            moved = max((a - b).abs().max().item() for a, b in zip(
                target.parameters(), start.parameters()))
            lag = max((a - b).abs().max().item() for a, b in zip(
                target.parameters(), online.parameters()))
            self.assertGreater(moved, 0)
            self.assertGreater(lag, 0)
        self.assertEqual(m.keep.tolist(), [True, True, True])
        # a variable that is constant in training is dropped, also when
        # it moves later
        X = self.X_train.copy()
        X[:, 1] = 7.0
        clf = TSJEPAOD(**dict(self.kw, epochs=1)).fit(X)
        self.assertEqual(clf.model_.keep.tolist(), [True, False, True])
        Xt = self.X_test.copy()
        Xt[:, 1] = 7.0
        base = clf.decision_function(Xt)
        Xt[300:, 1] = 9.0
        np.testing.assert_array_equal(clf.decision_function(Xt), base)

    def test_variant_params(self):
        clf = TSJEPAOD(**dict(self.kw, epochs=1),
                       variant_params={'n_codes': 16}).fit(self.X_train)
        self.assertEqual(clf.model_.codebook.codes.shape[0], 16)
        self.assertEqual(clf.emb_mean_.shape, (16,))
        with self.assertRaises(ValueError):
            TSJEPAOD(**dict(self.kw), variant_params={'curv_weight': 1.0}
                     ).fit(self.X_train)


if __name__ == '__main__':
    unittest.main()
