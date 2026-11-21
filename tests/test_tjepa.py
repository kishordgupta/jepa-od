# -*- coding: utf-8 -*-
"""PyOD-style unit tests for TJEPAOD (mirrors tests/test_jepa.py)."""
import os
import sys
import unittest

import numpy as np
import torch
from sklearn.base import clone
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyod.utils.data import generate_data
from jepa_od import JEPAOD, TJEPAOD
from jepa_od._tjepa_nets import (TJEPACore, tjepa_size_ranges,
                                 sample_tjepa_sizes, sample_tjepa_masks,
                                 eval_tjepa_masks)


class TestTJEPAMasks(unittest.TestCase):
    def test_size_ranges(self):
        self.assertEqual(tjepa_size_ranges(20, 0.2, 0.4, 0.2, 0.4),
                         ((4, 8), (4, 8)))
        # tiny tables still get one context and one target token
        self.assertEqual(tjepa_size_ranges(2, 0.2, 0.4, 0.2, 0.4),
                         ((1, 1), (1, 1)))
        for T in range(2, 70):
            (c_lo, c_hi), (t_lo, t_hi) = tjepa_size_ranges(T, 0.2, 0.4,
                                                           0.2, 0.4)
            self.assertTrue(1 <= c_lo <= c_hi and 1 <= t_lo <= t_hi)
            self.assertLessEqual(c_lo + t_lo, T)

    def test_sampled_sizes(self):
        gen = torch.Generator().manual_seed(0)
        for _ in range(200):
            n_ctx, n_tgt = sample_tjepa_sizes(10, (2, 9), (2, 9), gen)
            self.assertTrue(2 <= n_ctx <= 9 and 2 <= n_tgt <= 9)
            self.assertLessEqual(n_ctx + n_tgt, 10)

    def test_training_masks(self):
        gen = torch.Generator().manual_seed(0)
        ctx, tgt = sample_tjepa_masks(32, 20, 6, 5, 4, gen)
        self.assertEqual(tuple(ctx.shape), (32, 6))
        self.assertEqual(tuple(tgt.shape), (32, 4, 5))
        for b in range(32):
            c = set(ctx[b].tolist())
            self.assertEqual(len(c), 6)
            for p in range(4):
                t = set(tgt[b, p].tolist())
                self.assertEqual(len(t), 5)
                # targets never overlap the context
                self.assertFalse(c & t)
        # rows get different masks
        self.assertGreater(len({tuple(r.tolist()) for r in ctx}), 1)

    def test_eval_masks(self):
        gen = torch.Generator().manual_seed(0)
        ctx, tgt = eval_tjepa_masks(20, 6, 5, 8, gen)
        self.assertEqual(tuple(ctx.shape), (8, 6))
        self.assertEqual(tuple(tgt.shape), (8, 5))
        for k in range(8):
            self.assertFalse(set(ctx[k].tolist()) & set(tgt[k].tolist()))
        # the first full permutation covers every token once as a target
        self.assertEqual(sorted(tgt[:4].reshape(-1).tolist()),
                         list(range(20)))


class TestTJEPACore(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.net = TJEPACore(n_tokens=10, in_dim=1, d_model=16, n_heads=2,
                             n_layers=1, dim_feedforward=32, dropout=0.0,
                             pred_dim=16, pred_heads=2, pred_layers=1,
                             pred_feedforward=32, n_reg_tokens=1)
        self.x = torch.randn(6, 10, 1)

    def test_shapes(self):
        gen = torch.Generator().manual_seed(0)
        ctx, tgt = sample_tjepa_masks(6, 10, 3, 2, 4, gen)
        pred, h_tgt, h_all = self.net(self.x, ctx, tgt)
        self.assertEqual(tuple(pred.shape), (24, 2, 16))
        self.assertEqual(tuple(h_tgt.shape), (24, 2, 16))
        # the [REG] output is dropped
        self.assertEqual(tuple(h_all.shape), (6, 10, 16))
        self.assertEqual(tuple(self.net.encoder(self.x, ctx).shape),
                         (6, 3, 16))

    def test_target_encoder_is_ema_only(self):
        self.assertFalse(any(p.requires_grad
                             for p in self.net.target_encoder.parameters()))
        gen = torch.Generator().manual_seed(0)
        ctx, tgt = sample_tjepa_masks(6, 10, 3, 2, 4, gen)
        pred, h_tgt, _ = self.net(self.x, ctx, tgt)
        self.assertFalse(h_tgt.requires_grad)
        (pred - h_tgt).pow(2).mean().backward()
        self.assertIsNotNone(self.net.encoder.reg.grad)
        self.assertIsNone(self.net.target_encoder.reg.grad)

    def test_reg_token_changes_the_encoding(self):
        with torch.no_grad():
            self.net.eval()
            h1 = self.net.encoder(self.x)
            self.net.encoder.reg.add_(1.0)
            h2 = self.net.encoder(self.x)
        self.assertGreater((h1 - h2).abs().max().item(), 1e-4)

    def test_ema_update(self):
        from jepa_od._sigreg import ema_update
        before = self.net.target_encoder.tokenizer.weight.clone()
        with torch.no_grad():
            self.net.encoder.tokenizer.weight.copy_(before + 1.0)
        ema_update(self.net.target_encoder, self.net.encoder, 0.75)
        np.testing.assert_allclose(
            self.net.target_encoder.tokenizer.weight.numpy(),
            (before + 0.25).numpy(), atol=1e-6)

    def test_encoder_norm_switch(self):
        normed = TJEPACore(10, 1, d_model=16, n_heads=2, n_layers=1,
                           pred_dim=16, pred_heads=2, pred_layers=1)
        plain = TJEPACore(10, 1, d_model=16, n_heads=2, n_layers=1,
                          pred_dim=16, pred_heads=2, pred_layers=1,
                          encoder_norm=False)
        self.assertIsInstance(normed.encoder.body.norm2, torch.nn.LayerNorm)
        self.assertIsInstance(plain.encoder.body.norm2, torch.nn.Identity)


class TestTJEPAOD(unittest.TestCase):
    def assertHasAttr(self, obj, intended_attr):
        self.assertTrue(hasattr(obj, intended_attr))

    def assertInRange(self, data, lower, upper):
        self.assertGreaterEqual(data.min(), lower)
        self.assertLessEqual(data.max(), upper)

    def setUp(self):
        self.n_train = 1200
        self.n_test = 400
        self.n_features = 20
        self.contamination = 0.1
        self.roc_floor = 0.8
        self.X_train, self.X_test, self.y_train, self.y_test = generate_data(
            n_train=self.n_train, n_test=self.n_test,
            n_features=self.n_features, contamination=self.contamination,
            random_state=42)
        self.clf = TJEPAOD(epoch_num=5, contamination=self.contamination,
                           verbose=0)
        self.clf.fit(self.X_train)

    def test_is_family_member(self):
        self.assertIsInstance(self.clf, JEPAOD)
        self.assertIsInstance(self.clf.model, TJEPACore)

    def test_parameters(self):
        for attr in ('decision_scores_', 'labels_', 'threshold_', '_mu',
                     '_sigma', 'model', 'err_mean_', 'err_std_',
                     'emb_mean_', 'emb_prec_', 'ctx_range_', 'tgt_range_',
                     'n_ctx_', 'n_tgt_'):
            self.assertHasAttr(self.clf, attr)
            self.assertIsNotNone(getattr(self.clf, attr))
        self.assertEqual(self.clf.ctx_range_, (4, 8))
        self.assertEqual(self.clf.tgt_range_, (4, 8))
        self.assertEqual((self.clf.n_ctx_, self.clf.n_tgt_), (6, 6))

    def test_train_scores(self):
        self.assertEqual(len(self.clf.decision_scores_), self.X_train.shape[0])

    def test_prediction_scores(self):
        pred_scores = self.clf.decision_function(self.X_test)
        self.assertEqual(pred_scores.shape[0], self.X_test.shape[0])
        self.assertGreaterEqual(roc_auc_score(self.y_test, pred_scores),
                                self.roc_floor)

    def test_prediction_deterministic(self):
        s1 = self.clf.decision_function(self.X_test)
        s2 = self.clf.decision_function(self.X_test)
        np.testing.assert_allclose(s1, s2)

    def test_fit_reproducible(self):
        clf = TJEPAOD(epoch_num=5, contamination=self.contamination,
                      verbose=0).fit(self.X_train)
        np.testing.assert_allclose(clf.decision_scores_,
                                   self.clf.decision_scores_, rtol=1e-4,
                                   atol=1e-4)

    def test_prediction_labels(self):
        pred_labels = self.clf.predict(self.X_test)
        self.assertEqual(pred_labels.shape, self.y_test.shape)

    def test_prediction_proba(self):
        for method in ('linear', 'unify'):
            pred_proba = self.clf.predict_proba(self.X_test, method=method)
            self.assertInRange(pred_proba, 0, 1)

    def test_prediction_labels_confidence(self):
        pred_labels, confidence = self.clf.predict(self.X_test,
                                                   return_confidence=True)
        self.assertEqual(pred_labels.shape, self.y_test.shape)
        self.assertEqual(confidence.shape, self.y_test.shape)
        self.assertInRange(confidence, 0, 1)

    def test_fit_predict(self):
        pred_labels = self.clf.fit_predict(self.X_train)
        self.assertEqual(pred_labels.shape, self.y_train.shape)

    def test_model_clone(self):
        clone_clf = clone(self.clf)
        self.assertEqual(clone_clf.get_params(), self.clf.get_params())

    def test_score_variants(self):
        for score in ('pred', 'embed'):
            clf = TJEPAOD(epoch_num=5, score=score, verbose=0)
            clf.fit(self.X_train)
            s = clf.decision_function(self.X_test)
            self.assertEqual(s.shape[0], self.n_test)
            self.assertGreaterEqual(roc_auc_score(self.y_test, s),
                                    self.roc_floor)

    def test_variants(self):
        for kw in (dict(encoder_norm=False), dict(n_reg_tokens=0),
                   dict(n_reg_tokens=2, n_pred_masks=1),
                   dict(warmup_epochs=0, lr_schedule=False)):
            clf = TJEPAOD(epoch_num=5, verbose=0, **kw).fit(self.X_train)
            s = clf.decision_function(self.X_test)
            self.assertGreaterEqual(roc_auc_score(self.y_test, s),
                                    self.roc_floor)

    def test_token_embeddings(self):
        emb = self.clf.token_embeddings(self.X_test)
        self.assertEqual(emb.shape, (self.n_test, self.clf.n_tokens_,
                                     self.clf.d_model))
        np.testing.assert_allclose(emb,
                                   self.clf.token_embeddings(self.X_test))
        emb = JEPAOD(epoch_num=2, verbose=0).fit(
            self.X_train).token_embeddings(self.X_test)
        self.assertEqual(emb.shape, (self.n_test, self.n_features, 64))

    def test_two_features(self):
        Xtr, Xte, ytr, yte = generate_data(n_train=600, n_test=200,
                                           n_features=2, random_state=0)
        clf = TJEPAOD(epoch_num=5, verbose=0).fit(Xtr)
        self.assertEqual((clf.n_ctx_, clf.n_tgt_), (1, 1))
        self.assertEqual(clf.decision_function(Xte).shape[0], 200)

    def test_feature_grouping(self):
        Xtr, Xte, ytr, yte = generate_data(n_train=600, n_test=200,
                                           n_features=150, random_state=0)
        clf = TJEPAOD(epoch_num=3, max_tokens=32, verbose=0).fit(Xtr)
        self.assertEqual(clf.n_tokens_, 30)
        self.assertEqual(clf.token_dim_, 5)
        self.assertEqual(clf.decision_function(Xte).shape[0], 200)

    def test_invalid_params(self):
        for kw in (dict(score='bad'), dict(min_ctx_share=0.5,
                                           max_ctx_share=0.3),
                   dict(max_tgt_share=1.5), dict(n_pred_masks=0),
                   dict(n_reg_tokens=-1)):
            with self.assertRaises(ValueError):
                TJEPAOD(epoch_num=1, verbose=0, **kw).fit(self.X_train)
        with self.assertRaises(ValueError):
            TJEPAOD(epoch_num=1, verbose=0).fit(self.X_train[:, :1])

    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()
