# -*- coding: utf-8 -*-
"""PyOD-style unit tests for TSJEPAOD (mirrors pyod/test/test_ts_lstm.py)."""
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
from jepa_od._tsjepa_nets import ConvPatchTokenizer

VARIANTS = ('lejepa', 'ijepa', 'tsjepa', 'tjepa')
EMA_VARIANTS = ('ijepa', 'tsjepa', 'tjepa')


class TestTSJEPAOD(unittest.TestCase):
    def setUp(self):
        self.X_train, self.X_test, self.y_train, self.y_test = generate_ts_data(
            n_train=2000, n_test=800, n_channels=1, contamination=0.05,
            anomaly_type='point', random_state=42)
        self.clf = TSJEPAOD(window_size=40, patch_len=5, epochs=5,
                            contamination=0.05, verbose=0)
        self.clf.fit(self.X_train)

    def test_parameters(self):
        for attr in ('decision_scores_', 'labels_', 'threshold_', 'model_',
                     'err_mean_', 'err_std_', 'emb_mean_', 'emb_prec_'):
            self.assertTrue(hasattr(self.clf, attr))

    def test_train_scores(self):
        self.assertEqual(len(self.clf.decision_scores_), len(self.X_train))
        self.assertEqual(len(self.clf.labels_), len(self.X_train))
        self.assertTrue(np.isfinite(self.clf.decision_scores_).all())

    def test_prediction_scores(self):
        s = self.clf.decision_function(self.X_test)
        self.assertEqual(s.shape, (len(self.X_test),))
        self.assertTrue(np.isfinite(s).all())
        self.assertGreaterEqual(roc_auc_score(self.y_test, s), 0.8)

    def test_prediction_deterministic(self):
        np.testing.assert_allclose(self.clf.decision_function(self.X_test),
                                   self.clf.decision_function(self.X_test))

    def test_prediction_labels(self):
        labels = self.clf.predict(self.X_test)
        self.assertEqual(labels.shape, self.y_test.shape)
        self.assertTrue(set(np.unique(labels)) <= {0, 1})

    def test_prediction_proba(self):
        p = self.clf.predict_proba(self.X_test)
        self.assertGreaterEqual(p.min(), 0)
        self.assertLessEqual(p.max(), 1)

    def test_multivariate_and_modes(self):
        Xtr, Xte, ytr, yte = generate_ts_data(
            n_train=2000, n_test=800, n_channels=3, contamination=0.05,
            anomaly_type='both', random_state=1)
        for mode in ('block', 'forecast', 'random', 'mixed'):
            for score in ('pred', 'embed', 'combined'):
                clf = TSJEPAOD(window_size=40, patch_len=5, epochs=3,
                               mask_mode=mode, score=score, verbose=0)
                clf.fit(Xtr)
                s = clf.decision_function(Xte)
                self.assertEqual(s.shape, (len(Xte),))
                self.assertTrue(np.isfinite(s).all())

    def test_ema_max_aggregation(self):
        clf = TSJEPAOD(window_size=40, patch_len=5, epochs=3,
                       variant='ijepa', score_aggregation='max',
                       n_mask_samples=3, verbose=0).fit(self.X_train)
        self.assertTrue(np.isfinite(clf.decision_function(self.X_test)).all())

    def test_model_clone(self):
        c = clone(self.clf)
        self.assertEqual(c.get_params(), self.clf.get_params())

    def test_invalid(self):
        with self.assertRaises(ValueError):
            TSJEPAOD(window_size=45, patch_len=10).fit(self.X_train)
        with self.assertRaises(ValueError):
            TSJEPAOD(window_size=40, patch_len=5).fit(self.X_train[:30])
        with self.assertRaises(ValueError):
            self.clf.decision_function(np.random.randn(500, 2))
        with self.assertRaises(ValueError):
            TSJEPAOD(window_size=40, patch_len=5,
                     variant='vjepa').fit(self.X_train)
        for v in ('tsjepa', 'tjepa'):
            with self.assertRaises(ValueError):
                TSJEPAOD(window_size=40, patch_len=5, variant=v,
                         recon_weight=0.1).fit(self.X_train)


class TestTSJEPAODVariants(unittest.TestCase):
    """The JEPA versions selectable with ``variant``."""

    @classmethod
    def setUpClass(cls):
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = generate_ts_data(
            n_train=1500, n_test=600, n_channels=3, contamination=0.05,
            anomaly_type='both', random_state=1)
        cls.kw = dict(window_size=40, patch_len=5, epochs=3,
                      score='combined', random_state=3, verbose=0)
        cls.fitted = {v: TSJEPAOD(variant=v, **cls.kw).fit(cls.X_train)
                      for v in VARIANTS}

    def test_default_is_lejepa(self):
        self.assertEqual(TSJEPAOD().variant, 'lejepa')
        a = TSJEPAOD(**self.kw).fit(self.X_train)
        np.testing.assert_array_equal(a.decision_scores_,
                                      self.fitted['lejepa'].decision_scores_)

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
            self.assertGreaterEqual(auc, 0.8, v)

    def test_versions_differ(self):
        ref = self.fitted['lejepa'].decision_scores_
        for v in EMA_VARIANTS:
            self.assertFalse(np.allclose(ref, self.fitted[v].decision_scores_))

    def test_deterministic(self):
        for v in VARIANTS:
            again = TSJEPAOD(variant=v, **self.kw).fit(self.X_train)
            np.testing.assert_array_equal(again.decision_scores_,
                                          self.fitted[v].decision_scores_)
            np.testing.assert_array_equal(
                again.decision_function(self.X_test),
                self.fitted[v].decision_function(self.X_test))

    def test_resolved_defaults(self):
        expected = {'lejepa': (False, None), 'ijepa': (False, 0.996),
                    'tsjepa': (True, 0.998), 'tjepa': (True, 0.996)}
        for v, (norm, momentum) in expected.items():
            self.assertEqual(self.fitted[v].encoder_norm_, norm)
            self.assertEqual(self.fitted[v].ema_momentum_, momentum)
        clf = TSJEPAOD(variant='tsjepa', encoder_norm=False,
                       ema_momentum=0.99, **self.kw).fit(self.X_train)
        self.assertFalse(clf.encoder_norm_)
        self.assertEqual(clf.ema_momentum_, 0.99)
        self.assertTrue(np.isfinite(clf.decision_scores_).all())
        # the constructor arguments are stored as given
        self.assertIsNone(self.fitted['tsjepa'].encoder_norm)
        self.assertIsNone(self.fitted['tsjepa'].ema_momentum)

    def test_networks(self):
        self.assertIsNone(self.fitted['lejepa'].model_.target_encoder)
        for v in EMA_VARIANTS:
            m = self.fitted[v].model_
            self.assertFalse(any(p.requires_grad
                                 for p in m.target_encoder.parameters()))
        # the EMA covers the tokenizer for TS-JEPA and T-JEPA only
        self.assertFalse(hasattr(self.fitted['ijepa'].model_.target_encoder,
                                 'tokenizer'))
        for v in ('tsjepa', 'tjepa'):
            self.assertTrue(hasattr(self.fitted[v].model_.target_encoder,
                                    'tokenizer'))
        self.assertIsInstance(self.fitted['tsjepa'].model_.encoder.tokenizer,
                              ConvPatchTokenizer)
        self.assertEqual(self.fitted['tjepa'].model_.encoder.reg.shape[0], 1)

    def test_ema_target_follows_encoder(self):
        for v in EMA_VARIANTS:
            m = self.fitted[v].model_
            init = TSJEPAOD(variant=v, **dict(self.kw, epochs=0)).fit(
                self.X_train).model_
            moved = max((a - b).abs().max().item() for a, b in zip(
                m.target_encoder.parameters(),
                init.target_encoder.parameters()))
            lag = max((a - b).abs().max().item() for a, b in zip(
                m.target_encoder.parameters(), m.encoder.parameters()))
            self.assertGreater(moved, 0, v)   # the target was updated
            self.assertGreater(lag, 0, v)     # and is not the online copy

    def test_zero_epochs_keeps_initial_weights(self):
        for v in VARIANTS:
            clf = TSJEPAOD(variant=v, **dict(self.kw, epochs=0)).fit(
                self.X_train)
            torch.manual_seed(self.kw['random_state'])
            fresh = clf._build_model(clf.n_tokens_, clf.n_channels_)
            for a, b in zip(clf.model_.parameters(), fresh.parameters()):
                self.assertTrue(torch.equal(a, b), v)
            self.assertTrue(np.isfinite(clf.decision_scores_).all())

    def test_modes_and_paper_masking(self):
        for v in EMA_VARIANTS:
            for mode in ('forecast', 'random'):
                clf = TSJEPAOD(variant=v, **dict(self.kw, mask_mode=mode,
                                                 epochs=1)).fit(self.X_train)
                self.assertTrue(np.isfinite(
                    clf.decision_function(self.X_test)).all(), (v, mode))
        # the masking of the TS-JEPA paper: 70 % of the patches, at random
        clf = TSJEPAOD(variant='tsjepa', **dict(
            self.kw, mask_mode='random', mask_ratio=0.7, epochs=1,
            window_size=50)).fit(self.X_train)
        self.assertEqual((clf.n_tokens_, clf.n_mask_), (10, 7))
        self.assertTrue(np.isfinite(clf.decision_function(self.X_test)).all())

    def test_univariate(self):
        Xtr, Xte, _, _ = generate_ts_data(
            n_train=1000, n_test=400, n_channels=1, contamination=0.05,
            anomaly_type='point', random_state=0)
        for v in VARIANTS:
            clf = TSJEPAOD(variant=v, **dict(self.kw, epochs=1)).fit(Xtr)
            self.assertTrue(np.isfinite(clf.decision_function(Xte)).all(), v)

    def test_clone_and_pickle(self):
        for v, clf in self.fitted.items():
            c = clone(clf)
            self.assertEqual(c.get_params(), clf.get_params())
            self.assertEqual(c.variant, v)
            again = pickle.loads(pickle.dumps(clf))
            np.testing.assert_array_equal(
                again.decision_function(self.X_test),
                clf.decision_function(self.X_test))

    def test_conv_tokenizer_layout(self):
        # a patch arrives flattened time-major: [t0c0, t0c1, t1c0, ...]
        torch.manual_seed(0)
        tok = ConvPatchTokenizer(patch_len=6, n_channels=2, out_dim=8)
        x = torch.randn(4, 5, 12)
        out = tok(x)
        self.assertEqual(out.shape, (4, 5, 8))
        one = x[1, 3].view(6, 2).t().unsqueeze(0)          # (1, C, L)
        ref = tok.fc(tok.conv(one).flatten(1))[0]
        self.assertTrue(torch.allclose(out[1, 3], ref, atol=1e-6))
        # the kernel is capped at the patch length
        short = ConvPatchTokenizer(patch_len=2, n_channels=1, out_dim=8)
        self.assertEqual(short.conv.kernel_size, (2,))
        self.assertEqual(short(torch.randn(3, 4, 2)).shape, (3, 4, 8))


if __name__ == '__main__':
    unittest.main()
