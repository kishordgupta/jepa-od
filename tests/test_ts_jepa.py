# -*- coding: utf-8 -*-
"""PyOD-style unit tests for TSJEPAOD (mirrors pyod/test/test_ts_lstm.py)."""
import os
import sys
import unittest

import numpy as np
from sklearn.base import clone
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyod.utils.data import generate_ts_data
from jepa_od import TSJEPAOD


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
                       target_mode='ema', score_aggregation='max',
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


if __name__ == '__main__':
    unittest.main()
