# -*- coding: utf-8 -*-
"""PyOD-style unit tests for JEPAOD (mirrors pyod/test/test_auto_encoder.py)."""
import os
import sys
import unittest

import numpy as np
from sklearn.base import clone
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyod.utils.data import generate_data
from jepa_od import JEPAOD


class TestJEPAOD(unittest.TestCase):
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
        self.clf = JEPAOD(epoch_num=5, contamination=self.contamination,
                          verbose=0)
        self.clf.fit(self.X_train)

    def test_parameters(self):
        for attr in ('decision_scores_', 'labels_', 'threshold_', '_mu',
                     '_sigma', 'model', 'err_mean_', 'err_std_',
                     'emb_mean_', 'emb_prec_'):
            self.assertHasAttr(self.clf, attr)
            self.assertIsNotNone(getattr(self.clf, attr))

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

    def test_prediction_labels(self):
        pred_labels = self.clf.predict(self.X_test)
        self.assertEqual(pred_labels.shape, self.y_test.shape)

    def test_prediction_proba(self):
        for method in ('linear', 'unify'):
            pred_proba = self.clf.predict_proba(self.X_test, method=method)
            self.assertInRange(pred_proba, 0, 1)

    def test_prediction_proba_parameter(self):
        with self.assertRaises(ValueError):
            self.clf.predict_proba(self.X_test, method='something')

    def test_prediction_labels_confidence(self):
        pred_labels, confidence = self.clf.predict(self.X_test,
                                                   return_confidence=True)
        self.assertEqual(pred_labels.shape, self.y_test.shape)
        self.assertEqual(confidence.shape, self.y_test.shape)
        self.assertInRange(confidence, 0, 1)

    def test_fit_predict(self):
        pred_labels = self.clf.fit_predict(self.X_train)
        self.assertEqual(pred_labels.shape, self.y_train.shape)

    def test_fit_predict_score(self):
        self.clf.fit_predict_score(self.X_test, self.y_test)
        self.clf.fit_predict_score(self.X_test, self.y_test,
                                   scoring='roc_auc_score')
        self.clf.fit_predict_score(self.X_test, self.y_test,
                                   scoring='prc_n_score')
        with self.assertRaises(NotImplementedError):
            self.clf.fit_predict_score(self.X_test, self.y_test,
                                       scoring='something')

    def test_predict_rank(self):
        pred_scores = self.clf.decision_function(self.X_test)
        pred_ranks = self.clf._predict_rank(self.X_test)
        from scipy.stats import rankdata
        from pyod.utils.utility import check_parameter
        np.testing.assert_allclose(rankdata(pred_ranks), rankdata(pred_scores),
                                   atol=3)
        check_parameter(pred_ranks.min(), -1, self.n_test)

    def test_model_clone(self):
        clone_clf = clone(self.clf)
        self.assertEqual(clone_clf.get_params(), self.clf.get_params())

    def test_score_variants(self):
        for score in ('embed', 'combined'):
            clf = JEPAOD(epoch_num=5, score=score, verbose=0).fit(self.X_train)
            s = clf.decision_function(self.X_test)
            self.assertEqual(s.shape[0], self.n_test)
            self.assertGreaterEqual(roc_auc_score(self.y_test, s),
                                    self.roc_floor)

    def test_ema_and_recon(self):
        clf = JEPAOD(epoch_num=5, target_mode='ema', recon_weight=0.1,
                     verbose=0).fit(self.X_train)
        s = clf.decision_function(self.X_test)
        self.assertGreaterEqual(roc_auc_score(self.y_test, s), self.roc_floor)

    def test_two_features(self):
        Xtr, Xte, ytr, yte = generate_data(n_train=600, n_test=200,
                                           n_features=2, random_state=0)
        clf = JEPAOD(epoch_num=5, verbose=0).fit(Xtr)
        self.assertGreaterEqual(roc_auc_score(yte, clf.decision_function(Xte)),
                                self.roc_floor)

    def test_feature_grouping(self):
        Xtr, Xte, ytr, yte = generate_data(n_train=600, n_test=200,
                                           n_features=150, random_state=0)
        clf = JEPAOD(epoch_num=3, max_tokens=32, verbose=0).fit(Xtr)
        self.assertEqual(clf.n_tokens_, 30)
        self.assertEqual(clf.token_dim_, 5)
        self.assertEqual(clf.decision_function(Xte).shape[0], 200)

    def test_invalid_params(self):
        with self.assertRaises(ValueError):
            JEPAOD(epoch_num=1, score='bad', verbose=0).fit(self.X_train)
        with self.assertRaises(ValueError):
            JEPAOD(epoch_num=1, mask_mode='bad', verbose=0).fit(self.X_train)
        with self.assertRaises(ValueError):
            JEPAOD(epoch_num=1, verbose=0).fit(self.X_train[:, :1])

    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()
