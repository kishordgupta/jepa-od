# -*- coding: utf-8 -*-
"""Unit tests for jepa_od.metrics."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jepa_od import metrics
from jepa_od.metrics import (METRICS, ts_metrics, vus, standard_f1, pa_f1,
                             event_f1, range_f1, affiliation_f1, acf_window,
                             _affiliation, _runs)

FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fixtures',
                       'tsb_metrics.npz')


class TestMetrics(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(0)
        self.n = 3000
        self.label = np.zeros(self.n, dtype=int)
        for start, length in ((200, 40), (900, 1), (1500, 120), (2600, 15)):
            self.label[start:start + length] = 1
        self.score = rng.rand(self.n) + 0.7 * self.label * rng.rand(self.n)

    def test_against_tsb_ad(self):
        # values computed with TSB-AD's own code (commit 303dffb) by
        # benchmarks/check_metrics_tsb.py --fixture
        data = np.load(FIXTURE)
        self.assertEqual(tuple(data['names']), METRICS)
        self.assertGreaterEqual(int(data['n']), 30)
        for j in range(int(data['n'])):
            out = ts_metrics(data['label%d' % j], data['score%d' % j],
                             window=int(data['window%d' % j]))
            np.testing.assert_allclose([out[name] for name in METRICS],
                                       data['ref%d' % j], rtol=0, atol=1e-10,
                                       err_msg='case %d' % j)

    def test_all_measures(self):
        out = ts_metrics(self.label, self.score, window=30)
        self.assertEqual(tuple(out), METRICS)
        for name, value in out.items():
            self.assertTrue(0.0 <= value <= 1.0 + 1e-12, name)
        self.assertGreater(out['AUC-ROC'], 0.6)
        # point adjustment can only help; so can the label extension here
        self.assertGreaterEqual(out['PA-F1'], out['Standard-F1'] - 0.02)
        some = ts_metrics(self.label, self.score, metrics=('AUC-PR', 'PA-F1'))
        self.assertEqual(some, {k: out[k] for k in ('AUC-PR', 'PA-F1')})
        with self.assertRaises(ValueError):
            ts_metrics(self.label, self.score, metrics=('F1',))
        with self.assertRaises(ValueError):
            ts_metrics(self.label, self.score[:-1])

    def test_perfect_and_inverted_scores(self):
        good = ts_metrics(self.label, self.label.astype(float), window=10)
        for name in ('AUC-PR', 'AUC-ROC', 'PA-F1', 'Event-based-F1',
                     'R-based-F1', 'Affiliation-F'):
            self.assertAlmostEqual(good[name], 1.0, places=9, msg=name)
        self.assertGreater(good['Standard-F1'], 0.9999)
        self.assertGreater(good['VUS-ROC'], 0.97)
        bad = ts_metrics(self.label, 1.0 - self.label, window=10)
        self.assertLess(bad['AUC-ROC'], 1e-9)
        self.assertLess(bad['VUS-PR'], good['VUS-PR'])

    def test_scale_invariance(self):
        a = ts_metrics(self.label, self.score, window=20)
        b = ts_metrics(self.label, 4.0 * self.score, window=20)
        for name in METRICS:
            self.assertAlmostEqual(a[name], b[name], places=9, msg=name)

    def test_vus_window(self):
        roc0, pr0 = vus(self.label, self.score, window=0)
        self.assertAlmostEqual(roc0, metrics.auc_roc(self.label, self.score),
                               delta=0.02)
        # a score that peaks two timestamps after every one-point anomaly
        # gains from the label extension
        label = np.zeros(self.n, dtype=int)
        label[200::400] = 1
        late = 1e-3 * self.score
        late[202::400] = 1.0
        self.assertGreater(vus(label, late, window=20)[1],
                           vus(label, late, window=0)[1] + 0.05)
        with self.assertRaises(ValueError):
            vus(np.zeros(50, dtype=int), np.arange(50.0))

    def test_point_adjustment(self):
        label = np.array([0, 0, 1, 1, 1, 1, 0, 0, 0, 0])
        score = np.array([0, 0, 0, 9., 0, 0, 0, 0, 0, 0])   # one hit of four
        self.assertAlmostEqual(pa_f1(label, score), 1.0)
        # point-wise, flagging everything is best: precision 0.4, recall 1
        self.assertAlmostEqual(standard_f1(label, score), 0.8 / 1.4, places=4)
        self.assertAlmostEqual(event_f1(label, score), 1.0, places=9)
        # a false alarm of the same height halves the precision
        score[8] = 9.0
        self.assertAlmostEqual(pa_f1(label, score), 2 * 4 / (2 * 4 + 1))
        self.assertAlmostEqual(event_f1(label, score), 2 * 0.5 / 1.5,
                               places=9)

    def test_range_f1_by_hand(self):
        # one real range of four timestamps, half of it flagged by one
        # predicted range that also covers two normal timestamps
        label = np.array([0, 0, 1, 1, 1, 1, 0, 0, 0, 0])
        score = np.array([0, 0, 0, 0, 1., 1, 1, 1, 0, 0])
        recall = 0.2 * 1 + 0.8 * (2 / 4)
        precision = 2 / 4
        self.assertAlmostEqual(range_f1(label, score),
                               2 * recall * precision / (recall + precision))
        # the same hit split into two predicted ranges halves the overlap
        split = np.array([0, 0, 1., 0, 1, 0, 0, 0, 0, 0])
        recall = 0.2 * 1 + 0.8 * (2 / 4) / 2
        self.assertAlmostEqual(range_f1(label, split),
                               2 * recall * 1.0 / (recall + 1.0))

    def test_affiliation(self):
        label = np.zeros(100, dtype=int)
        label[40:50] = 1
        gs, ge = np.array([40]), np.array([50])
        bounds = np.array([0.0, 100.0])
        # the event itself: precision and recall are one
        p, r = _affiliation(gs, ge, bounds, np.array([40]), np.array([50]))
        self.assertAlmostEqual(p, 1.0)
        self.assertAlmostEqual(r, 1.0)
        # no prediction: no precision, recall zero
        p, r = _affiliation(gs, ge, bounds, np.array([], dtype=int),
                            np.array([], dtype=int))
        self.assertTrue(np.isnan(p))
        self.assertEqual(r, 0.0)
        # a prediction next to the event beats one far from it
        near = _affiliation(gs, ge, bounds, np.array([50]), np.array([55]))
        far = _affiliation(gs, ge, bounds, np.array([90]), np.array([95]))
        self.assertGreater(near[0], far[0])
        self.assertGreater(near[1], far[1])
        self.assertTrue(all(0.0 <= v <= 1.0 for v in near + far))
        # a prediction over the whole zone: recall one, precision is the
        # mean survival probability
        p, r = _affiliation(gs, ge, bounds, np.array([0]), np.array([100]))
        self.assertAlmostEqual(r, 1.0)
        self.assertTrue(0.3 < p < 0.8)
        score = np.zeros(100)
        score[48:53] = 1.0
        self.assertTrue(0.8 < affiliation_f1(label, score) < 1.0)
        with self.assertRaises(ValueError):
            affiliation_f1(np.zeros(20, dtype=int), np.arange(20.0))

    def test_runs(self):
        starts, ends = _runs(np.array([1, 1, 0, 0, 1, 0, 1]))
        self.assertEqual(starts.tolist(), [0, 4, 6])
        self.assertEqual(ends.tolist(), [1, 4, 6])
        self.assertEqual(len(_runs(np.zeros(5, dtype=int))[0]), 0)

    def test_acf_window(self):
        rng = np.random.RandomState(1)
        t = np.arange(6000)
        self.assertEqual(acf_window(np.sin(2 * np.pi * t / 48)
                                    + 0.1 * rng.randn(6000)), 48)
        self.assertEqual(acf_window(np.ones(500)), 125)       # constant
        self.assertEqual(acf_window(np.sin(2 * np.pi * t / 700)), 125)
        self.assertEqual(acf_window(rng.randn(100, 3)), 0)    # not one series


if __name__ == '__main__':
    unittest.main()
