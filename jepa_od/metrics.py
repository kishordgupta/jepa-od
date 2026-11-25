# -*- coding: utf-8 -*-
"""Evaluation measures for time-series anomaly scores.

The nine measures that the TSB-AD benchmark (Liu and Paparrizos, NeurIPS
2024) reports for a score vector, written with numpy so that a series of
a million timestamps takes seconds:

==================  ====================================================
``'AUC-ROC'``       area under the ROC curve
``'AUC-PR'``        average precision
``'VUS-ROC'``       volume under the ROC surface (Paparrizos et al.,
                    2022; Boniol et al., 2025): range-AUC with the
                    labels extended by ``0 .. window`` timestamps,
                    averaged over the extensions
``'VUS-PR'``        the same for the precision-recall curve; the measure
                    TSB-AD ranks by
``'Standard-F1'``   best point-wise F1 over all thresholds
``'PA-F1'``         best F1 after point adjustment: an anomaly counts as
                    found in full once one of its timestamps is flagged
``'Event-based-F1'``  best F1 of event recall and point precision
``'R-based-F1'``    best range-based F1 (Tatbul et al., 2018)
``'Affiliation-F'``  best affiliation F1 (Huet et al., 2022)
==================  ====================================================

The first four need no threshold. The other five are the best value over
100 thresholds spread evenly between the lowest and the highest score,
that is, with a threshold chosen with the labels, as TSB-AD reports them.

The definitions, including their edge cases, follow TSB-AD's
``TSB_AD.evaluation`` so that numbers can be compared with that
benchmark. ``benchmarks/check_metrics_tsb.py`` compares every function
with TSB-AD's own on random and on real score vectors; the notes below
name the edge cases that are TSB-AD's and not part of the measure's
definition.

    >>> from jepa_od.metrics import ts_metrics
    >>> ts_metrics(labels, scores, window=acf_window(series))  # doctest: +SKIP
"""
# License: BSD 2 clause

import numpy as np
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_auc_score)

__all__ = ['METRICS', 'ts_metrics', 'auc_roc', 'auc_pr', 'vus',
           'standard_f1', 'pa_f1', 'event_f1', 'range_f1', 'affiliation_f1',
           'acf_window']

METRICS = ('VUS-PR', 'VUS-ROC', 'AUC-PR', 'AUC-ROC', 'Standard-F1', 'PA-F1',
           'Event-based-F1', 'R-based-F1', 'Affiliation-F')


def _check(label, score):
    label = np.asarray(label).astype(int).ravel()
    score = np.asarray(score, dtype=np.float64).ravel()
    if label.shape != score.shape:
        raise ValueError('label and score must have the same length')
    return label, score


def _runs(binary):
    """First and last index of every run of ones."""
    b = np.asarray(binary).astype(np.int8)
    edge = np.diff(np.concatenate([[0], b, [0]]))
    return np.flatnonzero(edge == 1), np.flatnonzero(edge == -1) - 1


def _run_max(score, starts, ends):
    """Highest score in every run ``starts[i] .. ends[i]``; -inf for an
    empty run (``ends[i] < starts[i]``)."""
    out = np.full(len(starts), -np.inf)
    ok = ends >= starts
    if ok.any():
        bounds = np.stack([starts[ok], ends[ok] + 1], axis=1).ravel()
        out[ok] = np.maximum.reduceat(score, bounds[bounds < len(score)])[::2]
    return out


def _thresholds(score, n_thresholds):
    return np.linspace(score.min(), score.max(), n_thresholds)


def _count_above(sorted_values, thresholds):
    """Number of values strictly above every threshold."""
    return len(sorted_values) - np.searchsorted(sorted_values, thresholds,
                                                side='right')


def _best(values):
    """What Python's ``max`` gives for a list: NaN only if the first
    value is NaN (TSB-AD takes the best threshold that way)."""
    values = np.asarray(values, dtype=float)
    if np.isnan(values[0]):
        return float('nan')
    return float(np.nanmax(values))


# ----------------------------------------------------------------------
# threshold-free measures
# ----------------------------------------------------------------------
def auc_roc(label, score):
    """Area under the ROC curve."""
    label, score = _check(label, score)
    return float(roc_auc_score(label, score))


def auc_pr(label, score):
    """Average precision."""
    label, score = _check(label, score)
    return float(average_precision_score(label, score))


def _extend(label, starts, ends, window):
    """Labels with a square-root ramp of ``window // 2`` timestamps on
    both sides of every anomaly, capped at one."""
    out = label.astype(np.float64)
    half, n = window // 2, len(label)
    if half:
        d = np.arange(1, half + 1)
        ramp = np.sqrt(1.0 - d / window)
        after = ends[:, None] + d
        np.add.at(out, after[after < n], np.broadcast_to(ramp, after.shape)[after < n])
        before = starts[:, None] - d
        np.add.at(out, before[before >= 0], np.broadcast_to(ramp, before.shape)[before >= 0])
    return np.minimum(1.0, out)


def _extended_runs(starts, ends, window, n):
    """The anomalies widened by ``window // 2`` on both sides, merged
    where they touch."""
    half = window // 2
    gap = ends[:-1] + half < starts[1:] - half
    first = np.concatenate([[max(starts[0] - half, 0)],
                            starts[1:][gap] - half])
    last = np.concatenate([ends[:-1][gap] + half,
                           [min(ends[-1] + half, n - 1)]])
    return first, last


def vus(label, score, window=100, n_thresholds=250):
    """VUS-ROC and VUS-PR.

    Parameters
    ----------
    label : array of {0, 1}
    score : array of float, higher = more anomalous
    window : int
        Largest label extension; TSB-AD uses the dominant period of the
        series (:func:`acf_window`).
    n_thresholds : int
        Thresholds, taken at evenly spaced ranks of the scores.

    Returns
    -------
    vus_roc, vus_pr : float
    """
    label, score = _check(label, score)
    n, P = len(score), int(label.sum())
    starts, ends = _runs(label)
    if P == 0 or P == n:
        raise ValueError('the labels must hold both classes')
    order = np.argsort(-score, kind='stable')
    ranked = score[order]
    thresholds = ranked[np.linspace(0, n - 1, n_thresholds).astype(int)]
    n_pred = n - np.searchsorted(ranked[::-1], thresholds, side='left')
    inside = np.cumsum(label[order])[n_pred - 1]
    window = int(window)
    auc, ap = np.zeros(window + 1), np.zeros(window + 1)
    for w in range(window + 1):
        extended = _extend(label, starts, ends, w)
        first, last = _extended_runs(starts, ends, w, n)
        highest = np.sort(_run_max(score, first, last))
        existence = len(highest) - np.searchsorted(highest, thresholds,
                                                   side='left')
        TP = np.cumsum(extended[order])[n_pred - 1]
        P_new = (P + TP - inside + P) / 2
        tpr = np.minimum(TP / P_new, 1) * existence / len(first)
        fpr = (n_pred - TP) / (n - P_new)
        tpr = np.concatenate([[0.0], tpr, [1.0]])
        fpr = np.concatenate([[0.0], fpr, [1.0]])
        precision = np.concatenate([[1.0], TP / n_pred])
        auc[w] = np.dot(fpr[1:] - fpr[:-1], (tpr[1:] + tpr[:-1]) / 2)
        ap[w] = np.dot(tpr[1:-1] - tpr[:-2], precision[1:])
    return float(auc.mean()), float(ap.mean())


# ----------------------------------------------------------------------
# best F over thresholds
# ----------------------------------------------------------------------
def standard_f1(label, score):
    """Best point-wise F1 over all thresholds (every distinct score)."""
    label, score = _check(label, score)
    precision, recall, _ = precision_recall_curve(label, score)
    return float(np.max(2 * precision * recall
                        / (precision + recall + 0.00001)))


def pa_f1(label, score, n_thresholds=100):
    """Best F1 after point adjustment.

    For a threshold, every anomaly with at least one flagged timestamp
    counts as flagged in full; false positives are counted point-wise.
    (As in TSB-AD, the first timestamp of the series is not filled in
    when an anomaly starts there.)
    """
    label, score = _check(label, score)
    starts, ends = _runs(label)
    if not len(starts):
        return 0.0
    thr = _thresholds(score, n_thresholds)
    found = _run_max(score, starts, ends)[None, :] > thr[:, None]
    tp = (found * (ends - starts + 1)).sum(1)
    if starts[0] == 0:
        tp = tp - (found[:, 0] & (score[0] <= thr))
    fp = _count_above(np.sort(score[label == 0]), thr)
    fn = label.sum() - tp
    f1 = np.where(tp > 0, 2.0 * tp / np.maximum(2 * tp + fp + fn, 1), 0.0)
    return _best(f1)


def event_f1(label, score, n_thresholds=100):
    """Best F1 of event recall (share of anomalies with a flagged
    timestamp) and point-wise precision.

    (As in TSB-AD, an anomaly that reaches the end of the series is
    looked at without its last timestamp.)
    """
    label, score = _check(label, score)
    starts, ends = _runs(label)
    if not len(starts):
        return float('nan')
    ends = ends.copy()
    if ends[-1] == len(label) - 1:
        ends[-1] -= 1
    thr = _thresholds(score, n_thresholds)
    found = _run_max(score, starts, ends)[None, :] > thr[:, None]
    recall = found.sum(1) / len(starts)
    tp = _count_above(np.sort(score[label == 1]), thr)
    fp = _count_above(np.sort(score[label == 0]), thr)
    precision = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), 0.0)
    return _best(2 * recall * precision / (recall + precision + 1e-15))


def _range_recall(real_starts, real_ends, pred, pred_cum, pred_start_cum,
                  alpha):
    """Range-based recall of Tatbul et al. with a flat positional bias
    and the reciprocal cardinality factor: for every real range,
    ``alpha`` for being hit at all plus ``1 - alpha`` times the share of
    its timestamps that are flagged, divided by the number of predicted
    ranges that overlap it."""
    if not len(real_starts):
        return 0.0
    inside = pred_cum[real_ends + 1] - pred_cum[real_starts]
    card = (pred_start_cum[real_ends + 1] - pred_start_cum[real_starts + 1]
            + pred[real_starts])
    share = inside / (real_ends - real_starts + 1)
    overlap = np.where(card > 0, share / np.maximum(card, 1), 0.0).sum()
    existence = np.count_nonzero(inside)
    return (alpha * existence + (1 - alpha) * overlap) / len(real_starts)


def _cums(binary):
    """Cumulative count of ones and of run starts, both with a leading 0."""
    start = binary & ~np.concatenate([[False], binary[:-1]])
    return (np.concatenate([[0], np.cumsum(binary)]),
            np.concatenate([[0], np.cumsum(start)]), start)


def range_f1(label, score, n_thresholds=100, alpha=0.2):
    """Best range-based F1: range recall with existence weight ``alpha``
    and range precision (the same formula with the roles swapped and no
    existence term)."""
    label, score = _check(label, score)
    real = label.astype(bool)
    real_cum, real_start_cum, _ = _cums(real)
    real_starts, real_ends = _runs(label)
    out = []
    for t in _thresholds(score, n_thresholds):
        pred = score > t
        pred_cum, pred_start_cum, pred_start = _cums(pred)
        recall = _range_recall(real_starts, real_ends, pred.astype(int),
                               pred_cum, pred_start_cum, alpha)
        pred_starts = np.flatnonzero(pred_start)
        pred_ends = np.flatnonzero(
            pred & ~np.concatenate([pred[1:], [False]]))
        precision = _range_recall(pred_starts, pred_ends, label, real_cum,
                                  real_start_cum, 0.0)
        out.append(0.0 if precision + recall == 0
                   else 2 * recall * precision / (precision + recall))
    return _best(out)


# ----------------------------------------------------------------------
# affiliation
# ----------------------------------------------------------------------
def _affiliation(gs, ge, bounds, ps, pe):
    """Affiliation precision and recall (Huet et al., KDD 2022).

    ``gs, ge``: the real events as half-open intervals; ``bounds``: the
    edges of their affiliation zones (the midpoints between consecutive
    events, and the ends of the series); ``ps, pe``: the predicted events.
    """
    G = len(gs)
    if not len(ps):
        return float('nan'), 0.0
    # the pieces of the predicted events, one per zone they reach into
    j0 = np.searchsorted(bounds, ps, side='right') - 1
    j1 = np.searchsorted(bounds, pe, side='left') - 1
    count = j1 - j0 + 1
    event = np.repeat(np.arange(len(ps)), count)
    zone = np.repeat(j0, count) + np.arange(count.sum()) \
        - np.repeat(np.cumsum(count) - count, count)
    a = np.maximum(ps[event], bounds[zone]).astype(np.float64)
    b = np.minimum(pe[event], bounds[zone + 1]).astype(np.float64)
    jmin, jmax = gs[zone].astype(np.float64), ge[zone].astype(np.float64)
    emin, emax = bounds[zone], bounds[zone + 1]
    dE = emax - emin

    # precision: for a point of a piece, the probability that a random
    # point of the zone is at least as far from the event
    m = np.minimum(jmin - emin, emax - jmax)

    def outside(lo, hi):
        dmin = np.maximum(lo - jmax, jmin - hi)
        dmax = np.maximum(hi - jmax, jmin - lo)
        A = np.minimum(dmax, m) ** 2 - np.minimum(dmin, m) ** 2
        B = np.maximum(dmax, m) - np.maximum(dmin, m)
        return (hi - lo) - (0.5 * A + m * B + 0.5 * (dmax ** 2 - dmin ** 2)
                            + (jmax - jmin) * (hi - lo)) / dE

    part = (np.where(a < jmin, outside(a, np.minimum(b, jmin)), 0.0)
            + np.clip(np.minimum(b, jmax) - np.maximum(a, jmin), 0.0, None)
            + np.where(b > jmax, outside(np.maximum(a, jmax), b), 0.0))
    total = np.bincount(zone, weights=part, minlength=G)
    length = np.bincount(zone, weights=b - a, minlength=G)
    hit = length > 0
    precision = float((total[hit] / length[hit]).mean())

    # recall: every piece answers for the part of the event that is
    # closer to it than to the other pieces of the zone
    same_prev = np.concatenate([[False], zone[1:] == zone[:-1]])
    same_next = np.concatenate([zone[1:] == zone[:-1], [False]])
    left = np.where(same_prev, 0.5 * (np.roll(b, 1) + a), emin)
    right = np.where(same_next, 0.5 * (b + np.roll(a, -1)), emax)
    u, v = np.maximum(jmin, left), np.minimum(jmax, right)
    emean = 0.5 * (emin + emax)

    def cut(lo, hi, pivot):
        hi = np.maximum(lo, hi)
        c = np.clip(emean, lo, hi)
        m1 = np.clip(0.5 * (emin + pivot), lo, c)
        m2 = np.clip(0.5 * (emax + pivot), c, hi)
        return lo, m1, c, m2, hi

    lo, m1, c, m2, hi = cut(u, np.minimum(v, a), a)       # left of the piece
    before = (hi - lo) - (
        (a - emin) * (m1 - lo)
        + 2 * a * (c - m1) - (c ** 2 - m1 ** 2)
        + 2 * a * (m2 - c) - (m2 ** 2 - c ** 2)
        + (emax + a) * (hi - m2) - (hi ** 2 - m2 ** 2)) / dE
    lo, m1, c, m2, hi = cut(np.maximum(u, b), v, b)       # right of it
    after = (hi - lo) - (
        (m1 ** 2 - lo ** 2) - (emin + b) * (m1 - lo)
        + (c ** 2 - m1 ** 2) - 2 * b * (c - m1)
        + (m2 ** 2 - c ** 2) - 2 * b * (m2 - c)
        + (emax - b) * (hi - m2)) / dE
    part = np.where(u < v, before + after + np.clip(
        np.minimum(v, b) - np.maximum(u, a), 0.0, None), 0.0)
    recall = float((np.bincount(zone, weights=part, minlength=G)
                    / (ge - gs)).mean())
    return precision, recall


def affiliation_f1(label, score, n_thresholds=100):
    """Best affiliation F1.

    Events are half-open intervals of timestamps. Every real event owns
    the zone of the series that is closer to it than to any other real
    event; precision and recall are averaged over the zones (precision
    over the zones with a prediction).
    """
    label, score = _check(label, score)
    starts, ends = _runs(label)
    if not len(starts):
        raise ValueError('the labels hold no anomaly')
    gs, ge = starts, ends + 1
    bounds = np.concatenate([[0.0], 0.5 * (ge[:-1] + gs[1:]),
                             [float(len(label))]])
    out = []
    for t in _thresholds(score, n_thresholds):
        ps, pe = _runs(score > t)
        precision, recall = _affiliation(gs, ge, bounds, ps, pe + 1)
        out.append(2 * precision * recall / (precision + recall + 1e-15))
    return _best(out)


# ----------------------------------------------------------------------
def acf_window(x, max_lag=400, base=3):
    """Dominant period of a series, the label extension TSB-AD uses for
    the VUS measures: the lag (from ``base`` on) of the highest local
    maximum of the autocorrelation of the first 20,000 values; 125 if it
    is above 300 or if there is no local maximum.
    """
    x = np.asarray(x, dtype=np.float64).squeeze()
    if x.ndim > 1:
        return 0
    x = x[:20000]
    n = len(x)
    d = x - x.mean()
    size = 1 << int(np.ceil(np.log2(2 * n - 1)))
    f = np.fft.rfft(d, size)
    cov = np.fft.irfft(f * np.conj(f), size)[:max_lag + 1]
    if cov[0] <= 0:
        return 125
    acf = (cov / cov[0])[base:]
    peaks = np.flatnonzero((acf[1:-1] > acf[:-2]) & (acf[1:-1] > acf[2:])) + 1
    if not len(peaks):
        return 125
    period = int(peaks[np.argsort(acf[peaks])[::-1][0]] + base)
    return 125 if period > 300 else period


def ts_metrics(label, score, window=100, metrics=METRICS):
    """All measures of one score vector.

    Parameters
    ----------
    label : array of {0, 1} of shape (n_timestamps,)
    score : array of float of shape (n_timestamps,)
        Higher = more anomalous. TSB-AD scales scores to [0, 1] first;
        that changes none of the measures.
    window : int, optional (default=100)
        Label extension of the VUS measures, see :func:`acf_window`.
    metrics : sequence of str, optional
        Names from :data:`METRICS`.

    Returns
    -------
    dict : measure name -> float
    """
    label, score = _check(label, score)
    unknown = sorted(set(metrics) - set(METRICS))
    if unknown:
        raise ValueError('unknown measures %s; known: %s' % (unknown, METRICS))
    out = {}
    if 'VUS-PR' in metrics or 'VUS-ROC' in metrics:
        out['VUS-ROC'], out['VUS-PR'] = vus(label, score, window)
    simple = {'AUC-PR': auc_pr, 'AUC-ROC': auc_roc,
              'Standard-F1': standard_f1, 'PA-F1': pa_f1,
              'Event-based-F1': event_f1, 'R-based-F1': range_f1,
              'Affiliation-F': affiliation_f1}
    for name, func in simple.items():
        if name in metrics:
            out[name] = func(label, score)
    return {name: out[name] for name in metrics}
