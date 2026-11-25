"""TSB-AD metrics of the scores that benchmarks/bench_tsb.py wrote.

The measures are TSB-AD's (``TSB_AD.evaluation`` of the clone given with ``--tsb``), computed the way its
``benchmark_exp/Run_Detector_M.py`` does: scores are min-max scaled, and the window of the VUS measures is
``find_length_rank`` of the first channel. VUS-PR is the measure the benchmark ranks by; VUS-ROC, AUC-PR (average
precision) and AUC-ROC are written too.

TSB-AD's VUS function (``basic_metricor.RangeAUC_volume_opt``) loops in Python over 250 thresholds and over every
anomaly for each of up to 300 windows: minutes for one series with a thousand anomalies. ``vus`` below computes the
same two numbers from one sort and a cumulative sum per window. It uses TSB-AD's own helpers for the label
extension, and ``--check-vus N`` compares it with TSB-AD's function on N of the saved score vectors.

``--validate`` runs detectors of TSB-AD itself on a few files and compares with the numbers TSB-AD published.

    python -I benchmarks/eval_tsb.py --tsb <TSB-AD clone> --data <folder with TSB-AD-M/ and TSB-AD-U/> \
        --results <results folder> --jobs 64

Output: <results>/metrics/<suite>-<split>/<tag>.csv, one row per file and score type; finished rows are kept.
The score files of benchmarks/bench_tsb_ref.py (one vector, named ``score``) are evaluated the same way.
"""
import argparse, csv, glob, os, sys, time
from multiprocessing import Pool
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument('--tsb', required=True, help='clone of https://github.com/TheDatumOrg/TSB-AD')
ap.add_argument('--data', required=True)
ap.add_argument('--results', required=True)
ap.add_argument('--tags', default='*', help='pattern of the score folders to evaluate')
ap.add_argument('--jobs', type=int, default=8)
ap.add_argument('--validate', default=None, help='M or U: run TSB-AD detectors and compare with its published table')
ap.add_argument('--n-files', type=int, default=12)
ap.add_argument('--check-vus', type=int, default=0, help="compare vus() with TSB-AD's function on this many scores")
ap.add_argument('--slow', action='store_true', help="use TSB-AD's VUS function instead of vus()")
args = ap.parse_args()
sys.path.insert(0, os.path.abspath(args.tsb))
from TSB_AD.evaluation.basic_metrics import basic_metricor, generate_curve
from TSB_AD.utils.slidingWindows import find_length_rank

SCORES = ('pred', 'embed', 'combined')
FIELDS = ['file', 'score', 'VUS-PR', 'VUS-ROC', 'AUC-PR', 'AUC-ROC', 'window', 'nonfinite', 'constant', 'seconds']


def series(suite, name):
    """Data, labels and VUS window of one file, as Run_Detector_M.py reads them."""
    df = pd.read_csv(os.path.join(args.data, f'TSB-AD-{suite}', name)).dropna()
    data = df.iloc[:, :-1].values.astype(float)
    return data, df['Label'].astype(int).to_numpy(), int(find_length_rank(data[:, 0].reshape(-1, 1), rank=1))


def labels(suite, stem):
    """Labels and VUS window, cached: reading the largest files takes longer than the measures."""
    path = os.path.join(args.results, 'cache', suite, stem + '.npz')
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        _, label, window = series(suite, stem + '.csv')
        tmp = path[:-4] + f'.{os.getpid()}.tmp.npz'
        np.savez(tmp, label=label.astype(np.int8), window=window)
        os.replace(tmp, path)
    cached = np.load(path)
    return cached['label'].astype(int), int(cached['window'])


def vus(label, score, window_size, thre=250):
    """VUS-ROC and VUS-PR, equal to ``generate_curve(label, score, window_size, 'opt', thre)[-2:]``.

    For a threshold, TSB-AD's loop multiplies the extended label by the prediction inside the extended anomalies,
    sets it to one inside the anomalies, and sums over the anomalies extended by the full window. With e the
    extended label (one inside an anomaly, zero outside the extended ones) and p the prediction, that is

        TP = sum(e * p),   N_labels = TP - sum(label * p) + P,   existence = extended anomalies with any p.

    The predictions of the thresholds are nested (the highest scores first), so one cumulative sum over the
    timestamps in the order of the scores gives TP for every threshold, and an extended anomaly exists for a
    threshold when its highest score reaches it.
    """
    grader = basic_metricor()
    label = np.asarray(label).astype(int)
    n, P = len(score), label.sum()
    seq = grader.range_convers_new(label)
    order = np.argsort(-score, kind='stable')
    ascending = score[order][::-1]
    thresholds = score[order][np.linspace(0, n - 1, thre).astype(int)]
    n_pred = n - np.searchsorted(ascending, thresholds, side='left')    # sum(score >= threshold)
    inside = np.cumsum(label[order])[n_pred - 1]                        # sum(label * p)
    auc, ap = np.zeros(window_size + 1), np.zeros(window_size + 1)
    for window in range(window_size + 1):
        extended = grader.sequencing(label, seq, window)
        segments = np.asarray(grader.new_sequence(extended, seq, window))
        bounds = np.stack([segments[:, 0], segments[:, 1] + 1], axis=1).ravel()
        highest = np.maximum.reduceat(score, bounds[bounds < n])[::2]   # highest score of every extended anomaly
        existence = len(highest) - np.searchsorted(np.sort(highest), thresholds, side='left')
        TP = np.cumsum(extended[order])[n_pred - 1]
        P_new = (P + TP - inside + P) / 2
        tpr = np.minimum(TP / P_new, 1) * existence / len(segments)
        fpr = (n_pred - TP) / (n - P_new)
        tpr, fpr = np.concatenate([[0.0], tpr, [1.0]]), np.concatenate([[0.0], fpr, [1.0]])
        precision = np.concatenate([[1.0], TP / n_pred])
        auc[window] = np.dot(fpr[1:] - fpr[:-1], (tpr[1:] + tpr[:-1]) / 2)
        ap[window] = np.dot(tpr[1:-1] - tpr[:-2], precision[1:])
    return auc.sum() / (window_size + 1), ap.sum() / (window_size + 1)


def scaled(score):
    """Min-max scaled score with the non-finite values replaced; also their number and whether it is constant."""
    score = np.asarray(score, dtype=np.float64).copy()
    finite = np.isfinite(score)
    nonfinite = int((~finite).sum())
    if nonfinite:  # +-inf to the ends of the finite range, NaN to the median
        low, high = (score[finite].min(), score[finite].max()) if finite.any() else (0.0, 0.0)
        score = np.nan_to_num(score, nan=float(np.median(score[finite])) if finite.any() else 0.0,
                              posinf=high, neginf=low)
    low, high = score.min(), score.max()
    return (score - low) / (high - low) if high > low else np.zeros_like(score), nonfinite, int(high <= low)


def names(saved):
    """The score vectors of one file: the three score types, or whatever a reference detector saved."""
    return [name for name in SCORES if name in saved.files] or list(saved.files)


def measures(score, label, window):
    """TSB-AD's threshold-free measures of one score vector."""
    score, nonfinite, constant = scaled(score)
    grader = basic_metricor()
    out = {'AUC-ROC': grader.metric_ROC(label, score), 'AUC-PR': grader.metric_PR(label, score)}
    if args.slow:
        out['VUS-ROC'], out['VUS-PR'] = generate_curve(label.astype(int), score, window, 'opt', 250)[-2:]
    else:
        out['VUS-ROC'], out['VUS-PR'] = vus(label, score, window)
    out.update(window=window, nonfinite=nonfinite, constant=constant)
    return out


def check_vus(job):
    """Largest difference between vus() and TSB-AD's function over the three scores of one file."""
    group, tag, stem = job
    label, window = labels(group.split('-')[0], stem)
    saved = np.load(os.path.join(args.results, 'scores', group, tag, stem + '.npz'))
    worst, slow_s, fast_s = 0.0, 0.0, 0.0
    for name in names(saved):
        score = scaled(saved[name])[0]
        start = time.perf_counter()
        ref = generate_curve(label.astype(int), score, window, 'opt', 250)[-2:]
        slow_s += time.perf_counter() - start
        start = time.perf_counter()
        fast = vus(label, score, window)
        fast_s += time.perf_counter() - start
        worst = max(worst, abs(ref[0] - fast[0]), abs(ref[1] - fast[1]))
    return stem, len(label), window, worst, slow_s, fast_s


def task(job):
    group, tag, stem = job
    label, window = labels(group.split('-')[0], stem)
    saved = np.load(os.path.join(args.results, 'scores', group, tag, stem + '.npz'))
    rows = []
    for name in names(saved):
        start = time.perf_counter()
        row = dict(file=stem + '.csv', score=name)
        try:
            row.update(measures(saved[name], label, window))
        except Exception as exc:
            row['constant'] = repr(exc)[:200]
        row['seconds'] = round(time.perf_counter() - start, 2)
        rows.append(row)
    return group, tag, rows


def validate():
    """Detectors of TSB-AD on a spread of files against the published per-file VUS-PR."""
    from TSB_AD.model_wrapper import run_Unsupervise_AD
    from TSB_AD.HP_list import Optimal_Multi_algo_HP_dict, Optimal_Uni_algo_HP_dict
    from sklearn.preprocessing import MinMaxScaler
    suite = args.validate
    table = pd.read_csv(os.path.join(args.tsb, 'benchmark_exp', 'benchmark_eval_results',
                                     ('multi' if suite == 'M' else 'uni') + '_mergedTable_VUS-PR.csv'))
    table = table.sort_values('ts_len').iloc[np.linspace(0, len(table) * 0.6, args.n_files).astype(int)]
    params = Optimal_Multi_algo_HP_dict if suite == 'M' else Optimal_Uni_algo_HP_dict
    for column, model in (('PCA', 'PCA'), ('HBOS', 'HBOS')) if suite == 'M' else (('Sub-PCA', 'Sub_PCA'), ('SR', 'SR')):
        diffs = []
        for _, ref in table.iterrows():
            data, label, window = series(suite, ref['file'])
            out = run_Unsupervise_AD(model, data, **params[model])
            out = MinMaxScaler(feature_range=(0, 1)).fit_transform(np.asarray(out).reshape(-1, 1)).ravel()
            here = measures(out, label, window)['VUS-PR']
            diffs.append(abs(here - ref[column]))
            print(f'{model:8s} {ref["file"][:44]:44s} n={len(label):6d} window={window:3d} here {here:.4f} '
                  f'published {ref[column]:.4f}', flush=True)
        print(f'{model}: largest difference {max(diffs):.4f}, mean {np.mean(diffs):.4f} over {len(diffs)} files')


if __name__ == '__main__':
    if args.validate:
        validate()
        sys.exit()
    if args.check_vus:
        found = sorted(glob.glob(os.path.join(args.results, 'scores', '*', args.tags, '*.npz')))
        # the files with the most anomalies are the slow ones for TSB-AD's function: a spread of sizes instead
        found.sort(key=os.path.getsize)
        pick = [found[i] for i in np.linspace(0, len(found) * 0.9, args.check_vus).astype(int)]
        todo = [tuple(p[:-4].split(os.sep)[-3:]) for p in pick]
        with Pool(args.jobs) as pool:
            out = pool.map(check_vus, todo, chunksize=1)
        for stem, n, window, worst, slow_s, fast_s in out:
            print(f'{stem[:50]:50s} n={n:6d} window={window:3d} largest difference {worst:.1e}  '
                  f'TSB-AD {slow_s:7.1f} s, here {fast_s:5.2f} s', flush=True)
        print('largest difference over %d files: %.1e' % (len(out), max(o[3] for o in out)))
        sys.exit()
    jobs, done = [], {}
    for folder in sorted(glob.glob(os.path.join(args.results, 'scores', '*', args.tags))):
        group, tag = folder.split(os.sep)[-2:]
        path = os.path.join(args.results, 'metrics', group, tag + '.csv')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.exists(path):
            done[group, tag] = set(pd.read_csv(path)['file'])
        else:
            with open(path, 'w', newline='') as f:
                csv.writer(f).writerow(FIELDS)
        for npz in sorted(glob.glob(os.path.join(folder, '*.npz'))):
            stem = os.path.basename(npz)[:-4]
            if not stem.endswith('.tmp') and stem + '.csv' not in done.get((group, tag), ()):
                jobs.append((group, tag, stem))
    # longest series first, so that the pool ends evenly
    jobs.sort(key=lambda job: -os.path.getsize(os.path.join(args.results, 'scores', job[0], job[1], job[2] + '.npz')))
    print(len(jobs), 'series to evaluate', flush=True)
    with Pool(args.jobs) as pool:
        for i, (group, tag, rows) in enumerate(pool.imap_unordered(task, jobs)):
            with open(os.path.join(args.results, 'metrics', group, tag + '.csv'), 'a', newline='') as f:
                csv.DictWriter(f, FIELDS).writerows(rows)
            if (i + 1) % 500 == 0:
                print(i + 1, 'done', flush=True)
    print('ALL DONE')
