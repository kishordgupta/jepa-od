"""TSB-AD metrics of the scores that benchmarks/bench_tsb.py wrote.

The nine measures TSB-AD reports (``TSB_AD.evaluation.metrics.get_metrics``): VUS-PR, which the benchmark ranks by,
VUS-ROC, AUC-PR (average precision), AUC-ROC, and the best F over thresholds in five forms (Standard-F1, PA-F1,
Event-based-F1, R-based-F1, Affiliation-F). They are computed the way TSB-AD's ``benchmark_exp/Run_Detector_M.py``
does: scores are min-max scaled, and the window of the VUS measures is ``find_length_rank`` of the first channel.

TSB-AD's functions loop in Python over thresholds, anomalies and timestamps: minutes for one long series, and for
some series hours. The measures are therefore taken from ``jepa_od/metrics.py``, which computes the same numbers
with numpy (``benchmarks/check_metrics_tsb.py`` compares the two; ``--slow`` uses TSB-AD's functions here).

``--validate`` runs detectors of TSB-AD itself on a few files and compares with the numbers TSB-AD published.

    python -I benchmarks/eval_tsb.py --tsb <TSB-AD clone> --data <folder with TSB-AD-M/ and TSB-AD-U/> \
        --results <results folder> --jobs 64

Output: <results>/metrics/<suite>-<split>/<tag>.csv, one row per file and score type; finished rows are kept.
The score files of benchmarks/bench_tsb_ref.py (one vector, named ``score``) are evaluated the same way.
"""
import argparse, csv, glob, importlib.util, os, sys, time, warnings
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
ap.add_argument('--slow', action='store_true', help="use TSB-AD's functions instead of jepa_od/metrics.py")
ap.add_argument('--module', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'jepa_od',
                                                 'metrics.py'), help='path of jepa_od/metrics.py')
args = ap.parse_args()
sys.path.insert(0, os.path.abspath(args.tsb))
from TSB_AD.evaluation.metrics import get_metrics
from TSB_AD.utils.slidingWindows import find_length_rank
# loaded by its path: importing the package would also load the detectors
spec = importlib.util.spec_from_file_location('jepa_od_metrics', os.path.abspath(args.module))
fast = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fast)

SCORES = ('pred', 'embed', 'combined')
FIELDS = ['file', 'score'] + list(fast.METRICS) + ['window', 'nonfinite', 'constant', 'seconds']


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
    """TSB-AD's nine measures of one score vector."""
    score, nonfinite, constant = scaled(score)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        out = get_metrics(score, label, slidingWindow=window) if args.slow else fast.ts_metrics(label, score, window)
    out.update(window=window, nonfinite=nonfinite, constant=constant)
    return out


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
