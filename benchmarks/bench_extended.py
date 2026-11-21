"""Extended tabular benchmark: all 47 ADBench classical datasets, 5 seeds,
JEPA family with zero-epoch (untrained) controls and the PyOD baselines.

Protocol of bench_tabular.py: datasets with more than --max_rows rows are
subsampled per seed, stratified 60/40 split per seed, detectors fitted on
the contaminated training split without labels, ROC-AUC and average
precision on the test split.

Each (dataset, seed) pair is one task. Tasks run in a pool of worker
processes with one thread each, so results do not depend on the number
of workers, and every task writes results/extended_parts/<dataset>__s<seed>.csv:
an interrupted run resumes where it stopped. make_extended_tables.py
merges the parts into results/extended_results.csv.

Needs the 16 datasets of download_data.py and the 31 of
``download_data.py --all-adbench``.

Usage: python benchmarks/bench_extended.py [--workers 13] [--seeds 5]
"""
import os
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import argparse, glob, inspect, sys, time, warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARTS = os.path.join(ROOT, 'results', 'extended_parts')
SCORES = ('pred', 'embed', 'combined')


def make(cls, **kw):
    sig = inspect.signature(cls.__init__).parameters
    if 'epoch_num' in kw and 'epoch_num' not in sig and 'epochs' in sig:
        kw['epochs'] = kw.pop('epoch_num')
    return cls(**{k: v for k, v in kw.items() if k in sig})


def baselines(seed, n_features, epochs):
    from pyod.models.iforest import IForest
    from pyod.models.knn import KNN
    from pyod.models.lof import LOF
    from pyod.models.ecod import ECOD
    from pyod.models.copod import COPOD
    from pyod.models.pca import PCA
    from pyod.models.hbos import HBOS
    from pyod.models.ocsvm import OCSVM
    from pyod.models.auto_encoder import AutoEncoder
    from pyod.models.deep_svdd import DeepSVDD
    from pyod.models.lunar import LUNAR
    from pyod.models.dif import DIF
    return {
        'IForest': IForest(random_state=seed),
        'KNN': KNN(),
        'LOF': LOF(),
        'ECOD': ECOD(),
        'COPOD': COPOD(),
        'PCA': PCA(),
        'HBOS': HBOS(),
        'OCSVM': OCSVM(),
        'AutoEncoder': make(AutoEncoder, epoch_num=epochs, verbose=0, random_state=seed),
        'DeepSVDD': make(DeepSVDD, n_features=n_features, epoch_num=epochs, verbose=0, random_state=seed),
        'LUNAR': make(LUNAR, verbose=0),
        'DIF': DIF(random_state=seed),
    }


def part_path(name, seed):
    return os.path.join(PARTS, f'{name}__s{seed}.csv')


def run_task(path, seed, max_rows, epochs):
    """One dataset and seed: every detector, one CSV part."""
    warnings.filterwarnings('ignore')
    import torch
    torch.set_num_threads(1)
    sys.path.insert(0, ROOT)
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, average_precision_score
    from pyod.models.knn import KNN
    from jepa_od import JEPAOD, TJEPAOD

    t_task = time.time()
    name = os.path.basename(path)[:-4]
    d = np.load(path, allow_pickle=False)
    X, y = d['X'], d['y'].astype(int)
    rng = np.random.RandomState(seed)
    if len(X) > max_rows:
        keep = rng.choice(len(X), max_rows, replace=False)
        X, y = X[keep], y[keep]
    X = X.astype(np.float64)
    rows, notes = [], []

    def add(model, auc, apr, t):
        rows.append(dict(dataset=name, model=model, seed=seed, auc=auc, ap=apr, time=t,
                         n_train=n_train, n_features=X.shape[1]))

    def metrics(s):
        return roc_auc_score(yte, s), average_precision_score(yte, s)

    n_train = 0
    try:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, stratify=y, random_state=seed)
        n_train = len(Xtr)
    except ValueError as e:      # too few outliers in the subsample for a stratified split
        add('SKIPPED', np.nan, np.nan, 0.0)
        pd.DataFrame(rows).to_csv(part_path(name, seed), index=False)
        return f'{name} seed={seed} SKIPPED: {e}'

    for mname, clf in baselines(seed, Xtr.shape[1], epochs).items():
        t0 = time.time()
        try:
            clf.fit(Xtr)
            auc, apr = metrics(clf.decision_function(Xte))
        except Exception as e:
            notes.append(f'{mname} failed: {e!r}'[:160]); auc = apr = np.nan
        add(mname, auc, apr, time.time() - t0)

    family = {
        'JEPAOD': lambda: JEPAOD(verbose=0, random_state=seed, epoch_num=epochs),
        'JEPAOD-untrained': lambda: JEPAOD(verbose=0, random_state=seed, epoch_num=0),
        'TJEPAOD': lambda: TJEPAOD(verbose=0, random_state=seed, epoch_num=epochs),
        'TJEPAOD-untrained': lambda: TJEPAOD(verbose=0, random_state=seed, epoch_num=0),
    }
    for tag, build in family.items():
        t0 = time.time()
        try:
            clf = build().fit(Xtr)
            tfit = time.time() - t0
            for sc in SCORES:
                clf.score = sc
                add(f'{tag}[{sc}]', *metrics(clf.decision_function(Xte)), tfit)
            t1 = time.time()
            etr = clf.token_embeddings(Xtr, batch_size=1024).reshape(len(Xtr), -1)
            ete = clf.token_embeddings(Xte, batch_size=1024).reshape(len(Xte), -1)
            add(f'{tag}[emb+KNN]', *metrics(KNN().fit(etr).decision_function(ete)), tfit + time.time() - t1)
        except Exception as e:
            notes.append(f'{tag} failed: {e!r}'[:160])
            done = {r['model'] for r in rows}
            for sc in SCORES + ('emb+KNN',):
                if f'{tag}[{sc}]' not in done:
                    add(f'{tag}[{sc}]', np.nan, np.nan, time.time() - t0)

    tmp = part_path(name, seed) + '.part'
    pd.DataFrame(rows).to_csv(tmp, index=False)
    os.replace(tmp, part_path(name, seed))
    got = {r['model']: r['auc'] for r in rows}
    msg = (f'{name} seed={seed} n_train={n_train} d={X.shape[1]} {time.time() - t_task:.0f}s '
           + ' '.join(f'{k}={got[k]:.3f}' for k in ('KNN', 'IForest', 'JEPAOD[combined]', 'JEPAOD-untrained[combined]',
                                                    'TJEPAOD[combined]', 'TJEPAOD-untrained[embed]')))
    return msg + ''.join(f'\n  NOTE {n}' for n in notes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--max_rows', type=int, default=20000)
    ap.add_argument('--workers', type=int, default=13)
    ap.add_argument('--datasets', default='', help='comma separated name filter (substring match)')
    args = ap.parse_args()
    os.makedirs(PARTS, exist_ok=True)

    files = (glob.glob(os.path.join(ROOT, 'data', 'adbench', '*.npz'))
             + glob.glob(os.path.join(ROOT, 'data', 'adbench_extra', '*.npz')))
    if args.datasets:
        files = [f for f in files if any(k in os.path.basename(f) for k in args.datasets.split(','))]
    cost = {}
    for f in files:                     # longest tasks first: rows x tokens
        with np.load(f, allow_pickle=False) as d:
            n, p = d['X'].shape
        cost[f] = min(n, args.max_rows) * min(p, 64)
    tasks = [(f, s) for f in sorted(files, key=cost.get, reverse=True) for s in range(args.seeds)
             if not os.path.exists(part_path(os.path.basename(f)[:-4], s))]
    print(f'{len(files)} datasets, {len(tasks)} tasks to run, {args.workers} workers', flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_task, f, s, args.max_rows, args.epochs): (os.path.basename(f), s) for f, s in tasks}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                msg = fut.result()
            except Exception as e:
                msg = f'TASK FAILED {futs[fut]}: {e!r}'
            print(f'[{i}/{len(tasks)} {time.time() - t0:.0f}s] {msg}', flush=True)
    print('ALL DONE', flush=True)


if __name__ == '__main__':
    main()
