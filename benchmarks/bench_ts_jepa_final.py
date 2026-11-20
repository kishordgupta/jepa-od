"""Re-run only the TSJEPAOD variants of bench_ts.py with the final class defaults.
Results go to results/ts_results_jepa_final.csv and are merged by make_report_tables.py."""
import os, sys, json, glob, time, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, torch
from sklearn.metrics import roc_auc_score, average_precision_score
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from jepa_od import TSJEPAOD
from pyod.utils.data import generate_ts_data
torch.set_num_threads(3)
EPOCHS = 20
out = os.path.join(ROOT, 'results', 'ts_results_jepa_final.csv')
rows = []

def run(tag, Xtr, Xte, yte, window, patch, seed, protocol):
    for mode in ('block', 'forecast'):
        t0 = time.time()
        clf = TSJEPAOD(window_size=window, patch_len=patch, mask_mode=mode, score='combined',
                       random_state=seed, verbose=0).fit(Xtr)
        tfit = time.time() - t0
        for sc in ('pred', 'embed', 'combined'):
            clf.score = sc
            if Xte is None:
                Xs = (np.asarray(Xtr, float).reshape(-1, 1) - clf.X_mean_) / clf.X_std_
                s, valid = clf._score_series(Xs, clf._resolve_device()); s[~valid] = np.nanmedian(s[valid])
            else:
                s = clf.decision_function(Xte)
            rows.append(dict(dataset=tag, model=f'TSJEPAOD-{mode}[{sc}]', auc=roc_auc_score(yte, s),
                             ap=average_precision_score(yte, s), time=tfit, protocol=protocol, seed=seed))
    pd.DataFrame(rows).to_csv(out, index=False); print(tag, seed, 'done', flush=True)

for nch, atype in [(1, 'point'), (1, 'subsequence'), (3, 'both')]:
    for seed in range(2):
        Xtr, Xte, ytr, yte = generate_ts_data(n_train=3000, n_test=1500, n_channels=nch, contamination=0.05,
                                              anomaly_type=atype, random_state=seed)
        run(f'synthetic-{nch}ch-{atype}', Xtr, Xte, yte, 60, 5, seed, 'train/test')
labels = json.load(open(os.path.join(ROOT, 'data', 'nab', 'combined_windows.json')))
for f in sorted(glob.glob(os.path.join(ROOT, 'data', 'nab', '*.csv'))):
    name = os.path.basename(f)[:-4]; df = pd.read_csv(f, parse_dates=['timestamp'])
    y = np.zeros(len(df), dtype=int)
    for a, b in labels[f'realKnownCause/{name}.csv']:
        y[(df.timestamp >= pd.Timestamp(a)) & (df.timestamp <= pd.Timestamp(b))] = 1
    run(f'NAB-{name}', df['value'].values.astype(float), None, y, 100, 10, 0, 'transductive')
free = pd.read_csv(os.path.join(ROOT, 'data', 'skab', 'anomaly-free.csv'), sep=';')
feat = [c for c in free.columns if c not in ('datetime', 'anomaly', 'changepoint')]
Xtr = free[feat].values.astype(float)
fitted = {}
for mode in ('block', 'forecast'):
    t0 = time.time()
    fitted[mode] = (TSJEPAOD(window_size=60, patch_len=5, mask_mode=mode, score='combined',
                             random_state=0, verbose=0).fit(Xtr), time.time() - t0)
for f in sorted(glob.glob(os.path.join(ROOT, 'data', 'skab', 'valve1_*.csv'))):
    te = pd.read_csv(f, sep=';'); Xte, yte = te[feat].values.astype(float), te['anomaly'].values.astype(int)
    tag = 'SKAB-' + os.path.basename(f)[:-4]
    for mode, (clf, tfit) in fitted.items():
        for sc in ('pred', 'embed', 'combined'):
            clf.score = sc; s = clf.decision_function(Xte)
            rows.append(dict(dataset=tag, model=f'TSJEPAOD-{mode}[{sc}]', auc=roc_auc_score(yte, s),
                             ap=average_precision_score(yte, s), time=tfit, protocol='train-free/test', seed=0))
    pd.DataFrame(rows).to_csv(out, index=False); print(tag, 'done', flush=True)
print('ALL DONE')
