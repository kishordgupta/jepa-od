"""Diagnostics for the report.
(1) SKAB: is the test data shifted w.r.t. the anomaly-free training recording?  Compare patch-level
    Mahalanobis (embed score) of training patches vs normal test points vs anomalous test points, plus
    a 'per-file' protocol (fit on each valve1 file itself).
(2) NAB machine_temperature: does transductive training absorb the anomalous regime?  Compare
    transductive fit (all data) vs inductive fit (first 50 %), for embed / pred, and vs epochs.
"""
import os, sys, json, glob, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, torch
from sklearn.metrics import roc_auc_score
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from jepa_od import TSJEPAOD
from pyod.models.ts_od import TimeSeriesOD
torch.set_num_threads(2)

print('=== (1) SKAB shift diagnostic ===')
free = pd.read_csv(os.path.join(ROOT, 'data', 'skab', 'anomaly-free.csv'), sep=';')
feat = [c for c in free.columns if c not in ('datetime', 'anomaly', 'changepoint')]
Xtr = free[feat].values.astype(float)
clf = TSJEPAOD(window_size=60, patch_len=5, epochs=20, score='embed', verbose=0).fit(Xtr)
tr_scores = clf.decision_scores_
print(f'train embed score: median={np.median(tr_scores):.2f}  p90={np.percentile(tr_scores, 90):.2f}')
print(f'train channel means: {np.round(clf.X_mean_, 3).tolist()}')
rows = []
for f in sorted(glob.glob(os.path.join(ROOT, 'data', 'skab', 'valve1_*.csv')))[:6]:
    te = pd.read_csv(f, sep=';'); Xte, yte = te[feat].values.astype(float), te['anomaly'].values.astype(int)
    s = clf.decision_function(Xte)
    normal_shift = np.abs((Xte[yte == 0].mean(0) - clf.X_mean_) / clf.X_std_)
    clf.score = 'pred'; sp = clf.decision_function(Xte); clf.score = 'embed'
    # per-file (unsupervised) protocol: fit on the file itself
    own = TSJEPAOD(window_size=60, patch_len=5, epochs=20, score='combined', verbose=0).fit(Xte)
    own_auc = {}
    for sc in ('pred', 'embed', 'combined'):
        own.score = sc
        Xs = (Xte - own.X_mean_) / own.X_std_
        ss, valid = own._score_series(Xs, own._resolve_device()); ss[~valid] = np.nanmedian(ss[valid])
        own_auc[sc] = roc_auc_score(yte, ss)
    ifo = TimeSeriesOD('IForest', window_size=60).fit(Xtr)
    rows.append(dict(file=os.path.basename(f)[:-4], n=len(Xte), anom=f'{yte.mean():.0%}',
                     embed_median_normal=np.median(s[yte == 0]), embed_median_anom=np.median(s[yte == 1]),
                     max_normal_shift_in_std=normal_shift.max(), auc_embed=roc_auc_score(yte, s), auc_pred=roc_auc_score(yte, sp),
                     auc_ifo_trainfree=roc_auc_score(yte, ifo.decision_function(Xte)),
                     own_pred=own_auc['pred'], own_embed=own_auc['embed'], own_combined=own_auc['combined']))
    print(pd.DataFrame(rows).iloc[[-1]].round(3).to_string(index=False, header=(len(rows) == 1)), flush=True)
pd.DataFrame(rows).to_csv(os.path.join(ROOT, 'results', 'diag_skab.csv'), index=False)

print('\n=== (2) NAB absorption diagnostic (machine_temperature_system_failure) ===')
labels = json.load(open(os.path.join(ROOT, 'data', 'nab', 'combined_windows.json')))
df = pd.read_csv(os.path.join(ROOT, 'data', 'nab', 'machine_temperature_system_failure.csv'), parse_dates=['timestamp'])
y = np.zeros(len(df), dtype=int)
for a, b in labels['realKnownCause/machine_temperature_system_failure.csv']:
    y[(df.timestamp >= pd.Timestamp(a)) & (df.timestamp <= pd.Timestamp(b))] = 1
X = df['value'].values.astype(float)
half = len(X) // 2
print(f'n={len(X)}  anomalous={y.mean():.1%}  anomalous in first half={y[:half].mean():.1%}  second half={y[half:].mean():.1%}')
rows = []
for protocol in ('transductive', 'inductive-first-half'):
    for epochs in (5, 20):
        for sw in (0.05, 0.01):
            clf = TSJEPAOD(window_size=100, patch_len=10, epochs=epochs, score='combined', sigreg_weight=sw, verbose=0)
            Xfit = X if protocol == 'transductive' else X[:half]
            clf.fit(Xfit)
            Xs = (X.reshape(-1, 1) - clf.X_mean_) / clf.X_std_
            r = dict(protocol=protocol, epochs=epochs, sigreg_weight=sw)
            for sc in ('pred', 'embed', 'combined'):
                clf.score = sc
                s, valid = clf._score_series(Xs, clf._resolve_device()); s[~valid] = np.nanmedian(s[valid])
                mask = slice(None) if protocol == 'transductive' else slice(half, None)
                r[sc] = roc_auc_score(y[mask], s[mask])
            rows.append(r); print(pd.DataFrame([r]).round(3).to_string(index=False, header=(len(rows) == 1)), flush=True)
ifo = TimeSeriesOD('IForest', window_size=100).fit(X[:half])
print('IForest inductive second-half AUC:', round(roc_auc_score(y[half:], ifo.decision_function(X)[half:]), 3))
ifo = TimeSeriesOD('IForest', window_size=100).fit(X)
print('IForest transductive AUC:', round(roc_auc_score(y, ifo.decision_scores_), 3))
pd.DataFrame(rows).to_csv(os.path.join(ROOT, 'results', 'diag_nab.csv'), index=False)
print('DIAG DONE')
