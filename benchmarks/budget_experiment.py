"""Training-budget experiment for TSJEPAOD defaults: epochs x max_train_windows."""
import os, sys, json, warnings, time
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, torch
from sklearn.metrics import roc_auc_score
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from jepa_od import TSJEPAOD
from pyod.utils.data import generate_ts_data
torch.set_num_threads(2)
labels = json.load(open(os.path.join(ROOT, 'data', 'nab', 'combined_windows.json')))
def nab(name):
    df = pd.read_csv(os.path.join(ROOT, 'data', 'nab', name + '.csv'), parse_dates=['timestamp'])
    y = np.zeros(len(df), dtype=int)
    for a, b in labels[f'realKnownCause/{name}.csv']:
        y[(df.timestamp >= pd.Timestamp(a)) & (df.timestamp <= pd.Timestamp(b))] = 1
    return df['value'].values.astype(float), y
rows = []
configs = [(20, 20000), (10, 5000), (5, 5000), (10, 2000)]
for name in ('machine_temperature_system_failure', 'ambient_temperature_system_failure', 'cpu_utilization_asg_misconfiguration', 'nyc_taxi'):
    X, y = nab(name)
    for ep, mtw in configs:
        t0 = time.time()
        clf = TSJEPAOD(window_size=100, patch_len=10, epochs=ep, max_train_windows=mtw, score='combined', verbose=0).fit(X)
        Xs = (X.reshape(-1, 1) - clf.X_mean_) / clf.X_std_
        r = dict(dataset='NAB-' + name, epochs=ep, max_train_windows=mtw, time=round(time.time() - t0, 1))
        for sc in ('pred', 'embed', 'combined'):
            clf.score = sc; s, valid = clf._score_series(Xs, clf._resolve_device()); s[~valid] = np.nanmedian(s[valid])
            r[sc] = roc_auc_score(y, s)
        rows.append(r); print(pd.DataFrame([r]).round(3).to_string(index=False, header=(len(rows) == 1)), flush=True)
for nch, atype in [(1, 'subsequence'), (3, 'both')]:
    Xtr, Xte, ytr, yte = generate_ts_data(n_train=3000, n_test=1500, n_channels=nch, contamination=0.05, anomaly_type=atype, random_state=0)
    for ep, mtw in configs:
        t0 = time.time()
        clf = TSJEPAOD(window_size=60, patch_len=5, epochs=ep, max_train_windows=mtw, score='combined', verbose=0).fit(Xtr)
        r = dict(dataset=f'synthetic-{nch}ch-{atype}', epochs=ep, max_train_windows=mtw, time=round(time.time() - t0, 1))
        for sc in ('pred', 'embed', 'combined'):
            clf.score = sc; r[sc] = roc_auc_score(yte, clf.decision_function(Xte))
        rows.append(r); print(pd.DataFrame([r]).round(3).to_string(index=False, header=False), flush=True)
df = pd.DataFrame(rows); df.to_csv(os.path.join(ROOT, 'results', 'budget_experiment.csv'), index=False)
print(df.groupby(['epochs', 'max_train_windows'])[['pred', 'embed', 'combined', 'time']].mean().round(3).to_string())
print('BUDGET DONE')
