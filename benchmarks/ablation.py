"""Ablations for the report: encoder final LayerNorm x collapse-prevention mode.

Time series: synthetic (3 configs) + NAB realKnownCause (transductive), block masking.
Tabular: 8 small ADBench datasets, 60/40 split, seed 0.
"""
import os, sys, glob, json, time, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, torch
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from jepa_od import JEPAOD, TSJEPAOD
from pyod.utils.data import generate_ts_data
torch.set_num_threads(2)
rows = []
out = os.path.join(ROOT, 'results', 'ablation_results.csv')

def ts_eval(tag, Xtr, Xte, yte):
    for en in (False, True):
        for tm in ('sigreg', 'ema'):
            t0 = time.time()
            clf = TSJEPAOD(window_size=100 if Xte is None else 60, patch_len=10 if Xte is None else 5, epochs=20,
                           mask_mode='block', score='combined', encoder_norm=en, target_mode=tm, verbose=0)
            clf.fit(Xtr)
            for sc in ('pred', 'embed', 'combined'):
                clf.score = sc
                if Xte is None:
                    Xs = (np.asarray(Xtr, float).reshape(-1, 1) - clf.X_mean_) / clf.X_std_
                    s, valid = clf._score_series(Xs, clf._resolve_device()); s[~valid] = np.nanmedian(s[valid])
                else:
                    s = clf.decision_function(Xte)
                rows.append(dict(kind='ts', dataset=tag, encoder_norm=en, target_mode=tm, score=sc,
                                 auc=roc_auc_score(yte, s), ap=average_precision_score(yte, s), time=time.time() - t0))
    pd.DataFrame(rows).to_csv(out, index=False); print(tag, 'done', flush=True)

for nch, atype in [(1, 'point'), (1, 'subsequence'), (3, 'both')]:
    Xtr, Xte, ytr, yte = generate_ts_data(n_train=3000, n_test=1500, n_channels=nch, contamination=0.05, anomaly_type=atype, random_state=0)
    ts_eval(f'synthetic-{nch}ch-{atype}', Xtr, Xte, yte)
labels = json.load(open(os.path.join(ROOT, 'data', 'nab', 'combined_windows.json')))
for f in sorted(glob.glob(os.path.join(ROOT, 'data', 'nab', '*.csv'))):
    name = os.path.basename(f)[:-4]; df = pd.read_csv(f, parse_dates=['timestamp'])
    y = np.zeros(len(df), dtype=int)
    for a, b in labels[f'realKnownCause/{name}.csv']:
        y[(df.timestamp >= pd.Timestamp(a)) & (df.timestamp <= pd.Timestamp(b))] = 1
    ts_eval(f'NAB-{name}', df['value'].values.astype(float), None, y)

for name in ['6_cardio', '38_thyroid', '44_Wilt', '40_vowels', '42_WBC', '45_wine', '29_Pima', '18_Ionosphere']:
    d = np.load(os.path.join(ROOT, 'data', 'adbench', name + '.npz')); X, y = d['X'].astype(float), d['y'].astype(int)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, stratify=y, random_state=0)
    for en in (False, True):
        for tm in ('sigreg', 'ema'):
            t0 = time.time()
            clf = JEPAOD(epoch_num=30, verbose=0, score='combined', encoder_norm=en, target_mode=tm).fit(Xtr)
            for sc in ('pred', 'embed', 'combined'):
                clf.score = sc; s = clf.decision_function(Xte)
                rows.append(dict(kind='tabular', dataset=name, encoder_norm=en, target_mode=tm, score=sc,
                                 auc=roc_auc_score(yte, s), ap=average_precision_score(yte, s), time=time.time() - t0))
    pd.DataFrame(rows).to_csv(out, index=False); print(name, 'done', flush=True)

df = pd.DataFrame(rows)
print(df.groupby(['kind', 'encoder_norm', 'target_mode', 'score'])[['auc', 'ap']].mean().round(3).to_string())
