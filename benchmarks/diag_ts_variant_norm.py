"""Post hoc diagnostic for REPORT.md section 7.7: does switching off the normalised encoder output
(`encoder_norm=False`) repair the EMA versions of TSJEPAOD on SKAB, where the test recordings are in a
different operating regime than the training recording?

Same protocol as the SKAB part of bench_ts_jepa_final.py (block masking, 5 seeds, 2 threads).
Writes results/ts_variants/diag_skab_nonorm.csv.
"""
import os, sys, glob, time, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, torch
from sklearn.metrics import roc_auc_score, average_precision_score
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from jepa_od import TSJEPAOD
torch.set_num_threads(2)
out = os.path.join(ROOT, 'results', 'ts_variants', 'diag_skab_nonorm.csv')
free = pd.read_csv(os.path.join(ROOT, 'data', 'skab', 'anomaly-free.csv'), sep=';')
feat = [c for c in free.columns if c not in ('datetime', 'anomaly', 'changepoint')]
Xtr = free[feat].values.astype(float)
files = sorted(glob.glob(os.path.join(ROOT, 'data', 'skab', 'valve1_*.csv')))
rows = []
for variant in ('tsjepa', 'tjepa'):
    for seed in range(5):
        t0 = time.time()
        clf = TSJEPAOD(window_size=60, patch_len=5, mask_mode='block', score='combined', variant=variant,
                       encoder_norm=False, random_state=seed, verbose=0).fit(Xtr)
        tfit = time.time() - t0
        for f in files:
            te = pd.read_csv(f, sep=';'); Xte, yte = te[feat].values.astype(float), te['anomaly'].values.astype(int)
            for sc in ('pred', 'embed', 'combined'):
                clf.score = sc; s = clf.decision_function(Xte)
                rows.append(dict(dataset='SKAB-' + os.path.basename(f)[:-4], variant=variant, encoder_norm=False,
                                 score=sc, seed=seed, auc=roc_auc_score(yte, s),
                                 ap=average_precision_score(yte, s), time=tfit))
        pd.DataFrame(rows).to_csv(out, index=False); print(variant, seed, 'done', flush=True)
d = pd.DataFrame(rows)
print(d.groupby(['variant', 'score'])[['auc', 'ap']].mean().round(3).to_string())
