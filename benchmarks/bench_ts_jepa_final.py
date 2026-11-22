"""Re-run only the TSJEPAOD variants of bench_ts.py with the final class defaults.
Results go to results/ts_results_jepa_final.csv and are merged by make_report_tables.py.

Without arguments this is the committed run (2 synthetic seeds, seed 0 for NAB and SKAB).
The untrained control of REPORT.md section 7.6 uses the options:

    python benchmarks/bench_ts_jepa_final.py --seeds 5 --threads 2 --out results/ts_control_trained.csv
    python benchmarks/bench_ts_jepa_final.py --seeds 5 --threads 2 --epochs 0 --tag TSJEPAOD-untrained \
        --out results/ts_control_untrained.csv

The JEPA versions of REPORT.md section 7.7 (V = ijepa, tsjepa, tjepa; the two files above are V = lejepa):

    python benchmarks/bench_ts_jepa_final.py --seeds 5 --threads 2 --variant V --out results/ts_variants/V_trained.csv
    python benchmarks/bench_ts_jepa_final.py --seeds 5 --threads 2 --variant V --epochs 0 \
        --out results/ts_variants/V_untrained.csv

and, for the masking of the TS-JEPA paper, the same two commands with
``--variant tsjepa --mask random:0.7`` and the output files ``tsjepa-paper_trained.csv`` / ``tsjepa-paper_untrained.csv``.
"""
import os, sys, json, glob, time, warnings, argparse
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, torch
from sklearn.metrics import roc_auc_score, average_precision_score
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from jepa_od import TSJEPAOD
from pyod.utils.data import generate_ts_data
ap = argparse.ArgumentParser()
ap.add_argument('--epochs', type=int, default=None, help='default: the class default (10); 0 = untrained control')
ap.add_argument('--seeds', type=int, default=None,
                help='default: the committed run, 2 synthetic seeds and seed 0 for NAB and SKAB')
ap.add_argument('--tag', default='TSJEPAOD', help='model name prefix in the output')
ap.add_argument('--variant', default='lejepa', help='JEPA version: lejepa (class default), ijepa, tsjepa, tjepa')
ap.add_argument('--mask', default=None, help="MODE:RATIO, e.g. random:0.7; default: block and forecast at the "
                                             "class-default ratio")
ap.add_argument('--threads', type=int, default=3)
ap.add_argument('--out', default=os.path.join(ROOT, 'results', 'ts_results_jepa_final.csv'))
args = ap.parse_args()
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
torch.set_num_threads(args.threads)
KW = {} if args.epochs is None else {'epochs': args.epochs}
if args.variant != 'lejepa':
    KW['variant'] = args.variant
MODES = ('block', 'forecast')
if args.mask:
    MODES, KW['mask_ratio'] = (args.mask.split(':')[0],), float(args.mask.split(':')[1])
SYN_SEEDS = range(2 if args.seeds is None else args.seeds)
REAL_SEEDS = range(1 if args.seeds is None else args.seeds)
TAG = args.tag
out = args.out
rows = []

def run(tag, Xtr, Xte, yte, window, patch, seed, protocol):
    for mode in MODES:
        t0 = time.time()
        clf = TSJEPAOD(window_size=window, patch_len=patch, mask_mode=mode, score='combined',
                       random_state=seed, verbose=0, **KW).fit(Xtr)
        tfit = time.time() - t0
        for sc in ('pred', 'embed', 'combined'):
            clf.score = sc
            if Xte is None:
                Xs = (np.asarray(Xtr, float).reshape(-1, 1) - clf.X_mean_) / clf.X_std_
                s, valid = clf._score_series(Xs, clf._resolve_device()); s[~valid] = np.nanmedian(s[valid])
            else:
                s = clf.decision_function(Xte)
            rows.append(dict(dataset=tag, model=f'{TAG}-{mode}[{sc}]', auc=roc_auc_score(yte, s),
                             ap=average_precision_score(yte, s), time=tfit, protocol=protocol, seed=seed))
    pd.DataFrame(rows).to_csv(out, index=False); print(tag, seed, 'done', flush=True)

for nch, atype in [(1, 'point'), (1, 'subsequence'), (3, 'both')]:
    for seed in SYN_SEEDS:
        Xtr, Xte, ytr, yte = generate_ts_data(n_train=3000, n_test=1500, n_channels=nch, contamination=0.05,
                                              anomaly_type=atype, random_state=seed)
        run(f'synthetic-{nch}ch-{atype}', Xtr, Xte, yte, 60, 5, seed, 'train/test')
labels = json.load(open(os.path.join(ROOT, 'data', 'nab', 'combined_windows.json')))
for f in sorted(glob.glob(os.path.join(ROOT, 'data', 'nab', '*.csv'))):
    name = os.path.basename(f)[:-4]; df = pd.read_csv(f, parse_dates=['timestamp'])
    y = np.zeros(len(df), dtype=int)
    for a, b in labels[f'realKnownCause/{name}.csv']:
        y[(df.timestamp >= pd.Timestamp(a)) & (df.timestamp <= pd.Timestamp(b))] = 1
    for seed in REAL_SEEDS:
        run(f'NAB-{name}', df['value'].values.astype(float), None, y, 100, 10, seed, 'transductive')
free = pd.read_csv(os.path.join(ROOT, 'data', 'skab', 'anomaly-free.csv'), sep=';')
feat = [c for c in free.columns if c not in ('datetime', 'anomaly', 'changepoint')]
Xtr = free[feat].values.astype(float)
fitted = {}
for seed in REAL_SEEDS:
    for mode in MODES:
        t0 = time.time()
        fitted[mode, seed] = (TSJEPAOD(window_size=60, patch_len=5, mask_mode=mode, score='combined',
                                       random_state=seed, verbose=0, **KW).fit(Xtr), time.time() - t0)
for f in sorted(glob.glob(os.path.join(ROOT, 'data', 'skab', 'valve1_*.csv'))):
    te = pd.read_csv(f, sep=';'); Xte, yte = te[feat].values.astype(float), te['anomaly'].values.astype(int)
    tag = 'SKAB-' + os.path.basename(f)[:-4]
    for (mode, seed), (clf, tfit) in fitted.items():
        for sc in ('pred', 'embed', 'combined'):
            clf.score = sc; s = clf.decision_function(Xte)
            rows.append(dict(dataset=tag, model=f'{TAG}-{mode}[{sc}]', auc=roc_auc_score(yte, s),
                             ap=average_precision_score(yte, s), time=tfit, protocol='train-free/test', seed=seed))
    pd.DataFrame(rows).to_csv(out, index=False); print(tag, 'done', flush=True)
print('ALL DONE')
