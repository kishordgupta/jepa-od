"""Time-series benchmark: TSJEPAOD vs PyOD time-series detectors.

Datasets / protocols
* synthetic (pyod.utils.data.generate_ts_data): fit on train series, score test series
* NAB realKnownCause (7 series): unsupervised / transductive -- fit on the
  whole series, evaluate decision_scores_ against the NAB label windows
* SKAB valve1 (16 files, 8 channels): fit on anomaly-free.csv, score each file
Metrics: point-wise ROC-AUC and average precision.
"""
import os, sys, json, glob, time, warnings, inspect, argparse
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, torch
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from jepa_od import TSJEPAOD
from pyod.utils.data import generate_ts_data

torch.set_num_threads(4)
ap = argparse.ArgumentParser()
ap.add_argument('--epochs', type=int, default=20)
ap.add_argument('--out', default=os.path.join(ROOT, 'results', 'ts_results.csv'))
args = ap.parse_args()


def make(cls, **kw):
    sig = inspect.signature(cls.__init__).parameters
    return cls(**{k: v for k, v in kw.items() if k in sig})


def baselines(window, seed):
    from pyod.models.ts_lstm import LSTMAD
    from pyod.models.ts_od import TimeSeriesOD
    from pyod.models.ts_matrix_profile import MatrixProfile
    from pyod.models.ts_spectral_residual import SpectralResidual
    from pyod.models.ts_anomaly_transformer import AnomalyTransformer
    return {
        'LSTMAD': make(LSTMAD, window_size=window, epochs=args.epochs, random_state=seed),
        'TSOD-IForest': TimeSeriesOD('IForest', window_size=window),
        'TSOD-KNN': TimeSeriesOD('KNN', window_size=window),
        'TSOD-AutoEncoder': TimeSeriesOD('AutoEncoder', window_size=window),
        'MatrixProfile': make(MatrixProfile, window_size=window, random_state=seed),
        'SpectralResidual': make(SpectralResidual, window_size=window, random_state=seed),
        'AnomalyTransformer': make(AnomalyTransformer, window_size=window, d_model=64, n_heads=4,
                                   n_layers=2, epochs=5, batch_size=64, step=1),
    }


def jepa_variants(window, patch, seed):
    out = {}
    for mode in ('block', 'forecast'):
        out[mode] = TSJEPAOD(window_size=window, patch_len=patch, epochs=args.epochs, mask_mode=mode,
                             score='combined', random_state=seed, verbose=0)
    return out


def evaluate(models, jepas, Xtr, Xte, yte, tag, extra):
    rows = []
    for name, clf in models.items():
        t0 = time.time()
        try:
            clf.fit(Xtr)
            s = clf.decision_scores_ if Xte is None else clf.decision_function(Xte)
            auc, apr = roc_auc_score(yte, s), average_precision_score(yte, s)
        except Exception as e:
            print(f'  {tag} {name} FAILED: {type(e).__name__}: {e}'); auc = apr = np.nan
        rows.append(dict(dataset=tag, model=name, auc=auc, ap=apr, time=time.time() - t0, **extra))
    for mode, clf in jepas.items():
        t0 = time.time()
        try:
            clf.fit(Xtr); tfit = time.time() - t0
            for sc in ('pred', 'embed', 'combined'):
                clf.score = sc
                if Xte is None:
                    Xs = (clf_X := None)
                    s, valid = clf._score_series((Xtr_std := (validate(Xtr) - clf.X_mean_) / clf.X_std_), clf._resolve_device())
                    s[~valid] = np.nanmedian(s[valid])
                else:
                    s = clf.decision_function(Xte)
                rows.append(dict(dataset=tag, model=f'TSJEPAOD-{mode}[{sc}]', auc=roc_auc_score(yte, s),
                                 ap=average_precision_score(yte, s), time=tfit, **extra))
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f'  {tag} TSJEPAOD-{mode} FAILED: {type(e).__name__}: {e}')
    return rows


def validate(X):
    X = np.asarray(X, dtype=np.float64)
    return X.reshape(-1, 1) if X.ndim == 1 else X


rows = []
def flush(rows):
    pd.DataFrame(rows).to_csv(args.out, index=False)

# ---------------- synthetic ----------------
for nch, atype in [(1, 'point'), (1, 'subsequence'), (3, 'both')]:
    for seed in range(2):
        Xtr, Xte, ytr, yte = generate_ts_data(n_train=3000, n_test=1500, n_channels=nch, contamination=0.05,
                                              anomaly_type=atype, random_state=seed)
        tag = f'synthetic-{nch}ch-{atype}'
        rows += evaluate(baselines(60, seed), jepa_variants(60, 5, seed), Xtr, Xte, yte, tag,
                         dict(protocol='train/test', seed=seed))
        flush(rows); print(tag, seed, 'done', flush=True)

# ---------------- NAB realKnownCause (transductive) ----------------
labels = json.load(open(os.path.join(ROOT, 'data', 'nab', 'combined_windows.json')))
for f in sorted(glob.glob(os.path.join(ROOT, 'data', 'nab', '*.csv'))):
    name = os.path.basename(f)[:-4]
    df = pd.read_csv(f, parse_dates=['timestamp'])
    y = np.zeros(len(df), dtype=int)
    for a, b in labels[f'realKnownCause/{name}.csv']:
        y[(df.timestamp >= pd.Timestamp(a)) & (df.timestamp <= pd.Timestamp(b))] = 1
    X = df['value'].values.astype(np.float64)
    tag = f'NAB-{name}'
    rows += evaluate(baselines(100, 0), jepa_variants(100, 10, 0), X, None, y, tag,
                     dict(protocol='transductive', seed=0))
    flush(rows); print(tag, 'done', f'{y.mean():.2%} anomalous', flush=True)

# ---------------- SKAB (train on anomaly-free, test on valve1 files) ----------------
free = pd.read_csv(os.path.join(ROOT, 'data', 'skab', 'anomaly-free.csv'), sep=';')
feat = [c for c in free.columns if c not in ('datetime', 'anomaly', 'changepoint')]
Xtr = free[feat].values.astype(np.float64)
models = baselines(60, 0); jepas = jepa_variants(60, 5, 0)
fitted = {}
for name, clf in {**models, **{f'TSJEPAOD-{m}': c for m, c in jepas.items()}}.items():
    t0 = time.time()
    try:
        clf.fit(Xtr); fitted[name] = (clf, time.time() - t0)
    except Exception as e:
        print(f'  SKAB {name} FAILED on fit: {type(e).__name__}: {e}')
for f in sorted(glob.glob(os.path.join(ROOT, 'data', 'skab', 'valve1_*.csv'))):
    fname = os.path.basename(f)[:-4]
    te = pd.read_csv(f, sep=';')
    Xte, yte = te[feat].values.astype(np.float64), te['anomaly'].values.astype(int)
    tag = f'SKAB-{fname}'
    for name, (clf, tfit) in fitted.items():
        try:
            if name.startswith('TSJEPAOD'):
                for sc in ('pred', 'embed', 'combined'):
                    clf.score = sc; s = clf.decision_function(Xte)
                    rows.append(dict(dataset=tag, model=f'{name}[{sc}]', auc=roc_auc_score(yte, s),
                                     ap=average_precision_score(yte, s), time=tfit, protocol='train-free/test', seed=0))
            else:
                s = clf.decision_function(Xte)
                rows.append(dict(dataset=tag, model=name, auc=roc_auc_score(yte, s),
                                 ap=average_precision_score(yte, s), time=tfit, protocol='train-free/test', seed=0))
        except Exception as e:
            print(f'  {tag} {name} FAILED: {type(e).__name__}: {e}')
    flush(rows); print(tag, 'done', flush=True)

df = pd.DataFrame(rows)
df['group'] = df.dataset.str.split('-').str[0]
summ = df.groupby(['group', 'model'])[['auc', 'ap']].mean().reset_index()
piv = summ.pivot(index='group', columns='model', values='auc'); piv.loc['MEAN'] = piv.mean(0)
piv.round(3).to_csv(args.out.replace('.csv', '_auc_summary.csv'))
appiv = summ.pivot(index='group', columns='model', values='ap'); appiv.loc['MEAN'] = appiv.mean(0)
appiv.round(3).to_csv(args.out.replace('.csv', '_ap_summary.csv'))
print(piv.round(3).T.to_string()); print(appiv.round(3).T.to_string())
