"""Tabular benchmark: JEPAOD vs PyOD baselines on ADBench datasets.

Protocol: stratified 60/40 split per seed; detectors are fitted on the
(contaminated) training split without labels and evaluated on the test
split with ROC-AUC and average precision.
"""
import glob, os, sys, time, warnings, argparse
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from jepa_od import JEPAOD

torch.set_num_threads(4)
ap = argparse.ArgumentParser()
ap.add_argument('--seeds', type=int, default=3)
ap.add_argument('--epochs', type=int, default=30)
ap.add_argument('--max_rows', type=int, default=20000)
ap.add_argument('--out', default=os.path.join(ROOT, 'results', 'tabular_results.csv'))
args = ap.parse_args()

def make(cls, **kw):
    import inspect
    sig = inspect.signature(cls.__init__).parameters
    if 'epoch_num' in kw and 'epoch_num' not in sig and 'epochs' in sig:
        kw['epochs'] = kw.pop('epoch_num')
    return cls(**{k: v for k, v in kw.items() if k in sig})


def baselines(seed, n_features):
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
        'AutoEncoder': make(AutoEncoder, epoch_num=args.epochs, verbose=0, random_state=seed),
        'DeepSVDD': make(DeepSVDD, n_features=n_features, epoch_num=args.epochs, verbose=0, random_state=seed),
        'LUNAR': make(LUNAR, verbose=0),
        'DIF': DIF(random_state=seed),
    }

rows = []
files = sorted(glob.glob(os.path.join(ROOT, 'data', 'adbench', '*.npz')))
for f in files:
    name = os.path.basename(f)[:-4]
    d = np.load(f, allow_pickle=False)
    X, y = d['X'].astype(np.float64), d['y'].astype(int)
    for seed in range(args.seeds):
        rng = np.random.RandomState(seed)
        if len(X) > args.max_rows:
            keep = rng.choice(len(X), args.max_rows, replace=False)
            Xs, ys = X[keep], y[keep]
        else:
            Xs, ys = X, y
        Xtr, Xte, ytr, yte = train_test_split(Xs, ys, test_size=0.4, stratify=ys, random_state=seed)
        models = baselines(seed, Xtr.shape[1])
        for mname, clf in models.items():
            t0 = time.time()
            try:
                clf.fit(Xtr); s = clf.decision_function(Xte)
                auc, apr = roc_auc_score(yte, s), average_precision_score(yte, s)
            except Exception as e:
                print(f'  {name} {mname} FAILED: {e}'); auc = apr = np.nan
            rows.append(dict(dataset=name, model=mname, seed=seed, auc=auc, ap=apr, time=time.time()-t0))
        # JEPAOD: one fit per target mode, three scores
        for tag, kw in [('JEPAOD', dict(target_mode='sigreg')),
                        ('JEPAOD-ema', dict(target_mode='ema')),
                        ('JEPAOD-recon', dict(target_mode='sigreg', recon_weight=0.1))]:
            t0 = time.time()
            clf = JEPAOD(epoch_num=args.epochs, verbose=0, random_state=seed, score='combined', **kw)
            clf.fit(Xtr); tfit = time.time() - t0
            for sc in ('pred', 'embed', 'combined'):
                clf.score = sc
                s = clf.decision_function(Xte)
                rows.append(dict(dataset=name, model=f'{tag}[{sc}]', seed=seed,
                                 auc=roc_auc_score(yte, s), ap=average_precision_score(yte, s), time=tfit))
        pd.DataFrame(rows).to_csv(args.out, index=False)
        last = pd.DataFrame(rows); last = last[(last.dataset == name) & (last.seed == seed)]
        print(f'{name} seed={seed} n={Xs.shape} ' + ' '.join(f'{r.model}={r.auc:.3f}' for r in last.itertuples()), flush=True)

df = pd.DataFrame(rows)
summ = df.groupby(['dataset', 'model'])[['auc', 'ap', 'time']].mean().reset_index()
piv = summ.pivot(index='dataset', columns='model', values='auc')
piv.loc['MEAN'] = piv.mean(0)
ranks = summ.pivot(index='dataset', columns='model', values='auc').rank(axis=1, ascending=False)
piv.loc['AVG_RANK'] = ranks.mean(0)
piv.round(3).to_csv(args.out.replace('.csv', '_auc_summary.csv'))
ap_piv = summ.pivot(index='dataset', columns='model', values='ap'); ap_piv.loc['MEAN'] = ap_piv.mean(0)
ap_piv.round(3).to_csv(args.out.replace('.csv', '_ap_summary.csv'))
print(piv.round(3).to_string())
