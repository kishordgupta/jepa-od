"""T-JEPA family benchmark on the ADBench data sets of bench_tabular.py.

Same protocol as bench_tabular.py (stratified 60/40 split per seed, fit
on the contaminated training split without labels, ROC-AUC and average
precision on the test split), so the rows can be compared with
results/tabular_results.csv.

Each variant is fitted once per split and scored four ways:

* ``[pred]`` / ``[embed]`` / ``[combined]`` -- the JEPA-OD scores;
* ``[emb+KNN]`` -- PyOD ``KNN`` on the flattened token embeddings, the
  unsupervised counterpart of how T-JEPA uses its representation (a
  downstream model on frozen embeddings).

Variants (``--variants``, comma separated):

* ``JEPAOD``          -- reference, the default detector of this package
* ``TJEPAOD``         -- T-JEPA recipe at the budget of JEPAOD
* ``TJEPAOD-nonorm``  -- without the normalised encoder output
* ``TJEPAOD-noreg``   -- without the [REG] token
* ``TJEPAOD-repoopt`` -- optimisation defaults of the released T-JEPA
  code (batch size 512, learning rate 1e-4, 100 epochs, 5 warm-up epochs)
* ``JEPAOD-untrained`` / ``TJEPAOD-untrained`` -- controls with zero
  training epochs: the scores are computed from the randomly initialised
  networks, which shows how much of a result is due to training

The committed results were produced by five parallel jobs with 3 threads
each, one CSV per job in results/tjepa_parts/ (ref.csv: JEPAOD and the two
untrained controls; tjepa.csv, nonorm.csv, noreg.csv, repoopt.csv), e.g.

    python benchmarks/bench_tjepa.py --threads 3 --variants TJEPAOD \
        --out results/tjepa_parts/tjepa.csv

followed by ``python benchmarks/make_tjepa_tables.py --parts``. Results are
deterministic for a fixed seed and thread count only.
"""
import glob, os, sys, time, warnings, argparse
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from jepa_od import JEPAOD, TJEPAOD
from pyod.models.knn import KNN

VARIANTS = {
    'JEPAOD': lambda seed: JEPAOD(verbose=0, random_state=seed),
    'TJEPAOD': lambda seed: TJEPAOD(verbose=0, random_state=seed),
    'TJEPAOD-nonorm': lambda seed: TJEPAOD(verbose=0, random_state=seed, encoder_norm=False),
    'TJEPAOD-noreg': lambda seed: TJEPAOD(verbose=0, random_state=seed, n_reg_tokens=0),
    'TJEPAOD-repoopt': lambda seed: TJEPAOD(verbose=0, random_state=seed, batch_size=512, lr=1e-4,
                                             epoch_num=100, warmup_epochs=5),
    'JEPAOD-untrained': lambda seed: JEPAOD(verbose=0, random_state=seed, epoch_num=0),
    'TJEPAOD-untrained': lambda seed: TJEPAOD(verbose=0, random_state=seed, epoch_num=0),
}

ap = argparse.ArgumentParser()
ap.add_argument('--seeds', type=int, default=3)
ap.add_argument('--max_rows', type=int, default=20000)
ap.add_argument('--threads', type=int, default=4)
ap.add_argument('--variants', default=','.join(VARIANTS))
ap.add_argument('--datasets', default='', help='comma separated name filter (substring match)')
ap.add_argument('--out', default=os.path.join(ROOT, 'results', 'tjepa_results.csv'))
args = ap.parse_args()
torch.set_num_threads(args.threads)

rows = []
files = sorted(glob.glob(os.path.join(ROOT, 'data', 'adbench', '*.npz')))
if args.datasets:
    files = [f for f in files if any(k in os.path.basename(f) for k in args.datasets.split(','))]
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
        for tag in args.variants.split(','):
            t0 = time.time()
            clf = VARIANTS[tag](seed)
            clf.fit(Xtr); tfit = time.time() - t0
            for sc in ('pred', 'embed', 'combined'):
                clf.score = sc
                s = clf.decision_function(Xte)
                rows.append(dict(dataset=name, model=f'{tag}[{sc}]', seed=seed,
                                 auc=roc_auc_score(yte, s), ap=average_precision_score(yte, s), time=tfit))
            t1 = time.time()
            etr = clf.token_embeddings(Xtr, batch_size=1024).reshape(len(Xtr), -1)
            ete = clf.token_embeddings(Xte, batch_size=1024).reshape(len(Xte), -1)
            s = KNN().fit(etr).decision_function(ete)
            rows.append(dict(dataset=name, model=f'{tag}[emb+KNN]', seed=seed,
                             auc=roc_auc_score(yte, s), ap=average_precision_score(yte, s),
                             time=tfit + time.time() - t1))
        pd.DataFrame(rows).to_csv(args.out, index=False)
        last = pd.DataFrame(rows); last = last[(last.dataset == name) & (last.seed == seed)]
        print(f'{name} seed={seed} n={Xtr.shape} ' + ' '.join(f'{r.model}={r.auc:.3f}' for r in last.itertuples()), flush=True)

df = pd.DataFrame(rows)
summ = df.groupby(['dataset', 'model'])[['auc', 'ap', 'time']].mean().reset_index()
print(summ.groupby('model')[['auc', 'ap', 'time']].mean().round(3).to_string())
