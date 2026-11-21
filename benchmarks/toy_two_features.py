"""Two-feature toy problem of REPORT.md sections 4.6 and 7.5.

PyOD's ``generate_data`` with 2 features, 600 training and 300 test rows,
10 % outliers, seeds 0-2. Fits the JEPA family with and without a
normalised encoder output, plus zero-epoch (untrained) controls, and
writes results/toy_two_features.csv and results/toy_two_features.md.
"""
import os, warnings
import numpy as np, pandas as pd, torch
from sklearn.metrics import roc_auc_score
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.utils.data import generate_data

from jepa_od import JEPAOD, TJEPAOD

warnings.filterwarnings('ignore')
torch.set_num_threads(4)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, 'results')

VARIANTS = {
    'JEPAOD': lambda s: JEPAOD(verbose=0, random_state=s),
    'JEPAOD(encoder_norm=True)': lambda s: JEPAOD(verbose=0, random_state=s, encoder_norm=True),
    'JEPAOD-untrained': lambda s: JEPAOD(verbose=0, random_state=s, epoch_num=0),
    'TJEPAOD': lambda s: TJEPAOD(verbose=0, random_state=s),
    'TJEPAOD-nonorm': lambda s: TJEPAOD(verbose=0, random_state=s, encoder_norm=False),
    'TJEPAOD-untrained': lambda s: TJEPAOD(verbose=0, random_state=s, epoch_num=0),
}

rows = []
for seed in range(3):
    Xtr, Xte, ytr, yte = generate_data(n_train=600, n_test=300, n_features=2, contamination=0.1,
                                       random_state=seed)
    for name, det in (('KNN', KNN()), ('IForest', IForest(random_state=seed))):
        rows.append({'model': name, 'score': 'default', 'seed': seed,
                     'auc': roc_auc_score(yte, det.fit(Xtr).decision_function(Xte))})
    for name, make in VARIANTS.items():
        clf = make(seed).fit(Xtr)
        for sc in ('pred', 'embed', 'combined'):
            clf.score = sc
            rows.append({'model': name, 'score': sc, 'seed': seed,
                         'auc': roc_auc_score(yte, clf.decision_function(Xte))})
df = pd.DataFrame(rows)
df.to_csv(os.path.join(R, 'toy_two_features.csv'), index=False)

lines = ['| detector | pred | embed | combined |', '|---|---|---|---|']
for name in VARIANTS:
    cells = []
    for sc in ('pred', 'embed', 'combined'):
        a = df[(df.model == name) & (df.score == sc)].auc.values
        cells.append(f'{a.mean():.3f} ({" / ".join(f"{x:.2f}" for x in a)})')
    lines.append(f'| {name} | ' + ' | '.join(cells) + ' |')
for name in ('KNN', 'IForest'):
    a = df[df.model == name].auc.values
    lines.append(f'| {name} | | | {a.mean():.3f} ({" / ".join(f"{x:.2f}" for x in a)}) |')
text = ('#### Two-feature toy problem: mean ROC-AUC over 3 seeds (per-seed values in brackets)\n\n'
        + '\n'.join(lines) + '\n')
open(os.path.join(R, 'toy_two_features.md'), 'w').write(text)
print(text)
