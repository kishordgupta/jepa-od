"""Example: JEPAOD and TSJEPAOD side by side with PyOD baselines."""
import os, sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn.metrics import roc_auc_score, average_precision_score
from pyod.utils.data import generate_data, generate_ts_data
from pyod.models.auto_encoder import AutoEncoder
from pyod.models.iforest import IForest
from pyod.models.ts_lstm import LSTMAD
from jepa_od import JEPAOD, TSJEPAOD

def report(name, y, s):
    print(f'{name:28s} ROC-AUC={roc_auc_score(y, s):.3f}  AP={average_precision_score(y, s):.3f}')

print('--- tabular ---')
Xtr, Xte, ytr, yte = generate_data(n_train=2000, n_test=800, n_features=20, contamination=0.1, random_state=42)
for name, clf in [('JEPAOD', JEPAOD(epoch_num=20, verbose=0)),
                  ('AutoEncoder', AutoEncoder(epoch_num=20, verbose=0)),
                  ('IForest', IForest(random_state=0))]:
    clf.fit(Xtr); report(name, yte, clf.decision_function(Xte))

print('--- time series (3 channels, point + subsequence anomalies) ---')
Xtr, Xte, ytr, yte = generate_ts_data(n_train=3000, n_test=1500, n_channels=3, anomaly_type='both', random_state=42)
for name, clf in [('TSJEPAOD block/combined', TSJEPAOD(window_size=60, patch_len=5, epochs=15, score='combined')),
                  ('TSJEPAOD forecast/pred', TSJEPAOD(window_size=60, patch_len=5, epochs=15, mask_mode='forecast')),
                  ('LSTMAD', LSTMAD(window_size=60, epochs=15))]:
    clf.fit(Xtr); report(name, yte, clf.decision_function(Xte))
