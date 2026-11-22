"""Fit and scoring time of every TSJEPAOD version on one synthetic series, one after the other with 2 threads
(REPORT.md section 7.8; the series has a seed that the benchmark does not use).

    python benchmarks/time_ts_versions.py [version,version,...]
"""
import os, sys, time, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pyod.utils.data import generate_ts_data
from jepa_od import TSJEPAOD
from jepa_od.ts_jepa import _VARIANTS
torch.set_num_threads(2)
names = sys.argv[1].split(',') if len(sys.argv) > 1 else list(_VARIANTS)
Xtr, Xte, _, _ = generate_ts_data(n_train=3000, n_test=1500, n_channels=3, contamination=0.05,
                                  anomaly_type='both', random_state=100)
for v in names:
    t0 = time.time()
    clf = TSJEPAOD(variant=v, window_size=60, patch_len=5, score='combined', random_state=100).fit(Xtr)
    t1 = time.time(); s = clf.decision_function(Xte); t2 = time.time()
    print('%-10s fit %6.1fs  score %5.1fs  finite %s' % (v, t1 - t0, t2 - t1, np.isfinite(s).all()), flush=True)
