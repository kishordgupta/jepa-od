"""Are the patch embeddings of a next-patch version collapsed? (REPORT.md section 7.11, finding 4)

For TSB-AD series and the versions on the network of `lewm`, untrained and trained: the spread and the effective
rank (participation ratio of the covariance eigenvalues) of the patch embeddings on the training prefix, and the
mean prediction error on the training prefix, on normal patches after it, and on anomalous patches.

    python -I benchmarks/diag_ts_collapse.py --pyod <pyod checkout> --device cuda <TSB-AD csv file> [...]

The committed numbers are from 034_GHL_id_3_Sensor_tr_50000_1st_122001.csv and
086_SVDB_id_3_Medical_tr_533_1st_633.csv of TSB-AD-M.
"""
import argparse, os, sys, types
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument('files', nargs='+')
ap.add_argument('--pyod', default=None, help='PyOD checkout, if pyod is not installed')
ap.add_argument('--versions', default='lewm,subjepa,motionjepa,straight')
ap.add_argument('--device', default='cpu')
args = ap.parse_args()
if args.pyod:
    sys.path.insert(0, os.path.abspath(args.pyod))
# only the time-series modules: importing the package also loads the tabular detectors, which need tqdm
package = types.ModuleType('jepa_od')
package.__path__ = [os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'jepa_od')]
sys.modules['jepa_od'] = package
import torch
from jepa_od.ts_jepa import TSJEPAOD

for path in args.files:
    df = pd.read_csv(path).dropna()
    data, y = df.iloc[:, :-1].values.astype(float), df['Label'].values.astype(int)
    n_train = int(os.path.basename(path)[:-4].split('_')[-3])
    print(f'{os.path.basename(path)}: {len(data)} timestamps, {data.shape[1]} channels, training prefix {n_train}, '
          f'{100 * y.mean():.2f} % anomalous')
    for version in args.versions.split(','):
        for epochs in (0, 10):
            clf = TSJEPAOD(variant=version, epochs=epochs, device=args.device).fit(data[:n_train])
            L, P = clf.window_size, clf.n_tokens_
            windows = clf._windows((data - clf.X_mean_) / clf.X_std_, L, L)     # windows that do not overlap
            x = torch.from_numpy(clf._to_patches(windows)).to(clf._resolve_device())
            with torch.no_grad():
                err, z = clf.model_.multi_mask_errors(x, *clf.model_.eval_mask_set())
            err = err[:, 0].cpu().numpy()                                       # (windows, patches - 1)
            z = z.cpu().numpy().astype(np.float64)
            label = y[:len(windows) * L].reshape(len(windows), P, L // P).max(-1)[:, 1:]
            n_w = n_train // L
            eig = np.clip(np.linalg.eigvalsh(np.cov(z[:n_w].reshape(-1, z.shape[-1]), rowvar=False)), 0, None)
            after = err[n_w:][label[n_w:] == 0]
            print(f'  {version:10s} epochs {epochs:2d}: embedding std {z[:n_w].std((0, 1)).mean():.3f}, effective rank '
                  f'{eig.sum() ** 2 / (eig ** 2).sum():4.1f} of {z.shape[-1]}; error on the training prefix '
                  f'{err[:n_w].mean():.2e}, on normal patches after it {after.mean():.2e}, on anomalous patches '
                  f'{err[label == 1].mean() if (label == 1).any() else float("nan"):.2e}')
