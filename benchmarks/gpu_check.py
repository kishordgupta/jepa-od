"""CPU versus GPU check of ``pyod.models.ts_jepa.TSJEPA``.

Fits every value of ``variant`` on the CPU and on a CUDA device with the
same data and seed, and reports whether the GPU path runs, how close its
scores are to the CPU's, whether a second GPU fit repeats the first, and
how long each fit takes.

Usage (on a machine with a CUDA device)::

    CUDA_VISIBLE_DEVICES=<i> PYTHONPATH=<pyod checkout> \
        python gpu_check.py <out.json> [small|default|large ...]
"""
import json
import pickle
import platform
import sys
import time
import traceback

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from pyod.models.ts_jepa import TSJEPA, _VARIANTS
from pyod.utils.data import generate_ts_data

N_THREADS = 8

SETTINGS = {
    # the setting of pyod/test/test_ts_jepa_variants.py
    'small': dict(
        data=dict(n_train=1000, n_test=600, n_channels=3,
                  contamination=0.05, anomaly_type='both', random_state=1),
        params=dict(window_size=40, patch_len=5, epochs=2,
                    score='combined', random_state=3)),
    # the detector's default parameters
    'default': dict(
        data=dict(n_train=20000, n_test=5000, n_channels=3,
                  contamination=0.05, anomaly_type='both', random_state=1),
        params=dict()),
    # a wider and deeper network on every training window
    'large': dict(
        data=dict(n_train=20000, n_test=5000, n_channels=8,
                  contamination=0.05, anomaly_type='both', random_state=1),
        params=dict(d_model=256, n_heads=8, n_layers=4, batch_size=256,
                    max_train_windows=20000, epochs=5)),
}


def timed_fit(variant, device, X, params):
    cuda = device.startswith('cuda')
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    det = TSJEPA(variant=variant, device=device, **params).fit(X)
    if cuda:
        torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    peak = torch.cuda.max_memory_allocated() / 2 ** 20 if cuda else 0.0
    return det, seconds, peak


def devices_of(det):
    return sorted({t.device.type for t in list(det.model_.parameters())
                   + list(det.model_.buffers())})


def check_variant(variant, X_train, X_test, y_test, params, repeat):
    row = dict(variant=variant)
    cpu, row['cpu_s'], _ = timed_fit(variant, 'cpu', X_train, params)
    gpu, row['gpu_s'], row['gpu_peak_mb'] = timed_fit(
        variant, 'cuda', X_train, params)
    row['cpu_devices'] = devices_of(cpu)
    row['gpu_devices'] = devices_of(gpu)

    s_cpu = cpu.decision_function(X_test)
    s_gpu = gpu.decision_function(X_test)
    row['numpy_out'] = bool(isinstance(s_gpu, np.ndarray)
                            and isinstance(gpu.decision_scores_, np.ndarray))
    row['finite'] = bool(np.isfinite(gpu.decision_scores_).all()
                         and np.isfinite(s_gpu).all()
                         and np.isfinite(gpu.threshold_))
    row['auc_cpu'] = float(roc_auc_score(y_test, s_cpu))
    row['auc_gpu'] = float(roc_auc_score(y_test, s_gpu))
    row['spearman'] = float(spearmanr(s_cpu, s_gpu)[0])
    scale = float(np.abs(s_cpu).max())
    row['max_rel_diff'] = float(np.abs(s_cpu - s_gpu).max() / scale)
    row['train_max_rel_diff'] = float(
        np.abs(cpu.decision_scores_ - gpu.decision_scores_).max()
        / np.abs(cpu.decision_scores_).max())
    row['labels_agree'] = float(
        (cpu.predict(X_test) == gpu.predict(X_test)).mean())

    # the other public methods, on the GPU-fitted detector
    proba = gpu.predict_proba(X_test)
    row['proba_ok'] = bool(proba.shape == (len(X_test), 2)
                           and np.isfinite(proba).all())
    restored = pickle.loads(pickle.dumps(gpu))
    row['pickle_ok'] = bool(np.allclose(restored.decision_function(X_test),
                                        s_gpu))
    row['score_repeat_diff'] = float(
        np.abs(gpu.decision_function(X_test) - s_gpu).max() / scale)

    if repeat:
        again, _, _ = timed_fit(variant, 'cuda', X_train, params)
        row['gpu_refit_rel_diff'] = float(
            np.abs(again.decision_scores_ - gpu.decision_scores_).max()
            / np.abs(gpu.decision_scores_).max())
        row['gpu_refit_allclose'] = bool(np.allclose(
            again.decision_scores_, gpu.decision_scores_,
            rtol=1e-4, atol=1e-5))
        cpu_again, _, _ = timed_fit(variant, 'cpu', X_train, params)
        row['cpu_refit_rel_diff'] = float(
            np.abs(cpu_again.decision_scores_ - cpu.decision_scores_).max()
            / np.abs(cpu.decision_scores_).max())
    return row


def main():
    out_path = sys.argv[1]
    names = sys.argv[2:] or ['small', 'default']
    torch.set_num_threads(N_THREADS)
    assert torch.cuda.is_available(), 'no CUDA device is visible'
    report = dict(
        env=dict(python=platform.python_version(),
                 torch=torch.__version__, cuda=torch.version.cuda,
                 cudnn=torch.backends.cudnn.version(),
                 gpu=torch.cuda.get_device_name(0),
                 numpy=np.__version__, cpu_threads=N_THREADS,
                 platform=platform.platform()),
        settings={})

    # warm the CUDA context up so that the first timing is not the launch
    timed_fit('lejepa', 'cuda', generate_ts_data(
        n_train=400, n_test=100, random_state=0)[0],
        dict(window_size=40, patch_len=5, epochs=1))

    for name in names:
        setting = SETTINGS[name]
        X_train, X_test, _, y_test = generate_ts_data(**setting['data'])
        rows = []
        for variant in _VARIANTS:
            try:
                row = check_variant(variant, X_train, X_test, y_test,
                                    setting['params'],
                                    repeat=(name == 'small'))
            except Exception as exc:
                row = dict(variant=variant, error=repr(exc),
                           traceback=traceback.format_exc())
            rows.append(row)
            print(name, json.dumps({k: (round(v, 6) if isinstance(v, float)
                                        else v) for k, v in row.items()
                                    if k != 'traceback'}), flush=True)
            if 'traceback' in row:
                print(row['traceback'], flush=True)
        report['settings'][name] = rows
        with open(out_path, 'w') as f:
            json.dump(report, f, indent=1)

    # an explicit device string, and 'auto'
    X_train, X_test, _, _ = generate_ts_data(**SETTINGS['small']['data'])
    extra = {}
    for device in ('cuda:0', 'auto'):
        det = TSJEPA(device=device, **SETTINGS['small']['params']).fit(
            X_train)
        extra[device] = devices_of(det)
    report['device_strings'] = extra
    print('device strings', extra, flush=True)
    with open(out_path, 'w') as f:
        json.dump(report, f, indent=1)


if __name__ == '__main__':
    main()
