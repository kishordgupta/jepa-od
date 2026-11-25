"""Detectors of TSB-AD itself, re-run on any of its file lists: anomaly scores.

TSB-AD publishes per-file numbers for its evaluation lists only. For the other lists (the tuning lists, and the 472
univariate series in neither list) this script runs a detector of TSB-AD's own code (the clone given with ``--tsb``)
with the hyperparameters TSB-AD lists as optimal, the way its ``benchmark_exp/Run_Detector_U.py`` does, and writes
the scores where ``benchmarks/eval_tsb.py`` finds them. Run on an evaluation list, it shows how far today's code
reproduces the published column (``benchmarks/make_tsb_confirm_tables.py`` reports that).

    python -I benchmarks/bench_tsb_ref.py --tsb <TSB-AD clone> --data <folder with TSB-AD-M/ and TSB-AD-U/> \
        --suite U --split Rest --detector Sub_PCA --out <results folder> --jobs 8

Output: <results>/scores/<suite>-<split>/ref_<detector>[_<params>]/<file>.npz with one vector, ``score``.
"""
import argparse, csv, os, sys, time, traceback

ap = argparse.ArgumentParser()
ap.add_argument('--tsb', required=True, help='clone of https://github.com/TheDatumOrg/TSB-AD')
ap.add_argument('--data', required=True, help='folder with TSB-AD-M/ and TSB-AD-U/')
ap.add_argument('--suite', choices=('M', 'U'), required=True)
ap.add_argument('--split', default='Eva', choices=('Eva', 'Tuning', 'Rest'))
ap.add_argument('--detector', required=True, help="a name of TSB-AD's model_wrapper, e.g. Sub_PCA or PCA")
ap.add_argument('--params', default='', help="parameters on top of TSB-AD's optimal ones, e.g. slidingWindow=1")
ap.add_argument('--jobs', type=int, default=4)
ap.add_argument('--threads', type=int, default=4)
ap.add_argument('--limit', type=int, default=None)
ap.add_argument('--out', required=True)
args = ap.parse_args()
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = str(args.threads)
sys.path.insert(0, os.path.abspath(args.tsb))
from multiprocessing import Pool
import numpy as np, pandas as pd

FIELDS = ['file', 'n', 'n_channels', 'train_len', 'seconds', 'error']
tag = 'ref_' + '_'.join(filter(None, [args.detector, args.params.replace(',', '+').replace('=', '')]))
out_dir = os.path.join(args.out, 'scores', f'{args.suite}-{args.split}', tag)


def file_list(split):
    path = os.path.join(args.tsb, 'Datasets', 'File_List', f'TSB-AD-{args.suite}-{split}.csv')
    return pd.read_csv(path)['file_name'].tolist()


def run(name):
    from TSB_AD.HP_list import Optimal_Multi_algo_HP_dict, Optimal_Uni_algo_HP_dict
    from TSB_AD.model_wrapper import (Semisupervise_AD_Pool, Unsupervise_AD_Pool, run_Semisupervise_AD,
                                      run_Unsupervise_AD)
    params = dict((Optimal_Multi_algo_HP_dict if args.suite == 'M' else Optimal_Uni_algo_HP_dict)[args.detector])
    for item in filter(None, args.params.split(',')):
        key, value = item.split('=')
        params[key] = float(value) if '.' in value else int(value)
    df = pd.read_csv(os.path.join(args.data, f'TSB-AD-{args.suite}', name)).dropna()
    data = df.iloc[:, :-1].values.astype(float)
    train_len = int(name[:-4].split('_')[-3])
    row = dict(file=name, n=len(data), n_channels=data.shape[1], train_len=train_len)
    start = time.perf_counter()
    try:
        if args.detector in Semisupervise_AD_Pool:
            score = run_Semisupervise_AD(args.detector, data[:train_len], data, **params)
        else:
            assert args.detector in Unsupervise_AD_Pool, args.detector
            score = run_Unsupervise_AD(args.detector, data, **params)
        if not isinstance(score, np.ndarray):    # TSB-AD's wrappers return the error message
            raise RuntimeError(str(score))
        assert score.shape == (len(data),), score.shape
        row['seconds'] = round(time.perf_counter() - start, 3)
        tmp = os.path.join(out_dir, name[:-4] + '.tmp.npz')
        np.savez(tmp, score=score.astype(np.float64))
        os.replace(tmp, os.path.join(out_dir, name[:-4] + '.npz'))
    except Exception as exc:
        traceback.print_exc()
        row['error'] = repr(exc)[:300]
    return row


if __name__ == '__main__':
    os.makedirs(out_dir, exist_ok=True)
    if args.split == 'Rest':
        evaluation = set(file_list('Eva'))
        files = [f for f in file_list('Eva-Full') if f not in evaluation]
    else:
        files = file_list(args.split)
    files = [f for f in files[:args.limit] if not os.path.exists(os.path.join(out_dir, f[:-4] + '.npz'))]
    log_path = os.path.join(out_dir, '_runs.csv')
    if not os.path.exists(log_path):
        with open(log_path, 'w', newline='') as f:
            csv.writer(f).writerow(FIELDS)
    # the longest series first, so that the pool ends evenly
    files.sort(key=lambda f: -os.path.getsize(os.path.join(args.data, f'TSB-AD-{args.suite}', f)))
    with Pool(args.jobs) as pool:
        for i, row in enumerate(pool.imap_unordered(run, files, chunksize=1)):
            with open(log_path, 'a', newline='') as f:
                csv.DictWriter(f, FIELDS).writerow(row)
            print(f'{tag} {i + 1}/{len(files)} ' + ' '.join(f'{k}={v}' for k, v in row.items()), flush=True)
    print('ALL DONE')
