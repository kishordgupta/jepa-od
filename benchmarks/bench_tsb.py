"""TSJEPA on the TSB-AD benchmark (Liu and Paparrizos, NeurIPS 2024): anomaly scores.

TSB-AD has 180 multivariate (TSB-AD-M) and 350 univariate (TSB-AD-U) evaluation series, and publishes the
VUS-PR of its reference detectors on every one of them. This script follows the protocol of TSB-AD's
``benchmark_exp/Run_Detector_M.py`` for a semi-supervised detector: fit on the training prefix that the
file name gives (``..._tr_<n>_1st_<m>.csv``; it ends before the first anomaly in 476 of the 530 series), then score
the whole series. It writes the scores of the three
score types; ``benchmarks/eval_tsb.py`` turns them into TSB-AD's metrics.

The detector is the PyOD port (``pyod.models.ts_jepa.TSJEPA`` of the checkout given with ``--pyod``), with the
class defaults: no parameter is tuned on TSB-AD. Three things differ from the class, for speed on long series
(``Detector`` below): windows are scored ``--score-batch`` at a time instead of ``batch_size`` (64) at a time, they
are cut from a view of the series, and the Mahalanobis distance is one matrix product. Training is the class's own,
and the scores change by rounding only (``--check`` measures it against the class).

    python -I benchmarks/bench_tsb.py --pyod <pyod checkout> --data <folder with TSB-AD-M/ and TSB-AD-U/> \
        --lists <TSB-AD>/Datasets/File_List --suite M --variant lejepa --out <results folder>

The data files are third-party (https://www.thedatum.org/datasets/TSB-AD-M.zip and TSB-AD-U.zip); ``-I`` keeps
Python from importing anything from their folders.
"""
import argparse, csv, os, re, subprocess, sys, time, traceback
import numpy as np, pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

ap = argparse.ArgumentParser()
ap.add_argument('--pyod', required=True, help='PyOD checkout that has pyod/models/ts_jepa.py')
ap.add_argument('--data', required=True, help='folder with TSB-AD-M/ and TSB-AD-U/')
ap.add_argument('--lists', required=True, help="TSB-AD's Datasets/File_List folder")
ap.add_argument('--suite', choices=('M', 'U'), required=True)
ap.add_argument('--split', default='Eva', choices=('Eva', 'Tuning'))
ap.add_argument('--variant', default='lejepa')
ap.add_argument('--epochs', type=int, default=None, help='default: the class default (10); 0 = untrained control')
ap.add_argument('--seed', type=int, default=None, help='random_state; default: the class default (42)')
ap.add_argument('--mode', default='semi', choices=('semi', 'unsup'),
                help='semi: fit on the training prefix (the protocol); unsup: fit on the whole series')
ap.add_argument('--window', type=int, default=None, help='window_size; patch_len is a tenth of it')
ap.add_argument('--params', default='', help='further TSJEPA parameters, e.g. d_model=128,n_layers=4')
ap.add_argument('--rawpatch', action='store_true',
                help='control without a network: the embedding of a patch is the patch itself, so the embed score '
                     'is the Mahalanobis distance of raw patches to those of the training prefix (pred is constant)')
ap.add_argument('--unit-const', action='store_true',
                help='ablation: a channel that is constant in the training data is scaled by 1 when scoring, as '
                     'sklearn.preprocessing.StandardScaler does, instead of by the 1e-8 the detector adds to the '
                     'standard deviation')
ap.add_argument('--tag', default=None, help='name of the output folder; default: built from the options')
ap.add_argument('--device', default='cuda')
ap.add_argument('--gpus', default=None,
                help='shared machine: GPU indices to choose from, e.g. 2,5,6,7. The run takes one on which no other '
                     'job computes (it waits for one), holds memory on that one only, and looks again before '
                     'every file: when another job has come to its GPU it stops with exit code 75, which frees '
                     'its memory there, and is to be started again (benchmarks/run_tsb.sh does)')
ap.add_argument('--resident', default='bench_tsb|VLLM|vllm|docling|embed|comfyui',
                help='with --gpus: pattern of the processes that do not make a GPU busy (these runs, idle services)')
ap.add_argument('--score-batch', type=int, default=2048)
ap.add_argument('--threads', type=int, default=4)
ap.add_argument('--only', default=None, help='run only the files whose name contains this text')
ap.add_argument('--limit', type=int, default=None)
ap.add_argument('--check', action='store_true', help='also compare with the public decision_function')
ap.add_argument('--out', required=True)
args = ap.parse_args()

sys.path.insert(0, os.path.abspath(args.pyod))
if args.gpus:
    # only these GPUs exist for this process, so it never holds memory on another one
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
import torch
from pyod.models.ts_jepa import TSJEPA
from pyod.models._ts_utils import validate_ts_input

torch.set_num_threads(args.threads)


class Detector(TSJEPA):
    """TSJEPA with cheaper scoring of long series, in ``fit`` as well. Training is untouched."""

    def _raw(self, patches, device, ctx, tgt):
        if args.rawpatch:
            for start in range(0, len(patches), args.score_batch):
                batch = patches[start:start + args.score_batch].numpy()
                yield np.zeros((len(batch),) + tuple(tgt.shape), dtype=np.float32), batch
            return
        # 2048 windows a time: 4.5 times faster on a GPU for a series of 200,000 timestamps
        train_batch, self.batch_size = self.batch_size, max(self.batch_size, args.score_batch)
        try:
            yield from super()._raw(patches, device, ctx, tgt)
        finally:
            self.batch_size = train_batch

    @staticmethod
    def _windows(X, window_size, step):
        # the same array as the class builds by indexing, from a view: a third of the time, half the memory
        view = sliding_window_view(X.astype(np.float32), window_size, axis=0)[::step]
        return np.ascontiguousarray(view.transpose(0, 2, 1))

    def _mahalanobis(self, h_all):
        # the class contracts three arrays in one einsum, which does not use BLAS
        diff = h_all - self.emb_mean_
        return np.einsum('...i,...i->...', diff @ self.emb_prec_, diff)


KW = dict(variant=args.variant, device=args.device)
if args.epochs is not None:
    KW['epochs'] = args.epochs
if args.seed is not None:
    KW['random_state'] = args.seed
if args.window is not None:
    KW.update(window_size=args.window, patch_len=args.window // 10)
if args.rawpatch:
    KW.update(variant='lejepa', epochs=0)
for item in filter(None, args.params.split(',')):
    key, value = item.split('=')
    KW[key] = float(value) if '.' in value or 'e' in value else int(value)
tag = args.tag or '_'.join(filter(None, [
    'rawpatch' if args.rawpatch else args.variant,
    None if args.rawpatch else 'trained' if args.epochs is None else f'e{args.epochs}',
    f's{TSJEPA().random_state if args.seed is None else args.seed}', args.mode,
    None if args.window is None else f'w{args.window}', args.params.replace(',', '+').replace('=', '') or None,
    'unitconst' if args.unit_const else None]))
out_dir = os.path.join(args.out, 'scores', f'{args.suite}-{args.split}', tag)
os.makedirs(out_dir, exist_ok=True)
files = pd.read_csv(os.path.join(args.lists, f'TSB-AD-{args.suite}-{args.split}.csv'))['file_name'].tolist()
if args.only:
    files = [f for f in files if args.only in f]
files = files[:args.limit]
cuda = args.device.startswith('cuda')
FIELDS = ['file', 'n', 'n_channels', 'train_len', 'fit_s', 'score_s', 'pred_invalid', 'nonfinite', 'gpu_peak_mb',
          'error']
log_path = os.path.join(out_dir, '_runs.csv')
if not os.path.exists(log_path):
    with open(log_path, 'w', newline='') as f:
        csv.writer(f).writerow(FIELDS)


def sync():
    if cuda and torch.cuda.is_initialized():
        torch.cuda.synchronize(KW['device'])


def smi(what, fields):
    out = subprocess.run(['nvidia-smi', f'--query-{what}={fields}', '--format=csv,noheader,nounits'],
                         capture_output=True, text=True, timeout=60).stdout
    return [[x.strip() for x in line.split(',')] for line in out.splitlines() if line.strip()]


def pick_gpu(current):
    """A GPU of --gpus that has no job of somebody else: the current one if it still is such a GPU, else the one
    with the fewest of these runs; None when all are busy."""
    index = {uuid: int(i) for i, uuid in smi('gpu', 'index,uuid')}
    ours = {int(g): 0 for g in args.gpus.split(',')}
    busy = set()
    for uuid, pid, name in smi('compute-apps', 'gpu_uuid,pid,process_name'):
        try:
            with open(f'/proc/{pid}/cmdline', 'rb') as f:
                name += ' ' + f.read().replace(b'\0', b' ').decode(errors='replace')
        except OSError:
            pass
        gpu = index.get(uuid)
        if 'bench_tsb' in name:
            ours[gpu] = ours.get(gpu, 0) + (int(pid) != os.getpid())
        elif not re.search(args.resident, name):
            busy.add(gpu)
    free = [g for g in ours if g not in busy and g in index.values()]
    if current in free or not free:
        return current if free else None
    # runs that start together all see the same counts: the process id spreads them
    fewest = [g for g in free if ours[g] == min(ours[f] for f in free)]
    return fewest[os.getpid() % len(fewest)]


last_pick = [0.0, None]


def choose_device():
    """Take a GPU before the first file; later, at most every 20 seconds, stop if another job has come to it."""
    if not args.gpus:
        return
    if last_pick[1] is None:
        while True:
            gpu = pick_gpu(None)
            if gpu is not None:
                break
            print('  every GPU of --gpus has another job on it; waiting', flush=True)
            time.sleep(120)
        last_pick[:] = [time.time(), gpu]
        KW['device'] = 'cuda:%d' % args.gpus.split(',').index(str(gpu))   # the place among the visible GPUs
        torch.cuda.set_device(KW['device'])   # or PyTorch also opens the first visible GPU
        print(f'  on GPU {gpu}', flush=True)
    elif time.time() - last_pick[0] > 20:
        if pick_gpu(last_pick[1]) != last_pick[1]:
            print(f'  another job has come to GPU {last_pick[1]}: stopping to free it', flush=True)
            sys.exit(75)
        last_pick[0] = time.time()


def run(name):
    df = pd.read_csv(os.path.join(args.data, f'TSB-AD-{args.suite}', name)).dropna()
    data = df.iloc[:, :-1].values.astype(float)
    train_len = int(name[:-4].split('_')[-3])
    row = dict(file=name, n=len(data), n_channels=data.shape[1], train_len=train_len)
    choose_device()
    if cuda and torch.cuda.is_initialized():
        torch.cuda.reset_peak_memory_stats(KW['device'])
    start = time.perf_counter()
    X_train = data[:train_len] if args.mode == 'semi' else data
    det = Detector(**KW).fit(X_train)
    sync(); row['fit_s'] = time.perf_counter() - start
    if args.unit_const:
        # X_std_ is the standard deviation plus 1e-8; the training data of such a channel are all zero either way
        det.X_std_ = np.where(det.X_std_ <= 1.5e-8, 1.0, det.X_std_)

    # one pass per component; decision_function would repeat the first for every score type
    start = time.perf_counter()
    X_std = (validate_ts_input(data) - det.X_mean_) / det.X_std_
    device = det._resolve_device()
    det.score = 'pred'; s_pred, ok_pred = det._score_series(X_std, device)
    det.score = 'embed'; s_emb, ok_emb = det._score_series(X_std, device)
    sync(); row['score_s'] = time.perf_counter() - start
    assert ok_emb.all()
    # what decision_function returns for score='pred' (the threshold of fit is on that scale) ...
    pred = np.where(ok_pred, s_pred, det.threshold_)
    # ... and what _score_series computes for score='combined'
    z_pred = (s_pred - det.comp_mean_[0]) / det.comp_std_[0]
    z_emb = (s_emb - det.comp_mean_[1]) / det.comp_std_[1]
    combined = np.where(ok_pred, 0.5 * (z_pred + z_emb), z_emb)
    if args.check:
        det.score = 'pred'; assert np.array_equal(det.decision_function(data), pred)
        det.score = 'combined'; ref, ok = det._score_series(X_std, device)
        assert ok.all() and np.allclose(ref, combined, rtol=1e-6, atol=1e-9), np.abs(ref - combined).max()
        det.score = 'pred'
        # the class itself, which scores batch_size windows at a time
        start = time.perf_counter()
        plain = TSJEPA(**KW).fit(X_train)
        sync(); plain_fit = time.perf_counter() - start
        ref = plain.decision_function(data)
        sync(); plain_all = time.perf_counter() - start
        from scipy.stats import spearmanr
        print('  check ok: against the class, max difference %.1e of the score range, Spearman %.6f; the class '
              'takes %.1f s to fit and %.1f s to score once' % (
                  np.abs(ref - pred).max() / (ref.max() - ref.min()), spearmanr(ref, pred)[0], plain_fit,
                  plain_all - plain_fit), flush=True)
    scores = dict(pred=pred, embed=s_emb, combined=combined)
    row['pred_invalid'] = int((~ok_pred).sum())
    row['nonfinite'] = int(sum((~np.isfinite(s)).sum() for s in scores.values()))
    row['gpu_peak_mb'] = round(torch.cuda.max_memory_allocated(KW['device']) / 2 ** 20, 1) if cuda else 0
    tmp = os.path.join(out_dir, name[:-4] + '.tmp.npz')
    np.savez(tmp, **{k: v.astype(np.float32) for k, v in scores.items()})
    os.replace(tmp, os.path.join(out_dir, name[:-4] + '.npz'))
    del det
    if cuda:
        torch.cuda.empty_cache()   # hold no more than the context between files
    return row


for i, name in enumerate(files):
    if os.path.exists(os.path.join(out_dir, name[:-4] + '.npz')):
        continue
    try:
        row = run(name)
    except Exception as exc:
        traceback.print_exc()
        row = dict(file=name, error=repr(exc)[:300])
    with open(log_path, 'a', newline='') as f:
        csv.DictWriter(f, FIELDS).writerow({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()})
    print(f'{tag} {i + 1}/{len(files)} {name} ' + ' '.join(
        f'{k}={row[k]:.1f}' if isinstance(row.get(k), float) else f'{k}={row.get(k)}'
        for k in ('n', 'n_channels', 'train_len', 'fit_s', 'score_s', 'error') if row.get(k) is not None), flush=True)
print('ALL DONE')
