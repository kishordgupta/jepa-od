"""jepa_od.metrics against TSB-AD's own evaluation code.

Every measure of ``jepa_od/metrics.py`` is computed for (a) random label and score vectors, built to hit the edge
cases (an anomaly at the first or last timestamp, single-point anomalies, tied and constant scores, anomalies next to
each other), and (b) saved score vectors of benchmarks/bench_tsb.py, and compared with ``TSB_AD.evaluation``. It
also compares ``acf_window`` with TSB-AD's ``find_length_rank`` on the data files. Prints the largest absolute
difference per measure.

    python -I benchmarks/check_metrics_tsb.py --tsb <TSB-AD clone> [--results <results folder> --data <data folder>]
        [--random 400] [--real 60] [--max-len 60000] [--jobs 32] [--fixture tests/data/tsb_metrics_fixture.npz]

``--fixture`` writes a few of the random cases with TSB-AD's values, for tests/test_metrics.py.
"""
import argparse, glob, importlib.util, os, sys, time, warnings
from multiprocessing import Pool
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--tsb', required=True, help='clone of https://github.com/TheDatumOrg/TSB-AD')
ap.add_argument('--module', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'jepa_od',
                                                 'metrics.py'))
ap.add_argument('--results', default=None, help='results folder of bench_tsb.py, for the real score vectors')
ap.add_argument('--data', default=None, help='folder with TSB-AD-M/ and TSB-AD-U/, for the window check')
ap.add_argument('--random', type=int, default=400)
ap.add_argument('--real', type=int, default=60)
ap.add_argument('--max-len', type=int, default=60000, help="longest real series (TSB-AD's code is slow)")
ap.add_argument('--jobs', type=int, default=8)
ap.add_argument('--fixture', default=None)
args = ap.parse_args()
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.abspath(args.tsb))
spec = importlib.util.spec_from_file_location('jepa_od_metrics', os.path.abspath(args.module))
ours = importlib.util.module_from_spec(spec); spec.loader.exec_module(ours)
from TSB_AD.evaluation.metrics import get_metrics


def random_case(seed):
    """Labels and scores of one random case; the seed picks the edge cases."""
    rng = np.random.RandomState(seed)
    n = int(rng.choice([60, 200, 1000, 4000]))
    label = np.zeros(n, dtype=int)
    for _ in range(rng.randint(1, 9)):
        start = rng.randint(0, n - 1)
        label[start:start + int(rng.choice([1, 1, 2, 5, 20, n // 10 + 1]))] = 1
    if seed % 5 == 0:
        label[:rng.randint(1, 6)] = 1                    # an anomaly at the first timestamp
    if seed % 7 == 0:
        label[-rng.randint(1, 6):] = 1                   # ... and at the last one
    if label.all():
        label[n // 2] = 0
    kind = seed % 6
    score = rng.rand(n)
    if kind in (1, 2, 3):
        score = score + rng.rand() * 2 * label * rng.rand(n)   # informative
    if kind == 2:
        score = np.round(score, 1)                       # many ties
    if kind == 3:
        score = np.convolve(score, np.ones(9) / 9, mode='same')
    if kind == 4:
        score = (score > 0.9).astype(float)              # two values
    if kind == 5:
        score = np.cumsum(rng.randn(n))                  # a random walk
    low, high = score.min(), score.max()
    score = (score - low) / (high - low) if high > low else np.zeros(n)
    return label, score, int(rng.choice([0, 1, 5, 20, 50, 125]))


def compare(case):
    label, score, window = case
    start = time.perf_counter()
    try:
        ref = get_metrics(score, label, slidingWindow=window)
    except Exception as exc:
        ref = repr(exc)
    ref_s = time.perf_counter() - start
    start = time.perf_counter()
    try:
        here = ours.ts_metrics(label, score, window=window)
    except Exception as exc:
        here = repr(exc)
    return ref, here, ref_s, time.perf_counter() - start


def real_case(path):
    group, tag, stem = path[:-4].split(os.sep)[-3:]
    cached = np.load(os.path.join(args.results, 'cache', group.split('-')[0], stem + '.npz'))
    saved = np.load(path)
    name = [k for k in ('pred', 'embed', 'combined', 'score') if k in saved.files][hash(stem) % 3 % len(saved.files)]
    score = np.nan_to_num(saved[name].astype(np.float64))
    low, high = score.min(), score.max()
    score = (score - low) / (high - low) if high > low else np.zeros_like(score)
    return cached['label'].astype(int), score, int(cached['window'])


def report(title, cases, pool):
    out = pool.map(compare, cases, chunksize=1)
    worst = {name: 0.0 for name in ours.METRICS}
    failed = 0
    for ref, here, _, _ in out:
        if isinstance(ref, str) or isinstance(here, str):
            failed += 1
            if isinstance(ref, str) != isinstance(here, str):
                print('  only one side failed:', str(ref)[:90], '|', str(here)[:90])
            continue
        for name in ours.METRICS:
            a, b = ref[name], here[name]
            if not (np.isnan(a) and np.isnan(b)):
                worst[name] = max(worst[name], abs(a - b) if np.isfinite(a) and np.isfinite(b) else np.inf)
    print(f'{title}: {len(out)} cases, {failed} where both sides raise; TSB-AD {sum(o[2] for o in out):.0f} s, '
          f'here {sum(o[3] for o in out):.1f} s')
    print('  largest difference: ' + ', '.join(f'{k} {v:.1e}' for k, v in worst.items()))
    return out


def window_case(path):
    import pandas as pd
    from TSB_AD.utils.slidingWindows import find_length_rank
    data = pd.read_csv(path).dropna().iloc[:, 0].values.astype(float)
    return int(find_length_rank(data.reshape(-1, 1), rank=1)), ours.acf_window(data)


if __name__ == '__main__':
    with Pool(args.jobs) as pool:
        cases = [random_case(seed) for seed in range(args.random)]
        out = report('random', cases, pool)
        if args.fixture:
            keep = [i for i in range(0, len(cases), max(1, len(cases) // 40)) if not isinstance(out[i][0], str)]
            os.makedirs(os.path.dirname(os.path.abspath(args.fixture)), exist_ok=True)
            np.savez_compressed(
                args.fixture, n=len(keep), names=np.array(ours.METRICS),
                **{f'label{j}': cases[i][0].astype(np.int8) for j, i in enumerate(keep)},
                **{f'score{j}': cases[i][1] for j, i in enumerate(keep)},
                **{f'window{j}': cases[i][2] for j, i in enumerate(keep)},
                **{f'ref{j}': np.array([out[i][0][name] for name in ours.METRICS]) for j, i in enumerate(keep)})
            print('wrote', args.fixture, 'with', len(keep), 'cases')
        if args.results and args.real:
            found = sorted(glob.glob(os.path.join(args.results, 'scores', '*', '*', '*.npz')))
            found = [f for f in found if not f.endswith('.tmp.npz') and os.path.getsize(f) < args.max_len * 12]
            rng = np.random.RandomState(0)
            pick = [found[i] for i in rng.choice(len(found), min(args.real, len(found)), replace=False)]
            report('saved score vectors', [real_case(p) for p in pick], pool)
        if args.data:
            files = sorted(glob.glob(os.path.join(args.data, 'TSB-AD-*', '*.csv')))
            rng = np.random.RandomState(1)
            pick = [files[i] for i in rng.choice(len(files), min(300, len(files)), replace=False)]
            pairs = pool.map(window_case, pick, chunksize=4)
            print(f'window: {sum(a == b for a, b in pairs)} of {len(pairs)} files agree with find_length_rank; '
                  f'differing: {[p for p in pairs if p[0] != p[1]][:8]}')
