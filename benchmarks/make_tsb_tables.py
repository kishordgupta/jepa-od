"""Tables for the TSB-AD benchmark of the sixteen JEPA versions of TSJEPA (REPORT.md section 7.9).

Reads results/tsb/metrics/{M,U}-Eva/<run>.csv (benchmarks/eval_tsb.py), results/tsb/runs/ (the timing logs of
benchmarks/bench_tsb.py) and TSB-AD's published per-file VUS-PR (data/tsb_published/, fetched by
benchmarks/download_data.py --tsb-published). Writes results/tsb_tables.md and refreshes REPORT.md between the
TSB markers.

The 62 tests below were fixed when four of the 70 runs had been evaluated (the untrained default version on both
sets, and trained 'ijepa' and 'tsjepa' on TSB-AD-M): for each version and set, trained against its zero-epoch
control; for each of the fifteen other versions and set, trained against the trained default version 'lejepa'.
All on VUS-PR of the default score `pred`.
"""
import os, sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _tables import md, fmt_p, signed_rank, holm, boot_ci

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, 'results', 'tsb')
PUB = os.path.join(ROOT, 'data', 'tsb_published')
VERSIONS = ['lejepa', 'ijepa', 'tsjepa', 'tjepa', 'vjepa', 'vjepa21', 'cjepa', 'sjepa', 'salt', 'vicreg', 'kerjepa',
            'lpjepa', 'lambdajepa', 'lenepa', 'lewm', 'cfjepa']
SETS = {'M': ('TSB-AD-M', 'multi'), 'U': ('TSB-AD-U', 'uni')}
META = ['ts_len', 'anomaly_len', 'num_anomaly', 'avg_anomaly_len', 'anomaly_ratio', 'point_anomaly', 'seq_anomaly']
SCORES = ('pred', 'embed', 'combined')


def run(version, kind='trained', seed=42, mode='semi'):
    return f'{version}_{kind}_s{seed}_{mode}' if kind else f'{version}_s{seed}_{mode}'


published = {s: pd.read_csv(os.path.join(PUB, f'{name}_mergedTable_VUS-PR.csv')).set_index('file')
             for s, (_, name) in SETS.items()}
methods = {s: [c for c in published[s].columns if c not in META] for s in SETS}
cache = {}


def metric(s, tag, score='pred', col='VUS-PR'):
    """Per-file values of one run, in the order of the published table; None when the run is not complete."""
    if (s, tag) not in cache:
        path = os.path.join(R, 'metrics', f'{s}-Eva', tag + '.csv')
        cache[s, tag] = pd.read_csv(path) if os.path.exists(path) else None
    d = cache[s, tag]
    if d is None:
        return None
    x = d[d.score == score].drop_duplicates('file').set_index('file')[col].astype(float)
    return x.reindex(published[s].index) if set(published[s].index) <= set(x.index) else None


def mean(s, tag, score='pred', col='VUS-PR'):
    x = metric(s, tag, score, col)
    return '' if x is None else f'{x.mean():.3f}'


def pair(s, version, score, col='VUS-PR'):
    a, b = mean(s, run(version), score, col), mean(s, run(version, 'e0'), score, col)
    return f'{a} / {b}' if a or b else ''


def seconds(s, tag):
    path = os.path.join(R, 'runs', f'{s}-Eva', tag + '.csv')
    if not os.path.exists(path):
        return ''
    d = pd.read_csv(path).drop_duplicates('file', keep='last')
    return f'{(d.fit_s.sum() + d.score_s.sum()) / 60:.0f}'


def rank_among(s, value):
    """Place of a mean VUS-PR among the published means (1 = above all of them)."""
    return 1 + int((published[s][methods[s]].mean() > value).sum())


out = []
missing = [f'{s}: {run(v, k)}' for s in SETS for v in VERSIONS for k in ('trained', 'e0') if metric(s, run(v, k)) is None]
out.append('Coverage: ' + ', '.join(f'{SETS[s][0]}-Eva {len(published[s])} series' for s in SETS)
           + ('; every run has every series.' if not missing else f'; {len(missing)} RUNS ARE MISSING: '
              + ', '.join(missing)))

# table 1: the sixteen versions
for s, (name, _) in SETS.items():
    rows = [{'version': v, **{sc: pair(s, v, sc) for sc in SCORES}, 'AUC-PR of pred': pair(s, v, 'pred', 'AUC-PR'),
             'AUC-ROC of pred': pair(s, v, 'pred', 'AUC-ROC'), 'minutes': seconds(s, run(v))} for v in VERSIONS]
    out.append(f'#### {name}-Eva ({len(published[s])} series): mean VUS-PR of the three score types, trained / '
               'zero-epoch control (the default score is `pred`; minutes: fitting and scoring the whole set once, '
               'trained, on a shared H100)\n\n' + md(pd.DataFrame(rows).set_index('version')))

# table 2: next to the published detectors
rows = []
candidates = [('default detector: lejepa, trained, `pred`', run('lejepa'), 'pred'),
              ('lejepa, untrained, `pred`', run('lejepa', 'e0'), 'pred'),
              ('lejepa, trained, `embed`', run('lejepa'), 'embed'),
              ('lejepa, untrained, `embed`', run('lejepa', 'e0'), 'embed'),
              ('lejepa, trained, `combined`', run('lejepa'), 'combined'),
              ('no network: raw patches, `embed`', run('rawpatch', None), 'embed')]
for v in VERSIONS[1:]:
    candidates.append((f'{v}, trained, `pred`', run(v), 'pred'))
for label, tag, score in candidates:
    row = {'detector': label}
    for s, (name, _) in SETS.items():
        x = metric(s, tag, score)
        if x is None:
            row[f'{name}: VUS-PR'] = row[f'{name}: place'] = row[f'{name}: vs best published'] = ''
            continue
        best = published[s][methods[s]].mean().idxmax()
        d = x - published[s][best]
        row[f'{name}: VUS-PR'] = f'{x.mean():.3f}'
        row[f'{name}: place'] = f'{rank_among(s, x.mean())} of {len(methods[s]) + 1}'
        row[f'{name}: vs best published'] = (f'{d.mean():+.3f}, {int((d > 0).sum())} / {int((d < 0).sum())} series, '
                                             f'p {fmt_p(signed_rank(d))}')
    rows.append(row)
tops = '; '.join(f'{SETS[s][0]}: ' + ', '.join(f'{k} {v:.3f}' for k, v in
                 published[s][methods[s]].mean().sort_values(ascending=False).head(5).items()) for s in SETS)
out.append('#### Next to the detectors TSB-AD published (same files, same VUS-PR code; their numbers are taken from '
           'the TSB-AD repository, not re-run). Place: where the mean VUS-PR falls among the published means '
           f'({len(methods["M"])} detectors on TSB-AD-M, {len(methods["U"])} on TSB-AD-U). Last columns: difference '
           'to the best published detector of the set, series better / worse, and the p-value of a two-sided '
           f'Wilcoxon signed-rank test (not corrected). Published top five: {tops}\n\n'
           + md(pd.DataFrame(rows).set_index('detector')))

# table 3: the tests fixed in advance
rows = []
for kind in ('training', 'version'):
    for v in VERSIONS:
        if kind == 'version' and v == 'lejepa':
            continue
        for s, (name, _) in SETS.items():
            a = metric(s, run(v))
            b = metric(s, run(v, 'e0')) if kind == 'training' else metric(s, run('lejepa'))
            if a is None or b is None:
                continue
            d = a - b
            lo, hi = boot_ci(d.values)
            rows.append({'A': f'{v}, trained', 'B': f'{v}, untrained' if kind == 'training' else 'lejepa, trained',
                         'set': name, 'VUS-PR A / B': f'{a.mean():.3f} / {b.mean():.3f}',
                         'A − B [95 % CI]': f'{d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]',
                         'A / B better': f'{int((d > 0).sum())} / {int((d < 0).sum())}', '_p': signed_rank(d)})
if rows:
    t3 = pd.DataFrame(rows)
    t3['p'] = t3['_p'].map(fmt_p)
    t3['p (Holm)'] = [fmt_p(p) for p in holm(t3['_p'].values)]
    t3 = t3.drop(columns='_p')
    t3.index = range(1, len(t3) + 1); t3.index.name = ''
    out.append(f'#### Paired tests on VUS-PR of the default score `pred` (two-sided Wilcoxon signed-rank over series; '
               f'Holm correction over the {len(t3)} tests; CI: bootstrap over series)\n\n' + md(t3))

# table 4: by source dataset, the default version
for s, (name, _) in SETS.items():
    cols = {'trained `pred`': metric(s, run('lejepa')), 'untrained `pred`': metric(s, run('lejepa', 'e0')),
            'trained `embed`': metric(s, run('lejepa'), 'embed'),
            'untrained `embed`': metric(s, run('lejepa', 'e0'), 'embed'),
            'raw patches': metric(s, run('rawpatch', None), 'embed')}
    cols = {k: v for k, v in cols.items() if v is not None}
    if not cols:
        continue
    best = published[s][methods[s]].mean().idxmax()
    d = pd.DataFrame(cols)
    d[f'published {best}'] = published[s][best]
    source = pd.Series(d.index.str.split('_').str[1], index=d.index)
    d['best published of the dataset'] = published[s][methods[s]].groupby(source).transform('mean').max(axis=1)
    t = d.groupby(source).mean()
    t.insert(0, 'series', source.value_counts())
    t = t.sort_values('series', ascending=False)
    t.loc['all series'] = [len(d)] + list(d.mean())
    t['series'] = t['series'].astype(int)
    for c in t.columns[1:]:
        t[c] = t[c].map(lambda x: f'{x:.3f}')
    t.index.name = 'dataset'
    out.append(f'#### {name}-Eva by source dataset: mean VUS-PR of the default version `lejepa` (the last column is '
               'the best of the published detectors on that dataset, a different detector from row to row)\n\n'
               + md(t))

# table 5: seeds and the unsupervised protocol
rows = []
for label, tag in [('seed 42 (the class default)', run('lejepa')), ('seed 0', run('lejepa', seed=0)),
                   ('seed 1', run('lejepa', seed=1)),
                   ('seed 42, fitted on the whole series', run('lejepa', mode='unsup'))]:
    row = {'run of lejepa, trained': label}
    for s, (name, _) in SETS.items():
        for sc in ('pred', 'embed'):
            row[f'{name}: {sc}'] = mean(s, tag, sc)
    rows.append(row)
out.append('#### Seeds, and fitting on the whole series instead of the training prefix: mean VUS-PR\n\n'
           + md(pd.DataFrame(rows).set_index('run of lejepa, trained')))

text = '\n\n'.join(out) + '\n'
open(os.path.join(ROOT, 'results', 'tsb_tables.md'), 'w').write(text)
print(text)

narr = os.path.join(ROOT, 'results', 'tsb_narrative.md')
rp = os.path.join(ROOT, 'REPORT.md')
rep = open(rp).read()
B, E = '<!-- TSB:BEGIN -->', '<!-- TSB:END -->'
if os.path.exists(narr) and B in rep and E in rep:
    block = open(narr).read().strip() + '\n\n' + text.strip()
    rep = rep[:rep.index(B) + len(B)] + '\n\n' + block + '\n\n' + rep[rep.index(E):]
    open(rp, 'w').write(rep)
    print('REPORT.md section 7.9 refreshed')
