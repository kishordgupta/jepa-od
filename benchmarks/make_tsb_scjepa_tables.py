"""Tables for `scjepa`, the SC-JEPA (MTS-JEPA) version of 0.5.1 (REPORT.md section 7.13).

The tests were fixed in the addendum of results/tsb_more_plan.md before the runs: `scjepa`, trained, with the default
score `pred`, against the default version and against its untrained control, on the five TSB-AD lists (10 tests,
Holm over the 10) and in the three small benchmark groups (6 tests, Holm over the 6). Reads the same files as
benchmarks/make_tsb_more_tables.py. Writes results/tsb_scjepa_tables.md and refreshes REPORT.md between the SCJ
markers.
"""
import os, sys
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _tables import md, fmt_p, signed_rank, holm, boot_ci

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, 'results', 'tsb')
ALL = ['lejepa', 'ijepa', 'tsjepa', 'tjepa', 'vjepa', 'vjepa21', 'cjepa', 'sjepa', 'salt', 'vicreg', 'kerjepa',
       'lpjepa', 'lambdajepa', 'lenepa', 'lewm', 'cfjepa', 'ststjepa', 'hepa', 'subjepa', 'straight', 'motionjepa',
       'njepa', 'gmmjepa', 'variational', 'phijepa', 'harjepa', 'scjepa']
CONTROL = {'subjepa': 'lewm', 'straight': 'lewm', 'motionjepa': 'lewm'}
SCORES = ('pred', 'embed', 'combined')
LISTS = {'M-Eva': 180, 'M-Tuning': 20, 'U-Eva': 350, 'U-Tuning': 48, 'U-Rest': 472}
REF = {'M': ('ref_PCA_slidingWindow1', 'PCA (window 1), re-run'), 'U': ('ref_Sub_PCA', 'Sub-PCA, re-run')}
cache = {}


def run(version, trained=True):
    if not trained:
        version = CONTROL.get(version, version)
    return f'{version}_{"trained" if trained else "e0"}_s42_semi'


def table(lst, tag):
    if (lst, tag) not in cache:
        path = os.path.join(R, 'metrics', lst, tag + '.csv')
        cache[lst, tag] = pd.read_csv(path) if os.path.exists(path) else None
    return cache[lst, tag]


index = {lst: pd.Index(sorted(set(table(lst, run('lejepa')).file))) for lst in LISTS}


def metric(lst, tag, score='pred', col='VUS-PR'):
    d = table(lst, tag)
    if d is None:
        return None
    x = d[d.score == score].drop_duplicates('file').set_index('file')[col].astype(float)
    return x.reindex(index[lst]) if set(index[lst]) <= set(x.index) else None


def mean(x):
    return 'missing' if x is None else f'{x.mean():.3f}'


def log(lst, tag):
    path = os.path.join(R, 'runs', lst, tag + '.csv')
    return pd.read_csv(path).drop_duplicates('file', keep='last') if os.path.exists(path) else None


out = []
errors = seconds = peak = 0
for lst in LISTS:
    for t in (True, False):
        d = log(lst, run('scjepa', t))
        if d is not None:
            errors += int(d.error.notna().sum())
            seconds += d.fit_s.sum() + d.score_s.sum()
            peak = max(peak, d.gpu_peak_mb.max())
missing = [f'{lst} {"trained" if t else "untrained"}' for lst in LISTS for t in (True, False)
           if metric(lst, run('scjepa', t)) is None]
out.append(f'Coverage: `scjepa`, trained and untrained, on {len(LISTS)} lists ({sum(LISTS.values())} series); '
           + ('every run has every series' if not missing else 'MISSING: ' + ', '.join(missing))
           + f'; series with an error: {errors}; fitting and scoring took {seconds / 3600:.1f} hours in all; largest '
           f'GPU memory of a run {peak / 1024:.1f} GB.')

# means
rows = []
for v in ['lejepa', 'lewm', 'straight', 'hepa', 'scjepa']:
    for s in SCORES:
        row = {'version': v + (' (default)' if v == 'lejepa' else ''), 'score': f'`{s}`'}
        for lst in LISTS:
            row[lst] = f'{mean(metric(lst, run(v), s))} / {mean(metric(lst, run(v, False), s))}'
        rows.append(row)
NONORM = 'scjepa_trained_s42_semi_instance_norm0'    # the ablation: no normalisation of the two halves
if all(metric(lst, NONORM) is not None for lst in LISTS):
    for s in SCORES:
        rows.append({'version': 'scjepa without normalisation (ablation)', 'score': f'`{s}`',
                     **{lst: mean(metric(lst, NONORM, s)) for lst in LISTS}})
rows.append({'version': 'no network (raw patches)', 'score': '`embed`',
             **{lst: mean(metric(lst, 'rawpatch_s42_semi', 'embed')) for lst in LISTS}})
rows.append({'version': 'PCA / Sub-PCA, re-run', 'score': '',
             **{lst: mean(metric(lst, REF[lst[0]][0], 'score')) for lst in LISTS}})
out.append('#### Mean VUS-PR, trained / untrained (`straight` shares the untrained control of `lewm`; the ablation '
           'was run trained only); the other rows are from Sections 7.9 to 7.11\n\n'
           + md(pd.DataFrame(rows).set_index('version')))

# the ten tests
rows = []
for family, other in (('A', 'the default version `lejepa`, trained'), ('B', 'its untrained control')):
    for lst in LISTS:
        a = metric(lst, run('scjepa'))
        b = metric(lst, run('lejepa')) if family == 'A' else metric(lst, run('scjepa', False))
        if a is None or b is None:
            rows.append({'against': other, 'list': lst, '_p': np.nan, '_d': np.nan})
            continue
        d = a - b
        lo, hi = boot_ci(d.values)
        rows.append({'against': other, 'list': lst, 'series': len(d), 'VUS-PR A / B': f'{a.mean():.3f} / {b.mean():.3f}',
                     'A − B [95 % CI]': f'{d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]',
                     'A / B better': f'{int((d > 0).sum())} / {int((d < 0).sum())}', '_p': signed_rank(d),
                     '_d': d.mean()})
t = pd.DataFrame(rows)
t['p'] = t['_p'].map(fmt_p)
adj = holm(t['_p'].values)
t['p (Holm)'] = [fmt_p(p) for p in adj]
t['reading'] = ['n/a' if np.isnan(p) else 'not separated' if p >= 0.05 else
                (('ahead' if d > 0 else 'behind') if 'default' in o else
                 ('training helps' if d > 0 else 'training hurts')) for p, d, o in zip(adj, t['_d'], t['against'])]
t = t.drop(columns=['_p', '_d']).fillna('')
t.index = range(1, len(t) + 1); t.index.name = ''
out.append('#### The ten tests of the addendum: A = `scjepa`, trained, `pred` (two-sided Wilcoxon signed-rank over '
           'series, Holm over the ten; CI: bootstrap over series)\n\n' + md(t))

# place among the 27
rows = []
for lst in LISTS:
    row = {'list': lst}
    for s in SCORES:
        means = {v: metric(lst, run(v), s).mean() for v in ALL if metric(lst, run(v), s) is not None}
        order = sorted(means, key=means.get, reverse=True)
        if 'scjepa' in order:
            row[f'`{s}`'] = (f'{means["scjepa"]:.3f}, place {order.index("scjepa") + 1} of {len(order)} (first: '
                             f'{order[0]} {means[order[0]]:.3f})')
    rows.append(row)
out.append('#### Place of `scjepa` among the 27 trained versions, per score type\n\n'
           + md(pd.DataFrame(rows).set_index('list')))

# the ablation (not in the plan)
rows = []
for lst in LISTS:
    a, b, c = metric(lst, NONORM), metric(lst, run('scjepa')), metric(lst, run('lejepa'))
    if a is None or b is None:
        continue
    d, e = a - b, a - c
    rows.append({'list': lst, 'without / with normalisation': f'{a.mean():.3f} / {b.mean():.3f}',
                 'difference': f'{d.mean():+.3f}, {int((d > 0).sum())} / {int((d < 0).sum())}, p {fmt_p(signed_rank(d))}',
                 'without normalisation / default version': f'{a.mean():.3f} / {c.mean():.3f}',
                 'difference to the default': f'{e.mean():+.3f}, {int((e > 0).sum())} / {int((e < 0).sum())}, '
                                              f'p {fmt_p(signed_rank(e))}'})
if rows:
    out.append('#### Not in the plan: `scjepa` without the normalisation of the two halves '
               "(`variant_params={'instance_norm': False}`), trained, `pred`. Differences: mean, series better / "
               'worse, two-sided Wilcoxon signed-rank p, not corrected.\n\n' + md(pd.DataFrame(rows).set_index('list')))

# per source dataset (not in the plan)
SHOWN = ['lejepa', 'lewm', 'straight', 'hepa', 'scjepa']
for s, lists in (('M', ['M-Eva', 'M-Tuning']), ('U', ['U-Eva', 'U-Tuning', 'U-Rest'])):
    cols = {v: [metric(lst, run(v)) for lst in lists] for v in SHOWN}
    if any(x is None for c in cols.values() for x in c):
        continue
    d = pd.DataFrame({v: pd.concat(c) for v, c in cols.items()})
    d['dataset'] = d.index.str.split('_').str[1]
    g = d.groupby('dataset')
    t = g[SHOWN].mean()
    t.insert(0, 'series', g.size())
    t['scjepa better / worse than lejepa'] = [f'{int((x.scjepa > x.lejepa).sum())} / {int((x.scjepa < x.lejepa).sum())}'
                                              for _, x in g]
    t = t.sort_values('series', ascending=False)
    total = pd.DataFrame({**{v: [d[v].mean()] for v in SHOWN}, 'series': [len(d)],
                          'scjepa better / worse than lejepa': [
                              f'{int((d.scjepa > d.lejepa).sum())} / {int((d.scjepa < d.lejepa).sum())}']},
                         index=['all'])[t.columns]
    t = pd.concat([t, total])
    for v in SHOWN:
        t[v] = t[v].map(lambda x: f'{x:.3f}')
    t.index.name = 'dataset'
    out.append(f'#### Not in the plan: mean VUS-PR of trained `pred` per source dataset, all {len(d)} '
               f'{"multivariate" if s == "M" else "univariate"} series (the lists pooled)\n\n' + md(t))

# small benchmarks
V = os.path.join(ROOT, 'results', 'ts_variants')
GROUPS = ['synthetic', 'SKAB', 'NAB']


def small(version, trained, group, score='block[pred]'):
    kind = 'trained' if trained else 'untrained'
    path = (os.path.join(ROOT, 'results', f'ts_control_{kind}.csv') if version == 'lejepa'
            else os.path.join(V, f'{version}_{kind}.csv'))
    if not os.path.exists(path):
        return None
    d = pd.read_csv(path)
    d['group'] = d.dataset.str.split('-').str[0]
    d['series'] = np.where(d.group == 'synthetic', d.dataset + '-s' + d.seed.astype(str), d.dataset)
    d['score'] = d.model.str.split('-').str[-1]
    d = d.groupby(['group', 'series', 'score'])['auc'].mean().reset_index()
    return d[(d.group == group) & (d.score == score)].set_index('series')['auc']


if small('scjepa', True, 'NAB') is not None and small('scjepa', False, 'NAB') is not None:
    rows = []
    for v in ('lejepa', 'scjepa'):
        row = {'version': v + (' (default)' if v == 'lejepa' else '')}
        for grp in GROUPS:
            for s in SCORES:
                row[f'{grp} `{s}`'] = (f'{small(v, True, grp, f"block[{s}]").mean():.3f} / '
                                       f'{small(v, False, grp, f"block[{s}]").mean():.3f}')
        rows.append(row)
    if small('scjepa-nonorm', True, 'NAB') is not None:
        row = {'version': 'scjepa without normalisation (ablation, trained)'}
        for grp in GROUPS:
            for s in SCORES:
                row[f'{grp} `{s}`'] = f'{small("scjepa-nonorm", True, grp, f"block[{s}]").mean():.3f}'
        rows.append(row)
    out.append('#### Small benchmarks (protocol of Section 7.8, 5 seeds): mean ROC-AUC, trained / zero-epoch '
               'control\n\n' + md(pd.DataFrame(rows).set_index('version')))
    rows = []
    for other in ('its untrained control', 'lejepa, trained'):
        for grp in GROUPS:
            a = small('scjepa', True, grp)
            b = small('scjepa', False, grp) if other.startswith('its') else small('lejepa', True, grp)
            ok = a.index.intersection(b.index)
            d = a.loc[ok] - b.loc[ok]
            rows.append({'B': other, 'group': grp, 'series': len(ok),
                         'ROC-AUC A / B': f'{a.loc[ok].mean():.3f} / {b.loc[ok].mean():.3f}', 'A − B': f'{d.mean():+.3f}',
                         'A / B better': f'{int((d > 0).sum())} / {int((d < 0).sum())}', '_p': signed_rank(d)})
    t = pd.DataFrame(rows)
    t['p'] = t['_p'].map(fmt_p)
    t['p (Holm)'] = [fmt_p(p) for p in holm(t['_p'].values)]
    t = t.drop(columns='_p')
    t.index = range(1, len(t) + 1); t.index.name = ''
    out.append('#### Small benchmarks: the six tests of the addendum, A = `scjepa`, trained, `block[pred]` (Holm over '
               'the six)\n\n' + md(t))
else:
    out.append('Small benchmarks: NOT ALL RUNS ARE THERE.')

text = '\n\n'.join(out) + '\n'
open(os.path.join(ROOT, 'results', 'tsb_scjepa_tables.md'), 'w').write(text)
print(text)
narr = os.path.join(ROOT, 'results', 'tsb_scjepa_narrative.md')
rp = os.path.join(ROOT, 'REPORT.md')
rep = open(rp).read()
B, E = '<!-- SCJ:BEGIN -->', '<!-- SCJ:END -->'
if os.path.exists(narr) and B in rep and E in rep:
    block = open(narr).read().strip() + '\n\n' + text.strip()
    rep = rep[:rep.index(B) + len(B)] + '\n\n' + block + '\n\n' + rep[rep.index(E):]
    open(rp, 'w').write(rep)
    print('REPORT.md section 7.13 refreshed')
