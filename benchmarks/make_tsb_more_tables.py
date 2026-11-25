"""Tables for the ten JEPA versions of version 0.5.0 (REPORT.md section 7.11).

The plan, fixed before any result was read, is results/tsb_more_plan.md: the ten versions, trained and untrained, on
the five TSB-AD lists; 100 tests on VUS-PR of the default score `pred` (each version against the default version
and against its untrained control), Holm over the 100; and the small benchmarks of section 7.8 with 60 tests.
Reads results/tsb/metrics/<list>/<run>.csv (benchmarks/eval_tsb.py), results/tsb/runs/ (timing logs) and
results/ts_variants/ (benchmarks/run_ts_versions.sh). Writes results/tsb_more_tables.md and
results/tsb_more_tests.csv and refreshes REPORT.md between the TSBM markers.
"""
import os, sys
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _tables import md, fmt_p, signed_rank, holm, boot_ci

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, 'results', 'tsb')
OLD = ['lejepa', 'ijepa', 'tsjepa', 'tjepa', 'vjepa', 'vjepa21', 'cjepa', 'sjepa', 'salt', 'vicreg', 'kerjepa',
       'lpjepa', 'lambdajepa', 'lenepa', 'lewm', 'cfjepa']
NEW = ['ststjepa', 'hepa', 'subjepa', 'straight', 'motionjepa', 'njepa', 'gmmjepa', 'variational', 'phijepa',
       'harjepa']
# the untrained network of these three is that of lewm
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


index = {}
for lst, n in LISTS.items():
    index[lst] = pd.Index(sorted(set(table(lst, run('lejepa')).file)))
    assert len(index[lst]) == n, (lst, len(index[lst]))


def metric(lst, tag, score='pred', col='VUS-PR'):
    """Per-file values of one run in the order of the list; None when the run is missing or incomplete."""
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
# coverage ----------------------------------------------------------------------------------------------------------
missing = [f'{lst} {v} {"trained" if t else "untrained"}' for lst in LISTS for v in NEW for t in (True, False)
           if metric(lst, run(v, t)) is None]
errors, seconds, peak = 0, 0.0, 0.0
for lst in LISTS:
    for v in NEW:
        for t in ((True, False) if v not in CONTROL else (True,)):
            d = log(lst, run(v, t))
            if d is not None:
                errors += int(d.error.notna().sum())
                seconds += d.fit_s.sum() + d.score_s.sum()
                peak = max(peak, d.gpu_peak_mb.max())
n_runs = sum(len(NEW) * 2 - len(CONTROL) for _ in LISTS)
out.append(f'Coverage: {len(NEW)} versions on {len(LISTS)} lists ({sum(LISTS.values())} series), {n_runs} runs; '
           + ('every run has every series' if not missing else 'MISSING: ' + ', '.join(missing))
           + f'; series with an error: {errors}; fitting and scoring took {seconds / 3600:.1f} hours in all; '
           f'largest GPU memory of a run {peak / 1024:.1f} GB.')

# means ---------------------------------------------------------------------------------------------------------------
for lst, n in LISTS.items():
    rows = []
    for v in ['lejepa', 'vicreg', 'lewm'] + NEW:
        d = log(lst, run(v))
        rows.append({'version': v + (' (default)' if v == 'lejepa' else ''),
                     **{f'`{s}`': f'{mean(metric(lst, run(v), s))} / {mean(metric(lst, run(v, False), s))}'
                        for s in SCORES},
                     'minutes': '' if d is None else f'{(d.fit_s.sum() + d.score_s.sum()) / 60:.0f}'})
    rows.append({'version': 'no network (raw patches)', '`embed`': mean(metric(lst, 'rawpatch_s42_semi', 'embed'))})
    tag, name = REF[lst[0]]
    rows.append({'version': name, '`pred`': mean(metric(lst, tag, 'score'))})
    t = pd.DataFrame(rows).fillna('').set_index('version')
    out.append(f'#### {lst} ({n} series): mean VUS-PR, trained / untrained, for the three score types. The first '
               'three rows are from Sections 7.9 and 7.10; `subjepa`, `straight` and `motionjepa` share the untrained '
               'control of `lewm`. Minutes: fitting and scoring of the trained run.\n\n' + md(t))

# the tests of the plan ----------------------------------------------------------------------------------------------
rows = []
for family, name in (('A', 'default version'), ('B', 'untrained control')):
    for v in NEW:
        for lst in LISTS:
            a = metric(lst, run(v))
            b = metric(lst, run('lejepa')) if family == 'A' else metric(lst, run(v, False))
            if a is None or b is None:   # a run that is not there yet
                rows.append({'family': family, 'version': v, 'list': lst, 'against': name, 'series': 0,
                             'mean': np.nan, 'mean_other': np.nan, 'difference': np.nan, 'ci_low': np.nan,
                             'ci_high': np.nan, 'better': 0, 'worse': 0, '_p': np.nan})
                continue
            d = a - b
            lo, hi = boot_ci(d.values)
            rows.append({'family': family, 'version': v, 'list': lst, 'against': name, 'series': len(d),
                         'mean': a.mean(), 'mean_other': b.mean(), 'difference': d.mean(), 'ci_low': lo, 'ci_high': hi,
                         'better': int((d > 0).sum()), 'worse': int((d < 0).sum()), '_p': signed_rank(d)})
tests = pd.DataFrame(rows)
tests['p'] = tests['_p']
tests['p_holm'] = holm(tests['_p'].values)
tests = tests.drop(columns='_p')


def reading(r, up, down):
    if np.isnan(r.p_holm):
        return 'n/a'
    return 'not separated' if r.p_holm >= 0.05 else (up if r.difference > 0 else down)


tests['reading'] = [reading(r, *(('ahead', 'behind') if r.family == 'A' else ('training helps', 'training hurts')))
                    for r in tests.itertuples()]
tests.to_csv(os.path.join(ROOT, 'results', 'tsb_more_tests.csv'), index=False, float_format='%.6g')
for family, title in (('A', 'against the default version `lejepa` (trained)'),
                      ('B', 'against its untrained control')):
    sub = tests[tests.family == family]
    wide = {}
    for lst in LISTS:
        col = sub[sub.list == lst].set_index('version')
        wide[lst] = [f'{col.difference[v]:+.3f}, {int(col.better[v])} / {int(col.worse[v])}, p {fmt_p(col.p_holm[v])}'
                     + ('' if col.reading[v] == 'not separated' else f' **{col.reading[v]}**') for v in NEW]
    t = pd.DataFrame(wide, index=pd.Index(NEW, name='version'))
    out.append(f'#### Tests {family} of the plan: each new version, trained, `pred`, {title}. Cell: mean difference '
               'in VUS-PR, series better / worse, Holm-corrected p over all 100 tests (two-sided Wilcoxon '
               f'signed-rank); a reading is given where p < 0.05. All numbers: `results/tsb_more_tests.csv`.\n\n'
               + md(t))
tally = tests.groupby(['family', 'reading']).size().unstack(fill_value=0)
tally.index = ['A: against the default', 'B: against the untrained control']
tally.index.name = 'tests'
out.append('#### Tally of the 100 tests\n\n' + md(tally))

# where the new versions stand among all twenty-six ------------------------------------------------------------------
rows = []
for lst in LISTS:
    means = {v: metric(lst, run(v)).mean() for v in OLD + NEW if metric(lst, run(v)) is not None}
    order = sorted(means, key=means.get, reverse=True)
    if not set(NEW) <= set(means):
        continue
    cells = {(v, t, s): metric(lst, run(v, t), s).mean() for v in OLD + NEW for t in (True, False) for s in SCORES
             if metric(lst, run(v, t), s) is not None}
    top = max(cells, key=cells.get)
    rows.append({'list': lst, 'versions': len(means),
                 'highest `pred`, trained': ', '.join(f'{v} {means[v]:.3f}' for v in order[:3]),
                 'places of the ten': ', '.join(f'{v} {order.index(v) + 1}' for v in NEW if v in order),
                 'highest cell': f'{cells[top]:.3f} ({top[0]}, {"trained" if top[1] else "untrained"}, {top[2]})'})
if rows:
    out.append('#### All twenty-six versions: the three highest means of trained `pred`, the place of each new '
               'version among the twenty-six, and the highest of the 156 cells (version, trained or not, score '
               'type). The cell is picked after the results.\n\n' + md(pd.DataFrame(rows).set_index('list')))

# small benchmarks ----------------------------------------------------------------------------------------------------
V = os.path.join(ROOT, 'results', 'ts_variants')
GROUPS = ['synthetic', 'SKAB', 'NAB']


def small(version, trained):
    if not trained:
        version = CONTROL.get(version, version)
    kind = 'trained' if trained else 'untrained'
    path = (os.path.join(ROOT, 'results', f'ts_control_{kind}.csv') if version == 'lejepa'
            else os.path.join(V, f'{version}_{kind}.csv'))
    if not os.path.exists(path):
        return None
    d = pd.read_csv(path)
    d['group'] = d.dataset.str.split('-').str[0]
    d['series'] = np.where(d.group == 'synthetic', d.dataset + '-s' + d.seed.astype(str), d.dataset)
    d['score'] = d.model.str.split('-').str[-1]
    return d.groupby(['group', 'series', 'score'])[['auc', 'ap', 'time']].mean().reset_index()


def small_series(version, trained, group, score='block[pred]', col='auc'):
    d = small(version, trained)
    if d is None:
        return None
    return d[(d.group == group) & (d.score == score)].set_index('series')[col]


if all(small(v, True) is not None for v in NEW):
    rows = []
    for v in ['lejepa', 'vicreg', 'lewm'] + NEW:
        row = {'version': v + (' (default)' if v == 'lejepa' else '')}
        for grp in GROUPS:
            for s in SCORES:
                a, b = small_series(v, True, grp, f'block[{s}]'), small_series(v, False, grp, f'block[{s}]')
                row[f'{grp} `{s}`'] = f'{a.mean():.3f} / ' + ('missing' if b is None else f'{b.mean():.3f}')
        rows.append(row)
    out.append('#### Small benchmarks (protocol of Section 7.8, 5 seeds): mean ROC-AUC, trained / zero-epoch '
               'control\n\n' + md(pd.DataFrame(rows).set_index('version')))
    rows = []
    for kind in ('training', 'version'):
        for v in NEW:
            for grp in GROUPS:
                a = small_series(v, True, grp)
                b = small_series(v, False, grp) if kind == 'training' else small_series('lejepa', True, grp)
                ok = a.index.intersection(b.index)
                d = a.loc[ok] - b.loc[ok]
                rows.append({'A': f'{v}, trained', 'B': 'its untrained control' if kind == 'training'
                             else 'lejepa, trained', 'group': grp, 'series': len(ok),
                             'ROC-AUC A / B': f'{a.loc[ok].mean():.3f} / {b.loc[ok].mean():.3f}',
                             'A − B': f'{d.mean():+.3f}',
                             'A / B better': f'{int((d > 0).sum())} / {int((d < 0).sum())}', '_p': signed_rank(d)})
    t = pd.DataFrame(rows)
    t['p'] = t['_p'].map(fmt_p)
    t['p (Holm)'] = [fmt_p(p) for p in holm(t['_p'].values)]
    t = t.drop(columns='_p')
    t.index = range(1, len(t) + 1); t.index.name = ''
    out.append(f'#### Small benchmarks: the {len(t)} tests of the plan on `block[pred]` (two-sided Wilcoxon '
               'signed-rank over series, Holm over all of them)\n\n' + md(t))
else:
    out.append('Small benchmarks: NOT ALL RUNS ARE THERE.')

text = '\n\n'.join(out) + '\n'
open(os.path.join(ROOT, 'results', 'tsb_more_tables.md'), 'w').write(text)
print(text)

narr = os.path.join(ROOT, 'results', 'tsb_more_narrative.md')
rp = os.path.join(ROOT, 'REPORT.md')
rep = open(rp).read()
B, E = '<!-- TSBM:BEGIN -->', '<!-- TSBM:END -->'
if os.path.exists(narr) and B in rep and E in rep:
    block = open(narr).read().strip() + '\n\n' + text.strip()
    rep = rep[:rep.index(B) + len(B)] + '\n\n' + block + '\n\n' + rep[rep.index(E):]
    open(rp, 'w').write(rep)
    print('REPORT.md section 7.11 refreshed')
