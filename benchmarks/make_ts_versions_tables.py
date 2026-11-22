"""Tables for the twelve further JEPA versions of TSJEPAOD (REPORT.md section 7.8).

Reads the results of the first four versions (results/ts_control_{trained,untrained}.csv for 'lejepa',
results/ts_variants/{V}_{trained,untrained}.csv for the others) and those of the twelve new versions,
writes results/ts_versions_tables.md and refreshes REPORT.md between the TSVER markers.

The 72 tests below were fixed, and this script committed, before the new versions were benchmarked:
for each of the twelve new versions and each benchmark group, (a) trained against its zero-epoch
control and (b) trained against the trained default version ('lejepa'), both on the default score
`block[pred]`. 'lenepa', 'lewm' and 'cfjepa' do not mask patches; their one score is filed under `block`.
"""
import os, sys
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _tables import md, fmt_p, signed_rank, holm, boot_ci

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, 'results')
FIRST = ['lejepa', 'ijepa', 'tsjepa', 'tjepa']
NEW = ['vjepa', 'vjepa21', 'cjepa', 'sjepa', 'salt', 'vicreg', 'kerjepa', 'lpjepa', 'lambdajepa',
       'lenepa', 'lewm', 'cfjepa']
MASK_FREE = ['lenepa', 'lewm', 'cfjepa']
VERSIONS = FIRST + NEW
GROUPS = ['synthetic', 'SKAB', 'NAB']
DEFAULT = 'block[pred]'


def path(version, trained):
    kind = 'trained' if trained else 'untrained'
    return (os.path.join(R, f'ts_control_{kind}.csv') if version == 'lejepa'
            else os.path.join(R, 'ts_variants', f'{version}_{kind}.csv'))


def load(version, trained):
    d = pd.read_csv(path(version, trained))
    d['version'], d['trained'] = version, trained
    return d


missing = [os.path.relpath(path(v, t), ROOT) for v in VERSIONS for t in (True, False)
           if not os.path.exists(path(v, t))]
if missing:
    sys.exit('missing result files: ' + ', '.join(missing))
ts = pd.concat([load(v, t) for v in VERSIONS for t in (True, False)], ignore_index=True)
ts['group'] = ts.dataset.str.split('-').str[0]
# a series is one evaluation unit: synthetic series differ by seed, real ones are averaged over init seeds
ts['series'] = np.where(ts.group == 'synthetic', ts.dataset + '-s' + ts.seed.astype(str), ts.dataset)
ts['score'] = ts.model.str.split('-').str[-1]            # e.g. block[pred]
g = ts.groupby(['version', 'trained', 'group', 'series', 'score'])[['auc', 'ap', 'time']].mean().reset_index()
n_seed = ts.seed.nunique()


def series(version, trained, group, score, col='auc'):
    x = g[(g.version == version) & (g.trained == trained) & (g.group == group) & (g.score == score)]
    return x.set_index('series')[col]


def pair(version, group, score, col='auc'):
    a, b = series(version, True, group, score, col), series(version, False, group, score, col)
    return f'{a.mean():.3f} / {b.mean():.3f}' if len(a) else ''


out = []
counts = {grp: len(series('lejepa', True, grp, DEFAULT)) for grp in GROUPS}
complete = all(len(series(v, t, grp, DEFAULT)) == counts[grp]
               for v in VERSIONS for t in (True, False) for grp in GROUPS)
out.append(f'Coverage: {n_seed} seeds; series per group: '
           + ', '.join(f'{grp} {n}' for grp, n in counts.items())
           + ('; every version has every series.' if complete else '; SOME RUNS ARE INCOMPLETE.'))

# table 1: ROC-AUC, trained / untrained, all sixteen versions
rows = []
for grp in GROUPS:
    for v in VERSIONS:
        rows.append({'group': grp, 'version': v,
                     **{sc: pair(v, grp, f'block[{sc}]') for sc in ('pred', 'embed', 'combined')},
                     'forecast[pred]': '' if v in MASK_FREE else pair(v, grp, 'forecast[pred]'),
                     'AP of pred': pair(v, grp, DEFAULT, 'ap'),
                     'fit time (s)': f'{series(v, True, grp, DEFAULT, "time").mean():.1f}'})
t1 = pd.DataFrame(rows).set_index('group')
out.append(f'#### All sixteen versions of `TSJEPAOD`: mean ROC-AUC, trained / zero-epoch control, {n_seed} seeds '
           '(block masking for the versions that mask; the default score is `pred`; the first four rows of each '
           'group repeat Section 7.7)\n\n' + md(t1))

# table 2: mean rank of the trained versions on the default score, per group (descriptive)
rows = []
for grp in GROUPS:
    m = pd.DataFrame({v: series(v, True, grp, DEFAULT) for v in VERSIONS})
    rk = m.rank(axis=1, ascending=False).mean()
    best = (m.eq(m.max(axis=1), axis=0)).sum()
    for v in VERSIONS:
        rows.append({'group': grp, 'version': v, 'ROC-AUC': m[v].mean(), 'mean rank': rk[v], 'series won': int(best[v])})
t2 = pd.DataFrame(rows)
wide = t2.pivot(index='version', columns='group')
wide = wide.reindex(VERSIONS)
flat = pd.DataFrame({f'{grp}: {col}': wide[(col, grp)] for grp in GROUPS for col in ('ROC-AUC', 'mean rank', 'series won')})
for c in flat.columns:
    flat[c] = flat[c].map(lambda x: f'{x:.3f}' if 'ROC' in c else (f'{x:.1f}' if 'rank' in c else f'{int(x)}'))
flat.index.name = 'version'
out.append('#### Trained versions on the default score `pred`: mean ROC-AUC, mean rank among the sixteen (1 = best) '
           'and number of series on which the version is best\n\n' + md(flat))

# table 3: the tests fixed in advance
rows = []
for kind in ('training', 'version'):
    for v in NEW:
        for grp in GROUPS:
            a = series(v, True, grp, DEFAULT)
            b = series(v, False, grp, DEFAULT) if kind == 'training' else series('lejepa', True, grp, DEFAULT)
            ok = a.index.intersection(b.index)
            d = a.loc[ok] - b.loc[ok]
            lo, hi = boot_ci(d.values)
            rows.append({'A': f'{v}, trained',
                         'B': f'{v}, untrained' if kind == 'training' else 'lejepa, trained',
                         'group': grp, 'series': len(ok),
                         'ROC-AUC A / B': f'{a.loc[ok].mean():.3f} / {b.loc[ok].mean():.3f}',
                         'A − B [95 % CI]': f'{d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]',
                         'A / B better': f'{int((d > 0).sum())} / {int((d < 0).sum())}',
                         '_p': signed_rank(d)})
t3 = pd.DataFrame(rows)
t3['p'] = t3['_p'].map(fmt_p)
t3['p (Holm)'] = [fmt_p(p) for p in holm(t3['_p'].values)]
t3 = t3.drop(columns='_p')
t3.index = range(1, len(t3) + 1); t3.index.name = ''
out.append(f'#### Paired tests on the default score `pred` (two-sided Wilcoxon signed-rank over series; Holm '
           f'correction over the {len(t3)} tests, which were fixed in advance; CI: bootstrap over series)\n\n' + md(t3))

text = '\n\n'.join(out) + '\n'
open(os.path.join(R, 'ts_versions_tables.md'), 'w').write(text)
print(text)

narr = os.path.join(R, 'ts_versions_narrative.md')
rp = os.path.join(ROOT, 'REPORT.md')
rep = open(rp).read()
B, E = '<!-- TSVER:BEGIN -->', '<!-- TSVER:END -->'
if os.path.exists(narr) and B in rep and E in rep:
    block = open(narr).read().strip() + '\n\n' + text.strip()
    rep = rep[:rep.index(B) + len(B)] + '\n\n' + block + '\n\n' + rep[rep.index(E):]
    open(rp, 'w').write(rep)
    print('REPORT.md section 7.8 refreshed')
