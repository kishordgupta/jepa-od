"""Tables for the JEPA versions of TSJEPAOD (REPORT.md section 7.7).

Reads results/ts_control_{trained,untrained}.csv (variant 'lejepa') and
results/ts_variants/{V}_{trained,untrained}.csv, writes results/ts_variant_tables.md and
refreshes REPORT.md between the TSVAR markers.

The 18 tests below were fixed before the results were looked at: for each of the three new
versions and each benchmark group, (a) trained against its zero-epoch control and (b) trained
against the trained default version, both on the default score `block[pred]`.
"""
import os, sys
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _tables import md, fmt_p, signed_rank, holm, boot_ci

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, 'results')
VERSIONS = ['lejepa', 'ijepa', 'tsjepa', 'tjepa']
NEW = VERSIONS[1:]
PAPER = 'tsjepa-paper'   # variant='tsjepa' with the masking of its paper (random, 70 %)
GROUPS = ['synthetic', 'SKAB', 'NAB']
DEFAULT = 'block[pred]'


def load(version, trained):
    kind = 'trained' if trained else 'untrained'
    f = (os.path.join(R, f'ts_control_{kind}.csv') if version == 'lejepa'
         else os.path.join(R, 'ts_variants', f'{version}_{kind}.csv'))
    d = pd.read_csv(f)
    d['version'], d['trained'] = version, trained
    return d


ts = pd.concat([load(v, t) for v in VERSIONS + [PAPER] for t in (True, False)], ignore_index=True)
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
    return f'{a.mean():.3f} / {b.mean():.3f}'


out = []
counts = {grp: len(series('lejepa', True, grp, DEFAULT)) for grp in GROUPS}
complete = all(len(series(v, t, grp, DEFAULT)) == counts[grp]
               for v in VERSIONS for t in (True, False) for grp in GROUPS)
out.append(f'Coverage: {n_seed} seeds; series per group: '
           + ', '.join(f'{grp} {n}' for grp, n in counts.items())
           + ('; every version has every series.' if complete else '; SOME RUNS ARE INCOMPLETE.'))

# table 1: ROC-AUC, trained / untrained, default masking
rows = []
for grp in GROUPS:
    for v in VERSIONS:
        rows.append({'group': grp, 'version': v,
                     **{sc: pair(v, grp, f'block[{sc}]') for sc in ('pred', 'embed', 'combined')},
                     'forecast[pred]': pair(v, grp, 'forecast[pred]'),
                     'AP of block[pred]': pair(v, grp, DEFAULT, 'ap'),
                     'fit time (s)': f'{series(v, True, grp, DEFAULT, "time").mean():.1f}'})
    rows.append({'group': grp, 'version': 'tsjepa, masking of its paper (random, 70 %)',
                 **{sc: pair(PAPER, grp, f'random[{sc}]') for sc in ('pred', 'embed', 'combined')},
                 'forecast[pred]': '', 'AP of block[pred]': pair(PAPER, grp, 'random[pred]', 'ap'),
                 'fit time (s)': f'{series(PAPER, True, grp, "random[pred]", "time").mean():.1f}'})
t1 = pd.DataFrame(rows).set_index('group')
out.append(f'#### JEPA versions of `TSJEPAOD`: mean ROC-AUC, trained / zero-epoch control, {n_seed} seeds '
           '(block masking unless stated; the default score is `pred`; in the last row of each group the AP '
           'column is that of `random[pred]`)\n\n' + md(t1))

# table 2: the tests fixed in advance
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
t2 = pd.DataFrame(rows)
t2['p'] = t2['_p'].map(fmt_p)
t2['p (Holm)'] = [fmt_p(p) for p in holm(t2['_p'].values)]
t2 = t2.drop(columns='_p')
t2.index = range(1, len(t2) + 1); t2.index.name = ''
out.append(f'#### Paired tests on the default score `block[pred]` (two-sided Wilcoxon signed-rank over series; Holm '
           f'correction over the {len(t2)} tests, which were fixed in advance; CI: bootstrap over series)\n\n' + md(t2))

text = '\n\n'.join(out) + '\n'
open(os.path.join(R, 'ts_variant_tables.md'), 'w').write(text)
print(text)

narr = os.path.join(R, 'ts_variants_narrative.md')
rp = os.path.join(ROOT, 'REPORT.md')
rep = open(rp).read()
B, E = '<!-- TSVAR:BEGIN -->', '<!-- TSVAR:END -->'
if os.path.exists(narr) and B in rep and E in rep:
    block = open(narr).read().strip() + '\n\n' + text.strip()
    rep = rep[:rep.index(B) + len(B)] + '\n\n' + block + '\n\n' + rep[rep.index(E):]
    open(rp, 'w').write(rep)
    print('REPORT.md section 7.7 refreshed')
