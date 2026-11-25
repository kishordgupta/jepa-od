"""Tables for the confirmation of the TSB-AD results on series that no run had used (REPORT.md section 7.10).

The plan, fixed before the runs, is results/tsb_confirm_plan.md: three lists (M-Tuning, U-Tuning, U-Rest), the
sixteen versions trained and untrained plus the control without a network, 18 tests, and two detectors of TSB-AD's
own code as a reference. Reads results/tsb/metrics/<list>/<run>.csv (benchmarks/eval_tsb.py), results/tsb/runs/
(the timing logs of benchmarks/bench_tsb.py and bench_tsb_ref.py) and, for the evaluation lists, the files of
section 7.9 and TSB-AD's published per-file VUS-PR (data/tsb_published/). Writes results/tsb_confirm_tables.md and
refreshes REPORT.md between the TSBC markers.
"""
import os, sys
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _tables import md, fmt_p, signed_rank, holm, boot_ci

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, 'results', 'tsb')
PUB = os.path.join(ROOT, 'data', 'tsb_published')
VERSIONS = ['lejepa', 'ijepa', 'tsjepa', 'tjepa', 'vjepa', 'vjepa21', 'cjepa', 'sjepa', 'salt', 'vicreg', 'kerjepa',
            'lpjepa', 'lambdajepa', 'lenepa', 'lewm', 'cfjepa']
SCORES = ('pred', 'embed', 'combined')
LISTS = {'M-Tuning': ('M', 20), 'U-Tuning': ('U', 48), 'U-Rest': ('U', 472)}    # list: set, series
EVA = {'M': 'M-Eva', 'U': 'U-Eva'}
# the reference detector of each set: run, column of the published table, name
REF = {'M': ('ref_PCA_slidingWindow1', 'PCA', 'PCA (window 1)'), 'U': ('ref_Sub_PCA', 'Sub-PCA', 'Sub-PCA')}
# the comparisons of the plan, A against B: (version, kind, score type)
DEFAULT = ('lejepa', 'trained', 'pred')
PLANNED = [(('cfjepa', 'trained', 'embed'), DEFAULT), (('cfjepa', 'trained', 'embed'), ('rawpatch', None, 'embed')),
           (('vicreg', 'trained', 'pred'), DEFAULT), (('lambdajepa', 'trained', 'pred'), DEFAULT),
           (('lewm', 'trained', 'pred'), DEFAULT), (DEFAULT, ('lejepa', 'e0', 'pred'))]
UNTESTED = [(('kerjepa', 'trained', 'pred'), DEFAULT), (('cfjepa', 'trained', 'embed'), ('cfjepa', 'e0', 'embed')),
            (('cfjepa', 'trained', 'embed'), ('lejepa', 'trained', 'embed')), (('tsjepa', 'trained', 'embed'), DEFAULT)]
AGAINST_REF = [DEFAULT, ('lejepa', 'trained', 'embed'), ('vicreg', 'trained', 'pred'), ('cfjepa', 'trained', 'embed')]


def run(version, kind='trained'):
    return 'rawpatch_s42_semi' if version == 'rawpatch' else f'{version}_{kind}_s42_semi'


def label(spec):
    version, kind, score = spec
    if version == 'rawpatch':
        return 'raw patches, `embed`'
    return f'`{version}`, {"untrained" if kind == "e0" else "trained"}, `{score}`'


published = {s: pd.read_csv(os.path.join(PUB, f'{name}_mergedTable_VUS-PR.csv')).set_index('file')
             for s, name in (('M', 'multi'), ('U', 'uni'))}
index = {EVA[s]: published[s].index for s in EVA}
for lst, (s, n) in LISTS.items():
    files = pd.read_csv(os.path.join(R, 'metrics', lst, run('lejepa') + '.csv')).file
    index[lst] = pd.Index(sorted(set(files)))
    assert len(index[lst]) == n, (lst, len(index[lst]))
    assert not set(index[lst]) & set(index[EVA[s]]), lst
assert not set(index['U-Tuning']) & set(index['U-Rest'])
cache = {}


def metric(lst, tag, score='pred', col='VUS-PR'):
    """Per-file values of one run, in the order of the list; None when the run is not complete."""
    if (lst, tag) not in cache:
        path = os.path.join(R, 'metrics', lst, tag + '.csv')
        cache[lst, tag] = pd.read_csv(path) if os.path.exists(path) else None
    d = cache[lst, tag]
    if d is None:
        return None
    x = d[d.score == score].drop_duplicates('file').set_index('file')[col].astype(float)
    return x.reindex(index[lst]) if set(index[lst]) <= set(x.index) else None


def of(lst, spec):
    return metric(lst, run(spec[0], spec[1]), spec[2])


def ref(lst):
    return metric(lst, REF[lst[0]][0], 'score')


def mean(x):
    return '' if x is None else f'{x.mean():.3f}'


def pair(lst, version, score):
    return f'{mean(metric(lst, run(version), score))} / {mean(metric(lst, run(version, "e0"), score))}'


def log(lst, tag):
    path = os.path.join(R, 'runs', lst, tag + '.csv')
    return pd.read_csv(path).drop_duplicates('file', keep='last') if os.path.exists(path) else None


def minutes(lst, tag):
    d = log(lst, tag)
    return '' if d is None else f'{(d.fit_s.sum() + d.score_s.sum()) / 60:.0f}'


def compare(lst, a, b):
    """Row of a paired comparison of two per-file series."""
    d = a - b
    lo, hi = boot_ci(d.values)
    return {'list': lst, 'series': len(d), 'VUS-PR A / B': f'{a.mean():.3f} / {b.mean():.3f}',
            'A − B [95 % CI]': f'{d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]',
            'A / B better': f'{int((d > 0).sum())} / {int((d < 0).sum())}', '_p': signed_rank(d), '_d': d.mean()}


def eva_difference(spec_a, spec_b, s):
    return (of(EVA[s], spec_a) - of(EVA[s], spec_b)).mean()


out = []
# coverage
tags = [run(v, k) for v in VERSIONS for k in ('trained', 'e0')] + [run('rawpatch')]
missing = [f'{lst}: {tag}' for lst in LISTS for tag in tags
           if metric(lst, tag, 'embed') is None or metric(lst, tag, 'embed').isna().any()]
missing += [f'{lst}: {REF[lst[0]][0]}' for lst in list(LISTS) + list(EVA.values())
            if ref(lst) is None or ref(lst).isna().any()]
errors = []
for lst in LISTS:
    for tag in tags:
        d = log(lst, tag)
        if d is None or len(d) != LISTS[lst][1] or d.error.notna().any():
            errors.append(f'{lst}: {tag}')
out.append('Coverage: ' + ', '.join(f'{lst} {n} series' for lst, (_, n) in LISTS.items())
           + f'; {len(tags)} runs per list'
           + ('; every run has every series' if not missing else f'; {len(missing)} RUNS ARE MISSING: '
              + ', '.join(missing))
           + (', and no run log has an error.' if not errors else f'; RUN LOGS WITH ERRORS OR GAPS: '
              + ', '.join(errors)))

# table 1: the sixteen versions on each list
for lst, (s, n) in LISTS.items():
    rows = [{'version': v, **{sc: pair(lst, v, sc) for sc in SCORES}, 'minutes': minutes(lst, run(v))}
            for v in VERSIONS]
    out.append(f'#### {lst} ({n} series): mean VUS-PR of the three score types, trained / zero-epoch control '
               '(minutes: fitting and scoring the whole list once, trained, on a shared H100). Without a network '
               f'(raw patches, `embed`): {mean(metric(lst, run("rawpatch"), "embed"))}. Reference '
               f'({REF[s][2]}, re-run): {mean(ref(lst))}.\n\n' + md(pd.DataFrame(rows).set_index('version')))

# table 2: the tests of the plan
rows = []
for i, (spec_a, spec_b) in enumerate(PLANNED, 1):
    for lst, (s, _) in LISTS.items():
        a, b = of(lst, spec_a), of(lst, spec_b)
        if a is None or b is None:
            continue
        expected = eva_difference(spec_a, spec_b, s)
        assert expected > 0
        rows.append({'': i, 'A': label(spec_a), 'B': label(spec_b), **compare(lst, a, b),
                     'A − B on the evaluation list': f'{expected:+.3f}'})
if rows:
    t = pd.DataFrame(rows)
    t['p'] = t['_p'].map(fmt_p)
    adjusted = holm(t['_p'].values)
    t['p (Holm)'] = [fmt_p(p) for p in adjusted]
    t['reading'] = ['holds' if d > 0 and p < 0.05 else 'same direction' if d > 0 else 'not confirmed'
                    for d, p in zip(t['_d'], adjusted)]
    cols = [c for c in t.columns if c not in ('_p', '_d', 'A − B on the evaluation list')]
    t = t[cols[:-3] + ['A − B on the evaluation list'] + cols[-3:]]
    out.append(f'#### The {len(t)} tests fixed in the plan (VUS-PR; two-sided Wilcoxon signed-rank over series; Holm '
               f'correction over the {len(t)}; CI: bootstrap over series). Reading: "holds" when A − B is positive, '
               'as on the evaluation list, and the corrected p is below 0.05; "same direction" when only the sign '
               'agrees; "not confirmed" when A − B is negative.\n\n' + md(t.set_index('')))

# table 3: reported without a test in the plan
rows = []
for spec_a, spec_b in UNTESTED:
    for lst, (s, _) in LISTS.items():
        a, b = of(lst, spec_a), of(lst, spec_b)
        if a is None or b is None:
            continue
        row = {'A': label(spec_a), 'B': label(spec_b), **compare(lst, a, b),
               'A − B on the evaluation list': f'{eva_difference(spec_a, spec_b, s):+.3f}'}
        row['p (not corrected)'] = fmt_p(row.pop('_p')); row.pop('_d')
        rows.append(row)
if rows:
    out.append('#### Further comparisons named in the plan, outside the 18 tests\n\n'
               + md(pd.DataFrame(rows).set_index('A')))

# table 4: do the version means keep their order?
order = ['M-Eva', 'M-Tuning', 'U-Eva', 'U-Tuning', 'U-Rest']
for score in ('pred', 'embed'):
    t = pd.DataFrame({lst: {v: (lambda x: float('nan') if x is None else x.mean())(metric(lst, run(v), score))
                            for v in VERSIONS} for lst in order})
    if t.isna().any().any():
        continue
    rho = {lst: '' if lst.endswith('Eva') else f'{spearmanr(t[lst], t[EVA[lst[0]]])[0]:.2f}' for lst in order}
    place = {lst: ', '.join(t[lst].sort_values(ascending=False).index[:3]) for lst in order}
    t = t.map(lambda x: f'{x:.3f}')
    t.loc['rank correlation with the evaluation list'] = pd.Series(rho)
    t.loc['three highest'] = pd.Series(place)
    t.index.name = 'version'
    out.append(f'#### Mean VUS-PR of the sixteen trained versions, score `{score}`, on the evaluation lists (section '
               '7.9) and on the three lists of the confirmation. Rank correlation: Spearman, over the sixteen '
               'means.\n\n' + md(t))

# table 5: the reference detectors
rows = []
for s, (tag, column, name) in REF.items():
    x = ref(EVA[s])
    if x is None:
        continue
    d = x - published[s][column]
    rows.append({'detector': name, 'evaluation list': EVA[s], 'series': len(d),
                 'published mean': f'{published[s][column].mean():.3f}', 're-run mean': f'{x.mean():.3f}',
                 'series within 0.01': f'{int((d.abs() <= 0.01).sum())}',
                 'series more than 0.1 apart': f'{int((d.abs() > 0.1).sum())}',
                 'largest difference': f'{d.abs().max():.3f}',
                 'rank correlation': f'{spearmanr(x, published[s][column])[0]:.3f}'})
if rows:
    out.append("#### The reference detectors: TSB-AD's code of today against TSB-AD's published per-series VUS-PR, "
               'on the evaluation lists\n\n' + md(pd.DataFrame(rows).set_index('detector')))
rows = []
for lst in order:
    b = ref(lst)
    for spec in AGAINST_REF:
        a = of(lst, spec)
        if a is None or b is None:
            continue
        row = {'A': label(spec), 'B: reference, re-run': REF[lst[0]][2], **compare(lst, a, b)}
        row['p (not corrected)'] = fmt_p(row.pop('_p')); row.pop('_d')
        rows.append(row)
if rows:
    out.append('#### Against the re-run reference detector of the set, on the evaluation lists and on the three '
               'lists of the confirmation (the same series, the same VUS-PR code on both sides)\n\n'
               + md(pd.DataFrame(rows).set_index('A')))

# table 6: by source dataset
for lst, (s, n) in LISTS.items():
    cols = {'default: `lejepa` `pred`': of(lst, DEFAULT), '`lejepa` `embed`': of(lst, ('lejepa', 'trained', 'embed')),
            '`vicreg` `pred`': of(lst, ('vicreg', 'trained', 'pred')),
            '`cfjepa` `embed`': of(lst, ('cfjepa', 'trained', 'embed')),
            'raw patches': of(lst, ('rawpatch', None, 'embed')), f'{REF[s][2]}, re-run': ref(lst)}
    cols = {k: v for k, v in cols.items() if v is not None}
    if not cols:
        continue
    d = pd.DataFrame(cols)
    source = pd.Series(d.index.str.split('_').str[1], index=d.index)
    t = d.groupby(source).mean()
    t.insert(0, 'series', source.value_counts())
    t = t.sort_values('series', ascending=False)
    macro = list(t.drop(columns='series').mean())
    t.loc['all series'] = [len(d)] + list(d.mean())
    t.loc['mean of the dataset means'] = [len(d)] + macro
    t['series'] = t['series'].astype(int)
    for c in t.columns[1:]:
        t[c] = t[c].map(lambda x: f'{x:.3f}')
    t.index.name = 'dataset'
    out.append(f'#### {lst} by source dataset: mean VUS-PR (all trained)\n\n' + md(t))

# tables 7 and 8 are not in the plan: they were added after the results, to read them
shared = sorted(set(index['U-Rest'].str.split('_').str[1]))
rows = []
for dataset in shared:
    for lst in ('U-Eva', 'U-Rest'):
        keep = index[lst][index[lst].str.split('_').str[1] == dataset]
        row = {'dataset': dataset, 'list': lst, 'series': len(keep)}
        for name, x in (('default: `lejepa` `pred`', of(lst, DEFAULT)),
                        ('`lejepa` `embed`', of(lst, ('lejepa', 'trained', 'embed'))),
                        ('`vicreg` `pred`', of(lst, ('vicreg', 'trained', 'pred'))),
                        ('`cfjepa` `embed`', of(lst, ('cfjepa', 'trained', 'embed'))),
                        ('raw patches', of(lst, ('rawpatch', None, 'embed'))), ('Sub-PCA, re-run', ref(lst))):
            row[name] = '' if x is None else f'{x[keep].mean():.3f}'
        if lst == 'U-Eva':
            best = published['U'].loc[keep].drop(columns=[c for c in published['U'].columns
                                                          if c in ('ts_len', 'anomaly_len', 'num_anomaly',
                                                                   'avg_anomaly_len', 'anomaly_ratio',
                                                                   'point_anomaly', 'seq_anomaly')]).mean()
            row['best published on the dataset'] = f'{best.max():.3f} ({best.idxmax()})'
        else:
            row['best published on the dataset'] = ''
        rows.append(row)
out.append('#### Not in the plan: the four source datasets of U-Rest, on the evaluation list and on U-Rest (mean '
           'VUS-PR, all trained). The last column is the best of the 32 published detectors on the series of that '
           'dataset in the evaluation list; nothing is published for U-Rest.\n\n'
           + md(pd.DataFrame(rows).set_index('dataset')))

rows = []
for lst in order:
    a = {score: of(lst, ('lejepa', 'trained', score)) for score in SCORES}
    row = {'list': lst, 'series': len(index[lst]), **{f'`{score}`': mean(a[score]) for score in SCORES},
           'raw patches': mean(of(lst, ('rawpatch', None, 'embed')))}
    for score in ('embed', 'combined'):
        d = a[score] - a['pred']
        row[f'`{score}` − `pred`'] = (f'{d.mean():+.3f}, {int((d > 0).sum())} / {int((d < 0).sum())} series, '
                                      f'p {fmt_p(signed_rank(d))}')
    cells = {f'{v}, {"untrained" if k == "e0" else "trained"}, {score}': metric(lst, run(v, k), score).mean()
             for v in VERSIONS for k in ('trained', 'e0') for score in SCORES}
    top = max(cells, key=cells.get)
    row['highest of the 96 cells'] = f'{cells[top]:.3f} ({top})'
    rows.append(row)
out.append('#### Not in the plan: the three score types of the default version `lejepa` (trained), the control '
           'without a network, and the highest of the 96 cells (version, trained or not, score type) of each list. '
           'Differences: mean, series better / worse, two-sided Wilcoxon signed-rank p, not corrected.\n\n'
           + md(pd.DataFrame(rows).set_index('list')))

text = '\n\n'.join(out) + '\n'
open(os.path.join(ROOT, 'results', 'tsb_confirm_tables.md'), 'w').write(text)
print(text)

narr = os.path.join(ROOT, 'results', 'tsb_confirm_narrative.md')
rp = os.path.join(ROOT, 'REPORT.md')
rep = open(rp).read()
B, E = '<!-- TSBC:BEGIN -->', '<!-- TSBC:END -->'
if os.path.exists(narr) and B in rep and E in rep:
    block = open(narr).read().strip() + '\n\n' + text.strip()
    rep = rep[:rep.index(B) + len(B)] + '\n\n' + block + '\n\n' + rep[rep.index(E):]
    open(rp, 'w').write(rep)
    print('REPORT.md section 7.10 refreshed')
