"""Merge results/extended_parts/*.csv (when present) into results/extended_results.csv, write
results/extended_tables.md (tabular, 47 ADBench datasets) and results/ts_control_tables.md
(time-series untrained control), and refresh REPORT.md section 7.6.

The comparisons below were fixed before the results were looked at.
"""
import glob, os
import numpy as np, pandas as pd
from scipy.stats import spearmanr, wilcoxon

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, 'results')
BASELINES = ['KNN', 'IForest', 'LOF', 'ECOD', 'COPOD', 'PCA', 'HBOS', 'OCSVM', 'AutoEncoder', 'DeepSVDD', 'LUNAR', 'DIF']
FAMILY = ['JEPAOD', 'JEPAOD-untrained', 'TJEPAOD', 'TJEPAOD-untrained']
SCORES = ['pred', 'embed', 'combined', 'emb+KNN']
RANKED = BASELINES + ['JEPAOD[combined]', 'JEPAOD-untrained[combined]', 'TJEPAOD[combined]', 'TJEPAOD-untrained[embed]']
ORIGINAL16 = ['6_cardio', '12_fault', '17_InternetAds', '18_Ionosphere', '22_magic.gamma', '25_musk', '29_Pima',
              '31_satimage-2', '33_skin', '36_speech', '38_thyroid', '40_vowels', '41_Waveform', '42_WBC',
              '44_Wilt', '45_wine']
# (A, B, question), fixed in advance
PAIRS = [
    ('JEPAOD[combined]', 'JEPAOD-untrained[combined]', 'does training help the default detector'),
    ('JEPAOD[embed]', 'JEPAOD-untrained[embed]', 'does training help the embed score'),
    ('JEPAOD[pred]', 'JEPAOD-untrained[pred]', 'does training help the pred score'),
    ('JEPAOD[emb+KNN]', 'JEPAOD-untrained[emb+KNN]', 'does training help KNN on the embeddings'),
    ('TJEPAOD[combined]', 'TJEPAOD-untrained[embed]', 'trained T-JEPA recipe against its untrained embed score'),
    ('TJEPAOD[embed]', 'TJEPAOD-untrained[embed]', 'does training help the embed score (T-JEPA)'),
    ('TJEPAOD[pred]', 'TJEPAOD-untrained[pred]', 'does training help the pred score (T-JEPA)'),
    ('TJEPAOD[combined]', 'JEPAOD[combined]', 'T-JEPA recipe against JEPAOD'),
    ('JEPAOD[combined]', 'AutoEncoder', 'JEPAOD against the deep baseline'),
    ('JEPAOD[combined]', 'KNN', 'JEPAOD against KNN'),
    ('JEPAOD[combined]', 'IForest', 'JEPAOD against IForest'),
    ('TJEPAOD[combined]', 'KNN', 'T-JEPA recipe against KNN'),
    ('TJEPAOD[combined]', 'IForest', 'T-JEPA recipe against IForest'),
    ('TJEPAOD-untrained[embed]', 'KNN', 'untrained embed score against KNN'),
    ('TJEPAOD-untrained[embed]', 'IForest', 'untrained embed score against IForest'),
    ('TJEPAOD-untrained[embed]', 'PCA', 'untrained embed score against a linear Mahalanobis-type score'),
    ('JEPAOD-untrained[combined]', 'AutoEncoder', 'untrained JEPAOD network against the deep baseline'),
]


def md(df):
    cols = list(df.columns)
    lines = ['| ' + ' | '.join([df.index.name or ''] + [str(c) for c in cols]) + ' |',
             '|' + '---|' * (len(cols) + 1)]
    for idx, row in df.iterrows():
        lines.append('| ' + ' | '.join([str(idx)] + [str(v) for v in row.values]) + ' |')
    return '\n'.join(lines)


def fmt_p(p):
    return 'n/a' if np.isnan(p) else ('<0.001' if p < 1e-3 else f'{p:.3f}')


def signed_rank(d):
    d = np.asarray(d, float)
    if np.allclose(d, 0):
        return np.nan
    return wilcoxon(d).pvalue


def holm(ps):
    """Holm-adjusted p-values (NaN entries are left out of the family)."""
    ps = np.asarray(ps, float)
    adj = np.full(len(ps), np.nan)
    ok = np.where(~np.isnan(ps))[0]
    order = ok[np.argsort(ps[ok])]
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(order) - rank) * ps[i])
        adj[i] = min(1.0, running)
    return adj


def boot_ci(d, n=10000, seed=0):
    rng = np.random.RandomState(seed)
    d = np.asarray(d, float)
    means = d[rng.randint(0, len(d), (n, len(d)))].mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


def paired_table(auc, apr, pairs, unit):
    rows = []
    for a, b, q in pairs:
        if a not in auc.columns or b not in auc.columns:
            continue
        ok = auc[[a, b]].dropna().index
        d, dap = auc.loc[ok, a] - auc.loc[ok, b], apr.loc[ok, a] - apr.loc[ok, b]
        lo, hi = boot_ci(d.values)
        rows.append({'A': a, 'B': b, f'{unit}': len(ok),
                     'ROC-AUC A / B': f'{auc.loc[ok, a].mean():.3f} / {auc.loc[ok, b].mean():.3f}',
                     'A − B [95 % CI]': f'{d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]',
                     'A / B better': f'{int((d > 0).sum())} / {int((d < 0).sum())}',
                     '_p': signed_rank(d),
                     'AP A / B': f'{apr.loc[ok, a].mean():.3f} / {apr.loc[ok, b].mean():.3f}',
                     '_pap': signed_rank(dap)})
    t = pd.DataFrame(rows)
    if len(t) == 0:
        return t
    t.insert(t.columns.get_loc('_p') + 1, 'p', t['_p'].map(fmt_p))
    t.insert(t.columns.get_loc('p') + 1, 'p (Holm)', [fmt_p(p) for p in holm(t['_p'].values)])
    t['p AP'] = t['_pap'].map(fmt_p)
    t['p AP (Holm)'] = [fmt_p(p) for p in holm(t['_pap'].values)]
    t = t.drop(columns=['_p', '_pap'])
    t.index = range(1, len(t) + 1); t.index.name = ''
    return t


# ---------------------------------------------------------------- tabular
out = []
# the per-task files of bench_extended.py are merged when present (they are not in the repository);
# otherwise the committed merge is read
parts = sorted(glob.glob(os.path.join(R, 'extended_parts', '*.csv')))
if parts:
    df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    df.to_csv(os.path.join(R, 'extended_results.csv'), index=False)
else:
    df = pd.read_csv(os.path.join(R, 'extended_results.csv'))
skipped = df[df.model == 'SKIPPED']
df = df[df.model != 'SKIPPED']
n_sets, n_seeds = df.dataset.nunique(), df.seed.nunique()
tasks = df.groupby(['dataset', 'seed']).ngroups
failed = df[df.auc.isna()].groupby('model').size()
line = (f'Coverage: {n_sets} datasets, {n_seeds} seeds, {tasks} dataset-seed pairs with results'
        + (f'; {len(skipped)} pairs skipped because the subsample had too few outliers for a stratified split ('
           + ', '.join(f'{r.dataset} seed {r.seed}' for r in skipped.itertuples()) + ')' if len(skipped) else '')
        + '. Fits that raised an error: '
        + (', '.join(f'{k} {v}' for k, v in failed.items()) if len(failed) else 'none') + '.')
out.append(line)

m = df.groupby(['dataset', 'model'])[['auc', 'ap', 'time']].mean().reset_index()
auc = m.pivot(index='dataset', columns='model', values='auc')
apr = m.pivot(index='dataset', columns='model', values='ap')
tim = m.pivot(index='dataset', columns='model', values='time')
order = sorted(auc.index, key=lambda s: int(s.split('_')[0]))
auc, apr, tim = auc.loc[order], apr.loc[order], tim.loc[order]

# table 1: ranking
cols = [c for c in RANKED if c in auc.columns]
ranks = auc[cols].rank(axis=1, ascending=False)
t1 = pd.DataFrame({'mean ROC-AUC': auc[cols].mean(axis=0).map('{:.3f}'.format),
                   'mean AP': apr[cols].mean(axis=0).map('{:.3f}'.format),
                   'avg rank': ranks.mean(axis=0).map('{:.2f}'.format),
                   'wins': (ranks == 1).sum(axis=0),
                   'datasets': auc[cols].notna().sum(axis=0),
                   'mean time (s)': tim[cols].mean(axis=0).map('{:.1f}'.format)})
t1 = t1.loc[auc[cols].mean(axis=0).sort_values(ascending=False).index]
t1.index.name = 'detector'
out.append(f'#### Extended benchmark: {n_sets} ADBench datasets, mean of {n_seeds} seeds, ranked by mean ROC-AUC '
           f'({len(cols)} columns; time is fit plus scoring with one thread)\n\n' + md(t1))

# table 2: family x scores
rows = []
for v in FAMILY:
    if f'{v}[pred]' not in auc.columns:
        continue
    r = {'variant': v}
    for s in SCORES:
        c = f'{v}[{s}]'
        r[s] = f'{auc[c].mean():.3f} / {apr[c].mean():.3f}'
    by_seed = df[df.model == f'{v}[combined]'].groupby('seed').auc.mean()
    r['combined ROC-AUC by seed'] = ' / '.join(f'{x:.3f}' for x in by_seed.values)
    r['fit time (s)'] = f'{tim[f"{v}[pred]"].mean():.1f}'
    rows.append(r)
out.append(f'#### JEPA family and untrained controls: mean ROC-AUC / AP per score ({n_sets} datasets)\n\n'
           + md(pd.DataFrame(rows).set_index('variant')))

# table 3: paired comparisons
out.append(f'#### Paired comparisons over the datasets (two-sided Wilcoxon signed-rank on per-dataset means; '
           f'Holm correction over the {len(PAIRS)} comparisons, which were fixed in advance; '
           f'CI: bootstrap over datasets)\n\n' + md(paired_table(auc, apr, PAIRS, 'datasets')))

# table 4: original 16 vs the 31 added
keys = [c for c in ['KNN', 'IForest', 'AutoEncoder', 'JEPAOD[combined]', 'JEPAOD-untrained[combined]',
                    'TJEPAOD[combined]', 'TJEPAOD-untrained[embed]'] if c in auc.columns]
is16 = auc.index.isin(ORIGINAL16)
t4 = pd.DataFrame({f'original 16 (n={int(is16.sum())})': auc.loc[is16, keys].mean(axis=0).map('{:.3f}'.format),
                   f'added (n={int((~is16).sum())})': auc.loc[~is16, keys].mean(axis=0).map('{:.3f}'.format),
                   f'all (n={len(auc)})': auc[keys].mean(axis=0).map('{:.3f}'.format)})
t4.index.name = 'detector'
out.append('#### Mean ROC-AUC on the 16 datasets of Section 7.1 and on the datasets added here\n\n' + md(t4))

# table 5: post hoc, not one of the comparisons fixed in advance: does the effect of training depend on the size of
# the training split (the number of gradient steps is 30 epochs x n_train / 64)?
n_tr = df.groupby('dataset').n_train.first().loc[auc.index]
small = n_tr <= n_tr.median()
rows = []
for a, b in [('JEPAOD[combined]', 'JEPAOD-untrained[combined]'), ('JEPAOD[embed]', 'JEPAOD-untrained[embed]'),
             ('JEPAOD[pred]', 'JEPAOD-untrained[pred]'), ('TJEPAOD[combined]', 'TJEPAOD-untrained[embed]'),
             ('TJEPAOD[embed]', 'TJEPAOD-untrained[embed]'), ('TJEPAOD[pred]', 'TJEPAOD-untrained[pred]')]:
    d = auc[a] - auc[b]
    rho = spearmanr(d, n_tr)
    half = lambda k: f'{d[k].mean():+.3f} ({int((d[k] > 0).sum())} / {int((d[k] < 0).sum())})'
    rows.append({'trained': a, 'control': b, 'Spearman rho with n_train': f'{rho.statistic:+.2f} (p {"< 0.001" if rho.pvalue < 1e-3 else "= " + fmt_p(rho.pvalue)})',
                 f'smaller half (n_train <= {int(n_tr.median())}, {int(small.sum())} datasets)': half(small),
                 f'larger half ({int((~small).sum())} datasets)': half(~small)})
t5 = pd.DataFrame(rows).set_index('trained')
out.append('#### Post hoc: trained minus control ROC-AUC against the size of the training split (mean difference, '
           'with datasets where the trained model / the control is better in brackets; not corrected, not fixed in advance)'
           '\n\n' + md(t5))

# table 6: per data set
per = auc[keys].copy()
per.index = per.index.str.replace(r'^\d+_', '', regex=True)
tab = per.copy().astype(object)
for i, row in per.iterrows():
    for c in per.columns:
        tab.at[i, c] = 'n/a' if np.isnan(row[c]) else (f'**{row[c]:.3f}**' if row[c] == row.max() else f'{row[c]:.3f}')
tab.loc['**mean**'] = [f'{v:.3f}' for v in per.mean(axis=0)]
tab.index.name = 'dataset'
out.append('#### Per-dataset ROC-AUC (mean of seeds; best of the row in bold)\n\n' + md(tab))

text = '\n\n'.join(out) + '\n'
open(os.path.join(R, 'extended_tables.md'), 'w').write(text)
print(text)

# ---------------------------------------------------------------- time series
ts_text = ''
ftr, fun = os.path.join(R, 'ts_control_trained.csv'), os.path.join(R, 'ts_control_untrained.csv')
if os.path.exists(ftr) and os.path.exists(fun):
    ts = pd.concat([pd.read_csv(ftr), pd.read_csv(fun)], ignore_index=True)
    ts['group'] = ts.dataset.str.split('-').str[0].map({'synthetic': 'synthetic', 'NAB': 'NAB', 'SKAB': 'SKAB'})
    # a series is one evaluation unit: synthetic series differ by seed, real ones are averaged over init seeds
    ts['series'] = np.where(ts.group == 'synthetic', ts.dataset + '-s' + ts.seed.astype(str), ts.dataset)
    ts['trained'] = ~ts.model.str.contains('untrained')
    ts['variant'] = ts.model.str.replace('TSJEPAOD-untrained-', '', regex=False).str.replace('TSJEPAOD-', '', regex=False)
    g = ts.groupby(['group', 'series', 'variant', 'trained'])[['auc', 'ap']].mean().reset_index()
    rows = []
    for grp in ['synthetic', 'SKAB', 'NAB']:
        for var in ['block[pred]', 'block[embed]', 'block[combined]', 'forecast[pred]', 'forecast[embed]',
                    'forecast[combined]']:
            x = g[(g.group == grp) & (g.variant == var)]
            a = x[x.trained].set_index('series'); b = x[~x.trained].set_index('series')
            ok = a.index.intersection(b.index)
            if len(ok) == 0:
                continue
            d = a.loc[ok, 'auc'] - b.loc[ok, 'auc']
            lo, hi = boot_ci(d.values)
            rows.append({'group': grp, 'variant': var, 'series': len(ok),
                         'ROC-AUC trained / untrained': f'{a.loc[ok, "auc"].mean():.3f} / {b.loc[ok, "auc"].mean():.3f}',
                         'difference [95 % CI]': f'{d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]',
                         'trained / untrained better': f'{int((d > 0).sum())} / {int((d < 0).sum())}',
                         '_p': signed_rank(d),
                         'AP trained / untrained': f'{a.loc[ok, "ap"].mean():.3f} / {b.loc[ok, "ap"].mean():.3f}'})
    t = pd.DataFrame(rows)
    t.insert(t.columns.get_loc('_p') + 1, 'p', t['_p'].map(fmt_p))
    t.insert(t.columns.get_loc('p') + 1, 'p (Holm)', [fmt_p(p) for p in holm(t['_p'].values)])
    t = t.drop(columns='_p')
    t.index = range(1, len(t) + 1); t.index.name = ''
    n_seed = ts.seed.nunique()
    ts_text = (f'#### Time series: trained `TSJEPAOD` (class defaults) against the zero-epoch control, {n_seed} seeds '
               f'(Wilcoxon signed-rank over series; Holm correction over the {len(t)} rows; default variant: '
               f'`block[pred]`)\n\n' + md(t) + '\n')
    open(os.path.join(R, 'ts_control_tables.md'), 'w').write(ts_text)
    print(ts_text)

# ---------------------------------------------------------------- REPORT.md section 7.6
narr = os.path.join(R, 'extended_narrative.md')
rp = os.path.join(ROOT, 'REPORT.md')
rep = open(rp).read()
B, E = '<!-- EXTENDED:BEGIN -->', '<!-- EXTENDED:END -->'
if os.path.exists(narr) and B in rep and E in rep:
    block = '\n\n'.join(x.strip() for x in [open(narr).read(), text, ts_text] if x.strip())
    rep = rep[:rep.index(B) + len(B)] + '\n\n' + block + '\n\n' + rep[rep.index(E):]
    open(rp, 'w').write(rep)
    print('REPORT.md section 7.6 refreshed')
