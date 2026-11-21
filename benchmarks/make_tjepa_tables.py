"""Write results/tjepa_tables.md from results/tjepa_results.csv and refresh REPORT.md section 7.5.

With --parts, results/tjepa_results.csv is first rebuilt from results/tjepa_parts/*.csv.
"""
import argparse, glob, os
import numpy as np, pandas as pd
from scipy.stats import wilcoxon

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, 'results')
SCORES = ['pred', 'embed', 'combined', 'emb+KNN']
VARIANTS = ['JEPAOD', 'TJEPAOD', 'TJEPAOD-nonorm', 'TJEPAOD-noreg', 'TJEPAOD-repoopt',
            'JEPAOD-untrained', 'TJEPAOD-untrained']
BASELINES = ['AutoEncoder', 'DeepSVDD', 'LUNAR', 'DIF', 'IForest', 'KNN', 'LOF', 'ECOD', 'COPOD',
             'PCA', 'HBOS', 'OCSVM']


def md(df):
    cols = list(df.columns)
    lines = ['| ' + ' | '.join([df.index.name or ''] + [str(c) for c in cols]) + ' |',
             '|' + '---|' * (len(cols) + 1)]
    for idx, row in df.iterrows():
        lines.append('| ' + ' | '.join([str(idx)] + [str(v) for v in row.values]) + ' |')
    return '\n'.join(lines)


ap = argparse.ArgumentParser()
ap.add_argument('--parts', action='store_true',
                help='first merge results/tjepa_parts/*.csv into results/tjepa_results.csv')
args = ap.parse_args()
if args.parts:
    parts = sorted(glob.glob(os.path.join(R, 'tjepa_parts', '*.csv')))
    pd.concat([pd.read_csv(p) for p in parts], ignore_index=True).to_csv(
        os.path.join(R, 'tjepa_results.csv'), index=False)
df = pd.read_csv(os.path.join(R, 'tjepa_results.csv'))
old = pd.read_csv(os.path.join(R, 'tabular_results.csv'))
n_seeds, n_sets = df.seed.nunique(), df.dataset.nunique()
out = []

# reference check: how closely does the JEPAOD re-run reproduce results/tabular_results.csv
chk = df[df.model.isin([f'JEPAOD[{s}]' for s in SCORES[:3]])].merge(
    old, on=['dataset', 'model', 'seed'], suffixes=('', '_old'))
chk['d'] = np.abs(chk.auc - chk.auc_old)
moved = chk.groupby('dataset').d.max()
moved = moved[moved >= 1e-3].sort_values(ascending=False)
out.append(f'Reference check: {len(chk)} JEPAOD rows of this run against results/tabular_results.csv: '
           f'{int((chk.d < 1e-9).sum())} identical, {int((chk.d < 1e-3).sum())} within 0.001, '
           f'largest absolute ROC-AUC difference {chk.d.max():.3f}. Datasets with a difference of 0.001 or more: '
           + ', '.join(f'{k} ({v:.3f})' for k, v in moved.items()) + '.')

# per data set means over seeds
m = df.groupby(['dataset', 'model'])[['auc', 'ap', 'time']].mean().reset_index()
mo = old.groupby(['dataset', 'model'])[['auc', 'ap', 'time']].mean().reset_index()
auc = m.pivot(index='dataset', columns='model', values='auc')
apr = m.pivot(index='dataset', columns='model', values='ap')
tim = m.pivot(index='dataset', columns='model', values='time')

# --- table 1: variants x scores, mean over data sets
rows = []
for v in VARIANTS:
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
t1 = pd.DataFrame(rows).set_index('variant')
out.append(f'#### T-JEPA family, {n_sets} ADBench datasets, mean of {n_seeds} seeds: '
           f'mean ROC-AUC / AP per score\n\n' + md(t1))

# --- table 2: rank among the detectors of section 7.1
main = ['JEPAOD[combined]', 'TJEPAOD[combined]', 'TJEPAOD[embed]', 'TJEPAOD-nonorm[combined]',
        'TJEPAOD-nonorm[embed]', 'TJEPAOD-noreg[combined]', 'TJEPAOD-untrained[embed]',
        'JEPAOD-untrained[combined]']
main = [c for c in main if c in auc.columns]
base_auc = mo[mo.model.isin(BASELINES)].pivot(index='dataset', columns='model', values='auc')
base_ap = mo[mo.model.isin(BASELINES)].pivot(index='dataset', columns='model', values='ap')
all_auc = pd.concat([auc[main], base_auc], axis=1)
all_ap = pd.concat([apr[main], base_ap], axis=1)
ranks = all_auc.rank(axis=1, ascending=False)
t2 = pd.DataFrame({'mean ROC-AUC': all_auc.mean(axis=0).map('{:.3f}'.format),
                   'mean AP': all_ap.mean(axis=0).map('{:.3f}'.format),
                   'avg rank': ranks.mean(axis=0).map('{:.2f}'.format),
                   'wins': (ranks == 1).sum(axis=0)})
t2 = t2.loc[all_auc.mean(axis=0).sort_values(ascending=False).index]
t2.index.name = 'detector'
out.append(f'#### All detectors ranked by mean ROC-AUC ({len(t2)} columns; PCA has {int(base_auc["PCA"].notna().sum())} '
           f'datasets)\n\n' + md(t2))

# --- table 3: per data set ROC-AUC
cols = [c for c in ['JEPAOD[combined]', 'TJEPAOD[combined]', 'TJEPAOD[embed]', 'TJEPAOD-nonorm[combined]',
                    'TJEPAOD-noreg[combined]', 'TJEPAOD-untrained[embed]', 'JEPAOD-untrained[combined]']
        if c in auc.columns]
per = pd.concat([auc[cols], base_auc[['AutoEncoder', 'IForest', 'KNN']]], axis=1)
per.index = per.index.str.replace(r'^\d+_', '', regex=True)
tab = per.copy().astype(object)
for i, row in per.iterrows():
    for c in per.columns:
        tab.at[i, c] = f'**{row[c]:.3f}**' if row[c] == row.max() else f'{row[c]:.3f}'
tab.loc['**mean**'] = [f'{v:.3f}' for v in per.mean(axis=0)]
tab.index.name = 'dataset'
out.append('#### Per-dataset ROC-AUC (mean of seeds; best of the row in bold)\n\n' + md(tab))

# --- paired comparisons over data sets
def fmt_p(p):
    return 'n/a' if np.isnan(p) else ('<0.001' if p < 1e-3 else f'{p:.3f}')


def signed_rank(d):
    try:
        return wilcoxon(d).pvalue
    except ValueError:
        return np.nan


full_auc = pd.concat([auc, base_auc], axis=1)
full_ap = pd.concat([apr, base_ap], axis=1)
pairs = [('TJEPAOD[combined]', 'JEPAOD[combined]'), ('TJEPAOD[embed]', 'JEPAOD[embed]'),
         ('TJEPAOD[pred]', 'JEPAOD[pred]'),
         ('TJEPAOD-nonorm[combined]', 'TJEPAOD[combined]'), ('TJEPAOD-noreg[combined]', 'TJEPAOD[combined]'),
         ('TJEPAOD-repoopt[combined]', 'TJEPAOD[combined]'),
         ('TJEPAOD[embed]', 'TJEPAOD-untrained[embed]'), ('JEPAOD[embed]', 'JEPAOD-untrained[embed]'),
         ('TJEPAOD[pred]', 'TJEPAOD-untrained[pred]'), ('JEPAOD[pred]', 'JEPAOD-untrained[pred]'),
         ('JEPAOD[combined]', 'JEPAOD-untrained[combined]'),
         ('TJEPAOD[combined]', 'TJEPAOD-untrained[embed]'),
         ('JEPAOD[combined]', 'TJEPAOD-untrained[embed]'),
         ('TJEPAOD[emb+KNN]', 'TJEPAOD[embed]'), ('JEPAOD-untrained[emb+KNN]', 'JEPAOD[emb+KNN]'),
         ('TJEPAOD[combined]', 'KNN'), ('TJEPAOD[combined]', 'IForest'), ('TJEPAOD-untrained[embed]', 'PCA')]
rows = []
for a, b in pairs:
    if a not in full_auc.columns or b not in full_auc.columns:
        continue
    ok = full_auc[[a, b]].dropna().index
    d = full_auc.loc[ok, a] - full_auc.loc[ok, b]
    dap = full_ap.loc[ok, a] - full_ap.loc[ok, b]
    rows.append({'A': a, 'B': b,
                 'ROC-AUC A / B': f'{full_auc.loc[ok, a].mean():.3f} / {full_auc.loc[ok, b].mean():.3f}',
                 'datasets A / B better': f'{int((d > 0).sum())} / {int((d < 0).sum())}',
                 'p (ROC-AUC)': fmt_p(signed_rank(d)),
                 'AP A / B': f'{full_ap.loc[ok, a].mean():.3f} / {full_ap.loc[ok, b].mean():.3f}',
                 'p (AP)': fmt_p(signed_rank(dap))})
t4 = pd.DataFrame(rows)
t4.index = range(1, len(t4) + 1); t4.index.name = ''
out.append(f'#### Paired comparisons over the {n_sets} datasets (two-sided Wilcoxon signed-rank on the per-dataset '
           f'means, not corrected for multiple comparisons)\n\n' + md(t4))

text = '\n\n'.join(out) + '\n'
open(os.path.join(R, 'tjepa_tables.md'), 'w').write(text)
print(text)

# --- REPORT.md section 7.5: narrative + tables between the TJEPA markers (idempotent)
block = [open(os.path.join(R, 'tjepa_narrative.md')).read().strip(), text.strip()]
toy = os.path.join(R, 'toy_two_features.md')
if os.path.exists(toy):
    block.append(open(toy).read().strip())
block = '\n\n'.join(block)
rp = os.path.join(ROOT, 'REPORT.md')
rep = open(rp).read()
B, E = '<!-- TJEPA:BEGIN -->', '<!-- TJEPA:END -->'
if B in rep and E in rep:
    rep = rep[:rep.index(B) + len(B)] + '\n\n' + block + '\n\n' + rep[rep.index(E):]
    open(rp, 'w').write(rep)
    print('REPORT.md section 7.5 refreshed')
