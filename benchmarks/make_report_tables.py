"""Turn results/*.csv into the markdown tables used in REPORT.md section 7."""
import os, sys
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, 'results')

def md(df, floatfmt='{:.3f}'):
    cols = list(df.columns)
    lines = ['| ' + ' | '.join([df.index.name or ''] + [str(c) for c in cols]) + ' |',
             '|' + '---|' * (len(cols) + 1)]
    for idx, row in df.iterrows():
        vals = []
        for v in row.values:
            if isinstance(v, (float, np.floating)):
                vals.append('' if np.isnan(v) else floatfmt.format(v))
            else:
                vals.append(str(v))
        lines.append('| ' + ' | '.join([str(idx)] + vals) + ' |')
    return '\n'.join(lines)

def bold_best(df):
    """Bold the best value per row (higher is better)."""
    out = df.copy().astype(object)
    for i, row in df.iterrows():
        if row.notna().any():
            best = row.idxmax()
            for c in df.columns:
                v = row[c]
                out.at[i, c] = '' if pd.isna(v) else (f'**{v:.3f}**' if c == best else f'{v:.3f}')
    return out

parts = []
# ---------------- tabular ----------------
f = os.path.join(R, 'tabular_results.csv')
if os.path.exists(f):
    df = pd.read_csv(f)
    df['dataset'] = df['dataset'].str.replace(r'^\d+_', '', regex=True)
    main_models = ['JEPAOD[pred]', 'JEPAOD[embed]', 'JEPAOD[combined]', 'AutoEncoder', 'DeepSVDD', 'LUNAR', 'DIF',
                   'IForest', 'KNN', 'LOF', 'ECOD', 'COPOD', 'PCA', 'HBOS', 'OCSVM']
    m = df[df.model.isin(main_models)].groupby(['dataset', 'model'])[['auc', 'ap', 'time']].mean().reset_index()
    for metric in ('auc', 'ap'):
        piv = m.pivot(index='dataset', columns='model', values=metric)[[c for c in main_models if c in m.model.unique()]]
        ranks = piv.rank(axis=1, ascending=False)
        summary = pd.DataFrame({'mean': piv.mean(0), 'avg rank': ranks.mean(0), '#wins': (ranks == 1).sum(0)}).T
        piv.columns.name = None; piv.index.name = 'dataset'
        table = bold_best(piv)
        table.loc['**mean**'] = [f'**{v:.3f}**' if v == summary.loc['mean'].max() else f'{v:.3f}' for v in summary.loc['mean']]
        table.loc['**avg rank**'] = [f'**{v:.2f}**' if v == summary.loc['avg rank'].min() else f'{v:.2f}' for v in summary.loc['avg rank']]
        table.loc['**wins**'] = [f'{int(v)}' for v in summary.loc['#wins']]
        n_seeds = df.seed.nunique()
        parts.append(f'### 7.1 Tabular, ADBench ({piv.shape[0]} datasets, 60/40 split, mean of {n_seeds} seeds): {metric.upper()}\n\n' + md(table))
    # JEPA variants
    jv = df[df.model.str.startswith('JEPAOD')].groupby('model')[['auc', 'ap', 'time']].mean()
    jv.index.name = 'JEPAOD variant'
    parts.append('### 7.2 Tabular JEPA-OD variants (mean over datasets and seeds)\n\n' + md(jv.round(3)))
    t = df.groupby('model')['time'].mean().sort_values(); t.index.name = 'model'
    parts.append('### 7.3 Mean fit+score time per dataset split (s)\n\n' + md(t.to_frame('seconds').round(2)))

# ---------------- time series ----------------
f = os.path.join(R, 'ts_results.csv')
if os.path.exists(f):
    df = pd.read_csv(f)
    # bench_ts.py ran the JEPA variants with the final default (encoder_norm=False);
    # bench_ts_jepa_norm.py re-ran them with a final encoder LayerNorm -> label those '(LN)'.
    f2 = os.path.join(R, 'ts_results_jepa_norm.csv')
    if os.path.exists(f2):
        d2 = pd.read_csv(f2)
        d2['model'] = d2['model'].str.replace('TSJEPAOD-', 'TSJEPAOD(LN)-', regex=False)
        df = pd.concat([df, d2], ignore_index=True)
    f3 = os.path.join(R, 'ts_results_jepa_final.csv')
    if os.path.exists(f3):
        d3 = pd.read_csv(f3)
        d3['model'] = d3['model'].str.replace('TSJEPAOD-', 'TSJEPAOD(final)-', regex=False)
        df = pd.concat([df, d3], ignore_index=True)
    df['group'] = df.dataset.str.split('-').str[0]
    order = ['TSJEPAOD(final)-block[pred]', 'TSJEPAOD(final)-block[combined]',
             'TSJEPAOD-block[pred]', 'TSJEPAOD-block[embed]', 'TSJEPAOD-block[combined]', 'TSJEPAOD-forecast[pred]',
             'TSJEPAOD-forecast[embed]', 'TSJEPAOD-forecast[combined]',
             'TSJEPAOD(LN)-block[pred]', 'TSJEPAOD(LN)-block[embed]', 'TSJEPAOD(LN)-forecast[pred]',
             'LSTMAD', 'AnomalyTransformer', 'TSOD-AutoEncoder',
             'TSOD-IForest', 'TSOD-KNN', 'MatrixProfile', 'SpectralResidual']
    for metric in ('auc', 'ap'):
        g = df.groupby(['group', 'model'])[metric].mean().reset_index()
        piv = g.pivot(index='group', columns='model', values=metric)
        piv = piv[[c for c in order if c in piv.columns]]
        piv.columns.name = None; piv.index.name = 'dataset group'
        counts = df.groupby('group').dataset.nunique()
        piv.index = [f'{i} ({counts[i]} series)' for i in piv.index]
        table = bold_best(piv)
        parts.append(f'### 7.4 Time series, {metric.upper()} averaged per dataset group (point-wise)\n\n' + md(table))
    # per-series detail for NAB and SKAB with the best JEPA variant vs baselines
    for grp in ('NAB', 'SKAB'):
        sub = df[df.group == grp]
        if len(sub) == 0: continue
        piv = sub.pivot_table(index='dataset', columns='model', values='auc', aggfunc='mean')
        piv = piv[[c for c in order if c in piv.columns]]
        piv.columns.name = None; piv.index.name = 'series'
        piv.index = piv.index.str.replace(grp + '-', '')
        parts.append(f'### {grp} per-series ROC-AUC\n\n' + md(bold_best(piv)))
    t = df.groupby('model')['time'].mean().sort_values(); t.index.name = 'model'
    parts.append('### Mean fit time per series (s)\n\n' + md(t.to_frame('seconds').round(2)))

# ---------------- ablation ----------------
f = os.path.join(R, 'ablation_results.csv')
if os.path.exists(f):
    df = pd.read_csv(f)
    g = df.groupby(['kind', 'encoder_norm', 'target_mode', 'score'])[['auc', 'ap']].mean().round(3).reset_index()
    g.index = range(len(g)); g.index.name = ''
    parts.append('### 7.5 Ablation: encoder final LayerNorm x collapse prevention (mean ROC-AUC / AP)\n\n' + md(g))

open(os.path.join(R, 'report_tables.md'), 'w').write('\n\n'.join(parts) + '\n')
print('\n\n'.join(parts))
