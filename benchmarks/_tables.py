"""Markdown and statistics helpers shared by the table scripts."""
import numpy as np
from scipy.stats import wilcoxon


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
