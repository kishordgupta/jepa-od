"""Compose REPORT.md section 7 from results/*.csv (idempotent: rewrites between the RESULTS markers)."""
import os, re, subprocess, sys
import pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, 'results')
subprocess.run([sys.executable, '-W', 'ignore', os.path.join(ROOT, 'benchmarks', 'make_report_tables.py')],
               check=True, capture_output=True)
tables = open(os.path.join(R, 'report_tables.md')).read()

def section(title_prefix):
    m = re.search(r'(### ' + re.escape(title_prefix) + r'.*?)(?=\n### |\Z)', tables, re.S)
    return m.group(1).strip() if m else '(table not available yet)'

tab_narr = open(os.path.join(R, 'tabular_narrative.md')).read().strip() if os.path.exists(os.path.join(R, 'tabular_narrative.md')) else '*(tabular benchmark still running)*'

ts_narr = """
**Setup.** Point-wise ROC-AUC / AP. *Synthetic*: PyOD's `generate_ts_data` (3 configurations x 2 seeds, 3 000 training / 1 500 test steps, 5 % contamination; window 60, patch 5). *NAB realKnownCause*: 7 real univariate series (1.9k-22.7k steps), transductive protocol (fit on the whole series, score its own `decision_scores_`), NAB label windows (~10 % of points); window 100, patch 10. *SKAB valve1*: 8-channel industrial test-bed, fit on the `anomaly-free` recording, score each of the 16 valve1 files (~1 150 steps, 35 % anomalous); window 60, patch 5. Baselines are PyOD 3.6 detectors: `LSTMAD`, `AnomalyTransformer` (d_model 64), `MatrixProfile`, `SpectralResidual`, and `TimeSeriesOD` around `IForest`, `KNN`, `AutoEncoder`. `TSJEPAOD` rows: `block` / `forecast` masking x `pred` / `embed` / `combined` score, 20 epochs with the no-norm encoder (the bulk run); `(final)` = the shipped defaults (10 epochs, <= 5 000 training windows); `(LN)` = encoder with a final LayerNorm (ablation).

**Findings.**

* **Synthetic (structure-breaking anomalies):** every `TSJEPAOD` variant (0.93-0.96 AUC) beats all baselines; `LSTMAD`, the raw-space forecasting detector, is next at 0.87, windowed `KNN` / `AutoEncoder` reach 0.78, `AnomalyTransformer` 0.59. Latent prediction error is the right signal when anomalies are violations of temporal structure rather than level excursions.
* **SKAB (train on clean recording, strong regime shift at test time):** `TSJEPAOD-forecast[pred]` 0.859 and `block[pred]` 0.834 are on par with `LSTMAD` (0.853) and far above windowed `AutoEncoder` (0.68) and `IForest` (0.53). Windowed `KNN` is almost perfect (0.994): SKAB anomalies are large raw-space deviations, so plain nearest-neighbour distance on raw windows suffices. The JEPA `embed` score stays valid here (0.82-0.84) *only* with the no-norm encoder; with a final LayerNorm it collapses to 0.20 (inverted) because every shifted patch saturates to the same direction (diagnostic: all test patches, normal and anomalous alike, sit at Mahalanobis ~85 vs ~63 for training patches; one channel differs by ~58 training standard deviations between the recordings).
* **NAB (real univariate, level-shift / regime anomalies, transductive):** windowed `IForest` (0.825) and `AutoEncoder` (0.776) lead; `TSJEPAOD` reaches 0.62-0.67, above `LSTMAD` (0.56), `AnomalyTransformer` (0.46) and `SpectralResidual` (0.46), level with `MatrixProfile` (0.69). Two reasons, both verified in `results/diag_nab.csv`: (i) a sustained level shift is locally *predictable* once inside it, so the prediction error fires at the transition only, whereas a raw-window density method flags the whole shifted segment; (ii) with 10 % of the series anomalous, training on the whole series lets the model absorb the anomalous regime: on `machine_temperature` the `pred` AUC falls from 0.84 (5 epochs) to 0.55 (20 epochs), and fitting on the first half only (inductive) restores 0.93 (IForest: 0.96). This motivated the smaller default training budget.
* **Score choice.** `pred` is the most robust single score across the three protocols (0.95 / 0.83 / 0.63); `embed` wins only in-distribution (synthetic, NAB with LayerNorm) and is fragile under shift; `combined` is rarely the best of the three and never the worst. `pred` stays the default.
* **Masking.** `forecast` (causal) is slightly better than `block` on SKAB and synthetic (0.86 vs 0.83, 0.96 vs 0.95) but worse on NAB (0.59 vs 0.63); `block` stays the default for offline detection, `forecast` is the online option.
* **Training budget.** The shipped defaults (10 epochs, <= 5 000 windows) match the 20-epoch run within 0.01 AUC on all three groups at about half the fit time (`results/budget_experiment.csv`).
"""

abl_narr = """
`results/ablation_results.csv` (block masking, `combined`-trained models scored three ways; tabular: 8 small ADBench sets, seed 0; time series: 3 synthetic + 7 NAB series):

* **SIGReg vs EMA teacher.** SIGReg wins everywhere: tabular mean AUC 0.77-0.81 vs 0.67-0.75 (EMA), time series 0.72-0.77 vs 0.69-0.71. The EMA variant is also the one whose `embed` score is weakest, consistent with target-scale drift.
* **Encoder final LayerNorm.** Off is better for tabular data (0.804 vs 0.762 `combined`), on is better *in-distribution* for time series (0.767 vs 0.725) but catastrophic under shift (SKAB above). Both detectors ship with it off; the switch (`encoder_norm`) is kept for reproducibility.
* **Reconstruction anchor** (`recon_weight=0.1`, tabular bulk run): no gain over plain SIGReg (see 7.2), so it stays off.
"""

runtime = """
Laptop CPU (4 threads per run). `JEPAOD` (30 epochs, 8 inference masks, 3 scoring passes inside `fit`) is the slowest tabular detector in the comparison, roughly 10x `AutoEncoder`; `epoch_num=10-15` or `device='mps'/'cuda'` bring it to a few seconds. `TSJEPAOD` with the final defaults fits a 20k-step series in ~10 s, between `TimeSeriesOD(AutoEncoder)` and `AnomalyTransformer`.
"""

def tab_section(metric):
    m = re.search(r'(### 7\.1 Tabular, ADBench \([^\n]*?\): ' + metric + r'\n.*?)(?=\n### |\Z)', tables, re.S)
    if not m:
        return '(table not available yet)'
    return re.sub(r'### 7\.1 Tabular, ADBench \((.*?)\): ' + metric,
                  lambda mm: '#### ADBench, ' + mm.group(1) + ': ' + ('ROC-AUC' if metric == 'AUC' else 'AP'),
                  m.group(1).strip())

tab_auc, tab_ap = tab_section('AUC'), tab_section('AP')
tab_var = section('7.2 Tabular JEPA-OD variants').replace('### 7.2', '####')
ts_auc = section('7.4 Time series, AUC').replace('### 7.4 Time series, AUC averaged per dataset group (point-wise)', '#### ROC-AUC per dataset group')
ts_ap = section('7.4 Time series, AP').replace('### 7.4 Time series, AP averaged per dataset group (point-wise)', '#### AP per dataset group')
nab = section('NAB per-series').replace('### ', '#### ')
skab = section('SKAB per-series').replace('### ', '#### ')
abl = section('7.5 Ablation').replace('### 7.5 Ablation', '#### Ablation')
t_tab = section('7.3 Mean fit+score time').replace('### 7.3', '#### Tabular:')
t_ts = section('Mean fit time per series').replace('### ', '#### Time series: ')

out = "\n\n".join([
    "<!-- RESULTS:BEGIN -->",
    "### 7.1 Tabular (ADBench)", tab_narr, tab_auc, tab_ap, tab_var,
    "### 7.2 Time series (synthetic, NAB, SKAB)", ts_narr.strip(), ts_auc, ts_ap, nab, skab,
    "### 7.3 Ablations and diagnostics", abl_narr.strip(), abl,
    "### 7.4 Runtime", runtime.strip(), t_tab, t_ts,
    "<!-- RESULTS:END -->"])

p = os.path.join(ROOT, 'REPORT.md'); s = open(p).read()
s = re.sub(r'<!-- RESULTS:BEGIN -->.*?<!-- RESULTS:END -->', lambda m: out, s, flags=re.S)
open(p, 'w').write(s)
print('REPORT.md section 7 filled;', len(s.split()), 'words')
