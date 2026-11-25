# Plan: ten further JEPA versions and all nine TSB-AD measures

Written on 2026-11-25, after the ten versions were implemented and their runs were started, and before any result of
them was read (two series per version were run to see that the code works; only their run times were looked at).
The constants of the ten versions, including those the papers leave open, are in `jepa_od/_recipes2.py` as committed
with this file; none will be changed after results are seen.

## What is run

* **Versions.** The ten `variant` values added in version 0.5.0: `ststjepa` (STST-JEPA, arXiv 2607.06629), `hepa`
  (HEPA, 2605.11130), `subjepa` (Sub-JEPA, 2605.09241), `straight` (temporal straightening, 2603.12231),
  `motionjepa` (MotionJEPA, 2609.23881), `njepa` (N-JEPA, 2507.15216), `gmmjepa` (GMM-anchored JEPA, 2602.09040),
  `variational` (VJEPA, 2601.14354), `phijepa` (PhiJEPA, 2611.27516), `harjepa` (HAR-JEPA, 2607.16350).
* **Lists.** All five lists of sections 7.9 and 7.10: M-Eva (180 series), U-Eva (350), M-Tuning (20), U-Tuning (48)
  and U-Rest (472). No list is held out for these versions, and none has been used to choose anything about them.
* **Protocol.** That of section 7.9: class defaults, seed 42, fit on the training prefix, score the whole series,
  trained (10 epochs) and untrained (0 epochs). `subjepa`, `straight` and `motionjepa` start from the network of
  `lewm`, so their untrained control is `lewm`'s (a unit test checks that the scores are equal).
* **Class.** `jepa_od.TSJEPAOD`, not the PyOD port, which does not have the ten versions. For `lejepa` and `vicreg`
  the two classes gave identical scores on ten series (largest difference 0.0 for all three score types).

## Tests fixed in advance

Measure: VUS-PR, paired by series, two-sided Wilcoxon signed-rank test, default score `pred`.

* (A) each new version, trained, against the default version (`lejepa`, trained): 10 versions x 5 lists = 50 tests.
* (B) each new version, trained, against its untrained control: 50 tests.

Holm correction over all 100. Reading of a test: "ahead" or "behind" when the corrected p is below 0.05, otherwise
"not separated". M-Tuning has 20 series; a test there can only show large differences.

## Reported without a test

* The `embed` and `combined` scores of the ten versions, and the cell with the highest mean per list.
* The control without a network (raw patches) and the re-run reference detectors (PCA, Sub-PCA) next to them.
* **All nine measures of TSB-AD** (VUS-PR, VUS-ROC, AUC-PR, AUC-ROC, Standard-F1, PA-F1, Event-based-F1, R-based-F1,
  Affiliation-F) for every saved run: the sixteen older versions, the ten new ones, the controls and the reference
  detectors, on all five lists. They are computed from the saved score vectors with `jepa_od/metrics.py`, which is
  compared with TSB-AD's own code first (`benchmarks/check_metrics_tsb.py`). The four measures already in
  `results/tsb/metrics/` are recomputed and compared with the stored values.
  VUS-PR stays the measure the conclusions are drawn from. The five F measures use the best of 100 thresholds, chosen
  with the labels, as TSB-AD reports them; PA-F1 is known to reward a single hit in a long anomaly.
  TSB-AD publishes per-series numbers of its detectors for VUS-PR only, so places among published detectors can be
  given for that measure only.

## Small benchmarks (synthetic, SKAB, NAB)

The protocol of section 7.8: 5 seeds, ROC-AUC of `block[pred]`, for each new version and each of the three groups
(a) trained against its zero-epoch control and (b) trained against the trained default: 10 x 3 x 2 = 60 tests,
Holm over the 60.

## Not done

No tuning of any constant; one seed on TSB-AD; no re-run of further published detectors; the default `variant` and
the default `score` stay as they are whatever the outcome.
