# JEPA-OD in PyOD: port status and hand-off

## Status

**Read first (2026-11-21):** the extended benchmark in `REPORT.md` section 7.6 was run after this port.
On all 47 ADBench datasets the tabular detector is mid-table and an untrained network scores at least as
well, so the report no longer argues for adding `JEPA` on performance grounds. Issue 795 was written
before that result; the comment in `PYOD_ISSUE_COMMENT_DRAFT.md`, posted on the issue on 2026-11-21, tells
the maintainer and offers to narrow the proposal to `TSJEPA`, as a family with a selectable JEPA version
(`variant`, `REPORT.md` section 4.8). Wait for the answer before any pull request.

## The branch for the narrowed proposal: `feature/ts-jepa` (2026-11-21, second commit 2026-11-22)

Built so that a pull request can be opened as soon as the maintainer accepts the `TSJEPA`-only proposal:
https://github.com/kishordgupta/pyod/tree/feature/ts-jepa

* **Three commits** on top of `yzhao062/pyod:development` at `f785bc9` (PyOD 3.6.9): `455ca3c`, "Add TSJEPA
  detector for time series (experimental)", with the first four versions; `cb26e0c`, "TSJEPA: twelve
  further JEPA versions for `variant`" (next section); and `0fa3372`, "TSJEPA tests: follow the device of
  the fitted network" (the section after it). No tabular `JEPA`, no workflow file touched.
  Upstream has since added one merge commit without file changes (`e9d8670`), so GitHub shows the branch
  as behind by one. Rebase onto the then-current `development` before opening the pull request.
* **What the issue says about the branch.** The first comment describes four versions. The second, posted
  on 2026-11-22 (https://github.com/yzhao062/pyod/issues/795#issuecomment-7077175222, text in
  `PYOD_ISSUE_COMMENT2_DRAFT.md`), links this branch and proposes three more: `kerjepa`, `vicreg` and
  `lambdajepa`. It does not say that the branch holds the nine other versions too, so the maintainer may
  ask about them when opening the branch.

The bullets and the table below describe the first commit.

* **Content.** `pyod/models/ts_jepa.py` (`TSJEPA` with `variant` = `'lejepa'`, `'ijepa'`, `'tsjepa'`,
  `'tjepa'`; `encoder_norm=None` and `ema_momentum=None` resolve per variant; `target_mode` is gone),
  `pyod/models/_jepa_utils.py` (the networks of the four variants, SIGReg, EMA update, mask samplers),
  `pyod/test/test_ts_jepa.py` (28 tests), `examples/ts_jepa_example.py`, the API page, the model-table
  rows in `README.rst` and `docs/index.rst`, two bibliography entries, the two experimental-detector
  lists, and the `conftest.py` skip-list line.
* **Same patches here:** `integration/pyod-patches-ts-jepa/` (one per commit), applied with
  `git switch -c feature/ts-jepa upstream/development && git am --keep-cr <patches>`. Checked: the first
  two applied in order to `f785bc9` give the tree of `cb26e0c`; the third is `git format-patch` of `0fa3372`.
* **Parity with this repository.** On two series, four variants and five settings each (block, forecast
  and 70 % random masking, `encoder_norm=False` with another momentum, zero epochs), `TSJEPA` and
  `TSJEPAOD` return identical scores for `'lejepa'`, `'ijepa'` and `'tsjepa'`. For trained `'tjepa'` they
  agree to 2e-6: the prototype sends its single target set through the multi-target code of the tabular
  class, an identity reshaping that changes the last float digits; with that step added the port is
  identical too.

Checks that ran on `455ca3c` (Python 3.11.18, torch 2.15.0, macOS arm64, CPU):

| check | result |
|---|---|
| `pytest pyod/test/test_ts_jepa.py` | 28 passed, about 65 s |
| the time-series suites, `test_base`, `test_readme_rst`, `test_skill_api_refs`, `test_skill_kb_consistency`, `test_utility` | all passed |
| `python scripts/regen_skill.py --check` | skill files up to date |
| whole `pyod/test` directory | 1576 passed, 68 skipped, 0 failed. Not run here because optional packages are missing: `test_thresholds.py` (15 errors, `pythresh`) and five modules that need `combo`, `statsmodels` or `xgboost` |
| `examples/ts_jepa_example.py` | runs, ROC 0.954 |
| Sphinx build of the module page | no warnings from the new docstrings |
| `import pyod.models.ts_jepa` with torch blocked | imports without torch; `fit` raises the install hint |
| Python 3.9.6 `ast.parse` of the four Python files | parses |

Not run: PyOD's CI matrix on this branch (it needs a pull request, see below), the full documentation
build, Coveralls, any GPU.

Left out on purpose, as before: `CHANGES.txt`, the knowledge base and its detector counts, and the
`_DETECTOR_SHORTCUTS` line.

## Second commit on `feature/ts-jepa`: the twelve further versions (2026-11-22)

`cb26e0c` ports the twelve versions of `REPORT.md` section 4.9 (`vjepa`, `vjepa21`, `cjepa`, `sjepa`, `salt`,
`vicreg`, `kerjepa`, `lpjepa`, `lambdajepa`, `lenepa`, `lewm`, `cfjepa`), so `TSJEPA(variant=...)` has the same
sixteen values as `TSJEPAOD`. 2328 lines added, against 2120 in the first commit.

| this repo | PyOD | notes |
|---|---|---|
| `jepa_od/_regs.py`, `_variant_nets.py`, `_recipes.py` | `pyod/models/_jepa_variants.py` (new) | one private module in three parts: regularisers, networks, recipes. The networks keep only what the detector calls (no `token_embeddings`) |
| `jepa_od/ts_jepa.py` | `pyod/models/ts_jepa.py` | parameter `variant_params`, attribute `variant_params_`, `_recipe()`, `_loss()`; the variant names are a static tuple so the module still imports without torch |
| `momentum_at` in `_recipes.py` | `pyod/models/_jepa_utils.py` | next to `ema_update` |
| `tests/test_regs.py`, `tests/test_ts_jepa_versions.py` | `pyod/test/test_ts_jepa_variants.py` (new) | 27 tests on a smaller series (1000 timestamps, 2 epochs); listed in `conftest.py` |
| -- | `examples/ts_jepa_example.py` | one comment |

Differences from the prototype, none of which changes a score:

* SALT's `freeze_teacher` also clears the teacher's gradients, so the second stage does not depend on how
  the optimizer resets gradients.
* `Recipe.ema_pairs` is gone: every version with an EMA target updates `target_encoder` from `encoder`.
* The docstring of `TSJEPA` lists the sixteen versions with their references and names the ten versions
  whose normalised targets can fail under a regime change.
* Nothing was added to `README.rst`, `docs/index.rst`, the bibliography or the knowledge base: the detector's
  row and its two citations are those of the first commit.

**Parity with this repository.** On three settings (block masking with two channels, forecast masking with
one channel and a window of 100, and seven versions with overridden constants), 39 fits in all, `TSJEPA` and
`TSJEPAOD` return bit-identical training scores and `pred` / `embed` / `combined` test scores for fifteen
versions. `tjepa` differs by at most 1.5e-5, as in the first commit. The 72 score arrays of the first four
versions (three mask modes, with and without the reconstruction loss) are bit-identical before and after
the commit.

Checks that ran on `cb26e0c` (Python 3.11.18, torch 2.15.0, macOS arm64, CPU):

| check | result |
|---|---|
| `pytest pyod/test/test_ts_jepa.py pyod/test/test_ts_jepa_variants.py` | 55 passed (28 + 27), about 110 s |
| line coverage of `ts_jepa.py`, `_jepa_utils.py` and `_jepa_variants.py` under those two modules | 99.03 % (14 of 1439 lines missed: the torch-missing branch, an explicit device, a 2-D SIGReg input, three error branches, three abstract methods). Measured with a `sys.settrace` script, not with Coveralls |
| whole `pyod/test` directory | 1603 passed (the 1576 of the first commit plus the 27 new tests), 68 skipped, 0 failed. Not run here, as before, because optional packages are missing: `test_thresholds.py` (15 errors, `pythresh`) and five modules that need `combo`, `statsmodels` or `xgboost` |
| `python scripts/regen_skill.py --check` | skill files up to date |
| `examples/ts_jepa_example.py` | runs, ROC 0.954 |
| Sphinx build of the module page | no warnings from the new docstring |
| `import pyod.models.ts_jepa` with torch blocked | imports without torch and without `_jepa_variants`; `fit` raises the install hint for a recipe version too |
| Python 3.9.6 `ast.parse` of the six changed Python files | parses |

Not run: PyOD's CI matrix on this branch (it needs a pull request inside the fork; the author said "not
now" on 2026-11-21), the full documentation build, Coveralls. No GPU was used for this commit; the next
section is the GPU check.

## Third commit on `feature/ts-jepa`: the tests on a GPU (2026-11-24)

The branch had never run on a GPU. On a server with NVIDIA H100 cards (Linux, Python 3.10.12, torch
2.14.1 with CUDA 13.0) the detector ran on the first try, and six of the 55 tests failed: with
`device='auto'` the fitted network is on the GPU, and those tests compared it with a network on the CPU
or fed it CPU tensors. `0fa3372` moves the tensors of the tests to the device of the fitted network. The
detector's code is unchanged.

| check on the GPU server | result |
|---|---|
| `pytest pyod/test/test_ts_jepa.py pyod/test/test_ts_jepa_variants.py` on a GPU | 55 passed (6 failed before `0fa3372`) |
| the same with no GPU visible (`CUDA_VISIBLE_DEVICES=`) | 55 passed |
| the same on macOS (CPU) | 55 passed |
| all sixteen versions fitted on the CPU and on the GPU, three settings (`benchmarks/gpu_check.py`, `results/gpu/gpu_check.json`) | every fit runs; parameters and buffers are on the requested device; scores, `predict_proba` and a pickle round trip are fine |
| CPU scores against GPU scores, same seed | Spearman at least 0.996, ROC-AUC within 0.007, labels agree on at least 99.1 % of the timestamps |
| a second GPU fit with the same seed | identical scores for all sixteen versions |
| `device='cuda:0'` and `device='auto'` | both put the network on the GPU |

Speed of one fit, GPU against 8 CPU threads of the same server (median over the sixteen versions):

| setting | CPU time / GPU time |
|---|---|
| the test setting (1000 timestamps, 2 epochs) | 0.78: the GPU is slower |
| the class defaults on 20,000 timestamps | 0.53: the GPU takes about twice as long (only `cfjepa` is faster on it, 1.6) |
| `d_model=256`, 4 layers, batch 256, 20,000 training windows, 8 channels | 3.2 (2.0 to 10.9): the GPU is faster |

So with the default network a GPU does not pay, and `device='auto'` picks it anyway (PyOD's other
detectors have the same default).

Found on long series and **not changed in the branch** (`benchmarks/bench_tsb.py` works around the first
three in a subclass, with scores equal up to rounding):

1. Scoring runs `batch_size` (64) windows at a time. On the GPU a series of 200,000 timestamps takes 26 s
   to score; 2048 windows at a time take 5.7 s.
2. `_windows` builds all windows by indexing: three times slower than a copy of a strided view, with
   three times the peak memory. All windows of a series are in memory at once (timestamps x window x
   channels x 4 bytes: several GB for the longest multivariate TSB-AD series).
3. `_mahalanobis` contracts three arrays in one `einsum`, which does not use BLAS: most of the time of the
   `embed` and `combined` scores on long series.
4. A channel that is constant in the training data is divided by 1e-8 when scoring, so any later change
   of it gives a value of the order of 1e8. 63 of the 180 multivariate TSB-AD series have such a channel.
   The benchmark below measured it: on those 63 series the factor raises the mean VUS-PR from 0.298 to
   0.328 (it acts as an alarm for a channel that never moved in training), so replacing it by a unit
   scale, PyOD's usual guard, would cost 0.011 on the mean of the multivariate list. It is an accident
   all the same, and a decision for the maintainer.

## TSB-AD benchmark of the branch (2026-11-24)

`TSJEPA` of this branch (`0fa3372`) was run on the two evaluation lists of TSB-AD, 180 multivariate and
350 univariate series, at its class defaults and untuned, for all sixteen versions, trained and with zero
epochs: 75 runs of a whole list, all complete (`REPORT.md` section 7.9 of the prototype repository,
`results/tsb/`). What matters for a pull request:

| | TSB-AD-M | TSB-AD-U |
|---|---|---|
| best published mean VUS-PR | 0.313 (CNN) | 0.423 (Sub-PCA) |
| `TSJEPA()` trained, score `pred` (the defaults) | 0.351, above all 23 published means | 0.378, seventh of 33 |
| `TSJEPA(epochs=0)`, `pred` | 0.347 | 0.346 |
| `TSJEPA()`, score `embed` | 0.353 | 0.461 |
| `TSJEPA()`, score `combined` | 0.352 | 0.455 |
| `TSJEPA(variant='vicreg')`, `pred` | 0.360 | 0.417 |
| `TSJEPA(variant='lewm')`, `pred` | 0.386 | 0.394 |
| `TSJEPA(variant='cfjepa')`, `embed` (picked after the results) | 0.434 | 0.530 |
| seeds 42 / 0 / 1 of the default | 0.351 / 0.352 / 0.351 | 0.378 / 0.377 / 0.376 |
| minutes for the whole list, default version, one shared H100 | 46 | 49 |

* It is a result the proposal can cite, with its limits stated: the published numbers are TSB-AD's own
  and tuned, and they were not re-run (the confirmation below re-runs PCA and Sub-PCA); the lead on the
  multivariate list is in the mean and not per series (better than CNN on 87 of 180, p = 0.11); 0.011
  of it is the accident of item 4 above.
* It does not show that JEPA training is the reason. Training adds +0.004 and +0.032 to the default
  score (both significant), and a Mahalanobis distance on raw patches with no network scores 0.345 and
  0.465.
* Two defaults could be reconsidered before a pull request and were left as they are: `variant`
  (`vicreg` and `lambdajepa` are significantly ahead of `lejepa` on both lists and level with it on the
  earlier benchmarks) and `score` (`combined` is level with `pred` on TSB-AD-M and 0.077 ahead on
  TSB-AD-U).
* Nothing about this benchmark has been posted on issue 795.

### Confirmation on 540 series no run had used (2026-11-24)

The same code was then run on the rest of TSB-AD, with the lists, the 99 runs and 18 tests fixed and
committed beforehand (`results/tsb_confirm_plan.md`, `REPORT.md` section 7.10): the tuning lists (20
multivariate, 48 univariate series) and the 472 univariate series outside the evaluation and tuning lists
("U-Rest": YAHOO, UCR, WSD and Stock only). Mean VUS-PR:

| | M-Tuning (20) | U-Tuning (48) | U-Rest (472) |
|---|---|---|---|
| TSB-AD's own PCA / Sub-PCA, re-run | 0.323 | 0.326 | 0.146 |
| `TSJEPA()` trained, score `pred` (the defaults) | 0.363 | 0.337 | 0.177 |
| `TSJEPA(epochs=0)`, `pred` | 0.353 | 0.304 | 0.146 |
| `TSJEPA()`, score `embed` | 0.373 | 0.449 | 0.432 |
| `TSJEPA()`, score `combined` | 0.368 | 0.443 | 0.401 |
| no network: Mahalanobis distance of raw patches | 0.374 | 0.458 | 0.545 |
| `TSJEPA(variant='vicreg')`, `pred` | 0.390 | 0.410 | 0.242 |
| `TSJEPA(variant='lewm')`, `pred` | 0.441 | 0.398 | 0.213 |
| `TSJEPA(variant='cfjepa')`, `embed` | 0.528 | 0.508 | 0.333 |

* What a proposal can now say without the reservation about selection: `vicreg`, `lambdajepa` and `lewm`
  give a higher `pred` score than the default `lejepa` on all five lists (significant after correction on
  both new univariate lists for `vicreg` and `lambdajepa`, and on M-Tuning and U-Rest for `lewm`), and
  `kerjepa` is level with it on all five.
* What it should no longer say: that `cfjepa` with `embed` is the best configuration. It is ahead of the
  default detector on every list, and on U-Rest it is 0.21 below the control without a network. It gains
  on UCR, the ECG sets, SMAP and NAB and loses on YAHOO.
* The default `score` matters more than the default `variant` on univariate series: `combined` is 0.077,
  0.106 and 0.224 above `pred` on the evaluation list, U-Tuning and U-Rest, and level with it on the two
  multivariate lists. This comparison was not among the tests fixed in advance. Changing the default
  remains a decision for the maintainer and was not made.
* The control without a network is level with `embed` on four lists and 0.113 above it on U-Rest. A
  reviewer who asks what the network adds to `embed` will not find an answer in these numbers.
* Against TSB-AD's own detectors: the default is +0.040 above re-run PCA on M-Tuning (13 of 20 series,
  p = 0.23) and above re-run Sub-PCA on U-Tuning (+0.011, p = 0.23) and U-Rest (+0.031). Sub-PCA is not the
  strong published detector on the four sources of U-Rest, and CNN and OmniAnomaly were not re-run, so
  the place among published detectors is still known for the evaluation lists only.
* Nothing about the confirmation has been posted on issue 795.

## The first port: `feature/jepa-od`

This is the branch linked in issue 795. It has the tabular `JEPA` and a `TSJEPA` without `variant`
(it still has `target_mode`). It is kept as it was.

The detectors are ported to PyOD as two commits on the branch `feature/jepa-od`, based on
`yzhao062/pyod` branch `development` at commit `9680a8b` (PyOD 3.6.8).

| commit | content |
|---|---|
| `Add JEPA detector for tabular data (experimental)` | `pyod.models.jepa.JEPA`, shared helpers, tests, example, docs |
| `Add TSJEPA detector for time series (experimental)` | `pyod.models.ts_jepa.TSJEPA`, tests, example, docs |

**The branch is on GitHub:** https://github.com/kishordgupta/pyod/tree/feature/jepa-od. It is two
commits ahead of `yzhao062/pyod:development` and zero behind, and touches no workflow file. The
proposal was posted on 2026-11-21 as https://github.com/yzhao062/pyod/issues/795. No pull request has
been opened on PyOD.

Getting it there needed one extra step. The fork was 924 commits behind upstream, and a push that
carries upstream's changes to `.github/workflows/` is refused for a `gh` login without the
`workflow` scope. The fork's `development` branch was therefore synced first with "Sync fork" on
GitHub, after which the push went through. The same applies whenever the fork falls behind again.

The same two commits are also kept as patches in `integration/pyod-patches/`. They apply to an
up-to-date PyOD checkout with:

```bash
git switch -c feature/jepa-od upstream/development
git am --keep-cr /path/to/jepa-od/integration/pyod-patches/*.patch
```

`--keep-cr` is needed because `README.rst` and `docs/index.rst` use CRLF line endings in PyOD.

PyOD's test workflow runs on pushes to `master` and `development` and on pull requests against
them, so the push of `feature/jepa-od` did not start CI. To run it without involving upstream, a
draft pull request was opened inside the fork, from `feature/jepa-od` into the fork's own
`development`: https://github.com/kishordgupta/pyod/pull/1. It exists only to run the matrix:
Ubuntu and Windows on Python 3.9 to 3.13, plus one macOS job on Python 3.9. The 44 new tests run
in all of them. The workflow means to leave torch out on macOS, but the job log shows torch being
installed, so the torch-dependent tests ran there as well. Do not merge the pull request, because
that would make the fork's `development` diverge from upstream.

Result of the run on 2026-11-21 (commit 5a225ef): all 12 jobs passed. Each of the 11 test-job logs
shows 1714 passed, 18 skipped, with every test in `test_jepa.py` and `test_ts_jepa.py` passing.
Test time per job ranged from 14 to 42 minutes, against a 60-minute job limit. Coveralls reported
98.36% patch coverage (840 of 854 changed lines) and overall coverage up 0.3 points to 92.993%.

## File map

| this repo | PyOD | notes |
|---|---|---|
| `jepa_od/_nets.py`, `jepa_od/_sigreg.py` | `pyod/models/_jepa_utils.py` | merged into one private helper module, like `_ts_utils.py` |
| `jepa_od/jepa.py` (`JEPAOD`) | `pyod/models/jepa.py` (`JEPA`) | on `BaseDeepLearningDetector`; every parameter documented |
| `jepa_od/ts_jepa.py` (`TSJEPAOD`) | `pyod/models/ts_jepa.py` (`TSJEPA`) | torch imported lazily, so the module imports without torch |
| `tests/test_jepa.py` | `pyod/test/test_jepa.py` | 27 tests, structure of `test_auto_encoder.py` |
| `tests/test_ts_jepa.py` | `pyod/test/test_ts_jepa.py` | 17 tests, structure of `test_ts_lstm.py` |
| `examples/jepa_example.py` | `examples/jepa_example.py`, `examples/ts_jepa_example.py` | one example per detector |
| -- | `pyod/test/conftest.py` | both test modules added to the torch-dependent skip list |
| -- | `docs/pyod.models.tabular.rst`, `docs/pyod.models.timeseries.rst` | API reference entries |
| -- | `README.rst`, `docs/index.rst` | model table rows, marked experimental |
| -- | `docs/zreferences.bib` | LeCun 2022, Assran et al. 2023, Balestriero & LeCun 2025 |
| -- | `docs/examples/timeseries.rst`, `docs/benchmark.rst` | lists of experimental time-series detectors |

Behaviour is unchanged from this repo: on the same data and seeds the PyOD classes return scores
identical to `JEPAOD` and `TSJEPAOD` (maximum absolute difference 0.0 in five configurations). The
only functional addition is a `ValueError` for `max_tokens < 2`.

## Checks that ran

Environment: Python 3.11.18, torch 2.15.0, numpy 2.4.6, scikit-learn 1.9.1, macOS arm64, CPU.

| check | result |
|---|---|
| `pytest pyod/test/test_jepa.py pyod/test/test_ts_jepa.py` | 44 passed, about 50 s |
| the two files plus `test_readme_rst`, `test_base_dl`, `test_ts_lstm`, `test_ts_anomaly_transformer`, `test_ts_od`, `test_knowledge`, `test_skill_kb_consistency`, `test_regen_skill`, `test_skill_api_refs`, `test_save_load_clone` | 106 passed |
| both example scripts | run to completion |
| Sphinx build of the two modules, the two table rows and the bibliography | no warnings from the new content |
| `import pyod.models.ts_jepa` with torch unloaded | does not import torch |
| Python 3.9.6 `ast.parse` of the seven new files | parses |

Not run: the full PyOD test suite, the full documentation build, any Python version other than
3.11 beyond the 3.9 syntax check, and any GPU.

ROC-AUC / average precision on PyOD's bundled data (`notebooks/data/`, one stratified 60/40 split,
seed 42, default parameters):

| data | JEPA | AutoEncoder | IForest | KNN | JEPA fit+score time |
|---|---|---|---|---|---|
| cardio | 0.919 / 0.483 | 0.739 / 0.296 | 0.945 / 0.631 | 0.726 / 0.337 | 14 s |
| pima | 0.657 / 0.489 | 0.575 / 0.434 | 0.638 / 0.462 | 0.579 / 0.419 | 3 s |
| shuttle | 0.887 / 0.485 | 0.991 / 0.843 | 0.997 / 0.980 | 0.740 / 0.205 | 220 s |

## Left out on purpose

* `CHANGES.txt`: PyOD's `CONTRIBUTING.md` says the maintainer writes the release notes.
* Knowledge base (`pyod/utils/knowledge/algorithms.json`) and the skill files generated from it.
  PyOD's headline "61 detectors" in `README.rst`, `docs/index.rst`, `pyproject.toml` and
  `docs/install.rst` counts the shipped plus experimental entries there, so registering two more
  changes those numbers. That is the maintainer's call. Entries that match the schema are drafted
  in `integration/pyod_knowledge_entries.json`; after adding them, run `python scripts/regen_skill.py`.
* `pyod/models/embedding.py::_DETECTOR_SHORTCUTS`: one optional line,
  `'JEPA': ('pyod.models.jepa', 'JEPA', {})`, lets `TimeSeriesOD(detector='JEPA')` and
  `EmbeddingOD(detector='JEPA')` resolve by name.

## From the branch to a pull request

PyOD's process (`CONTRIBUTING.md` on `development`):

1. Open an issue first. For a new detector it must state the inclusion criterion or a commitment to
   maintain the detector for at least two years. JEPA-OD is new, so it needs that commitment.
   Done on 2026-11-21: https://github.com/yzhao062/pyod/issues/795. The text as posted is in
   `integration/PYOD_ISSUE_DRAFT.md`.
2. Wait until the maintainer labels the issue `accepted` and assigns it. A pull request without
   such an issue may be closed without review.
3. Open one pull request against `development` and link the issue. "One change per pull request":
   if asked to split, the two commits are separable. The second needs `_jepa_utils.py` from the first.
4. The pull request template asks for an AI disclosure. Tools that contributed: Claude Code. Checks
   that ran: the table above. Whether a person read the full diff is for the author to answer.

## DeepOD alternative

`deepod.core.base_model.BaseDeepAD`: implement `training_prepare(X, y)` -> (loader, net, criterion),
`training_forward(batch_x, net, criterion)`, `inference_prepare(X)` -> loader and
`inference_forward(batch_x, net, criterion)` -> (z, score); set `data_type='ts'` and use DeepOD's
`seq_len` / `stride` windowing. `JEPACore` plugs in as `net` unchanged, and the SIGReg term is
computed inside `training_forward`.
