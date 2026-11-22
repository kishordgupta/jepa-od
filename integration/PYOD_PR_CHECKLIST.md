# JEPA-OD in PyOD: port status and hand-off

## Status

**Read first (2026-11-21):** the extended benchmark in `REPORT.md` section 7.6 was run after this port.
On all 47 ADBench datasets the tabular detector is mid-table and an untrained network scores at least as
well, so the report no longer argues for adding `JEPA` on performance grounds. Issue 795 was written
before that result; the comment in `PYOD_ISSUE_COMMENT_DRAFT.md`, posted on the issue on 2026-11-21, tells
the maintainer and offers to narrow the proposal to `TSJEPA`, as a family with a selectable JEPA version
(`variant`, `REPORT.md` section 4.8). Wait for the answer before any pull request.

## The branch for the narrowed proposal: `feature/ts-jepa` (2026-11-21)

> **2026-11-22.** The prototype has since gained twelve more versions (`jepa_od` 0.4.0, `REPORT.md` sections 4.9
> and 7.8). They are **not** in this branch, which still has the four versions that the comment on the issue
> describes. Porting any of them is a separate decision.

Built so that a pull request can be opened as soon as the maintainer accepts the `TSJEPA`-only proposal:
https://github.com/kishordgupta/pyod/tree/feature/ts-jepa

* **One commit** (`455ca3c`, "Add TSJEPA detector for time series (experimental)") on top of
  `yzhao062/pyod:development` at `f785bc9` (PyOD 3.6.9). No tabular `JEPA`, no workflow file touched.
  Upstream has since added one merge commit without file changes (`e9d8670`), so GitHub shows the branch
  as one ahead and one behind. Rebase onto the then-current `development` before opening the pull request.
* **Content.** `pyod/models/ts_jepa.py` (`TSJEPA` with `variant` = `'lejepa'`, `'ijepa'`, `'tsjepa'`,
  `'tjepa'`; `encoder_norm=None` and `ema_momentum=None` resolve per variant; `target_mode` is gone),
  `pyod/models/_jepa_utils.py` (the networks of the four variants, SIGReg, EMA update, mask samplers),
  `pyod/test/test_ts_jepa.py` (28 tests), `examples/ts_jepa_example.py`, the API page, the model-table
  rows in `README.rst` and `docs/index.rst`, two bibliography entries, the two experimental-detector
  lists, and the `conftest.py` skip-list line.
* **Same patch here:** `integration/pyod-patches-ts-jepa/`, applied with
  `git switch -c feature/ts-jepa upstream/development && git am --keep-cr <patch>`.
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
| whole `pyod/test` directory | 1576 passed, 68 skipped, 0 failed. Not run here because optional packages are missing: `test_thresholds.py` (15 errors, `pythresh`) and five modules that need `xgboost`, `combo` or `suod` |
| `examples/ts_jepa_example.py` | runs, ROC 0.954 |
| Sphinx build of the module page | no warnings from the new docstrings |
| `import pyod.models.ts_jepa` with torch blocked | imports without torch; `fit` raises the install hint |
| Python 3.9.6 `ast.parse` of the four Python files | parses |

Not run: PyOD's CI matrix on this branch (it needs a pull request, see below), the full documentation
build, Coveralls, any GPU.

Left out on purpose, as before: `CHANGES.txt`, the knowledge base and its detector counts, and the
`_DETECTOR_SHORTCUTS` line.

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
