# JEPA-OD in PyOD: port status and hand-off

## Status

The detectors are ported to PyOD as two commits on the branch `feature/jepa-od`, based on
`yzhao062/pyod` branch `development` at commit `9680a8b` (PyOD 3.6.8).

| commit | content |
|---|---|
| `Add JEPA detector for tabular data (experimental)` | `pyod.models.jepa.JEPA`, shared helpers, tests, example, docs |
| `Add TSJEPA detector for time series (experimental)` | `pyod.models.ts_jepa.TSJEPA`, tests, example, docs |

**The branch is on GitHub:** https://github.com/kishordgupta/pyod/tree/feature/jepa-od. It is two
commits ahead of `yzhao062/pyod:development` and zero behind, and touches no workflow file. No issue
or pull request has been opened on PyOD.

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
them, so the push of `feature/jepa-od` did not start CI. A pull request inside the fork, from
`feature/jepa-od` into the fork's own `development`, would run the full matrix (Python 3.9 to 3.13)
without involving upstream.

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
