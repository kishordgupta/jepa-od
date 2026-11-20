# Adding JEPA-OD to PyOD: file map and checklist

PyOD 3.6.x layout observed in the installed package (`pyod/models/`, `pyod/utils/`).

| this repo | PyOD destination | notes |
|---|---|---|
| `jepa_od/_nets.py`, `jepa_od/_sigreg.py` | `pyod/models/_jepa_utils.py` | merge into one private helper module (PyOD convention: `_ts_utils.py`, `_pyg_utils.py`) |
| `jepa_od/jepa.py` (`JEPAOD`) | `pyod/models/jepa.py` (`class JEPA`) | class name follows `AutoEncoder`, `DeepSVDD`, `LUNAR`; imports become `from .base_dl import ...` |
| `jepa_od/ts_jepa.py` (`TSJEPAOD`) | `pyod/models/ts_jepa.py` (`class TSJEPA`) | follows `ts_lstm.py::LSTMAD`, `ts_anomaly_transformer.py::AnomalyTransformer` |
| `tests/test_jepa.py` | `pyod/test/test_jepa.py` | same structure as `pyod/test/test_auto_encoder.py` |
| `tests/test_ts_jepa.py` | `pyod/test/test_ts_jepa.py` | same structure as `pyod/test/test_ts_lstm.py` |
| `integration/pyod_knowledge_entries.json` | append to `pyod/utils/knowledge/algorithms.json` | used by PyOD 3's LLM model selector (`auto_model_selector.py`) |
| -- | `pyod/models/embedding.py::_DETECTOR_SHORTCUTS` | add `'JEPA': ('pyod.models.jepa', 'JEPA', {})` so `TimeSeriesOD(detector='JEPA')` and `EmbeddingOD(detector='JEPA')` resolve by name |
| -- | `docs/pyod.models.rst`, `docs/api_cc.rst`, `README.rst` model table, `docs/zreferences.bib` | documentation entries; cite LeCun 2022, Assran et al. 2023, Balestriero & LeCun 2025 |
| -- | `examples/jepa_example.py`, `examples/ts_jepa_example.py` | PyOD ships one example per detector |

Process (from `CONTRIBUTING.md` on master, fetched 2026-11-20):
1. Open an issue first; the maintainer accepts/assigns within 14 days (label `accepted`), max 5 active outside contributions.
2. Branch from `development`, one PR against `development`, link the issue (`Fixes #...`).
3. Inclusion criteria (FAQ): "at least two years since publication, 50+ citations, practical usefulness",
   **or** the authors of a newly proposed detector commit to two years of maintenance. JEPA-OD is a new
   detector, so the PR must state that maintenance commitment, and it helps to attach the benchmark CSVs.
4. The PR needs tests, docs, an example, and a statement of the test command and results.

Design constraints respected by the prototype:
* `__init__` only stores parameters (sklearn `clone` / `get_params` work; verified in tests).
* `fit` -> `decision_scores_`, `threshold_`, `labels_` via `_process_decision_scores()`.
* Deterministic `decision_function` (seeded mask set), picklable (`BaseDeepLearningDetector.save/load`).
* Torch is imported lazily only inside PyOD's `try/except` convention in `base_dl.py`; the TS detector
  should wrap `import torch` the same way `ts_anomaly_transformer.py` does before merging.
* Device handling mirrors PyOD: `None/'auto'` -> CUDA else CPU, explicit `'mps'` allowed.

DeepOD alternative (`deepod.core.base_model.BaseDeepAD`): implement `training_prepare(X, y)` -> (loader, net, criterion),
`training_forward(batch_x, net, criterion)`, `inference_prepare(X)` -> loader, `inference_forward(batch_x, net, criterion)`
-> (z, score); set `data_type='ts'` and use DeepOD's `seq_len`/`stride` windowing. `JEPACore` plugs in as `net`
unchanged; the SIGReg term is computed inside `training_forward`.
