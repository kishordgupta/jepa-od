# -*- coding: utf-8 -*-
"""JEPA-OD: Joint-Embedding Predictive Architectures for outlier detection.

* :class:`jepa_od.JEPAOD`   -- tabular detector (PyOD BaseDeepLearningDetector)
* :class:`jepa_od.TJEPAOD`  -- tabular detector trained with the T-JEPA recipe
* :class:`jepa_od.TSJEPAOD` -- time-series detector (PyOD time-series API);
  ``variant`` selects the JEPA version (LeJEPA, I-JEPA, TS-JEPA, T-JEPA and twelve more)
"""
from .jepa import JEPAOD
from .tjepa import TJEPAOD
from .ts_jepa import TSJEPAOD

__all__ = ['JEPAOD', 'TJEPAOD', 'TSJEPAOD']
__version__ = '0.4.0'
