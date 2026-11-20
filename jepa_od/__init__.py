# -*- coding: utf-8 -*-
"""JEPA-OD: Joint-Embedding Predictive Architectures for outlier detection.

* :class:`jepa_od.JEPAOD`   -- tabular detector (PyOD BaseDeepLearningDetector)
* :class:`jepa_od.TSJEPAOD` -- time-series detector (PyOD time-series API)
"""
from .jepa import JEPAOD
from .ts_jepa import TSJEPAOD

__all__ = ['JEPAOD', 'TSJEPAOD']
__version__ = '0.1.0'
