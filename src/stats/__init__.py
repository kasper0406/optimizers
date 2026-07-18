"""Tracked-pair statistics: EMAs, autocorrelation, regime classifier.

Implemented and validated in WP0.5 (synthetic-signal suite) before any
instrumented run.  WP1.1 must use this module, never a reimplementation.

Public API:
    BiasCorrectedEma, ema_effective_sample_size   (ema.py)
    DirectionStats                                (direction_stats.py)
    Regime, RegimeClassifier                      (classifier.py)
    ArrayDirectionStats, BatchRegimeClassifier    (batch.py; vectorized k-at-
                                                   once equivalents, equivalence-
                                                   tested against the scalar path)
    ar1, drifting_mean, oscillation,
    gaussian_noise, concat_segments               (generators.py)
"""

from .batch import ArrayDirectionStats, BatchRegimeClassifier
from .classifier import Regime, RegimeClassifier
from .direction_stats import DirectionStats
from .ema import BiasCorrectedEma, ema_effective_sample_size
from .generators import ar1, concat_segments, drifting_mean, gaussian_noise, oscillation

__all__ = [
    "BiasCorrectedEma",
    "ema_effective_sample_size",
    "DirectionStats",
    "Regime",
    "RegimeClassifier",
    "ArrayDirectionStats",
    "BatchRegimeClassifier",
    "ar1",
    "drifting_mean",
    "oscillation",
    "gaussian_noise",
    "concat_segments",
]
