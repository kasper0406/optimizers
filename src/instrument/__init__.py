"""Instrumentation: tracked-pair machinery, logging, and plotting (WP1.1).

Public API:
    TrackedSubspace, RefreshResult              (subspace.py)
    DirectionTrack, MatrixTracker,
    InstrumentationHub, hub_from_config         (tracker.py)
    validate_instrumentation, write_sidecar,
    load_instrumentation, iter_directions,
    sidecar_path                                (schema.py)
    make_all_plots                              (plots.py)

All statistics (EMAs, autocorrelation, classification) are computed by the
WP0.5-validated ``src.stats`` module; nothing statistical is reimplemented
here.  HVP curvature probes are Phase-1 validation only and are forbidden in
any routing/update path (distributed invariant 3).

``plots`` is imported lazily (``from src.instrument import plots``) so that
importing the hub does not pull in matplotlib on the training path.
"""

from src.instrument.schema import (
    INSTRUMENTATION_SCHEMA_VERSION,
    InstrumentationValidationError,
    iter_directions,
    load_instrumentation,
    sidecar_path,
    validate_instrumentation,
    write_sidecar,
)
from src.instrument.subspace import RefreshResult, TrackedSubspace
from src.instrument.tracker import (
    DirectionTrack,
    InstrumentationHub,
    MatrixTracker,
    hub_from_config,
)

__all__ = [
    "INSTRUMENTATION_SCHEMA_VERSION",
    "InstrumentationValidationError",
    "TrackedSubspace",
    "RefreshResult",
    "DirectionTrack",
    "MatrixTracker",
    "InstrumentationHub",
    "hub_from_config",
    "validate_instrumentation",
    "write_sidecar",
    "load_instrumentation",
    "iter_directions",
    "sidecar_path",
]
