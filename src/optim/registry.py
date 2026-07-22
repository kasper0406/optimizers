"""Optimizer registry (WP0.4): configs select optimizers by name.

``scripts/run.py`` (and any experiment adapter) resolves
``config["optimizer"]["name"]`` through :data:`OPTIMIZER_REGISTRY`.
WP2.1 registers ``routed`` here once implemented.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Type

from src.optim.adamuon import AdaMuon
from src.optim.adamw import AdamW
from src.optim.dynmuon import DynMuon
from src.optim.interface import MatrixOptimizer, NoOpOptimizer
from src.optim.muon import Muon
from src.optim.normuon import NorMuon
from src.optim.routed import RoutedMuon
from src.optim.tempomuon import TempoMuon

OPTIMIZER_REGISTRY: Dict[str, Type[MatrixOptimizer]] = {
    "noop": NoOpOptimizer,
    "muon": Muon,
    "adamw": AdamW,
    "dynmuon": DynMuon,
    "adamuon": AdaMuon,
    "normuon": NorMuon,
    "routed": RoutedMuon,  # WP2.1
    "tempomuon": TempoMuon,  # program #8
}


def build_optimizer(
    name: str, params: Iterable, kwargs: Dict[str, Any]
) -> MatrixOptimizer:
    """Instantiate a registered optimizer by name (config-driven path)."""
    try:
        cls = OPTIMIZER_REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"Unknown optimizer {name!r}; known: {sorted(OPTIMIZER_REGISTRY)}"
        ) from None
    return cls(params, **kwargs)
