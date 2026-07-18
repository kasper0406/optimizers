"""Instrumentation log schema, sidecar writer, and reader (WP1.1).

The per-direction time series produced by :class:`InstrumentationHub.to_log`
are written next to the run's results JSON as a **sidecar** file

    <results_stem>.instrumentation.json

following the ``src.results_io`` conventions: append-only (never overwrite),
atomic write via a temp file, sorted keys.  The main results JSON stays small;
its ``metrics`` dict carries a pointer ``{"instrumentation_sidecar": <name>}``
so aggregation tooling can find the sidecar from the results file alone.

Schema (version 1)::

    {
      "instrumentation_schema_version": 1,
      "betas": ["0.9", "0.99"],
      "hvp_enabled": bool,
      "matrices": {
        <name>: {
          "shape": [m, n], "k1": int, "k2": int, "t_refresh": int,
          "align_min": float, "snapshot_every": int,
          "steps": [int], "grad_fro_norm": [float], "top_sigma_m": [float],
          "refresh_steps": [int],
          "directions": [
            {
              "index": int, "kind": "top"|"bulk",
              "s": [float],                      # every step
              "reset_steps": [int],              # innovation resets
              "refresh_alignment": {"step": [int], "value": [float]},
              "sigma": {"step": [int], "value": [float]},
              "lambda_hvp": {"step": [int], "value": [float]},   # per refresh
              "per_beta": {                      # snapshot cadence
                "0.9": {"step": [...], "regime": [...], "mu": [...],
                         "var": [...], "rho": [...], "t_stat": [...],
                         "amplitude_ratio": [...],
                         "implied_eta_lambda": [...], "ess": [...],
                         "n_since_reset": [...]},
                "0.99": {...}
              }
            }, ...
          ]
        }, ...
      }
    }

The reader (:func:`load_instrumentation`, :func:`iter_directions`) is what
``src.instrument.plots`` consumes -- plots run from JSON alone, no live
training required.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, Tuple

INSTRUMENTATION_SCHEMA_VERSION = 1

SIDECAR_SUFFIX = ".instrumentation.json"

_REQUIRED_TOP = ("instrumentation_schema_version", "betas", "hvp_enabled", "matrices")
_REQUIRED_MATRIX = (
    "shape",
    "k1",
    "k2",
    "t_refresh",
    "steps",
    "grad_fro_norm",
    "top_sigma_m",
    "refresh_steps",
    "directions",
)
_REQUIRED_DIRECTION = (
    "index",
    "kind",
    "s",
    "reset_steps",
    "refresh_alignment",
    "sigma",
    "lambda_hvp",
    "per_beta",
)
_REQUIRED_BETA_SERIES = (
    "step",
    "regime",
    "mu",
    "var",
    "rho",
    "t_stat",
    "amplitude_ratio",
    "implied_eta_lambda",
    "ess",
    "n_since_reset",
)


class InstrumentationValidationError(ValueError):
    """Raised when an instrumentation log does not conform to the schema."""


def validate_instrumentation(log: Dict[str, Any]) -> Dict[str, Any]:
    """Validate an instrumentation log dict; return it unchanged or raise."""
    problems = []
    if not isinstance(log, dict):
        raise InstrumentationValidationError(
            f"log must be a dict, got {type(log).__name__}"
        )
    for key in _REQUIRED_TOP:
        if key not in log:
            problems.append(f"missing top-level key {key!r}")
    version = log.get("instrumentation_schema_version")
    if version is not None and version != INSTRUMENTATION_SCHEMA_VERSION:
        problems.append(
            f"schema version {version!r} != {INSTRUMENTATION_SCHEMA_VERSION}"
        )
    for name, mat in (log.get("matrices") or {}).items():
        for key in _REQUIRED_MATRIX:
            if key not in mat:
                problems.append(f"matrix {name!r}: missing key {key!r}")
        for d in mat.get("directions", []):
            for key in _REQUIRED_DIRECTION:
                if key not in d:
                    problems.append(
                        f"matrix {name!r} direction {d.get('index')}: "
                        f"missing key {key!r}"
                    )
            for beta_key, series in (d.get("per_beta") or {}).items():
                for key in _REQUIRED_BETA_SERIES:
                    if key not in series:
                        problems.append(
                            f"matrix {name!r} direction {d.get('index')} "
                            f"beta {beta_key}: missing series {key!r}"
                        )
            n_steps = len(mat.get("steps", []))
            if len(d.get("s", [])) != n_steps:
                problems.append(
                    f"matrix {name!r} direction {d.get('index')}: "
                    f"len(s)={len(d.get('s', []))} != len(steps)={n_steps}"
                )
    if problems:
        raise InstrumentationValidationError(
            "invalid instrumentation log:\n  - " + "\n  - ".join(problems)
        )
    return log


def sidecar_path(results_json_path: Path) -> Path:
    """Sidecar path for a results JSON: <stem>.instrumentation.json."""
    p = Path(results_json_path)
    return p.with_name(p.name[: -len(".json")] + SIDECAR_SUFFIX if p.name.endswith(".json") else p.name + SIDECAR_SUFFIX)


def write_sidecar(log: Dict[str, Any], results_json_path: Path) -> Path:
    """Validate and write the instrumentation sidecar next to a results JSON.

    Append-only, like everything under results/: refuses to overwrite.
    Returns the sidecar path; callers should record
    ``metrics["instrumentation_sidecar"] = path.name`` in the results JSON.
    """
    validate_instrumentation(log)
    out = sidecar_path(results_json_path)
    if out.exists():
        raise FileExistsError(
            f"{out} already exists; instrumentation sidecars are append-only"
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    with open(tmp, "w") as fh:
        json.dump(log, fh, indent=2, sort_keys=True)
        fh.write("\n")
    tmp.rename(out)
    return out


def load_instrumentation(path: Path) -> Dict[str, Any]:
    """Load and validate an instrumentation log JSON (sidecar or embedded).

    Accepts either a sidecar file (the log at top level) or a full results
    JSON whose ``metrics`` embeds the log under ``"instrumentation"``.
    """
    with open(path) as fh:
        obj = json.load(fh)
    if "instrumentation_schema_version" not in obj:
        embedded = obj.get("metrics", {}).get("instrumentation")
        if embedded is None:
            raise InstrumentationValidationError(
                f"{path}: neither an instrumentation log nor a results JSON "
                "with metrics.instrumentation"
            )
        obj = embedded
    return validate_instrumentation(obj)


def iter_directions(
    log: Dict[str, Any]
) -> Iterator[Tuple[str, Dict[str, Any], Dict[str, Any]]]:
    """Yield (matrix_name, matrix_record, direction_record) over the log."""
    for name in sorted(log.get("matrices", {})):
        mat = log["matrices"][name]
        for d in mat.get("directions", []):
            yield name, mat, d
