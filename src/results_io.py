"""Shared results-JSON schema, writer, and validator (WP0.0).

Every run — local or cloud — produces exactly one JSON file in ``results/``
conforming to this schema. All tooling (runner, sweep aggregation, plotting)
must go through this module rather than reinventing the schema.

Required top-level keys (see ``REQUIRED_KEYS``):

- ``schema_version``: int, currently 1.
- ``experiment``: str, experiment name from the config.
- ``config``: dict with ``path`` (str, repo-relative), ``sha256`` (str, hash of
  the config file bytes), and ``contents`` (dict, full parsed copy).
- ``git_sha``: str, 40-hex commit SHA of the code that ran (or "unknown" only
  if provenance is genuinely unavailable — validation warns loudly).
- ``git_dirty``: bool, whether the working tree had uncommitted changes.
- ``seed``: int, the seed this run used.
- ``gpu_type``: str, torch device string / CUDA device name
  (e.g. "cpu", "mps", "NVIDIA A100-SXM4-80GB").
- ``wall_time_s``: float, wall time of the run in seconds.
- ``cost_usd``: float or None. None for local runs; the human fills in the
  cloud cost after a Hyperstack run.
- ``started_at`` / ``finished_at``: ISO-8601 UTC timestamps.
- ``metrics``: dict of run metrics (experiment-defined).

Results files are append-only: never modify an existing file in ``results/``.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

SCHEMA_VERSION = 1

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"

REQUIRED_KEYS = {
    "schema_version": int,
    "experiment": str,
    "config": dict,
    "git_sha": str,
    "git_dirty": bool,
    "seed": int,
    "gpu_type": str,
    "wall_time_s": (int, float),
    "cost_usd": (int, float, type(None)),
    "started_at": str,
    "finished_at": str,
    "metrics": dict,
}

REQUIRED_CONFIG_KEYS = {
    "path": str,
    "sha256": str,
    "contents": dict,
}


class ResultsValidationError(ValueError):
    """Raised when a results object does not conform to the schema."""


def utc_now_iso() -> str:
    """Current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def git_provenance(repo_root: Optional[Path] = None) -> Dict[str, Any]:
    """Return {'git_sha': str, 'git_dirty': bool} for the given repo.

    Falls back to the RM_GIT_SHA environment variable (set at Docker build
    time) and then to "unknown" when no .git directory is available.
    """
    import os

    root = Path(repo_root) if repo_root is not None else REPO_ROOT
    try:
        sha = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty_out = subprocess.check_output(
            ["git", "-C", str(root), "status", "--porcelain"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return {"git_sha": sha, "git_dirty": bool(dirty_out.strip())}
    except (subprocess.CalledProcessError, FileNotFoundError):
        env_sha = os.environ.get("RM_GIT_SHA")
        if env_sha:
            return {"git_sha": env_sha, "git_dirty": False}
        return {"git_sha": "unknown", "git_dirty": True}


def config_record(config_path: Path, parsed_contents: Dict[str, Any]) -> Dict[str, Any]:
    """Build the ``config`` sub-record: repo-relative path, sha256, full copy."""
    config_path = Path(config_path).resolve()
    try:
        rel = str(config_path.relative_to(REPO_ROOT))
    except ValueError:
        rel = str(config_path)
    digest = hashlib.sha256(config_path.read_bytes()).hexdigest()
    return {"path": rel, "sha256": digest, "contents": parsed_contents}


def validate(result: Dict[str, Any]) -> Dict[str, Any]:
    """Validate a results object against the schema.

    Returns the object unchanged on success; raises
    :class:`ResultsValidationError` with all problems listed otherwise.
    """
    problems: List[str] = []

    if not isinstance(result, dict):
        raise ResultsValidationError(f"results object must be a dict, got {type(result)}")

    for key, expected_type in REQUIRED_KEYS.items():
        if key not in result:
            problems.append(f"missing required key: {key!r}")
        elif not isinstance(result[key], expected_type):
            problems.append(
                f"key {key!r} has type {type(result[key]).__name__}, "
                f"expected {expected_type}"
            )
        elif expected_type is int and isinstance(result[key], bool):
            problems.append(f"key {key!r} is bool, expected int")

    if isinstance(result.get("config"), dict):
        for key, expected_type in REQUIRED_CONFIG_KEYS.items():
            if key not in result["config"]:
                problems.append(f"missing required config key: config.{key!r}")
            elif not isinstance(result["config"][key], expected_type):
                problems.append(
                    f"config.{key!r} has type "
                    f"{type(result['config'][key]).__name__}, expected {expected_type}"
                )

    if result.get("schema_version") not in (None, SCHEMA_VERSION):
        problems.append(
            f"schema_version {result['schema_version']!r} != {SCHEMA_VERSION}"
        )

    for ts_key in ("started_at", "finished_at"):
        value = result.get(ts_key)
        if isinstance(value, str):
            try:
                datetime.fromisoformat(value)
            except ValueError:
                problems.append(f"{ts_key!r} is not ISO-8601: {value!r}")

    wall = result.get("wall_time_s")
    if isinstance(wall, (int, float)) and not isinstance(wall, bool) and wall < 0:
        problems.append(f"wall_time_s must be >= 0, got {wall}")

    if problems:
        raise ResultsValidationError(
            "invalid results object:\n  - " + "\n  - ".join(problems)
        )
    return result


def write_result(result: Dict[str, Any], out_path: Path) -> Path:
    """Validate and write a results JSON. Refuses to overwrite (append-only)."""
    validate(result)
    out_path = Path(out_path)
    if out_path.exists():
        raise FileExistsError(
            f"{out_path} already exists; results/ is append-only — never overwrite"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp, "w") as fh:
        json.dump(result, fh, indent=2, sort_keys=True)
        fh.write("\n")
    tmp.rename(out_path)
    return out_path


def load_result(path: Path) -> Dict[str, Any]:
    """Load and validate a results JSON."""
    with open(path) as fh:
        return validate(json.load(fh))
