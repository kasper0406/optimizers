#!/usr/bin/env python
"""WP2.2 helper: materialize stage-B placeholder configs from stage-A output.

Usage:
    uv run python scripts/plan_wp22.py fill-tuneB \
        --csv <stageA_aggregate.csv> --manifest <stageA_manifest.json> \
        --target configs/wp22_tuneB_muon.yaml [--fix optimizer.lr=0.24] \
        [--dry-run]

Inputs:
- ``--csv``: scripts/aggregate.py --out-csv output for the stage-A dev sweep
  (one row per materialized variant config; ``metric_mean``/``metric_ci95``).
- ``--manifest``: the stage-A sweep out-dir ``manifest.json`` (scripts/sweep.py),
  which maps each variant config to its grid overrides.
- ``--target``: a placeholder config carrying ``TBD-STAGE-A`` values
  (wp22_tuneB_*.yaml, wp22_null_muon_wd.yaml).
- ``--fix key=value`` (repeatable): restrict candidate variants to those whose
  override for ``key`` equals ``value`` (e.g. hold lr at the record for the
  wp22_null_muon_wd fill), OR whose target-config value already equals it.

Behavior: picks the argmax ``metric_mean`` variant among candidates, rewrites
the target's ``TBD-STAGE-A`` scalars TEXTUALLY (comments preserved), drops the
``status:`` placeholder marker, and appends a provenance comment. Ties within
the winner's 95% CI are reported for human adjudication (the argmax is still
written; the human eyeballs before launch per docs/wp22-run-plan.md).

This tool only ever edits placeholder configs that explicitly carry
``TBD-STAGE-A`` markers; it refuses anything else. It never touches
criteria/ or results/.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent

TBD = "TBD-STAGE-A"
STATUS_RE = re.compile(r"^status:\s*.*$", re.MULTILINE)


class PlanError(ValueError):
    pass


def _parse_fix(items: List[str]) -> Dict[str, float]:
    fixes: Dict[str, float] = {}
    for item in items:
        if "=" not in item:
            raise PlanError(f"--fix expects key=value, got {item!r}")
        key, _, raw = item.partition("=")
        try:
            fixes[key.strip()] = float(raw)
        except ValueError as exc:
            raise PlanError(f"--fix value must be numeric, got {raw!r}") from exc
    return fixes


def _load_rows(csv_path: Path) -> List[Dict[str, str]]:
    with open(csv_path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise PlanError(f"{csv_path}: no rows")
    for col in ("config", "metric_mean", "metric_ci95"):
        if col not in rows[0]:
            raise PlanError(f"{csv_path}: missing column {col!r} (aggregate.py --out-csv?)")
    return rows


def _match_variant(row_config: str, variants: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    stem = Path(row_config).name
    for variant in variants:
        if Path(variant["config_path"]).name == stem or f"{variant['name']}.yaml" == stem:
            return variant
    return None


def fill_tuneb(args: argparse.Namespace) -> int:
    fixes = _parse_fix(args.fix or [])
    manifest = json.loads(Path(args.manifest).read_text())
    variants = manifest.get("variants", [])
    if not variants:
        raise PlanError(f"{args.manifest}: no variants")

    target = Path(args.target)
    text = target.read_text()
    if TBD not in text:
        raise PlanError(f"{target}: no {TBD} markers — refusing to edit a non-placeholder config")
    target_cfg = yaml.safe_load(text)

    # Which dotted keys need filling? (scalar leaves equal to the TBD string)
    tbd_keys: List[str] = []

    def walk(node: Any, prefix: str) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                walk(value, f"{prefix}.{key}" if prefix else str(key))
        elif node == TBD:
            tbd_keys.append(prefix)

    walk(target_cfg, "")
    if not tbd_keys:
        raise PlanError(f"{target}: found no {TBD} scalars to fill")

    # Candidate rows: joinable to a variant, satisfying --fix constraints.
    rows = _load_rows(Path(args.csv))
    candidates = []
    for row in rows:
        variant = _match_variant(row["config"], variants)
        if variant is None:
            continue
        overrides = variant["overrides"]
        ok = True
        for key, want in fixes.items():
            have = overrides.get(key, _dotted_get(target_cfg, key))
            if have is None or abs(float(have) - want) > 1e-12:
                ok = False
                break
        if ok and row["metric_mean"] not in ("", None):
            candidates.append((float(row["metric_mean"]), float(row["metric_ci95"] or 0.0), row, variant))
    if not candidates:
        raise PlanError("no stage-A rows matched the manifest variants and --fix constraints")

    candidates.sort(key=lambda c: c[0], reverse=True)
    best_mean, best_ci, best_row, best_variant = candidates[0]
    ties = [c for c in candidates[1:] if best_mean - c[0] <= best_ci]

    print(f"stage-A cells considered: {len(candidates)}")
    print(f"argmax: {best_variant['name']}  mean={best_mean:.5f}  ci95={best_ci:.5f}")
    print(f"        overrides: {best_variant['overrides']}")
    if ties:
        print("WARNING: tie(s) within the winner's 95% CI — human adjudication required:")
        for mean, _, _, variant in ties:
            print(f"  {variant['name']}  mean={mean:.5f}  overrides: {variant['overrides']}")

    # Fill each TBD key from the winning overrides (last path component match).
    new_text = text
    filled: Dict[str, Any] = {}
    for dotted in tbd_keys:
        leaf = dotted.split(".")[-1]
        value = None
        for okey, oval in best_variant["overrides"].items():
            if okey.split(".")[-1] == leaf or okey == dotted:
                value = oval
        if value is None:
            raise PlanError(
                f"{target}: cannot fill {dotted!r} — no matching override in the "
                f"winning variant {best_variant['overrides']!r}"
            )
        pattern = re.compile(rf"^(\s*{re.escape(leaf)}:\s*){re.escape(TBD)}", re.MULTILINE)
        new_text, n_subs = pattern.subn(rf"\g<1>{value}", new_text, count=1)
        if n_subs != 1:
            raise PlanError(f"{target}: could not textually locate '{leaf}: {TBD}'")
        filled[dotted] = value

    new_text = STATUS_RE.sub("", new_text, count=1).replace("\n\n\n", "\n\n")
    provenance = (
        f"# FILLED by scripts/plan_wp22.py from {Path(args.csv).name} "
        f"(argmax {best_variant['name']}, mean {best_mean:.5f} +- ci95 {best_ci:.5f}"
        f"{'; TIES — see launch log' if ties else ''}): "
        + ", ".join(f"{k}={v}" for k, v in filled.items())
        + "\n"
    )
    new_text = provenance + new_text

    if args.dry_run:
        print("--dry-run: not writing. Would fill:", filled)
        return 0
    target.write_text(new_text)
    print(f"filled {filled} -> {target} (status marker removed; human confirms before launch)")
    return 0


def _dotted_get(config: Dict[str, Any], dotted: str) -> Any:
    node: Any = config
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    fill = sub.add_parser("fill-tuneB", help="fill a TBD-STAGE-A placeholder from stage-A output")
    fill.add_argument("--csv", required=True, type=Path)
    fill.add_argument("--manifest", required=True, type=Path)
    fill.add_argument("--target", required=True, type=Path)
    fill.add_argument("--fix", action="append", default=[], metavar="KEY=VALUE")
    fill.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "fill-tuneB":
            return fill_tuneb(args)
        raise PlanError(f"unknown command {args.command!r}")
    except PlanError as exc:
        print(f"plan_wp22.py: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
