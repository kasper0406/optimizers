#!/usr/bin/env python
"""Sweep expander: (seed policy x hyperparameter grid) -> per-run configs + run.py invocations.

Usage:
    uv run python scripts/sweep.py <config.yaml> [--dry-run] [--out-dir DIR] [--execute]

The input is an experiment YAML (same shape scripts/run.py consumes) with an
extra ``sweep:`` block:

    sweep:
      seeds: eval                  # seed policy, resolved HERE at launch time
      # or:  seeds: dev
      # or:  seeds: {policy: dev, num_seeds: 5}
      # or:  seeds: [1000, 1001, 1002]   # explicit list, dev seeds only
      grid:                        # optional; dotted keys into the config
        optimizer.lr: [0.05, 0.1]
        optimizer.weight_decay: [0.0, 0.01]

Seed policies (CLAUDE.md ground rule 2):

- ``eval``: the evaluation set, seeds 0-99, exactly 100 seeds. Used only for
  comparison tables. Eval seeds are resolved here at launch time and passed to
  scripts/run.py via ``--seed``; they are NEVER written into any config file.
- ``dev``: development seeds, the documented range starting at
  ``DEV_SEED_BASE = 1000`` (seeds 1000, 1001, ... 1000+n-1; default n = 10).
  Used for debugging, smoke runs, and hyperparameter exploration.

Enforcement: this tool refuses any input config containing a literal eval seed
(an integer 0-99 under any seed-like key), and asserts that every materialized
per-run config it writes contains no seed key at all — seeds only ever travel
on the run.py command line.

Outputs (under --out-dir, default sweeps/<config-stem>/, never under configs/):

- one materialized YAML per grid variant (sweep block stripped, overrides applied)
- ``manifest.json``: the full expansion (variants, seeds, commands)
- ``run_all.sh``: one ``uv run python scripts/run.py <cfg> --seed N`` per run

``--dry-run`` prints the plan and writes nothing. ``--execute`` runs every
command sequentially after writing (local sweeps; cloud runs execute
run_all.sh inside the container instead).
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent

# ----------------------------------------------------------- seed policies

EVAL_SEED_MIN = 0
EVAL_SEED_MAX = 99
EVAL_SEED_COUNT = 100  # the eval set is exactly seeds 0-99, never a subset

DEV_SEED_BASE = 1000  # documented dev range: DEV_SEED_BASE + i
DEV_SEED_DEFAULT_COUNT = 10

# Keys that contain "seed" but denote a *count* of seeds, not a seed value.
SEED_COUNT_KEY_EXEMPTIONS = {"num_seeds", "n_seeds", "seed_count", "seeds_per_config"}


class SweepConfigError(ValueError):
    """Raised for invalid sweep configs (including eval-seed literals)."""


def find_eval_seed_literals(obj: Any, path: str = "", under_seed_key: bool = False) -> List[str]:
    """Return a list of '<path> = <value>' strings for every literal integer in
    [0, 99] found underneath a seed-like key anywhere in a parsed config."""
    hits: List[str] = []
    if isinstance(obj, bool):
        return hits
    if isinstance(obj, int):
        if under_seed_key and EVAL_SEED_MIN <= obj <= EVAL_SEED_MAX:
            hits.append(f"{path} = {obj}")
        return hits
    if isinstance(obj, dict):
        for key, value in obj.items():
            key_l = str(key).lower()
            child_path = f"{path}.{key}" if path else str(key)
            if key_l in SEED_COUNT_KEY_EXEMPTIONS:
                # e.g. num_seeds: 5 is a count, not a seed value — and it must
                # not inherit seed-context from a parent 'seeds:' mapping.
                child_under = False
            else:
                child_under = under_seed_key or "seed" in key_l
            hits.extend(find_eval_seed_literals(value, child_path, child_under))
    elif isinstance(obj, (list, tuple)):
        for i, value in enumerate(obj):
            hits.extend(find_eval_seed_literals(value, f"{path}[{i}]", under_seed_key))
    return hits


def refuse_eval_seed_literals(config: Dict[str, Any], source: str) -> None:
    hits = find_eval_seed_literals(config)
    if hits:
        raise SweepConfigError(
            f"{source} contains literal eval seed(s) — seeds 0-99 must never "
            "appear in config files; use a seed policy ('eval'/'dev') instead:\n  - "
            + "\n  - ".join(hits)
        )


def resolve_seed_policy(spec: Any) -> Tuple[str, List[int]]:
    """Resolve a ``sweep.seeds`` spec to (policy_name, concrete seed list)."""
    if isinstance(spec, str):
        spec = {"policy": spec}
    if isinstance(spec, (list, tuple)):
        seeds = list(spec)
        bad = [s for s in seeds if not isinstance(s, int) or isinstance(s, bool) or s < DEV_SEED_BASE]
        if bad:
            raise SweepConfigError(
                f"explicit seed lists may contain dev seeds (>= {DEV_SEED_BASE}) only; "
                f"offending entries: {bad}. Use the 'eval' policy for eval seeds."
            )
        if not seeds:
            raise SweepConfigError("explicit seed list is empty")
        return "explicit-dev", seeds
    if not isinstance(spec, dict) or "policy" not in spec:
        raise SweepConfigError(
            f"sweep.seeds must be 'eval', 'dev', a policy mapping, or a list of "
            f"dev seeds; got {spec!r}"
        )
    policy = spec["policy"]
    num = spec.get("num_seeds")
    if policy == "eval":
        if num is not None and num != EVAL_SEED_COUNT:
            raise SweepConfigError(
                f"the eval set is exactly seeds {EVAL_SEED_MIN}-{EVAL_SEED_MAX} "
                f"({EVAL_SEED_COUNT} seeds); num_seeds={num!r} is not allowed "
                "(no partial eval sweeps — use policy 'dev' for development)."
            )
        return "eval", list(range(EVAL_SEED_MIN, EVAL_SEED_MAX + 1))
    if policy == "dev":
        count = DEV_SEED_DEFAULT_COUNT if num is None else num
        if not isinstance(count, int) or isinstance(count, bool) or count < 1:
            raise SweepConfigError(f"dev num_seeds must be a positive int, got {num!r}")
        return "dev", list(range(DEV_SEED_BASE, DEV_SEED_BASE + count))
    raise SweepConfigError(f"unknown seed policy {policy!r}; known: 'eval', 'dev'")


# ------------------------------------------------------------ grid expansion


def set_dotted(config: Dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    node = config
    for part in parts[:-1]:
        node = node.setdefault(part, {})
        if not isinstance(node, dict):
            raise SweepConfigError(
                f"grid key {dotted_key!r}: {part!r} is not a mapping in the base config"
            )
    node[parts[-1]] = value


def _value_slug(value: Any) -> str:
    text = f"{value}"
    return re.sub(r"[^A-Za-z0-9.+-]+", "-", text)


def expand_grid(grid: Dict[str, Sequence[Any]]) -> List[Dict[str, Any]]:
    """Cartesian product of a {dotted_key: [values]} grid -> list of overrides."""
    if not grid:
        return [{}]
    keys = list(grid)
    for key in keys:
        values = grid[key]
        if not isinstance(values, (list, tuple)) or len(values) == 0:
            raise SweepConfigError(f"grid entry {key!r} must be a non-empty list")
    return [
        dict(zip(keys, combo))
        for combo in itertools.product(*(grid[k] for k in keys))
    ]


def variant_name(stem: str, overrides: Dict[str, Any]) -> str:
    if not overrides:
        return stem
    parts = [f"{k.split('.')[-1]}{_value_slug(v)}" for k, v in sorted(overrides.items())]
    return f"{stem}__" + "_".join(parts)


# ---------------------------------------------------------------- expansion


def expand_sweep(config: Dict[str, Any], config_path: Path) -> Dict[str, Any]:
    """Expand a sweep config into a plan dict (no files written).

    Plan shape:
      {"name", "source_config", "seed_policy", "seeds",
       "variants": [{"name", "overrides", "config"}],
       "runs": [{"variant", "seed"}]}
    """
    if not isinstance(config, dict):
        raise SweepConfigError(f"{config_path} did not parse to a mapping")
    refuse_eval_seed_literals(config, str(config_path))
    sweep_block = config.get("sweep")
    if not isinstance(sweep_block, dict):
        raise SweepConfigError(f"{config_path} has no 'sweep:' block")

    policy_name, seeds = resolve_seed_policy(sweep_block.get("seeds"))
    overrides_list = expand_grid(sweep_block.get("grid") or {})

    base = copy.deepcopy(config)
    base.pop("sweep", None)
    # Seeds travel exclusively via `run.py --seed`; materialized configs carry
    # no seed key at all (a base dev seed, if present, is dropped).
    base.pop("seed", None)

    stem = Path(config_path).stem
    variants = []
    for overrides in overrides_list:
        cfg = copy.deepcopy(base)
        for key, value in overrides.items():
            set_dotted(cfg, key, value)
        # Defense in depth: a grid must not smuggle eval seeds in either.
        refuse_eval_seed_literals(cfg, f"materialized variant {overrides!r}")
        if "seed" in cfg:
            raise SweepConfigError("materialized config must not contain a 'seed' key")
        variants.append(
            {"name": variant_name(stem, overrides), "overrides": overrides, "config": cfg}
        )

    names = [v["name"] for v in variants]
    if len(set(names)) != len(names):
        raise SweepConfigError(f"variant name collision in grid expansion: {names}")

    runs = [
        {"variant": v["name"], "seed": s} for v in variants for s in seeds
    ]
    try:
        source = str(Path(config_path).resolve().relative_to(REPO_ROOT))
    except ValueError:
        source = str(config_path)
    return {
        "name": stem,
        "source_config": source,
        "seed_policy": policy_name,
        "seeds": seeds,
        "variants": variants,
        "runs": runs,
    }


# ------------------------------------------------------------------ writing


def _repo_relative_or_abs(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def run_command(config_path: Path, seed: int) -> List[str]:
    return [
        "uv", "run", "python", "scripts/run.py",
        _repo_relative_or_abs(config_path),
        "--seed", str(seed),
    ]


def write_plan(plan: Dict[str, Any], out_dir: Path) -> Dict[str, Any]:
    """Write variant configs, manifest.json, and run_all.sh into out_dir.

    Returns the manifest dict. Refuses to write under configs/ (hand-written
    experiment configs only) and refuses to overwrite an existing manifest.
    """
    out_dir = Path(out_dir).resolve()
    configs_dir = (REPO_ROOT / "configs").resolve()
    if out_dir == configs_dir or configs_dir in out_dir.parents:
        raise SweepConfigError(
            f"refusing to write generated sweep output under {configs_dir} — "
            "configs/ holds hand-written experiment configs only"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists():
        raise SweepConfigError(
            f"{manifest_path} already exists; use a fresh --out-dir per expansion"
        )

    variant_paths: Dict[str, Path] = {}
    for variant in plan["variants"]:
        cfg_path = out_dir / f"{variant['name']}.yaml"
        text = yaml.safe_dump(variant["config"], sort_keys=True)
        # Final gate before anything touches disk.
        refuse_eval_seed_literals(yaml.safe_load(text), str(cfg_path))
        cfg_path.write_text(
            "# Generated by scripts/sweep.py from "
            f"{plan['source_config']} — do not edit; seeds are passed via --seed.\n"
            + text
        )
        variant_paths[variant["name"]] = cfg_path

    commands = []
    for run in plan["runs"]:
        cmd = run_command(variant_paths[run["variant"]], run["seed"])
        commands.append({**run, "command": cmd})

    manifest = {
        "sweep": plan["name"],
        "source_config": plan["source_config"],
        "seed_policy": plan["seed_policy"],
        "seeds": plan["seeds"],
        "n_variants": len(plan["variants"]),
        "n_runs": len(commands),
        "variants": [
            {
                "name": v["name"],
                "overrides": v["overrides"],
                "config_path": _repo_relative_or_abs(variant_paths[v["name"]]),
            }
            for v in plan["variants"]
        ],
        "runs": commands,
    }
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2)
        fh.write("\n")

    if configs_dir in out_dir.parents or out_dir == configs_dir:  # pragma: no cover
        raise AssertionError("unreachable")

    try:
        rel_root = os.path.relpath(REPO_ROOT, out_dir)
        cd_line = f'cd "$(cd "$(dirname "$0")" && pwd)/{rel_root}"'
    except ValueError:  # pragma: no cover - different drive on Windows
        cd_line = f'cd "{REPO_ROOT}"'
    script_lines = [
        "#!/usr/bin/env bash",
        f"# Generated by scripts/sweep.py from {plan['source_config']}",
        f"# seed policy: {plan['seed_policy']} ({len(plan['seeds'])} seeds), "
        f"{manifest['n_variants']} variant(s), {manifest['n_runs']} run(s)",
        "set -euo pipefail",
        cd_line,
    ]
    script_lines += [shlex.join(c["command"]) for c in commands]
    run_all = out_dir / "run_all.sh"
    run_all.write_text("\n".join(script_lines) + "\n")
    run_all.chmod(0o755)
    return manifest


# --------------------------------------------------------------------- main


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Expand a sweep config into per-run configs + run.py invocations."
    )
    parser.add_argument("config", type=Path, help="Experiment YAML with a sweep: block")
    parser.add_argument(
        "--out-dir", type=Path, default=None,
        help="Output directory (default: sweeps/<config-stem>/; never under configs/)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the expansion plan without writing anything",
    )
    parser.add_argument(
        "--execute", action="store_true",
        help="After writing, run every command sequentially (local sweeps)",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    with open(args.config) as fh:
        config = yaml.safe_load(fh)

    try:
        plan = expand_sweep(config, args.config)
    except SweepConfigError as exc:
        print(f"sweep.py: {exc}", file=sys.stderr)
        return 2

    n_runs = len(plan["runs"])
    print(
        f"sweep '{plan['name']}': seed policy {plan['seed_policy']} "
        f"({len(plan['seeds'])} seeds) x {len(plan['variants'])} variant(s) "
        f"= {n_runs} run(s)"
    )

    if args.dry_run:
        for variant in plan["variants"]:
            print(f"  variant {variant['name']}: overrides {variant['overrides']}")
        preview = plan["runs"][:3]
        for run in preview:
            print(f"  e.g. run.py <{run['variant']}.yaml> --seed {run['seed']}")
        if n_runs > len(preview):
            print(f"  ... and {n_runs - len(preview)} more")
        print("dry run: nothing written")
        return 0

    out_dir = args.out_dir or (REPO_ROOT / "sweeps" / plan["name"])
    try:
        manifest = write_plan(plan, out_dir)
    except SweepConfigError as exc:
        print(f"sweep.py: {exc}", file=sys.stderr)
        return 2
    print(f"wrote {manifest['n_variants']} config(s), manifest.json, run_all.sh -> {out_dir}")

    if args.execute:
        for i, run in enumerate(manifest["runs"], 1):
            print(f"[{i}/{n_runs}] {shlex.join(run['command'])}")
            subprocess.run(run["command"], cwd=REPO_ROOT, check=True)
    else:
        print(f"execute with: bash {Path(out_dir) / 'run_all.sh'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
