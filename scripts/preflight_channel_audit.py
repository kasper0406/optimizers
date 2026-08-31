#!/usr/bin/env python
"""Launch gate for the program #23 channel audit (15 instrumented GPU runs).

Usage:
    uv run --no-sync python scripts/preflight_channel_audit.py [--verbose]

Exits 0 only if EVERY check passes; any BLOCKED check exits 1 and names the
human action that clears it. Nothing here trains, touches the GPU, or writes a
file -- it reads `reports/channel-audit-preregistration.md`, the five
`configs/dev/instrumented_airbench_demod*.yaml` configs and
`criteria/phase1_preregistration.md` and compares them.

Checks, in fixed order:

1. ``phase1_preregistration``      criteria/phase1_preregistration.md exists
                                   (scripts/run.py refuses airbench_instrumented
                                   without it -- WP1.2, CLAUDE.md).
2. ``prereg_status``               the pre-registration status line reads
                                   REGISTERED, not DRAFT.
3. ``threshold_freeze_table``      every row of its Appendix threshold-freeze
                                   table carries a frozen value.
4. ``run_set_matches_prereg``      the configs on disk still expand to the run
                                   set section 3 registers: seeds, run and
                                   variant counts, batch size, epochs, derived
                                   step count and lr rungs.
5. ``instrumentation_matches``     the instrumentation block is identical
                                   across the five configs and equals the
                                   registered settings (frozen probes, hvp,
                                   smoothness, the scalar block, compile).

Checks 2 and 3 are cleared by a HUMAN freezing the thresholds and flipping the
status line (CLAUDE.md ground rules 1 and 3); this script never edits either
document. Check 4 exists because a config/pre-registration drift is exactly
the failure that a header comment cannot be trusted to catch.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.optim.airbench_zoo import AIRBENCH_STOCK_LR  # noqa: E402

PREREG = REPO_ROOT / "reports" / "channel-audit-preregistration.md"
PHASE1_PREREG = REPO_ROOT / "criteria" / "phase1_preregistration.md"
CONFIG_DIR = REPO_ROOT / "configs" / "dev"
CONFIG_STEM = "instrumented_airbench_demod"
# CifarLoader(train=True) drops the last partial batch, so steps_per_epoch is
# a floor division of the CIFAR-10 training set size.
TRAIN_SET_SIZE = 50000
REQUIRED_STATUS = "REGISTERED"
# The instrumentation keys section 3 registers as a literal scalar block.
REGISTERED_SCALAR_KEYS = (
    "align_min",
    "betas",
    "k1",
    "k2",
    "min_dim",
    "momentum_key",
    "seed",
    "snapshot_every",
    "subspace_iters",
    "t_refresh",
)


class PreflightError(RuntimeError):
    """A document could not be parsed at all -- reported as BLOCKED."""


def _load_sweep_module():
    spec = importlib.util.spec_from_file_location(
        "sweep_module_preflight", REPO_ROOT / "scripts" / "sweep.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------------ parsing


def _cells(line: str) -> List[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _plain(cell: str) -> str:
    """Markdown cell -> bare text (backticks and bold markers removed)."""
    return re.sub(r"[`*]", "", cell).strip()


def _is_separator(cells: Sequence[str]) -> bool:
    return all(set(c) <= set("-: ") and c for c in cells)


def parse_status(text: str) -> str:
    match = re.search(r"^Status:\s*\*\*([A-Za-z]+)", text, flags=re.MULTILINE)
    if match is None:
        raise PreflightError(f"no 'Status: **...' line in {PREREG.name}")
    return match.group(1).upper()


def parse_freeze_table(text: str) -> List[Dict[str, str]]:
    """The Appendix threshold-freeze checklist as one dict per row."""
    head = re.search(r"^## Appendix.*$", text, flags=re.MULTILINE)
    if head is None:
        raise PreflightError(f"no '## Appendix' heading in {PREREG.name}")
    rows = []
    for line in text[head.end():].splitlines():
        line = line.strip()
        if not line.startswith("|"):
            if rows:
                break
            continue
        cells = _cells(line)
        if len(cells) != 4 or _is_separator(cells) or _plain(cells[0]) == "#":
            continue
        rows.append(
            {
                "row": _plain(cells[0]),
                "threshold": _plain(cells[1]),
                "proposed": _plain(cells[2]),
                "frozen": _plain(cells[3]),
            }
        )
    if not rows:
        raise PreflightError(f"empty threshold-freeze table in {PREREG.name}")
    return rows


def parse_run_set(text: str) -> Dict[str, Dict[str, Any]]:
    """The section-3 run-set table, keyed by config filename."""
    registered: Dict[str, Dict[str, Any]] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("|") or CONFIG_STEM not in line:
            continue
        cells = _cells(line)
        if len(cells) != 8:
            raise PreflightError(f"unexpected run-set row shape: {line}")
        name = _plain(cells[0])
        variants_cell = _plain(cells[1])
        lr_cell = _plain(cells[7])
        rungs = None
        if "lr" in variants_cell:
            rungs = sorted(float(v) for v in re.findall(r"\d+\.\d+", variants_cell))
        registered[name] = {
            "variants": int(re.match(r"\d+", variants_cell).group(0)),
            "seeds": [int(s) for s in re.findall(r"\d+", _plain(cells[2]))],
            "runs": int(_plain(cells[3])),
            "batch_size": int(_plain(cells[4])),
            "epochs": int(_plain(cells[5])),
            "steps": int(_plain(cells[6])),
            "lr_rungs": rungs,
            "lr_stock": None if rungs else float(re.match(r"[\d.]+", lr_cell).group(0)),
        }
    if not registered:
        raise PreflightError(f"no {CONFIG_STEM}*.yaml rows in {PREREG.name} section 3")
    return registered


def parse_registered_instrumentation(text: str) -> Dict[str, Any]:
    """The instrumentation settings section 3 registers, as a dict."""
    frozen = re.search(r"`frozen_probes:\s*(\{[^`]*?\})`", text, flags=re.DOTALL)
    if frozen is None:
        raise PreflightError(f"no `frozen_probes: {{...}}` mapping in {PREREG.name}")
    scalars = re.search(r"`(k1:\s*\d+[^`]*?)`", text, flags=re.DOTALL)
    if scalars is None:
        raise PreflightError(f"no `k1: ...` settings block in {PREREG.name}")
    block = yaml.safe_load("{%s}" % " ".join(scalars.group(1).split()))
    block["frozen_probes"] = yaml.safe_load(" ".join(frozen.group(1).split()))
    return block


# ------------------------------------------------------------------- checks


def check_phase1_preregistration() -> Tuple[bool, str]:
    if PHASE1_PREREG.is_file():
        return True, f"{PHASE1_PREREG.relative_to(REPO_ROOT)} present"
    return False, (
        f"{PHASE1_PREREG.relative_to(REPO_ROOT)} is missing; scripts/run.py "
        "refuses to launch airbench_instrumented without it (WP1.2)"
    )


def check_prereg_status(text: str) -> Tuple[bool, str]:
    status = parse_status(text)
    if status == REQUIRED_STATUS:
        return True, f"{PREREG.name} reads {status}"
    return False, (
        f"{PREREG.name} reads {status}, not {REQUIRED_STATUS}. Its own standing "
        "instruction is that no Phase B run may launch while it says DRAFT. "
        "Freezing the thresholds, flipping the status line and committing is a "
        "HUMAN act (CLAUDE.md ground rules 1 and 3)."
    )


def check_threshold_freeze_table(text: str) -> Tuple[bool, str]:
    rows = parse_freeze_table(text)
    unfrozen = [r["row"] for r in rows if not r["frozen"]]
    if not unfrozen:
        return True, f"all {len(rows)} threshold rows carry a frozen value"
    return False, (
        f"{len(unfrozen)} of {len(rows)} threshold-freeze rows are empty "
        f"(rows {', '.join(unfrozen)}); until the human fills them Phase B is "
        "exploratory, not confirmatory"
    )


def _config_path(name: str) -> Path:
    return CONFIG_DIR / name


def check_run_set(text: str, sweep) -> Tuple[bool, str]:
    registered = parse_run_set(text)
    problems: List[str] = []
    total_runs = 0
    for name in sorted(registered):
        want = registered[name]
        path = _config_path(name)
        if not path.is_file():
            problems.append(f"{name}: registered but not on disk")
            continue
        cfg = yaml.safe_load(path.read_text())
        plan = sweep.expand_sweep(cfg, path)
        total_runs += len(plan["runs"])
        got_seeds = sorted(plan["seeds"])
        if got_seeds != sorted(want["seeds"]):
            problems.append(f"{name}: seeds {got_seeds} != registered {want['seeds']}")
        if len(plan["variants"]) != want["variants"]:
            problems.append(
                f"{name}: {len(plan['variants'])} variants != registered {want['variants']}"
            )
        if len(plan["runs"]) != want["runs"]:
            problems.append(
                f"{name}: {len(plan['runs'])} runs != registered {want['runs']}"
            )
        batch = int(cfg["train"]["batch_size"])
        epochs = int(cfg["train"]["epochs"])
        if batch != want["batch_size"]:
            problems.append(f"{name}: batch_size {batch} != registered {want['batch_size']}")
        if epochs != want["epochs"]:
            problems.append(f"{name}: epochs {epochs} != registered {want['epochs']}")
        steps = math.ceil(epochs * (TRAIN_SET_SIZE // batch))
        if steps != want["steps"]:
            problems.append(f"{name}: {steps} derived steps != registered {want['steps']}")
        grid = (cfg.get("sweep") or {}).get("grid") or {}
        rungs = grid.get("probe_overrides.lr")
        if want["lr_rungs"] is None:
            if rungs is not None:
                problems.append(f"{name}: registered at a single lr but grids {rungs}")
            override = (cfg.get("probe_overrides") or {}).get("lr")
            lr = AIRBENCH_STOCK_LR if override is None else float(override)
            if lr != want["lr_stock"]:
                problems.append(f"{name}: lr {lr} != registered {want['lr_stock']}")
        elif rungs is None or sorted(float(v) for v in rungs) != want["lr_rungs"]:
            problems.append(f"{name}: lr rungs {rungs} != registered {want['lr_rungs']}")
    for path in sorted(CONFIG_DIR.glob(f"{CONFIG_STEM}*.yaml")):
        if path.name not in registered:
            problems.append(f"{path.name}: on disk but not registered in section 3")
    if problems:
        return False, (
            "the run set on disk is not the run set section 3 registers -- "
            "reconciling them is a HUMAN decision: " + "; ".join(problems)
        )
    return True, (
        f"{len(registered)} configs expand to {total_runs} runs, matching section 3 "
        "on seeds, variants, batch size, epochs, derived steps and lr"
    )


def check_instrumentation(text: str) -> Tuple[bool, str]:
    want = parse_registered_instrumentation(text)
    problems: List[str] = []
    blocks: Dict[str, Any] = {}
    for path in sorted(CONFIG_DIR.glob(f"{CONFIG_STEM}*.yaml")):
        cfg = yaml.safe_load(path.read_text())
        instr = cfg["instrumentation"]
        blocks[path.name] = instr
        for key in REGISTERED_SCALAR_KEYS:
            if instr.get(key) != want[key]:
                problems.append(f"{path.name}: {key} {instr.get(key)!r} != {want[key]!r}")
        if instr.get("frozen_probes") != want["frozen_probes"]:
            problems.append(
                f"{path.name}: frozen_probes {instr.get('frozen_probes')!r} != "
                f"{want['frozen_probes']!r}"
            )
        if instr.get("hvp") is not False:
            problems.append(f"{path.name}: hvp {instr.get('hvp')!r} != False")
        if (instr.get("smoothness") or {}).get("enabled") is not False:
            problems.append(f"{path.name}: smoothness.enabled != False")
        if cfg["recipe"]["compile"] != want["recipe.compile"]:
            problems.append(
                f"{path.name}: recipe.compile {cfg['recipe']['compile']!r} != "
                f"{want['recipe.compile']!r}"
            )
    reference = sorted(blocks)[0]
    for name in sorted(blocks):
        if blocks[name] != blocks[reference]:
            problems.append(f"{name}: instrumentation block differs from {reference}")
    if problems:
        return False, "instrumentation drift: " + "; ".join(problems)
    return True, (
        f"instrumentation identical across {len(blocks)} configs and equal to the "
        "registered settings (k3 %d, max_lag %d, decimate %d, hvp off, smoothness off)"
        % (
            want["frozen_probes"]["k3"],
            want["frozen_probes"]["max_lag"],
            want["frozen_probes"]["decimate"],
        )
    )


# -------------------------------------------------------------------- main


def run_checks() -> List[Tuple[str, bool, str]]:
    """Every check, in fixed order, as (name, ok, detail)."""
    sweep = _load_sweep_module()
    text = PREREG.read_text() if PREREG.is_file() else None
    checks = [
        ("phase1_preregistration", check_phase1_preregistration, ()),
        ("prereg_status", check_prereg_status, (text,)),
        ("threshold_freeze_table", check_threshold_freeze_table, (text,)),
        ("run_set_matches_prereg", check_run_set, (text, sweep)),
        ("instrumentation_matches", check_instrumentation, (text,)),
    ]
    results = []
    for name, fn, args in checks:
        if text is None and args:
            results.append((name, False, f"{PREREG.relative_to(REPO_ROOT)} is missing"))
            continue
        try:
            ok, detail = fn(*args)
        except Exception as exc:  # a gate fails closed
            ok, detail = False, f"{type(exc).__name__}: {exc}"
        results.append((name, ok, detail))
    return results


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--verbose", action="store_true", help="print the detail of passing checks too"
    )
    args = parser.parse_args(argv)

    results = run_checks()
    width = max(len(name) for name, _, _ in results)
    for name, ok, detail in results:
        label = "OK     " if ok else "BLOCKED"
        if ok and not args.verbose:
            print(f"{label} {name.ljust(width)}")
        else:
            print(f"{label} {name.ljust(width)}  {detail}")
    blocked = [name for name, ok, _ in results if not ok]
    if blocked:
        print(
            f"\nBLOCKED on {len(blocked)} of {len(results)} checks "
            f"({', '.join(blocked)}). Do not launch; report (CLAUDE.md ground rule 6)."
        )
        return 1
    print(f"\nAll {len(results)} checks pass. Launching remains the human's call.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
