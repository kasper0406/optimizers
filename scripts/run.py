#!/usr/bin/env python
"""Config-driven experiment runner (WP0.0).

Usage:
    uv run python scripts/run.py <config.yaml> [--seed N] [--out-dir DIR]

Reads a YAML config, runs the experiment it names, and writes exactly one
results JSON (schema: src/results_io.py) into results/.

Seed policy (CLAUDE.md ground rule 2): configs carry a literal dev seed
(>= 1000). Evaluation seeds 0-99 are never written into configs; sweep/launch
tooling passes them at run time via --seed. A literal config seed < 1000 is
rejected.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import yaml

from src import results_io
from src.optim import NoOpOptimizer
from src.optim.registry import OPTIMIZER_REGISTRY as _ZOO_REGISTRY

OPTIMIZER_REGISTRY = {
    "noop": NoOpOptimizer,
    **_ZOO_REGISTRY,  # WP0.4 zoo: muon/adamw/dynmuon/adamuon/normuon
    # WP2.1 registers routed here.
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def gpu_type_string(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    return device.type


def build_optimizer(name: str, params, kwargs: Dict[str, Any]):
    try:
        cls = OPTIMIZER_REGISTRY[name]
    except KeyError:
        raise SystemExit(
            f"Unknown optimizer {name!r}; known: {sorted(OPTIMIZER_REGISTRY)}"
        )
    return cls(params, **kwargs)


# --------------------------------------------------------------- experiments


def run_smoke(config: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """Tiny MLP on random data through the full optimizer-interface path.

    A '10-step training no-op': with the NoOpOptimizer the parameters must not
    change; the point is to exercise config -> trainer -> optimizer hooks ->
    metrics end to end.
    """
    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})
    train_cfg = config.get("train", {})
    opt_cfg = dict(config.get("optimizer", {}))
    opt_name = opt_cfg.pop("name", "noop")

    in_dim = int(model_cfg.get("in_dim", 32))
    hidden_dim = int(model_cfg.get("hidden_dim", 64))
    out_dim = int(model_cfg.get("out_dim", 4))
    batch_size = int(data_cfg.get("batch_size", 16))
    steps = int(train_cfg.get("steps", 10))
    # model.bias: false gives an all-2-D-parameter MLP so matrix-only
    # optimizers (Muon family / routed) can smoke-run on CPU -- mirrors the
    # airbench harness's filter-params split. Default true (original smoke).
    use_bias = bool(model_cfg.get("bias", True))

    model = torch.nn.Sequential(
        torch.nn.Linear(in_dim, hidden_dim, bias=use_bias),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_dim, out_dim, bias=use_bias),
    ).to(device)
    optimizer = build_optimizer(opt_name, model.parameters(), opt_cfg)
    loss_fn = torch.nn.CrossEntropyLoss()

    initial_params = [p.detach().clone() for p in model.parameters()]

    losses = []
    for _ in range(steps):
        x = torch.randn(batch_size, in_dim, device=device)
        y = torch.randint(0, out_dim, (batch_size,), device=device)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

    max_param_delta = max(
        float((p.detach() - p0).abs().max().item())
        for p, p0 in zip(model.parameters(), initial_params)
    )

    metrics = {
        "steps": steps,
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "losses": losses,
        "max_param_delta": max_param_delta,
        "optimizer": opt_name,
    }
    if hasattr(optimizer, "routing_stats"):
        # Gate-1 amendment A5 (mirrors the airbench harness wiring): routed
        # optimizers report per-channel occupancy / treated-fraction / gain
        # telemetry; the smoke experiment is the CPU-verifiable path.
        metrics["routing_stats"] = optimizer.routing_stats()
    return metrics


from src.optim.airbench_zoo import (  # noqa: E402
    run_airbench,
    run_airbench_instrumented,
    run_airbench_smoke,
)


def _load_probe_divergence():
    """Load scripts/probe_divergence.py (not a package) for its experiment
    function, so the twin-trajectory probe runs through the standard runner /
    container entrypoint. Import-time only; no side effects."""
    import importlib.util

    path = REPO_ROOT / "scripts" / "probe_divergence.py"
    spec = importlib.util.spec_from_file_location("routed_muon_probe_divergence", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.run_probe_divergence


EXPERIMENT_REGISTRY = {
    "smoke": run_smoke,
    "airbench": run_airbench,  # WP0.1 stock baseline (vendored Muon, compile on)
    "airbench_smoke": run_airbench_smoke,  # WP0.4 zoo smoke harness
    "airbench_instrumented": run_airbench_instrumented,  # WP1.2 measurement runs
    "probe_divergence": _load_probe_divergence(),  # twin-trajectory probe
    # Later WPs register nanogpt experiments here.
}

# WP1.2 launch precondition (CLAUDE.md): the human-authored, pre-registered
# Phase-1 criteria must exist BEFORE any instrumented measurement run.  The
# agent never authors this file; main() refuses to run the instrumented
# experiment while it is missing.
PHASE1_PREREG = REPO_ROOT / "criteria" / "phase1_preregistration.md"
PREREG_GATED_EXPERIMENTS = {"airbench_instrumented"}


def check_phase1_preregistration(experiment: str) -> None:
    """Refuse pre-registration-gated experiments while criteria are absent."""
    if experiment in PREREG_GATED_EXPERIMENTS and not PHASE1_PREREG.exists():
        raise SystemExit(
            f"experiment {experiment!r} is a WP1.2 Phase-1 measurement run and "
            f"requires the human-authored pre-registration file "
            f"{PHASE1_PREREG} to exist (committed before the first run). "
            "It is missing; refusing to launch. The agent must not create it."
        )


# ---------------------------------------------------------------------- main


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Path to experiment YAML config")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed override (used by sweep/launch tooling, incl. eval seeds)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=results_io.RESULTS_DIR,
        help="Directory for the results JSON (default: results/)",
    )
    args = parser.parse_args(argv)

    with open(args.config) as fh:
        config = yaml.safe_load(fh)
    if not isinstance(config, dict):
        raise SystemExit(f"Config {args.config} did not parse to a mapping")

    experiment = config.get("experiment")
    if experiment not in EXPERIMENT_REGISTRY:
        raise SystemExit(
            f"Unknown experiment {experiment!r}; known: {sorted(EXPERIMENT_REGISTRY)}"
        )
    check_phase1_preregistration(experiment)

    config_seed = config.get("seed")
    if args.seed is None:
        if not isinstance(config_seed, int):
            raise SystemExit("Config must contain an integer 'seed' (>= 1000)")
        if config_seed < 1000:
            raise SystemExit(
                f"Config seed {config_seed} < 1000. Literal config seeds must be "
                "dev seeds (>= 1000); eval seeds 0-99 are passed by launch "
                "tooling via --seed."
            )
        seed = config_seed
    else:
        seed = args.seed

    device = resolve_device(str(config.get("device", "cpu")))
    set_seed(seed)

    started_at = results_io.utc_now_iso()
    t0 = time.perf_counter()
    metrics = EXPERIMENT_REGISTRY[experiment](config, device)
    wall_time_s = time.perf_counter() - t0
    finished_at = results_io.utc_now_iso()

    stamp = started_at.replace(":", "").replace("-", "").split(".")[0]
    out_path = args.out_dir / f"{experiment}_seed{seed}_{stamp}.json"

    # Instrumented experiments return the full per-direction log under a
    # private key; it is written as a sidecar next to the results JSON
    # (src.instrument.schema) and referenced by filename from metrics.
    instr_log = metrics.pop("_instrumentation_log", None)
    if instr_log is not None:
        from src.instrument.schema import write_sidecar

        sidecar = write_sidecar(instr_log, out_path)
        metrics["instrumentation_sidecar"] = sidecar.name
        print(f"Wrote {sidecar}")

    result = {
        "schema_version": results_io.SCHEMA_VERSION,
        "experiment": experiment,
        "config": results_io.config_record(args.config, config),
        **results_io.git_provenance(),
        "seed": seed,
        "gpu_type": gpu_type_string(device),
        "wall_time_s": wall_time_s,
        "cost_usd": None,  # human-filled for cloud runs
        "started_at": started_at,
        "finished_at": finished_at,
        "metrics": metrics,
    }

    results_io.write_result(result, out_path)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
