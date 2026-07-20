#!/usr/bin/env python
"""Twin-trajectory divergence probe (brainstorm program #1: compounding).

Trains TWO models in lockstep in a single process -- identical initialization
(same seed), identical batch sequence -- differing ONLY in the optimizer used
for the matrix (filter / linear) parameters:

    twin A: stock Muon (the reference trajectory);
    twin B: a chosen intervention (optimizer name + kwargs from the config),
            e.g. output-side routed damping or the new state-side damping.

Per step it records, over the matrix parameters:

    rel_dist(t) = ||W_A - W_B||_F / ||W_A||_F   (per matrix and aggregated),
    update-direction cosine = cos(Delta W_A, Delta W_B).

This is a MEASUREMENT, not an accuracy claim. It answers whether an
intervention actually moves the training trajectory, and how that scales with
step count -- the prerequisite the compounding hypothesis (see below) must
clear before any accuracy comparison is worth running.

================================ HYPOTHESIS ==================================
Routed Muon v0 damps per-direction components in the Newton-Schulz OUTPUT O_t
= NS(M_t) only; the momentum state M_t is never edited, so each step's output
modification is re-derived from an unmodified buffer next step and cannot
compound. Error-feedback theory (EF21-Muon) proves the accumulate-or-fail
dichotomy; DeMo / Dion subtract applied components from the buffer (state-side)
for compression. State-side damping (routed.py: state_damping=True) writes the
attenuation into the buffer itself, so it persists across steps.

========================= PRE-REGISTERED PREDICTION ==========================
Output-mode relative divergence stays small (same order as the goscconst-
output arm's known behavioral equivalence, i.e. well below 1e-2 by the end of
the 200-step run) while state-mode divergence GROWS with step count and ends
at LEAST 10x the output-mode divergence at matched gain (the goscconst-0.50
output-vs-state pair, identical constant gain, only the application point
differing). If instead BOTH are small or BOTH are large, the compounding
explanation of the Gate-2 null is WRONG and must be reported as refuted.
=============================================================================

Harnesses:
    harness: airbench  -- the vendored airbench94 recipe (CUDA required), the
        standard record-config path; only the filter-optimizer differs A vs B.
    harness: mlp       -- a self-contained tiny-MLP twin on synthetic data
        (CPU-friendly, fully deterministic) that exercises the whole probe
        path for testing.

Run (standard container / local path -- experiment registered in scripts/run.py):
    uv run python scripts/run.py configs/dev/probe_divergence_state.yaml
    uv run python scripts/probe_divergence.py configs/dev/probe_divergence_state.yaml

Dev seeds only (>= 1000): this is a measurement probe, never a comparison-table
entry.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.optim.registry import build_optimizer  # noqa: E402

# The pre-registered prediction, verbatim, stored into every results JSON.
PREDICTION = (
    "Output-mode relative divergence stays small (same order as the "
    "goscconst-output arm's known behavioral equivalence, i.e. well below "
    "1e-2 by the end of the 200-step run) while state-mode divergence grows "
    "with step count and ends at least 10x the output-mode divergence at "
    "matched gain. If instead both are small or both are large, the "
    "compounding explanation of the Gate-2 null is wrong and must be reported "
    "as refuted."
)

_DTYPES = {
    "float32": torch.float32,
    "float": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "half": torch.float16,
}


def _resolve_spec(spec: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """Split a twin optimizer spec into (name, kwargs), resolving ns_dtype
    from a string (YAML cannot express a torch.dtype)."""
    kw = dict(spec)
    name = kw.pop("name")
    if "ns_dtype" in kw and isinstance(kw["ns_dtype"], str):
        try:
            kw["ns_dtype"] = _DTYPES[kw["ns_dtype"].lower()]
        except KeyError:
            raise SystemExit(
                f"unknown ns_dtype {kw['ns_dtype']!r}; known: {sorted(_DTYPES)}"
            )
    return name, kw


# ------------------------------------------------------------ divergence math


def _matrix_divergence(
    pairs: List[Tuple[str, torch.Tensor, torch.Tensor]],
) -> Tuple[float, Dict[str, float]]:
    """Aggregate + per-matrix relative Frobenius distance ||A-B||/||A||."""
    num_sq = 0.0
    den_sq = 0.0
    per: Dict[str, float] = {}
    for name, a, b in pairs:
        a = a.detach().float()
        b = b.detach().float()
        d = float((a - b).norm())
        na = float(a.norm())
        per[name] = d / na if na > 0 else 0.0
        num_sq += d * d
        den_sq += na * na
    agg = (num_sq ** 0.5) / (den_sq ** 0.5) if den_sq > 0 else 0.0
    return agg, per


def _update_cosine(
    deltas_a: List[torch.Tensor], deltas_b: List[torch.Tensor]
) -> float:
    """Cosine between the concatenated matrix-parameter updates of the twins."""
    va = torch.cat([d.detach().float().reshape(-1) for d in deltas_a])
    vb = torch.cat([d.detach().float().reshape(-1) for d in deltas_b])
    na = float(va.norm())
    nb = float(vb.norm())
    if na == 0.0 or nb == 0.0:
        return float("nan")
    return float(torch.dot(va, vb) / (na * nb))


# ------------------------------------------------------------- MLP twin harness


def run_mlp_twin(config: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """Self-contained twin probe: a bias-free 2-layer MLP (all params are
    matrices) on synthetic classification data. Deterministic on CPU: with
    identical twin specs the trajectories must not diverge at all (the
    determinism control asserted by the CPU test)."""
    model_cfg = config.get("model", {})
    train_cfg = config.get("train", {})
    in_dim = int(model_cfg.get("in_dim", 32))
    hidden = int(model_cfg.get("hidden_dim", 48))
    out_dim = int(model_cfg.get("out_dim", 4))
    batch_size = int(train_cfg.get("batch_size", 32))
    steps = int(config.get("steps", 200))

    def make_model():
        return torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden, bias=False),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, out_dim, bias=False),
        ).to(device)

    model_a = make_model()
    model_b = make_model()
    model_b.load_state_dict(model_a.state_dict())  # identical init

    name_a, kw_a = _resolve_spec(config["twin_a"])
    name_b, kw_b = _resolve_spec(config["twin_b"])
    opt_a = build_optimizer(name_a, model_a.parameters(), kw_a)
    opt_b = build_optimizer(name_b, model_b.parameters(), kw_b)
    loss_fn = torch.nn.CrossEntropyLoss()

    def named_matrices(model):
        return [
            (f"linear{i}", p)
            for i, p in enumerate(model.parameters())
            if p.ndim >= 2
        ]

    gen = torch.Generator(device="cpu").manual_seed(int(config.get("seed", 1600)))
    divergence = []
    loss_a = loss_b = float("nan")
    for step in range(1, steps + 1):
        x = torch.randn(batch_size, in_dim, generator=gen).to(device)
        y = torch.randint(0, out_dim, (batch_size,), generator=gen).to(device)

        before_a = [p.detach().clone() for _, p in named_matrices(model_a)]
        before_b = [p.detach().clone() for _, p in named_matrices(model_b)]

        opt_a.zero_grad(set_to_none=True)
        out_a = loss_fn(model_a(x), y)
        out_a.backward()
        opt_a.step()
        loss_a = float(out_a.detach())

        opt_b.zero_grad(set_to_none=True)
        out_b = loss_fn(model_b(x), y)
        out_b.backward()
        opt_b.step()
        loss_b = float(out_b.detach())

        mats_a = named_matrices(model_a)
        mats_b = named_matrices(model_b)
        deltas_a = [p.detach() - b0 for (_, p), b0 in zip(mats_a, before_a)]
        deltas_b = [p.detach() - b0 for (_, p), b0 in zip(mats_b, before_b)]
        pairs = [(na, pa, pb) for (na, pa), (_, pb) in zip(mats_a, mats_b)]
        agg, per = _matrix_divergence(pairs)
        divergence.append(
            {
                "step": step,
                "rel_dist": agg,
                "update_cosine": _update_cosine(deltas_a, deltas_b),
                "per_matrix_rel_dist": per,
            }
        )

    return _finish(config, divergence, name_a, kw_a, name_b, kw_b,
                   final_a={"loss": loss_a}, final_b={"loss": loss_b},
                   harness="mlp")


# -------------------------------------------------------- airbench twin harness


def _build_airbench_twin(ab, config, filter_spec, data_root, batch_size):
    """Construct one airbench94 model + its (SGD side, matrix) optimizers."""
    recipe_cfg = config.get("recipe", {})
    bias_lr = float(recipe_cfg.get("bias_lr", 0.053))
    head_lr = float(recipe_cfg.get("head_lr", 0.67))
    wd = float(recipe_cfg.get("sgd_weight_decay", 2e-6)) * batch_size

    model = ab.CifarNet().cuda().to(memory_format=torch.channels_last)
    filter_params = [
        p for p in model.parameters() if len(p.shape) == 4 and p.requires_grad
    ]
    norm_biases = [
        p for n, p in model.named_parameters() if "norm" in n and p.requires_grad
    ]
    param_configs = [
        dict(params=[model.whiten.bias], lr=bias_lr, weight_decay=wd / bias_lr),
        dict(params=norm_biases, lr=bias_lr, weight_decay=wd / bias_lr),
        dict(params=[model.head.weight], lr=head_lr, weight_decay=wd / head_lr),
    ]
    opt1 = torch.optim.SGD(param_configs, momentum=0.85, nesterov=True, fused=True)
    name, kw = _resolve_spec(filter_spec)
    opt2 = build_optimizer(name, filter_params, kw)
    for opt in (opt1, opt2):
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]
    return model, opt1, opt2, filter_params, (name, kw)


def run_airbench_twin(config: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """Twin probe on the vendored airbench94 recipe (CUDA). Both twins share
    an identical initialization, an identical per-step batch, and identical
    schedules; only the filter-parameter optimizer differs (A: stock Muon,
    B: the intervention). 200-step budget by default (record-config scope)."""
    import math

    from src.optim.airbench_zoo import load_vendor_airbench

    if device.type != "cuda":
        raise SystemExit(
            "harness 'airbench' requires a CUDA device (vendored CifarLoader "
            "maps data to cuda; the model is half precision). Use harness 'mlp' "
            "for CPU testing."
        )
    ab = load_vendor_airbench()

    train_cfg = config.get("train", {})
    recipe_cfg = config.get("recipe", {})
    data_root = str(config.get("data", {}).get("root", "data/cifar10"))
    batch_size = int(train_cfg.get("batch_size", 2000))
    steps = int(config.get("steps", 200))
    normalize_filter_weights = bool(recipe_cfg.get("normalize_filter_weights", True))
    tta_level = int(recipe_cfg.get("tta_level", 2))

    test_loader = ab.CifarLoader(data_root, train=False, batch_size=2000)
    train_loader = ab.CifarLoader(
        data_root, train=True, batch_size=batch_size, aug=dict(flip=True, translate=2)
    )
    total_train_steps = steps
    whiten_bias_train_steps = min(math.ceil(3 * len(train_loader)), total_train_steps)

    # Twin A = stock Muon; twin B = the intervention. Default twin_a to the
    # record-config vendored Muon hyperparameters if unspecified.
    spec_a = config.get(
        "twin_a", dict(name="muon", lr=0.24, momentum=0.6, nesterov=True, ns_steps=3)
    )
    spec_b = config["twin_b"]

    model_a, opt1_a, opt2_a, filt_a, resolved_a = _build_airbench_twin(
        ab, config, spec_a, data_root, batch_size
    )
    model_b, opt1_b, opt2_b, filt_b, resolved_b = _build_airbench_twin(
        ab, config, spec_b, data_root, batch_size
    )
    # Identical initialization: reset A, copy its state into B, whiten both
    # from the same training images.
    #
    # ``twin_b_seed`` (calibration arm only) instead re-seeds and resets B
    # independently, so the twins differ by INIT rather than by optimizer.
    # That measures the reference scale of the divergence metric: how far
    # apart do two equally-good stock runs end up? See
    # configs/dev/probe_divergence_seedcal.yaml.
    model_a.reset()
    twin_b_seed = config.get("twin_b_seed")
    if twin_b_seed is None:
        model_b.load_state_dict(model_a.state_dict())
    else:
        if int(twin_b_seed) < 1000:
            raise SystemExit("twin_b_seed must be a dev seed (>= 1000)")
        torch.manual_seed(int(twin_b_seed))
        torch.cuda.manual_seed_all(int(twin_b_seed))
        model_b.reset()
    train_images = train_loader.normalize(train_loader.images[:5000])
    model_a.init_whiten(train_images)
    model_b.init_whiten(train_images)

    names_a = {id(p): n for n, p in model_a.named_parameters()}
    filt_names = [names_a.get(id(p), f"filter{i}") for i, p in enumerate(filt_a)]

    def step_twin(model, opt1, opt2, filt, inputs, labels, step):
        outputs = model(inputs, whiten_bias_grad=(step < whiten_bias_train_steps))
        torch.nn.functional.cross_entropy(
            outputs, labels, label_smoothing=0.2, reduction="sum"
        ).backward()
        for group in opt1.param_groups[:1]:
            group["lr"] = group["initial_lr"] * (1 - step / whiten_bias_train_steps)
        for group in opt1.param_groups[1:] + opt2.param_groups:
            group["lr"] = group["initial_lr"] * (1 - step / total_train_steps)
        if normalize_filter_weights:
            for p in filt:
                p.data.mul_(len(p.data) ** 0.5 / p.data.norm())
        before = [p.detach().clone() for p in filt]
        opt1.step()
        opt2.step()
        deltas = [p.detach() - b0 for p, b0 in zip(filt, before)]
        model.zero_grad(set_to_none=True)
        return deltas

    divergence = []
    step = 0
    done = False
    for _epoch in range(math.ceil(total_train_steps / len(train_loader))):
        model_a.train()
        model_b.train()
        for inputs, labels in train_loader:
            deltas_a = step_twin(
                model_a, opt1_a, opt2_a, filt_a, inputs, labels, step
            )
            deltas_b = step_twin(
                model_b, opt1_b, opt2_b, filt_b, inputs, labels, step
            )
            step += 1
            pairs = list(zip(filt_names, filt_a, filt_b))
            agg, per = _matrix_divergence(pairs)
            divergence.append(
                {
                    "step": step,
                    "rel_dist": agg,
                    "update_cosine": _update_cosine(deltas_a, deltas_b),
                    "per_matrix_rel_dist": per,
                }
            )
            if step >= total_train_steps:
                done = True
                break
        if done:
            break

    model_a.eval()
    model_b.eval()
    final_a = {"tta_val_acc": ab.evaluate(model_a, test_loader, tta_level=tta_level)}
    final_b = {"tta_val_acc": ab.evaluate(model_b, test_loader, tta_level=tta_level)}
    return _finish(config, divergence, *resolved_a, *resolved_b,
                   final_a=final_a, final_b=final_b, harness="airbench")


# ------------------------------------------------------------------- assembly


def _finish(config, divergence, name_a, kw_a, name_b, kw_b, *,
            final_a, final_b, harness) -> Dict[str, Any]:
    final_rel = divergence[-1]["rel_dist"] if divergence else float("nan")
    max_rel = max((d["rel_dist"] for d in divergence), default=float("nan"))
    return {
        "harness": harness,
        "steps": len(divergence),
        "prediction": PREDICTION,
        "twin_a": {"name": name_a, **_jsonable_kwargs(kw_a)},
        "twin_b": {"name": name_b, **_jsonable_kwargs(kw_b)},
        "divergence": divergence,
        "final": {
            "rel_dist": final_rel,
            "max_rel_dist": max_rel,
            "twin_a": final_a,
            "twin_b": final_b,
        },
    }


def _jsonable_kwargs(kw: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in kw.items():
        out[k] = str(v) if isinstance(v, torch.dtype) else v
    return out


def run_probe_divergence(config: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """Experiment entrypoint (registered in scripts/run.py)."""
    if "twin_b" not in config:
        raise SystemExit("probe_divergence config requires a 'twin_b' spec")
    harness = config.get("harness", "airbench")
    if harness == "mlp":
        return run_mlp_twin(config, device)
    if harness == "airbench":
        return run_airbench_twin(config, device)
    raise SystemExit(f"unknown harness {harness!r}; expected 'mlp' or 'airbench'")


# ----------------------------------------------------------------- entrypoint


def _load_run_module():
    import importlib.util

    path = REPO_ROOT / "scripts" / "run.py"
    spec = importlib.util.spec_from_file_location("routed_muon_run_probe", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main(argv=None) -> int:
    """Delegate to scripts/run.py with the probe experiment registered.

    Usage: uv run python scripts/probe_divergence.py <config.yaml> [--seed N]
    """
    run_mod = _load_run_module()
    run_mod.EXPERIMENT_REGISTRY["probe_divergence"] = run_probe_divergence
    return run_mod.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
