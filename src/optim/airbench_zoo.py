"""Airbench smoke-run harness for the baseline zoo (WP0.4, part b).

Runs the vendored airbench94 recipe (vendor/airbench/airbench94_muon.py) with
the filter-parameter optimizer swapped for any zoo optimizer from
``src.optim.registry``. Model, data pipeline, augmentation, SGD side-optimizer
for biases/head, LR schedules, and evaluation (incl. TTA) are all the vendored
code / faithful ports of its ``main()`` (lines 340-432).

Requires CUDA: the vendored ``CifarLoader`` maps the dataset to "cuda"
(airbench94_muon.py:128) and the model runs in half precision. On the dev Mac
this module only needs to *import* and parse configs; the actual run happens
on a GPU box via::

    bash scripts/launch_local.sh configs/dev/airbench_smoke_muon.yaml
    # (once scripts/run.py registers this experiment -- see WIRING below), or
    uv run python -m src.optim.airbench_zoo configs/dev/airbench_smoke_muon.yaml

WIRING: scripts/run.py is owned by the WP0.0/WP0.1 tooling; to register this
experiment there, add::

    from src.optim import OPTIMIZER_REGISTRY as ZOO
    from src.optim.airbench_zoo import run_airbench_smoke
    OPTIMIZER_REGISTRY.update(ZOO)
    EXPERIMENT_REGISTRY["airbench_smoke"] = run_airbench_smoke

Until then, the ``python -m src.optim.airbench_zoo`` entrypoint below performs
the same registration at runtime (without editing scripts/run.py) and
delegates to ``scripts/run.py:main`` so results JSONs share the WP0.0 schema
and provenance fields.

torchvision note: the vendored script imports torchvision only for
``transforms.Normalize`` and the one-time CIFAR-10 download. torchvision is
not in pyproject.toml (locked; cannot be edited by WP0.4), so when it is
absent a minimal stand-in is installed into ``sys.modules`` before the vendor
import: Normalize as the standard (x - mean)/std, and a CIFAR10 class that
downloads/parses the canonical cifar-10-python.tar.gz with numpy only. With
real torchvision installed the stand-in is never used.
"""

from __future__ import annotations

import math
import pickle
import sys
import tarfile
import types
import urllib.request
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
VENDOR_AIRBENCH = REPO_ROOT / "vendor" / "airbench"

CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"


# ------------------------------------------------------- torchvision stand-in


class _Normalize:
    """transforms.Normalize equivalent for CHW tensors: (x - mean) / std."""

    def __init__(self, mean, std):
        self.mean = torch.as_tensor(mean)
        self.std = torch.as_tensor(std)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        mean = self.mean.to(device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
        std = self.std.to(device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
        return (x - mean) / std


class _CIFAR10:
    """torchvision.datasets.CIFAR10 equivalent (data/targets/classes only).

    Downloads and parses the canonical CIFAR-10 python tarball with numpy;
    exposes exactly the attributes the vendored CifarLoader uses
    (airbench94_muon.py:122-126): .data (N,32,32,3 uint8), .targets, .classes.
    """

    def __init__(self, root, download: bool = False, train: bool = True):
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        tar_path = root / "cifar-10-python.tar.gz"
        if not tar_path.exists():
            if not download:
                raise RuntimeError(f"CIFAR-10 not found at {tar_path}")
            urllib.request.urlretrieve(CIFAR10_URL, tar_path)
        batch_names = (
            [f"data_batch_{i}" for i in range(1, 6)] if train else ["test_batch"]
        )
        images, targets = [], []
        with tarfile.open(tar_path, "r:gz") as tar:
            for name in batch_names:
                with tar.extractfile(f"cifar-10-batches-py/{name}") as fh:
                    batch = pickle.load(fh, encoding="latin1")
                images.append(
                    np.asarray(batch["data"], dtype=np.uint8).reshape(-1, 3, 32, 32)
                )
                targets.extend(batch["labels"])
            with tar.extractfile("cifar-10-batches-py/batches.meta") as fh:
                meta = pickle.load(fh, encoding="latin1")
        self.data = np.concatenate(images).transpose(0, 2, 3, 1)  # NHWC uint8
        self.targets = list(targets)
        self.classes = list(meta["label_names"])


def _ensure_torchvision() -> None:
    """Install a minimal torchvision stand-in if the real one is missing."""
    try:
        import torchvision  # noqa: F401

        return
    except ModuleNotFoundError:
        pass
    tv = types.ModuleType("torchvision")
    transforms = types.ModuleType("torchvision.transforms")
    datasets = types.ModuleType("torchvision.datasets")
    transforms.Normalize = _Normalize
    datasets.CIFAR10 = _CIFAR10
    tv.transforms = transforms
    tv.datasets = datasets
    sys.modules["torchvision"] = tv
    sys.modules["torchvision.transforms"] = transforms
    sys.modules["torchvision.datasets"] = datasets


# ------------------------------------------------------------- vendor loading

_VENDOR_CACHE = None


def load_vendor_airbench():
    """Import vendor/airbench/airbench94_muon.py as a module (idempotent).

    Module-level code only defines the model/loader/eval helpers (training is
    under ``if __name__ == "__main__"``), so importing is side-effect free
    apart from reading sys.argv[0] for its self-logging feature.
    """
    global _VENDOR_CACHE
    if _VENDOR_CACHE is not None:
        return _VENDOR_CACHE
    _ensure_torchvision()
    import importlib.util

    path = VENDOR_AIRBENCH / "airbench94_muon.py"
    spec = importlib.util.spec_from_file_location("airbench94_muon_vendored", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    # The vendored script self-logs its own source via ``open(sys.argv[0])`` at
    # import time (airbench94_muon.py:14). Point argv[0] at the vendored file
    # during import so (a) the import survives contexts where argv[0] is not a
    # readable file (e.g. ``python -c``) and (b) the logged code is the actual
    # vendored source, matching the reference's intent.
    argv0 = sys.argv[0]
    sys.argv[0] = str(path)
    try:
        spec.loader.exec_module(module)
    finally:
        sys.argv[0] = argv0
    _VENDOR_CACHE = module
    return module


# ----------------------------------------------------------------- experiment


def run_airbench_smoke(
    config: Dict[str, Any],
    device: torch.device,
    _hub_factory=None,
) -> Dict[str, Any]:
    """One airbench94 training run with a zoo optimizer on the filter params.

    Faithful port of vendor/airbench/airbench94_muon.py:main() (lines
    340-432) with:
    - optimizer2 (filter params) built from src.optim.registry per config;
    - the reference's per-step weight renormalization (line 83, inside the
      vendored Muon.step) applied harness-side to filter params so every zoo
      optimizer trains under the identical recipe (config-switchable);
    - torch.compile optional (smoke default: off);
    - wall-clock timing via CUDA events as in the reference.

    ``_hub_factory`` (internal; used by :func:`run_airbench_instrumented`):
    callable ``(model, optimizer2, filter_params) -> InstrumentationHub``.
    When given, the hub observes every step -- ``capture_grads()`` right
    before the optimizer steps (the raw PRE-momentum gradient; the vendored
    Muon's ``step()`` must never be assumed to leave ``p.grad`` intact) and
    ``after_step()`` right after, before ``zero_grad``.  Instrumentation is
    strictly read-only: it never modifies parameters, gradients, optimizer
    state, or any update.
    """
    from src.optim.registry import build_optimizer

    if device.type != "cuda":
        raise SystemExit(
            "airbench_smoke requires a CUDA device: the vendored CifarLoader "
            "maps data to cuda (airbench94_muon.py:128) and the model is half "
            "precision. Run this config on a GPU box."
        )

    ab = load_vendor_airbench()

    opt_cfg = dict(config.get("optimizer", {}))
    opt_name = opt_cfg.pop("name")
    train_cfg = config.get("train", {})
    recipe_cfg = config.get("recipe", {})
    data_root = str(config.get("data", {}).get("root", "data/cifar10"))

    epochs = float(train_cfg.get("epochs", 8))
    batch_size = int(train_cfg.get("batch_size", 2000))
    bias_lr = float(recipe_cfg.get("bias_lr", 0.053))
    head_lr = float(recipe_cfg.get("head_lr", 0.67))
    wd = float(recipe_cfg.get("sgd_weight_decay", 2e-6)) * batch_size
    normalize_filter_weights = bool(recipe_cfg.get("normalize_filter_weights", True))
    tta_level = int(recipe_cfg.get("tta_level", 2))

    model = ab.CifarNet().cuda().to(memory_format=torch.channels_last)
    if bool(recipe_cfg.get("compile", False)):
        model.compile()

    test_loader = ab.CifarLoader(data_root, train=False, batch_size=2000)
    train_loader = ab.CifarLoader(
        data_root, train=True, batch_size=batch_size, aug=dict(flip=True, translate=2)
    )
    total_train_steps = math.ceil(epochs * len(train_loader))
    whiten_bias_train_steps = min(
        math.ceil(3 * len(train_loader)), total_train_steps
    )

    # Parameter split identical to the reference (lines 356-361)
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
    optimizer1 = torch.optim.SGD(
        param_configs, momentum=0.85, nesterov=True, fused=(device.type == "cuda")
    )
    if opt_name == "vendor_muon":
        # WP0.1 baseline: the vendored Muon itself (airbench94_muon.py:56-84).
        # It renormalizes filter weights inside step() (line 83), so the
        # harness-side renormalization must stay off to avoid applying it twice.
        if normalize_filter_weights:
            raise SystemExit(
                "optimizer 'vendor_muon' renormalizes weights inside step(); "
                "set recipe.normalize_filter_weights: false"
            )
        optimizer2 = ab.Muon(filter_params, **opt_cfg)
    else:
        optimizer2 = build_optimizer(opt_name, filter_params, opt_cfg)
    optimizers = [optimizer1, optimizer2]
    for opt in optimizers:
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]

    hub = None
    if _hub_factory is not None:
        hub = _hub_factory(model, optimizer2, filter_params)

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    time_seconds = 0.0

    def start_timer():
        starter.record()

    def stop_timer():
        nonlocal time_seconds
        ender.record()
        torch.cuda.synchronize()
        time_seconds += 1e-3 * starter.elapsed_time(ender)

    model.reset()
    step = 0

    start_timer()
    train_images = train_loader.normalize(train_loader.images[:5000])
    model.init_whiten(train_images)
    stop_timer()

    val_accs = []
    train_acc = float("nan")
    for _epoch in range(math.ceil(total_train_steps / len(train_loader))):
        start_timer()
        model.train()
        for inputs, labels in train_loader:
            outputs = model(inputs, whiten_bias_grad=(step < whiten_bias_train_steps))
            torch.nn.functional.cross_entropy(
                outputs, labels, label_smoothing=0.2, reduction="sum"
            ).backward()
            for group in optimizer1.param_groups[:1]:
                group["lr"] = group["initial_lr"] * (
                    1 - step / whiten_bias_train_steps
                )
            for group in optimizer1.param_groups[1:] + optimizer2.param_groups:
                group["lr"] = group["initial_lr"] * (1 - step / total_train_steps)
            if normalize_filter_weights:
                # airbench94_muon.py:83 (recipe step, applied uniformly)
                for p in filter_params:
                    p.data.mul_(len(p.data) ** 0.5 / p.data.norm())
            if hub is not None:
                hub.capture_grads()  # raw pre-momentum G, before any step()
            for opt in optimizers:
                opt.step()
            if hub is not None:
                hub.after_step()  # reads captured G + post-step momentum
            model.zero_grad(set_to_none=True)
            step += 1
            if step >= total_train_steps:
                break
        stop_timer()

        train_acc = (outputs.detach().argmax(1) == labels).float().mean().item()
        val_accs.append(ab.evaluate(model, test_loader, tta_level=0))
        if step >= total_train_steps:
            break

    start_timer()
    tta_val_acc = (
        ab.evaluate(model, test_loader, tta_level=tta_level) if tta_level else None
    )
    stop_timer()

    return {
        "optimizer": opt_name,
        "epochs": epochs,
        "steps": step,
        "train_acc_last": train_acc,
        "val_accs": val_accs,
        "val_acc": val_accs[-1],
        "tta_val_acc": tta_val_acc,
        "time_seconds": time_seconds,
    }


def run_airbench(config: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """WP0.1 baseline: the stock vendored airbench94 recipe, unmodified.

    Same harness as :func:`run_airbench_smoke` but pinned to the vendored
    Muon at the record hyperparameters (airbench94_muon.py:362 —
    lr=0.24, momentum=0.6, nesterov=True) with torch.compile on, exactly as
    the reference ``main()``. Config may not override the optimizer: the
    point of WP0.1 is the untouched reference distribution.
    """
    if "optimizer" in config:
        raise SystemExit(
            "experiment 'airbench' is the stock WP0.1 baseline; it does not "
            "accept an optimizer override (use experiment 'airbench_smoke')."
        )
    merged = dict(config)
    merged["optimizer"] = dict(
        name="vendor_muon", lr=0.24, momentum=0.6, nesterov=True
    )
    recipe = dict(config.get("recipe", {}))
    recipe["normalize_filter_weights"] = False  # vendored Muon.step does it
    recipe.setdefault("compile", True)  # the reference compiles the model
    merged["recipe"] = recipe
    return run_airbench_smoke(merged, device)


AIRBENCH_STOCK_LR = 0.24  # vendored record hyperparameter (airbench94_muon.py:362)


def run_airbench_instrumented(
    config: Dict[str, Any], device: torch.device
) -> Dict[str, Any]:
    """WP1.2 instrumented airbench: the IDENTICAL stock WP0.1 recipe
    (vendored Muon lr=0.24 momentum=0.6 nesterov, compile on) with an
    InstrumentationHub observing the filter-parameter matrices.

    Per plan section 1.1: top-k1 + k2 bulk tracked pairs of the vendored
    Muon's momentum buffers; per-step raw PRE-momentum gradient projections
    s_i = u^T G v (grabbed via ``hub.capture_grads()`` before
    ``optimizer2.step()``); per-matrix top sigma and ||G||_F; both betas.
    The HVP callback stays None on airbench (the amplitude-ratio path is the
    core estimator; HVPs are validation-only and optional here).

    Zero behavior change to training itself: the hub is read-only and the
    recipe, schedules, and optimizers are exactly those of the ``airbench``
    experiment.  The returned metrics carry the full instrumentation log
    under the private key ``"_instrumentation_log"``; scripts/run.py pops it
    and writes the src.instrument.schema sidecar next to the results JSON.

    LAUNCH PRECONDITION (WP1.2, enforced in scripts/run.py): the
    human-authored ``criteria/phase1_preregistration.md`` must exist before
    any run of this experiment.
    """
    from src.instrument import hub_from_config

    if "optimizer" in config:
        raise SystemExit(
            "experiment 'airbench_instrumented' is the stock WP0.1 recipe "
            "plus read-only instrumentation; it does not accept an optimizer "
            "override."
        )
    instr_cfg = config.get("instrumentation")
    if not isinstance(instr_cfg, dict):
        raise SystemExit(
            "experiment 'airbench_instrumented' requires an 'instrumentation' "
            "block in the config (k1, k2, t_refresh, betas, classifier, ...)"
        )

    merged = dict(config)
    merged["optimizer"] = dict(
        name="vendor_muon", lr=AIRBENCH_STOCK_LR, momentum=0.6, nesterov=True
    )
    recipe = dict(config.get("recipe", {}))
    recipe["normalize_filter_weights"] = False  # vendored Muon.step does it
    recipe.setdefault("compile", True)  # the reference compiles the model
    merged["recipe"] = recipe

    holder: Dict[str, Any] = {}

    def factory(model, optimizer2, filter_params):
        names = {id(p): n for n, p in model.named_parameters()}
        named = [
            (names.get(id(p), f"filter_{i}"), p)
            for i, p in enumerate(filter_params)
        ]
        # HVP callback stays None for airbench (validation-only, optional).
        holder["hub"] = hub_from_config(instr_cfg, named, optimizer2, hvp_fn=None)
        return holder["hub"]

    metrics = run_airbench_smoke(merged, device, _hub_factory=factory)
    metrics["instrumented"] = True
    metrics["optimizer_lr"] = AIRBENCH_STOCK_LR  # for the eta*lambda plot
    metrics["_instrumentation_log"] = holder["hub"].to_log()
    return metrics


# ----------------------------------------------------------------- entrypoint


def _load_run_module():
    """Import scripts/run.py (the WP0.0 runner) without modifying it."""
    import importlib.util

    path = REPO_ROOT / "scripts" / "run.py"
    spec = importlib.util.spec_from_file_location("routed_muon_run", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main(argv=None) -> int:
    """Delegate to scripts/run.py with the zoo registered at runtime.

    Usage: uv run python -m src.optim.airbench_zoo <config.yaml> [--seed N]
    """
    from src.optim.registry import OPTIMIZER_REGISTRY

    run_mod = _load_run_module()
    run_mod.OPTIMIZER_REGISTRY.update(OPTIMIZER_REGISTRY)
    run_mod.EXPERIMENT_REGISTRY["airbench_smoke"] = run_airbench_smoke
    run_mod.EXPERIMENT_REGISTRY["airbench"] = run_airbench
    run_mod.EXPERIMENT_REGISTRY["airbench_instrumented"] = run_airbench_instrumented
    return run_mod.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
